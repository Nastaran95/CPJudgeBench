"""LLM-as-judge evaluation of generated candidate CP models."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CORRECTNESS_LABELS, ExperimentContext, LABEL_GUIDANCE
from .llm import ModelSpec, get_openrouter_llm, llm_response_to_text
from .parsing import normalize_label, parse_judge_json

logger = logging.getLogger(__name__)

JUDGE_STATUS_FILENAME = "judge-status.csv"
JUDGE_STATUS_COLUMNS = [
    "problem_id",
    "generator_llm",
    "language",
    "label",
    "judge_llm",
    "approach",
    "judgment",
    "succeed",
    "description",
    "updated_at",
]

# Allowed approach identifiers
JUDGE_APPROACH_REFERENCE_FREE = "reference_free"
JUDGE_APPROACH_REFERENCE_BASED = "reference_based"
JUDGE_APPROACHES = [JUDGE_APPROACH_REFERENCE_FREE, JUDGE_APPROACH_REFERENCE_BASED]


def _judge_status_path() -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / JUDGE_STATUS_FILENAME


def _judge_status_key(
    problem_id: str,
    generator_llm: str,
    language: str,
    label: str,
    judge_llm: str,
    approach: str = JUDGE_APPROACH_REFERENCE_FREE,
) -> tuple[str, str, str, str, str, str]:
    return (problem_id, generator_llm, language, label, judge_llm, approach)


def _load_judge_status() -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    path = _judge_status_path()
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        logger.warning("[judge] could not read status file %s (%s); starting fresh", path, e)
        return {}

    rows: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        # Older rows may lack the "approach" column; default to reference_free.
        approach_val = str(row.get("approach", "") or JUDGE_APPROACH_REFERENCE_FREE)
        key = _judge_status_key(
            str(row.get("problem_id", "")),
            str(row.get("generator_llm", "")),
            str(row.get("language", "")),
            str(row.get("label", "")),
            str(row.get("judge_llm", "")),
            approach_val,
        )
        if all(k for k in key[:5]):  # first five fields must be non-empty
            rows[key] = row
    return rows


def _save_judge_status(rows: dict[tuple[str, str, str, str, str, str], dict[str, Any]]) -> None:
    path = _judge_status_path()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    records = [rows[k] for k in sorted(rows.keys())]
    df = pd.DataFrame(records)
    for col in JUDGE_STATUS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[JUDGE_STATUS_COLUMNS]
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, path)


def _update_judge_status(
    rows: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
    *,
    problem_id: str,
    generator_llm: str,
    language: str,
    label: str,
    judge_llm: str,
    approach: str,
    judgment: str,
    succeed: bool,
    description: str,
) -> None:
    key = _judge_status_key(problem_id, generator_llm, language, label, judge_llm, approach)
    rows[key] = {
        "problem_id": problem_id,
        "generator_llm": generator_llm,
        "language": language,
        "label": label,
        "judge_llm": judge_llm,
        "approach": approach,
        "judgment": judgment,
        "succeed": "yes" if succeed else "no",
        "description": description,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_judge_status(rows)


def _judge_prompt(context: ExperimentContext, candidate_code: str, language: str) -> str:
    labels_text = ", ".join(CORRECTNESS_LABELS)
    label_guidance_text = "\n".join(f"- {label}: {guidance}" for label, guidance in LABEL_GUIDANCE.items())
    return f"""
You are an expert CP model judge.
Classify the candidate with exactly one allowed correctness label.

Allowed labels: {labels_text}

Label guidance:
{label_guidance_text}

Reasoning protocol (perform internally before answering):
1) Check executability for the given language (syntax/API shape + obvious runtime issues).
2) Infer intended semantics from the problem description.
3) Compare candidate constraints to intended semantics.
4) Decide the appropriate label.

Output format (STRICT JSON only):
{{"label": "<one allowed label>", "confidence": <0..1>, "rationale": "<1-3 short sentences>"}}

Problem description:
{context.problem_description}

Candidate ({language}):
{candidate_code}
""".strip()


def _judge_prompt_reference_based(
    context: ExperimentContext, candidate_code: str, language: str
) -> str:
    labels_text = ", ".join(CORRECTNESS_LABELS)
    label_guidance_text = "\n".join(f"- {label}: {guidance}" for label, guidance in LABEL_GUIDANCE.items())
    return f"""
You are an expert CP model judge.
Classify the candidate with exactly one allowed correctness label.
Use the reference model below as the ground-truth encoding of the problem.

Allowed labels: {labels_text}

Label guidance:
{label_guidance_text}

Reasoning protocol (perform internally before answering):
1) Understand the reference model's constraints and the solution space it encodes.
2) Check whether the candidate is executable (syntax/API shape + obvious runtime issues).
3) Compare the candidate's constraints to the reference model, identifying false positives
   (solutions allowed by candidate but rejected by reference) and false negatives
   (solutions accepted by reference but rejected by candidate).
4) Decide the appropriate label.

Output format (STRICT JSON only):
{{"label": "<one allowed label>", "confidence": <0..1>, "rationale": "<1-3 short sentences>"}}

Problem description:
{context.problem_description}

Reference model ({context.reference_language}):
{context.reference_cp_model}

Candidate ({language}):
{candidate_code}
""".strip()


def direct_llm_judge(
    context: ExperimentContext,
    spec: ModelSpec,
    candidate_code: str,
    language: str,
    approach: str = JUDGE_APPROACH_REFERENCE_FREE,
) -> tuple[str, dict[str, Any], str]:
    """Ask a single LLM to label one candidate model.

    ``approach`` selects the prompt variant:
      - ``"reference_free"``  – problem description only (default, existing behaviour).
      - ``"reference_based"`` – also includes the reference CP model in the prompt.
    """
    llm = get_openrouter_llm(spec)
    if approach == JUDGE_APPROACH_REFERENCE_BASED:
        prompt = _judge_prompt_reference_based(context, candidate_code, language)
    else:
        prompt = _judge_prompt(context, candidate_code, language)
    raw_reply = llm_response_to_text(llm.invoke(prompt)).strip()
    parsed = parse_judge_json(raw_reply)
    predicted_label = normalize_label(str(parsed.get("label", "")))
    return predicted_label, parsed, raw_reply


def _load_plain_payload(context: ExperimentContext) -> dict[str, Any]:
    path = context.output_dir / "candidate-models.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate payload not found at {path}. "
            f"Run the data-generation phase first (e.g. `python main.py --data-generation`)."
        )
    logger.info("[judge] loading candidates from disk: %s", path)
    return json.loads(path.read_text(encoding="utf-8"))


def _save_judge_results(
    context: ExperimentContext, judge_df: pd.DataFrame, summary_df: pd.DataFrame
) -> None:
    output_dir = context.output_dir
    details_path = output_dir / "judge-results.csv"
    summary_path = output_dir / "judge-summary.csv"
    judge_df.to_csv(details_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info("[judge] saved per-candidate results: %s", details_path)
    logger.info("[judge] saved rich summary table:    %s", summary_path)


def _candidate_is_judgeable(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("ok") is False:
        return False
    return bool(str(meta.get("code", "")).strip())


def _aggregate_accuracy(judge_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if judge_df.empty:
        cols = group_cols + ["correct", "total", "accuracy"]
        return pd.DataFrame(columns=cols)
    if not group_cols:
        total = int(len(judge_df))
        correct = int(judge_df["match"].sum())
        accuracy = (correct / total) if total else 0.0
        return pd.DataFrame([{"correct": correct, "total": total, "accuracy": accuracy}])

    agg = (
        judge_df.groupby(group_cols, dropna=False)["match"]
        .agg(correct="sum", total="count")
        .reset_index()
    )
    agg["correct"] = agg["correct"].astype(int)
    agg["total"] = agg["total"].astype(int)
    agg["accuracy"] = agg["correct"] / agg["total"]
    return agg


def _summary_dataframe(judge_df: pd.DataFrame) -> pd.DataFrame:
    """Single rich summary file with multiple breakdown views."""
    base_cols = ["view", "approach", "judge_llm", "language", "label", "correct", "total", "accuracy"]

    def _pack(view: str, df: pd.DataFrame, rename: dict[str, str] | None = None) -> pd.DataFrame:
        rename = rename or {}
        out = df.rename(columns=rename).copy()
        out["view"] = view
        for col in ("approach", "judge_llm", "language", "label"):
            if col not in out.columns:
                out[col] = ""
        out = out[base_cols]
        return out

    if judge_df.empty:
        return pd.DataFrame(columns=base_cols)

    pieces = [
        _pack("overall", _aggregate_accuracy(judge_df, [])),
        _pack("by_judge_language_approach", _aggregate_accuracy(judge_df, ["judge_llm", "approach", "language"])),
        _pack("by_judge_language_approach_label", _aggregate_accuracy(judge_df, ["judge_llm", "approach", "language", "claimed_label"]), rename={"claimed_label": "label"}),
    ]
    return pd.concat(pieces, ignore_index=True)


def run_judge_evaluation(
    context: ExperimentContext,
    judge_specs: list[ModelSpec],
    plain_payload: dict[str, Any] | None = None,
    approach: str = JUDGE_APPROACH_REFERENCE_FREE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every judge over every captured candidate and aggregate accuracy.

    ``approach`` selects the prompt variant used for every judge call:
      - ``"reference_free"``  – problem description only (default).
      - ``"reference_based"`` – also includes the reference CP model.

    Results are saved to ``judge-results.csv`` (with an ``approach`` column)
    and ``judge-summary.csv``.
    """
    if approach not in JUDGE_APPROACHES:
        raise ValueError(f"Unknown approach '{approach}'. Must be one of {JUDGE_APPROACHES}.")

    if plain_payload is None:
        plain_payload = _load_plain_payload(context)
    status_rows = _load_judge_status()
    if status_rows:
        logger.info(
            "[judge] loaded existing judge status: %d row(s) from %s",
            len(status_rows),
            _judge_status_path(),
        )
    else:
        logger.info("[judge] creating judge status file: %s", _judge_status_path())
    _save_judge_status(status_rows)

    total_candidates = sum(
        len(by_label)
        for by_language in plain_payload.values()
        for by_label in by_language.values()
    )
    logger.info(
        "[judge] starting: approach=%s, judges=%d, generators=%d, candidate slots=%d",
        approach,
        len(judge_specs),
        len(plain_payload),
        total_candidates,
    )

    judge_rows: list[dict[str, Any]] = []
    for judge_spec in judge_specs:
        judge_prefix = f"[judge={judge_spec.raw}][approach={approach}]"
        logger.info("%s starting", judge_prefix)

        judged_count = 0
        match_count = 0
        excluded_count = 0
        failed_count = 0

        for generator_llm, by_language in plain_payload.items():
            for language, by_label in by_language.items():
                for claimed_label, meta in by_label.items():
                    slot_suffix = (
                        f"gen={generator_llm} | {language} | claimed='{claimed_label}'"
                    )

                    if not _candidate_is_judgeable(meta):
                        logger.info("%s %s -> excluded (no usable code)", judge_prefix, slot_suffix)
                        _update_judge_status(
                            status_rows,
                            problem_id=context.targeted_id,
                            generator_llm=generator_llm,
                            language=language,
                            label=claimed_label,
                            judge_llm=judge_spec.raw,
                            approach=approach,
                            judgment="excluded",
                            succeed=False,
                            description="No usable candidate code to judge",
                        )
                        excluded_count += 1
                        continue

                    candidate_code = meta["code"]
                    try:
                        predicted_label, parsed, raw_reply = direct_llm_judge(
                            context, judge_spec, candidate_code, language, approach=approach
                        )
                    except Exception as e:
                        logger.exception(
                            "%s %s -> failed (%s: %s)",
                            judge_prefix,
                            slot_suffix,
                            type(e).__name__,
                            e,
                        )
                        _update_judge_status(
                            status_rows,
                            problem_id=context.targeted_id,
                            generator_llm=generator_llm,
                            language=language,
                            label=claimed_label,
                            judge_llm=judge_spec.raw,
                            approach=approach,
                            judgment="judge-error",
                            succeed=False,
                            description=f"{type(e).__name__}: {e}",
                        )
                        failed_count += 1
                        continue

                    matched = predicted_label == claimed_label
                    confidence = parsed.get("confidence")
                    confidence_text = (
                        f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
                    )
                    logger.info(
                        "%s %s -> predicted='%s' match=%s confidence=%s",
                        judge_prefix,
                        slot_suffix,
                        predicted_label,
                        matched,
                        confidence_text,
                    )
                    _update_judge_status(
                        status_rows,
                        problem_id=context.targeted_id,
                        generator_llm=generator_llm,
                        language=language,
                        label=claimed_label,
                        judge_llm=judge_spec.raw,
                        approach=approach,
                        judgment=predicted_label,
                        succeed=matched,
                        description=str(parsed.get("rationale", "")),
                    )

                    judged_count += 1
                    match_count += int(matched)
                    judge_rows.append({
                        "judge_llm": judge_spec.raw,
                        "approach": approach,
                        "generator_llm": generator_llm,
                        "language": language,
                        "claimed_label": claimed_label,
                        "predicted_label": predicted_label,
                        "match": matched,
                        "confidence": confidence,
                        "rationale": parsed.get("rationale", ""),
                        "raw_reply": raw_reply,
                    })

        accuracy_text = f"{match_count / judged_count:.3f}" if judged_count else "n/a"
        logger.info(
            "%s done: judged=%d matched=%d excluded=%d failed=%d accuracy=%s",
            judge_prefix,
            judged_count,
            match_count,
            excluded_count,
            failed_count,
            accuracy_text,
        )

    judge_df = pd.DataFrame(judge_rows)
    summary_df = _summary_dataframe(judge_df)
    _save_judge_results(context, judge_df, summary_df)
    logger.info("[judge] complete: %d judgement(s) collected", len(judge_rows))
    return judge_df, summary_df


def run_judge_evaluation_all_approaches(
    context: ExperimentContext,
    judge_specs: list[ModelSpec],
    plain_payload: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both reference-free and reference-based judges and save combined results.

    Calls ``run_judge_evaluation`` for each approach, concatenates the result
    DataFrames, and saves the combined files (overwriting what each individual
    run already saved).

    Returns ``(combined_judge_df, combined_summary_df)``.
    """
    all_dfs: list[pd.DataFrame] = []
    for appr in JUDGE_APPROACHES:
        df, _ = run_judge_evaluation(
            context,
            judge_specs,
            plain_payload=plain_payload,
            approach=appr,
        )
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    combined_summary = _summary_dataframe(combined_df)
    _save_judge_results(context, combined_df, combined_summary)
    logger.info(
        "[judge] all-approaches complete: %d total judgement(s)", len(combined_df)
    )
    return combined_df, combined_summary
