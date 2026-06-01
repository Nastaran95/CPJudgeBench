"""LLM-as-judge evaluation of generated candidate CP models."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentContext
from .llm import ModelSpec, get_openrouter_llm, llm_response_to_text
from .parsing import (
    normalize_label,
    parse_judge_json,
    parse_score_json,
    parse_binary_json,
    strip_code_comments,
)
from .prompts import (
    label_judge_prompt_reference_free,
    label_judge_prompt_reference_based,
    score_judge_prompt_reference_free,
    score_judge_prompt_reference_based,
    binary_judge_prompt_reference_free,
    binary_judge_prompt_reference_based,
)

logger = logging.getLogger(__name__)

# change here
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

# ---------------------------------------------------------------------------
# Approach identifiers
# ---------------------------------------------------------------------------

JUDGE_APPROACH_REFERENCE_FREE = "reference_free"
JUDGE_APPROACH_REFERENCE_BASED = "reference_based"
JUDGE_APPROACH_SCORE_REFERENCE_FREE = "score_reference_free"
JUDGE_APPROACH_SCORE_REFERENCE_BASED = "score_reference_based"
JUDGE_APPROACH_BINARY_REFERENCE_FREE = "binary_reference_free"
JUDGE_APPROACH_BINARY_REFERENCE_BASED = "binary_reference_based"

LABEL_APPROACHES: list[str] = [
    JUDGE_APPROACH_REFERENCE_FREE,
    JUDGE_APPROACH_REFERENCE_BASED,
]
SCORE_APPROACHES: list[str] = [
    JUDGE_APPROACH_SCORE_REFERENCE_FREE,
    JUDGE_APPROACH_SCORE_REFERENCE_BASED,
]
BINARY_APPROACHES: list[str] = [
    JUDGE_APPROACH_BINARY_REFERENCE_FREE,
    JUDGE_APPROACH_BINARY_REFERENCE_BASED,
]

JUDGE_APPROACHES: list[str] = LABEL_APPROACHES + SCORE_APPROACHES + BINARY_APPROACHES

# Result / summary file names by judge type
_LABEL_RESULTS_FILE = "judge-results.csv"
_LABEL_SUMMARY_FILE = "judge-summary.csv"
_SCORE_RESULTS_FILE = "score-judge-results.csv"
_SCORE_SUMMARY_FILE = "score-judge-summary.csv"
_BINARY_RESULTS_FILE = "binary-judge-results.csv"
_BINARY_SUMMARY_FILE = "binary-judge-summary.csv"


# ---------------------------------------------------------------------------
# Status file helpers  (shared across all judge types)
# ---------------------------------------------------------------------------

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
        approach_val = str(row.get("approach", "") or JUDGE_APPROACH_REFERENCE_FREE)
        key = _judge_status_key(
            str(row.get("problem_id", "")),
            str(row.get("generator_llm", "")),
            str(row.get("language", "")),
            str(row.get("label", "")),
            str(row.get("judge_llm", "")),
            approach_val,
        )
        if all(k for k in key[:5]):
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


# ---------------------------------------------------------------------------
# Prompt dispatch helpers
# ---------------------------------------------------------------------------

def _build_label_prompt(
    context: ExperimentContext, candidate_code: str, language: str, approach: str
) -> str:
    if approach == JUDGE_APPROACH_REFERENCE_BASED:
        return label_judge_prompt_reference_based(context, candidate_code, language)
    return label_judge_prompt_reference_free(context, candidate_code, language)


def _build_score_prompt(
    context: ExperimentContext, candidate_code: str, language: str, approach: str
) -> str:
    if approach == JUDGE_APPROACH_SCORE_REFERENCE_BASED:
        return score_judge_prompt_reference_based(context, candidate_code, language)
    return score_judge_prompt_reference_free(context, candidate_code, language)


def _build_binary_prompt(
    context: ExperimentContext, candidate_code: str, language: str, approach: str
) -> str:
    if approach == JUDGE_APPROACH_BINARY_REFERENCE_BASED:
        return binary_judge_prompt_reference_based(context, candidate_code, language)
    return binary_judge_prompt_reference_free(context, candidate_code, language)


# ---------------------------------------------------------------------------
# Core judge call functions
# ---------------------------------------------------------------------------

def direct_llm_judge(
    context: ExperimentContext,
    spec: ModelSpec,
    candidate_code: str,
    language: str,
    approach: str = JUDGE_APPROACH_REFERENCE_FREE,
) -> tuple[str, dict[str, Any], str]:
    """Label judge: ask one LLM to assign a correctness label to a candidate.

    Comments are stripped from *candidate_code* before the prompt is sent.
    Returns ``(predicted_label, parsed_dict, raw_reply)``.
    """
    clean_code = strip_code_comments(candidate_code, language)
    llm = get_openrouter_llm(spec)
    prompt = _build_label_prompt(context, clean_code, language, approach)
    raw_reply = llm_response_to_text(llm.invoke(prompt)).strip()
    parsed = parse_judge_json(raw_reply)
    predicted_label = normalize_label(str(parsed.get("label", "")))
    return predicted_label, parsed, raw_reply


def direct_llm_score_judge(
    context: ExperimentContext,
    spec: ModelSpec,
    candidate_code: str,
    language: str,
    approach: str = JUDGE_APPROACH_SCORE_REFERENCE_FREE,
) -> tuple[int | None, dict[str, Any], str]:
    """Score judge: ask one LLM to rate a candidate on a 1–5 scale.

    Comments are stripped from *candidate_code* before the prompt is sent.
    Returns ``(score, parsed_dict, raw_reply)`` where *score* is 1–5 or ``None``.
    """
    clean_code = strip_code_comments(candidate_code, language)
    llm = get_openrouter_llm(spec)
    prompt = _build_score_prompt(context, clean_code, language, approach)
    raw_reply = llm_response_to_text(llm.invoke(prompt)).strip()
    parsed = parse_score_json(raw_reply)
    return parsed.get("score"), parsed, raw_reply


def direct_llm_binary_judge(
    context: ExperimentContext,
    spec: ModelSpec,
    candidate_code: str,
    language: str,
    approach: str = JUDGE_APPROACH_BINARY_REFERENCE_FREE,
) -> tuple[str, dict[str, Any], str]:
    """Binary judge: ask one LLM for a correct/incorrect verdict.

    Comments are stripped from *candidate_code* before the prompt is sent.
    Returns ``(verdict, parsed_dict, raw_reply)`` where *verdict* is
    ``"correct"``, ``"incorrect"``, or ``"unknown"``.
    """
    clean_code = strip_code_comments(candidate_code, language)
    llm = get_openrouter_llm(spec)
    prompt = _build_binary_prompt(context, clean_code, language, approach)
    raw_reply = llm_response_to_text(llm.invoke(prompt)).strip()
    parsed = parse_binary_json(raw_reply)
    return str(parsed.get("verdict", "unknown")), parsed, raw_reply


# ---------------------------------------------------------------------------
# Shared helpers: payload loading, saving, candidate filtering
# ---------------------------------------------------------------------------

def _load_plain_payload(context: ExperimentContext) -> dict[str, Any]:
    path = context.output_dir / "candidate-models.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate payload not found at {path}. "
            "Run the data-generation phase first (e.g. `python main.py --data-generation`)."
        )
    logger.info("[judge] loading candidates from disk: %s", path)
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_is_judgeable(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("ok") is False:
        return False
    return bool(str(meta.get("code", "")).strip())


def _save_results(
    context: ExperimentContext,
    judge_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    results_filename: str,
    summary_filename: str,
) -> None:
    output_dir = context.output_dir
    details_path = output_dir / results_filename
    summary_path = output_dir / summary_filename
    judge_df.to_csv(details_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info("[judge] saved results:  %s", details_path)
    logger.info("[judge] saved summary:  %s", summary_path)


# ---------------------------------------------------------------------------
# Label judge runner
# ---------------------------------------------------------------------------

def _aggregate_accuracy(judge_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if judge_df.empty:
        return pd.DataFrame(columns=group_cols + ["correct", "total", "accuracy"])
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


def _label_summary_dataframe(judge_df: pd.DataFrame) -> pd.DataFrame:
    """Single rich summary file with multiple breakdown views for label judges."""
    base_cols = ["view", "approach", "judge_llm", "language", "label", "correct", "total", "accuracy"]

    def _pack(view: str, df: pd.DataFrame, rename: dict[str, str] | None = None) -> pd.DataFrame:
        rename = rename or {}
        out = df.rename(columns=rename).copy()
        out["view"] = view
        for col in ("approach", "judge_llm", "language", "label"):
            if col not in out.columns:
                out[col] = ""
        return out[base_cols]

    if judge_df.empty:
        return pd.DataFrame(columns=base_cols)

    pieces = [
        _pack("overall", _aggregate_accuracy(judge_df, [])),
        _pack("by_judge_language_approach", _aggregate_accuracy(
            judge_df, ["judge_llm", "approach", "language"]
        )),
        _pack(
            "by_judge_language_approach_label",
            _aggregate_accuracy(judge_df, ["judge_llm", "approach", "language", "claimed_label"]),
            rename={"claimed_label": "label"},
        ),
    ]
    return pd.concat(pieces, ignore_index=True)


def _run_label_judge_evaluation(
    context: ExperimentContext,
    judge_specs: list[ModelSpec],
    plain_payload: dict[str, Any],
    approach: str,
    status_rows: dict,
) -> pd.DataFrame:
    """Inner loop for label-based judge approaches. Returns a rows DataFrame."""
    judge_rows: list[dict[str, Any]] = []
    judge_prefix_base = f"[approach={approach}]"

    for judge_spec in judge_specs:
        judge_prefix = f"[judge={judge_spec.raw}]{judge_prefix_base}"
        logger.info("%s starting", judge_prefix)

        judged_count = match_count = excluded_count = failed_count = 0

        for generator_llm, by_language in plain_payload.items():
            for language, by_label in by_language.items():
                # # change here
                # if language != "minizinc":
                #     continue
                for claimed_label, meta in by_label.items():
                    slot = f"gen={generator_llm} | {language} | claimed='{claimed_label}'"

                    if not _candidate_is_judgeable(meta):
                        logger.info("%s %s -> excluded (no usable code)", judge_prefix, slot)
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

                    try:
                        predicted_label, parsed, raw_reply = direct_llm_judge(
                            context, judge_spec, meta["code"], language, approach=approach
                        )
                    except Exception as e:
                        logger.exception(
                            "%s %s -> failed (%s: %s)", judge_prefix, slot, type(e).__name__, e
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
                        judge_prefix, slot, predicted_label, matched, confidence_text,
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
            judge_prefix, judged_count, match_count, excluded_count, failed_count, accuracy_text,
        )

    return pd.DataFrame(judge_rows)


# ---------------------------------------------------------------------------
# Score judge runner
# ---------------------------------------------------------------------------

def _score_summary_dataframe(score_df: pd.DataFrame) -> pd.DataFrame:
    """Summary for score judges: mean score broken down by label."""
    base_cols = ["view", "approach", "judge_llm", "language", "label", "mean_score", "count"]

    def _pack(view: str, df: pd.DataFrame, rename: dict[str, str] | None = None) -> pd.DataFrame:
        rename = rename or {}
        out = df.rename(columns=rename).copy()
        out["view"] = view
        for col in ("approach", "judge_llm", "language", "label"):
            if col not in out.columns:
                out[col] = ""
        return out[base_cols]

    if score_df.empty or "score" not in score_df.columns:
        return pd.DataFrame(columns=base_cols)

    valid = score_df.dropna(subset=["score"]).copy()
    valid["score"] = pd.to_numeric(valid["score"], errors="coerce")

    def _agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=group_cols + ["mean_score", "count"])
        if not group_cols:
            return pd.DataFrame([{
                "mean_score": round(float(df["score"].mean()), 3),
                "count": len(df),
            }])
        agg = (
            df.groupby(group_cols, dropna=False)["score"]
            .agg(mean_score="mean", count="count")
            .reset_index()
        )
        agg["mean_score"] = agg["mean_score"].round(3)
        return agg

    pieces = [
        _pack("overall", _agg(valid, [])),
        _pack("by_judge_language_approach", _agg(valid, ["judge_llm", "approach", "language"])),
        _pack(
            "by_judge_language_approach_label",
            _agg(valid, ["judge_llm", "approach", "language", "claimed_label"]),
            rename={"claimed_label": "label"},
        ),
    ]
    return pd.concat(pieces, ignore_index=True)


def _run_score_judge_evaluation(
    context: ExperimentContext,
    judge_specs: list[ModelSpec],
    plain_payload: dict[str, Any],
    approach: str,
    status_rows: dict,
) -> pd.DataFrame:
    """Inner loop for score-based judge approaches. Returns a rows DataFrame."""
    score_rows: list[dict[str, Any]] = []
    judge_prefix_base = f"[approach={approach}]"

    for judge_spec in judge_specs:
        judge_prefix = f"[judge={judge_spec.raw}]{judge_prefix_base}"
        logger.info("%s starting", judge_prefix)

        judged_count = excluded_count = failed_count = 0

        for generator_llm, by_language in plain_payload.items():
            for language, by_label in by_language.items():
                for claimed_label, meta in by_label.items():
                    slot = f"gen={generator_llm} | {language} | claimed='{claimed_label}'"

                    if not _candidate_is_judgeable(meta):
                        logger.info("%s %s -> excluded (no usable code)", judge_prefix, slot)
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

                    try:
                        score, parsed, raw_reply = direct_llm_score_judge(
                            context, judge_spec, meta["code"], language, approach=approach
                        )
                    except Exception as e:
                        logger.exception(
                            "%s %s -> failed (%s: %s)", judge_prefix, slot, type(e).__name__, e
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

                    score_text = str(score) if score is not None else "n/a"
                    logger.info(
                        "%s %s -> score=%s", judge_prefix, slot, score_text
                    )
                    _update_judge_status(
                        status_rows,
                        problem_id=context.targeted_id,
                        generator_llm=generator_llm,
                        language=language,
                        label=claimed_label,
                        judge_llm=judge_spec.raw,
                        approach=approach,
                        judgment=score_text,
                        succeed=score is not None,
                        description=str(parsed.get("rationale", "")),
                    )

                    judged_count += 1
                    score_rows.append({
                        "judge_llm": judge_spec.raw,
                        "approach": approach,
                        "generator_llm": generator_llm,
                        "language": language,
                        "claimed_label": claimed_label,
                        "score": score,
                        "rationale": parsed.get("rationale", ""),
                        "raw_reply": raw_reply,
                    })

        logger.info(
            "%s done: judged=%d excluded=%d failed=%d",
            judge_prefix, judged_count, excluded_count, failed_count,
        )

    return pd.DataFrame(score_rows)


# ---------------------------------------------------------------------------
# Binary judge runner
# ---------------------------------------------------------------------------

def _binary_summary_dataframe(binary_df: pd.DataFrame) -> pd.DataFrame:
    """Summary for binary judges: accuracy broken down by label."""
    base_cols = ["view", "approach", "judge_llm", "language", "label", "correct", "total", "accuracy"]

    def _pack(view: str, df: pd.DataFrame, rename: dict[str, str] | None = None) -> pd.DataFrame:
        rename = rename or {}
        out = df.rename(columns=rename).copy()
        out["view"] = view
        for col in ("approach", "judge_llm", "language", "label"):
            if col not in out.columns:
                out[col] = ""
        return out[base_cols]

    if binary_df.empty or "match" not in binary_df.columns:
        return pd.DataFrame(columns=base_cols)

    pieces = [
        _pack("overall", _aggregate_accuracy(binary_df, [])),
        _pack("by_judge_language_approach", _aggregate_accuracy(
            binary_df, ["judge_llm", "approach", "language"]
        )),
        _pack(
            "by_judge_language_approach_label",
            _aggregate_accuracy(binary_df, ["judge_llm", "approach", "language", "claimed_label"]),
            rename={"claimed_label": "label"},
        ),
    ]
    return pd.concat(pieces, ignore_index=True)


def _run_binary_judge_evaluation(
    context: ExperimentContext,
    judge_specs: list[ModelSpec],
    plain_payload: dict[str, Any],
    approach: str,
    status_rows: dict,
) -> pd.DataFrame:
    """Inner loop for binary judge approaches. Returns a rows DataFrame."""
    binary_rows: list[dict[str, Any]] = []
    judge_prefix_base = f"[approach={approach}]"

    for judge_spec in judge_specs:
        judge_prefix = f"[judge={judge_spec.raw}]{judge_prefix_base}"
        logger.info("%s starting", judge_prefix)

        judged_count = match_count = excluded_count = failed_count = 0

        for generator_llm, by_language in plain_payload.items():
            for language, by_label in by_language.items():
                for claimed_label, meta in by_label.items():
                    slot = f"gen={generator_llm} | {language} | claimed='{claimed_label}'"

                    if not _candidate_is_judgeable(meta):
                        logger.info("%s %s -> excluded (no usable code)", judge_prefix, slot)
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

                    try:
                        verdict, parsed, raw_reply = direct_llm_binary_judge(
                            context, judge_spec, meta["code"], language, approach=approach
                        )
                    except Exception as e:
                        logger.exception(
                            "%s %s -> failed (%s: %s)", judge_prefix, slot, type(e).__name__, e
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

                    ground_truth_binary = "correct" if claimed_label == "equivalent" else "incorrect"
                    matched = verdict == ground_truth_binary
                    confidence = parsed.get("confidence")
                    confidence_text = (
                        f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
                    )
                    logger.info(
                        "%s %s -> verdict='%s' ground_truth='%s' match=%s confidence=%s",
                        judge_prefix, slot, verdict, ground_truth_binary, matched, confidence_text,
                    )
                    _update_judge_status(
                        status_rows,
                        problem_id=context.targeted_id,
                        generator_llm=generator_llm,
                        language=language,
                        label=claimed_label,
                        judge_llm=judge_spec.raw,
                        approach=approach,
                        judgment=verdict,
                        succeed=matched,
                        description=str(parsed.get("rationale", "")),
                    )

                    judged_count += 1
                    match_count += int(matched)
                    binary_rows.append({
                        "judge_llm": judge_spec.raw,
                        "approach": approach,
                        "generator_llm": generator_llm,
                        "language": language,
                        "claimed_label": claimed_label,
                        "ground_truth_binary": ground_truth_binary,
                        "predicted_binary": verdict,
                        "match": matched,
                        "confidence": confidence,
                        "rationale": parsed.get("rationale", ""),
                        "raw_reply": raw_reply,
                    })

        accuracy_text = f"{match_count / judged_count:.3f}" if judged_count else "n/a"
        logger.info(
            "%s done: judged=%d matched=%d excluded=%d failed=%d accuracy=%s",
            judge_prefix, judged_count, match_count, excluded_count, failed_count, accuracy_text,
        )

    return pd.DataFrame(binary_rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_judge_evaluation(
    context: ExperimentContext,
    judge_specs: list[ModelSpec],
    plain_payload: dict[str, Any] | None = None,
    approach: str = JUDGE_APPROACH_REFERENCE_FREE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every judge over every captured candidate for the given *approach*.

    Dispatches internally to the label, score, or binary runner depending on
    the approach type.  Results are saved to per-type CSV files and the shared
    status file is updated after each judgment.

    Returns ``(results_df, summary_df)``.
    """
    if approach not in JUDGE_APPROACHES:
        raise ValueError(f"Unknown approach '{approach}'. Must be one of {JUDGE_APPROACHES}.")

    if plain_payload is None:
        plain_payload = _load_plain_payload(context)

    status_rows = _load_judge_status()
    if status_rows:
        logger.info(
            "[judge] loaded existing judge status: %d row(s) from %s",
            len(status_rows), _judge_status_path(),
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
        approach, len(judge_specs), len(plain_payload), total_candidates,
    )

    if approach in LABEL_APPROACHES:
        results_df = _run_label_judge_evaluation(
            context, judge_specs, plain_payload, approach, status_rows
        )
        summary_df = _label_summary_dataframe(results_df)
        _save_results(context, results_df, summary_df, _LABEL_RESULTS_FILE, _LABEL_SUMMARY_FILE)

    elif approach in SCORE_APPROACHES:
        results_df = _run_score_judge_evaluation(
            context, judge_specs, plain_payload, approach, status_rows
        )
        summary_df = _score_summary_dataframe(results_df)
        _save_results(context, results_df, summary_df, _SCORE_RESULTS_FILE, _SCORE_SUMMARY_FILE)

    else:  # BINARY_APPROACHES
        results_df = _run_binary_judge_evaluation(
            context, judge_specs, plain_payload, approach, status_rows
        )
        summary_df = _binary_summary_dataframe(results_df)
        _save_results(context, results_df, summary_df, _BINARY_RESULTS_FILE, _BINARY_SUMMARY_FILE)

    logger.info("[judge] complete: %d result(s) collected for approach=%s", len(results_df), approach)
    return results_df, summary_df


def run_judge_evaluation_all_approaches(
    context: ExperimentContext,
    judge_specs: list[ModelSpec],
    plain_payload: dict[str, Any] | None = None,
    approaches: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run multiple judge approaches and save combined results per judge type.

    *approaches* defaults to all entries in ``JUDGE_APPROACHES`` when omitted.
    Only result files for the judge types actually run are written.

    Returns ``(combined_label_df, combined_label_summary_df)`` when any label
    approach was requested; otherwise returns the first non-empty result type.
    """
    if plain_payload is None:
        plain_payload = _load_plain_payload(context)

    selected = approaches if approaches is not None else JUDGE_APPROACHES
    unknown = [a for a in selected if a not in JUDGE_APPROACHES]
    if unknown:
        raise ValueError(f"Unknown approach(s) {unknown}. Must be one of {JUDGE_APPROACHES}.")

    label_dfs: list[pd.DataFrame] = []
    score_dfs: list[pd.DataFrame] = []
    binary_dfs: list[pd.DataFrame] = []

    status_rows = _load_judge_status()
    _save_judge_status(status_rows)

    for appr in selected:
        logger.info("[judge] ── approach: %s ──", appr)
        if appr in LABEL_APPROACHES:
            df = _run_label_judge_evaluation(
                context, judge_specs, plain_payload, appr, status_rows
            )
            label_dfs.append(df)
        elif appr in SCORE_APPROACHES:
            df = _run_score_judge_evaluation(
                context, judge_specs, plain_payload, appr, status_rows
            )
            score_dfs.append(df)
        else:
            df = _run_binary_judge_evaluation(
                context, judge_specs, plain_payload, appr, status_rows
            )
            binary_dfs.append(df)

    combined_label = pd.concat(label_dfs, ignore_index=True) if label_dfs else pd.DataFrame()
    combined_score = pd.concat(score_dfs, ignore_index=True) if score_dfs else pd.DataFrame()
    combined_binary = pd.concat(binary_dfs, ignore_index=True) if binary_dfs else pd.DataFrame()

    label_summary = _label_summary_dataframe(combined_label)
    score_summary = _score_summary_dataframe(combined_score)
    binary_summary = _binary_summary_dataframe(combined_binary)

    if label_dfs:
        _save_results(context, combined_label, label_summary, _LABEL_RESULTS_FILE, _LABEL_SUMMARY_FILE)
    if score_dfs:
        _save_results(context, combined_score, score_summary, _SCORE_RESULTS_FILE, _SCORE_SUMMARY_FILE)
    if binary_dfs:
        _save_results(context, combined_binary, binary_summary, _BINARY_RESULTS_FILE, _BINARY_SUMMARY_FILE)

    logger.info(
        "[judge] multi-approach complete: label=%d score=%d binary=%d",
        len(combined_label), len(combined_score), len(combined_binary),
    )
    if label_dfs:
        return combined_label, label_summary
    if score_dfs:
        return combined_score, score_summary
    return combined_binary, binary_summary
