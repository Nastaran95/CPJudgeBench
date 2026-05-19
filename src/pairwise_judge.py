"""Pairwise LLM-as-judge: compare two CP model candidates head-to-head.

The judge sees both candidate codes (labelled A and B) together with the
problem description and picks the better one, or declares a tie.

Output per pair: winner in {"A", "B", "tie"}, confidence, rationale.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentContext
from .llm import ModelSpec, get_openrouter_llm, llm_response_to_text
from .parsing import parse_pairwise_judge_json

logger = logging.getLogger(__name__)

PAIRWISE_STATUS_FILENAME = "pairwise-judge-status.csv"
PAIRWISE_STATUS_COLUMNS = [
    "problem_id",
    "language",
    "generator_llm_a",
    "claimed_label_a",
    "generator_llm_b",
    "claimed_label_b",
    "judge_llm",
    "winner",
    "succeed",
    "description",
    "updated_at",
]

PAIRWISE_RESULTS_FILENAME = "pairwise-results.csv"
PAIRWISE_SUMMARY_FILENAME = "pairwise-summary.csv"

_PAIRWISE_RESULT_COLUMNS = [
    "judge_llm",
    "language",
    "generator_llm_a",
    "claimed_label_a",
    "generator_llm_b",
    "claimed_label_b",
    "winner",
    "confidence",
    "rationale",
    "raw_reply",
]


# ---------------------------------------------------------------------------
# Status file helpers
# ---------------------------------------------------------------------------

def _pairwise_status_path() -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / PAIRWISE_STATUS_FILENAME


_StatusKey = tuple[str, str, str, str, str, str, str]


def _pairwise_status_key(
    problem_id: str,
    language: str,
    gen_a: str,
    label_a: str,
    gen_b: str,
    label_b: str,
    judge_llm: str,
) -> _StatusKey:
    return (problem_id, language, gen_a, label_a, gen_b, label_b, judge_llm)


def _load_pairwise_status() -> dict[_StatusKey, dict[str, Any]]:
    path = _pairwise_status_path()
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        logger.warning("[pairwise] could not read status file %s (%s); starting fresh", path, e)
        return {}

    rows: dict[_StatusKey, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        key = _pairwise_status_key(
            str(row.get("problem_id", "")),
            str(row.get("language", "")),
            str(row.get("generator_llm_a", "")),
            str(row.get("claimed_label_a", "")),
            str(row.get("generator_llm_b", "")),
            str(row.get("claimed_label_b", "")),
            str(row.get("judge_llm", "")),
        )
        if all(k for k in key[:4]):
            rows[key] = row
    return rows


def _save_pairwise_status(rows: dict[_StatusKey, dict[str, Any]]) -> None:
    path = _pairwise_status_path()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    records = [rows[k] for k in sorted(rows.keys())]
    df = pd.DataFrame(records)
    for col in PAIRWISE_STATUS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[PAIRWISE_STATUS_COLUMNS]
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, path)


def _update_pairwise_status(
    rows: dict[_StatusKey, dict[str, Any]],
    *,
    problem_id: str,
    language: str,
    gen_a: str,
    label_a: str,
    gen_b: str,
    label_b: str,
    judge_llm: str,
    winner: str,
    succeed: bool,
    description: str,
) -> None:
    key = _pairwise_status_key(problem_id, language, gen_a, label_a, gen_b, label_b, judge_llm)
    rows[key] = {
        "problem_id": problem_id,
        "language": language,
        "generator_llm_a": gen_a,
        "claimed_label_a": label_a,
        "generator_llm_b": gen_b,
        "claimed_label_b": label_b,
        "judge_llm": judge_llm,
        "winner": winner,
        "succeed": "yes" if succeed else "no",
        "description": description,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_pairwise_status(rows)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _pairwise_prompt(
    context: ExperimentContext,
    code_a: str,
    code_b: str,
    language: str,
) -> str:
    return f"""
You are an expert CP model judge.
Compare the two candidate models below and decide which better solves the problem.

Allowed decisions:
  "A"   : Candidate A is clearly better (more correct, more complete, or more sound).
  "B"   : Candidate B is clearly better.
  "tie" : Both are equally good or equally flawed; no meaningful difference.

Output format (STRICT JSON only):
{{"winner": "<A|B|tie>", "confidence": <0..1>, "rationale": "<1-3 short sentences>"}}

Problem description:
{context.problem_description}

Candidate A ({language}):
{code_a}

Candidate B ({language}):
{code_b}
""".strip()


# ---------------------------------------------------------------------------
# Core judge call
# ---------------------------------------------------------------------------

def direct_pairwise_judge(
    context: ExperimentContext,
    spec: ModelSpec,
    code_a: str,
    code_b: str,
    language: str,
) -> tuple[str, dict[str, Any], str]:
    """Ask a single LLM to compare two candidates.

    Returns ``(winner, parsed_dict, raw_reply)`` where
    ``winner`` is one of ``"A"``, ``"B"``, ``"tie"``, or ``"unknown"``.
    """
    llm = get_openrouter_llm(spec)
    raw_reply = llm_response_to_text(
        llm.invoke(_pairwise_prompt(context, code_a, code_b, language))
    ).strip()
    parsed = parse_pairwise_judge_json(raw_reply)
    winner = str(parsed.get("winner", "unknown"))
    return winner, parsed, raw_reply


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def _collect_judgeable_candidates(
    plain_payload: dict[str, Any],
) -> dict[str, list[tuple[str, str, str]]]:
    """Return {language: [(generator_llm, claimed_label, code), ...]} for executable candidates."""
    by_language: dict[str, list[tuple[str, str, str]]] = {}
    for generator_llm, by_lang in plain_payload.items():
        for language, by_label in by_lang.items():
            for claimed_label, meta in by_label.items():
                if (
                    isinstance(meta, dict)
                    and meta.get("ok")
                    and str(meta.get("code", "")).strip()
                ):
                    by_language.setdefault(language, []).append(
                        (generator_llm, claimed_label, str(meta["code"]))
                    )
    return by_language


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _build_pairwise_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-candidate win/loss/tie counts and win rate across all judges."""
    if results_df.empty:
        return pd.DataFrame(
            columns=["judge_llm", "language", "generator_llm", "claimed_label",
                     "wins", "losses", "ties", "total", "win_rate"]
        )

    records: list[dict[str, Any]] = []

    def _add(judge_llm: str, language: str, gen: str, label: str,
             wins: int, losses: int, ties: int) -> None:
        total = wins + losses + ties
        records.append({
            "judge_llm": judge_llm,
            "language": language,
            "generator_llm": gen,
            "claimed_label": label,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "total": total,
            "win_rate": round(wins / (wins + losses), 4) if (wins + losses) > 0 else None,
        })

    grouped = results_df.groupby(["judge_llm", "language"])
    for (judge_llm, language), grp in grouped:
        # Collect all unique (generator_llm, claimed_label) identifiers in this group.
        candidates: set[tuple[str, str]] = set()
        for _, row in grp.iterrows():
            candidates.add((str(row["generator_llm_a"]), str(row["claimed_label_a"])))
            candidates.add((str(row["generator_llm_b"]), str(row["claimed_label_b"])))

        for gen, label in sorted(candidates):
            wins = int((
                ((grp["generator_llm_a"] == gen) & (grp["claimed_label_a"] == label) & (grp["winner"] == "A"))
                | ((grp["generator_llm_b"] == gen) & (grp["claimed_label_b"] == label) & (grp["winner"] == "B"))
            ).sum())
            losses = int((
                ((grp["generator_llm_a"] == gen) & (grp["claimed_label_a"] == label) & (grp["winner"] == "B"))
                | ((grp["generator_llm_b"] == gen) & (grp["claimed_label_b"] == label) & (grp["winner"] == "A"))
            ).sum())
            ties = int((
                ((grp["generator_llm_a"] == gen) & (grp["claimed_label_a"] == label) & (grp["winner"] == "tie"))
                | ((grp["generator_llm_b"] == gen) & (grp["claimed_label_b"] == label) & (grp["winner"] == "tie"))
            ).sum())
            _add(str(judge_llm), str(language), gen, label, wins, losses, ties)

    return pd.DataFrame(records)


def _save_pairwise_results(
    context: ExperimentContext,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    output_dir = context.output_dir
    res_path = output_dir / PAIRWISE_RESULTS_FILENAME
    sum_path = output_dir / PAIRWISE_SUMMARY_FILENAME

    tmp_r = res_path.with_suffix(res_path.suffix + ".tmp")
    tmp_s = sum_path.with_suffix(sum_path.suffix + ".tmp")
    results_df.to_csv(tmp_r, index=False, encoding="utf-8")
    summary_df.to_csv(tmp_s, index=False, encoding="utf-8")
    os.replace(tmp_r, res_path)
    os.replace(tmp_s, sum_path)
    logger.info("[pairwise] results  -> %s", res_path)
    logger.info("[pairwise] summary  -> %s", sum_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pairwise_evaluation(
    context: ExperimentContext,
    judge_specs: list[ModelSpec],
    plain_payload: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run pairwise comparison for all candidate pairs within the same language.

    For every unique pair of executable candidates (within the same language) and
    every judge model, calls ``direct_pairwise_judge`` and records the winner.

    Uses a status file (``logs/pairwise-judge-status.csv``) to skip pairs that
    have already been evaluated.

    Returns ``(results_df, summary_df)``.
    """
    if plain_payload is None:
        path = context.output_dir / "candidate-models.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Candidate payload not found at {path}. "
                "Run the data-generation phase first (--data-generation)."
            )
        plain_payload = json.loads(path.read_text(encoding="utf-8"))

    status_rows = _load_pairwise_status()
    if status_rows:
        logger.info(
            "[pairwise] loaded existing status: %d row(s) from %s",
            len(status_rows),
            _pairwise_status_path(),
        )
    _save_pairwise_status(status_rows)

    by_language = _collect_judgeable_candidates(plain_payload)
    total_pairs = sum(
        len(list(itertools.combinations(cands, 2)))
        for cands in by_language.values()
    )
    logger.info(
        "[pairwise] starting: judges=%d, languages=%d, total pairs=%d",
        len(judge_specs),
        len(by_language),
        total_pairs,
    )

    result_rows: list[dict[str, Any]] = []

    for judge_spec in judge_specs:
        judge_prefix = f"[pairwise-judge={judge_spec.raw}]"
        judged_count = 0
        skipped_count = 0
        failed_count = 0

        for language, candidates in by_language.items():
            for (gen_a, label_a, code_a), (gen_b, label_b, code_b) in itertools.combinations(candidates, 2):
                slot = f"{language} | A=({gen_a},{label_a}) vs B=({gen_b},{label_b})"
                key = _pairwise_status_key(
                    context.targeted_id, language, gen_a, label_a, gen_b, label_b, judge_spec.raw
                )
                if key in status_rows:
                    logger.info("%s %s -> skipped (already done)", judge_prefix, slot)
                    skipped_count += 1
                    continue

                try:
                    winner, parsed, raw_reply = direct_pairwise_judge(
                        context, judge_spec, code_a, code_b, language
                    )
                except Exception as e:
                    logger.exception(
                        "%s %s -> failed (%s: %s)", judge_prefix, slot, type(e).__name__, e
                    )
                    _update_pairwise_status(
                        status_rows,
                        problem_id=context.targeted_id,
                        language=language,
                        gen_a=gen_a,
                        label_a=label_a,
                        gen_b=gen_b,
                        label_b=label_b,
                        judge_llm=judge_spec.raw,
                        winner="error",
                        succeed=False,
                        description=f"{type(e).__name__}: {e}",
                    )
                    failed_count += 1
                    continue

                confidence = parsed.get("confidence")
                confidence_text = (
                    f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
                )
                logger.info(
                    "%s %s -> winner=%s confidence=%s",
                    judge_prefix, slot, winner, confidence_text
                )
                _update_pairwise_status(
                    status_rows,
                    problem_id=context.targeted_id,
                    language=language,
                    gen_a=gen_a,
                    label_a=label_a,
                    gen_b=gen_b,
                    label_b=label_b,
                    judge_llm=judge_spec.raw,
                    winner=winner,
                    succeed=winner in {"A", "B", "tie"},
                    description=str(parsed.get("rationale", "")),
                )

                judged_count += 1
                result_rows.append({
                    "judge_llm": judge_spec.raw,
                    "language": language,
                    "generator_llm_a": gen_a,
                    "claimed_label_a": label_a,
                    "generator_llm_b": gen_b,
                    "claimed_label_b": label_b,
                    "winner": winner,
                    "confidence": confidence,
                    "rationale": parsed.get("rationale", ""),
                    "raw_reply": raw_reply,
                })

        logger.info(
            "%s done: judged=%d skipped=%d failed=%d",
            judge_prefix, judged_count, skipped_count, failed_count,
        )

    results_df = (
        pd.DataFrame(result_rows, columns=_PAIRWISE_RESULT_COLUMNS)
        if result_rows
        else pd.DataFrame(columns=_PAIRWISE_RESULT_COLUMNS)
    )
    summary_df = _build_pairwise_summary(results_df)
    _save_pairwise_results(context, results_df, summary_df)
    logger.info("[pairwise] complete: %d comparison(s) collected", len(result_rows))
    return results_df, summary_df
