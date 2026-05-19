"""SAT-only meta-evaluation: compare a lightweight solve-once approach against the
full solution-space ground truth.

The SAT-only approach labels a candidate as *correct* when:
  - Candidate is UNSAT  AND reference space is empty (UNSAT), OR
  - Candidate is SAT    AND reference space is non-empty (SAT) AND
    the single returned solution is present in the reference space.

This module runs that labelling for every executable candidate and compares it
against the ground-truth correctness derived from the full-space enumeration
(``final_observed_label == "equivalent"``), producing precision/recall/F1/accuracy
statistics so the two evaluation strategies can be compared.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentContext
from .executors import (
    enumerate_solution_space,
    evaluate_candidate_code,
    evaluate_candidate_sat_only,
    instance_to_dict,
)

logger = logging.getLogger(__name__)

SAT_META_EVAL_RESULTS_FILENAME = "sat-meta-eval-results.csv"
SAT_META_EVAL_SUMMARY_FILENAME = "sat-meta-eval-summary.csv"

_RESULT_COLUMNS = [
    "generator_llm",
    "language",
    "claimed_label",
    "full_space_observed_label",
    "full_space_correct",
    "exec_status",
    "reference_sat",
    "candidate_sat",
    "solution_in_ref",
    "sat_only_correct",
    "error",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_plain_payload(context: ExperimentContext) -> dict[str, Any]:
    path = context.output_dir / "candidate-models.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate payload not found at {path}. "
            "Run the data-generation phase first (--data-generation)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_data_gen_payload(context: ExperimentContext) -> dict[str, Any] | None:
    path = context.output_dir / "candidate-models-data-generation.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[sat-meta-eval] could not load data-gen payload (%s); will re-run full-space eval", e)
        return None


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _confusion_row(
    view: str,
    label: str,
    language: str,
    sub_df: pd.DataFrame,
) -> dict[str, Any]:
    total = len(sub_df)
    valid = sub_df[sub_df["exec_status"] == "ok"]
    tp = int(((valid["sat_only_correct"] == True) & (valid["full_space_correct"] == True)).sum())  # noqa: E712
    fp = int(((valid["sat_only_correct"] == True) & (valid["full_space_correct"] == False)).sum())  # noqa: E712
    tn = int(((valid["sat_only_correct"] == False) & (valid["full_space_correct"] == False)).sum())  # noqa: E712
    fn = int(((valid["sat_only_correct"] == False) & (valid["full_space_correct"] == True)).sum())  # noqa: E712
    n_valid = tp + fp + tn + fn
    accuracy = (tp + tn) / n_valid if n_valid > 0 else 0.0
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    return {
        "view": view,
        "label": label,
        "language": language,
        "total": total,
        "evaluated": n_valid,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rows.append(_confusion_row("overall", "", "", df))

    for lang, g in df.groupby("language", dropna=False):
        rows.append(_confusion_row("by_language", "", str(lang), g))

    for lbl, g in df.groupby("claimed_label", dropna=False):
        rows.append(_confusion_row("by_claimed_label", str(lbl), "", g))

    for (lang, lbl), g in df.groupby(["language", "claimed_label"], dropna=False):
        rows.append(_confusion_row("by_language_label", str(lbl), str(lang), g))

    return pd.DataFrame(rows)


def _save_results(
    context: ExperimentContext,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    output_dir = context.output_dir
    results_path = output_dir / SAT_META_EVAL_RESULTS_FILENAME
    summary_path = output_dir / SAT_META_EVAL_SUMMARY_FILENAME

    tmp_r = results_path.with_suffix(results_path.suffix + ".tmp")
    tmp_s = summary_path.with_suffix(summary_path.suffix + ".tmp")
    results_df.to_csv(tmp_r, index=False, encoding="utf-8")
    summary_df.to_csv(tmp_s, index=False, encoding="utf-8")
    os.replace(tmp_r, results_path)
    os.replace(tmp_s, summary_path)
    logger.info("[sat-meta-eval] results  -> %s", results_path)
    logger.info("[sat-meta-eval] summary  -> %s", summary_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sat_meta_evaluation(
    context: ExperimentContext,
    plain_payload: dict[str, Any] | None = None,
    data_gen_payload: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the SAT-only approach against the full-space ground truth.

    For every executable candidate the function:
    1. Determines *full_space_correct* (``final_observed_label == "equivalent"``),
       falling back to a fresh full-space evaluation when the data-gen payload is
       not available.
    2. Runs ``evaluate_candidate_sat_only`` to obtain *sat_only_correct*.
    3. Saves per-candidate results and a confusion-matrix summary.

    Returns ``(results_df, summary_df)``.
    """
    if plain_payload is None:
        plain_payload = _load_plain_payload(context)
    if data_gen_payload is None:
        data_gen_payload = _load_data_gen_payload(context)

    # Build reference space once (uses first available instance).
    instance_dict = instance_to_dict(context.example_instance, context.data_instances)
    logger.info("[sat-meta-eval] enumerating reference space …")
    ref_exec, reference_space = enumerate_solution_space(
        code_text=context.reference_cp_model,
        language=context.reference_language,
        instance_dict=instance_dict,
        decision_var_names=context.decision_variables,
        solution_limit=context.solution_limit,
        time_limit_sec=context.time_limit_for(context.reference_language),
    )
    if not ref_exec["ok"]:
        raise RuntimeError(
            f"[sat-meta-eval] Reference model failed to execute: {ref_exec.get('error')}"
        )
    logger.info("[sat-meta-eval] reference space size: %d", len(reference_space))

    total_candidates = sum(
        len(by_label)
        for by_language in plain_payload.values()
        for by_label in by_language.values()
    )
    logger.info(
        "[sat-meta-eval] starting: %d generator(s), %d candidate slots",
        len(plain_payload),
        total_candidates,
    )

    result_rows: list[dict[str, Any]] = []

    for generator_llm, by_language in plain_payload.items():
        for language, by_label in by_language.items():
            for claimed_label, meta in by_label.items():
                slot = f"gen={generator_llm} | {language} | claimed='{claimed_label}'"

                if not isinstance(meta, dict) or not meta.get("ok") or not str(meta.get("code", "")).strip():
                    logger.info("[sat-meta-eval] %s -> skipped (no usable code)", slot)
                    continue

                code = str(meta["code"])

                # --- ground-truth full-space label ----------------------------
                full_space_observed_label: str | None = None
                if data_gen_payload:
                    dg = (
                        data_gen_payload
                        .get(generator_llm, {})
                        .get(language, {})
                        .get(claimed_label, {})
                    )
                    full_space_observed_label = dg.get("final_observed_label")

                if full_space_observed_label is None:
                    logger.info("[sat-meta-eval] %s -> re-running full-space eval (no data-gen record)", slot)
                    full_eval = evaluate_candidate_code(
                        context=context,
                        language=language,
                        code=code,
                        claimed_label=claimed_label,
                        instance_dict=instance_dict,
                        reference_space=reference_space,
                    )
                    full_space_observed_label = str(full_eval.get("observed_label", "unknown"))

                full_space_correct: bool = full_space_observed_label == "equivalent"

                # --- SAT-only evaluation --------------------------------------
                sat_result = evaluate_candidate_sat_only(
                    context=context,
                    language=language,
                    code=code,
                    instance_dict=instance_dict,
                    reference_space=reference_space,
                )
                logger.info(
                    "[sat-meta-eval] %s -> full_space=%s sat_only=%s",
                    slot,
                    full_space_observed_label,
                    sat_result.get("sat_only_correct"),
                )

                result_rows.append({
                    "generator_llm": generator_llm,
                    "language": language,
                    "claimed_label": claimed_label,
                    "full_space_observed_label": full_space_observed_label,
                    "full_space_correct": full_space_correct,
                    "exec_status": sat_result.get("exec_status"),
                    "reference_sat": sat_result.get("reference_sat"),
                    "candidate_sat": sat_result.get("candidate_sat"),
                    "solution_in_ref": sat_result.get("solution_in_ref"),
                    "sat_only_correct": sat_result.get("sat_only_correct"),
                    "error": sat_result.get("error", ""),
                })

    results_df = pd.DataFrame(result_rows, columns=_RESULT_COLUMNS) if result_rows else pd.DataFrame(columns=_RESULT_COLUMNS)
    summary_df = _build_summary(results_df) if not results_df.empty else pd.DataFrame()

    _save_results(context, results_df, summary_df)

    if not results_df.empty:
        valid = results_df[results_df["exec_status"] == "ok"]
        n = len(valid)
        if n > 0:
            agree = int((valid["sat_only_correct"] == valid["full_space_correct"]).sum())
            logger.info(
                "[sat-meta-eval] agreement: %d/%d (%.1f%%)",
                agree, n, 100.0 * agree / n,
            )

    logger.info("[sat-meta-eval] complete: %d candidate(s) evaluated", len(result_rows))
    return results_df, summary_df
