"""Generate, validate, and capture candidate CP models per (LLM, language, label).

This is the data-generation phase: each round asks an LLM for a candidate model,
runs it, classifies the result, and stores it under a correctness label.

Two optimizations keep the workload small:
  1. Skip a target label that has already been captured for the current (LLM, language).
  2. Opportunistic capture: when a generated candidate happens to produce a different
     (but still useful) correctness label, store it under that label instead of discarding.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CORRECTNESS_LABELS, LANGUAGE_LABELS, ExperimentContext
from .executors import enumerate_solution_space, evaluate_candidate_code, instance_to_dict
from .generation import generate_candidate_model
from .llm import ModelSpec

logger = logging.getLogger(__name__)


DETAILED_PAYLOAD_FILENAME = "candidate-models-data-generation.json"
PLAIN_PAYLOAD_FILENAME = "candidate-models.json"
BENCHMARK_STATUS_FILENAME = "benchmark-status.csv"
BENCHMARK_STATUS_COLUMNS = [
    "problem_id",
    "generator_llm",
    "language",
    "target_label",
    "attempts",
    "succeed",
    "reference_space_size",
    "candidate_space_size",
    "fp",
    "fn",
    "updated_at",
]


def _benchmark_status_path() -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / BENCHMARK_STATUS_FILENAME


# ---------------------------------------------------------------------------
# Per-attempt helpers
# ---------------------------------------------------------------------------

def _build_feedback_refinement(eval_result: dict[str, Any], target_label: str) -> tuple[str, str]:
    if eval_result["exec_status"] == "non_executable":
        if eval_result.get("error_category") == "syntax_error":
            return (
                "syntax_repair",
                (
                    f"Previous attempt failed with syntax/parsing errors "
                    f"(detail='{eval_result['error_summary']}'). "
                    "Fix syntax only while preserving model intent and behavior."
                ),
            )
        return (
            "solver_validation_mismatch",
            (
                f"Previous attempt failed to execute "
                f"(category='{eval_result['error_category']}', detail='{eval_result['error_summary']}'). "
                f"Fix executable/runtime issues and keep targeting label '{target_label}'."
            ),
        )
    return (
        "solver_validation_mismatch",
        (
            f"Previous candidate produced observed label '{eval_result['observed_label']}' "
            f"(fp={eval_result['fp']}, fn={eval_result['fn']}), but the target is '{target_label}'. "
            f"Adjust constraints so the solution-space relation matches the requested label."
        ),
    )


def _build_candidate_record(
    code: str,
    eval_result: dict[str, Any],
    claimed_label: str,
    target_label: str,
    attempts_used: int,
    feedback_note: str,
    attempt_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "ok": True,
        "code": code,
        "claimed_label": claimed_label,
        "target_label_attempted": target_label,
        "attempts_used": attempts_used,
        "final_observed_label": eval_result["observed_label"],
        "final_exec_status": eval_result["exec_status"],
        "final_candidate_space_size": eval_result.get("candidate_space_size"),
        "final_fp": eval_result.get("fp"),
        "final_fn": eval_result.get("fn"),
        "final_candidate_truncated": eval_result.get("candidate_truncated", False),
        "final_error_category": eval_result["error_category"],
        "final_error_summary": eval_result["error_summary"],
        "feedback_last": feedback_note,
        "attempt_history": list(attempt_history),
    }


def _attempt_history_entry(attempt_idx: int, eval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt": attempt_idx,
        "exec_status": eval_result["exec_status"],
        "observed_label": eval_result["observed_label"],
        "label_match": eval_result["label_match"],
        "error_category": eval_result["error_category"],
        "error_summary": eval_result["error_summary"],
    }


# ---------------------------------------------------------------------------
# Per-(llm, language) capture loop
# ---------------------------------------------------------------------------

def _capture_labels_for_language(
    context: ExperimentContext,
    spec: ModelSpec,
    language: str,
    correctness_labels: list[str],
    instance_dict: dict[str, Any],
    reference_space: set,
    max_generation_rounds: int,
    initial_captured: dict[str, dict[str, Any]] | None = None,
    on_progress: Any = None,
    on_target_complete: Any = None,
) -> dict[str, dict[str, Any]]:
    slot_prefix = f"[{spec.raw} | {language}]"
    captured_by_label: dict[str, dict[str, Any]] = dict(initial_captured or {})
    if captured_by_label:
        logger.info(
            "%s resuming from disk: %d/%d label(s) already captured [%s]",
            slot_prefix,
            len(captured_by_label),
            len(correctness_labels),
            ", ".join(captured_by_label.keys()),
        )
    logger.info(
        "%s starting candidate generation for %d target labels",
        slot_prefix,
        len(correctness_labels),
    )

    for target_label in correctness_labels:
        if target_label in captured_by_label:
            logger.info(
                "%s target='%s': already captured opportunistically, skipping",
                slot_prefix,
                target_label,
            )
            if on_target_complete is not None:
                existing = captured_by_label[target_label]
                try:
                    on_target_complete(
                        target_label=target_label,
                        attempts=int(existing.get("attempts_used", 0) or 0),
                        succeed=True,
                        last_exec_status=str(existing.get("final_exec_status", "")),
                        candidate_space_size=existing.get("final_candidate_space_size"),
                        fp=existing.get("final_fp"),
                        fn=existing.get("final_fn"),
                    )
                except Exception:
                    logger.exception("%s status update failed for target='%s'", slot_prefix, target_label)
            continue

        logger.info(
            "%s target='%s': begin (max %d attempt(s))",
            slot_prefix,
            target_label,
            max_generation_rounds,
        )
        feedback_note = ""
        feedback_mode = ""
        attempt_history: list[dict[str, Any]] = []
        previous_attempt = ""
        last_eval_result: dict[str, Any] | None = None
        for attempt_idx in range(1, max_generation_rounds + 1):
            candidate_code = generate_candidate_model(
                context=context,
                spec=spec,
                language=language,
                label=target_label,
                feedback_notes=feedback_note,
                feedback_mode=feedback_mode,
                previous_attempt=previous_attempt,
            )
            previous_attempt = candidate_code
            eval_result = evaluate_candidate_code(
                context=context,
                language=language,
                code=candidate_code,
                claimed_label=target_label,
                instance_dict=instance_dict,
                reference_space=reference_space,
            )
            last_eval_result = eval_result
            attempt_history.append(_attempt_history_entry(attempt_idx, eval_result))
            observed_label = eval_result["observed_label"]
            logger.info(
                "%s target='%s' attempt %d/%d: observed='%s', match=%s, space_size=%s, fp=%s, fn=%s",
                slot_prefix,
                target_label,
                attempt_idx,
                max_generation_rounds,
                observed_label,
                eval_result["label_match"],
                eval_result["candidate_space_size"],
                eval_result["fp"],
                eval_result["fn"],
            )

            captured_changed = False
            if observed_label in correctness_labels and observed_label not in captured_by_label:
                captured_by_label[observed_label] = _build_candidate_record(
                    code=candidate_code,
                    eval_result=eval_result,
                    claimed_label=observed_label,
                    target_label=target_label,
                    attempts_used=attempt_idx,
                    feedback_note=feedback_note,
                    attempt_history=attempt_history,
                )
                captured_changed = True
                if observed_label != target_label:
                    logger.info(
                        "%s target='%s': opportunistic capture under '%s'",
                        slot_prefix,
                        target_label,
                        observed_label,
                    )

            if eval_result["label_match"] and target_label not in captured_by_label:
                captured_by_label[target_label] = _build_candidate_record(
                    code=candidate_code,
                    eval_result=eval_result,
                    claimed_label=target_label,
                    target_label=target_label,
                    attempts_used=attempt_idx,
                    feedback_note=feedback_note,
                    attempt_history=attempt_history,
                )
                captured_changed = True

            if captured_changed and on_progress is not None:
                try:
                    on_progress(captured_by_label)
                except Exception:
                    logger.exception("%s checkpoint write failed; continuing", slot_prefix)

            if target_label in captured_by_label:
                logger.info(
                    "%s target='%s': captured after %d attempt(s)",
                    slot_prefix,
                    target_label,
                    attempt_idx,
                )
                if on_target_complete is not None:
                    record = captured_by_label[target_label]
                    try:
                        on_target_complete(
                            target_label=target_label,
                            attempts=attempt_idx,
                            succeed=True,
                            last_exec_status=str(record.get("final_exec_status", "")),
                            candidate_space_size=record.get("final_candidate_space_size"),
                            fp=record.get("final_fp"),
                            fn=record.get("final_fn"),
                        )
                    except Exception:
                        logger.exception("%s status update failed for target='%s'", slot_prefix, target_label)
                break

            feedback_mode, feedback_note = _build_feedback_refinement(eval_result, target_label)
        else:
            logger.info(
                "%s target='%s': not captured after %d attempt(s)",
                slot_prefix,
                target_label,
                max_generation_rounds,
            )
            if on_target_complete is not None:
                last = last_eval_result or {}
                try:
                    on_target_complete(
                        target_label=target_label,
                        attempts=max_generation_rounds,
                        succeed=False,
                        last_exec_status=str(last.get("exec_status", "")),
                        candidate_space_size=last.get("candidate_space_size"),
                        fp=last.get("fp"),
                        fn=last.get("fn"),
                    )
                except Exception:
                    logger.exception("%s status update failed for target='%s'", slot_prefix, target_label)

    logger.info(
        "%s done: captured %d/%d labels [%s]",
        slot_prefix,
        len(captured_by_label),
        len(correctness_labels),
        ", ".join(captured_by_label.keys()) or "-",
    )
    return captured_by_label


# ---------------------------------------------------------------------------
# Output shaping
# ---------------------------------------------------------------------------

def _row_from_record(
    llm_name: str, language: str, label: str, record: dict[str, Any]
) -> dict[str, Any]:
    return {
        "llm": llm_name,
        "language": language,
        "claimed_label": label,
        "target_label_attempted": record["target_label_attempted"],
        "attempts_used": record["attempts_used"],
        "exec_status": record["final_exec_status"],
        "observed_label": record["final_observed_label"],
        "label_match": record["ok"],
        "candidate_space_size": record.get("final_candidate_space_size"),
        "fp": record.get("final_fp"),
        "fn": record.get("final_fn"),
        "candidate_truncated": record.get("final_candidate_truncated", False),
        "error_category": record["final_error_category"],
        "error_summary": record["final_error_summary"],
    }


def _summary_dataframe(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(columns=["llm", "language", "match_rate"])
    return (
        rows_df.groupby(["llm", "language"], dropna=False)["label_match"]
        .mean()
        .reset_index(name="match_rate")
        .sort_values(["llm", "language"])
    )


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON via a temp file + rename so a crash mid-write cannot corrupt `path`."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _plain_from_detailed(detailed_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        llm_name: {
            language: {
                label: {"ok": record["ok"], "code": record["code"]}
                for label, record in by_label.items()
            }
            for language, by_label in by_language.items()
        }
        for llm_name, by_language in detailed_payload.items()
    }


def _save_payloads(
    context: ExperimentContext,
    detailed_payload: dict[str, Any],
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Persist both detailed and plain payloads atomically. Returns plain payload."""
    output_dir = context.output_dir
    detailed_path = output_dir / DETAILED_PAYLOAD_FILENAME
    plain_path = output_dir / PLAIN_PAYLOAD_FILENAME
    plain_payload = _plain_from_detailed(detailed_payload)
    _atomic_write_json(detailed_path, detailed_payload)
    _atomic_write_json(plain_path, plain_payload)
    if verbose:
        logger.info("[data-gen] saved detailed payload: %s", detailed_path)
        logger.info("[data-gen] saved plain payload:    %s", plain_path)
    return plain_payload


def _load_existing_detailed_payload(context: ExperimentContext) -> dict[str, Any]:
    """Load a previously-saved detailed payload if present, else return empty."""
    path = context.output_dir / DETAILED_PAYLOAD_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "[data-gen] could not load existing payload at %s (%s); starting fresh",
            path,
            e,
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "[data-gen] existing payload at %s is not a JSON object; starting fresh",
            path,
        )
        return {}
    return data


def _status_key(
    problem_id: str, generator_llm: str, language: str, target_label: str
) -> tuple[str, str, str, str]:
    return (problem_id, generator_llm, language, target_label)


def _load_benchmark_status(context: ExperimentContext) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    path = _benchmark_status_path()
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        logger.warning(
            "[data-gen] could not read benchmark status at %s (%s); starting fresh",
            path,
            e,
        )
        return {}

    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        key = _status_key(
            str(row.get("problem_id", "")),
            str(row.get("generator_llm", "")),
            str(row.get("language", "")),
            str(row.get("target_label", "")),
        )
        if all(key):
            rows[key] = row
    return rows


def _save_benchmark_status(
    context: ExperimentContext, status_rows: dict[tuple[str, str, str, str], dict[str, Any]]
) -> None:
    path = _benchmark_status_path()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    records = [status_rows[k] for k in sorted(status_rows.keys())]
    df = pd.DataFrame(records)
    for col in BENCHMARK_STATUS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[BENCHMARK_STATUS_COLUMNS]
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, path)


def _update_benchmark_status(
    context: ExperimentContext,
    status_rows: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    generator_llm: str,
    language: str,
    target_label: str,
    attempts: int,
    succeed: bool,
    last_exec_status: str,
    reference_space_size: int | None = None,
    candidate_space_size: int | None = None,
    fp: int | None = None,
    fn: int | None = None,
) -> None:
    _ = last_exec_status
    key = _status_key(context.targeted_id, generator_llm, language, target_label)
    status_rows[key] = {
        "problem_id": context.targeted_id,
        "generator_llm": generator_llm,
        "language": language,
        "target_label": target_label,
        "attempts": str(attempts),
        "succeed": "yes" if succeed else "no",
        "reference_space_size": "" if reference_space_size is None else str(reference_space_size),
        "candidate_space_size": "" if candidate_space_size is None else str(candidate_space_size),
        "fp": "" if fp is None else str(fp),
        "fn": "" if fn is None else str(fn),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_benchmark_status(context, status_rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_data_generation(
    context: ExperimentContext,
    model_specs: list[ModelSpec],
    language_labels: list[str] | None = None,
    correctness_labels: list[str] | None = None,
    max_generation_rounds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    language_labels = language_labels or LANGUAGE_LABELS
    correctness_labels = correctness_labels or CORRECTNESS_LABELS

    logger.info(
        "[data-gen] starting: problem='%s', llms=%d, languages=%d, labels=%d, max_rounds=%d",
        context.targeted_id,
        len(model_specs),
        len(language_labels),
        len(correctness_labels),
        max_generation_rounds,
    )

    instance_dict = instance_to_dict(context.example_instance, context.data_instances)
    reference_exec, reference_space = enumerate_solution_space(
        code_text=context.reference_cp_model,
        language=context.reference_language,
        instance_dict=instance_dict,
        decision_var_names=context.decision_variables,
        solution_limit=context.solution_limit,
        time_limit_sec=context.time_limit_for(context.reference_language),
    )
    if not reference_exec["ok"]:
        raise RuntimeError(
            f"Reference model failed: {reference_exec.get('error', 'unknown error')}"
        )
    reference_space_size = len(reference_space)
    logger.info(
        "[data-gen] reference model OK (language='%s', solution-space size=%d)",
        context.reference_language,
        reference_space_size,
    )

    detailed_payload: dict[str, Any] = _load_existing_detailed_payload(context)
    if detailed_payload:
        already_captured = sum(
            len(by_label)
            for by_language in detailed_payload.values()
            for by_label in by_language.values()
        )
        logger.info(
            "[data-gen] resuming from existing payload: %d candidate(s) already on disk",
            already_captured,
        )
    status_rows = _load_benchmark_status(context)
    if status_rows:
        logger.info(
            "[data-gen] loaded benchmark status: %d row(s) from %s",
            len(status_rows),
            _benchmark_status_path(),
        )
    else:
        logger.info("[data-gen] creating benchmark status file: %s", _benchmark_status_path())
    _save_benchmark_status(context, status_rows)

    plain_payload: dict[str, Any] = {}
    try:
        for spec in model_specs:
            llm_name = spec.raw
            detailed_payload.setdefault(llm_name, {})

            for language in language_labels:
                existing_for_slot = detailed_payload[llm_name].get(language, {})

                def _checkpoint(captured: dict[str, dict[str, Any]],
                                 _llm: str = llm_name,
                                 _lang: str = language) -> None:
                    detailed_payload[_llm][_lang] = captured
                    _save_payloads(context, detailed_payload, verbose=False)

                def _status_checkpoint(
                    *,
                    target_label: str,
                    attempts: int,
                    succeed: bool,
                    last_exec_status: str,
                    candidate_space_size: int | None = None,
                    fp: int | None = None,
                    fn: int | None = None,
                    _llm: str = llm_name,
                    _lang: str = language,
                ) -> None:
                    _update_benchmark_status(
                        context,
                        status_rows,
                        generator_llm=_llm,
                        language=_lang,
                        target_label=target_label,
                        attempts=attempts,
                        succeed=succeed,
                        last_exec_status=last_exec_status,
                        reference_space_size=reference_space_size,
                        candidate_space_size=candidate_space_size,
                        fp=fp,
                        fn=fn,
                    )

                captured_by_label = _capture_labels_for_language(
                    context=context,
                    spec=spec,
                    language=language,
                    correctness_labels=correctness_labels,
                    instance_dict=instance_dict,
                    reference_space=reference_space,
                    max_generation_rounds=max_generation_rounds,
                    initial_captured=existing_for_slot,
                    on_progress=_checkpoint,
                    on_target_complete=_status_checkpoint,
                )

                detailed_payload[llm_name][language] = captured_by_label
                _save_payloads(context, detailed_payload)
    except BaseException:
        logger.exception(
            "[data-gen] unhandled error during run; saving current state before re-raising"
        )
        raise
    finally:
        plain_payload = _save_payloads(context, detailed_payload, verbose=False)

    rows = [
        _row_from_record(llm_name, language, label, record)
        for llm_name, by_language in detailed_payload.items()
        for language, by_label in by_language.items()
        for label, record in by_label.items()
    ]
    results_df = pd.DataFrame(rows)
    logger.info("[data-gen] complete: %d candidate(s) captured across all slots", len(rows))
    return results_df, _summary_dataframe(results_df), plain_payload
