"""Build one attempt-level CSV for benchmark construction reliability analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_ROOT = Path("data-storage")
BENCHMARK_STATUS = Path("logs/benchmark-status.csv")
OUTPUT_PATH = Path("extra_files/preprocessed/data_generation_attempts.csv")


def _load_payloads() -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    for json_path in sorted(DATA_ROOT.glob("*/candidates/candidate-models-data-generation.json")):
        payloads[json_path.parts[-3]] = json.loads(json_path.read_text(encoding="utf-8"))
    return payloads


def _record_for_target(
    payload: dict, llm: str, language: str, target_label: str
) -> dict | None:
    by_label = payload.get(llm, {}).get(language, {})
    for claimed_label, record in by_label.items():
        if claimed_label == target_label:
            return record
    return None


def _session_history(record: dict | None, target_label: str) -> list[dict]:
    """Return attempt history for the generation session that filled *target_label*."""
    if record is None:
        return []
    if record.get("target_label_attempted") == target_label:
        return list(record.get("attempt_history", []))
    # Opportunistically pre-filled while targeting another label.
    return list(record.get("attempt_history", []))


def load_capture_rows(payloads: dict[str, dict]) -> pd.DataFrame:
    capture_rows: list[dict] = []
    for problem_id, payload in payloads.items():
        for llm, by_lang in payload.items():
            for language, by_label in by_lang.items():
                for claimed_label, record in by_label.items():
                    target = record.get("target_label_attempted", claimed_label)
                    capture_rows.append(
                        {
                            "problem_id": problem_id,
                            "generator_llm": llm,
                            "language": language,
                            "slot_label": claimed_label,
                            "target_label_attempted": target,
                            "opportunistic_capture": claimed_label != target,
                        }
                    )
    return pd.DataFrame(capture_rows)


def build_targets_df(captures_df: pd.DataFrame) -> pd.DataFrame:
    status_df = pd.read_csv(BENCHMARK_STATUS)
    status_df["succeed"] = status_df["succeed"].map({"yes": True, "no": False})

    cap_lookup = (
        captures_df.groupby(["problem_id", "generator_llm", "language", "slot_label"], dropna=False)
        .agg(
            opportunistic_capture=("opportunistic_capture", "any"),
            target_label_attempted=("target_label_attempted", "first"),
        )
        .reset_index()
        .rename(columns={"slot_label": "target_label"})
    )

    targets_df = status_df.merge(
        cap_lookup,
        on=["problem_id", "generator_llm", "language", "target_label"],
        how="left",
    )
    targets_df["opportunistic_capture"] = (
        targets_df["opportunistic_capture"]
        .map({True: True, False: False, "True": True, "False": False})
        .fillna(False)
        .astype(bool)
    )
    targets_df.loc[~targets_df["succeed"], "opportunistic_capture"] = False
    targets_df["capture_mode"] = targets_df.apply(
        lambda row: (
            "failed"
            if not row["succeed"]
            else ("opportunistic" if row["opportunistic_capture"] else "direct")
        ),
        axis=1,
    )
    return targets_df


def build_attempts_df(payloads: dict[str, dict], targets_df: pd.DataFrame) -> pd.DataFrame:
    """One row per feedback attempt, with target-level context on every row."""
    rows: list[dict] = []
    for target in targets_df.itertuples(index=False):
        record = _record_for_target(
            payloads.get(target.problem_id, {}),
            target.generator_llm,
            target.language,
            target.target_label,
        )
        history = _session_history(record, target.target_label)
        target_attempts_total = int(target.attempts)

        target_context = {
            "problem_id": target.problem_id,
            "generator_llm": target.generator_llm,
            "language": target.language,
            "target_label": target.target_label,
            "target_succeeded": target.succeed,
            "capture_mode": target.capture_mode,
            "opportunistic_capture": target.opportunistic_capture,
            "target_label_attempted": target.target_label_attempted,
            "target_attempts_total": target_attempts_total,
            "reference_space_size": target.reference_space_size,
            "candidate_space_size": target.candidate_space_size,
            "fp": target.fp,
            "fn": target.fn,
            "updated_at": target.updated_at,
        }

        for attempt_entry in history:
            attempt_num = attempt_entry.get("attempt")
            rows.append(
                {
                    **target_context,
                    "attempt": attempt_num,
                    "is_final_attempt": attempt_num == target_attempts_total,
                    "detected_label": attempt_entry.get("observed_label"),
                    "label_match": attempt_entry.get("label_match"),
                    "attempt_outcome": "success" if attempt_entry.get("label_match") else "failed",
                    "exec_status": attempt_entry.get("exec_status"),
                    "error_category": attempt_entry.get("error_category"),
                    "error_summary": attempt_entry.get("error_summary", ""),
                }
            )

        if not history and not target.succeed:
            for attempt_num in range(1, target_attempts_total + 1):
                rows.append(
                    {
                        **target_context,
                        "attempt": attempt_num,
                        "is_final_attempt": attempt_num == target_attempts_total,
                        "detected_label": "",
                        "label_match": False,
                        "attempt_outcome": "failed",
                        "exec_status": "",
                        "error_category": "",
                        "error_summary": "",
                    }
                )

    return pd.DataFrame(rows)


TARGET_KEYS = ["problem_id", "generator_llm", "language", "target_label"]
MAX_GENERATION_ROUNDS = 5


def targets_from_attempts(attempts_df: pd.DataFrame) -> pd.DataFrame:
    return (
        attempts_df.sort_values(TARGET_KEYS + ["attempt"])
        .drop_duplicates(TARGET_KEYS, keep="last")
        .copy()
    )


def summarize_construction_group(
    attempt_rows: pd.DataFrame, target_rows: pd.DataFrame
) -> dict[str, float | int]:
    n_targets = len(target_rows)
    succeeded = int(target_rows["target_succeeded"].sum())
    failed = n_targets - succeeded
    return {
        "total_attempts": len(attempt_rows),
        "succeeded_targets": succeeded,
        "failed_targets": failed,
        "success_rate": round(succeeded / n_targets, 4) if n_targets else 0.0,
        "opportunistic": int(
            target_rows.loc[target_rows["target_succeeded"], "opportunistic_capture"].sum()
        ),
        "succeed_first_attempt": int(
            (target_rows["target_succeeded"] & (target_rows["target_attempts_total"] == 1)).sum()
        ),
        "failed_after_5_attempts": int(
            (
                ~target_rows["target_succeeded"]
                & (target_rows["target_attempts_total"] == MAX_GENERATION_ROUNDS)
            ).sum()
        ),
        "avg_num_attempts": round(target_rows["target_attempts_total"].mean(), 2),
        "median_num_attempts": target_rows["target_attempts_total"].median(),
    }


def build_construction_summary_table(attempts_df: pd.DataFrame) -> pd.DataFrame:
    """Summary rows: total, by language, by language+target_label."""
    targets_df = targets_from_attempts(attempts_df)
    rows: list[dict[str, float | int | str]] = [
        {"group": "total", **summarize_construction_group(attempts_df, targets_df)}
    ]

    for language, lang_attempts in attempts_df.groupby("language", sort=True):
        lang_targets = targets_df[targets_df["language"] == language]
        rows.append(
            {"group": language, **summarize_construction_group(lang_attempts, lang_targets)}
        )

    for (language, target_label), ll_attempts in attempts_df.groupby(
        ["language", "target_label"], sort=True
    ):
        ll_targets = targets_df[
            (targets_df["language"] == language) & (targets_df["target_label"] == target_label)
        ]
        rows.append(
            {
                "group": f"{language} / {target_label}",
                **summarize_construction_group(ll_attempts, ll_targets),
            }
        )

    return pd.DataFrame(rows)


def _print_summary(targets_df: pd.DataFrame, attempts_df: pd.DataFrame) -> None:
    succeeded = int(targets_df["succeed"].sum())
    opportunistic = int(targets_df.loc[targets_df["succeed"], "opportunistic_capture"].sum())

    print(f"Target slots: {len(targets_df)} ({succeeded} succeeded, {len(targets_df) - succeeded} failed)")
    print(f"Capture modes: direct={succeeded - opportunistic}, opportunistic={opportunistic}, failed={(~targets_df['succeed']).sum()}")
    print(f"Attempts: {len(attempts_df)} total, {int(attempts_df['label_match'].sum())} label matches")
    print()
    print("By language (target slots):")
    print(
        targets_df.groupby("language")["succeed"]
        .agg(total="count", succeeded="sum")
        .assign(failed=lambda df: df["total"] - df["succeeded"])
    )
    print()
    print("By target label (target slots):")
    print(
        targets_df.groupby("target_label")["succeed"]
        .agg(total="count", succeeded="sum")
        .assign(failed=lambda df: df["total"] - df["succeeded"])
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    payloads = _load_payloads()
    captures_df = load_capture_rows(payloads)
    targets_df = build_targets_df(captures_df)
    attempts_df = build_attempts_df(payloads, targets_df)

    attempts_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {OUTPUT_PATH} ({len(attempts_df)} rows)")
    print()
    _print_summary(targets_df, attempts_df)


if __name__ == "__main__":
    main()
