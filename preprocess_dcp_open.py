import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.executors import (
    classify_problem_type_strict,
    enumerate_solution_space_for_entry,
    instance_to_dict,
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_metadata_field(metadata: List[str], key: str) -> str:
    prefix = f"# {key}:"
    for line in metadata or []:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def main():
    input_path = Path("extra_files/dcp-bench-open.jsonl")
    out_dir = Path("extra_files/preprocessed")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dcp_bench_open_summary.csv"

    solution_limit = 1_000_000
    time_limit_sec = 60 * 5

    rows: List[Dict[str, Any]] = []

    current_info = pd.read_csv("extra_files/preprocessed/dcp_bench_open_summary.csv")

    for entry in load_jsonl(input_path):
        pid = entry.get("id", "")


        if pid in current_info["id"].values:
            stop_reason = current_info.loc[current_info["id"] == pid, "stop_reason"].values[0]
            if stop_reason in ["complete","time_limit"]: # "time_limit"
                print(f"problem {pid} already processed!")
                rows.append(current_info.loc[current_info["id"] == pid].to_dict(orient="records")[0])
                continue

        model_code = entry.get("model", "") or ""
        decision_vars = entry.get("decision_variables", []) or []
        metadata = entry.get("metadata", []) or []
        instances = entry.get("instances", []) or []
        example_instance = entry.get("example_instance", "")

        problem_type, is_optimization, type_reason = classify_problem_type_strict(model_code)
        instance_dict = instance_to_dict(example_instance, instances)

        exec_result = enumerate_solution_space_for_entry(
            model_code=model_code,
            instance_dict=instance_dict,
            decision_vars=decision_vars,
            problem_type=problem_type,
            solution_limit=solution_limit,
            time_limit_sec=time_limit_sec,
        )

        if exec_result.get("ok", False):
            solution_space_size = exec_result.get("num_solutions", 0)
            is_complete = exec_result.get("is_solution_space_complete", False)
            exec_status = "ok"
            exec_error = exec_result.get("error", "")
            stop_reason = exec_result.get("stop_reason", "")
            exit_status = exec_result.get("exit_status", "")
            runtime_sec = exec_result.get("runtime_sec", "")
        else:
            solution_space_size = ""
            is_complete = False
            err = exec_result.get("error", "") or ""
            if "SyntaxError" in err or "No `model`" in err or "missing" in err.lower():
                exec_status = "non_executable"
            else:
                exec_status = "enumeration_failed"
            exec_error = err
            stop_reason = ""
            exit_status = ""
            runtime_sec = ""

        print(pid)
        print(problem_type)
        print(solution_space_size)
        print(is_complete)
        print(stop_reason)
        print(exec_error)
        print("-------------------------------")

        row = {
            "id": pid,
            "problem_type": problem_type,
            "is_optimization": is_optimization,
            "type_reason": type_reason,
            "solution_space_size": solution_space_size,
            "is_solution_space_complete": is_complete,
            "stop_reason": stop_reason,
            "exit_status": exit_status,
            "runtime_sec": runtime_sec,
            "space_counting_mode": "feasible",
            "exec_status": exec_status,
            "exec_error": exec_error,
            "num_decision_variables": len(decision_vars),
            "num_instances": len(instances),
            "has_example_instance": bool(example_instance),
            "category": parse_metadata_field(metadata, "Category"),
            "source": parse_metadata_field(metadata, "Source"),
        }
        rows.append(row)

    fieldnames = [
        "id",
        "problem_type",
        "is_optimization",
        "type_reason",
        "solution_space_size",
        "is_solution_space_complete",
        "stop_reason",
        "exit_status",
        "runtime_sec",
        "space_counting_mode",
        "exec_status",
        "exec_error",
        "num_decision_variables",
        "num_instances",
        "has_example_instance",
        "category",
        "source",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} problems.")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
