import contextlib
import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from cpmpy import Model
from cpmpy.solvers.solver_interface import ExitStatus

from src.executors import (
    _cpmpy_solution_key,
    _effective_solution_limit,
    _exec_failure,
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


def classify_problem_type_strict(model_code: str) -> Tuple[str, bool, str]:
    if re.search(r"\bmodel\.(minimize|maximize)\s*\(", model_code):
        return "COP", True, "explicit model.minimize/maximize in code"
    if re.search(r"\b(minimize|maximize)\s*\(", model_code):
        return "COP", True, "explicit minimize/maximize in code"
    return "CSP", False, "no explicit objective call in code"


def analyze_solve_all_stop(
    enum_model: Model,
    n_found: int,
    max_solutions: int,
    time_limit_sec: int,
) -> Dict[str, Any]:
    """Infer why solveAll stopped (complete, solution_limit, time_limit, unsat, partial)."""
    st = enum_model.status()
    runtime_sec = st.runtime
    exit_status = st.exitstatus.name if st.exitstatus else "NOT_RUN"

    hit_solution_limit = n_found >= max_solutions
    hit_time_limit = st.exitstatus == ExitStatus.UNKNOWN or (
        runtime_sec is not None and runtime_sec >= 0.99 * time_limit_sec
    )
    enumeration_complete = (
        n_found > 0
        and st.exitstatus == ExitStatus.OPTIMAL
        and not hit_solution_limit
    )

    if hit_solution_limit:
        stop_reason = "solution_limit"
    elif hit_time_limit:
        stop_reason = "time_limit"
    elif st.exitstatus == ExitStatus.UNSATISFIABLE:
        stop_reason = "unsat"
    elif enumeration_complete:
        stop_reason = "complete"
    elif n_found > 0:
        stop_reason = "partial"
    else:
        stop_reason = "none"

    error = ""
    if hit_solution_limit:
        error = f"Reached solution_limit={max_solutions}"
    elif hit_time_limit:
        error = f"Reached time_limit={time_limit_sec}s (runtime={runtime_sec})"

    is_complete = enumeration_complete and not hit_time_limit

    return {
        "error": error,
        "stop_reason": stop_reason,
        "exit_status": exit_status,
        "runtime_sec": runtime_sec,
        "is_solution_space_complete": is_complete,
        "hit_solution_limit": hit_solution_limit,
        "hit_time_limit": hit_time_limit,
    }


def execute_cpmpy_enumerate(
    code_text: str,
    instance_dict: dict[str, Any],
    decision_var_names: list[str],
    solution_limit: int,
    time_limit_sec: int,
    *,
    strip_objective: bool = False,
) -> dict[str, Any]:
    """
    Run solveAll and count solutions.

    For COP models, strip_objective=True builds Model(constraints) so OR-Tools
    enumerates feasible solutions instead of all optimal ones (unsupported).
    """
    namespace = dict(instance_dict)
    max_solutions = _effective_solution_limit(solution_limit)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code_text, namespace, namespace)
    except SyntaxError as e:
        detail = f"SyntaxError at line {e.lineno}, offset {e.offset}: {e.msg}"
        if e.text:
            detail += f" | source: {e.text.strip()}"
        return _exec_failure(detail)
    except Exception as e:
        return _exec_failure(f"{type(e).__name__}: {e}")

    model = namespace.get("model")
    if model is None:
        return _exec_failure("No `model` object found")
    if not isinstance(model, Model):
        return _exec_failure("`model` is not a CPMpy Model")

    missing = [v for v in decision_var_names if v not in namespace]
    if missing:
        return _exec_failure(f"Decision variables missing: {missing}")

    enum_model = Model(model.constraints) if strip_objective else model

    solution_space: set = set()
    solution_count = [0]

    def _on_solution() -> None:
        solution_count[0] += 1
        solution_space.add(_cpmpy_solution_key(decision_var_names, namespace))

    try:
        n_found = enum_model.solveAll(
            display=_on_solution,
            solution_limit=max_solutions,
            time_limit=time_limit_sec,
        )
    except Exception as e:
        return _exec_failure(f"Solver crashed: {type(e).__name__}: {e}")

    stop_meta = analyze_solve_all_stop(enum_model, n_found, max_solutions, time_limit_sec)

    return {
        "ok": True,
        "space": solution_space,
        "num_solutions": n_found,
        **stop_meta,
    }


def enumerate_solution_space_for_entry(
    model_code: str,
    instance_dict: dict[str, Any],
    decision_vars: list[str],
    problem_type: str,
    solution_limit: int,
    time_limit_sec: int,
) -> dict[str, Any]:
    return execute_cpmpy_enumerate(
        code_text=model_code,
        instance_dict=instance_dict,
        decision_var_names=decision_vars,
        solution_limit=solution_limit,
        time_limit_sec=time_limit_sec,
        strip_objective=(problem_type == "COP"),
    )


def main():
    input_path = Path("extra_files/dcp-bench-open.jsonl")
    out_dir = Path("extra_files/preprocessed")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dcp_bench_open_summary.csv"

    solution_limit = 100_000
    time_limit_sec = 60

    rows: List[Dict[str, Any]] = []

    for entry in load_jsonl(input_path):
        pid = entry.get("id", "")
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
