"""Run reference and candidate CP models; enumerate their solution spaces."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from cpmpy import Model

from .config import ExperimentContext


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def instance_to_dict(example_instance_text: str, data_instances_list: list[Any]) -> dict[str, Any]:
    """Pick the first instance from `data_instances_list`, or eval the example snippet."""
    if isinstance(data_instances_list, list) and data_instances_list:
        first = data_instances_list[0]
        if isinstance(first, dict):
            return first

    snippet = str(example_instance_text or "").strip()
    if not snippet:
        return {}

    namespace: dict[str, Any] = {}
    exec(snippet, {}, namespace)  # benchmark-only: trusted source snippet
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


# ---------------------------------------------------------------------------
# Label classification
# ---------------------------------------------------------------------------

def observed_label_from_sets(reference_space: set, candidate_space: set) -> tuple[str, int, int]:
    false_positives = len(candidate_space - reference_space)
    false_negatives = len(reference_space - candidate_space)

    if false_positives == 0 and false_negatives == 0:
        observed = "equivalent"
    elif false_positives > 0 and false_negatives == 0:
        observed = "unsound"
    elif false_positives == 0 and false_negatives > 0:
        observed = "incomplete"
    else:
        observed = "unsound-incomplete"
    return observed, false_positives, false_negatives


def label_match_rule(
    claimed: str, observed: str, reference_space: set, candidate_space: set
) -> bool:
    if claimed == "status-only correct":
        ref_sat = len(reference_space) > 0
        cand_sat = len(candidate_space) > 0
        return ref_sat == cand_sat and observed != "equivalent"
    if claimed == "non-executable":
        return False
    return claimed == observed


def classify_error_message(message: str) -> tuple[str, str]:
    text = str(message or "")
    lowered = text.lower()

    if any(token in lowered for token in (
        "syntax error", "indentation error", "syntax check failed", "parse error",
    )):
        category = "syntax_error"
    elif "timeout" in lowered or "time limit" in lowered or "timed out" in lowered:
        category = "timeout"
    elif "not found" in lowered:
        category = "environment"
    else:
        category = "runtime"

    compact = " | ".join([ln.strip() for ln in text.splitlines() if ln.strip()][:3])
    return category, compact[:700]


# ---------------------------------------------------------------------------
# Value flattening helpers
# ---------------------------------------------------------------------------

def _flatten_cpmpy_vars(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return [item for inner in value for item in _flatten_cpmpy_vars(inner)]
    if isinstance(value, np.ndarray):
        return list(value.flat)
    if hasattr(value, "flatten") and not hasattr(value, "value"):
        try:
            return list(value.flatten())
        except Exception:
            pass
    return [value]


def _flatten_value_to_ints(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(item for inner in value for item in _flatten_value_to_ints(inner))
    if isinstance(value, bool):
        return (1 if value else 0,)
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    try:
        return (int(value),)
    except (TypeError, ValueError):
        return (hash(str(value)),)


_UNSET_VAR_SENTINEL = -(1 << 31)


def _safe_var_value(var: Any) -> int:
    """Read a CPMpy variable's value, falling back to a sentinel when unset.

    Some LLM-generated candidates declare a decision variable but never
    constrain it; OR-Tools then leaves it unbound and `v.value()` returns
    `None`, which used to crash the enumeration callback. We substitute a
    sentinel so the run can continue.
    """
    try:
        raw = var.value()
    except Exception:
        return _UNSET_VAR_SENTINEL
    if raw is None:
        return _UNSET_VAR_SENTINEL
    try:
        return int(raw)
    except (TypeError, ValueError):
        return _UNSET_VAR_SENTINEL


def _cpmpy_solution_key(var_names: list[str], namespace: dict[str, Any]) -> tuple:
    parts: list[tuple[str, tuple[int, ...]]] = []
    for name in var_names:
        flat = _flatten_cpmpy_vars(namespace[name])
        parts.append((name, tuple(_safe_var_value(v) for v in flat)))
    return tuple(parts)


def _exec_failure(error: str) -> dict[str, Any]:
    return {"ok": False, "error": error, "space": None, "num_solutions": 0}


def _effective_solution_limit(solution_limit: int) -> int:
    """Normalize solution-limit inputs to a positive bound."""
    try:
        limit = int(solution_limit)
    except (TypeError, ValueError):
        return 1
    return max(1, limit)


def _assigned_python_names(code_text: str) -> set[str]:
    """Return variable names assigned in Python code."""
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return set()

    assigned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
    return assigned


def _build_cpmpy_namespace(code_text: str, instance_dict: dict[str, Any]) -> dict[str, Any]:
    """Inject all instance fields into the execution namespace.

    We inject unconditionally so that patterns like
        capabilities = cpm_array(capabilities)
    work correctly: the instance value is available on the RHS, then the model
    code overwrites it with the wrapped form.  Filtering by `ast.Store` targets
    was too aggressive and broke any assignment that reads the same name it writes.
    """
    return dict(instance_dict)


def _model_defines_minizinc_symbol(code_text: str, symbol: str) -> bool:
    """Best-effort check for in-model MiniZinc symbol definitions.

    We treat declarations with assignments like `int: m = 4;` as defined and skip
    passing that symbol via data file to avoid duplicate assignments.
    """
    symbol_esc = re.escape(symbol)
    pattern = re.compile(rf":[^;\n]*\b{symbol_esc}\b\s*=")
    return bool(pattern.search(code_text))


def _to_minizinc_dzn_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return "[" + ", ".join(_to_minizinc_dzn_value(v) for v in value) + "]"
    if isinstance(value, dict):
        raise ValueError("MiniZinc data generation does not support dict values directly")
    return str(value)


def _render_minizinc_dzn(instance_dict: dict[str, Any], code_text: str) -> str:
    lines: list[str] = []
    for key, value in instance_dict.items():
        if _model_defines_minizinc_symbol(code_text, key):
            continue
        lines.append(f"{key} = {_to_minizinc_dzn_value(value)};")
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# CPMpy execution
# ---------------------------------------------------------------------------

def execute_cpmpy_and_enumerate(
    code_text: str,
    instance_dict: dict[str, Any],
    decision_var_names: list[str],
    solution_limit: int,
    time_limit_sec: int,
) -> dict[str, Any]:
    namespace = _build_cpmpy_namespace(code_text, instance_dict)
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

    solution_space: set = set()
    solution_count = [0]

    def _on_solution() -> None:
        solution_count[0] += 1
        solution_space.add(_cpmpy_solution_key(decision_var_names, namespace))

    try:
        model.solveAll(
            display=_on_solution,
            solution_limit=max_solutions,
            time_limit=time_limit_sec,
        )
    except Exception as e:
        return _exec_failure(f"Solver crashed: {type(e).__name__}: {e}")

    error = ""
    if solution_count[0] >= max_solutions:
        error = f"Reached solution_limit={max_solutions}; full space may be truncated"
    return {
        "ok": True,
        "error": error,
        "space": solution_space,
        "num_solutions": solution_count[0],
    }


# ---------------------------------------------------------------------------
# MiniZinc execution
# ---------------------------------------------------------------------------

def _check_minizinc_syntax(
    model_path: Path, data_path: Path | None = None, timeout_sec: int = 60
) -> tuple[bool, str]:
    cmd = ["minizinc", "--compile", "--solver", "gecode", str(model_path)]
    if data_path is not None:
        cmd.append(str(data_path))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        print(f"MiniZinc result: {result.returncode}")
        print(f"MiniZinc stdout: {result.stdout}")
        print(f"MiniZinc stderr: {result.stderr}")
    except FileNotFoundError:
        print("MiniZinc executable not found")
        return True, "MiniZinc executable not found; syntax not checked."
    except subprocess.TimeoutExpired:
        print(f"MiniZinc process timed out after {timeout_sec}s")
        return False, "MiniZinc compilation timed out."

    if result.returncode == 0:
        return True, ""
    combined = (result.stderr or "") + (result.stdout or "")
    return False, combined.strip()


def _iter_json_objects(text: str) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        try:
            obj, j = decoder.raw_decode(text, i)
            yield obj
            i = j
        except json.JSONDecodeError:
            i += 1


def _minizinc_solutions_from_stream(stdout_text: str, decision_var_names: list[str]) -> set:
    space: set = set()
    for obj in _iter_json_objects(stdout_text):
        candidate = None
        if isinstance(obj, dict):
            output_json = obj.get("output", {}).get("json") if isinstance(obj.get("output"), dict) else None
            if isinstance(output_json, dict):
                candidate = output_json
            elif isinstance(obj.get("solution"), dict):
                candidate = obj["solution"]
            elif all(k in obj for k in decision_var_names):
                candidate = obj

        if isinstance(candidate, dict) and all(k in candidate for k in decision_var_names):
            key = tuple((name, _flatten_value_to_ints(candidate[name])) for name in decision_var_names)
            space.add(key)
    return space


def execute_minizinc_and_enumerate(
    code_text: str,
    instance_dict: dict[str, Any],
    decision_var_names: list[str],
    solution_limit: int,
    time_limit_sec: int,
) -> dict[str, Any]:
    max_solutions = _effective_solution_limit(solution_limit)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "candidate.mzn"
        data_path = Path(tmpdir) / "candidate.dzn"

        print(f"Writing model to {model_path}")
        print(f"Writing data to {data_path}")
        print(f"Code text: \n--------------------------------\n\n{code_text}\n\n--------------------------------\n\n")

        model_path.write_text(code_text, encoding="utf-8")
        data_path.write_text(_render_minizinc_dzn(instance_dict, code_text), encoding="utf-8")

        ok_syntax, syntax_error = _check_minizinc_syntax(model_path, data_path=data_path)
        if not ok_syntax:
            return _exec_failure(f"MiniZinc syntax check failed: {syntax_error}")

        try:
            result = subprocess.run(
                [
                    "minizinc",
                    "--solver", "gecode",
                    "--all-solutions",
                    "--num-solutions", str(max_solutions),
                    "--time-limit", str(time_limit_sec * 1000),
                    "--output-mode", "json",
                    str(model_path),
                    str(data_path),
                ],
                capture_output=True,
                text=True,
                timeout=time_limit_sec + 30,
            )
            print(f"MiniZinc result: {result.returncode}")
            # print(f"MiniZinc stdout: {result.stdout}")
            print(f"MiniZinc stderr: {result.stderr}")
        except FileNotFoundError:
            print("MiniZinc executable not found")
            return _exec_failure("MiniZinc executable not found")
        except subprocess.TimeoutExpired:
            print(f"MiniZinc process timed out after {time_limit_sec + 30}s")
            return _exec_failure(f"MiniZinc process timed out after {time_limit_sec + 30}s")

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout + "\n" + stderr

        if result.returncode != 0 and not stdout.strip():
            return _exec_failure((stderr.strip() or "MiniZinc run failed"))

        space = _minizinc_solutions_from_stream(stdout, decision_var_names)
        if not space:
            if "=====UNSATISFIABLE=====" in combined:
                return {"ok": True, "error": "", "space": set(), "num_solutions": 0}
            if "=====UNKNOWN=====" in combined:
                return _exec_failure(
                    "MiniZinc solver timed out (=====UNKNOWN=====); "
                    "try increasing time_limit_minizinc_sec"
                )
            if "=====ERROR=====" in combined:
                return _exec_failure(
                    f"MiniZinc solver error: {(stderr.strip() or stdout.strip())}"
                )
            return _exec_failure("Could not parse MiniZinc solutions")

        error = ""
        if len(space) >= max_solutions:
            error = f"Reached solution_limit={max_solutions}; full space may be truncated"
        return {
            "ok": True,
            "error": error,
            "space": space,
            "num_solutions": len(space),
        }


# ---------------------------------------------------------------------------
# High-level dispatchers
# ---------------------------------------------------------------------------

_SUPPORTED_LANGUAGES = {"cpmpy", "minizinc"}


def enumerate_solution_space(
    code_text: str,
    language: str,
    instance_dict: dict[str, Any],
    decision_var_names: list[str],
    solution_limit: int,
    time_limit_sec: int,
) -> tuple[dict[str, Any], set]:
    language_key = language.strip().lower()
    if language_key == "cpmpy":
        result = execute_cpmpy_and_enumerate(
            code_text=code_text,
            instance_dict=instance_dict,
            decision_var_names=decision_var_names,
            solution_limit=solution_limit,
            time_limit_sec=time_limit_sec,
        )
    elif language_key == "minizinc":
        result = execute_minizinc_and_enumerate(
            code_text=code_text,
            instance_dict=instance_dict,
            decision_var_names=decision_var_names,
            solution_limit=solution_limit,
            time_limit_sec=time_limit_sec,
        )
    else:
        raise ValueError(f"Unsupported language: {language}")

    space = result["space"] if result["ok"] else set()
    return result, space


def evaluate_candidate_code(
    context: ExperimentContext,
    language: str,
    code: str,
    claimed_label: str,
    instance_dict: dict[str, Any],
    reference_space: set,
) -> dict[str, Any]:
    """Execute a candidate, classify its solution space, and decide label-match."""
    if language.strip().lower() not in _SUPPORTED_LANGUAGES:
        return {
            "exec_status": "not_evaluated",
            "candidate_space_size": None,
            "fp": None,
            "fn": None,
            "observed_label": "unknown",
            "label_match": None,
            "note": "Executor not implemented for this language",
            "error_category": "unsupported",
            "error_summary": "Executor not implemented for this language",
        }

    exec_result, candidate_space = enumerate_solution_space(
        code_text=code,
        language=language,
        instance_dict=instance_dict,
        decision_var_names=context.decision_variables,
        solution_limit=context.solution_limit,
        time_limit_sec=context.time_limit_for(language),
    )

    if not exec_result["ok"]:
        error_category, error_summary = classify_error_message(exec_result.get("error", ""))
        return {
            "exec_status": "non_executable",
            "candidate_space_size": None,
            "fp": None,
            "fn": None,
            "observed_label": "non-executable",
            "label_match": claimed_label == "non-executable",
            "candidate_truncated": False,
            "note": exec_result.get("error", ""),
            "error_category": error_category,
            "error_summary": error_summary,
        }

    # Detect whether the candidate's enumeration was cut short by the solution limit.
    # When truncated, any false-negative count is unreliable: we simply may not have
    # enumerated far enough to find all reference solutions, so we must NOT conclude
    # incompleteness from such a run.
    candidate_truncated = "Reached solution_limit" in exec_result.get("error", "")

    observed, false_positives, false_negatives = observed_label_from_sets(
        reference_space, candidate_space
    )

    truncation_note = ""
    if candidate_truncated and false_negatives > 0:
        truncation_note = (
            f"Candidate enumeration stopped at solution_limit={context.solution_limit}; "
            f"{false_negatives} apparent missing solution(s) may be an artefact of truncation "
            f"and do not reliably indicate incompleteness. "
        )
        if false_positives > 0:
            # The unsound part (FPs) is confirmed; the incomplete part is not.
            observed = "unsound"
        else:
            # All enumerated solutions are valid but we cannot confirm coverage.
            # Use a sentinel that does not match any real correctness label so the
            # candidate is not opportunistically captured as "incomplete".
            observed = "truncated"
        false_negatives = 0

    return {
        "exec_status": "ok",
        "candidate_space_size": len(candidate_space),
        "fp": false_positives,
        "fn": false_negatives,
        "observed_label": observed,
        "label_match": label_match_rule(claimed_label, observed, reference_space, candidate_space),
        "candidate_truncated": candidate_truncated,
        "note": truncation_note,
        "error_category": "",
        "error_summary": "",
    }


# ---------------------------------------------------------------------------
# SAT-only execution (single solve, not solveAll)
# ---------------------------------------------------------------------------

def execute_cpmpy_sat_only(
    code_text: str,
    instance_dict: dict[str, Any],
    decision_var_names: list[str],
    time_limit_sec: int,
) -> dict[str, Any]:
    """Run a CPMpy SAT check: single solve() call, then read the first solution."""
    namespace = _build_cpmpy_namespace(code_text, instance_dict)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code_text, namespace, namespace)
    except SyntaxError as e:
        detail = f"SyntaxError at line {e.lineno}, offset {e.offset}: {e.msg}"
        if e.text:
            detail += f" | source: {e.text.strip()}"
        return {"ok": False, "sat": None, "solution": None, "error": detail}
    except Exception as e:
        return {"ok": False, "sat": None, "solution": None, "error": f"{type(e).__name__}: {e}"}

    model = namespace.get("model")
    if model is None:
        return {"ok": False, "sat": None, "solution": None, "error": "No `model` object found"}
    if not isinstance(model, Model):
        return {"ok": False, "sat": None, "solution": None, "error": "`model` is not a CPMpy Model"}

    missing = [v for v in decision_var_names if v not in namespace]
    if missing:
        return {"ok": False, "sat": None, "solution": None, "error": f"Decision variables missing: {missing}"}

    try:
        sat = model.solve(time_limit=time_limit_sec)
    except Exception as e:
        return {"ok": False, "sat": None, "solution": None, "error": f"Solver crashed: {type(e).__name__}: {e}"}

    if not sat:
        return {"ok": True, "sat": False, "solution": None, "error": ""}

    first_solution = {
        name: tuple(_safe_var_value(v) for v in _flatten_cpmpy_vars(namespace[name]))
        for name in decision_var_names
    }
    return {"ok": True, "sat": True, "solution": first_solution, "error": ""}


def execute_minizinc_sat_only(
    code_text: str,
    instance_dict: dict[str, Any],
    decision_var_names: list[str],
    time_limit_sec: int,
) -> dict[str, Any]:
    """Run a MiniZinc SAT check: ask for at most 1 solution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "candidate.mzn"
        data_path = Path(tmpdir) / "candidate.dzn"
        model_path.write_text(code_text, encoding="utf-8")
        data_path.write_text(_render_minizinc_dzn(instance_dict, code_text), encoding="utf-8")

        ok_syntax, syntax_error = _check_minizinc_syntax(model_path, data_path=data_path)
        if not ok_syntax:
            return {
                "ok": False, "sat": None, "solution": None,
                "error": f"MiniZinc syntax check failed: {syntax_error}",
            }

        try:
            result = subprocess.run(
                [
                    "minizinc",
                    "--solver", "gecode",
                    "--num-solutions", "1",
                    "--time-limit", str(time_limit_sec * 1000),
                    "--output-mode", "json",
                    str(model_path),
                    str(data_path),
                ],
                capture_output=True,
                text=True,
                timeout=time_limit_sec + 30,
            )
        except FileNotFoundError:
            return {"ok": False, "sat": None, "solution": None, "error": "MiniZinc executable not found"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "sat": None, "solution": None, "error": f"MiniZinc process timed out after {time_limit_sec + 30}s"}

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout + "\n" + stderr

        if "=====UNSATISFIABLE=====" in combined:
            return {"ok": True, "sat": False, "solution": None, "error": ""}

        if "=====UNKNOWN=====" in combined:
            return {
                "ok": False, "sat": None, "solution": None,
                "error": "MiniZinc solver timed out (=====UNKNOWN=====); try increasing time_limit_minizinc_sec",
            }

        if result.returncode != 0 and not stdout.strip():
            return {"ok": False, "sat": None, "solution": None, "error": stderr.strip() or "MiniZinc run failed"}

        for obj in _iter_json_objects(stdout):
            if not isinstance(obj, dict):
                continue
            output_json = (
                obj.get("output", {}).get("json")
                if isinstance(obj.get("output"), dict)
                else None
            )
            candidate_dict = (
                output_json
                or obj.get("solution")
                or (obj if all(k in obj for k in decision_var_names) else None)
            )
            if isinstance(candidate_dict, dict) and all(k in candidate_dict for k in decision_var_names):
                solution = {
                    name: _flatten_value_to_ints(candidate_dict[name])
                    for name in decision_var_names
                }
                return {"ok": True, "sat": True, "solution": solution, "error": ""}

        return {"ok": False, "sat": None, "solution": None, "error": "Could not parse MiniZinc output"}


def evaluate_candidate_sat_only(
    context: ExperimentContext,
    language: str,
    code: str,
    instance_dict: dict[str, Any],
    reference_space: set,
) -> dict[str, Any]:
    """Evaluate a candidate using SAT-only: single solve, check SAT/UNSAT parity and solution validity.

    Returns a dict with keys:
        exec_status        – "ok" | "non_executable" | "not_evaluated"
        reference_sat      – bool: reference space is non-empty
        candidate_sat      – bool | None
        solution_in_ref    – bool | None  (None when UNSAT or non-executable)
        sat_only_correct   – bool | None
        error / error_category / error_summary
    """
    reference_sat = len(reference_space) > 0
    language_key = language.strip().lower()

    if language_key not in _SUPPORTED_LANGUAGES:
        return {
            "exec_status": "not_evaluated",
            "reference_sat": reference_sat,
            "candidate_sat": None,
            "solution_in_ref": None,
            "sat_only_correct": None,
            "error": "Executor not implemented for this language",
            "error_category": "unsupported",
            "error_summary": "Executor not implemented for this language",
        }

    if language_key == "cpmpy":
        result = execute_cpmpy_sat_only(
            code_text=code,
            instance_dict=instance_dict,
            decision_var_names=context.decision_variables,
            time_limit_sec=context.time_limit_for(language),
        )
    else:
        result = execute_minizinc_sat_only(
            code_text=code,
            instance_dict=instance_dict,
            decision_var_names=context.decision_variables,
            time_limit_sec=context.time_limit_for(language),
        )

    if not result["ok"]:
        error_cat, error_sum = classify_error_message(result.get("error", ""))
        return {
            "exec_status": "non_executable",
            "reference_sat": reference_sat,
            "candidate_sat": None,
            "solution_in_ref": None,
            "sat_only_correct": False,
            "error": result.get("error", ""),
            "error_category": error_cat,
            "error_summary": error_sum,
        }

    candidate_sat: bool = result["sat"]

    if not candidate_sat:
        return {
            "exec_status": "ok",
            "reference_sat": reference_sat,
            "candidate_sat": False,
            "solution_in_ref": None,
            "sat_only_correct": not reference_sat,
            "error": "",
        }

    solution = result["solution"]
    sol_key = tuple((name, solution[name]) for name in context.decision_variables)
    solution_in_ref = sol_key in reference_space
    return {
        "exec_status": "ok",
        "reference_sat": reference_sat,
        "candidate_sat": True,
        "solution_in_ref": solution_in_ref,
        "sat_only_correct": reference_sat and solution_in_ref,
        "error": "",
    }
