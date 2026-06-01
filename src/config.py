"""Static configuration and the per-experiment `ExperimentContext`."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CORRECTNESS_LABELS: list[str] = [
    "equivalent",
    "unsound",
    "incomplete",
    "unsound-incomplete",
    "non-executable",
]

LANGUAGE_LABELS: list[str] = [
    "minizinc", 
    "CPMpy",
    # "pyCSP3",
]

GENERATOR_MODELS: list[str] = [
    "openai/gpt-5.4-mini",
    # "openai/gpt-5.1-codex-mini",
    # "openai/gpt-4o-mini",
]

JUDGE_MODELS: list[str] = [
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-2.5-pro",
    "meta-llama/llama-4-scout",
    "qwen/qwen3-32b",
    "qwen/qwen3-coder-next",
]

# JUDGE_MODELS = [
#     # Commercial / closed models
#     "openai/gpt-4o-mini",              # small, commercial, cheap general baseline, non-reasoning
#     "openai/gpt-4.1-mini",             # mid-size commercial, stronger coding/instruction-following baseline
#     "anthropic/claude-sonnet-4.6",      # commercial, strong general + coding + agentic reasoning
#     "google/gemini-2.5-pro",            # commercial, large reasoning model, strong coding/math/science
#     "google/gemini-2.5-flash",          # commercial, cheaper/faster reasoning-capable Gemini baseline

#     # Open-weight / model-weights-available general or reasoning models
#     "meta-llama/llama-4-scout",         # open-weight, MoE, general, multilingual/multimodal, non-code-focused
#     "qwen/qwen3-32b",                   # open-weight, dense 32B, general reasoning + coding
#     "deepseek/deepseek-chat-v3.1",      # open-weight, large MoE, hybrid reasoning/non-reasoning, strong coding
#     "microsoft/phi-4",                  # open-weight, efficient smaller reasoning/general model
#     "mistralai/mistral-small-3.2-24b-instruct",  # open-weight, 24B, general + structured output + coding/STEM

#     # Code-specialized models
#     "qwen/qwen3-coder-next",            # open-weight, code-specialized, efficient MoE coder
#     "mistralai/codestral-2508",         # code-specialized, low-latency code correction/test-generation baseline

#     # Added strong recent models
#     "openai/gpt-5.3-codex",             # commercial, strong code-specialized reasoning judge
#     "minimax/minimax-m2.7",             # MoE, 230B total / 10B active, agentic/general judge
#     "moonshotai/kimi-k2.6",             # open-weight MoE, 1T total / 32B active, strong reasoning/coding
# ]

LABEL_GUIDANCE: dict[str, str] = {
    "equivalent": "Semantically equivalent to the intended model.",
    "unsound": "Allows invalid solutions (false positives).",
    "incomplete": "Misses valid solutions (false negatives).",
    "unsound-incomplete": "Has both false positives and false negatives.",
    "non-executable": "Syntactically/API invalid and cannot execute.",
    # "status-only correct": "Likely preserves SAT/UNSAT status but not exact solution space.",
}

DATA_ROOT = Path("data-storage")


@dataclass
class ExperimentContext:
    """Everything one CP problem needs to be benchmarked end-to-end."""

    targeted_id: str
    problem_description: str
    reference_cp_model: str
    reference_language: str
    example_instance: str
    decision_variables: list[str]
    data_instances: list[Any] = field(default_factory=list)
    n_instances: int = 3
    solution_limit: int = 1000000
    time_limit_cpmpy_sec: int = 60
    time_limit_minizinc_sec: int = 60

    @property
    def problem_root(self) -> Path:
        return DATA_ROOT / self.targeted_id

    @property
    def output_dir(self) -> Path:
        path = self.problem_root / "candidates"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def instances_dir(self) -> Path:
        path = self.problem_root / "instances"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def time_limit_for(self, language: str) -> int:
        if language.strip().lower() == "cpmpy":
            return self.time_limit_cpmpy_sec
        return self.time_limit_minizinc_sec


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file as a list of dicts."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_problem_by_id(records: list[dict[str, Any]], problem_id: str) -> dict[str, Any]:
    """Return the first record whose `id` matches `problem_id`."""
    for record in records:
        if record.get("id") == problem_id:
            return record
    raise ValueError(f"Problem id not found: {problem_id}")


def context_from_record(record: dict[str, Any]) -> ExperimentContext:
    """Build an :class:`ExperimentContext` from a single JSONL benchmark record.

    The JSONL schema produced by *dcp-bench-open* contains at minimum:
    ``id``, ``description``, ``model``, ``example_instance``,
    ``decision_variables``, and ``instances``.
    """
    return ExperimentContext(
        targeted_id=record["id"],
        problem_description=record.get("description", ""),
        reference_cp_model=record.get("model", ""),
        reference_language="CPMpy",
        example_instance=record.get("example_instance", ""),
        decision_variables=record.get("decision_variables", []),
        data_instances=record.get("instances", []),
    )


def contexts_from_jsonl(
    path: Path,
    num_problems: int | None = None,
    problem_ids: list[str] | None = None,
) -> list[ExperimentContext]:
    """Load benchmark problems from a JSONL file and return :class:`ExperimentContext` objects.

    Args:
        path: Path to the ``.jsonl`` file.
        num_problems: Cap on how many problems to return (applied after ID
            filtering).  ``None`` means no limit.
        problem_ids: If given, only records whose ``id`` is in this list are
            returned.  Order from the file is preserved.

    Returns:
        A list of :class:`ExperimentContext` objects ready for the benchmark.
    """
    records = load_jsonl(path)
    if problem_ids is not None:
        id_set = set(problem_ids)
        records = [r for r in records if r.get("id") in id_set]
    if num_problems is not None:
        records = records[:num_problems]
    return [context_from_record(r) for r in records]


def default_context() -> ExperimentContext:
    """Hard-coded fallback used when no problem file is supplied."""
    return ExperimentContext(
        targeted_id="domino_tiling",
        problem_description=_DOMINO_TILING_DESCRIPTION,
        reference_cp_model=_DOMINO_TILING_REFERENCE_MODEL,
        reference_language="CPMpy",
        example_instance="m = 4\nn = 6",
        decision_variables=["h", "v"],
        time_limit_minizinc_sec=60,
        time_limit_cpmpy_sec=60,
        solution_limit=1000000,
        n_instances=3,
        data_instances=[],
    )


_DOMINO_TILING_DESCRIPTION = """
Consider an m x n rectangular chessboard. We want to tile this board with dominoes,
where each domino is a 2 x 1 rectangle. A tiling is a placement of dominoes such that
every square of the board is covered exactly once, no dominoes overlap, and no domino
extends beyond the boundary of the board.

Print one valid tiling of the chessboard.
""".strip()


_DOMINO_TILING_REFERENCE_MODEL = """
from cpmpy import *

h = boolvar(shape=(m, n - 1), name="h")  # horizontal domino starts
v = boolvar(shape=(m - 1, n), name="v")  # vertical domino starts

model = Model()
for i in range(m):
    for j in range(n):
        cover = []
        if j > 0:
            cover.append(h[i, j - 1])
        if j < n - 1:
            cover.append(h[i, j])
        if i > 0:
            cover.append(v[i - 1, j])
        if i < m - 1:
            cover.append(v[i, j])
        model += (sum(cover) == 1)

if model.solve():
    print(h.value())
    print(v.value())
""".strip()
