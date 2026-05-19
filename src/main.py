"""CLI entry point for the CPJudgeBench notebook workflow."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from .config import GENERATOR_MODELS, JUDGE_MODELS, contexts_from_jsonl, default_context
from .data_generation import run_data_generation
from .instances import generate_and_save_instances
from .judge import JUDGE_APPROACHES, run_judge_evaluation, run_judge_evaluation_all_approaches
from .llm import ModelSpec
from .meta_eval import run_sat_meta_evaluation
from .pairwise_judge import run_pairwise_evaluation


logger = logging.getLogger(__name__)


def _parse_models(values: list[str]) -> list[ModelSpec]:
    return [ModelSpec.from_string(v) for v in values]


def _build_log_path() -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return logs_dir / f"run-{timestamp}.log"


def _configure_logging(log_path: Path) -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


_DEFAULT_PROBLEMS_FILE = Path("extra_files/dcp-bench-open.jsonl")
_DEFAULT_NUM_PROBLEMS = 10


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPJudgeBench notebook workflow as Python modules.",
    )

    # ── Problem source ────────────────────────────────────────────────────────
    problem_group = parser.add_argument_group("problem source")
    problem_group.add_argument(
        "--problems-file",
        type=Path,
        default=_DEFAULT_PROBLEMS_FILE,
        metavar="PATH",
        help=(
            f"Path to a JSONL file of benchmark problems "
            f"(default: {_DEFAULT_PROBLEMS_FILE})."
        ),
    )
    problem_group.add_argument(
        "--num-problems",
        type=int,
        default=_DEFAULT_NUM_PROBLEMS,
        metavar="N",
        help=(
            f"Maximum number of problems to run from --problems-file "
            f"(default: {_DEFAULT_NUM_PROBLEMS}).  Ignored when --problem-ids is given."
        ),
    )
    problem_group.add_argument(
        "--problem-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Run only these specific problem IDs from --problems-file.  "
            "When set, --num-problems is ignored."
        ),
    )

    # ── Pipeline steps ────────────────────────────────────────────────────────
    parser.add_argument(
        "--generate-instances",
        action="store_true",
        help="Generate benchmark instances with LLMs.",
    )
    parser.add_argument(
        "--data-generation",
        action="store_true",
        help="Run candidate generation + validation loop.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM-as-judge evaluation (use --judge-approach to select variant).",
    )
    parser.add_argument(
        "--judge-approach",
        nargs="+",
        choices=JUDGE_APPROACHES,
        default=["reference_free"],
        metavar="APPROACH",
        help=(
            "Which judge prompt variant(s) to run when --judge is active. "
            f"Choices: {JUDGE_APPROACHES}. "
            "Pass multiple values to run both "
            "(e.g. --judge-approach reference_free reference_based). "
            "Default: reference_free."
        ),
    )
    parser.add_argument(
        "--pairwise-judge",
        action="store_true",
        help=(
            "Run pairwise LLM-as-judge evaluation: compare all pairs of candidates "
            "within the same language and pick the better one (or tie)."
        ),
    )
    parser.add_argument(
        "--sat-meta-eval",
        action="store_true",
        help=(
            "Run SAT-only meta-evaluation: label candidates via a single solve() call "
            "and compare against the full solution-space ground truth."
        ),
    )
    return parser


def load_contexts(args: argparse.Namespace) -> list:
    """Return the list of :class:`ExperimentContext` objects to benchmark.

    When *--problems-file* resolves to an existing JSONL file the contexts are
    derived from that file (filtered by *--problem-ids* and capped by
    *--num-problems*).  Otherwise the built-in ``domino_tiling`` fallback is
    used.
    """
    problems_file: Path = args.problems_file
    if problems_file.exists():
        num = None if args.problem_ids else args.num_problems
        contexts = contexts_from_jsonl(
            problems_file,
            num_problems=num,
            problem_ids=args.problem_ids,
        )
        logger.info(
            "[run] loaded %d problem(s) from %s", len(contexts), problems_file
        )
        return contexts

    logger.warning(
        "[run] --problems-file %s not found; falling back to default context",
        problems_file,
    )
    return [default_context()]


def _run_pipeline(
    args: argparse.Namespace,
    context,
    generator_specs: list[ModelSpec],
    judge_specs: list[ModelSpec],
    run_all: bool,
) -> None:
    """Execute the requested pipeline steps for a single *context*."""
    plain_payload = None

    if args.generate_instances or run_all:
        instance_results = generate_and_save_instances(context, generator_specs)
        logger.info("%s", instance_results)

    if args.data_generation or run_all:
        results_df, summary_df, plain_payload = run_data_generation(context, generator_specs)
        logger.info("\n%s", results_df)
        logger.info("\n%s", summary_df)

    if args.judge or run_all:
        approaches = args.judge_approach if not run_all else ["reference_free"]
        if len(approaches) > 1:
            judge_df, judge_summary_df = run_judge_evaluation_all_approaches(
                context, judge_specs, plain_payload=plain_payload
            )
        else:
            judge_df, judge_summary_df = run_judge_evaluation(
                context, judge_specs, plain_payload=plain_payload, approach=approaches[0]
            )
        logger.info("\n%s", judge_df)
        logger.info("\n%s", judge_summary_df)

    if args.pairwise_judge:
        pairwise_df, pairwise_summary_df = run_pairwise_evaluation(
            context, judge_specs, plain_payload=plain_payload
        )
        logger.info("\n%s", pairwise_df)
        logger.info("\n%s", pairwise_summary_df)

    if args.sat_meta_eval:
        meta_df, meta_summary_df = run_sat_meta_evaluation(
            context, plain_payload=plain_payload
        )
        logger.info("\n%s", meta_df)
        logger.info("\n%s", meta_summary_df)


def main() -> None:
    args = _build_parser().parse_args()
    log_path = _build_log_path()
    _configure_logging(log_path)
    generator_specs = _parse_models(GENERATOR_MODELS)
    judge_specs = _parse_models(JUDGE_MODELS)
    logger.info("[run] log file: %s", log_path)

    any_explicit = (
        args.generate_instances
        or args.data_generation
        or args.judge
        or args.pairwise_judge
        or args.sat_meta_eval
    )
    run_all = not any_explicit

    contexts = load_contexts(args)
    total = len(contexts)
    for idx, context in enumerate(contexts, start=1):
        # if idx == 4:
        #     continue
        logger.info(
            "[run] ── problem %d/%d: %s ──", idx, total, context.targeted_id
        )
        _run_pipeline(args, context, generator_specs, judge_specs, run_all)

    logger.info("[run] logs saved: %s", log_path)


if __name__ == "__main__":
    main()
