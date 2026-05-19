"""CPJudgeBench - benchmark CP-model generation and judging with LLMs."""
from __future__ import annotations

from .config import (
    CORRECTNESS_LABELS,
    GENERATOR_MODELS,
    JUDGE_MODELS,
    LABEL_GUIDANCE,
    LANGUAGE_LABELS,
    ExperimentContext,
    default_context,
    get_problem_by_id,
    load_jsonl,
)
from .data_generation import run_data_generation
from .instances import generate_and_save_instances
from .judge import (
    JUDGE_APPROACHES,
    direct_llm_judge,
    run_judge_evaluation,
    run_judge_evaluation_all_approaches,
)
from .executors import evaluate_candidate_sat_only
from .llm import ModelSpec, get_openrouter_llm, llm_response_to_text
from .meta_eval import run_sat_meta_evaluation
from .pairwise_judge import direct_pairwise_judge, run_pairwise_evaluation

__all__ = [
    # config
    "CORRECTNESS_LABELS",
    "GENERATOR_MODELS",
    "JUDGE_MODELS",
    "JUDGE_APPROACHES",
    "LABEL_GUIDANCE",
    "LANGUAGE_LABELS",
    "ExperimentContext",
    "ModelSpec",
    "default_context",
    "get_problem_by_id",
    "load_jsonl",
    # llm
    "get_openrouter_llm",
    "llm_response_to_text",
    # pipeline
    "generate_and_save_instances",
    "run_data_generation",
    # executors
    "evaluate_candidate_sat_only",
    # pointwise judge
    "direct_llm_judge",
    "run_judge_evaluation",
    "run_judge_evaluation_all_approaches",
    # pairwise judge
    "direct_pairwise_judge",
    "run_pairwise_evaluation",
    # meta-evaluation
    "run_sat_meta_evaluation",
]
