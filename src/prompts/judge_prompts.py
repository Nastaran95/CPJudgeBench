"""Prompt builders for all LLM-as-judge evaluation variants."""
from __future__ import annotations

from ..config import CORRECTNESS_LABELS, ExperimentContext, LABEL_GUIDANCE


# ---------------------------------------------------------------------------
# Label judges  (output: one of CORRECTNESS_LABELS)
# ---------------------------------------------------------------------------

def label_judge_prompt_reference_free(
    context: ExperimentContext,
    candidate_code: str,
    language: str,
) -> str:
    """Reference-free label judge: classify using problem description only."""
    labels_text = ", ".join(CORRECTNESS_LABELS)
    label_guidance_text = "\n".join(
        f"- {label}: {guidance}" for label, guidance in LABEL_GUIDANCE.items()
    )
    return f"""
You are an expert CP model judge.
Classify the candidate with exactly one allowed correctness label.

Allowed labels: {labels_text}

Label guidance:
{label_guidance_text}

Reasoning protocol (perform internally before answering):
1) Check executability for the given language (syntax/API shape + obvious runtime issues).
2) Infer intended semantics from the problem description.
3) Compare candidate constraints to intended semantics.
4) Decide the appropriate label.

Output format (STRICT JSON only):
{{"label": "<one allowed label>", "confidence": <0..1>, "rationale": "<1-3 short sentences>"}}

Problem description:
{context.problem_description}

Candidate ({language}):
{candidate_code}
""".strip()


def label_judge_prompt_reference_based(
    context: ExperimentContext,
    candidate_code: str,
    language: str,
) -> str:
    """Reference-based label judge: classify using problem description + reference model."""
    labels_text = ", ".join(CORRECTNESS_LABELS)
    label_guidance_text = "\n".join(
        f"- {label}: {guidance}" for label, guidance in LABEL_GUIDANCE.items()
    )
    return f"""
You are an expert CP model judge.
Classify the candidate with exactly one allowed correctness label.
Use the reference model below as the ground-truth encoding of the problem.

Allowed labels: {labels_text}

Label guidance:
{label_guidance_text}

Reasoning protocol (perform internally before answering):
1) Understand the reference model's constraints and the solution space it encodes.
2) Check whether the candidate is executable (syntax/API shape + obvious runtime issues).
3) Compare the candidate's constraints to the reference model, identifying false positives
   (solutions allowed by candidate but rejected by reference) and false negatives
   (solutions accepted by reference but rejected by candidate).
4) Decide the appropriate label.

Output format (STRICT JSON only):
{{"label": "<one allowed label>", "confidence": <0..1>, "rationale": "<1-3 short sentences>"}}

Problem description:
{context.problem_description}

Reference model ({context.reference_language}):
{context.reference_cp_model}

Candidate ({language}):
{candidate_code}
""".strip()


# ---------------------------------------------------------------------------
# Score judges  (output: integer 1–5)
# ---------------------------------------------------------------------------

def score_judge_prompt_reference_free(
    context: ExperimentContext,
    candidate_code: str,
    language: str,
) -> str:
    """Reference-free score judge: rate 1–5 against problem description semantics."""
    return f"""
You are an expert CP model evaluator.
Score the candidate constraint programming model on a scale of 1–5 based on how well
its solution space matches the intended semantics described in the problem description.

Scoring rubric:
5 - Semantically equivalent: the solution space exactly matches what the problem requires
4 - Mostly correct: minor issues that slightly restrict or expand the solution space
3 - Partially correct: captures key constraints but with notable omissions or additions
2 - Mostly incorrect: fundamentally misses the problem semantics in significant ways
1 - Completely wrong or non-executable: does not model the problem at all

Output format (STRICT JSON only):
{{"score": <integer 1..5>, "rationale": "<1-3 short sentences>"}}

Problem description:
{context.problem_description}

Candidate ({language}):
{candidate_code}
""".strip()


def score_judge_prompt_reference_based(
    context: ExperimentContext,
    candidate_code: str,
    language: str,
) -> str:
    """Reference-based score judge: rate 1–5 against the reference model's solution space."""
    return f"""
You are an expert CP model evaluator.
Score the candidate constraint programming model on a scale of 1–5 based on how closely
its solution space matches the solution space of the provided reference model.

Scoring rubric:
5 - Equivalent solution spaces: accepts exactly the same solutions as the reference
4 - Near-equivalent: very minor differences (edge cases or trivially different encodings)
3 - Partial overlap: solution spaces overlap significantly but with notable differences
2 - Poor match: solution spaces differ substantially
1 - No meaningful overlap or non-executable

Output format (STRICT JSON only):
{{"score": <integer 1..5>, "rationale": "<1-3 short sentences>"}}

Problem description:
{context.problem_description}

Reference model ({context.reference_language}):
{context.reference_cp_model}

Candidate ({language}):
{candidate_code}
""".strip()


# ---------------------------------------------------------------------------
# Binary judges  (output: "correct" | "incorrect")
# ---------------------------------------------------------------------------

def binary_judge_prompt_reference_free(
    context: ExperimentContext,
    candidate_code: str,
    language: str,
) -> str:
    """Reference-free binary judge: correct/incorrect using problem description only."""
    return f"""
You are an expert CP model judge.
Determine whether the candidate constraint programming model is CORRECT or INCORRECT
based solely on the problem description.

Definitions:
- CORRECT: the model's solution space semantically matches what the problem requires
  (it accepts all and only the valid solutions described).
- INCORRECT: the model allows invalid solutions, misses valid solutions, or cannot execute.

Output format (STRICT JSON only):
{{"verdict": "correct" or "incorrect", "confidence": <0..1>, "rationale": "<1-3 short sentences>"}}

Problem description:
{context.problem_description}

Candidate ({language}):
{candidate_code}
""".strip()


def binary_judge_prompt_reference_based(
    context: ExperimentContext,
    candidate_code: str,
    language: str,
) -> str:
    """Reference-based binary judge: correct/incorrect by comparing to the reference model."""
    return f"""
You are an expert CP model judge.
Determine whether the candidate constraint programming model is CORRECT or INCORRECT
by comparing it against the provided reference model.

Definitions:
- CORRECT: the candidate's solution space is equivalent to the reference model's solution space.
- INCORRECT: the candidate allows solutions the reference rejects, misses solutions the reference
  accepts, or cannot execute.

Output format (STRICT JSON only):
{{"verdict": "correct" or "incorrect", "confidence": <0..1>, "rationale": "<1-3 short sentences>"}}

Problem description:
{context.problem_description}

Reference model ({context.reference_language}):
{context.reference_cp_model}

Candidate ({language}):
{candidate_code}
""".strip()
