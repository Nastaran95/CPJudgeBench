"""LLM-driven generators for benchmark instances and candidate CP models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentContext, LABEL_GUIDANCE
from .llm import ModelSpec, get_openrouter_llm, llm_response_to_text
from .parsing import extract_code_block, extract_json_array


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def _normalize_instance(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Instance #{index} is not a JSON object")

    variables = raw.get("variables")
    if not isinstance(variables, dict):
        raise ValueError(f"Instance #{index} must contain object field `variables`")

    case_note = raw.get("case_note")
    if not isinstance(case_note, str) or not case_note.strip():
        raise ValueError(f"Instance #{index} must contain non-empty string field `case_note`")

    return {"variables": variables, "case_note": case_note.strip()}


def _instance_generation_prompt(context: ExperimentContext, count: int) -> str:
    return f"""
Generate exactly {count} small, valid, diverse benchmark instances for the constraint-programming problem below.

Problem:
{context.problem_description}

Reference model:
{context.reference_cp_model}

Example instance:
{context.example_instance}

Infer the input schema and invariants from the reference model. Include typical, boundary, near-degenerate, modest stress, and bug-revealing cases. Keep instances solver-validatable within the time limit. Avoid large scales.

Return only this JSON shape:
[
  {{
    "variables": {{ "<input_field>": <value> }},
    "case_note": "<short description>"
  }}
]

Strict rules:
- Output a JSON array with exactly {count} objects.
- Each object has only "variables" and "case_note".
- All instance inputs go inside "variables".
- Use the exact key names "variables" and "case_note".
- No markdown, prose, or code fences.
""".strip()


def generate_instances(
    context: ExperimentContext, spec: ModelSpec, count: int = 5
) -> list[dict[str, Any]]:
    llm = get_openrouter_llm(spec)
    text = llm_response_to_text(llm.invoke(_instance_generation_prompt(context, count)))
    instances = extract_json_array(text)
    normalized = [_normalize_instance(item, idx) for idx, item in enumerate(instances, start=1)]
    if len(normalized) != count:
        raise ValueError(f"Expected exactly {count} instances, got {len(normalized)}")
    return normalized


# ---------------------------------------------------------------------------
# Candidate model generation
# ---------------------------------------------------------------------------

_DATA_GENERATION_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "data_generation"
_LABEL_PROMPT_FILES: dict[str, str] = {
    "equivalent": "equivalent.txt",
    "unsound": "unsound.txt",
    "incomplete": "incomplete.txt",
    "unsound-incomplete": "unsound-incomplete.txt",
    "non-executable": "non-executable.txt",
    "status-only correct": "status-only-correct.txt",
}


def _load_data_generation_prompt_template(label: str) -> str:
    filename = _LABEL_PROMPT_FILES.get(label)
    if filename is None:
        raise ValueError(f"Unknown correctness label: {label}")
    prompt_path = _DATA_GENERATION_PROMPTS_DIR / filename
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Missing data-generation prompt template for label '{label}': {prompt_path}"
        )
    return prompt_path.read_text(encoding="utf-8")


def _syntax_repair_prompt(
    *,
    language: str,
    label: str,
    feedback_notes: str,
    previous_attempt: str,
) -> str:
    return f"""
Revise the following CP model code to fix syntax/parsing issues only.

Language: {language}
Target label must remain: {label}

Instructions:
- Preserve the model's original formulation, constraints, and intent.
- Make the minimum edits required for syntactic validity and parser/compile success.
- Do not redesign the model, change semantics, or switch formulation.
- Keep variable names and structure as close as possible.

Failure detail:
{feedback_notes}

Code to revise:
{previous_attempt}

Output only raw code. No prose or markdown fences.
""".strip()


def _candidate_model_prompt(
    context: ExperimentContext,
    language: str,
    label: str,
    feedback_notes: str,
    feedback_mode: str,
    previous_attempt: str,
) -> str:
    if feedback_notes and feedback_mode == "syntax_repair":
        return _syntax_repair_prompt(
            language=language,
            label=label,
            feedback_notes=feedback_notes,
            previous_attempt=previous_attempt,
        )

    template = _load_data_generation_prompt_template(label)
    if feedback_notes:
        feedback_section = (
            "Feedback from previous attempt:\n"
            f"{feedback_notes}\n\n"
            "Feedback type: solver/validation mismatch.\n"
            "Refine constraints and modeling choices to match the target label behavior.\n"
        )
        previous_attempt_section = f"Previous attempt:\n{previous_attempt}\n"
    else:
        feedback_section = ""
        previous_attempt_section = ""
    return template.format(
        problem_description=context.problem_description,
        reference_cp_model=context.reference_cp_model,
        example_instance=context.example_instance,
        language=language,
        label=label,
        label_guidance=LABEL_GUIDANCE[label],
        label_guidance_dict=LABEL_GUIDANCE,
        feedback_section=feedback_section,
        previous_attempt_section=previous_attempt_section,
    ).strip()


def generate_candidate_model(
    context: ExperimentContext,
    spec: ModelSpec,
    language: str,
    label: str,
    feedback_notes: str = "",
    feedback_mode: str = "",
    previous_attempt: str = "",
) -> str:
    llm = get_openrouter_llm(spec)
    prompt = _candidate_model_prompt(
        context,
        language,
        label,
        feedback_notes,
        feedback_mode,
        previous_attempt,
    )
    text = llm_response_to_text(llm.invoke(prompt))
    return extract_code_block(text)
