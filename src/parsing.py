"""Tolerant extractors for structured data embedded in LLM text output."""
from __future__ import annotations

import json
import re
from typing import Any

from .config import CORRECTNESS_LABELS


_CODE_FENCE_LANGUAGE_HINTS = {"python", "minizinc"}


def extract_json_array(text: str) -> list[Any]:
    """Parse a JSON array from `text`, tolerating leading/trailing prose."""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("No JSON array found in model output")
    return json.loads(text[start : end + 1])


def extract_code_block(text: str) -> str:
    """Return the longest fenced-code block, or the trimmed text if none exist."""
    if "```" not in text:
        return text.strip()

    parts = text.split("```")
    if len(parts) < 3:
        return text.strip()

    candidate = max(parts[1::2], key=len).strip()
    lines = candidate.splitlines()
    if lines and lines[0].strip().lower() in _CODE_FENCE_LANGUAGE_HINTS:
        lines = lines[1:]
    return "\n".join(lines).strip()


def normalize_label(text: str) -> str:
    """Map free-form judge output to a canonical correctness label."""
    lowered = (text or "").lower()
    for label in sorted(CORRECTNESS_LABELS, key=len, reverse=True):
        if label.lower() in lowered:
            return label
    return "unknown"


def parse_judge_json(raw_text: str) -> dict[str, Any]:
    """Parse a judge response into a dict; fall back to best-effort extraction."""
    text = (raw_text or "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {
        "label": normalize_label(text),
        "confidence": None,
        "rationale": text[:300],
    }


_PAIRWISE_WINNERS = {"a", "b", "tie"}


def _normalize_winner(raw: str) -> str:
    """Map free-form pairwise output to 'A', 'B', 'tie', or 'unknown'."""
    v = str(raw or "").strip().lower()
    if v in _PAIRWISE_WINNERS:
        return v.upper() if v in {"a", "b"} else v
    for w in ("tie", "b", "a"):
        if w in v:
            return w.upper() if w in {"a", "b"} else w
    return "unknown"


def parse_pairwise_judge_json(raw_text: str) -> dict[str, Any]:
    """Parse a pairwise judge response; fall back to best-effort extraction.

    Expected LLM output: {"winner": "A"|"B"|"tie", "confidence": 0..1, "rationale": "..."}
    Normalises winner to "A", "B", "tie", or "unknown".
    """
    text = (raw_text or "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed["winner"] = _normalize_winner(str(parsed.get("winner", "")))
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                parsed["winner"] = _normalize_winner(str(parsed.get("winner", "")))
                return parsed
        except json.JSONDecodeError:
            pass

    return {
        "winner": _normalize_winner(text),
        "confidence": None,
        "rationale": text[:300],
    }
