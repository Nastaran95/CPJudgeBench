"""Tolerant extractors for structured data embedded in LLM text output."""
from __future__ import annotations

import io
import json
import re
import tokenize as _tokenize
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


# ---------------------------------------------------------------------------
# Comment stripping
# ---------------------------------------------------------------------------

def _strip_python_comments(code: str) -> str:
    """Remove # comments from Python/CPMpy code using the tokenize module.

    The original whitespace and indentation of non-comment tokens is
    preserved by re-deriving it from character offsets.
    """
    try:
        source = code if code.endswith("\n") else code + "\n"
        lines = source.splitlines(keepends=True)

        # Build cumulative character offsets for each line (0-indexed line → start offset).
        line_offsets: list[int] = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        comment_ranges: list[tuple[int, int]] = []
        for ttype, _, (sr, sc), (er, ec), _ in _tokenize.generate_tokens(
            io.StringIO(source).readline
        ):
            if ttype == _tokenize.COMMENT:
                start = line_offsets[sr - 1] + sc
                end = line_offsets[er - 1] + ec
                comment_ranges.append((start, end))

        if not comment_ranges:
            return code.strip()

        chars = list(source)
        for start, end in comment_ranges:
            for i in range(start, min(end, len(chars))):
                chars[i] = ""

        stripped = "".join(chars)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped.strip()
    except _tokenize.TokenError:
        return code.strip()


def _strip_minizinc_comments(code: str) -> str:
    """Remove % line comments and /* */ block comments from MiniZinc code."""
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"%[^\n]*", "", code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def strip_code_comments(code: str, language: str) -> str:
    """Remove comments from *code* before it is sent to a judge LLM.

    Dispatches to a language-appropriate stripper:
    - ``"cpmpy"`` / ``"python"`` → tokenize-based # comment removal
    - ``"minizinc"`` → regex-based % / /* */ removal
    - anything else → code returned unchanged
    """
    lang = language.strip().lower()
    if lang in ("cpmpy", "python"):
        return _strip_python_comments(code)
    if lang == "minizinc":
        return _strip_minizinc_comments(code)
    return code


# ---------------------------------------------------------------------------
# Score and binary response parsers
# ---------------------------------------------------------------------------

def parse_score_json(raw_text: str) -> dict[str, Any]:
    """Parse a score-judge response; returns ``{"score": int, "rationale": str}``.

    ``score`` is clamped to [1, 5] if the LLM returns something out of range.
    Falls back to ``score=None`` when no valid integer is found.
    """
    text = (raw_text or "").strip()

    def _clamp(v: Any) -> int | None:
        try:
            n = int(v)
            return max(1, min(5, n))
        except (TypeError, ValueError):
            return None

    for attempt in (text, None):
        try:
            parsed = json.loads(attempt or text)
            if isinstance(parsed, dict):
                return {
                    "score": _clamp(parsed.get("score")),
                    "rationale": str(parsed.get("rationale", "")),
                }
        except json.JSONDecodeError:
            pass
        if attempt is None:
            break
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
        else:
            break

    # Last resort: search for a bare digit 1-5
    m = re.search(r"\b([1-5])\b", raw_text or "")
    return {
        "score": int(m.group(1)) if m else None,
        "rationale": (raw_text or "")[:300],
    }


_BINARY_VERDICTS = {"correct", "incorrect"}


def _normalize_verdict(raw: str) -> str:
    v = str(raw or "").strip().lower()
    if v in _BINARY_VERDICTS:
        return v
    if "incorrect" in v or "wrong" in v or "false" in v:
        return "incorrect"
    if "correct" in v or "right" in v or "true" in v:
        return "correct"
    return "unknown"


def parse_binary_json(raw_text: str) -> dict[str, Any]:
    """Parse a binary-judge response; returns ``{"verdict": "correct"|"incorrect", ...}``.

    Falls back to best-effort verdict extraction when the LLM does not return
    clean JSON.
    """
    text = (raw_text or "").strip()

    for attempt in (text, None):
        try:
            parsed = json.loads(attempt or text)
            if isinstance(parsed, dict):
                parsed["verdict"] = _normalize_verdict(str(parsed.get("verdict", "")))
                return parsed
        except json.JSONDecodeError:
            pass
        if attempt is None:
            break
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
        else:
            break

    return {
        "verdict": _normalize_verdict(raw_text or ""),
        "confidence": None,
        "rationale": (raw_text or "")[:300],
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
