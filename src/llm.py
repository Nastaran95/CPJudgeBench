"""OpenRouter chat-model client and lightweight model-spec wrapper."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import dotenv
from langchain.chat_models import init_chat_model


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelSpec:
    """Parsed reference to an LLM exposed via OpenRouter."""

    provider: str
    name: str
    tag: str | None = None
    raw: str = ""

    @classmethod
    def from_string(cls, model_str: str) -> "ModelSpec":
        if "/" not in model_str:
            raise ValueError(f"Model must be '<provider>/<model>[:tag]', got: {model_str!r}")
        provider, rest = model_str.split("/", 1)
        if ":" in rest:
            name, tag = rest.split(":", 1)
        else:
            name, tag = rest, None
        return cls(provider=provider, name=name, tag=tag, raw=model_str)

    @property
    def openrouter_model_name(self) -> str:
        model_id = f"{self.provider}/{self.name}"
        return f"{model_id}:{self.tag}" if self.tag else model_id


def _require_api_key() -> str:
    dotenv.load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment/.env")
    return api_key


def get_openrouter_llm(spec: ModelSpec):
    """Return a LangChain chat model bound to OpenRouter."""
    # OpenRouter exposes an OpenAI-compatible API for all providers; route
    # every model through the OpenAI client with the full OpenRouter model id.
    return init_chat_model(
        model=spec.openrouter_model_name,
        model_provider="openai",
        base_url=OPENROUTER_BASE_URL,
        api_key=_require_api_key(),
    )


def llm_response_to_text(response: Any) -> str:
    """Best-effort conversion of a chat-model response into a plain string."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts) if parts else str(content)

    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value

    return str(content)
