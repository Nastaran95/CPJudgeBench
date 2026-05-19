"""Generate, save, and merge benchmark instances from multiple LLMs."""
from __future__ import annotations

import json
import logging
from typing import Any

from .config import ExperimentContext
from .generation import generate_instances
from .llm import ModelSpec

logger = logging.getLogger(__name__)


def _safe_filename(model_label: str) -> str:
    return model_label.replace("/", "_").replace(":", "_")


def _save_per_model_instances(
    context: ExperimentContext, model_specs: list[ModelSpec]
) -> dict[str, dict[str, Any]]:
    """Generate and persist one JSON file per LLM."""
    results: dict[str, dict[str, Any]] = {}
    for spec in model_specs:
        out_path = context.instances_dir / f"{_safe_filename(spec.raw)}.json"
        if out_path.exists():
            logger.info("Skip existing: %s", out_path)
            continue

        try:
            generated = generate_instances(context, spec, count=context.n_instances)
            out_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
            results[spec.raw] = {"ok": True, "count": len(generated), "path": str(out_path)}
        except Exception as e:
            results[spec.raw] = {"ok": False, "error": str(e)}
    return results


def _merge_instances(
    context: ExperimentContext, model_specs: list[ModelSpec]
) -> list[dict[str, Any]]:
    """Union per-model files, deduplicate by `variables`, keep model provenance."""
    union: list[dict[str, Any]] = []
    for spec in model_specs:
        in_path = context.instances_dir / f"{_safe_filename(spec.raw)}.json"
        if not in_path.exists():
            continue

        loaded = json.loads(in_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            continue

        for instance in loaded:
            if not isinstance(instance, dict):
                continue
            for existing in union:
                if existing["variables"] == instance["variables"]:
                    existing["model_names"].append(spec.raw)
                    existing["case_notes"].append(instance["case_note"])
                    break
            else:
                union.append({
                    "variables": instance["variables"],
                    "case_notes": [instance["case_note"]],
                    "model_names": [spec.raw],
                })
    return union


def generate_and_save_instances(
    context: ExperimentContext, model_specs: list[ModelSpec]
) -> dict[str, dict[str, Any]]:
    """Generate per-model instances, save them, and write a deduplicated union file."""
    results = _save_per_model_instances(context, model_specs)
    union = _merge_instances(context, model_specs)

    union_path = context.instances_dir / "all_instances.json"
    union_path.write_text(json.dumps(union, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved all instances: %s with %d instances", union_path, len(union))
    return results
