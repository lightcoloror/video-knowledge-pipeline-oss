from __future__ import annotations

import importlib
import inspect

from pathlib import Path
from typing import Any

from .models import now_iso
from .online_model_gateway import online_model_api_call
from .storage import write_json


SCHEMA = "video_knowledge_pipeline.model_task_gateway.v1"
COVERAGE_SCHEMA = "video_knowledge_pipeline.model_task_coverage.v1"


# Authoritative task -> model -> gateway contract used by agents and tests.
MODEL_TASKS: dict[str, dict[str, Any]] = {
    "cloud_asr": {"model_type": "asr", "modality": "audio", "module": "cloud_asr", "execution": "online", "migration_status": "unified", "providers": ["litellm", "openai_compatible_asr"]},
    "local_asr_service": {"model_type": "asr", "modality": "audio", "module": "local_asr_service_adapter", "execution": "local_http_or_explicit_remote", "migration_status": "unified", "providers": ["speaches_openai_compatible", "custom_openai_compatible_asr"]},
    "multimodal_frame_analysis": {"model_type": "semantic_frame", "modality": "image", "module": "multimodal_frame_analyzer", "execution": "online_or_local_http", "migration_status": "unified", "providers": ["litellm", "gemini", "openai_compatible", "local_qwen_vl"]},
    "temporal_visual_analysis": {"model_type": "temporal_sequence", "modality": "multi_image", "module": "temporal_visual_analyzer", "execution": "online_or_local_http", "migration_status": "unified", "providers": ["litellm", "gemini", "openai_compatible", "local_qwen_vl"]},
    "smart_summary_rewrite": {"model_type": "summary_rewrite", "modality": "text", "module": "smart_summary_codex", "execution": "online_or_agent_import", "migration_status": "unified", "providers": ["litellm", "openai_compatible"]},
    "smart_summary_section_rewrite": {"model_type": "summary_rewrite", "modality": "text", "module": "smart_summary_section_llm", "execution": "online", "migration_status": "unified", "providers": ["litellm", "openai_compatible"]},
    "smart_summary_global_reduce": {"model_type": "summary_rewrite", "modality": "text", "module": "smart_summary_global_reduce", "execution": "online", "migration_status": "unified", "providers": ["litellm", "openai_compatible"]},
    "transcript_readable_polish": {"model_type": "text_llm", "modality": "text", "module": "transcript_readable_llm", "execution": "online_or_agent_substitute", "migration_status": "unified", "providers": ["litellm", "openai_compatible"]},
    "transcript_correction_pack": {"model_type": "transcript_correction", "modality": "text", "module": "transcript_correction_pack", "execution": "online_or_import", "migration_status": "unified", "providers": ["litellm", "openai_compatible"]},
    "transcript_candidate_discovery": {"model_type": "transcript_correction", "modality": "text", "module": "transcript_semantic_correction", "execution": "online_or_agent_substitute", "migration_status": "unified", "providers": ["litellm", "openai_compatible"]},
    "transcript_semantic_correction": {"model_type": "transcript_correction", "modality": "text", "module": "transcript_semantic_correction", "execution": "online_or_agent_substitute", "migration_status": "unified", "providers": ["litellm", "openai_compatible"]},
    "term_arbitration": {"model_type": "transcript_correction", "modality": "text", "module": "model_task_automation", "execution": "online_or_import", "migration_status": "unified", "providers": ["litellm", "openai_compatible"]},
    "bilinote_mind_map": {"model_type": "text_llm", "modality": "text", "module": "model_task_automation", "execution": "online_or_prompt_import", "migration_status": "unified", "providers": ["litellm", "openai_compatible"]},
    "provider_task_benchmark": {"model_type": "text_llm", "modality": "text", "module": "volcengine_model_task_matrix", "execution": "online_benchmark", "migration_status": "unified", "providers": ["litellm", "openai_compatible"]},
    "native_video_segment": {"model_type": "video_segment", "modality": "video", "module": "online_model_gateway", "execution": "online_provider_capability", "migration_status": "unified", "providers": ["gemini"]},
}


MODEL_TASKS["online_ocr"] = {
    "model_type": "ocr",
    "modality": "multi_image",
    "module": "ocr_route",
    "execution": "online_via_consented_connector",
    "migration_status": "unified",
    "providers": ["litellm", "gemini", "openai_compatible", "local_qwen_vl"],
}

def model_task_api_call(
    task: str,
    *,
    provider_config: dict[str, Any] | None = None,
    prompt: str = "",
    input_text: str = "",
    messages: list[dict[str, Any]] | None = None,
    image_paths: list[str] | None = None,
    audio_path: str = "",
    video_path: str = "",
    execute: bool = False,
    temperature: float = 0,
    response_format: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    max_retries: int | None = None,
    output_dir: str | Path | None = None,
    allowed_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
    write: bool = False,
) -> dict[str, Any]:
    key = _normalise_task(task)
    spec = MODEL_TASKS[key]
    result = online_model_api_call(
        str(spec["model_type"]), provider_config=provider_config, prompt=prompt,
        input_text=input_text, messages=messages, image_paths=image_paths,
        audio_path=audio_path, video_path=video_path, execute=execute, temperature=temperature,
        response_format=response_format, max_tokens=max_tokens, max_retries=max_retries,
        output_dir=output_dir, allowed_roots=allowed_roots, write=write,
    )
    return {**result, "schema": SCHEMA, "task": key, "model_type": spec["model_type"], "task_contract": {"modality": spec["modality"], "module": spec["module"], "execution": spec["execution"], "migration_status": spec["migration_status"]}}


def model_task_coverage_audit(*, output_dir: str | Path | None = None, write: bool = True) -> dict[str, Any]:
    rows = []
    for task, spec in MODEL_TASKS.items():
        declared_status = str(spec["migration_status"])
        implementation = _implementation_check(spec)
        status = "contract_drift" if declared_status == "unified" and not implementation["uses_model_task_gateway"] else declared_status
        rows.append({"task": task, "model_type": spec["model_type"], "modality": spec["modality"], "module": spec["module"], "gateway": "model_task_gateway" if status == "unified" else ("online_model_gateway" if status == "deferred" else "legacy_direct_adapter"), "execution": spec["execution"], "migration_status": status, "declared_migration_status": declared_status, "implementation": implementation, "providers": list(spec["providers"]), "online_api_adapter": bool(spec["providers"]), "needs_migration": status in {"legacy_adapter", "contract_drift"}, "deferred": status == "deferred"})
    counts = {"total": len(rows), "unified": sum(row["migration_status"] == "unified" for row in rows), "legacy_adapter": sum(row["migration_status"] == "legacy_adapter" for row in rows), "deferred": sum(row["migration_status"] == "deferred" for row in rows), "contract_drift": sum(row["migration_status"] == "contract_drift" for row in rows), "with_online_adapter": sum(bool(row["online_api_adapter"]) for row in rows)}
    result = {"schema": COVERAGE_SCHEMA, "status": "complete" if counts["legacy_adapter"] == 0 and counts["contract_drift"] == 0 else "migration_required", "counts": counts, "rows": rows, "deferred_policy": "Native whole-video understanding is capability-routed: supported Gemini routes upload an explicitly consented video through the Gemini Files API; unsupported providers report their actual capability response.", "updated_at": now_iso()}
    if output_dir and write:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "model-task-coverage.json", result)
        (root / "model-task-coverage.md").write_text(_render_coverage_markdown(result), encoding="utf-8")
        result["artifacts"] = {"json": str(root / "model-task-coverage.json"), "markdown": str(root / "model-task-coverage.md")}
    return result


def _implementation_check(spec: dict[str, Any]) -> dict[str, Any]:
    module_name = str(spec.get("module") or "").strip()
    try:
        module = importlib.import_module(f".{module_name}", __package__)
        source = inspect.getsource(module)
    except Exception as exc:
        return {"module_imported": False, "uses_model_task_gateway": False, "error": str(exc)}
    return {
        "module_imported": True,
        "uses_model_task_gateway": module_name == "online_model_gateway" or "model_task_api_call(" in source,
        "source_module": module_name,
    }

def _normalise_task(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    if key not in MODEL_TASKS:
        raise ValueError(f"unsupported model task: {value}; expected one of: {', '.join(MODEL_TASKS)}")
    return key


def _render_coverage_markdown(result: dict[str, Any]) -> str:
    counts = result["counts"]
    lines = ["# Model Task Gateway Coverage", "", f"- Status: `{result['status']}`", f"- Total: `{counts['total']}`", f"- Unified: `{counts['unified']}`", f"- Legacy adapter: `{counts['legacy_adapter']}`", f"- Deferred: `{counts['deferred']}`", f"- Contract drift: `{counts['contract_drift']}`", "", "| Task | Model type | Modality | Gateway | Providers | Status |", "|---|---|---|---|---|---|"]
    for row in result["rows"]:
        providers = ", ".join(row["providers"]) or "none"
        lines.append(f"| `{row['task']}` | `{row['model_type']}` | `{row['modality']}` | `{row['gateway']}` | {providers} | `{row['migration_status']}` |")
    lines.extend(["", f"> {result['deferred_policy']}", ""])
    return "\n".join(lines)
