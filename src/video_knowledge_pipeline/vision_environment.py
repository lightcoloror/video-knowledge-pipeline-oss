from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import project_root
from .model_defaults import GEMINI_DEFAULT_MODEL
from .storage import write_json
from .vision_api import provider_requires_api_key
from .vision_api import redact_url_secrets
from .vision_api import resolve_provider_config


VISION_ENV_FILES = (".local/vision.env", ".local/video-knowledge.env")


def vision_environment_status(
    *,
    provider: str = "",
    model: str = "",
    write_template: bool = False,
    template_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Inspect no-secret local vision provider configuration."""
    root = project_root()
    env_files = [_env_file_status(root / relative) for relative in VISION_ENV_FILES]
    requested = {key: value for key, value in {"provider": provider, "model": model}.items() if str(value or "").strip()}
    cfg = resolve_provider_config(requested or None)
    template_target = Path(template_path).expanduser() if template_path else root / ".local" / "vision.env"
    template_result = _write_template(template_target, provider=provider, model=model, overwrite=overwrite) if write_template else {}
    if write_template:
        env_files = [_env_file_status(root / relative) for relative in VISION_ENV_FILES]
    result = {
        "schema": "lecture_vision_environment_status.v1",
        "project_root": str(root),
        "provider": {
            "provider": cfg.get("provider"),
            "base_url": redact_url_secrets(str(cfg.get("base_url") or "")),
            "model": cfg.get("model"),
            "api_key_required": provider_requires_api_key(cfg),
            "api_key_configured": bool(cfg.get("api_key")),
            "timeout_seconds": cfg.get("timeout_seconds"),
        },
        "sanitized_provider_config": _sanitized_provider_config(cfg),
        "env_files": env_files,
        "template_path": str(template_target),
        "template_written": bool(template_result.get("written")),
        "template_status": template_result.get("status", "not_requested"),
        "next_action": _next_action(cfg, env_files, template_target),
    }
    if write_template:
        handoff = root / ".local" / "vision-env-status.json"
        write_json(handoff, result)
        result["status_json_path"] = str(handoff)
    return result


def _env_file_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    keys: list[str] = []
    if exists:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = _parse_env_name(line)
            if parsed:
                keys.append(parsed)
    return {
        "path": str(path),
        "exists": exists,
        "keys": sorted(set(keys)),
        "contains_api_key_name": any(name.endswith("_API_KEY") or name == "LECTURE_VISION_API_KEY" for name in keys),
    }


def _write_template(path: Path, *, provider: str = "", model: str = "", overwrite: bool = False) -> dict[str, Any]:
    if path.exists() and not overwrite:
        return {"status": "exists", "written": False, "path": str(path)}
    selected_provider = str(provider or os.environ.get("LECTURE_VISION_PROVIDER") or "agnes").strip()
    selected_model = str(model or os.environ.get("LECTURE_VISION_MODEL") or _default_model(selected_provider)).strip()
    key_name = _key_name(selected_provider)
    lines = [
        "# Local-only vision provider settings. Do not commit this file.",
        f"LECTURE_VISION_PROVIDER={selected_provider}",
        *([f"{key_name}=<your key>"] if key_name else []),
        *(_provider_extra_env_lines(selected_provider)),
        f"LECTURE_VISION_MODEL={selected_model}",
        "LECTURE_VISION_TIMEOUT_SECONDS=90",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"status": "written", "written": True, "path": str(path)}


def _next_action(cfg: dict[str, Any], env_files: list[dict[str, Any]], template_target: Path) -> dict[str, Any]:
    if not provider_requires_api_key(cfg) or cfg.get("api_key"):
        return {
            "key": "run_vision_provider_matrix",
            "label": "Run provider matrix before real vision execution",
            "command": ".\\scripts\\video-knowledge.ps1 vision-provider-matrix --providers \"local_qwen_vl,volcengine_coding_plan,gemini,openai,agnes\" --bundle-dir <bundle-dir> --image-probe-max-edge 512 --image-probe-jpeg-quality 55",
            "mcp_tool": "vision_provider_matrix",
            "human_required": False,
        }
    if not any(item.get("exists") for item in env_files):
        return {
            "key": "write_template",
            "label": "Write local vision env template",
            "command": ".\\scripts\\video-knowledge.ps1 vision-env-status --write-template",
            "template_path": str(template_target),
            "human_required": False,
        }
    existing = next((item for item in env_files if item.get("exists")), {})
    return {
        "key": "fill_api_key",
        "label": "Fill the local API key placeholder",
        "template_path": str(existing.get("path") or template_target),
        "human_required": True,
    }


def _sanitized_provider_config(cfg: dict[str, Any]) -> dict[str, Any]:
    result = {
        "provider": cfg.get("provider"),
        "base_url": redact_url_secrets(str(cfg.get("base_url") or "")),
        "model": cfg.get("model"),
        "timeout_seconds": cfg.get("timeout_seconds"),
    }
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def _parse_env_name(line: str) -> str:
    text = line.strip()
    if not text or text.startswith("#") or "=" not in text:
        return ""
    name = text.split("=", 1)[0].strip()
    if name.startswith("$env:"):
        name = name[5:]
    if not name.replace("_", "").isalnum():
        return ""
    return name


def _key_name(provider: str) -> str:
    normalised = str(provider or "").strip().lower().replace("-", "_")
    if normalised == "gemini":
        return "GEMINI_API_KEY"
    if normalised in {"openai", "openai_compatible", "custom_openai_compatible"}:
        return "OPENAI_API_KEY"
    if normalised == "agnes":
        return "AGNES_API_KEY"
    if normalised in {"local_qwen", "local_qwen_vl", "qwen_local", "qwen_vl_local", "qwen2_5_vl_local", "qwen25_vl_local", "local_vlm", "local_openai_compatible", "localhost_vlm"}:
        return ""
    if normalised in {"volcengine", "volcengine_coding_plan", "ark", "ark_coding_plan", "coding_plan"}:
        return "ARK_API_KEY"
    return "LECTURE_VISION_API_KEY"


def _provider_extra_env_lines(provider: str) -> list[str]:
    normalised = str(provider or "").strip().lower().replace("-", "_")
    if normalised in {"volcengine", "volcengine_coding_plan", "ark", "ark_coding_plan", "coding_plan"}:
        return [
            "LECTURE_VISION_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3",
            "# Also compatible with semantic-merge style LLM_API_KEY / LLM_BASE_URL / LLM_MODEL.",
        ]
    if normalised in {"local_qwen", "local_qwen_vl", "qwen_local", "qwen_vl_local", "qwen2_5_vl_local", "qwen25_vl_local"}:
        return [
            "LECTURE_VISION_BASE_URL=http://127.0.0.1:8000/v1",
            "LECTURE_VISION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct",
            "# Optional if your local OpenAI-compatible server requires auth:",
            "# LOCAL_QWEN_VL_API_KEY=<your local key>",
        ]
    if normalised in {"local_vlm", "local_openai_compatible", "localhost_vlm"}:
        return [
            "LECTURE_VISION_BASE_URL=http://127.0.0.1:8000/v1",
            "LECTURE_VISION_MODEL=local-vlm",
            "# Optional if your local OpenAI-compatible server requires auth:",
            "# LOCAL_VLM_API_KEY=<your local key>",
        ]
    return []


def _default_model(provider: str) -> str:
    normalised = str(provider or "").strip().lower().replace("-", "_")
    if normalised == "gemini":
        return GEMINI_DEFAULT_MODEL
    if normalised == "agnes":
        return "agnes-1.5-flash"
    if normalised in {"local_qwen", "local_qwen_vl", "qwen_local", "qwen_vl_local", "qwen2_5_vl_local", "qwen25_vl_local"}:
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    if normalised in {"local_vlm", "local_openai_compatible", "localhost_vlm"}:
        return "local-vlm"
    if normalised in {"volcengine", "volcengine_coding_plan", "ark", "ark_coding_plan", "coding_plan"}:
        return "ark-code-latest"
    return "gpt-4o-mini"
