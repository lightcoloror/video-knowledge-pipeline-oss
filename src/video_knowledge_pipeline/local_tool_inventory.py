from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from .asr_runner import detect_asr_runners
from .captiocr_resolver import resolve_captiocr_root
from .general_tagger_adapter import general_tagger_status
from .markdown_text import markdown_table_cell as _md_cell
from .media_tools import resolve_media_tool, resolve_tesseract
from .models import now_iso
from .path_defaults import source_reviews_root, tool_source_review_root, workspace_root
from .storage import write_json
from .tool_research import recommended_trial_order

LOCAL_TOOL_INVENTORY_SCHEMA = "lecture_local_tool_inventory.v1"
LOCAL_RUNTIME_PREFLIGHT_SCHEMA = "video_knowledge_pipeline.local_runtime_preflight.v1"

_HEAVY_MODEL_PROBE_SCRIPT = r"""
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path

module_names = ["torch", "transformers", "timm", "fairscale", "PIL"]
modules = {name: importlib.util.find_spec(name) is not None for name in module_names}
required = {
    "modeling_utils.PreTrainedModel": False,
    "pytorch_utils.apply_chunking_to_forward": False,
    "pytorch_utils.find_pruneable_heads_and_indices": False,
    "pytorch_utils.prune_linear_layer": False,
}
version = ""
error = ""
if modules["transformers"]:
    try:
        version = importlib.metadata.version("transformers")
        distribution = importlib.metadata.distribution("transformers")
        package_root = Path(distribution.locate_file("transformers"))
        modeling_source = (package_root / "modeling_utils.py").read_text(encoding="utf-8", errors="replace")
        pytorch_source = (package_root / "pytorch_utils.py").read_text(encoding="utf-8", errors="replace")
        required["modeling_utils.PreTrainedModel"] = "class PreTrainedModel" in modeling_source
        required["pytorch_utils.apply_chunking_to_forward"] = "def apply_chunking_to_forward" in pytorch_source
        required["pytorch_utils.find_pruneable_heads_and_indices"] = "def find_pruneable_heads_and_indices" in pytorch_source
        required["pytorch_utils.prune_linear_layer"] = "def prune_linear_layer" in pytorch_source
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
compatible = all(modules.values()) and all(required.values()) and not error
print(json.dumps({
    "ok": compatible,
    "status": "ready" if compatible else "incompatible",
    "python_version": platform.python_version(),
    "modules": modules,
    "transformers": {"version": version, "compatible": all(required.values()) and not error, "required_symbols": required},
    "error": error,
}))
""".strip()


def local_runtime_preflight(
    project_root: str | Path | None = None,
    *,
    source_inventory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect local runtimes without installs, media processing, model loading, or GPU setup."""
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    executable = Path(sys.executable).expanduser()
    resolved_executable = executable.resolve()
    prefix = Path(sys.prefix).expanduser().resolve()
    base_prefix = Path(sys.base_prefix).expanduser().resolve()
    pyvenv_cfg = prefix / "pyvenv.cfg"
    pyproject_path = root / "pyproject.toml"
    project = _read_pyproject(pyproject_path)
    dependency_rows = _core_dependency_rows(project)
    missing_dependencies = [row["import_name"] for row in dependency_rows if not row["available"]]
    ffmpeg = resolve_media_tool("ffmpeg")
    ffprobe = resolve_media_tool("ffprobe")
    tesseract = resolve_tesseract()
    uv = shutil.which("uv") or ""
    inventory_path = (
        Path(source_inventory_path).expanduser().resolve()
        if source_inventory_path is not None
        else (source_reviews_root() / "SOURCE_INVENTORY.json").resolve()
    )
    tagger = general_tagger_status(source_inventory_path=inventory_path)
    heavy_python = _resolve_heavy_model_python(root)
    heavy_runtime = _probe_heavy_model_runtime(Path(str(heavy_python.get("executable") or "")))
    heavy_modules_ready = all(bool(value) for value in (heavy_runtime.get("modules") or {}).values())
    transformers_compatible = bool((heavy_runtime.get("transformers") or {}).get("compatible"))
    checks = [
        _runtime_check("python:version", sys.version_info >= (3, 11), f"Python {'.'.join(str(value) for value in sys.version_info[:3])}"),
        _runtime_check("python:executable", resolved_executable.is_absolute() and resolved_executable.exists(), str(resolved_executable)),
        _runtime_check("project:pyproject", pyproject_path.is_file(), str(pyproject_path)),
        _runtime_check("dependencies:core", not missing_dependencies, ",".join(missing_dependencies) or "all core imports available"),
        _runtime_check("media:ffmpeg", bool(ffmpeg), ffmpeg or "FFMPEG_BINARY/LECTURE_FFMPEG_DIR not resolved"),
        _runtime_check("media:ffprobe", bool(ffprobe), ffprobe or "FFPROBE_BINARY/LECTURE_FFMPEG_DIR not resolved"),
        _runtime_check(
            "general_tagger:assets",
            tagger.get("status") == "ready",
            ",".join(str(item) for item in tagger.get("blockers") or []) or "RAM++ source, checkpoint, and tokenizer discovered",
        ),
        _runtime_check(
            "heavy_model:python",
            bool(heavy_python.get("exists")),
            str(heavy_python.get("source") or "heavy-model interpreter not found"),
        ),
        _runtime_check(
            "heavy_model:dependencies",
            heavy_modules_ready,
            ",".join(name for name, available in (heavy_runtime.get("modules") or {}).items() if not available)
            or "heavy-model imports available",
        ),
        _runtime_check(
            "heavy_model:transformers_compatibility",
            transformers_compatible,
            str((heavy_runtime.get("transformers") or {}).get("version") or heavy_runtime.get("error") or "not compatible"),
        ),
    ]
    failed_checks = [row["check_id"] for row in checks if row["status"] == "failed"]
    recovery_commands: list[dict[str, str]] = []
    if not resolved_executable.exists() or not pyvenv_cfg.exists():
        recovery_commands.append(
            {
                "key": "prepare_uv_venv",
                "command": "uv venv .venv --python 3.11",
                "reason": "Create the project-local virtual environment; this preflight never executes the command.",
            }
        )
    if missing_dependencies:
        recovery_commands.append(
            {
                "key": "sync_project_dependencies",
                "command": "uv sync --extra dev --python <venv-python>",
                "reason": "Install only after explicit operator approval; missing imports: " + ", ".join(missing_dependencies),
            }
        )
    if not ffmpeg or not ffprobe:
        recovery_commands.append(
            {
                "key": "configure_media_tools",
                "command": "Set FFMPEG_BINARY=<ffmpeg-path> and FFPROBE_BINARY=<ffprobe-path>, then rerun local-runtime-preflight.",
                "reason": "VKP resolves media binaries through media_tools; no machine path is embedded here.",
            }
        )
    if tagger.get("status") != "ready":
        recovery_commands.append(
            {
                "key": "configure_ram_plus_assets",
                "command": "Set VKP_LOCAL_MODEL_ROOT=<model-root> or repair the recognize-anything deployment_path in SOURCE_INVENTORY.json, then rerun local-runtime-preflight.",
                "reason": "RAM++ blockers: " + ", ".join(str(item) for item in tagger.get("blockers") or []),
            }
        )
    if not heavy_python.get("exists") or not heavy_modules_ready or not transformers_compatible:
        recovery_commands.append(
            {
                "key": "repair_heavy_model_runtime",
                "command": "Set VKP_HEAVY_MODEL_PYTHON=<compatible-python>, verify torch/transformers/timm/fairscale/Pillow, and apply the RAM++ transformers import-layout patch before rerunning local-runtime-preflight.",
                "reason": "The doctor reports compatibility only; it never installs into or mutates an existing heavy-model environment.",
            }
        )
    ok = not failed_checks
    return {
        "schema": LOCAL_RUNTIME_PREFLIGHT_SCHEMA,
        "checked_at": now_iso(),
        "ok": ok,
        "status": "ready" if ok else "not_ready",
        "project_root": str(root),
        "runtime": {
            "python": {
                "executable": str(resolved_executable),
                "exists": resolved_executable.exists(),
                "absolute": resolved_executable.is_absolute(),
                "version": ".".join(str(value) for value in sys.version_info[:3]),
                "required_python": str(project.get("requires-python") or ""),
            },
            "venv": {
                "active": prefix != base_prefix,
                "prefix": str(prefix),
                "base_prefix": str(base_prefix),
                "pyvenv_cfg": str(pyvenv_cfg),
                "pyvenv_cfg_exists": pyvenv_cfg.is_file(),
            },
            "uv": {"available": bool(uv), "path": str(Path(uv).resolve()) if uv else ""},
            "heavy_model_python": heavy_python,
        },
        "capabilities": {
            "media": {
                "ffmpeg": {"available": bool(ffmpeg), "path": ffmpeg},
                "ffprobe": {"available": bool(ffprobe), "path": ffprobe},
                "tesseract": {"available": bool(tesseract), "path": tesseract},
            },
            "dependencies": {
                "required": dependency_rows,
                "missing": missing_dependencies,
            },
            "general_tagger": tagger,
            "heavy_model_runtime": heavy_runtime,
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "recovery_commands": recovery_commands,
        "boundaries": {
            "installs_dependencies": False,
            "starts_service": False,
            "processes_media": False,
            "reads_configuration_secrets": False,
            "makes_network_request": False,
            "loads_model": False,
            "initializes_gpu": False,
            "starts_probe_process": bool(heavy_python.get("exists")),
        },
    }


def _resolve_heavy_model_python(project_root: Path) -> dict[str, Any]:
    configured = str(os.environ.get("VKP_HEAVY_MODEL_PYTHON") or "").strip()
    candidates: list[tuple[str, Path]] = []
    if configured:
        candidates.append(("VKP_HEAVY_MODEL_PYTHON", Path(configured).expanduser()))
    candidates.extend(
        [
            ("workspace_heavy_venv", workspace_root() / "tools" / "mineru-venv" / "Scripts" / "python.exe"),
            ("workspace_heavy_venv", workspace_root() / "tools" / "mineru-venv" / "bin" / "python"),
            ("project_venv", project_root / ".venv" / "Scripts" / "python.exe"),
            ("project_venv", project_root / ".venv" / "bin" / "python"),
            ("current_python", Path(sys.executable).expanduser()),
        ]
    )
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for source, candidate in candidates:
        key = str(candidate).casefold()
        if key in seen:
            continue
        seen.add(key)
        exists = candidate.is_file()
        checked.append({"source": source, "path": str(candidate), "exists": exists})
        if exists:
            return {
                "executable": str(candidate.resolve()),
                "exists": True,
                "source": source,
                "checked": checked,
            }
    return {"executable": "", "exists": False, "source": "not_found", "checked": checked}


def _probe_heavy_model_runtime(executable: Path) -> dict[str, Any]:
    empty = {
        "ok": False,
        "status": "interpreter_unavailable",
        "python_version": "",
        "modules": {name: False for name in ("torch", "transformers", "timm", "fairscale", "PIL")},
        "transformers": {"version": "", "compatible": False, "required_symbols": {}},
        "error": "heavy-model interpreter not found",
    }
    if not executable.is_file():
        return empty
    try:
        completed = subprocess.run(
            [str(executable), "-c", _HEAVY_MODEL_PROBE_SCRIPT],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {**empty, "status": "probe_failed", "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {
            **empty,
            "status": "probe_failed",
            "error": str(completed.stderr or completed.stdout or f"exit {completed.returncode}").strip(),
        }
    try:
        payload = json.loads(str(completed.stdout or "").strip())
    except json.JSONDecodeError as exc:
        return {**empty, "status": "probe_failed", "error": f"invalid probe JSON: {exc}"}
    return dict(payload) if isinstance(payload, dict) else empty


def local_tool_inventory(output_dir: str | Path | None = None, *, write: bool = False) -> dict[str, Any]:
    """Inspect local reusable tools for the lecture extraction stack.

    The inventory is deliberately local-only. It reuses the existing tool matrix,
    ASR detector, media-tool resolver, and known project mirror paths so agents
    can decide what to run before installing or developing anything new.
    """
    tool_rows = recommended_trial_order()
    by_name = {str(row.get("name") or "").lower(): row for row in tool_rows}
    asr = detect_asr_runners()
    media = _media_tools()
    runtime_preflight = local_runtime_preflight()
    tools = [
        _local_project_tool(
            "BiliNote",
            "product_ui",
            "产品壳 / UI",
            ["LECTURE_BILINOTE_ROOT"],
            [
                tool_source_review_root() / "BiliNote",
                workspace_root() / "BiliNote",
            ],
            evidence_files=["BillNote_frontend/package.json", "backend/app"],
            reuse_role="直接作为 Web/Tauri UI、任务系统和人工复核入口。",
        ),
        _matrix_tool(by_name, "vidclaude", "visual_timeline", "视觉/时间线提取"),
        _matrix_tool(by_name, "peepshow", "fast_frame_ocr", "快速帧/OCR/报告"),
        _matrix_tool(by_name, "vidwise", "fallback_extractor", "轻量素材抽取 fallback"),
        _local_project_tool(
            "content-core",
            "architecture_reference",
            "CLI/MCP 架构参考",
            ["LECTURE_CONTENT_CORE_ROOT"],
            [
                tool_source_review_root() / "content-core",
                workspace_root() / "content-core",
                Path.home() / "GitHub" / "content-core",
            ],
            evidence_files=["pyproject.toml", "package.json", "src"],
            reuse_role="只借 CLI/MCP 架构经验；不把音频摘要逻辑当视频全量抽取能力。",
        ),
        _matrix_tool(by_name, "FunClip", "asr_timestamp_reference", "ASR/时间戳参考"),
        _captiocr_tool(),
    ]
    runnable_asr = [item for item in asr.get("tools") or [] if isinstance(item, dict) and item.get("runnable")]
    summary = {
        "schema": LOCAL_TOOL_INVENTORY_SCHEMA,
        "checked_at": now_iso(),
        "ready_for_ui": _available(tools, "BiliNote"),
        "ready_for_visual_extraction": any(_available(tools, name) for name in ("vidclaude", "peepshow", "vidwise")),
        "ready_for_asr": bool(runnable_asr),
        "ready_for_media": bool(media.get("ffmpeg") and media.get("ffprobe")),
        "ready_for_ocr": bool(media.get("tesseract") or _available(tools, "CaptiOCR")),
        "available_tools": [tool["name"] for tool in tools if tool.get("available")],
        "missing_tools": [tool["name"] for tool in tools if not tool.get("available")],
        "recommended_route": _recommended_route(tools, asr, media),
    }
    result = {
        "schema": LOCAL_TOOL_INVENTORY_SCHEMA,
        "checked_at": summary["checked_at"],
        "summary": summary,
        "tools": tools,
        "asr": asr,
        "media": media,
        "runtime_preflight": runtime_preflight,
        "next_action": _next_action(summary, tools, asr, media),
    }
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "local-tool-inventory.json"
        markdown_path = out_dir / "local-tool-inventory.md"
        args_path = out_dir / "mcp-local-tool-inventory.args.json"
        result["output_json"] = str(json_path)
        result["output_markdown"] = str(markdown_path)
        result["mcp_args_path"] = str(args_path)
        if write:
            write_json(json_path, result)
            markdown_path.write_text(render_local_tool_inventory_markdown(result), encoding="utf-8")
            write_json(args_path, {"output_dir": str(out_dir), "write": True})
    return result


def render_local_tool_inventory_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    next_action = result.get("next_action") if isinstance(result.get("next_action"), dict) else {}
    lines = [
        "# Local Lecture Tool Inventory",
        "",
        f"- Schema: `{result.get('schema', '')}`",
        f"- Checked: `{result.get('checked_at', '')}`",
        f"- UI ready: `{summary.get('ready_for_ui', False)}`",
        f"- Visual extraction ready: `{summary.get('ready_for_visual_extraction', False)}`",
        f"- ASR ready: `{summary.get('ready_for_asr', False)}`",
        f"- OCR ready: `{summary.get('ready_for_ocr', False)}`",
        f"- Next: `{next_action.get('key', '')}` / {next_action.get('label', '')}",
        "",
        "## Recommended Route",
        "",
    ]
    route = summary.get("recommended_route") if isinstance(summary.get("recommended_route"), list) else []
    for item in route:
        if not isinstance(item, dict):
            continue
        status = "ready" if item.get("ready") else "missing"
        lines.append(f"- `{status}` **{item.get('layer', '')}**: {item.get('tool', '')} - {item.get('use', '')}")
    lines.extend(
        [
            "",
            "## Tools",
            "",
            "| Tool | Layer | Available | Role | Evidence | Install / Configure |",
            "|---|---|---|---|---|---|",
        ]
    )
    for tool in result.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        evidence = ", ".join(str(item) for item in tool.get("evidence") or []) or "-"
        install = str(tool.get("install_hint") or tool.get("configure_hint") or "")
        lines.append(
            "| {name} | {layer} | `{available}` | {role} | {evidence} | `{install}` |".format(
                name=_md_cell(str(tool.get("name") or "")),
                layer=_md_cell(str(tool.get("layer") or "")),
                available=bool(tool.get("available")),
                role=_md_cell(str(tool.get("reuse_role") or "")),
                evidence=_md_cell(evidence),
                install=_md_cell(install),
            )
        )
    lines.extend(["", "## ASR", "", "| Preset | Runnable | Command | Notes |", "|---|---|---|---|"])
    asr = result.get("asr") if isinstance(result.get("asr"), dict) else {}
    for runner in asr.get("tools") or []:
        if not isinstance(runner, dict):
            continue
        lines.append(
            "| {name} | `{runnable}` | `{command}` | {notes} |".format(
                name=_md_cell(str(runner.get("name") or "")),
                runnable=bool(runner.get("runnable")),
                command=_md_cell(str(runner.get("command_path") or runner.get("command") or "")),
                notes=_md_cell(str(runner.get("notes") or "")),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _matrix_tool(by_name: dict[str, dict[str, Any]], name: str, kind: str, layer: str) -> dict[str, Any]:
    row = by_name.get(name.lower()) or {}
    return {
        "name": name,
        "kind": kind,
        "layer": layer,
        "available": bool(row.get("installed")),
        "evidence": list(row.get("installed_paths") or []),
        "reuse_role": str(row.get("reuse_role") or ""),
        "best_for": str(row.get("best_for") or ""),
        "install_hint": str(row.get("install_hint") or ""),
        "url": str(row.get("url") or ""),
        "source": "tool_matrix",
    }


def _local_project_tool(
    name: str,
    kind: str,
    layer: str,
    env_names: list[str],
    candidates: list[Path],
    *,
    evidence_files: list[str],
    reuse_role: str,
) -> dict[str, Any]:
    paths: list[Path] = []
    for env_name in env_names:
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            paths.append(Path(env_value).expanduser())
    paths.extend(candidates)
    evidence: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        matched = [relative for relative in evidence_files if (path / relative).exists()]
        if matched:
            evidence.append(str(path.resolve()))
            evidence.extend(str((path / relative).resolve()) for relative in matched[:3])
            break
    return {
        "name": name,
        "kind": kind,
        "layer": layer,
        "available": bool(evidence),
        "evidence": evidence,
        "reuse_role": reuse_role,
        "configure_hint": f"Set {env_names[0]} to the local {name} root." if env_names else "",
        "source": "local_project_probe",
    }


def _captiocr_tool() -> dict[str, Any]:
    resolved = resolve_captiocr_root()
    evidence = list(resolved.get("evidence") or [])
    if resolved.get("root"):
        evidence.insert(0, str(resolved.get("root")))
    command = shutil.which("captiocr") or ""
    if command:
        evidence.append(command)
    return {
        "name": "CaptiOCR",
        "kind": "ocr_widget",
        "layer": "OCR 小组件 / 人工校对",
        "available": bool(evidence or importlib.util.find_spec("PIL")),
        "evidence": evidence,
        "reuse_role": "复用 Tk 截屏 OCR 和人工校对思路；作为 OCR backfill 的局部工具。",
        "configure_hint": str(resolved.get("configure_hint") or ""),
        "command_hint": str(resolved.get("command_hint") or command or ""),
        "checked_paths": list(resolved.get("checked") or []),
        "source": "local_project_probe",
    }


def _media_tools() -> dict[str, str]:
    return {
        "ffmpeg": resolve_media_tool("ffmpeg"),
        "ffprobe": resolve_media_tool("ffprobe"),
        "tesseract": resolve_tesseract(),
    }


def _recommended_route(tools: list[dict[str, Any]], asr: dict[str, Any], media: dict[str, str]) -> list[dict[str, Any]]:
    runnable_asr = next((item for item in asr.get("tools") or [] if isinstance(item, dict) and item.get("runnable")), {})
    visual = _first_available(tools, ["vidclaude", "peepshow", "vidwise"])
    return [
        {"layer": "产品壳 / UI", "tool": "BiliNote", "ready": _available(tools, "BiliNote"), "use": "课程任务、人工复核、WebUI。"},
        {
            "layer": "视觉/时间线提取",
            "tool": visual.get("name") or "vidclaude / peepshow / vidwise",
            "ready": bool(visual),
            "use": "把视频转成时间线、OCR、关键帧、原始证据。",
        },
        {
            "layer": "中文 ASR",
            "tool": runnable_asr.get("name") or "FunASR / SenseVoice / WhisperX",
            "ready": bool(runnable_asr),
            "use": "生成带时间戳的中文转写。",
        },
        {
            "layer": "媒体/OCR 基础设施",
            "tool": "ffmpeg + ffprobe + tesseract",
            "ready": bool(media.get("ffmpeg") and media.get("ffprobe")),
            "use": "抽帧、探测视频、OCR 回填。",
        },
        {"layer": "调度层", "tool": "lecture-extract", "ready": True, "use": "统一数据模型、CLI/MCP、Obsidian 导出。"},
    ]


def _next_action(summary: dict[str, Any], tools: list[dict[str, Any]], asr: dict[str, Any], media: dict[str, str]) -> dict[str, Any]:
    if not summary.get("ready_for_media"):
        return {
            "key": "configure_media_tools",
            "label": "配置 ffmpeg/ffprobe",
            "hint": "先让抽帧和视频探测可用；设置 FFMPEG_BINARY/FFPROBE_BINARY 或 LECTURE_FFMPEG_DIR。",
        }
    if not summary.get("ready_for_visual_extraction"):
        return {
            "key": "prepare_visual_extractor",
            "label": "准备 vidclaude/peepshow/vidwise",
            "hint": "优先复用本地已验证的 vidclaude 或 peepshow；没有就安装/拉取其中一个。",
        }
    if not summary.get("ready_for_asr"):
        return {
            "key": "prepare_asr",
            "label": "准备中文 ASR",
            "hint": "先运行 asr-env-status 写出 ASR 环境交接；优先 FunASR/SenseVoice，需要英文或对齐能力时再用 WhisperX/faster-whisper。",
        }
    if not _available(tools, "BiliNote"):
        return {
            "key": "configure_bilinote_root",
            "label": "配置 BiliNote UI",
            "hint": "设置 LECTURE_BILINOTE_ROOT 或把本地 BiliNote 镜像放到默认路径。",
        }
    return {
        "key": "ready_to_run_pipeline",
        "label": "可以运行课程抽取流水线",
        "hint": "用 detect-extractor-output / run-detected-lecture-pipeline 或 BiliNote 任务入口开始处理视频。",
    }


def _first_available(tools: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    for name in names:
        for tool in tools:
            if str(tool.get("name") or "").lower() == name.lower() and tool.get("available"):
                return tool
    return {}


def _available(tools: list[dict[str, Any]], name: str) -> bool:
    return any(str(tool.get("name") or "").lower() == name.lower() and tool.get("available") for tool in tools)


def _runtime_check(check_id: str, ok: bool, detail: str) -> dict[str, str]:
    return {"check_id": check_id, "status": "passed" if ok else "failed", "detail": detail}


def _read_pyproject(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    project = payload.get("project") if isinstance(payload, dict) else {}
    return dict(project) if isinstance(project, dict) else {}


def _core_dependency_rows(project: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for requirement in project.get("dependencies") or []:
        package = re.split(r"[<>=!~;\s\[]", str(requirement), maxsplit=1)[0].strip()
        if not package:
            continue
        import_name = {
            "markdown-it-py": "markdown_it",
        }.get(package.lower(), package.replace("-", "_"))
        rows.append(
            {
                "requirement": str(requirement),
                "package": package,
                "import_name": import_name,
                "available": importlib.util.find_spec(import_name) is not None,
            }
        )
    return rows
