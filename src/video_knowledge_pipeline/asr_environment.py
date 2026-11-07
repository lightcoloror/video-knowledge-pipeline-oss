from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .asr_runner import (
    _command_runtime_probe,
    _model_ready,
    _resolve_command_path,
    default_local_asr_device,
)
from .cuda_runtime import cuda_dll_discovery_status
from .markdown_text import markdown_table_cell as _md_cell
from .media_tools import resolve_media_tool
from .models import now_iso
from .storage import write_json

ASR_ENV_SCHEMA = "lecture_asr_environment.v1"
ASR_TOOLS = (
    {
        "name": "funasr",
        "role": "primary Chinese ASR and SenseVoice runner",
        "module": "funasr",
        "command_name": "funasr",
        "env_command": "LECTURE_FUNASR_COMMAND",
        "install_switch": "-InstallFunASR",
    },
    {
        "name": "sensevoice",
        "role": "SenseVoice through FunASR CLI",
        "module": "funasr",
        "command_name": "funasr",
        "env_command": "LECTURE_FUNASR_COMMAND",
        "install_switch": "-InstallFunASR",
    },
    {
        "name": "qwen3-asr",
        "role": "independent quality ASR and forced alignment",
        "module": "qwen_asr",
        "command_name": "qwen-asr",
        "env_command": "LECTURE_QWEN3_ASR_COMMAND",
        "install_switch": "",
        "pip_package": "qwen-asr",
    },
    {
        "name": "whisperx",
        "role": "alignment-capable multilingual fallback",
        "module": "whisperx",
        "command_name": "whisperx",
        "env_command": "LECTURE_WHISPERX_COMMAND",
        "install_switch": "-InstallWhisperX",
    },
    {
        "name": "faster-whisper",
        "role": "lighter local Whisper fallback",
        "module": "faster_whisper",
        "command_name": "faster-whisper",
        "env_command": "LECTURE_FASTER_WHISPER_COMMAND",
        "install_switch": "-InstallFasterWhisper",
    },
    {
        "name": "moss-transcribe-diarize",
        "role": "explicit long-form multi-speaker ASR challenger",
        "module": "moss_transcribe_diarize",
        "command_name": "mtd-subtitle",
        "env_command": "LECTURE_MOSS_TRANSCRIBE_COMMAND",
        "install_switch": "",
        "install_command": "See docs/moss-transcribe-diarize-adapter-2026-07-27.md",
    },
)


def asr_environment_status(
    venv_dir: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    write: bool = False,
    python_version: str = "3.11",
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    env_dir = Path(venv_dir).expanduser() if venv_dir else _default_env_dir(root)
    env_dir = env_dir.resolve()
    scripts_dir = env_dir / "Scripts"
    python_path = _managed_python(env_dir)
    venv_exists = python_path.exists()
    python_info = _python_info(python_path) if venv_exists else {
        "executable": str(python_path),
        "version": "",
        "major": None,
        "minor": None,
        "asr_recommended": False,
        "warning": "environment not created",
    }
    tools = [_tool_status(tool, scripts_dir=scripts_dir, python_path=python_path, venv_exists=venv_exists) for tool in ASR_TOOLS]
    runtime = _runtime_info(python_path, venv_exists=venv_exists)
    default_device = default_local_asr_device(str(python_path)) if venv_exists else "unknown"
    cuda_dll = cuda_dll_discovery_status()
    model_caches = _model_cache_status()
    primary_model = _model_ready(preset="sensevoice", model="iic/SenseVoiceSmall")
    quality_models = {
        "qwen3_asr_1_7b": _model_ready(preset="qwen3-asr-1.7b", model="Qwen/Qwen3-ASR-1.7B"),
        "qwen3_asr_0_6b": _model_ready(preset="qwen3-asr-0.6b", model="Qwen/Qwen3-ASR-0.6B"),
        "qwen3_forced_aligner": _model_ready(preset="qwen3-forced-aligner", model="Qwen/Qwen3-ForcedAligner-0.6B"),
        "moss_transcribe_diarize": _model_ready(
            preset="moss-transcribe-diarize",
            model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        ),
    }
    model_download_allowed = _allow_model_download()
    ffmpeg_path = resolve_media_tool("ffmpeg")
    available_tools = [
        tool["name"]
        for tool in tools
        if tool["module"]["available"] or tool.get("runtime_ready", tool["command_exists"])
    ]
    command_tools = [
        tool["name"]
        for tool in tools
        if tool.get("runtime_ready", tool["command_exists"])
    ]
    runtime_blocked_tools = [
        tool["name"]
        for tool in tools
        if tool["command_exists"] and not tool.get("runtime_ready", tool["command_exists"])
    ]
    readiness = _readiness_checklist(
        venv_exists=venv_exists,
        python_info=python_info,
        available_tools=available_tools,
        command_tools=command_tools,
        ffmpeg_available=bool(ffmpeg_path),
        runtime=runtime,
        primary_model=primary_model,
        model_download_allowed=model_download_allowed,
    )
    env_snippet = _env_snippet(scripts_dir)
    install_script = ".\\scripts\\install-local-asr-env.ps1"
    result = {
        "schema": ASR_ENV_SCHEMA,
        "checked_at": now_iso(),
        "ok": bool(venv_exists and command_tools),
        "module_ready": bool(venv_exists and available_tools),
        "command_ready": bool(command_tools),
        "venv_exists": bool(venv_exists),
        "venv_dir": str(env_dir),
        "venv_python": str(python_path),
        "venv_bin": str(scripts_dir),
        "python": python_info,
        "runtime": runtime,
        "default_local_asr_device": default_device,
        "cuda_dll": cuda_dll,
        "ffmpeg": {"path": ffmpeg_path, "available": bool(ffmpeg_path)},
        "model_caches": model_caches,
        "primary_model": primary_model,
        "quality_models": quality_models,
        "model_download_allowed": model_download_allowed,
        "readiness": readiness,
        "tools": tools,
        "available_tools": available_tools,
        "command_tools": command_tools,
        "runtime_blocked_tools": runtime_blocked_tools,
        "recommended_order": [
            "sensevoice",
            "moss-transcribe-diarize",
            "qwen3-asr",
            "whisperx",
            "faster-whisper",
        ],
        "create_command": f'{install_script} -VenvDir "{env_dir}" -CreateVenv',
        "create_conda_command": f'{install_script} -VenvDir "{env_dir}" -CreateCondaEnv -PythonVersion {python_version}',
        "install_funasr_command": f'{install_script} -VenvDir "{env_dir}" -InstallFunASR',
        "install_cuda_torch_command": f'{install_script} -VenvDir "{env_dir}" -InstallCudaTorch',
        "install_whisperx_command": f'{install_script} -VenvDir "{env_dir}" -InstallWhisperX',
        "install_faster_whisper_command": f'{install_script} -VenvDir "{env_dir}" -InstallFasterWhisper',
        "smoke_command": ".\\scripts\\video-knowledge.ps1 asr-smoke <short-audio-or-video.mp4>",
        "install_report": _install_report(env_dir=env_dir, python_version=python_version, model_download_allowed=model_download_allowed),
        "privacy": "Local ASR runs on this machine. Audio is not uploaded by this pipeline; first-run model download is controlled by LECTURE_ASR_ALLOW_MODEL_DOWNLOAD.",
        "env_snippet": env_snippet,
        "next_action": _next_action(
            venv_exists=venv_exists,
            python_info=python_info,
            available_tools=available_tools,
            command_tools=command_tools,
            ffmpeg_available=bool(ffmpeg_path),
            primary_model=primary_model,
            model_download_allowed=model_download_allowed,
        ),
    }
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "asr-environment.json"
        markdown_path = out_dir / "asr-environment.md"
        env_path = out_dir / "asr-env.ps1"
        args_path = out_dir / "mcp-asr-environment-status.args.json"
        result["output_json"] = str(json_path)
        result["output_markdown"] = str(markdown_path)
        result["env_script_path"] = str(env_path)
        result["mcp_args_path"] = str(args_path)
        if write:
            write_json(json_path, result)
            markdown_path.write_text(render_asr_environment_markdown(result), encoding="utf-8")
            env_path.write_text("\n".join(env_snippet).rstrip() + "\n", encoding="utf-8")
            write_json(args_path, {"venv_dir": str(env_dir), "output_dir": str(out_dir), "write": True, "python_version": python_version})
    return result


def _default_env_dir(root: Path) -> Path:
    conda_dir = root / ".conda-lecture-asr"
    venv_dir = root / ".venv-lecture-asr"
    if _managed_python(conda_dir).exists():
        return conda_dir
    if _managed_python(venv_dir).exists():
        return venv_dir
    return conda_dir


def render_asr_environment_markdown(result: dict[str, Any]) -> str:
    next_action = result.get("next_action") if isinstance(result.get("next_action"), dict) else {}
    lines = [
        "# ASR Environment Status",
        "",
        f"- Schema: `{result.get('schema', '')}`",
        f"- Checked: `{result.get('checked_at', '')}`",
        f"- Env: `{result.get('venv_dir', '')}`",
        f"- Exists: `{result.get('venv_exists', False)}`",
        f"- Model caches: `{_format_model_caches(result.get('model_caches'))}`",
        f"- Primary model: `{(result.get('primary_model') or {}).get('model', '')}` / `{(result.get('primary_model') or {}).get('status', '')}`",
        f"- Quality models: `{_format_quality_models(result.get('quality_models'))}`",
        f"- Model download allowed: `{result.get('model_download_allowed', False)}`",
        f"- Module ready: `{result.get('module_ready', False)}`",
        f"- Command ready: `{result.get('command_ready', False)}`",
        f"- Runtime blocked tools: `{', '.join(result.get('runtime_blocked_tools') or [])}`",
        f"- FFmpeg: `{(result.get('ffmpeg') or {}).get('available', False)}`",
        f"- Default local ASR device: `{result.get('default_local_asr_device', '')}`",
        f"- CUDA DLL dirs discovered: `{(result.get('cuda_dll') or {}).get('discovered_count', 0)}`",
        f"- Next: `{next_action.get('key', '')}` / {next_action.get('label', '')}",
        f"- Privacy: {result.get('privacy', '')}",
        "",
        "## Readiness Checklist",
        "",
        "| Check | OK | Message |",
        "|---|---:|---|",
    ]
    for item in result.get("readiness") or []:
        if not isinstance(item, dict):
            continue
        lines.append(f"| `{_md_cell(str(item.get('key') or ''))}` | `{bool(item.get('ok'))}` | {_md_cell(str(item.get('message') or ''))} |")
    report = result.get("install_report") if isinstance(result.get("install_report"), dict) else {}
    lines.extend(
        [
            "",
            "## Install Report",
            "",
            f"- Install command: `{report.get('install_command', '')}`",
            f"- Model download command: `{report.get('model_download_command', '')}`",
            f"- Cache path: `{report.get('cache_path', '')}`",
            f"- Expected disk usage: `{report.get('expected_disk_usage', '')}`",
            "",
        "## Commands",
        "",
        f"- Create venv: `{result.get('create_command', '')}`",
        f"- Create conda env: `{result.get('create_conda_command', '')}`",
        f"- Install FunASR/SenseVoice: `{result.get('install_funasr_command', '')}`",
        f"- Install WhisperX: `{result.get('install_whisperx_command', '')}`",
        f"- Install faster-whisper: `{result.get('install_faster_whisper_command', '')}`",
        "",
        "## Env Snippet",
        "",
        "```powershell",
        *[str(line) for line in result.get("env_snippet") or []],
        "```",
        "",
        f"- Smoke test: `{result.get('smoke_command', '')}`",
        "",
        "## Tools",
        "",
        "| Tool | Module | Command | Runtime | Blocker | Role | Install |",
        "|---|---|---|---|---|---|---|",
    ]
    )
    for tool in result.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        module = tool.get("module") if isinstance(tool.get("module"), dict) else {}
        runtime_probe = (
            tool.get("runtime_probe")
            if isinstance(tool.get("runtime_probe"), dict)
            else {}
        )
        lines.append(
            "| {name} | `{module}` | `{command}` | `{runtime}` | `{blocker}` | {role} | `{install}` |".format(
                name=_md_cell(str(tool.get("name") or "")),
                module=bool(module.get("available")),
                command=bool(tool.get("command_exists")),
                runtime=bool(tool.get("runtime_ready", tool.get("command_exists"))),
                blocker=_md_cell(str(runtime_probe.get("blocker") or "")),
                role=_md_cell(str(tool.get("role") or "")),
                install=_md_cell(str(tool.get("install_command") or "")),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _managed_python(env_dir: Path) -> Path:
    venv_python = env_dir / "Scripts" / "python.exe"
    conda_python = env_dir / "python.exe"
    if venv_python.exists():
        return venv_python
    if conda_python.exists():
        return conda_python
    return venv_python


def _python_info(python_path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(python_path), "-c", "import json, sys; print(json.dumps({'version': sys.version.split()[0], 'major': sys.version_info.major, 'minor': sys.version_info.minor}))"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
            check=False,
        )
        payload = json.loads(completed.stdout.strip()) if completed.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        payload = {}
    major = payload.get("major")
    minor = payload.get("minor")
    recommended = bool(major == 3 and isinstance(minor, int) and 10 <= minor <= 12)
    return {
        "executable": str(python_path),
        "version": str(payload.get("version") or ""),
        "major": major,
        "minor": minor,
        "asr_recommended": recommended,
        "warning": "" if recommended else f"FunASR/WhisperX dependencies are more reliable on Python 3.10-3.12; detected {payload.get('version') or 'unknown'}.",
    }


def _tool_status(tool: dict[str, str], *, scripts_dir: Path, python_path: Path, venv_exists: bool) -> dict[str, Any]:
    env_command = str(tool["env_command"])
    configured = os.environ.get(env_command, "").strip()
    resolved_command = _resolve_command_path(
        {
            "command": str(tool["command_name"]),
            "env_command": env_command,
        }
    )
    command = Path(resolved_command).expanduser() if resolved_command else (
        scripts_dir / f"{tool['command_name']}.exe"
    )
    command_exists = command.exists()
    module = _module_status(
        str(tool["module"]),
        python_path=python_path,
        venv_exists=venv_exists,
    )
    runtime_probe = _command_runtime_probe(
        preset=str(tool["name"]),
        command_path=str(command.resolve()) if command_exists else "",
    )
    install_script = r".\scripts\install-local-asr-env.ps1"
    pip_package = str(tool.get("pip_package") or "").strip()
    explicit_install = str(tool.get("install_command") or "").strip()
    install_command = explicit_install or (
        f'"{python_path}" -m pip install {pip_package}'
        if pip_package
        else f'{install_script} -VenvDir "{scripts_dir.parent}" {tool["install_switch"]}'
    )
    return {
        "name": tool["name"],
        "role": tool["role"],
        "module": module,
        "command": str(command),
        "command_exists": command_exists,
        "runtime_ready": bool(command_exists and runtime_probe.get("ready")),
        "runtime_probe": runtime_probe,
        "env_command": env_command,
        "configured_by_environment": bool(configured),
        "install_command": install_command,
    }


def _module_status(module: str, *, python_path: Path, venv_exists: bool) -> dict[str, Any]:
    if not venv_exists:
        return {"module": module, "available": False, "returncode": None}
    try:
        completed = subprocess.run(
            [str(python_path), "-c", f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module!r}) else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        return {"module": module, "available": completed.returncode == 0, "returncode": completed.returncode}
    except (OSError, subprocess.TimeoutExpired):
        return {"module": module, "available": False, "returncode": None}


def _runtime_info(python_path: Path, *, venv_exists: bool) -> dict[str, Any]:
    if not venv_exists:
        return {"device": "unknown", "cuda_available": False, "torch_available": False}
    code = (
        "import json\n"
        "payload={'torch_available': False, 'cuda_available': False, 'device': 'cpu'}\n"
        "try:\n"
        " import torch\n"
        " payload['torch_available']=True\n"
        " payload['cuda_available']=bool(torch.cuda.is_available())\n"
        " payload['device']='cuda' if payload['cuda_available'] else 'cpu'\n"
        "except Exception:\n"
        " pass\n"
        "print(json.dumps(payload))\n"
    )
    try:
        completed = subprocess.run(
            [str(python_path), "-c", code],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            payload = json.loads(completed.stdout.strip())
            if isinstance(payload, dict):
                return payload
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return {"device": "unknown", "cuda_available": False, "torch_available": False}


def _model_cache_status() -> list[dict[str, Any]]:
    paths = []
    if os.environ.get("MODELSCOPE_CACHE"):
        paths.append(("MODELSCOPE_CACHE", Path(os.environ["MODELSCOPE_CACHE"]).expanduser()))
    paths.extend(
        [
            ("modelscope", Path.home() / ".cache" / "modelscope"),
            ("modelscope_hub_models", Path.home() / ".cache" / "modelscope" / "hub" / "models"),
            ("huggingface_hub", Path.home() / ".cache" / "huggingface" / "hub"),
        ]
    )
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for name, path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append({"name": name, "path": resolved, "exists": path.exists()})
    return result


def _format_quality_models(value: Any) -> str:
    if not isinstance(value, dict):
        return "unknown"
    return ", ".join(f"{key}={str((row or {}).get('status') or 'unknown')}" for key, row in value.items())


def _format_model_caches(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    existing = [str(item.get("path")) for item in value if isinstance(item, dict) and item.get("exists")]
    return "; ".join(existing)


def _env_snippet(scripts_dir: Path) -> list[str]:
    return [
        f'$env:LECTURE_ASR_BIN_DIR="{scripts_dir}"',
        f'$env:LECTURE_FUNASR_COMMAND="{scripts_dir / "funasr.exe"}"',
        f'$env:LECTURE_WHISPERX_COMMAND="{scripts_dir / "whisperx.exe"}"',
        f'$env:LECTURE_FASTER_WHISPER_COMMAND="{scripts_dir / "faster-whisper.exe"}"',
    ]


def _readiness_checklist(
    *,
    venv_exists: bool,
    python_info: dict[str, Any],
    available_tools: list[str],
    command_tools: list[str],
    ffmpeg_available: bool,
    runtime: dict[str, Any],
    primary_model: dict[str, Any],
    model_download_allowed: bool,
) -> list[dict[str, Any]]:
    module_ready = bool(venv_exists and available_tools)
    command_ready = bool(command_tools)
    primary_model_ready = bool(primary_model.get("ready"))
    return [
        {
            "key": "python_environment_ready",
            "ok": bool(venv_exists and python_info.get("asr_recommended")),
            "message": "Python 3.10-3.12 ASR environment exists." if venv_exists else "Create a local ASR environment first.",
        },
        {
            "key": "python_package_missing",
            "ok": module_ready,
            "message": "At least one local ASR package is importable." if module_ready else "Install FunASR/SenseVoice first for Chinese lecture videos.",
        },
        {
            "key": "asr_command_available",
            "ok": command_ready,
            "message": "ASR command entrypoint exists." if command_ready else "Command wrapper is missing; module runner may still work, but install/repair is recommended.",
        },
        {
            "key": "ffmpeg_missing",
            "ok": bool(ffmpeg_available),
            "message": "ffmpeg is available for audio extraction and short smoke clips." if ffmpeg_available else "Install or expose ffmpeg before running ASR smoke tests.",
        },
        {
            "key": "model_cache_missing",
            "ok": primary_model_ready or model_download_allowed,
            "message": "Primary SenseVoice model cache exists or first-run download is allowed."
            if primary_model_ready or model_download_allowed
            else "Primary SenseVoice model cache was not found and model download is disabled.",
        },
        {
            "key": "model_download_disabled",
            "ok": primary_model_ready or model_download_allowed,
            "message": "Model download is allowed or the model is already cached."
            if primary_model_ready or model_download_allowed
            else "Set LECTURE_ASR_ALLOW_MODEL_DOWNLOAD=1 only if you want the first smoke run to download the model.",
        },
        {
            "key": "cpu_ready",
            "ok": bool(module_ready and ffmpeg_available and runtime.get("device") == "cpu"),
            "message": "CPU ASR path is available." if module_ready and ffmpeg_available and runtime.get("device") == "cpu" else "CPU fallback is not the active path; CUDA may be active, or ASR package/ffmpeg is missing.",
        },
        {
            "key": "cuda_ready",
            "ok": bool(module_ready and ffmpeg_available and runtime.get("cuda_available")),
            "message": "CUDA ASR path is available." if runtime.get("cuda_available") else "CUDA is not available; CPU mode can still be used.",
        },
    ]


def _install_report(*, env_dir: Path, python_version: str, model_download_allowed: bool) -> dict[str, str]:
    cache_path = os.environ.get("MODELSCOPE_CACHE") or str(Path.home() / ".cache" / "modelscope")
    return {
        "install_command": f'.\\scripts\\install-local-asr-env.ps1 -VenvDir "{env_dir}" -CreateCondaEnv -PythonVersion {python_version} -InstallFunASR',
        "install_cuda_torch_command": f'.\\scripts\\install-local-asr-env.ps1 -VenvDir "{env_dir}" -InstallCudaTorch',
        "model_download_command": '$env:LECTURE_ASR_ALLOW_MODEL_DOWNLOAD="1"; .\\scripts\\video-knowledge.ps1 asr-smoke <short-audio-or-video.mp4> --execute',
        "cache_path": cache_path,
        "expected_disk_usage": "SenseVoiceSmall roughly 1-2 GB; Whisper large models can require several GB.",
        "model_download_allowed": str(bool(model_download_allowed)).lower(),
    }


def _allow_model_download() -> bool:
    return str(os.environ.get("LECTURE_ASR_ALLOW_MODEL_DOWNLOAD", "")).strip().lower() in {"1", "true", "yes", "on"}


def _next_action(
    *,
    venv_exists: bool,
    python_info: dict[str, Any],
    available_tools: list[str],
    command_tools: list[str],
    ffmpeg_available: bool,
    primary_model: dict[str, Any],
    model_download_allowed: bool,
) -> dict[str, str]:
    if not venv_exists:
        return {
            "key": "create_asr_environment",
            "label": "创建 ASR 环境",
            "hint": "优先用 Python 3.11 conda env，再安装 FunASR/SenseVoice 或 WhisperX。",
        }
    if not ffmpeg_available:
        return {
            "key": "install_ffmpeg",
            "label": "安装或暴露 ffmpeg",
            "hint": "ASR smoke 和视频音频抽取需要 ffmpeg。",
        }
    if not python_info.get("asr_recommended"):
        return {
            "key": "recreate_asr_environment",
            "label": "重建 Python 3.10-3.12 ASR 环境",
            "hint": str(python_info.get("warning") or ""),
        }
    if not available_tools:
        return {
            "key": "install_funasr",
            "label": "安装 FunASR/SenseVoice",
            "hint": "这是中文知识类讲解视频的优先 ASR 路线。",
        }
    if not command_tools:
        return {
            "key": "repair_asr_command",
            "label": "修复 ASR CLI 命令",
            "hint": "模块已存在但命令不存在；重新安装对应包，或设置 LECTURE_*_COMMAND 指向可执行脚本。",
        }
    if not primary_model.get("ready") and not model_download_allowed:
        return {
            "key": "prepare_asr_model_cache",
            "label": "准备 SenseVoice 模型缓存",
            "hint": "当前未找到 iic/SenseVoiceSmall 缓存；设置 LECTURE_ASR_ALLOW_MODEL_DOWNLOAD=1 后运行 asr-smoke，可触发首次下载。",
        }
    return {
        "key": "run_asr_smoke",
        "label": "运行本地 ASR 短片段 smoke",
        "hint": "使用 asr-smoke 对短片段验证转写链路；默认只在本机执行，不上传音频。",
    }
