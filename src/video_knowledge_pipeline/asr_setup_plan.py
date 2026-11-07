from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_environment import asr_environment_status
from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .storage import write_json

ASR_SETUP_PLAN_SCHEMA = "lecture_asr_setup_plan.v1"


def plan_asr_setup(
    venv_dir: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    write: bool = False,
    python_version: str = "3.11",
    preferred: str = "funasr",
) -> dict[str, Any]:
    """Create a preview-only ASR environment setup handoff from current state."""
    env = asr_environment_status(venv_dir=venv_dir, python_version=python_version)
    preferred_tool = _select_preferred_tool(env, preferred=preferred)
    target_dir = _target_venv_dir(env)
    env_snippet = _env_snippet_for(target_dir)
    steps = _steps(env, preferred_tool=preferred_tool, target_dir=target_dir, env_snippet=env_snippet)
    result = {
        "schema": ASR_SETUP_PLAN_SCHEMA,
        "checked_at": now_iso(),
        "preferred": preferred_tool.get("name", preferred),
        "status": _status(env, steps),
        "environment": env,
        "target_venv_dir": str(target_dir),
        "steps": steps,
        "next_step": next((step for step in steps if step.get("status") != "done"), steps[-1] if steps else {}),
        "env_snippet": env_snippet,
        "recommended_order": list(env.get("recommended_order") or []),
    }
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "asr-setup-plan.json"
        markdown_path = out_dir / "asr-setup-plan.md"
        args_path = out_dir / "mcp-plan-asr-setup.args.json"
        result["output_json"] = str(json_path)
        result["output_markdown"] = str(markdown_path)
        result["mcp_args_path"] = str(args_path)
        if write:
            write_json(json_path, result)
            markdown_path.write_text(render_asr_setup_plan_markdown(result), encoding="utf-8")
            write_json(args_path, {"venv_dir": str(target_dir), "output_dir": str(out_dir), "write": True, "python_version": python_version, "preferred": preferred_tool.get("name", preferred)})
    return result


def render_asr_setup_plan_markdown(result: dict[str, Any]) -> str:
    env = result.get("environment") if isinstance(result.get("environment"), dict) else {}
    next_step = result.get("next_step") if isinstance(result.get("next_step"), dict) else {}
    lines = [
        "# ASR Setup Plan",
        "",
        f"- Schema: `{result.get('schema', '')}`",
        f"- Checked: `{result.get('checked_at', '')}`",
        f"- Status: `{result.get('status', '')}`",
        f"- Preferred: `{result.get('preferred', '')}`",
        f"- Env: `{env.get('venv_dir', '')}`",
        f"- Next: `{next_step.get('key', '')}` / {next_step.get('label', '')}",
        "",
        "## Steps",
        "",
        "| Step | Status | Command | Why |",
        "|---|---|---|---|",
    ]
    for step in result.get("steps") or []:
        if not isinstance(step, dict):
            continue
        lines.append(
            "| {key} | `{status}` | `{command}` | {reason} |".format(
                key=_md_cell(str(step.get("key") or "")),
                status=bool(step.get("status") == "done") and "done" or str(step.get("status") or ""),
                command=_md_cell(str(step.get("command") or "")),
                reason=_md_cell(str(step.get("reason") or "")),
            )
        )
    lines.extend(["", "## Env Snippet", "", "```powershell"])
    lines.extend(str(line) for line in result.get("env_snippet") or [])
    lines.extend(["```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _select_preferred_tool(env: dict[str, Any], *, preferred: str) -> dict[str, Any]:
    tools = [tool for tool in env.get("tools") or [] if isinstance(tool, dict)]
    for tool in tools:
        if str(tool.get("name") or "").lower() == preferred.lower():
            return tool
    for name in env.get("recommended_order") or []:
        for tool in tools:
            if str(tool.get("name") or "").lower() == str(name).lower():
                return tool
    return tools[0] if tools else {"name": preferred, "install_command": ""}


def _steps(env: dict[str, Any], *, preferred_tool: dict[str, Any], target_dir: Path, env_snippet: list[str]) -> list[dict[str, Any]]:
    venv_exists = bool(env.get("venv_exists"))
    current_dir = Path(str(env.get("venv_dir") or ".venv-lecture-asr"))
    target_is_current = target_dir == current_dir
    target_exists = target_dir.exists() if not target_is_current else venv_exists
    python = env.get("python") if isinstance(env.get("python"), dict) else {}
    recommended_python = bool(python.get("asr_recommended"))
    command_ready = bool(env.get("command_ready"))
    module_ready = bool(env.get("module_ready"))
    preferred_module = preferred_tool.get("module") if isinstance(preferred_tool.get("module"), dict) else {}
    preferred_command_ready = bool(preferred_tool.get("command_exists"))
    preferred_module_ready = bool(preferred_module.get("available"))
    install_command = _install_command_for(preferred_tool, target_dir)
    create_command = f'.\\scripts\\install-local-asr-env.ps1 -VenvDir "{target_dir}" -CreateCondaEnv -PythonVersion 3.11'
    steps = [
        {
            "key": "create_environment",
            "label": "创建 ASR 环境",
            "status": "done" if target_exists else "todo",
            "command": create_command,
            "reason": "FunASR/SenseVoice/WhisperX 在隔离的 Python 3.10-3.12 环境里更稳。",
        },
        {
            "key": "verify_python_version",
            "label": "确认 Python 版本",
            "status": "done" if target_is_current and venv_exists and recommended_python else "todo",
            "command": create_command,
            "reason": str(python.get("warning") or "Python version is in the recommended range."),
        },
        {
            "key": "install_preferred_asr",
            "label": f"安装 {preferred_tool.get('name', 'ASR')}",
            "status": "done" if preferred_module_ready or preferred_command_ready else "todo",
            "command": install_command,
            "reason": "中文知识类视频优先 FunASR/SenseVoice；需要对齐/多语种时再装 WhisperX 或 faster-whisper。",
        },
        {
            "key": "apply_env_snippet",
            "label": "加载 ASR 命令环境变量",
            "status": "done" if command_ready else "todo",
            "command": "; ".join(env_snippet),
            "reason": "让 CLI、MCP、BiliNote 的 plan-asr/run-asr-plan 能找到同一个 ASR runner。",
        },
        {
            "key": "run_plan_asr",
            "label": "生成并预览 ASR 运行计划",
            "status": "ready" if command_ready or module_ready else "blocked",
            "command": "video-knowledge plan-asr <media> <output-dir> --preset " + str(preferred_tool.get("name") or "funasr"),
            "reason": "先生成 guarded run plan，再决定是否执行真实转写。",
        },
    ]
    return steps


def _status(env: dict[str, Any], steps: list[dict[str, Any]]) -> str:
    if bool(env.get("command_ready")):
        return "ready"
    python = env.get("python") if isinstance(env.get("python"), dict) else {}
    if bool(env.get("venv_exists")) and not bool(python.get("asr_recommended")):
        return "needs_recreate_environment"
    if any(step.get("status") == "todo" for step in steps[:3]):
        return "needs_setup"
    return "needs_env"


def _target_venv_dir(env: dict[str, Any]) -> Path:
    current = Path(str(env.get("venv_dir") or ".venv-lecture-asr"))
    python = env.get("python") if isinstance(env.get("python"), dict) else {}
    if bool(env.get("venv_exists")) and not bool(python.get("asr_recommended")):
        return current.with_name(current.name + "-py311")
    return current


def _install_command_for(tool: dict[str, Any], target_dir: Path) -> str:
    name = str(tool.get("name") or "funasr").lower()
    switch = "-InstallFunASR" if name in {"funasr", "sensevoice"} else "-InstallWhisperX" if name == "whisperx" else "-InstallFasterWhisper"
    return f'.\\scripts\\install-local-asr-env.ps1 -VenvDir "{target_dir}" {switch}'


def _env_snippet_for(target_dir: Path) -> list[str]:
    scripts = target_dir / "Scripts"
    return [
        f'$env:LECTURE_ASR_BIN_DIR="{scripts}"',
        f'$env:LECTURE_FUNASR_COMMAND="{scripts / "funasr.exe"}"',
        f'$env:LECTURE_WHISPERX_COMMAND="{scripts / "whisperx.exe"}"',
        f'$env:LECTURE_FASTER_WHISPER_COMMAND="{scripts / "faster-whisper.exe"}"',
    ]
