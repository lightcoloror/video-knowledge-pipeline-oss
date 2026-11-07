from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .config import processing_profile
from .models import now_iso
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.quality_console.v1"


def export_quality_console(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    manifest = _mapping(manifest_path)
    stages = [
        _stage(root, manifest, "dual_asr", "双 ASR 共识", "asr_consensus_json", "asr-consensus.json", "asr-consensus.md", _asr_consensus_command(root, manifest)),
        _stage(root, manifest, "conflicts", "证据冲突与纠错补丁", "evidence_conflict_index_json", "evidence-conflict-index.json", "evidence-conflict-index.md", _command(root, "transcript-evidence-correction-pipeline", "--quality-profile quality")),
        _stage(root, manifest, "transcript_gate", "纠正版逐字稿质量门禁", "transcript_quality_gate_json", "transcript-quality-gate.json", "transcript-quality-gate.md", _command(root, "transcript-quality-gate")),
        _stage(root, manifest, "semantic_chapters", "语义章节", "semantic_chapter_plan_json", "exports/semantic-chapter-plan.json", "exports/semantic-chapter-plan.md", _command(root, "semantic-chapter-plan")),
        _stage(root, manifest, "summary_sections", "章节级 LLM 总结", "smart_summary_section_llm_rewrite_json", "exports/smart-summary-section-llm-rewrite.json", "exports/smart-summary-section-llm-rewrite.md", _command(root, "run-smart-summary-section-llm-rewrite", "--auto-from-profile")),
        _stage(root, manifest, "summary_gate", "智能总结质量门禁", "smart_summary_quality_json", "exports/smart-summary-quality.json", "exports/smart-summary-quality.md", _command(root, "smart-summary-quality-check")),
    ]
    benchmark_candidates = [root / "quality-benchmark.json", root / "quality-benchmark" / "quality-benchmark.json"]
    benchmark = next((path for path in benchmark_candidates if path.exists()), None)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "profile": processing_profile(),
        "status": "ready" if all(row["status"] in {"completed", "passed", "ready"} for row in stages) else "needs_work",
        "stages": stages,
        "benchmark": {"exists": bool(benchmark), "path": str(benchmark or ""), "status": _artifact_status(benchmark) if benchmark else "not_built"},
        "navigation": {"task_console": "task-console.html", "video_workbench": "video-workbench.html", "review": "review.html"},
        "operator_boundary": {"static_read_only_ui": True, "cloud_calls_not_executed": True, "retry_buttons_copy_commands_only": True},
        "updated_at": now_iso(),
        "artifacts": {"json": "quality-console.json", "html": "quality-console.html", "mcp_args": "mcp-export-quality-console.args.json"},
    }
    if write:
        write_json(root / "quality-console.json", result)
        (root / "quality-console.html").write_text(_render_html(result), encoding="utf-8")
        write_json(root / "mcp-export-quality-console.args.json", {"bundle_dir": str(root), "write": True})
        manifest["quality_console"] = "quality-console.html"
        manifest["quality_console_json"] = "quality-console.json"
        manifest["mcp_export_quality_console_args"] = "mcp-export-quality-console.args.json"
        write_json(manifest_path, manifest)
    return result


def _stage(root: Path, manifest: dict[str, Any], key: str, label: str, manifest_key: str, fallback_json: str, markdown: str, command: str) -> dict[str, Any]:
    value = str(manifest.get(manifest_key) or fallback_json)
    path = _bundle_path(root, value)
    md_path = _bundle_path(root, markdown)
    return {
        "key": key,
        "label": label,
        "status": _artifact_status(path),
        "exists": path.exists(),
        "json_path": str(path),
        "json_href": _href(root, path),
        "markdown_path": str(md_path),
        "markdown_href": _href(root, md_path),
        "retry_command": command,
    }


def _artifact_status(path: Path | None) -> str:
    if not path or not path.exists():
        return "not_run"
    try:
        data = read_json(path)
    except Exception:
        return "artifact_exists"
    if not isinstance(data, dict):
        return "artifact_exists"
    return str(data.get("status") or ("passed" if data.get("ok") else "needs_review"))


def _asr_consensus_command(root: Path, manifest: dict[str, Any]) -> str:
    primary = str(manifest.get("asr_primary_transcript") or "<sensevoice-transcript.json>")
    secondary = str(manifest.get("asr_secondary_transcript") or "<qwen3-transcript.json>")
    return f".\\scripts\\video-knowledge.ps1 asr-consensus '{root}' '{primary}' '{secondary}'"


def _command(root: Path, command: str, extra: str = "") -> str:
    suffix = f" {extra.strip()}" if extra.strip() else ""
    return f".\\scripts\\video-knowledge.ps1 {command} '{root}'{suffix}"


def _render_html(result: dict[str, Any]) -> str:
    rows = []
    for stage in result.get("stages") or []:
        status = html.escape(str(stage.get("status") or "unknown"))
        label = html.escape(str(stage.get("label") or ""))
        json_href = html.escape(str(stage.get("json_href") or ""))
        md_href = html.escape(str(stage.get("markdown_href") or ""))
        command = html.escape(str(stage.get("retry_command") or ""), quote=True)
        rows.append(
            f'<tr><td>{label}</td><td><code>{status}</code></td><td><a href="{json_href}">JSON</a> <a href="{md_href}">报告</a></td><td><button data-command="{command}" onclick="copyCommand(this)">复制重试命令</button></td></tr>'
        )
    profile = result.get("profile") if isinstance(result.get("profile"), dict) else {}
    profile_text = html.escape(json.dumps(profile, ensure_ascii=False, indent=2))
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VKP 质量控制台</title>
<style>
body{{margin:0;font:14px/1.55 system-ui,"Microsoft YaHei",sans-serif;color:#202124;background:#f4f6f8}}header{{background:#fff;border-bottom:1px solid #d8dde3;padding:14px 22px;position:sticky;top:0}}main{{max-width:1180px;margin:0 auto;padding:22px}}nav a{{margin-right:14px;color:#1261a0}}section{{background:#fff;border:1px solid #d8dde3;border-radius:6px;padding:16px;margin-bottom:16px}}h1{{font-size:22px;margin:0 0 8px}}h2{{font-size:17px;margin:0 0 12px}}table{{width:100%;border-collapse:collapse;table-layout:fixed}}th,td{{padding:10px;border-bottom:1px solid #e5e8eb;text-align:left;vertical-align:top;word-break:break-word}}button{{border:1px solid #9aa4ad;background:#fff;padding:6px 10px;border-radius:4px;cursor:pointer}}button:hover{{background:#eef3f7}}pre{{white-space:pre-wrap;overflow:auto;background:#f7f8fa;padding:12px;border:1px solid #e1e5e9}}.muted{{color:#687078}}
</style></head><body>
<header><h1>VKP 质量控制台</h1><nav><a href="task-console.html">任务控制台</a><a href="video-workbench.html">视频工作台</a><a href="review.html">审核页</a></nav></header>
<main><section><h2>质量执行状态</h2><div class="muted">状态：{html.escape(str(result.get('status')))}。按钮只复制命令，不会自动发送本地材料或调用云模型。</div>
<table><thead><tr><th>阶段</th><th>状态</th><th>产物</th><th>操作</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>处理配置</h2><pre>{profile_text}</pre></section></main>
<script>async function copyCommand(button){{const value=button.dataset.command||'';await navigator.clipboard.writeText(value);const old=button.textContent;button.textContent='已复制';setTimeout(()=>button.textContent=old,1200);}}</script>
</body></html>'''


def _mapping(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _href(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_uri()
