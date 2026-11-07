from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .models import now_iso
from .path_defaults import provider_env_file
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .storage import read_json_object_or_empty as _read_object
from .vision_review_triage import vision_review_triage

SCHEMA = "video_knowledge_pipeline.vision_review_queue.v1"


def vision_review_queue(
    bundle_dir: str | Path,
    *,
    min_score: int = 10,
    batch_size: int = 10,
    max_items: int = 0,
    provider: str = "volcengine_coding_plan",
    env_file: str = str(provider_env_file()),
    write: bool = True,
    refresh_triage: bool = False,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    timeline_path = root / "timeline.json"
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")
    if refresh_triage or not (root / "vision-review-triage.json").exists():
        vision_review_triage(root, min_score=3, write=True)
    timeline = _timeline_by_index(read_json(timeline_path))
    triage = _read_object(root / "vision-review-triage.json")
    candidates = _pending_semantic_candidates(triage.get("semantic_candidates") or [], timeline, min_score=max(0, int(min_score or 0)))
    candidates = _merge_failed_visual_items(candidates, timeline, min_score=max(0, int(min_score or 0)))
    if max_items and max_items > 0:
        candidates = candidates[: int(max_items)]
    batches = _build_batches(candidates, root=root, batch_size=max(1, int(batch_size or 10)), env_file=env_file, provider=provider, timeline=timeline)
    completed_batches = sum(1 for batch in batches if batch["status"] == "completed")
    pending_batches = sum(1 for batch in batches if batch["status"] == "pending")
    partial_batches = sum(1 for batch in batches if batch["status"] == "partial")
    failed_batches = sum(1 for batch in batches if batch["status"] == "failed")
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "generated_at": now_iso(),
        "provider": provider,
        "env_file": env_file,
        "min_score": int(min_score or 0),
        "batch_size": int(batch_size or 10),
        "max_items": int(max_items or 0),
        "triage_path": str(root / "vision-review-triage.json"),
        "total_candidates": len(candidates),
        "completed_items": sum(1 for row in candidates if _has_ok_visual(timeline.get(row["index"], {}))),
        "pending_items": sum(1 for row in candidates if not _has_ok_visual(timeline.get(row["index"], {}))),
        "failed_or_incomplete_items": sum(1 for row in candidates if _has_failed_visual(timeline.get(row["index"], {}))),
        "batch_counts": {"total": len(batches), "completed": completed_batches, "partial": partial_batches, "pending": pending_batches, "failed": failed_batches},
        "batches": batches,
        "operator_boundary": {
            "default": "preview_and_copy_commands",
            "execute": "Only the generated PowerShell commands with -Execute send frames to a cloud provider.",
            "secrets": "API keys are read from env/private env files and are not written into queue artifacts.",
        },
    }
    if write:
        _write_outputs(root, result)
    return result


def _write_outputs(root: Path, result: dict[str, Any]) -> None:
    json_path = root / "vision-review-queue.json"
    md_path = root / "vision-review-queue.md"
    html_path = root / "vision-review-queue.html"
    args_path = root / "mcp-vision-review-queue.args.json"
    run_script = root / "vision-review-queue-run.ps1"
    write_json(json_path, result)
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    html_path.write_text(_render_html(result), encoding="utf-8")
    run_script.write_text(_render_run_script(result), encoding="utf-8")
    write_json(args_path, {"bundle_dir": str(root), "min_score": result["min_score"], "batch_size": result["batch_size"], "max_items": result["max_items"], "provider": result["provider"], "env_file": result["env_file"], "write": True})
    manifest_path = root / "manifest.json"
    manifest = _read_object(manifest_path)
    manifest.update({
        "vision_review_queue_json": "vision-review-queue.json",
        "vision_review_queue_report": "vision-review-queue.md",
        "vision_review_queue_html": "vision-review-queue.html",
        "vision_review_queue_run_script": "vision-review-queue-run.ps1",
        "mcp_vision_review_queue_args": "mcp-vision-review-queue.args.json",
    })
    write_json(manifest_path, manifest)
    register_bundle_run(
        root,
        run_type="vision_review_queue",
        run_id="vision-review-queue",
        status=_queue_run_status(result),
        title="Vision review queue",
        summary=f"{result['total_candidates']} candidates split into {result['batch_counts']['total']} batches.",
        inputs={"triage_path": result.get("triage_path", "")},
        parameters={
            "provider": result.get("provider", ""),
            "min_score": result.get("min_score", 0),
            "batch_size": result.get("batch_size", 0),
            "max_items": result.get("max_items", 0),
        },
        artifacts=[
            {"key": "vision_review_queue_json", "path": "vision-review-queue.json"},
            {"key": "vision_review_queue_report", "path": "vision-review-queue.md"},
            {"key": "vision_review_queue_html", "path": "vision-review-queue.html"},
            {"key": "vision_review_queue_run_script", "path": "vision-review-queue-run.ps1"},
            {"key": "mcp_vision_review_queue_args", "path": "mcp-vision-review-queue.args.json"},
        ],
        failed_items=_queue_failed_items(result),
        retry_command=_queue_retry_command(result, run_script),
        next_actions=[
            "Run selected batch commands with -Execute only after provider/API confirmation.",
            "Refresh vision-review-queue after each completed batch to update pending/failed counts.",
        ],
        operator_boundary=result.get("operator_boundary") or {},
        write=True,
    )



def _queue_run_status(result: dict[str, Any]) -> str:
    counts = result.get("batch_counts") or {}
    if int(counts.get("failed") or 0) > 0 or int(counts.get("partial") or 0) > 0:
        return "needs_retry"
    if int(result.get("pending_items") or 0) > 0:
        return "needs_execution"
    return "completed"


def _queue_failed_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in result.get("batches") or []:
        pending_indexes = [int(value) for value in (batch.get("pending_indexes") or []) if _int(value) > 0]
        failed_indexes = {int(value) for value in (batch.get("failed_or_incomplete_indexes") or []) if _int(value) > 0}
        for index in pending_indexes:
            is_failed = index in failed_indexes
            rows.append(
                {
                    "index": index,
                    "batch_id": batch.get("batch_id"),
                    "batch_status": batch.get("status"),
                    "reason": "visual_understanding_failed_or_incomplete" if is_failed else "visual_understanding_pending",
                    "detail": f"Batch {batch.get('batch_id')} pending indexes: {_csv(pending_indexes)}",
                    "pending_indexes": pending_indexes,
                    "suggested_next_tool": "run_multimodal_frame_analysis",
                    "suggested_retry_command": batch.get("retry_command") or batch.get("execute_command") or "",
                }
            )
    return rows


def _queue_retry_command(result: dict[str, Any], run_script: Path) -> str:
    for batch in result.get("batches") or []:
        if batch.get("status") in {"pending", "partial", "failed"}:
            command = str(batch.get("retry_command") or batch.get("execute_command") or "").strip()
            if command:
                return command
    return f"& {_ps_quote(str(run_script))}"
def _pending_semantic_candidates(rows: list[Any], timeline: dict[int, dict[str, Any]], *, min_score: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = _int(row.get("index"))
        if index <= 0:
            continue
        score = _int(row.get("score"))
        if score < min_score:
            continue
        item = timeline.get(index, {})
        if _has_ok_visual(item):
            continue
        out.append({
            "index": index,
            "score": score,
            "priority": row.get("priority") or _priority(score),
            "reasons": row.get("reasons") or row.get("semantic_reasons") or [],
            "transcript_excerpt": row.get("transcript_excerpt") or "",
        })
    return out


def _merge_failed_visual_items(candidates: list[dict[str, Any]], timeline: dict[int, dict[str, Any]], *, min_score: int) -> list[dict[str, Any]]:
    existing = {int(row["index"]) for row in candidates if row.get("index") is not None}
    failed_rows = []
    for index, item in sorted(timeline.items()):
        if index in existing or not _has_failed_visual(item):
            continue
        route = str(item.get("visual_route") or "")
        if route not in {"semantic_frame", "mixed"}:
            continue
        failed_rows.append({
            "index": index,
            "score": max(int(min_score or 0), 10),
            "priority": "high",
            "reasons": ["previous_visual_understanding_failed_or_incomplete"],
            "transcript_excerpt": str(item.get("transcript") or item.get("text") or "")[:160],
        })
    return failed_rows + candidates


def _build_batches(candidates: list[dict[str, Any]], *, root: Path, batch_size: int, env_file: str, provider: str, timeline: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    batches = []
    for offset in range(0, len(candidates), batch_size):
        rows = candidates[offset : offset + batch_size]
        indexes = [int(row["index"]) for row in rows]
        completed = [idx for idx in indexes if _has_ok_visual(timeline.get(idx, {}))]
        failed = [idx for idx in indexes if _has_failed_visual(timeline.get(idx, {}))]
        pending = [idx for idx in indexes if idx not in completed]
        if failed:
            status = "failed" if len(failed) == len(indexes) else "partial"
        elif not pending:
            status = "completed"
        else:
            status = "pending" if len(pending) == len(indexes) else "partial"
        index_csv = ",".join(str(idx) for idx in pending or indexes)
        command = _batch_command(root, env_file=env_file, indexes=index_csv, limit=len(pending or indexes), execute=False)
        execute_command = command + " -Execute"
        batches.append({
            "batch_id": len(batches) + 1,
            "status": status,
            "indexes": indexes,
            "pending_indexes": pending,
            "completed_indexes": completed,
            "failed_or_incomplete_indexes": failed,
            "limit": len(pending or indexes),
            "provider": provider,
            "preview_command": command,
            "execute_command": execute_command,
            "retry_command": execute_command,
            "scores": {str(row["index"]): row.get("score") for row in rows},
            "reasons": {str(row["index"]): row.get("reasons") or [] for row in rows},
        })
    return batches


def _batch_command(root: Path, *, env_file: str, indexes: str, limit: int, execute: bool) -> str:
    cmd = f".\\scripts\\run-volcengine-vision-batch.ps1 {_ps_quote(str(root))} -EnvFile {_ps_quote(env_file)} -Indexes {_ps_quote(indexes)} -Limit {int(limit)}"
    if execute:
        cmd += " -Execute"
    return cmd


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Vision Review Queue",
        "",
        f"- Bundle: `{result['bundle_dir']}`",
        f"- Provider: `{result['provider']}`",
        f"- Min score: `{result['min_score']}`",
        f"- Total candidates: `{result['total_candidates']}`",
        f"- Pending items: `{result['pending_items']}`",
        f"- Failed/incomplete items: `{result['failed_or_incomplete_items']}`",
        f"- Batches: `{result['batch_counts']['total']}` total, `{result['batch_counts']['completed']}` completed, `{result['batch_counts']['partial']}` partial, `{result['batch_counts']['pending']}` pending, `{result['batch_counts']['failed']}` failed",
        "",
        "## One-shot runner",
        "",
        "```powershell",
        f". {Path(result['bundle_dir']) / 'vision-review-queue-run.ps1'}",
        "```",
        "",
        "## Batches",
        "",
        "| Batch | Status | Pending | Failed | Command |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for batch in result.get("batches") or []:
        lines.append(f"| {batch['batch_id']} | {batch['status']} | `{_csv(batch['pending_indexes']) or '-'}` | `{_csv(batch['failed_or_incomplete_indexes']) or '-'}` | `{batch['retry_command']}` |")
    return "\n".join(lines).rstrip() + "\n"


def _render_html(result: dict[str, Any]) -> str:
    rows = "\n".join(_batch_html(batch) for batch in result.get("batches") or [])
    title = "疑难点多模态队列"
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <style>
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f7f8fa;color:#172026}}
    header{{background:#fff;border-bottom:1px solid #d8dee8;padding:22px 30px}}main{{max-width:1180px;margin:0 auto;padding:20px 24px}}
    .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.metric,.batch{{background:#fff;border:1px solid #d8dee8;border-radius:8px;padding:12px;margin:10px 0}}
    .metric span{{display:block;color:#667085;font-size:13px}}.metric strong{{font-size:24px}}code{{display:block;white-space:pre-wrap;word-break:break-word;background:#f1f4f8;border:1px solid #d8dee8;border-radius:6px;padding:8px}}
    button{{border:1px solid #d8dee8;background:#fff;border-radius:6px;padding:7px 10px;margin-top:8px;cursor:pointer}}.pending{{border-left:5px solid #995c00}}.failed{{border-left:5px solid #b42318}}.partial{{border-left:5px solid #2557a7}}.completed{{border-left:5px solid #0f6b4f}}.muted{{color:#667085}}
  </style>
</head>
<body>
<header><h1>{title}</h1><div class=\"muted\">根据 vision-review-triage 自动分批。页面不会直接调用云端；按钮只复制命令。</div></header>
<main>
  <section class=\"metrics\">
    <div class=\"metric\"><span>候选</span><strong>{result['total_candidates']}</strong></div>
    <div class=\"metric\"><span>待处理</span><strong>{result['pending_items']}</strong></div>
    <div class=\"metric\"><span>失败/不完整</span><strong>{result['failed_or_incomplete_items']}</strong></div>
    <div class=\"metric\"><span>批次</span><strong>{result['batch_counts']['total']}</strong></div>
  </section>
  <section class=\"batch\"><strong>一次跑完整队列</strong><code id=\"queueRun\">& .\\vision-review-queue-run.ps1</code><button onclick=\"copyText('queueRun')\">复制</button></section>
  <section>{rows}</section>
</main>
<script>async function copyText(id){{const el=document.getElementById(id); if(el) await navigator.clipboard.writeText(el.innerText);}}</script>
</body>
</html>"""


def _batch_html(batch: dict[str, Any]) -> str:
    bid = int(batch.get("batch_id") or 0)
    status = html.escape(str(batch.get("status") or "pending"))
    pending = html.escape(_csv(batch.get("pending_indexes") or []) or "-")
    failed = html.escape(_csv(batch.get("failed_or_incomplete_indexes") or []) or "-")
    command = html.escape(str(batch.get("retry_command") or ""))
    return f"""<div class=\"batch {status}\">
  <h2>Batch {bid}: {status}</h2>
  <div class=\"muted\">Pending: <code style=\"display:inline\">{pending}</code></div>
  <div class=\"muted\">Failed/incomplete: <code style=\"display:inline\">{failed}</code></div>
  <code id=\"batch-{bid}\">{command}</code>
  <button onclick=\"copyText('batch-{bid}')\">重试/执行这一批</button>
</div>"""


def _render_run_script(result: dict[str, Any]) -> str:
    lines = [
        "$ErrorActionPreference = \"Stop\"",
        "$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot \"..\\..\\..\\..\")).Path",
        "Set-Location -LiteralPath $RepoRoot",
        "Write-Host \"Running VKP vision review queue...\"",
    ]
    for batch in result.get("batches") or []:
        if not batch.get("pending_indexes") and not batch.get("failed_or_incomplete_indexes"):
            continue
        lines.extend([
            f"Write-Host \"Batch {batch['batch_id']} indexes: {_csv(batch.get('pending_indexes') or batch.get('indexes') or [])}\"",
            str(batch.get("retry_command") or ""),
        ])
    lines.append("Write-Host \"Queue finished. Refreshing queue status...\"")
    lines.append(f".\\scripts\\video-knowledge.ps1 vision-review-queue {_ps_quote(result['bundle_dir'])} --min-score {int(result['min_score'])} --batch-size {int(result['batch_size'])}")
    return "\n".join(lines).rstrip() + "\n"


def _timeline_by_index(value: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    rows = {}
    for pos, item in enumerate(value, start=1):
        if isinstance(item, dict):
            idx = _int(item.get("index")) or pos
            rows[idx] = item
    return rows


def _has_ok_visual(item: dict[str, Any]) -> bool:
    value = item.get("visual_understanding")
    if not isinstance(value, dict) or not value:
        return False
    if value.get("parse_failed") is True:
        return False
    if str(value.get("validation_status") or "ok") not in {"", "ok"}:
        return False
    return True


def _has_failed_visual(item: dict[str, Any]) -> bool:
    value = item.get("visual_understanding")
    issues = set(item.get("quality_issues") or [])
    if isinstance(value, dict) and value:
        if value.get("parse_failed") is True:
            return True
        if str(value.get("validation_status") or "ok") not in {"", "ok"}:
            return True
    return bool(issues.intersection({"model_output_parse_failed", "visual_understanding_incomplete"}))



def _int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _priority(score: int) -> str:
    if score >= 10:
        return "high"
    if score >= 6:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def _csv(values: list[Any]) -> str:
    return ",".join(str(v) for v in values if str(v).strip())
