from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import read_json, write_json
from .external_reuse_run_artifacts import ps_quote, register_external_reuse_run
from .transcript import format_timestamp
from .video_moment_index import build_video_moment_index

SCHEMA = "video_knowledge_pipeline.long_video_memory_pack.v1"


def build_long_video_memory_pack(
    bundle_dir: str | Path,
    *,
    target_window_seconds: float = 300.0,
    max_chunk_chars: int = 3600,
    long_group_size: int = 6,
    write: bool = True,
) -> dict[str, Any]:
    """Build a MovieChat-style text memory layer for long lecture videos.

    MovieChat compresses dense visual tokens into short/long memory. For VKP we
    reuse the idea at the evidence layer: timeline moments become short memories,
    grouped long memories, then a final prompt-ready pack.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")

    index = build_video_moment_index(
        root,
        target_window_seconds=target_window_seconds,
        max_chunk_chars=max_chunk_chars,
        write=write,
    )
    chunks = [chunk for chunk in index.get("chunks", []) if isinstance(chunk, dict)]
    short_memories = [_short_memory(chunk) for chunk in chunks]
    long_memories = _long_memories(short_memories, long_group_size=max(1, long_group_size))
    final_map = _final_memory_map(short_memories, long_memories)

    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "created_at": now_iso(),
        "inspired_by": [
            "MovieChat: dense-to-sparse memory for long video understanding",
            "VideoRAG: segment-level evidence retained for retrieval",
        ],
        "parameters": {
            "target_window_seconds": target_window_seconds,
            "max_chunk_chars": max_chunk_chars,
            "long_group_size": long_group_size,
        },
        "summary": {
            "short_memories": len(short_memories),
            "long_memories": len(long_memories),
            "duration_seconds": index.get("summary", {}).get("duration_seconds", 0.0),
            "coverage_start": short_memories[0]["start"] if short_memories else 0.0,
            "coverage_end": short_memories[-1]["end"] if short_memories else 0.0,
            "visual_evidence_short_memories": sum(1 for row in short_memories if row["has_visual_evidence"]),
            "temporal_evidence_short_memories": sum(1 for row in short_memories if row["has_temporal_evidence"]),
        },
        "short_memories": short_memories,
        "long_memories": long_memories,
        "final_memory_map": final_map,
        "codex_prompt": _codex_prompt(manifest, result_stub_title=str(manifest.get("title") or root.name)),
        "artifacts": {
            "json": str(root / "exports" / "long-video-memory-pack.json"),
            "markdown": str(root / "exports" / "long-video-memory-pack.md"),
        },
        "write": bool(write),
    }
    result["codex_prompt"] = _codex_prompt(manifest, result)
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "long-video-memory-pack.json", result)
        (exports / "long-video-memory-pack.md").write_text(_render_markdown(result), encoding="utf-8")
        manifest["long_video_memory_pack"] = "exports/long-video-memory-pack.json"
        manifest["long_video_memory_pack_markdown"] = "exports/long-video-memory-pack.md"
        manifest["mcp_long_video_memory_pack_args"] = "mcp-long-video-memory-pack.args.json"
        write_json(
            root / "mcp-long-video-memory-pack.args.json",
            {
                "bundle_dir": str(root),
                "target_window_seconds": target_window_seconds,
                "max_chunk_chars": max_chunk_chars,
                "long_group_size": long_group_size,
                "write": True,
            },
        )
        write_json(manifest_path, manifest)
        failed_items = [] if short_memories else [{"id": "memory_pack", "reason": "no_short_memories", "detail": "Moment index produced no short memories for the MovieChat-style memory pack."}]
        register_external_reuse_run(
            root,
            run_type="long_video_memory_pack",
            title="Long video memory pack",
            result=result,
            status="needs_input" if failed_items else "completed",
            failed_items=failed_items,
            retry_command=f".\\scripts\\video-knowledge.ps1 long-video-memory-pack {ps_quote(root)}",
            next_actions=[] if not failed_items else ["Build or repair video-moment-index, then rerun long-video-memory-pack."],
            write=True,
        )
    return result


def _short_memory(chunk: dict[str, Any]) -> dict[str, Any]:
    transcript = str(chunk.get("transcript_text") or "")
    visual = str(chunk.get("visual_text") or "")
    temporal = str(chunk.get("temporal_text") or "")
    bullets = _extractive_bullets(transcript, limit=4)
    visual_bullets = _extractive_bullets(" ".join([visual, temporal]), limit=3)
    return {
        "memory_id": f"M{int(chunk.get('chunk_index') or 0):04d}",
        "chunk_index": chunk.get("chunk_index"),
        "start": chunk.get("start", 0.0),
        "end": chunk.get("end", 0.0),
        "start_time": chunk.get("start_time") or format_timestamp(chunk.get("start", 0.0)),
        "end_time": chunk.get("end_time") or format_timestamp(chunk.get("end", 0.0)),
        "timeline_indexes": chunk.get("timeline_indexes") or [],
        "topic_hint": _topic_hint(chunk),
        "summary_bullets": bullets,
        "visual_bullets": visual_bullets,
        "keywords": chunk.get("keywords") or [],
        "has_visual_evidence": bool(chunk.get("has_visual_evidence")),
        "has_temporal_evidence": bool(chunk.get("has_temporal_evidence")),
        "evidence_paths": chunk.get("evidence_paths") or [],
    }


def _long_memories(short_memories: list[dict[str, Any]], *, long_group_size: int) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for i in range(0, len(short_memories), long_group_size):
        group = short_memories[i : i + long_group_size]
        if not group:
            continue
        keywords = _merge_keywords(group)
        bullets = []
        for row in group:
            bullets.extend(row.get("summary_bullets") or [])
        groups.append(
            {
                "long_memory_id": f"L{len(groups) + 1:03d}",
                "short_memory_ids": [row["memory_id"] for row in group],
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "start_time": group[0]["start_time"],
                "end_time": group[-1]["end_time"],
                "topic_hint": " / ".join(_dedupe([row["topic_hint"] for row in group if row.get("topic_hint")])[:5]),
                "merged_bullets": _dedupe(bullets)[:8],
                "keywords": keywords[:40],
                "visual_evidence_count": sum(1 for row in group if row.get("has_visual_evidence")),
                "temporal_evidence_count": sum(1 for row in group if row.get("has_temporal_evidence")),
            }
        )
    return groups


def _final_memory_map(short_memories: list[dict[str, Any]], long_memories: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "coverage": {
            "start_time": short_memories[0]["start_time"] if short_memories else "00:00:00.000",
            "end_time": short_memories[-1]["end_time"] if short_memories else "00:00:00.000",
            "short_memory_count": len(short_memories),
            "long_memory_count": len(long_memories),
        },
        "course_flow": [
            {
                "time": f"{row['start_time']} - {row['end_time']}",
                "topic_hint": row["topic_hint"],
                "key_points": row["merged_bullets"][:4],
                "evidence": {
                    "visual": row["visual_evidence_count"],
                    "temporal": row["temporal_evidence_count"],
                },
            }
            for row in long_memories
        ],
        "low_confidence_boundaries": [
            {
                "memory_id": row["memory_id"],
                "time": f"{row['start_time']} - {row['end_time']}",
                "reason": "missing_visual_or_temporal_evidence",
            }
            for row in short_memories
            if not row.get("has_visual_evidence") and not row.get("has_temporal_evidence")
        ][:80],
    }


def _codex_prompt(manifest: dict[str, Any], result: dict[str, Any] | None = None, *, result_stub_title: str = "") -> str:
    title = str(manifest.get("title") or result_stub_title or "视频")
    if result is None:
        return ""
    return "\n".join(
        [
            f"请基于 long-video-memory-pack 为《{title}》生成最终 smart-summary。",
            "",
            "要求：",
            "- 覆盖完整视频，不要只总结前几段。",
            "- 用分段时间戳组织课程主线。",
            "- 把视觉证据缺失或低置信内容放到末尾待复核。",
            "- 不要机械复制逐字稿。",
            "- 输出可直接进入 Obsidian/Logseq 的层级 Markdown。",
            "",
            "可用材料：",
            "- short_memories：每个时间窗的局部记忆。",
            "- long_memories：跨多个时间窗的压缩记忆。",
            "- final_memory_map：全片课程流和低置信边界。",
        ]
    )


def _topic_hint(chunk: dict[str, Any]) -> str:
    keywords = [str(token) for token in chunk.get("keywords") or [] if len(str(token)) >= 2]
    tags = [str(token) for token in chunk.get("tags") or [] if str(token).strip()]
    values = _dedupe(tags + keywords)
    if values:
        return "、".join(values[:6])
    text = str(chunk.get("transcript_text") or "")
    return _clip(text, 40) or "未识别主题"


def _extractive_bullets(text: str, *, limit: int) -> list[str]:
    value = _clean(text)
    if not value:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s*", value) if part.strip()]
    if len(sentences) <= 1:
        sentences = [value[i : i + 90].strip() for i in range(0, len(value), 90) if value[i : i + 90].strip()]
    scored = []
    for pos, sentence in enumerate(sentences):
        score = 0
        for marker in ("核心", "关键", "步骤", "方法", "原则", "注意", "问题", "案例", "总结", "所以", "因为", "必须", "不要"):
            if marker in sentence:
                score += 2
        score += min(3, len(sentence) / 60)
        score -= pos * 0.02
        scored.append((score, pos, sentence))
    scored.sort(key=lambda row: (-row[0], row[1]))
    picked = sorted(scored[:limit], key=lambda row: row[1])
    return [_clip(row[2], 180) for row in picked]


def _merge_keywords(rows: list[dict[str, Any]]) -> list[str]:
    values = []
    for row in rows:
        values.extend(str(token) for token in row.get("keywords") or [])
    return _dedupe(values)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = _clean(value).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(_clean(value))
    return result


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _clip(text: str, limit: int) -> str:
    value = _clean(text)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Long Video Memory Pack",
        "",
        f"- Bundle: `{result['bundle_dir']}`",
        f"- Title: {result['title']}",
        f"- Created: {result['created_at']}",
        f"- Duration: `{format_timestamp(result['summary']['duration_seconds'])}`",
        f"- Short memories: `{result['summary']['short_memories']}`",
        f"- Long memories: `{result['summary']['long_memories']}`",
        f"- Visual evidence memories: `{result['summary']['visual_evidence_short_memories']}`",
        f"- Temporal evidence memories: `{result['summary']['temporal_evidence_short_memories']}`",
        "",
        "## Long Memories",
        "",
    ]
    for row in result["long_memories"]:
        lines.extend(
            [
                f"### {row['long_memory_id']} {row['start_time']} - {row['end_time']}",
                "",
                f"- Short memories: `{','.join(row['short_memory_ids'])}`",
                f"- Topic hint: {row['topic_hint'] or '-'}",
                f"- Keywords: {', '.join(row['keywords'][:16])}",
                "",
            ]
        )
        for bullet in row["merged_bullets"][:8]:
            lines.append(f"- {bullet}")
        lines.append("")
    lines.extend(["## Low Confidence Boundaries", ""])
    low = result["final_memory_map"]["low_confidence_boundaries"]
    if not low:
        lines.append("No missing-visual/temporal memory boundary detected.")
    else:
        for row in low[:80]:
            lines.append(f"- `{row['memory_id']}` {row['time']}: {row['reason']}")
    lines.extend(["", "## Codex Prompt", "", "```text", result["codex_prompt"], "```"])
    return "\n".join(lines).rstrip() + "\n"
