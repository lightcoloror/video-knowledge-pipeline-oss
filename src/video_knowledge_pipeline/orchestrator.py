from __future__ import annotations

from .config import DEFAULT_LOCAL_FRAME_BUDGET, DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS, DEFAULT_LOCAL_FRAME_SAMPLING_MODE, LOCAL_FRAME_SAMPLING_MODES
import csv
import json
import shutil
from pathlib import Path

from .models import ClaimRecord, EvidenceRecord, ResearchProject, SourceRecord, SourceReview, dataclass_to_dict, new_id, now_iso, project_paths
from .storage import append_jsonl, ensure_project_dirs, load_project, read_jsonl, save_project, write_json
from .transcript import format_timestamp, parse_transcript, semantic_timepoints
from .video import build_segments, extract_segment_frames, fixed_timepoints, merge_timepoints, probe_video
from .scene_detection_adapter import pyscenedetect_timepoints


def init_project(root: str | Path, question: str) -> ResearchProject:
    project = ResearchProject(id=new_id("research"), question=question, created_at=now_iso())
    save_project(root, project)
    return project


def add_source(
    root: str | Path,
    *,
    title: str,
    url: str = "",
    path: str = "",
    source_type: str = "web",
    quality: str = "unknown",
    notes: str = "",
) -> SourceRecord:
    paths = ensure_project_dirs(root)
    source = SourceRecord(
        id=new_id("source"),
        source_type=source_type,
        title=title,
        url=url,
        path=path,
        quality=quality,
        notes=notes,
    )
    append_jsonl(paths["sources"], [dataclass_to_dict(source)])
    return source


def review_source(
    root: str | Path,
    *,
    source_id: str,
    quality: str,
    reason: str = "",
    reviewer: str = "human",
) -> SourceReview:
    if quality not in {"good", "mixed", "poor", "unknown"}:
        raise ValueError("quality must be good, mixed, poor, or unknown")
    paths = ensure_project_dirs(root)
    sources = read_jsonl(paths["sources"])
    matched = False
    for source in sources:
        if source.get("id") == source_id:
            source["quality"] = quality
            source["quality_reason"] = reason
            matched = True
            break
    if not matched:
        raise ValueError("source not found")
    write_jsonl(paths["sources"], sources)
    review = SourceReview(id=new_id("source_review"), source_id=source_id, quality=quality, reason=reason, reviewer=reviewer)
    append_jsonl(paths["source_reviews"], [dataclass_to_dict(review)])
    return review


def add_claim(root: str | Path, *, text: str, notes: str = "") -> ClaimRecord:
    paths = ensure_project_dirs(root)
    claim = ClaimRecord(id=new_id("claim"), text=text, notes=notes)
    append_jsonl(paths["claims"], [dataclass_to_dict(claim)])
    append_jsonl(
        paths["graph"],
        [
            {
                "from_id": f"question:{load_project(root).question}",
                "to_id": f"claim:{claim.id}",
                "edge_type": "raises_claim",
                "status": "candidate",
                "reason": "manual claim added",
                "source_ref": "",
                "review_note": "",
                "reviewed_at": "",
            }
        ],
    )
    return claim


def add_evidence(
    root: str | Path,
    *,
    claim_id: str,
    source_id: str,
    quote_or_note: str,
    relation: str = "unknown",
    confidence: str = "unknown",
    notes: str = "",
) -> EvidenceRecord:
    if relation not in {"supports", "contradicts", "context", "unknown"}:
        raise ValueError("relation must be supports, contradicts, context, or unknown")
    paths = ensure_project_dirs(root)
    _require_record(paths["claims"], claim_id, "claim")
    _require_record(paths["sources"], source_id, "source")
    evidence = EvidenceRecord(
        id=new_id("evidence"),
        claim_id=claim_id,
        source_id=source_id,
        quote_or_note=quote_or_note,
        relation=relation,
        confidence=confidence,
        notes=notes,
    )
    append_jsonl(paths["evidence"], [dataclass_to_dict(evidence)])
    append_jsonl(
        paths["graph"],
        [
            {
                "from_id": f"source:{source_id}",
                "to_id": f"evidence:{evidence.id}",
                "edge_type": "contains_evidence",
                "status": "candidate",
                "reason": relation,
                "source_ref": source_id,
                "review_note": "",
                "reviewed_at": "",
            },
            {
                "from_id": f"evidence:{evidence.id}",
                "to_id": f"claim:{claim_id}",
                "edge_type": relation,
                "status": "candidate",
                "reason": "manual evidence link",
                "source_ref": source_id,
                "review_note": "",
                "reviewed_at": "",
            },
        ],
    )
    card_path = paths["notes"] / f"{evidence.id}-evidence-card.md"
    card_path.write_text(render_evidence_card(root, dataclass_to_dict(evidence)), encoding="utf-8")
    return evidence


def review_evidence(
    root: str | Path,
    *,
    evidence_id: str,
    status: str,
    relation: str | None = None,
    confidence: str | None = None,
    notes: str = "",
) -> dict:
    if status not in {"candidate", "verified", "rejected"}:
        raise ValueError("status must be candidate, verified, or rejected")
    if relation is not None and relation not in {"supports", "contradicts", "context", "unknown"}:
        raise ValueError("relation must be supports, contradicts, context, or unknown")
    paths = ensure_project_dirs(root)
    evidence_rows = read_jsonl(paths["evidence"])
    updated_evidence: dict | None = None
    for row in evidence_rows:
        if row.get("id") == evidence_id:
            row["status"] = status
            if relation is not None:
                row["relation"] = relation
            if confidence is not None:
                row["confidence"] = confidence
            if notes:
                row["review_note"] = notes
            row["reviewed_at"] = now_iso()
            updated_evidence = row
            break
    if not updated_evidence:
        raise ValueError("evidence not found")
    write_jsonl(paths["evidence"], evidence_rows)

    graph_rows = read_jsonl(paths["graph"])
    for row in graph_rows:
        if row.get("from_id") == f"evidence:{evidence_id}" or row.get("to_id") == f"evidence:{evidence_id}":
            row["status"] = status
            row["review_note"] = notes
            row["reviewed_at"] = updated_evidence["reviewed_at"]
            if relation is not None and row.get("from_id") == f"evidence:{evidence_id}":
                row["edge_type"] = relation
                row["reason"] = "evidence review"
    write_jsonl(paths["graph"], graph_rows)
    append_jsonl(
        paths["reviews"],
        [
            {
                "kind": "evidence",
                "evidence_id": evidence_id,
                "status": status,
                "relation": updated_evidence.get("relation", "unknown"),
                "confidence": updated_evidence.get("confidence", "unknown"),
                "note": notes,
                "reviewed_at": updated_evidence["reviewed_at"],
            }
        ],
    )
    card_path = paths["notes"] / f"{evidence_id}-evidence-card.md"
    card_path.write_text(render_evidence_card(root, updated_evidence), encoding="utf-8")
    return {"updated": 1, "evidence": updated_evidence}


def list_records(root: str | Path, kind: str) -> list[dict]:
    paths = project_paths(root)
    mapping = {
        "sources": paths["sources"],
        "claims": paths["claims"],
        "evidence": paths["evidence"],
        "graph": paths["graph"],
        "reviews": paths["reviews"],
    }
    if kind not in mapping:
        raise ValueError("kind must be sources, claims, evidence, graph, or reviews")
    return read_jsonl(mapping[kind])


def draft_answer(root: str | Path, *, include_candidate: bool = False) -> str:
    paths = project_paths(root)
    project = load_project(root)
    claims = read_jsonl(paths["claims"])
    evidence_rows = read_jsonl(paths["evidence"])
    sources = {source.get("id"): source for source in read_jsonl(paths["sources"])}
    evidence_by_claim: dict[str, list[dict]] = {}
    for evidence in evidence_rows:
        if not include_candidate and evidence.get("status") != "verified":
            continue
        evidence_by_claim.setdefault(str(evidence.get("claim_id", "")), []).append(evidence)

    lines = [
        "---",
        "type: note",
        f'title: "{project.question} - 阶段性回答草稿"',
        "tags: [ai-research, answer-draft]",
        "status: active",
        f"created: {now_iso()}",
        "---",
        "",
        f"# 阶段性回答草稿：{project.question}",
        "",
        "> 这不是最终结论。它只汇总当前已审核证据，并保留反证、上下文和缺口。",
        "",
        "## 证据状态",
        "",
        f"- 使用候选证据：{include_candidate}",
        f"- 主张数量：{len(claims)}",
        f"- 纳入证据数量：{sum(len(items) for items in evidence_by_claim.values())}",
        "",
        "## 按主张汇总",
        "",
    ]
    if not claims:
        lines.append("- 暂无主张")
    for claim in claims:
        claim_id = str(claim.get("id", ""))
        items = evidence_by_claim.get(claim_id, [])
        lines.extend([f"### {claim.get('text', '')}", "", f"- 主张 ID：`{claim_id}`", f"- 当前状态：`{claim.get('status', 'open')}`", ""])
        if not items:
            lines.extend(["当前没有已纳入的证据。", ""])
            continue
        for evidence in items:
            source = sources.get(evidence.get("source_id"), {})
            lines.extend(
                [
                    f"- `{evidence.get('relation', 'unknown')}` / `{evidence.get('confidence', 'unknown')}` / 来源质量 `{source.get('quality', 'unknown')}`",
                    f"  - 来源：{source.get('title', evidence.get('source_id', ''))}",
                    f"  - 摘录：{evidence.get('quote_or_note', '')}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## 待人工处理",
            "",
            "- [ ] 检查是否存在高质量反证",
            "- [ ] 检查来源质量是否足够支撑结论",
            "- [ ] 把仍然只是推论的内容标记出来",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def import_sources(root: str | Path, import_path: str | Path, *, default_quality: str = "unknown") -> dict:
    rows = _load_source_import_rows(import_path)
    imported: list[dict] = []
    for row in rows:
        title = str(row.get("title") or row.get("name") or row.get("url") or row.get("path") or "Untitled source")
        imported_source = add_source(
            root,
            title=title,
            url=str(row.get("url") or row.get("link") or ""),
            path=str(row.get("path") or ""),
            source_type=str(row.get("source_type") or row.get("type") or "web"),
            quality=str(row.get("quality") or default_quality),
            notes=str(row.get("notes") or row.get("snippet") or row.get("summary") or ""),
        )
        imported.append(dataclass_to_dict(imported_source))
    return {"imported_count": len(imported), "sources": imported}



def _frame_sampling_plan(duration: float, sample_interval: float, max_frames: int, sample_mode: str) -> dict:
    mode = sample_mode if sample_mode in LOCAL_FRAME_SAMPLING_MODES else DEFAULT_LOCAL_FRAME_SAMPLING_MODE
    requested_interval = max(1.0, float(sample_interval or DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS))
    requested_budget = max(1, int(max_frames or DEFAULT_LOCAL_FRAME_BUDGET))
    duration = max(0.0, float(duration or 0.0))
    effective_budget = requested_budget

    if mode == "dense-local":
        desired_budget = int(duration // requested_interval) + 1 if duration > 0 else 1
        if duration > 0 and duration % requested_interval > 0:
            desired_budget += 1
        effective_budget = max(requested_budget, desired_budget)
        fixed_budget = effective_budget
        effective_interval = requested_interval
    elif mode == "triage-first":
        fixed_budget = max(1, min(requested_budget, round(requested_budget * 0.7)))
        effective_interval = _balanced_sampling_interval(duration, requested_interval, fixed_budget)
    else:
        fixed_budget = requested_budget
        effective_interval = _balanced_sampling_interval(duration, requested_interval, fixed_budget)

    return {
        "mode": mode,
        "duration_seconds": duration,
        "requested_interval_seconds": requested_interval,
        "requested_max_frames": requested_budget,
        "effective_interval_seconds": effective_interval,
        "effective_max_frames": effective_budget,
        "fixed_budget": fixed_budget,
        "reserved_for_scene_or_semantic": max(0, effective_budget - fixed_budget),
    }


def _balanced_sampling_interval(duration: float, requested_interval: float, budget: int) -> float:
    if duration <= 0 or budget <= 1:
        return requested_interval
    return max(requested_interval, duration / max(1, budget - 1))


def add_video(
    root: str | Path,
    video_path: str | Path,
    *,
    topic: str,
    transcript_path: str | Path | None = None,
    sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
    max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
    sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
    detect_scenes: bool = True,
    extract_frames: bool = True,
) -> dict:
    paths = ensure_project_dirs(root)
    metadata = probe_video(video_path)
    from .media_applicability import inspect_media_source
    media_capability = inspect_media_source(video_path, expected_sha256=metadata.sha256)
    if media_capability.get("media_kind") == "audio_only" and media_capability.get("source_bound"):
        detect_scenes = False
        extract_frames = False
    video_dir = paths["videos"] / metadata.id
    frame_dir = video_dir / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)

    cues = parse_transcript(transcript_path) if transcript_path else []
    sampling_plan = _frame_sampling_plan(metadata.duration_seconds, sample_interval, max_frames, sample_mode)
    sampling_plan["media_capability"] = media_capability
    fixed_points = fixed_timepoints(metadata.duration_seconds, sampling_plan["effective_interval_seconds"], sampling_plan["fixed_budget"])
    extra_budget = max(0, sampling_plan["effective_max_frames"] - len(fixed_points))
    extra_points = []
    sampling_plan["scene_detection"] = {
        "backend": "disabled" if not detect_scenes else "not_run",
        "boundary_count": 0,
        "fallback_reason": "",
    }
    if detect_scenes and extra_budget > 0:
        scene_budget = extra_budget if sampling_plan["mode"] != "triage-first" else max(0, extra_budget // 2)
        if scene_budget > 0:
            scene_points, scene_status = pyscenedetect_timepoints(video_path, max_points=scene_budget)
            sampling_plan["scene_detection"] = scene_status
        else:
            scene_points = []
        extra_points.extend(scene_points)
        extra_budget = max(0, sampling_plan["effective_max_frames"] - len(fixed_points) - len(extra_points))
    if extra_budget > 0:
        extra_points.extend(semantic_timepoints(cues, topic=topic, max_points=extra_budget))
    points = fixed_points + extra_points
    merged = merge_timepoints(points, window_seconds=2.0, duration=metadata.duration_seconds, max_segments=sampling_plan["effective_max_frames"])
    sampling_plan["fixed_points"] = len(fixed_points)
    sampling_plan["extra_points"] = len(extra_points)
    sampling_plan["merged_segments"] = len(merged)
    segments = build_segments(video_id=metadata.id, duration=metadata.duration_seconds, timepoints=merged, cues=cues)

    if extract_frames:
        extract_segment_frames(video_path, frame_dir, segments)

    write_json(video_dir / "metadata.json", dataclass_to_dict(metadata))
    write_json(video_dir / "sampling-plan.json", sampling_plan)
    write_json(video_dir / "segments.json", [dataclass_to_dict(segment) for segment in segments])
    if transcript_path:
        shutil.copy2(transcript_path, video_dir / Path(transcript_path).name)

    evidence_card = render_video_evidence_card(metadata=dataclass_to_dict(metadata), segments=[dataclass_to_dict(s) for s in segments], topic=topic)
    card_path = paths["notes"] / f"{metadata.id}-video-evidence-card.md"
    card_path.write_text(evidence_card, encoding="utf-8")

    graph_rows = graph_candidates_for_video(metadata=dataclass_to_dict(metadata), segments=[dataclass_to_dict(s) for s in segments], topic=topic)
    append_jsonl(paths["graph"], graph_rows)

    return {
        "video_id": metadata.id,
        "metadata_path": str(video_dir / "metadata.json"),
        "segments_path": str(video_dir / "segments.json"),
        "sampling_plan_path": str(video_dir / "sampling-plan.json"),
        "sampling_plan": sampling_plan,
        "card_path": str(card_path),
        "graph_path": str(paths["graph"]),
        "segment_count": len(segments),
    }


def export_obsidian(root: str | Path, vault: str | Path, folder: str = "00_Inbox/AI/问题导向研究流程POC") -> dict:
    paths = project_paths(root)
    vault_path = Path(vault)
    target_folder = _safe_relative_folder(folder)
    output_dir = vault_path / target_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = render_project_overview(root)
    overview_path = paths["notes"] / "research-project-overview.md"
    overview_path.write_text(overview, encoding="utf-8")
    answer_path = paths["notes"] / "answer-draft.md"
    answer_path.write_text(draft_answer(root), encoding="utf-8")
    exported = []
    for note in sorted(paths["notes"].glob("*.md")):
        target = output_dir / note.name
        target.write_text(note.read_text(encoding="utf-8"), encoding="utf-8")
        exported.append(str(target))
    return {"exported": exported, "folder": str(output_dir)}


def write_graph_csv(root: str | Path) -> Path:
    paths = project_paths(root)
    rows = read_jsonl(paths["graph"])
    target = paths["root"] / "graph-candidates.csv"
    fieldnames = ["from_id", "to_id", "edge_type", "status", "reason", "source_ref", "review_note", "reviewed_at"]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return target


def review_graph_edge(
    root: str | Path,
    *,
    from_id: str,
    to_id: str,
    status: str,
    note: str = "",
) -> dict:
    if status not in {"candidate", "verified", "rejected"}:
        raise ValueError("status must be candidate, verified, or rejected")
    paths = ensure_project_dirs(root)
    rows = read_jsonl(paths["graph"])
    reviewed_at = now_iso()
    updated = 0
    for row in rows:
        if row.get("from_id") == from_id and row.get("to_id") == to_id:
            row["status"] = status
            row["review_note"] = note
            row["reviewed_at"] = reviewed_at
            updated += 1
    if not updated:
        raise ValueError("graph edge not found")
    write_jsonl(paths["graph"], rows)
    review_record = {"from_id": from_id, "to_id": to_id, "status": status, "note": note, "reviewed_at": reviewed_at}
    append_jsonl(paths["reviews"], [review_record])
    return {"updated": updated, "review": review_record}


def render_project_overview(root: str | Path) -> str:
    paths = project_paths(root)
    project = load_project(root)
    sources = read_jsonl(paths["sources"])
    claims = read_jsonl(paths["claims"])
    evidence = read_jsonl(paths["evidence"])
    source_reviews = read_jsonl(paths["source_reviews"])
    graph_rows = read_jsonl(paths["graph"])
    reviews = read_jsonl(paths["reviews"])
    status_counts: dict[str, int] = {}
    for row in graph_rows:
        status = str(row.get("status", "candidate"))
        status_counts[status] = status_counts.get(status, 0) + 1
    lines = [
        "---",
        "type: note",
        f'title: "{project.question}"',
        "tags: [ai-research, question-led-research]",
        "status: active",
        f"created: {now_iso()}",
        "---",
        "",
        f"# 研究项目总览：{project.question}",
        "",
        f"- 项目 ID：`{project.id}`",
        f"- 状态：`{project.status}`",
        f"- 创建时间：`{project.created_at}`",
        f"- 来源数量：{len(sources)}",
        f"- 主张数量：{len(claims)}",
        f"- 证据数量：{len(evidence)}",
        f"- 来源质量标注：{len(source_reviews)}",
        f"- 图谱候选边：{len(graph_rows)}",
        f"- 人工审核记录：{len(reviews)}",
        "",
        "## 当前图谱状态",
        "",
    ]
    if status_counts:
        lines.extend(f"- `{status}`：{count}" for status, count in sorted(status_counts.items()))
    else:
        lines.append("- 暂无图谱候选边")
    lines.extend(["", "## 来源清单", ""])
    if sources:
        for source in sources:
            ref = source.get("url") or source.get("path") or ""
            lines.append(f"- `{source.get('quality', 'unknown')}` {source.get('title', '')} {ref}")
    else:
        lines.append("- 暂无来源")
    lines.extend(["", "## 主张清单", ""])
    if claims:
        for claim in claims:
            lines.append(f"- `{claim.get('status', 'open')}` {claim.get('text', '')} (`{claim.get('id', '')}`)")
    else:
        lines.append("- 暂无主张")
    lines.extend(
        [
            "",
            "## 人工审核队列",
            "",
            "- [ ] 复核 `candidate` 边是否支持研究问题",
            "- [ ] 把确认有效的边改为 `verified`",
            "- [ ] 把错误、无关或误导的边改为 `rejected` 并保留原因",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_video_evidence_card(*, metadata: dict, segments: list[dict], topic: str) -> str:
    lines = [
        "---",
        'type: note',
        f'title: "{metadata.get("title", "video evidence")}"',
        "tags: [ai-research, video-evidence]",
        "status: active",
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# 视频证据卡：{metadata.get('title', '')}",
        "",
        f"- 研究问题：{topic}",
        f"- 视频文件：`{metadata.get('path', '')}`",
        f"- 视频 ID：`{metadata.get('id', '')}`",
        f"- 时长：{metadata.get('duration_seconds', 0):.2f} 秒",
        f"- 分辨率：{metadata.get('width') or '?'} x {metadata.get('height') or '?'}",
        f"- SHA256：`{metadata.get('sha256', '')}`",
        "",
        "## 候选关键片段",
        "",
    ]
    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                f"### 片段 {index}: {format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}",
                "",
                f"- 状态：`{segment.get('status', 'candidate')}`",
                f"- 触发信号：{', '.join(segment.get('signals') or [])}",
                f"- 时间点：`{format_timestamp(segment.get('midpoint', 0))}`",
                f"- 需要人工复核：{segment.get('needs_human_review', True)}",
                f"- 不确定性：{segment.get('uncertainty', '')}",
                "",
                "#### 转录摘录",
                "",
                segment.get("transcript_excerpt") or "（无转录或该时间段未命中转录）",
                "",
                "#### 关键帧",
                "",
            ]
        )
        frame_paths = segment.get("frame_paths") or []
        if frame_paths:
            lines.extend(f"- `{frame}`" for frame in frame_paths)
        else:
            lines.append("- （未抽帧或抽帧失败）")
        lines.extend(["", "#### 人工观察", "", "- [ ] 补充画面事实", "- [ ] 标记支持/反驳/无关", ""])
    return "\n".join(lines).rstrip() + "\n"


def graph_candidates_for_video(*, metadata: dict, segments: list[dict], topic: str) -> list[dict]:
    video_id = metadata.get("id", "")
    rows = [
        {
            "from_id": f"question:{topic}",
            "to_id": f"source:{video_id}",
            "edge_type": "uses_source",
            "status": "candidate",
            "reason": "local video added to research project",
            "source_ref": metadata.get("path", ""),
            "review_note": "",
            "reviewed_at": "",
        }
    ]
    for segment in segments:
        segment_id = segment.get("id", "")
        rows.append(
            {
                "from_id": f"source:{video_id}",
                "to_id": f"evidence:{segment_id}",
                "edge_type": "contains_evidence",
                "status": "candidate",
                "reason": ",".join(segment.get("signals") or []),
                "source_ref": f"{metadata.get('path', '')}#{format_timestamp(segment.get('midpoint', 0))}",
                "review_note": "",
                "reviewed_at": "",
            }
        )
    return rows


def render_evidence_card(root: str | Path, evidence: dict) -> str:
    paths = project_paths(root)
    claim = _find_record(paths["claims"], evidence.get("claim_id", ""))
    source = _find_record(paths["sources"], evidence.get("source_id", ""))
    lines = [
        "---",
        "type: note",
        f'title: "{evidence.get("id", "evidence")}"',
        "tags: [ai-research, evidence]",
        "status: active",
        f"created: {now_iso()}",
        "---",
        "",
        f"# 证据卡：{evidence.get('id', '')}",
        "",
        f"- 主张：{claim.get('text', '') if claim else evidence.get('claim_id', '')}",
        f"- 来源：{source.get('title', '') if source else evidence.get('source_id', '')}",
        f"- 来源质量：`{source.get('quality', 'unknown') if source else 'unknown'}`",
        f"- 关系：`{evidence.get('relation', 'unknown')}`",
        f"- 状态：`{evidence.get('status', 'candidate')}`",
        f"- 置信度：`{evidence.get('confidence', 'unknown')}`",
        "",
        "## 原文或记录",
        "",
        evidence.get("quote_or_note", ""),
        "",
        "## 人工审核",
        "",
        "- [ ] 这条证据是否直接支持/反驳主张？",
        "- [ ] 来源质量是否足够？",
        "- [ ] 是否需要补充反例或更高质量来源？",
        "",
        "## 备注",
        "",
        evidence.get("notes", ""),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_source_import_rows(import_path: str | Path) -> list[dict]:
    path = Path(import_path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            for key in ("sources", "results", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
        raise ValueError("JSON import must be a list or contain sources/results/items")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            rows.append({"url": value, "title": value})
    return rows


def _find_record(path: Path, record_id: str) -> dict | None:
    for row in read_jsonl(path):
        if row.get("id") == record_id:
            return row
    return None


def _require_record(path: Path, record_id: str, label: str) -> None:
    if not _find_record(path, record_id):
        raise ValueError(f"{label} not found: {record_id}")


def _safe_relative_folder(folder: str) -> Path:
    cleaned = Path(folder.strip() or "00_Inbox/AI")
    if cleaned.is_absolute() or any(part == ".." for part in cleaned.parts):
        raise ValueError("Obsidian folder must be relative to the vault")
    return cleaned
