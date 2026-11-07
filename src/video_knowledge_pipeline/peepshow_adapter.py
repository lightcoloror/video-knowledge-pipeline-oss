from __future__ import annotations

import json
import shutil
from pathlib import Path

from .models import EvidenceSegment, VideoMetadata, dataclass_to_dict, new_id, now_iso
from .orchestrator import graph_candidates_for_video, render_video_evidence_card
from .source_artifacts import build_source_artifact_index, collect_source_artifacts, render_source_artifact_index_markdown
from .storage import append_jsonl, ensure_project_dirs, write_json


def import_peepshow_output(root: str | Path, output_dir: str | Path, *, topic: str) -> dict:
    """Import a peepshow output directory as research video evidence."""
    paths = ensure_project_dirs(root)
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"peepshow output missing manifest.json: {output}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = _metadata_from_manifest(manifest)
    segments = _segments_from_manifest(metadata, manifest)

    video_dir = paths["videos"] / metadata.id
    video_dir.mkdir(parents=True, exist_ok=True)
    write_json(video_dir / "metadata.json", dataclass_to_dict(metadata))
    write_json(video_dir / "segments.json", [dataclass_to_dict(segment) for segment in segments])

    report_path = output / "report.html"
    copied_report_path = ""
    if report_path.exists():
        copied = paths["notes"] / f"{metadata.id}-peepshow-report.html"
        shutil.copyfile(report_path, copied)
        copied_report_path = str(copied)
    source_artifacts = collect_source_artifacts("peepshow", output, copied_paths={"report": copied_report_path})
    source_artifacts_path = video_dir / "source-artifacts.json"
    write_json(source_artifacts_path, source_artifacts)

    card = render_video_evidence_card(
        metadata=dataclass_to_dict(metadata),
        segments=[dataclass_to_dict(segment) for segment in segments],
        topic=topic,
    )
    card_path = paths["notes"] / f"{metadata.id}-peepshow-video-evidence-card.md"
    card_path.write_text(card, encoding="utf-8")

    graph_rows = graph_candidates_for_video(
        metadata=dataclass_to_dict(metadata),
        segments=[dataclass_to_dict(segment) for segment in segments],
        topic=topic,
    )
    append_jsonl(paths["graph"], graph_rows)

    return {
        "video_id": metadata.id,
        "imported_from": str(output),
        "segment_count": len(segments),
        "card_path": str(card_path),
        "copied_report_path": copied_report_path,
        "source_artifacts_path": str(source_artifacts_path),
        "source_artifact_count": source_artifacts["available_count"],
        "segments_path": str(video_dir / "segments.json"),
        "graph_path": str(paths["graph"]),
    }


def attach_peepshow_output_to_bundle(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    write: bool = True,
) -> dict:
    """Attach Peepshow outputs to a WebUI bundle as source evidence only."""
    bundle = Path(bundle_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    timeline_path = bundle / "timeline.json"
    peepshow_manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {bundle}")
    if not peepshow_manifest_path.exists():
        raise FileNotFoundError(f"peepshow output missing manifest.json: {output}")
    bundle_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(bundle_manifest, dict):
        raise ValueError("bundle manifest must be a JSON object")
    peepshow_manifest = json.loads(peepshow_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(peepshow_manifest, dict):
        raise ValueError("peepshow manifest must be a JSON object")
    timeline = []
    if timeline_path.exists():
        timeline_data = json.loads(timeline_path.read_text(encoding="utf-8"))
        timeline = [item for item in timeline_data if isinstance(item, dict)] if isinstance(timeline_data, list) else []

    report_path = output / "report.html"
    copied_report = ""
    if report_path.exists():
        copied_report_path = bundle / "source-peepshow-report.html"
        copied_report = str(copied_report_path)
        if write:
            shutil.copyfile(report_path, copied_report_path)
    metadata = _metadata_from_manifest(peepshow_manifest)
    source_artifacts = collect_source_artifacts("peepshow", output, copied_paths={"report": copied_report})
    evidence = _bundle_peepshow_evidence(peepshow_manifest, output, metadata=metadata)
    evidence_json_path = bundle / "peepshow-evidence.json"
    evidence_md_path = bundle / "peepshow-evidence.md"
    source_entry = {
        "video_id": str(metadata.id),
        "title": str(bundle_manifest.get("title") or metadata.title),
        "path": str((peepshow_manifest.get("input") or {}).get("originalPath") or (peepshow_manifest.get("input") or {}).get("url") or ""),
        "source_artifacts": source_artifacts,
        "peepshow_evidence_json": str(evidence_json_path),
        "peepshow_evidence_markdown": str(evidence_md_path),
        "peepshow_import_role": "optional_evidence_extractor",
        "notes": [
            "Peepshow evidence is preserved as source material only.",
            "Do not treat Peepshow OCR as structured_visual unless imported through run_visual_structure_plan.",
            "Do not treat Peepshow per-frame analysis as final visual_understanding unless explicitly imported and reviewed.",
        ],
    }
    result = {
        "schema": "peepshow_bundle_attach.v1",
        "bundle_dir": str(bundle),
        "output_dir": str(output),
        "manifest_path": str(manifest_path),
        "peepshow_manifest_path": str(peepshow_manifest_path),
        "report_path": str(report_path),
        "copied_report_path": copied_report,
        "evidence_json_path": str(evidence_json_path),
        "evidence_markdown_path": str(evidence_md_path),
        "frame_evidence_count": len(evidence["frames"]),
        "source_artifact_count": source_artifacts["available_count"],
        "timeline_items": len(timeline),
        "write": write,
    }
    if write:
        sources = bundle_manifest.get("sources") if isinstance(bundle_manifest.get("sources"), list) else []
        sources = [source for source in sources if not _same_peepshow_source(source, output)]
        sources.append(source_entry)
        bundle_manifest["sources"] = sources
        bundle_manifest["peepshow_optional_evidence"] = {
            "schema": "peepshow_optional_evidence.v1",
            "output_dir": str(output),
            "evidence_json": str(evidence_json_path),
            "evidence_markdown": str(evidence_md_path),
            "source_artifacts": source_artifacts,
            "attached_at": now_iso(),
        }
        write_json(evidence_json_path, evidence)
        evidence_md_path.write_text(_render_peepshow_evidence_markdown(evidence), encoding="utf-8")
        source_index = build_source_artifact_index({**bundle_manifest, "bundle_dir": str(bundle), "timeline_path": str(timeline_path)})
        write_json(bundle / "source-artifacts.json", source_index)
        (bundle / "source-artifacts.md").write_text(render_source_artifact_index_markdown(source_index), encoding="utf-8")
        write_json(manifest_path, bundle_manifest)
    return result


def _bundle_peepshow_evidence(manifest: dict, output_dir: Path, *, metadata: VideoMetadata | None = None) -> dict:
    metadata = metadata or _metadata_from_manifest(manifest)
    frames = manifest.get("frames") if isinstance(manifest.get("frames"), list) else []
    per_frame = ((manifest.get("analysis") or {}).get("perFrame") if isinstance(manifest.get("analysis"), dict) else None) or []
    if not isinstance(per_frame, list):
        per_frame = []
    rows = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        midpoint = _frame_midpoint(frame, index, len(frames), metadata.duration_seconds)
        analysis = per_frame[index] if index < len(per_frame) and isinstance(per_frame[index], dict) else {}
        rows.append(
            {
                "index": index + 1,
                "timestamp_seconds": midpoint,
                "frame_path": _frame_path(frame, output_dir),
                "ocr_text": _ocr_text(frame),
                "analysis": analysis,
                "tags": _tags_from_frame(frame, analysis),
                "transcript_excerpt": _transcript_excerpt(manifest, max(0.0, midpoint - 0.5), midpoint + 0.5),
            }
        )
    return {
        "schema": "peepshow_source_evidence.v1",
        "tool": "peepshow",
        "source_dir": str(output_dir),
        "video": dataclass_to_dict(metadata),
        "summary": (manifest.get("analysis") or {}).get("summary", "") if isinstance(manifest.get("analysis"), dict) else "",
        "frames": rows,
        "notes": [
            "This is source evidence for routing, OCR import, multimodal analysis, and review.",
            "It is not final timeline fusion and does not overwrite visual_understanding.",
        ],
    }


def _tags_from_frame(frame: dict, analysis: dict) -> list[str]:
    values: list[str] = []
    for source in (frame, analysis):
        raw = source.get("tags") if isinstance(source, dict) else None
        if isinstance(raw, str):
            values.extend(part.strip() for part in raw.split(",") if part.strip())
        elif isinstance(raw, list):
            values.extend(str(item).strip() for item in raw if str(item).strip())
    return list(dict.fromkeys(values))


def _same_peepshow_source(source: object, output: Path) -> bool:
    if not isinstance(source, dict):
        return False
    source_artifacts = source.get("source_artifacts") if isinstance(source.get("source_artifacts"), dict) else {}
    return source_artifacts.get("tool") == "peepshow" and Path(str(source_artifacts.get("source_dir") or "")).resolve() == output


def _render_peepshow_evidence_markdown(evidence: dict) -> str:
    video = evidence.get("video") if isinstance(evidence.get("video"), dict) else {}
    lines = [
        "# Peepshow Source Evidence",
        "",
        f"- Source dir: `{evidence.get('source_dir', '')}`",
        f"- Video: {video.get('title', '')}",
        f"- Frames: {len(evidence.get('frames') or [])}",
        "",
        "Peepshow evidence is preserved as source material. It does not replace visual routing, document screenshot parsing, multimodal frame analysis, temporal analysis, or human review.",
        "",
        "## Summary",
        "",
        str(evidence.get("summary") or "（无 Peepshow summary。）"),
        "",
        "## Frames",
        "",
    ]
    for frame in evidence.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        lines.extend(
            [
                f"### {frame.get('index')}. {frame.get('timestamp_seconds', 0):.3f}s",
                "",
                f"- Frame: `{frame.get('frame_path', '')}`",
                f"- Tags: {', '.join(frame.get('tags') or []) or 'none'}",
                f"- Transcript: {frame.get('transcript_excerpt') or 'none'}",
                f"- OCR: {frame.get('ocr_text') or 'none'}",
                f"- Analysis: {json.dumps(frame.get('analysis') or {}, ensure_ascii=False)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _metadata_from_manifest(manifest: dict) -> VideoMetadata:
    video = manifest.get("video", {})
    input_info = manifest.get("input", {})
    path = str(input_info.get("originalPath") or input_info.get("url") or "")
    title = str(input_info.get("filename") or Path(path).name or "peepshow-video")
    return VideoMetadata(
        id=new_id("video"),
        path=path,
        title=title,
        duration_seconds=float(video.get("durationSeconds") or 0),
        width=_optional_int(video.get("width")),
        height=_optional_int(video.get("height")),
        fps=_optional_float(video.get("fps")),
    )


def _segments_from_manifest(metadata: VideoMetadata, manifest: dict) -> list[EvidenceSegment]:
    frames = manifest.get("frames", [])
    if not isinstance(frames, list):
        return []

    duration = metadata.duration_seconds
    count = len(frames)
    output_dir = Path(str(manifest.get("outputDir") or ""))
    segments = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        midpoint = _frame_midpoint(frame, index, count, duration)
        start = max(0.0, midpoint - 0.5)
        end = min(duration, midpoint + 0.5) if duration else midpoint + 0.5
        ocr_text = _ocr_text(frame)
        signals = [f"peepshow_{manifest.get('strategy', 'frame')}"]
        if ocr_text:
            signals.append("peepshow_ocr")
        frame_path = _frame_path(frame, output_dir)
        segments.append(
            EvidenceSegment(
                id=new_id("segment"),
                video_id=metadata.id,
                start=start,
                end=end,
                midpoint=midpoint,
                signals=signals,
                transcript_excerpt=_transcript_excerpt(manifest, start, end),
                frame_paths=[frame_path] if frame_path else [],
                visual_observation="",
                uncertainty=_timing_uncertainty(frame),
            )
        )
    return segments


def _frame_midpoint(frame: dict, index: int, count: int, duration: float) -> float:
    for key in ("timestampSeconds", "timestamp_seconds", "timeSeconds", "time_seconds", "ptsTime", "pts_time", "time", "second"):
        value = _optional_float(frame.get(key))
        if value is not None:
            return max(0.0, value)
    for key in ("timestampMs", "timestamp_ms", "timeMs", "time_ms", "ptsMs", "pts_ms", "millis", "ms"):
        value = _optional_float(frame.get(key))
        if value is not None:
            return max(0.0, value / 1000)
    return _midpoint_for_index(index, count, duration)


def _midpoint_for_index(index: int, count: int, duration: float) -> float:
    if count <= 1 or duration <= 0:
        return 0.0
    return duration * index / (count - 1)


def _ocr_text(frame: dict) -> str:
    ocr = frame.get("ocr")
    if isinstance(ocr, str):
        return ocr.strip()
    if isinstance(ocr, list):
        return _join_text_values(ocr)
    if not isinstance(ocr, dict):
        return ""
    return _join_text_values(
        [
            ocr.get("text"),
            ocr.get("markdown"),
            ocr.get("content"),
            ocr.get("plainText"),
            ocr.get("plain_text"),
            ocr.get("fullText"),
            ocr.get("full_text"),
            ocr.get("lines"),
            ocr.get("blocks"),
            ocr.get("regions"),
            ocr.get("words"),
        ]
    )


def _join_text_values(values: object) -> str:
    texts: list[str] = []

    def visit(value: object) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                texts.append(text)
            return
        if isinstance(value, dict):
            for key in ("text", "markdown", "content", "value", "label"):
                if key in value:
                    visit(value.get(key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item)

    visit(values)
    return "\n".join(dict.fromkeys(texts))


def _frame_path(frame: dict, output_dir: Path) -> str:
    raw = frame.get("path") or frame.get("file") or frame.get("filename") or frame.get("image") or frame.get("imagePath")
    if not raw:
        return ""
    path = Path(str(raw))
    if path.is_absolute() or not str(output_dir):
        return str(path)
    return str(output_dir / path)


def _timing_uncertainty(frame: dict) -> str:
    timestamp_keys = (
        "timestampSeconds",
        "timestamp_seconds",
        "timeSeconds",
        "time_seconds",
        "ptsTime",
        "pts_time",
        "time",
        "second",
        "timestampMs",
        "timestamp_ms",
        "timeMs",
        "time_ms",
        "ptsMs",
        "pts_ms",
        "millis",
        "ms",
    )
    if any(frame.get(key) is not None for key in timestamp_keys):
        return "imported from peepshow with frame timestamps; frame content requires routed OCR/multimodal analysis"
    return "imported from peepshow; frame timing is estimated from frame order; frame content requires routed OCR/multimodal analysis"


def _transcript_excerpt(manifest: dict, start: float, end: float) -> str:
    transcript = manifest.get("audio", {}).get("transcript")
    if not isinstance(transcript, dict):
        return ""
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        return str(transcript.get("text", "")).strip()
    texts = []
    for cue in segments:
        if not isinstance(cue, dict):
            continue
        cue_start = float(cue.get("start") or 0)
        cue_end = float(cue.get("end") or cue_start)
        if cue_end >= start and cue_start <= end:
            text = str(cue.get("text", "")).strip()
            if text:
                texts.append(text)
    return " ".join(texts)


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
