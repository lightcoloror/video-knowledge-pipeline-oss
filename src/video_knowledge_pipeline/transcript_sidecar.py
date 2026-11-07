from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import write_json
from .transcript import format_timestamp, parse_transcript

TRANSCRIPT_KEYS = (
    "source_arbitrated_transcript_json",
    "source_arbitrated_transcript_srt",
    "human_corrected_transcript_json",
    "human_corrected_transcript_srt",
    "llm_corrected_transcript_json",
    "llm_corrected_transcript_srt",
    "corrected_transcript_json",
    "corrected_transcript_srt",
    "readable_transcript_json",
    "readable_transcript_srt",
    "normalized_transcript_json",
    "normalized_transcript_srt",
    "transcript_json",
    "transcript_srt",
    "source_transcript",
    "transcript_path",
)


def ensure_review_transcript_sidecar(
    bundle_dir: str | Path,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    *,
    title: str = "",
    project_root: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Ensure a review-friendly transcript JSON/SRT is available in the bundle.

    Corrected or normalized ASR files win. If they are not already in the bundle,
    this tries to copy them from the project workspace. Only when no real sidecar
    exists does it create a timeline-derived fallback.
    """
    root = Path(bundle_dir).expanduser().resolve()
    existing_real = _existing_transcript(root, manifest, include_timeline=False)
    if existing_real:
        return _record_existing(manifest, existing_real)

    source = _source_transcript(root, manifest, project_root=project_root)
    if source:
        return _copy_source_transcript(root, manifest, source, write=write)

    existing_any = _existing_transcript(root, manifest, include_timeline=True)
    if existing_any:
        return _record_existing(manifest, existing_any)

    return _create_timeline_sidecar(root, manifest, timeline, title=title, write=write)


def transcript_source_kind(root: str | Path, path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    bundle = Path(root).expanduser().resolve()
    try:
        name = resolved.relative_to(bundle).as_posix().lower()
    except ValueError:
        name = resolved.name.lower()
    if "source-arbitrated-transcript" in name:
        return "source_arbitrated"
    if "corrected-transcript" in name:
        return "corrected"
    if "readable-transcript" in name:
        return "readable"
    if "timeline-transcript" in name:
        return "timeline_fallback"
    return "asr"


def _record_existing(manifest: dict[str, Any], existing: tuple[str, Path]) -> dict[str, Any]:
    result = {
        "schema": "video_knowledge_pipeline.review_transcript_sidecar.v1",
        "status": "existing",
        "source": existing[0],
        "path": str(existing[1]),
        "updated_at": now_iso(),
    }
    manifest["review_transcript_sidecar"] = result
    return result


def _copy_source_transcript(root: Path, manifest: dict[str, Any], source: Path, *, write: bool) -> dict[str, Any]:
    kind = transcript_source_kind(root, source)
    if kind == "source_arbitrated":
        json_name = "source-arbitrated-transcript.json"
        srt_name = "source-arbitrated-transcript.srt"
        json_key = "source_arbitrated_transcript_json"
        srt_key = "source_arbitrated_transcript_srt"
        source_label = "source_arbitrated_transcript"
    elif kind == "corrected":
        json_name = "corrected-transcript.json"
        srt_name = "corrected-transcript.srt"
        json_key = "corrected_transcript_json"
        srt_key = "corrected_transcript_srt"
        source_label = "corrected_transcript"
    elif kind == "readable":
        json_name = "readable-transcript.json"
        srt_name = "readable-transcript.srt"
        json_key = "readable_transcript_json"
        srt_key = "readable_transcript_srt"
        source_label = "readable_transcript"
    else:
        json_name = "normalized-transcript.json"
        srt_name = "normalized-transcript.srt"
        json_key = "normalized_transcript_json"
        srt_key = "normalized_transcript_srt"
        source_label = "normalized_asr"

    json_dest = root / json_name
    srt_source = source.with_suffix(".srt")
    srt_dest = root / srt_name
    if write:
        if source.resolve() != json_dest.resolve():
            shutil.copy2(source, json_dest)
        if srt_source.exists() and srt_source.resolve() != srt_dest.resolve():
            shutil.copy2(srt_source, srt_dest)
    manifest[json_key] = json_name
    if srt_source.exists() or srt_dest.exists():
        manifest[srt_key] = srt_name
    # Keep generic keys for consumers that only know transcript_json/srt.
    manifest["transcript_json"] = json_name
    if srt_source.exists() or srt_dest.exists():
        manifest["transcript_srt"] = srt_name
    result = {
        "schema": "video_knowledge_pipeline.review_transcript_sidecar.v1",
        "status": "copied" if write else "copy_planned",
        "source": source_label,
        "source_path": str(source),
        "json_path": json_name,
        "srt_path": srt_name if srt_source.exists() or srt_dest.exists() else "",
        "segment_count": _count_segments(source),
        "updated_at": now_iso(),
    }
    manifest["review_transcript_sidecar"] = result
    return result


def _create_timeline_sidecar(root: Path, manifest: dict[str, Any], timeline: list[dict[str, Any]], *, title: str, write: bool) -> dict[str, Any]:
    segments = _segments_from_timeline(timeline)
    json_path = root / "timeline-transcript.json"
    srt_path = root / "timeline-transcript.srt"
    payload = {
        "schema": "video_knowledge_pipeline.timeline_transcript.v1",
        "title": title or str(manifest.get("title") or root.name),
        "provider": "timeline_fallback",
        "source": "timeline.json",
        "created_at": now_iso(),
        "segments": segments,
    }
    result = {
        "schema": "video_knowledge_pipeline.review_transcript_sidecar.v1",
        "status": "created" if segments else "empty",
        "source": "timeline_fallback",
        "json_path": "timeline-transcript.json",
        "srt_path": "timeline-transcript.srt",
        "segment_count": len(segments),
        "updated_at": payload["created_at"],
    }
    manifest["transcript_json"] = "timeline-transcript.json"
    manifest["transcript_srt"] = "timeline-transcript.srt"
    manifest["review_transcript_sidecar"] = result
    if write:
        write_json(json_path, payload)
        srt_path.write_text(_render_srt(segments), encoding="utf-8")
    return result


def _existing_transcript(root: Path, manifest: dict[str, Any], *, include_timeline: bool) -> tuple[str, Path] | None:
    priority_keys = (
        "source_arbitrated_transcript_json",
        "source_arbitrated_transcript_srt",
        "human_corrected_transcript_json",
        "human_corrected_transcript_srt",
        "llm_corrected_transcript_json",
        "llm_corrected_transcript_srt",
        "corrected_transcript_json",
        "corrected_transcript_srt",
        "readable_transcript_json",
        "readable_transcript_srt",
        "normalized_transcript_json",
        "normalized_transcript_srt",
        "source_transcript",
        "transcript_path",
        "transcript_json",
        "transcript_srt",
    )
    for key in priority_keys:
        value = manifest.get(key)
        if not value:
            continue
        path = _bundle_path(root, str(value))
        if not include_timeline and "timeline-transcript" in path.name.lower():
            continue
        if path.exists():
            return key, path
    for path in (
        root / "source-arbitrated-transcript.json",
        root / "source-arbitrated-transcript.srt",
        root / "corrected-transcript.json",
        root / "corrected-transcript.srt",
        root / "normalized-transcript.json",
        root / "normalized-transcript.srt",
        root / "transcript.json",
        root / "transcript.srt",
        root / "timeline-transcript.json",
        root / "timeline-transcript.srt",
    ):
        if not include_timeline and "timeline-transcript" in path.name.lower():
            continue
        if path.exists():
            return path.name, path
    return None


def _source_transcript(root: Path, manifest: dict[str, Any], *, project_root: str | Path | None) -> Path | None:
    projects = _project_roots(root, manifest, project_root=project_root)
    for project in projects:
        for path in _video_transcript_candidates(project, manifest):
            if path.exists() and path.is_file():
                return path.resolve()
        transcript_root = project / "transcripts"
        if transcript_root.exists():
            candidates = sorted(transcript_root.glob("**/normalized-transcript.json"), key=lambda item: item.stat().st_mtime, reverse=True)
            if candidates:
                return candidates[0].resolve()
    return None


def _project_roots(root: Path, manifest: dict[str, Any], *, project_root: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if project_root:
        roots.append(Path(project_root).expanduser().resolve())
    source_package = str(manifest.get("source_package") or "").strip()
    if source_package:
        package_path = _bundle_path(root, source_package)
        if package_path.name == "lecture-package.json" and package_path.parent.name == "lecture-packages":
            roots.append(package_path.parent.parent.resolve())
    # Common case for openclaw-runs/<run>/webui-bundle.
    if root.parent.name == "lecture-packages":
        roots.append(root.parent.parent.resolve())
    else:
        roots.append(root.parent.resolve())
    return _dedupe_paths(roots)


def _video_transcript_candidates(project: Path, manifest: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    sources = manifest.get("sources") if isinstance(manifest.get("sources"), list) else []
    for source in sources:
        if not isinstance(source, dict):
            continue
        video_id = str(source.get("video_id") or "").strip()
        if not video_id:
            continue
        video_dir = project / "videos" / video_id
        candidates.extend(
            [
                video_dir / "source-arbitrated-transcript.json",
                video_dir / "corrected-transcript.json",
                video_dir / "normalized-transcript.json",
                video_dir / "transcript.json",
            ]
        )
    return candidates


def _count_segments(path: Path) -> int:
    try:
        return len(parse_transcript(path))
    except Exception:
        return 0


def _segments_from_timeline(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("corrected_transcript") or item.get("transcript") or item.get("text") or "").strip()
        if not text:
            continue
        start = _seconds(item.get("start"))
        end = _seconds(item.get("end"))
        if end <= start:
            end = start
        rows.append(
            {
                "index": item.get("index") or position,
                "start": start,
                "end": end,
                "text": text,
                "source": "timeline",
            }
        )
    rows.sort(key=lambda row: (float(row.get("start") or 0), int(row.get("index") or 0)))
    return rows


def _render_srt(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = _srt_timestamp(_seconds(segment.get("start")))
        end = _srt_timestamp(max(_seconds(segment.get("start")), _seconds(segment.get("end"))))
        blocks.extend([str(index), f"{start} --> {end}", text, ""])
    return "\n".join(blocks).rstrip() + "\n" if blocks else ""


def _srt_timestamp(seconds: float) -> str:
    return format_timestamp(seconds).replace(".", ",")


def _seconds(value: Any) -> float:
    try:
        return max(0.0, float(value or 0))
    except Exception:
        return 0.0


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out
