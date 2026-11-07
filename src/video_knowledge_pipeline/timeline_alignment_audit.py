from __future__ import annotations

from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .markdown_text import markdown_table_cell as _md
from .models import TranscriptCue, now_iso
from .powershell import quote_powershell_literal as _ps_quote
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .transcript import format_timestamp, parse_transcript, transcript_excerpt

SCHEMA = "video_knowledge_pipeline.timeline_alignment_audit.v1"


def timeline_alignment_audit(
    bundle_dir: str | Path,
    *,
    tolerance_seconds: float = 2.0,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"bundle missing timeline.json: {root}")
    manifest = _read_object(manifest_path)
    timeline_value = read_json(timeline_path)
    timeline = [item for item in timeline_value if isinstance(item, dict)] if isinstance(timeline_value, list) else []
    transcript_path = _transcript_path(root, manifest)
    transcript_sha256 = (
        sha256_file(transcript_path)
        if transcript_path is not None and transcript_path.is_file()
        else ""
    )
    timeline_sha256 = sha256_file(timeline_path)
    cues = _read_cues(transcript_path)
    tolerance = max(0.0, float(tolerance_seconds or 0))
    items = [_audit_item(item, cues=cues, tolerance=tolerance) for item in timeline]
    summary = _summary(items, cues=cues, transcript_path=transcript_path)
    json_path = root / "timeline-alignment-audit.json"
    md_path = root / "timeline-alignment-audit.md"
    args_path = root / "mcp-timeline-alignment-audit.args.json"
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "generated_at": now_iso(),
        "tolerance_seconds": tolerance,
        "transcript_path": str(transcript_path) if transcript_path else "",
        "transcript_sha256": transcript_sha256,
        "timeline_sha256": timeline_sha256,
        "summary": summary,
        "items": items,
        "json_path": str(json_path),
        "report_path": str(md_path),
        "mcp_args_path": str(args_path),
        "operator_boundary": {
            "timeline_writeback": "none",
            "purpose": "Audit ASR segment start/end, frame time, tagger time, and review_start alignment.",
        },
    }
    if write:
        write_json(json_path, result)
        md_path.write_text(render_timeline_alignment_audit_markdown(result), encoding="utf-8")
        write_json(args_path, {"bundle_dir": str(root), "tolerance_seconds": tolerance, "write": True})
        manifest["timeline_alignment_audit_json"] = "timeline-alignment-audit.json"
        manifest["timeline_alignment_audit_report"] = "timeline-alignment-audit.md"
        manifest["mcp_timeline_alignment_audit_args"] = "mcp-timeline-alignment-audit.args.json"
        write_json(manifest_path, manifest)
        _register_run(root, result)
    return result


def resolve_canonical_transcript_path(
    bundle_dir: str | Path,
    manifest: dict[str, Any] | None = None,
) -> Path | None:
    """Resolve the transcript currently bound to a Bundle without modifying it."""

    root = Path(bundle_dir).expanduser().resolve()
    current_manifest = manifest
    if current_manifest is None:
        manifest_path = root / "manifest.json"
        current_manifest = _read_object(manifest_path) if manifest_path.is_file() else {}
    return _transcript_path(root, current_manifest)


def timeline_alignment_audit_is_current(
    bundle_dir: str | Path,
    payload: dict[str, Any] | None = None,
) -> bool:
    """Fail closed unless an audit names the exact timeline and transcript bytes."""

    root = Path(bundle_dir).expanduser().resolve()
    timeline_path = root / "timeline.json"
    transcript_path = resolve_canonical_transcript_path(root)
    if not timeline_path.is_file() or transcript_path is None or not transcript_path.is_file():
        return False
    current = payload
    if current is None:
        audit_path = root / "timeline-alignment-audit.json"
        if not audit_path.is_file():
            return False
        try:
            current = _read_object(audit_path)
        except (OSError, ValueError):
            return False
    try:
        recorded_transcript = Path(str(current.get("transcript_path") or "")).resolve()
        return (
            recorded_transcript == transcript_path.resolve()
            and str(current.get("transcript_sha256") or "")
            == sha256_file(transcript_path)
            and str(current.get("timeline_sha256") or "") == sha256_file(timeline_path)
        )
    except (OSError, ValueError):
        return False


def render_timeline_alignment_audit_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        "# Timeline Alignment Audit",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Transcript: `{result.get('transcript_path', '')}`",
        f"- Tolerance seconds: `{result.get('tolerance_seconds', 0)}`",
        f"- Items: `{summary.get('items', 0)}`",
        f"- Items with issues: `{summary.get('items_with_issues', 0)}`",
        f"- Missing ASR overlap: `{summary.get('missing_asr_overlap', 0)}`",
        f"- Review start mismatches: `{summary.get('review_start_mismatch', 0)}`",
        f"- Tagger time conflicts: `{summary.get('tagger_time_conflict', 0)}`",
        "",
        "## Issue Counts",
        "",
        "| Issue | Count |",
        "| --- | ---: |",
    ]
    issue_counts = summary.get("issue_counts") if isinstance(summary.get("issue_counts"), dict) else {}
    if issue_counts:
        for key, value in sorted(issue_counts.items()):
            lines.append(f"| `{key}` | {value} |")
    else:
        lines.append("| - | 0 |")
    lines.extend(["", "## Items With Issues", "", "| Index | Segment | Review Start | ASR Start | Frame Time | Tagger Times | Issues |", "| ---: | --- | --- | --- | --- | --- | --- |"])
    for item in result.get("items") or []:
        if not isinstance(item, dict) or not item.get("issues"):
            continue
        lines.append(
            "| {index} | `{segment}` | `{review}` | `{asr}` | `{frame}` | {tagger} | {issues} |".format(
                index=item.get("index", ""),
                segment=f"{_ts(item.get('start'))}-{_ts(item.get('end'))}",
                review=_ts(item.get("review_start")),
                asr=_ts(item.get("asr_first_start")),
                frame=_ts(item.get("frame_time")),
                tagger=_md(", ".join(_ts(value) for value in item.get("tagger_times") or [])),
                issues=_md(", ".join(str(issue) for issue in item.get("issues") or [])),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _audit_item(item: dict[str, Any], *, cues: list[TranscriptCue], tolerance: float) -> dict[str, Any]:
    start = _seconds(item.get("start"))
    end = _seconds(item.get("end"))
    if end < start:
        end = start
    frame_time = _first_seconds(item, ("frame_time", "frame_time_seconds", "midpoint"))
    review_start = _first_seconds(item, ("review_start", "transcript_start", "subtitle_start", "asr_start"))
    overlapping = [cue for cue in cues if cue.end >= start and cue.start <= end]
    asr_first_start = overlapping[0].start if overlapping else None
    tagger_times = _tagger_times(item)
    issues: list[str] = []
    if cues and not overlapping:
        issues.append("missing_asr_overlap")
    if review_start is not None and end > start and (review_start < start - tolerance or review_start > end + tolerance):
        issues.append("review_start_outside_segment")
    if asr_first_start is not None and review_start is not None and abs(review_start - asr_first_start) > tolerance:
        issues.append("review_start_mismatch")
    if frame_time is not None and end > start and (frame_time < start - tolerance or frame_time > end + tolerance):
        issues.append("frame_time_outside_segment")
    for value in tagger_times:
        if end > start and (value < start - tolerance or value > end + tolerance):
            issues.append("tagger_time_conflict")
            break
    return {
        "index": _int(item.get("index")),
        "start": start,
        "end": end,
        "frame_time": frame_time,
        "review_start": review_start,
        "review_start_source": str(item.get("review_start_source") or ""),
        "asr_first_start": asr_first_start,
        "asr_overlap_count": len(overlapping),
        "asr_excerpt": transcript_excerpt(overlapping, start, end)[:240] if overlapping else "",
        "tagger_times": tagger_times,
        "issues": _dedupe(issues),
    }


def _summary(items: list[dict[str, Any]], *, cues: list[TranscriptCue], transcript_path: Path | None) -> dict[str, Any]:
    issue_counts: dict[str, int] = {}
    for item in items:
        for issue in item.get("issues") or []:
            issue_counts[str(issue)] = issue_counts.get(str(issue), 0) + 1
    return {
        "items": len(items),
        "transcript_cues": len(cues),
        "transcript_available": bool(transcript_path and cues),
        "items_with_issues": sum(1 for item in items if item.get("issues")),
        "missing_asr_overlap": issue_counts.get("missing_asr_overlap", 0),
        "review_start_mismatch": issue_counts.get("review_start_mismatch", 0),
        "tagger_time_conflict": issue_counts.get("tagger_time_conflict", 0),
        "issue_counts": dict(sorted(issue_counts.items())),
    }



def _register_run(root: Path, result: dict[str, Any]) -> None:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    issue_count = int(summary.get("items_with_issues") or 0)
    transcript_available = bool(summary.get("transcript_available"))
    if not transcript_available:
        status = "needs_input"
    elif issue_count:
        status = "needs_review"
    else:
        status = "completed"
    retry = f".\\scripts\\video-knowledge.ps1 timeline-alignment-audit {_ps_quote(str(root))} --tolerance-seconds {result.get('tolerance_seconds', 2)}"
    register_bundle_run(
        root,
        run_type="timeline_alignment_audit",
        run_id="timeline-alignment-audit",
        status=status,
        title="Timeline alignment audit",
        summary=f"Found {issue_count} timeline items with ASR/frame/tagger/review timestamp alignment issues.",
        inputs={"transcript_path": result.get("transcript_path", "")},
        parameters={"tolerance_seconds": result.get("tolerance_seconds", 2)},
        artifacts=[
            {"key": "timeline_alignment_audit_json", "path": result.get("json_path", "")},
            {"key": "timeline_alignment_audit_report", "path": result.get("report_path", "")},
            {"key": "mcp_args", "path": result.get("mcp_args_path", "")},
        ],
        failed_items=_alignment_failed_items(result),
        retry_command=retry if status != "completed" else "",
        next_actions=_alignment_next_actions(status),
        operator_boundary={
            "local_only": True,
            "no_cloud_call": True,
            "no_timeline_writeback": True,
            "purpose": "Expose ASR/frame/tagger/review timestamp conflicts to task console and review workflow.",
        },
        write=True,
    )


def _alignment_failed_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for item in result.get("items") or []:
        if not isinstance(item, dict) or not item.get("issues"):
            continue
        failed.append(
            {
                "index": item.get("index", ""),
                "reason": ",".join(str(issue) for issue in item.get("issues") or []),
                "detail": f"segment={_ts(item.get('start'))}-{_ts(item.get('end'))}; review_start={_ts(item.get('review_start'))}; asr_start={_ts(item.get('asr_first_start'))}",
            }
        )
    return failed


def _alignment_next_actions(status: str) -> list[str]:
    if status == "completed":
        return ["No timeline alignment issues found under current tolerance."]
    if status == "needs_input":
        return ["Attach or generate normalized-transcript.json / corrected-transcript.json, then rerun timeline-alignment-audit."]
    return ["Open timeline-alignment-audit.md before manually editing review_start fields.", "Prefer ASR segment start as review_start when transcript overlap is reliable."]


def _transcript_path(root: Path, manifest: dict[str, Any]) -> Path | None:
    for key in ("corrected_transcript_json", "normalized_transcript_json", "transcript_json", "source_transcript", "transcript_path"):
        value = manifest.get(key)
        if value:
            path = _bundle_path(root, str(value))
            if path.exists():
                return path
    for path in (root / "corrected-transcript.json", root / "normalized-transcript.json", root / "transcript.json", root / "timeline-transcript.json"):
        if path.exists():
            return path
    return None


def _read_cues(path: Path | None) -> list[TranscriptCue]:
    if not path:
        return []
    try:
        return parse_transcript(path)
    except Exception:
        return []


def _tagger_times(item: dict[str, Any]) -> list[float]:
    rows: list[Any] = []
    for key in ("tagger_time_axis", "tagger_annotations"):
        value = item.get(key)
        if isinstance(value, list):
            rows.extend(value)
    integrated = item.get("integrated_visual") if isinstance(item.get("integrated_visual"), dict) else {}
    for key in ("tagger_time_axis", "tagger_annotations"):
        value = integrated.get(key)
        if isinstance(value, list):
            rows.extend(value)
    result: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _first_seconds(row, ("time", "start", "start_seconds", "seconds"))
        if value is not None:
            result.append(value)
    return _dedupe_floats(result)


def _first_seconds(item: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in item:
            value = _seconds_or_none(item.get(key))
            if value is not None:
                return value
    return None


def _seconds(value: Any) -> float:
    return _seconds_or_none(value) or 0.0


def _seconds_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _read_object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _dedupe_floats(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        if not any(abs(value - existing) < 0.001 for existing in result):
            result.append(value)
    return result


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _ts(value: Any) -> str:
    if value is None:
        return ""
    return format_timestamp(_seconds(value))
