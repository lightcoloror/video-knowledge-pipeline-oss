from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .visual_integration import integrated_visual
from .vision_review_triage import _load_tagger_annotations, _unique, TAGGER_PRODUCER_FIELDS

SCHEMA = "video_knowledge_pipeline.tagger_annotation_import.v1"


def import_tagger_annotations(
    bundle_dir: str | Path,
    tagger_json: str | Path,
    *,
    source: str = "qinglong",
    write: bool = True,
) -> dict[str, Any]:
    """Import Qinglong/manual tagger timeline annotations into bundle timeline."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must be a JSON array")

    tagger_path = Path(tagger_json).expanduser().resolve()
    annotations = _load_tagger_annotations(tagger_path, [item for item in timeline if isinstance(item, dict)])
    index_to_position = {_item_index(item, position): position - 1 for position, item in enumerate(timeline, start=1) if isinstance(item, dict)}
    updated_indexes: list[int] = []
    imported_rows: list[dict[str, Any]] = []
    from .general_tagger_adapter import validate_tagger_recognition
    for index, rows in annotations.items():
        for row in rows:
            validate_tagger_recognition(root,row,expected_index=index)

    for index, rows in sorted(annotations.items()):
        position = index_to_position.get(index)
        if position is None:
            continue
        item = timeline[position]
        if not isinstance(item, dict):
            continue
        normalised_rows = [_with_source(row, source=source) for row in rows]
        existing = item.get("tagger_annotations") if isinstance(item.get("tagger_annotations"), list) else []
        merged = _merge_annotations(existing, normalised_rows)
        item["tagger_annotations"] = merged
        item["tagger_tags"] = _unique([tag for row in merged for tag in row.get("tags", [])])
        _merge_item_tags(item, item["tagger_tags"])
        item["tagger_visual_summary"] = _join_text(row.get("text") for row in merged)
        item["tagger_time_axis"] = _time_axis_rows(merged)
        item["tagger_annotations_updated_at"] = now_iso()
        _mark_quality_from_tags(item, item["tagger_tags"])
        item["integrated_visual"] = integrated_visual(item)
        updated_indexes.append(index)
        imported_rows.extend({**row, "timeline_index": index} for row in normalised_rows)

    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "tagger_json": str(tagger_path),
        "source": source,
        "write": bool(write),
        "annotation_count": sum(len(rows) for rows in annotations.values()),
        "matched_indexes": sorted(annotations),
        "updated_indexes": updated_indexes,
        "updated_count": len(updated_indexes),
        "report_path": str(root / "tagger-import-report.md"),
        "json_path": str(root / "tagger-import.json"),
    }

    if write:
        with bundle_write_lock(root, operation="import_tagger_annotations", timeout_seconds=1.0):
            for index, rows in annotations.items():
                for row in rows:
                    validate_tagger_recognition(root,row,expected_index=index)
            write_json(timeline_path, timeline)
            write_json(root / "tagger-import.json", {**result, "imported_rows": imported_rows})
            (root / "tagger-import-report.md").write_text(_render_report(result, timeline), encoding="utf-8")
            write_json(root / "mcp-import-tagger-annotations.args.json", {"bundle_dir": str(root), "tagger_json": str(tagger_path), "source": source, "write": True})
            manifest["tagger_annotations"] = {
                "schema": SCHEMA,
                "source": source,
                "tagger_json": str(tagger_path),
                "annotation_count": result["annotation_count"],
                "updated_indexes": updated_indexes,
                "updated_at": now_iso(),
            }
            manifest["tagger_import_report"] = "tagger-import-report.md"
            manifest["tagger_import_json"] = "tagger-import.json"
            manifest["mcp_import_tagger_annotations_args"] = "mcp-import-tagger-annotations.args.json"
            write_json(manifest_path, manifest)
    return result


def _with_source(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.timeline_tagger_annotation.v1",
        **{key:row[key] for key in TAGGER_PRODUCER_FIELDS if key in row},
        "source": source,
        "tags": [str(tag) for tag in row.get("tags", []) if str(tag)],
        "text": str(row.get("text") or "").strip(),
        "priority": str(row.get("priority") or "").strip(),
        "raw_index": row.get("raw_index"),
        "start": row.get("start"),
        "end": row.get("end"),
        "time": row.get("time"),
        "source_path": row.get("source_path"),
        "model": str(row.get("model") or "").strip(),
        "model_revision": str(row.get("model_revision") or "").strip(),
        "artifact_path": str(row.get("artifact_path") or "").strip(),
        "artifact_sha256": str(row.get("artifact_sha256") or "").strip(),
        "tag_vocabulary": str(row.get("tag_vocabulary") or "").strip(),
        "labels_en": [str(value) for value in row.get("labels_en", []) if str(value)]
        if isinstance(row.get("labels_en"), list) else [],
        "labels_zh": [str(value) for value in row.get("labels_zh", []) if str(value)]
        if isinstance(row.get("labels_zh"), list) else [],
        "candidate_only": bool(row.get("candidate_only", True)),
        "human_review_required": bool(row.get("human_review_required", True)),
    }


def _merge_annotations(existing: list[Any], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in existing if isinstance(row, dict)]
    seen = {_annotation_key(row) for row in rows}
    for row in incoming:
        key = _annotation_key(row)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def _annotation_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("source") or ""),
        ",".join(str(tag) for tag in row.get("tags", [])),
        str(row.get("text") or ""),
        str(row.get("time") or row.get("start") or row.get("raw_index") or ""),
        str(row.get("model_revision") or ""),
        str(row.get("artifact_sha256") or ""),
    )


def _merge_item_tags(item: dict[str, Any], tags: list[str]) -> None:
    existing = item.get("tags") if isinstance(item.get("tags"), list) else []
    item["tags"] = _unique([*[str(tag) for tag in existing if str(tag)], *tags])


def _mark_quality_from_tags(item: dict[str, Any], tags: list[str]) -> None:
    joined = " ".join(tags)
    issues = item.get("quality_issues") if isinstance(item.get("quality_issues"), list) else []
    additions = []
    if any(token in joined for token in ["疑难", "易错", "复核"]):
        additions.append("tagger_marked_risky")
    if any(token in joined for token in ["工具名", "术语", "名称"]):
        additions.append("tagger_marked_term_sensitive")
    if any(token in joined for token in ["OCR", "屏幕文字", "表格", "公式", "代码"]):
        additions.append("tagger_marked_document_visual")
    item["quality_issues"] = _unique([*[str(issue) for issue in issues if str(issue)], *additions])


def _time_axis_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row.get("start") is not None or row.get("end") is not None or row.get("time") is not None:
            result.append({"start": row.get("start"), "end": row.get("end"), "time": row.get("time"), "source": row.get("source")})
    return result


def _join_text(values: Any) -> str:
    return "；".join(str(value).strip() for value in values if str(value or "").strip())


def _item_index(item: dict[str, Any], position: int) -> int:
    try:
        value = int(item.get("index") or 0)
    except Exception:
        value = 0
    return value or position


def _render_report(result: dict[str, Any], timeline: list[Any]) -> str:
    lines = [
        "# Tagger Annotation Import",
        "",
        f"- Source: `{result.get('source', '')}`",
        f"- Tagger JSON: `{result.get('tagger_json', '')}`",
        f"- Annotation count: `{result.get('annotation_count', 0)}`",
        f"- Updated timeline items: `{result.get('updated_count', 0)}`",
        "",
        "| Index | Tags | Text |",
        "| ---: | --- | --- |",
    ]
    updated = set(result.get("updated_indexes") or [])
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        index = _item_index(item, position)
        if index not in updated:
            continue
        lines.append(f"| {index} | {_md(', '.join(item.get('tagger_tags') or []))} | {_md(item.get('tagger_visual_summary') or '')} |")
    return "\n".join(lines).rstrip() + "\n"


def _md(value: Any) -> str:
    return str(value or "-").replace("\n", " ").replace("|", "\\|")
