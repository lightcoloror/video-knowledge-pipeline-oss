from __future__ import annotations

from pathlib import Path
from typing import Any

from .knowledge_coverage import audit_knowledge_coverage
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json


SCHEMA = "video_knowledge_pipeline.tile_result_merge.v1"
INPUT_SCHEMA = "video_knowledge_pipeline.tile_result_import.v1"
LOW_CONFIDENCE_THRESHOLD = 0.65


def run_tile_result_merge(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    execute: bool = False,
    min_confidence: float = LOW_CONFIDENCE_THRESHOLD,
    write: bool = True,
) -> dict[str, Any]:
    """Merge high-res tile OCR/VLM/human results back into bundle review data.

    Preview by default. Only writes timeline fields when execute=True. Empty,
    wrapper-only, or low-confidence results become review targets and do not
    clear OCR/screen-text blockers.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"bundle missing timeline.json: {root}")
    manifest = _read_object(manifest_path)
    timeline_data = read_json(timeline_path)
    timeline = [item for item in timeline_data if isinstance(item, dict)] if isinstance(timeline_data, list) else []
    input_payload = _read_input(input_json) if input_json else _default_preview_payload(root)
    results = _normalise_tile_results(input_payload)
    by_index = {_int(item.get("index")): item for item in timeline if item.get("index") is not None}
    min_confidence = max(0.0, min(1.0, float(min_confidence or LOW_CONFIDENCE_THRESHOLD)))

    updates: list[dict[str, Any]] = []
    review_targets: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for result in results:
        index = _first_int(result.get("index"), result.get("timeline_index"))
        item = by_index.get(index)
        if not item:
            skipped.append({"index": index, "reason": "timeline_index_not_found", "tile_id": result.get("tile_id", "")})
            continue
        decision = _classify_result(result, min_confidence=min_confidence)
        update = _build_update(item, result, decision)
        if decision["action"] == "merge":
            updates.append(update)
        else:
            review_targets.append(update)

    result_payload = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "created_at": now_iso(),
        "input_json": str(Path(input_json).expanduser().resolve()) if input_json else "",
        "execute": bool(execute),
        "min_confidence": min_confidence,
        "summary": {
            "input_results": len(results),
            "updates": len(updates),
            "review_targets": len(review_targets),
            "skipped": len(skipped),
        },
        "updates": updates,
        "review_targets": review_targets,
        "skipped": skipped,
        "operator_boundary": {
            "timeline_writeback": "execute_only",
            "no_cloud_call": True,
            "no_ocr_success_claim_for_empty_or_low_confidence": True,
            "purpose": "Consume high-res tile OCR/VLM/human results produced by local or approved external tools.",
        },
    }
    json_path = root / "tile-result-merge.json"
    report_path = root / "tile-result-merge.md"
    args_path = root / "mcp-tile-result-merge.args.json"
    template_path = root / "tile-result-import.template.json"
    result_payload.update({"json_path": str(json_path), "report_path": str(report_path), "mcp_args_path": str(args_path), "input_template_path": str(template_path)})

    if execute:
        _apply_updates(timeline, updates, review_targets)

    if write:
        with bundle_write_lock(root, operation="tile_result_merge"):
            if execute:
                write_json(timeline_path, timeline)
                manifest["tile_result_merge_last_applied_at"] = result_payload["created_at"]
            write_json(json_path, result_payload)
            report_path.write_text(render_tile_result_merge_markdown(result_payload), encoding="utf-8")
            write_json(args_path, {"bundle_dir": str(root), "input_json": str(input_json or ""), "execute": False, "min_confidence": min_confidence, "write": True})
            if not template_path.exists():
                write_json(template_path, _input_template(root))
            manifest["tile_result_merge"] = {
                "schema": SCHEMA,
                "last_run_at": result_payload["created_at"],
                "json_path": "tile-result-merge.json",
                "report_path": "tile-result-merge.md",
                "input_template_path": "tile-result-import.template.json",
                "mcp_args_path": "mcp-tile-result-merge.args.json",
                "summary": result_payload["summary"],
            }
            manifest["mcp_tile_result_merge_args"] = "mcp-tile-result-merge.args.json"
            write_json(manifest_path, manifest)
            if execute:
                audit_knowledge_coverage(root, write=True)
            _register_run(root, result_payload)
    return result_payload


def render_tile_result_merge_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        "# Tile Result Merge",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Execute: `{result.get('execute')}`",
        f"- Input JSON: `{result.get('input_json', '')}`",
        f"- Min confidence: `{result.get('min_confidence', '')}`",
        f"- Updates: `{summary.get('updates', 0)}`",
        f"- Review targets: `{summary.get('review_targets', 0)}`",
        f"- Skipped: `{summary.get('skipped', 0)}`",
        "",
        "This step merges high-res tile OCR/VLM/human results only when the result is meaningful and confident enough. Low-confidence, empty, or wrapper-only outputs are kept as review evidence and do not clear OCR blockers.",
        "",
        "## Updates",
        "",
        "| Index | Tile | Confidence | Text chars | Structured | Action | Evidence |",
        "| ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in result.get("updates") or []:
        if isinstance(item, dict):
            lines.append(_merge_row(item))
    if not result.get("updates"):
        lines.append("| - | - | - | - | - | - | - |")
    lines.extend(["", "## Review Targets", "", "| Index | Tile | Reason | Confidence | Evidence |", "| ---: | --- | --- | ---: | --- |"])
    for item in result.get("review_targets") or []:
        if isinstance(item, dict):
            lines.append(
                "| {index} | `{tile}` | {reason} | {confidence} | `{evidence}` |".format(
                    index=item.get("index", ""),
                    tile=_md(str(item.get("tile_id") or "")),
                    reason=_md(", ".join(str(value) for value in item.get("reasons") or [])),
                    confidence=item.get("confidence", ""),
                    evidence=_md(str(item.get("evidence_path") or "")),
                )
            )
    if not result.get("review_targets"):
        lines.append("| - | - | - | - | - |")
    return "\n".join(lines).rstrip() + "\n"


def _merge_row(item: dict[str, Any]) -> str:
    return "| {index} | `{tile}` | {confidence} | {chars} | {structured} | `{action}` | `{evidence}` |".format(
        index=item.get("index", ""),
        tile=_md(str(item.get("tile_id") or "")),
        confidence=item.get("confidence", ""),
        chars=len(str(item.get("text") or "")),
        structured="yes" if item.get("structured_visual") else "no",
        action=item.get("action", ""),
        evidence=_md(str(item.get("evidence_path") or "")),
    )


def _apply_updates(timeline: list[dict[str, Any]], updates: list[dict[str, Any]], review_targets: list[dict[str, Any]]) -> None:
    by_index = {_int(item.get("index")): item for item in timeline if item.get("index") is not None}
    for update in updates:
        item = by_index.get(_int(update.get("index")))
        if not item:
            continue
        tile_merges = item.setdefault("tile_result_merges", [])
        if isinstance(tile_merges, list):
            tile_merges.append(_timeline_merge_record(update))
        text = str(update.get("text") or "").strip()
        if text:
            item["visual_text"] = _append_text(str(item.get("visual_text") or ""), _tile_markdown(update))
        structured = update.get("structured_visual")
        if isinstance(structured, dict) and structured:
            values = item.setdefault("structured_visual", [])
            if isinstance(values, list):
                values.append(structured)
        issues = [str(issue) for issue in item.get("quality_issues") or []]
        if text or structured:
            removable = {"missing_visual_text", "visual_text_empty", "ocr_text_empty", "ocr_wrapper_only", "structured_visual_without_structure"}
            item["quality_issues"] = [issue for issue in issues if issue not in removable]
    for target in review_targets:
        item = by_index.get(_int(target.get("index")))
        if not item:
            continue
        pending = item.setdefault("tile_review_targets", [])
        if isinstance(pending, list):
            pending.append(_timeline_merge_record(target))
        issues = [str(issue) for issue in item.get("quality_issues") or []]
        for issue in ("tile_result_needs_review", "missing_visual_text"):
            if issue not in issues:
                issues.append(issue)
        item["quality_issues"] = issues
        item["needs_human_review"] = True


def _build_update(item: dict[str, Any], result: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    structured = _normalise_structured(result, item)
    text = str(result.get("text") or result.get("visual_text") or result.get("markdown") or "").strip()
    evidence = str(result.get("evidence_path") or result.get("tile_path") or result.get("output_path") or "").strip()
    return {
        "index": _first_int(result.get("index"), result.get("timeline_index"), item.get("index")),
        "tile_id": str(result.get("tile_id") or ""),
        "action": decision.get("action"),
        "reasons": decision.get("reasons") or [],
        "confidence": _confidence(result),
        "text": text,
        "structured_visual": structured,
        "evidence_path": evidence,
        "source": str(result.get("source") or result.get("provider") or "tile_result_import"),
        "applied_at": now_iso(),
    }


def _classify_result(result: dict[str, Any], *, min_confidence: float) -> dict[str, Any]:
    reasons: list[str] = []
    confidence = _confidence(result)
    text = str(result.get("text") or result.get("visual_text") or result.get("markdown") or "").strip()
    structured = result.get("structured_visual")
    if confidence is not None and confidence < min_confidence:
        reasons.append("tile_result_low_confidence")
    if not _meaningful_text(text) and not _has_structured_payload(structured):
        reasons.append("tile_result_empty")
    if _wrapper_only(text):
        reasons.append("tile_result_wrapper_only")
    status = str(result.get("status") or "").strip().lower()
    if status in {"failed", "error", "parse_failed"}:
        reasons.append(status)
    return {"action": "review" if reasons else "merge", "reasons": reasons}


def _normalise_structured(result: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    value = result.get("structured_visual")
    if isinstance(value, dict):
        structured = dict(value)
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        structured = dict(value[0])
    else:
        structured = {}
    text = str(result.get("text") or result.get("visual_text") or result.get("markdown") or "").strip()
    if text and not structured:
        structured = {"type": "tile_text", "markdown": text}
    if structured:
        structured.setdefault("type", "tile_text")
        structured.setdefault("source", "high_res_tile")
        structured.setdefault("timeline_index", _int(item.get("index")))
        if result.get("tile_id"):
            structured.setdefault("tile_id", str(result.get("tile_id")))
        evidence = str(result.get("evidence_path") or result.get("tile_path") or result.get("output_path") or "").strip()
        if evidence:
            structured.setdefault("evidence_path", evidence)
    return structured


def _timeline_merge_record(update: dict[str, Any]) -> dict[str, Any]:
    return {
        "tile_id": update.get("tile_id", ""),
        "action": update.get("action", ""),
        "reasons": update.get("reasons") or [],
        "confidence": update.get("confidence"),
        "evidence_path": update.get("evidence_path", ""),
        "source": update.get("source", ""),
        "applied_at": update.get("applied_at") or now_iso(),
    }


def _tile_markdown(update: dict[str, Any]) -> str:
    text = str(update.get("text") or "").strip()
    evidence = str(update.get("evidence_path") or "").strip()
    tile_id = str(update.get("tile_id") or "").strip()
    lines = ["", "## High-res tile result"]
    if tile_id:
        lines.append(f"- Tile: `{tile_id}`")
    if evidence:
        lines.append(f"- Evidence: `{evidence}`")
    lines.extend(["", text])
    return "\n".join(lines).strip()


def _append_text(current: str, addition: str) -> str:
    current = str(current or "").strip()
    addition = str(addition or "").strip()
    if not current:
        return addition
    if addition in current:
        return current
    return current + "\n\n" + addition


def _read_input(input_json: str | Path) -> Any:
    path = Path(input_json).expanduser().resolve()
    return read_json(path)


def _normalise_tile_results(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("tile_results", "results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _default_preview_payload(root: Path) -> dict[str, Any]:
    plan_path = root / "high-res-tile-plan.json"
    items: list[dict[str, Any]] = []
    if plan_path.exists():
        try:
            plan = read_json(plan_path)
            for item in plan.get("items") or []:
                if not isinstance(item, dict):
                    continue
                for tile in item.get("tiles") or []:
                    if not isinstance(tile, dict):
                        continue
                    items.append(
                        {
                            "timeline_index": item.get("index"),
                            "tile_id": tile.get("tile_id", ""),
                            "tile_path": tile.get("output_path") or tile.get("planned_output") or "",
                            "status": "pending_result",
                            "confidence": 0.0,
                            "text": "",
                        }
                    )
        except Exception:
            items = []
    return {"schema": INPUT_SCHEMA, "tile_results": items}


def _input_template(root: Path) -> dict[str, Any]:
    return {
        "schema": INPUT_SCHEMA,
        "bundle_dir": str(root),
        "tile_results": [
            {
                "timeline_index": 1,
                "tile_id": "0001-01",
                "tile_path": "high-res-tiles/timeline-0001/tile-01.jpg",
                "source": "human|local_ocr|local_vlm|approved_cloud_vlm",
                "status": "ok",
                "confidence": 0.9,
                "text": "识别出的 tile 小字、表格单元格、代码或界面标签。",
                "structured_visual": {"type": "tile_text", "markdown": "...", "evidence_path": "..."},
            }
        ],
    }


def _register_run(root: Path, result: dict[str, Any]) -> None:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    status = "completed" if result.get("execute") else "needs_execution"
    if summary.get("review_targets") and not summary.get("updates"):
        status = "needs_review"
    register_bundle_run(
        root,
        run_type="tile_result_merge",
        run_id="tile-result-merge",
        status=status,
        title="Tile result merge",
        summary=f"Prepared {summary.get('updates', 0)} tile updates and {summary.get('review_targets', 0)} review targets.",
        inputs={"input_json": result.get("input_json", "")},
        parameters={"execute": result.get("execute"), "min_confidence": result.get("min_confidence")},
        artifacts=[
            {"key": "tile_result_merge_json", "path": result.get("json_path", "")},
            {"key": "tile_result_merge_report", "path": result.get("report_path", "")},
            {"key": "input_template", "path": result.get("input_template_path", "")},
            {"key": "mcp_args", "path": result.get("mcp_args_path", "")},
        ],
        failed_items=_merge_failed_items(root, result),
        retry_command=f".\\scripts\\video-knowledge.ps1 tile-result-merge '{root}' --input-json <tile-result-import.json> --execute",
        next_actions=_next_actions(status),
        operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {},
        write=True,
    )


def _merge_failed_items(root: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    import_json = str(result.get("input_json") or "<tile-result-import.json>")
    for item in result.get("review_targets") or []:
        if not isinstance(item, dict):
            continue
        reasons = item.get("reasons") if isinstance(item.get("reasons"), list) else []
        evidence = str(item.get("evidence_path") or item.get("tile_path") or "").strip()
        row = {
            "index": item.get("index"),
            "reason": ",".join(str(reason) for reason in reasons),
            "detail": "Tile result needs review before it can clear screen-text/OCR blockers.",
            "tile_id": item.get("tile_id", ""),
            "confidence": item.get("confidence"),
            "suggested_next_tool": "prepare_review_session",
            "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 tile-result-merge '{root}' --input-json {import_json} --execute",
            "tile_result_merge_command": f".\\scripts\\video-knowledge.ps1 tile-result-merge '{root}' --input-json {import_json} --execute",
            "tile_result_import_command": f".\\scripts\\video-knowledge.ps1 tile-result-import-build '{root}' --results-dir <tile-results-dir>",
            "review_command": f".\\scripts\\video-knowledge.ps1 prepare-review-session '{root}' --limit 0 --group-by reason",
            "evidence_paths": [evidence] if evidence else [],
        }
        rows.append(row)
    return rows


def _next_actions(status: str) -> list[str]:
    if status == "needs_execution":
        return ["Fill tile-result-import.template.json with OCR/VLM/human tile results, then rerun tile-result-merge --execute."]
    if status == "needs_review":
        return ["Open tile-result-merge.md and review low-confidence or empty tile outputs before applying."]
    return ["Refresh review session and knowledge export after tile result merge."]


def _confidence(result: dict[str, Any]) -> float | None:
    value = result.get("confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _meaningful_text(value: str) -> bool:
    text = str(value or "").strip()
    if len(text) < 2:
        return False
    wrappers = {"ok", "none", "n/a", "无", "空", "未识别", "no text"}
    return text.lower() not in wrappers


def _wrapper_only(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    markers = ["<!-- source:", "![source image]", "```", "# tile", "# image"]
    if any(marker in text for marker in markers) and len(text) < 120:
        return True
    return False


def _has_structured_payload(value: Any) -> bool:
    if isinstance(value, dict):
        return any(v not in (None, "", [], {}) for v in value.values())
    if isinstance(value, list):
        return any(_has_structured_payload(item) for item in value)
    return False


def _read_object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _first_int(*values: Any) -> int:
    for value in values:
        if value is None or value == "":
            continue
        return _int(value)
    return 0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _md(value: str) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
