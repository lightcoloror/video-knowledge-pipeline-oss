from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json
from .tile_result_merge import INPUT_SCHEMA


SCHEMA = "video_knowledge_pipeline.tile_result_import_builder.v1"


def build_tile_result_import(
    bundle_dir: str | Path,
    *,
    results_dir: str | Path | None = None,
    output_json: str | Path | None = None,
    default_source: str = "tile_result_import_builder",
    default_confidence: float = 0.0,
    write: bool = True,
) -> dict[str, Any]:
    """Build tile-result-import.json from a high-res tile plan and result files.

    This is a local glue step. It does not run OCR/VLM. It reads existing
    tile-level .json/.txt/.md outputs and normalises them into the import
    schema consumed by tile-result-merge.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    plan_path = root / "high-res-tile-plan.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    if not plan_path.exists():
        raise FileNotFoundError(f"bundle missing high-res-tile-plan.json: {root}")
    manifest = _read_object(manifest_path)
    plan = _read_object(plan_path)
    results_root = Path(results_dir).expanduser().resolve() if results_dir else None
    result_files = _index_result_files(results_root) if results_root else {}
    tile_results: list[dict[str, Any]] = []
    matched = 0
    pending = 0
    for item in plan.get("items") or []:
        if not isinstance(item, dict):
            continue
        index = _int(item.get("index"))
        for tile in item.get("tiles") or []:
            if not isinstance(tile, dict):
                continue
            row, is_matched = _tile_result_row(
                root=root,
                timeline_index=index,
                tile=tile,
                result_files=result_files,
                default_source=default_source,
                default_confidence=default_confidence,
            )
            tile_results.append(row)
            if is_matched:
                matched += 1
            else:
                pending += 1
    output_path = Path(output_json).expanduser().resolve() if output_json else root / "tile-result-import.json"
    report_path = root / "tile-result-import.md"
    args_path = root / "mcp-tile-result-import-build.args.json"
    payload = {
        "schema": INPUT_SCHEMA,
        "bundle_dir": str(root),
        "created_at": now_iso(),
        "source_schema": SCHEMA,
        "source_plan": str(plan_path),
        "results_dir": str(results_root) if results_root else "",
        "tile_results": tile_results,
    }
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "created_at": payload["created_at"],
        "plan_path": str(plan_path),
        "results_dir": str(results_root) if results_root else "",
        "output_json_path": str(output_path),
        "report_path": str(report_path),
        "mcp_args_path": str(args_path),
        "summary": {
            "tiles": len(tile_results),
            "matched_results": matched,
            "pending_results": pending,
            "result_files_indexed": len(result_files),
        },
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "no_ocr_or_vlm_execution": True,
            "purpose": "Normalise existing tile OCR/VLM/human outputs into tile-result-import.json.",
        },
        "pending_items": _pending_failed_items(tile_results, results_root=results_root),
    }
    if write:
        with bundle_write_lock(root, operation="tile_result_import_build"):
            write_json(output_path, payload)
            report_path.write_text(render_tile_result_import_markdown(result, payload), encoding="utf-8")
            write_json(
                args_path,
                {
                    "bundle_dir": str(root),
                    "results_dir": str(results_root) if results_root else "",
                    "output_json": str(output_path),
                    "default_source": default_source,
                    "default_confidence": default_confidence,
                    "write": True,
                },
            )
            manifest["tile_result_import_json"] = _relative_or_abs(root, output_path)
            manifest["tile_result_import_report"] = "tile-result-import.md"
            manifest["mcp_tile_result_import_build_args"] = "mcp-tile-result-import-build.args.json"
            write_json(manifest_path, manifest)
            _register_run(root, result)
    return {**result, "tile_results_preview": tile_results[:20]}


def render_tile_result_import_markdown(result: dict[str, Any], payload: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        "# Tile Result Import Build",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Plan: `{result.get('plan_path', '')}`",
        f"- Results dir: `{result.get('results_dir', '')}`",
        f"- Output JSON: `{result.get('output_json_path', '')}`",
        f"- Tiles: `{summary.get('tiles', 0)}`",
        f"- Matched results: `{summary.get('matched_results', 0)}`",
        f"- Pending results: `{summary.get('pending_results', 0)}`",
        "",
        "This step only normalises existing tile result files. It does not run OCR, VLM, or cloud APIs.",
        "",
        "## Tile Results",
        "",
        "| Index | Tile | Status | Confidence | Source | Result file | Text chars |",
        "| ---: | --- | --- | ---: | --- | --- | ---: |",
    ]
    for row in payload.get("tile_results") or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {index} | `{tile}` | `{status}` | {confidence} | `{source}` | `{file}` | {chars} |".format(
                index=row.get("timeline_index", ""),
                tile=_md(str(row.get("tile_id") or "")),
                status=_md(str(row.get("status") or "")),
                confidence=row.get("confidence", ""),
                source=_md(str(row.get("source") or "")),
                file=_md(str(row.get("result_file") or "")),
                chars=len(str(row.get("text") or row.get("visual_text") or row.get("markdown") or "")),
            )
        )
    if not payload.get("tile_results"):
        lines.append("| - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Next Command",
            "",
            "```powershell",
            f".\\scripts\\video-knowledge.ps1 tile-result-merge '{result.get('bundle_dir', '')}' --input-json '{result.get('output_json_path', '')}' --execute",
            "```",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _tile_result_row(
    *,
    root: Path,
    timeline_index: int,
    tile: dict[str, Any],
    result_files: dict[str, Path],
    default_source: str,
    default_confidence: float,
) -> tuple[dict[str, Any], bool]:
    tile_id = str(tile.get("tile_id") or "").strip()
    tile_path = str(tile.get("output_path") or tile.get("planned_output") or "").strip()
    result_file = _find_result_file(tile_id, tile_path, result_files)
    base = {
        "timeline_index": timeline_index,
        "tile_id": tile_id,
        "tile_path": _relative_or_abs(root, Path(tile_path)) if tile_path else "",
        "source": default_source,
        "status": "pending_result",
        "confidence": 0.0,
        "text": "",
    }
    if not result_file:
        return base, False
    parsed = _parse_result_file(result_file, default_confidence=default_confidence)
    row = {**base, **parsed}
    row["result_file"] = str(result_file)
    row.setdefault("source", default_source)
    row.setdefault("status", "ok")
    if row.get("confidence") is None:
        row["confidence"] = default_confidence
    evidence = str(row.get("evidence_path") or "").strip()
    if not evidence:
        row["evidence_path"] = base["tile_path"]
    return row, True


def _parse_result_file(path: Path, *, default_confidence: float = 0.0) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            return {"status": "parse_failed", "confidence": 0.0, "text": "", "error": str(exc)}
        text = _extract_result_text(value).strip()
        confidence = _extract_result_confidence(value, default=float(default_confidence or 0.0))
        structured = _extract_structured_visual(value, text=text)
        if not isinstance(value, dict):
            return {
                "status": "ok" if text else "empty",
                "confidence": confidence,
                "text": text,
                "structured_visual": structured,
                "source": "tile_result_json_list",
                "parse_source": "json_list",
            }
        return {
            "status": str(value.get("status") or ("ok" if text or structured else "empty")),
            "confidence": confidence,
            "text": text,
            "structured_visual": structured,
            "source": str(value.get("source") or value.get("provider") or value.get("model") or _infer_source(value)),
            "evidence_path": str(value.get("evidence_path") or value.get("tile_path") or ""),
            "parse_source": _infer_parse_source(value),
        }
    text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    return {
        "status": "ok" if text else "empty",
        "confidence": 0.8 if text else 0.0,
        "text": text,
        "source": f"tile_result_{suffix.lstrip('.') or 'text'}",
    }


def _extract_result_text(value: Any) -> str:
    texts: list[str] = []
    _collect_result_texts(value, texts)
    return _join_texts(texts)


def _collect_result_texts(value: Any, texts: list[str]) -> None:
    if isinstance(value, str):
        if value.strip():
            texts.append(value.strip())
        return
    if isinstance(value, list):
        for item in value:
            _collect_ocr_entry_text(item, texts)
        return
    if not isinstance(value, dict):
        return

    for key in ("text", "visual_text", "markdown", "content", "output", "response", "description", "raw_text", "recognized_text"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            texts.append(item.strip())

    choices = value.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            content = message.get("content") or choice.get("text")
            _collect_message_content_text(content, texts)

    candidates = value.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, dict):
                            _collect_message_content_text(part.get("text"), texts)
                        else:
                            _collect_message_content_text(part, texts)
            _collect_message_content_text(candidate.get("text"), texts)

    rec_texts = value.get("rec_texts")
    if isinstance(rec_texts, list):
        for text in rec_texts:
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    for key in ("result", "ocr_result", "data", "items", "lines", "blocks"):
        item = value.get(key)
        if isinstance(item, list):
            for entry in item:
                _collect_ocr_entry_text(entry, texts)

    visual = value.get("visual_understanding")
    if isinstance(visual, dict):
        for key in ("summary", "description", "screen_text", "non_text_information", "markdown"):
            item = visual.get(key)
            if isinstance(item, str) and item.strip():
                texts.append(item.strip())


def _collect_message_content_text(content: Any, texts: list[str]) -> None:
    if isinstance(content, str) and content.strip():
        texts.append(content.strip())
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                _collect_message_content_text(part.get("text") or part.get("content"), texts)
            else:
                _collect_message_content_text(part, texts)


def _collect_ocr_entry_text(entry: Any, texts: list[str]) -> None:
    if isinstance(entry, str):
        if entry.strip():
            texts.append(entry.strip())
        return
    if isinstance(entry, dict):
        for key in ("text", "visual_text", "markdown", "content", "recognized_text", "raw_text"):
            text = entry.get(key)
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
                return
        for key in ("result", "ocr_result", "data", "items", "lines", "blocks"):
            nested = entry.get(key)
            if isinstance(nested, list):
                for item in nested:
                    _collect_ocr_entry_text(item, texts)
        return
    if isinstance(entry, list):
        if len(entry) >= 2 and isinstance(entry[1], (list, tuple)) and entry[1] and isinstance(entry[1][0], str):
            if entry[1][0].strip():
                texts.append(entry[1][0].strip())
            return
        if len(entry) >= 2 and isinstance(entry[1], str):
            if entry[1].strip():
                texts.append(entry[1].strip())
            return
        for item in entry:
            _collect_ocr_entry_text(item, texts)


def _extract_result_confidence(value: Any, *, default: float) -> float:
    if isinstance(value, dict):
        for key in ("confidence", "score", "probability", "avg_confidence", "mean_confidence"):
            if key in value:
                return _float_or(value.get(key), default)
        scores: list[float] = []
        _collect_confidences(value, scores)
        if scores:
            return round(sum(scores) / len(scores), 4)
    elif isinstance(value, list):
        scores = []
        _collect_confidences(value, scores)
        if scores:
            return round(sum(scores) / len(scores), 4)
    return float(default)


def _collect_confidences(value: Any, scores: list[float]) -> None:
    if isinstance(value, dict):
        for key in ("confidence", "score", "probability"):
            if key in value:
                scores.append(_float_or(value.get(key), 0.0))
                return
        rec_scores = value.get("rec_scores")
        if isinstance(rec_scores, list):
            for item in rec_scores:
                scores.append(_float_or(item, 0.0))
        for key in ("result", "ocr_result", "data", "items", "lines", "blocks"):
            item = value.get(key)
            if isinstance(item, list):
                _collect_confidences(item, scores)
    elif isinstance(value, list):
        if len(value) >= 2 and isinstance(value[1], (list, tuple)) and len(value[1]) >= 2:
            if isinstance(value[1][0], str):
                scores.append(_float_or(value[1][1], 0.0))
                return
        if len(value) >= 3 and isinstance(value[1], str):
            scores.append(_float_or(value[2], 0.0))
            return
        for item in value:
            _collect_confidences(item, scores)


def _extract_structured_visual(value: Any, *, text: str) -> dict[str, Any] | list[Any]:
    if isinstance(value, dict):
        structured = value.get("structured_visual")
        if isinstance(structured, (dict, list)):
            return structured
        visual = value.get("visual_understanding")
        if isinstance(visual, dict):
            payload = {"type": "tile_visual_understanding", "visual_understanding": visual}
            if text:
                payload["markdown"] = text
            return payload
    return {}


def _infer_source(value: dict[str, Any]) -> str:
    if isinstance(value.get("choices"), list):
        return "openai_compatible_tile_result"
    if isinstance(value.get("candidates"), list):
        return "gemini_tile_result"
    if any(isinstance(value.get(key), list) for key in ("result", "ocr_result", "rec_texts")):
        return "ocr_tile_result"
    if isinstance(value.get("visual_understanding"), dict):
        return "vlm_tile_result"
    return "tile_result_json"


def _infer_parse_source(value: dict[str, Any]) -> str:
    if isinstance(value.get("choices"), list):
        return "openai_choices"
    if isinstance(value.get("candidates"), list):
        return "gemini_candidates"
    if isinstance(value.get("rec_texts"), list):
        return "ocr_rec_texts"
    if any(isinstance(value.get(key), list) for key in ("result", "ocr_result", "data", "items", "lines", "blocks")):
        return "ocr_entries"
    if isinstance(value.get("visual_understanding"), dict):
        return "visual_understanding"
    return "direct_fields"


def _join_texts(texts: list[str]) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for text in texts:
        clean = " ".join(str(text or "").split())
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return "\n".join(result)

def _index_result_files(results_root: Path) -> dict[str, Path]:
    if not results_root.exists():
        return {}
    files: dict[str, Path] = {}
    for path in results_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".txt", ".md"}:
            continue
        stem = path.stem.lower()
        files.setdefault(stem, path)
    return files


def _find_result_file(tile_id: str, tile_path: str, result_files: dict[str, Path]) -> Path | None:
    candidates = []
    if tile_id:
        candidates.extend([tile_id.lower(), tile_id.lower().replace("-", "_"), tile_id.lower().replace("-", "")])
    if tile_path:
        candidates.append(Path(tile_path).stem.lower())
    for candidate in candidates:
        if candidate in result_files:
            return result_files[candidate]
    for key, path in result_files.items():
        if tile_id and tile_id.lower() in key:
            return path
    return None


def _register_run(root: Path, result: dict[str, Any]) -> None:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    status = "needs_input" if summary.get("pending_results") else ("completed" if summary.get("matched_results") else "needs_input")
    register_bundle_run(
        root,
        run_type="tile_result_import_build",
        run_id="tile-result-import-build",
        status=status,
        title="Tile result import build",
        summary=f"Built tile-result-import.json with {summary.get('matched_results', 0)} matched and {summary.get('pending_results', 0)} pending tile results.",
        inputs={"results_dir": result.get("results_dir", ""), "plan_path": result.get("plan_path", "")},
        parameters={},
        artifacts=[
            {"key": "tile_result_import_json", "path": result.get("output_json_path", "")},
            {"key": "tile_result_import_report", "path": result.get("report_path", "")},
            {"key": "mcp_args", "path": result.get("mcp_args_path", "")},
        ],
        failed_items=result.get("pending_items") if isinstance(result.get("pending_items"), list) else [],
        retry_command=f".\\scripts\\video-knowledge.ps1 tile-result-import-build '{root}' --results-dir <tile-results-dir>",
        next_actions=_next_actions(summary),
        operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {},
        write=True,
    )


def _pending_failed_items(tile_results: list[dict[str, Any]], *, results_root: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in tile_results:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").strip().lower() != "pending_result":
            continue
        tile_id = str(row.get("tile_id") or "")
        result_hint = str(results_root) if results_root else "<tile-results-dir>"
        bundle_root = str(row.get("bundle_dir") or "").strip()
        tile_path = str(row.get("tile_path") or "").strip()
        rows.append(
            {
                "index": row.get("timeline_index"),
                "reason": "tile_result_pending",
                "tile_id": tile_id,
                "detail": f"No result file found for tile {tile_id}. Put a .json/.txt/.md result in {result_hint} and rerun tile-result-import-build.",
                "tile_path": tile_path,
                "suggested_next_tool": "tile_result_import_build",
                "suggested_retry_command": f".\\scripts\\video-knowledge.ps1 tile-result-import-build '{bundle_root or '<webui-bundle>'}' --results-dir {result_hint}",
                "tile_result_import_command": f".\\scripts\\video-knowledge.ps1 tile-result-import-build '{bundle_root or '<webui-bundle>'}' --results-dir {result_hint}",
                "tile_result_merge_command": f".\\scripts\\video-knowledge.ps1 tile-result-merge '{bundle_root or '<webui-bundle>'}' --input-json tile-result-import.json --execute",
                "review_command": f".\\scripts\\video-knowledge.ps1 prepare-review-session '{bundle_root or '<webui-bundle>'}' --limit 0 --group-by reason",
                "evidence_paths": [tile_path] if tile_path else [],
            }
        )
    return rows


def _next_actions(summary: dict[str, Any]) -> list[str]:
    if int(summary.get("pending_results") or 0) > 0:
        return [
            "Generate OCR/VLM/human result files for pending tiles, then rerun tile-result-import-build.",
            "Run tile-result-merge only after pending or low-quality tile outputs have been reviewed.",
        ]
    return ["Run tile-result-merge with the generated tile-result-import.json."]

def _read_object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _relative_or_abs(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _md(value: str) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
