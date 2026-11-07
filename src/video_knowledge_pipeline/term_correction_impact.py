from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json
from .term_text import load_bundle_term_replacements

SCHEMA = "video_knowledge_pipeline.term_correction_impact.v1"


SOURCE_CANDIDATES = [
    ("normalized_transcript_json", "normalized-transcript.json"),
    ("normalized_transcript_srt", "normalized-transcript.srt"),
    ("transcript_json", "transcript.json"),
    ("transcript_srt", "transcript.srt"),
    ("platform_subtitle", "platform-subtitle.json"),
    ("platform_subtitle_path", "platform-subtitle.srt"),
    ("subtitle_json", "subtitle.json"),
    ("subtitle_srt", "subtitle.srt"),
    ("timeline_json", "timeline.json"),
]

OUTPUT_CANDIDATES = [
    ("source_arbitrated_transcript_json", "source-arbitrated-transcript.json"),
    ("source_arbitrated_transcript_markdown", "source-arbitrated-transcript.md"),
    ("full_transcript", "exports/full-transcript.md"),
    ("smart_summary", "exports/smart-summary.md"),
    ("smart_summary_codex", "exports/smart-summary.codex.md"),
]

SOURCE_TEXT_KEYS = {
    "text",
    "raw_text",
    "original_text",
    "corrected_text",
    "punctuated_text",
    "transcript",
    "subtitle",
    "caption",
    "visual_text",
    "summary",
    "content",
}

OUTPUT_TEXT_KEYS = {
    "text",
    "corrected_text",
    "punctuated_text",
    "summary",
    "content",
}


def term_correction_impact_report(
    bundle_dir: str | Path,
    *,
    min_confidence: float = 0.88,
    write: bool = True,
) -> dict[str, Any]:
    """Report whether reviewed terminology corrections reached final outputs."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")

    replacements = load_bundle_term_replacements(root, min_confidence=min_confidence)
    replacement_rows = _replacement_rows(replacements)
    source_docs = _documents(root, manifest, SOURCE_CANDIDATES, text_keys=SOURCE_TEXT_KEYS)
    output_docs = _documents(root, manifest, OUTPUT_CANDIDATES, text_keys=OUTPUT_TEXT_KEYS)
    source_counts = _count_documents(source_docs, replacement_rows)
    output_counts = _count_documents(output_docs, replacement_rows)
    term_rows = _term_rows(replacement_rows, source_counts, output_counts)
    source_total = sum(row["source_alias_count"] for row in term_rows)
    output_total = sum(row["output_alias_count"] for row in term_rows)
    final_doc_names = {"full_transcript", "smart_summary", "smart_summary_codex"}
    final_residual_total = sum(
        count
        for row in term_rows
        for doc_name, count in row.get("output_by_document", {}).items()
        if doc_name in final_doc_names
    )
    corrected_docs_present = any(doc.get("key") == "source_arbitrated_transcript_json" for doc in output_docs)
    smart_summary_present = any(doc.get("key") in {"smart_summary", "smart_summary_codex"} for doc in output_docs)
    status = _status(
        replacements=replacement_rows,
        source_total=source_total,
        output_total=output_total,
        final_residual_total=final_residual_total,
        corrected_docs_present=corrected_docs_present,
        smart_summary_present=smart_summary_present,
    )
    reduction_rate = round((source_total - output_total) / source_total, 4) if source_total > 0 else None
    final_clean_rate = round((source_total - final_residual_total) / source_total, 4) if source_total > 0 else None
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": status,
        "ok": status in {"passed", "no_source_aliases"},
        "min_confidence": float(min_confidence),
        "replacement_count": len(replacement_rows),
        "source_alias_total": source_total,
        "output_alias_total": output_total,
        "final_export_alias_total": final_residual_total,
        "reduction_rate": reduction_rate,
        "final_clean_rate": final_clean_rate,
        "corrected_docs_present": corrected_docs_present,
        "smart_summary_present": smart_summary_present,
        "source_documents": [{k: v for k, v in doc.items() if k != "text"} for doc in source_docs],
        "output_documents": [{k: v for k, v in doc.items() if k != "text"} for doc in output_docs],
        "terms": term_rows,
        "artifacts": {
            "json": str(root / "term-correction-impact-report.json"),
            "markdown": str(root / "term-correction-impact-report.md"),
            "mcp_args": str(root / "mcp-term-correction-impact-report.args.json"),
        },
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "does_not_modify_raw_sources": True,
            "purpose": "Verify whether reviewed term decisions reached corrected transcript and final human-readable exports.",
        },
        "next_actions": _next_actions(status),
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="term_correction_impact_report", timeout_seconds=1.0):
            write_json(root / "term-correction-impact-report.json", result)
            (root / "term-correction-impact-report.md").write_text(_render_markdown(result), encoding="utf-8")
            write_json(
                root / "mcp-term-correction-impact-report.args.json",
                {"bundle_dir": str(root), "min_confidence": min_confidence, "write": True},
            )
            manifest["term_correction_impact_report_json"] = "term-correction-impact-report.json"
            manifest["term_correction_impact_report_markdown"] = "term-correction-impact-report.md"
            manifest["mcp_term_correction_impact_report_args"] = "mcp-term-correction-impact-report.args.json"
            manifest["term_correction_impact_summary"] = {
                "status": status,
                "replacement_count": len(replacement_rows),
                "source_alias_total": source_total,
                "output_alias_total": output_total,
                "final_export_alias_total": final_residual_total,
                "reduction_rate": reduction_rate,
                "final_clean_rate": final_clean_rate,
                "updated_at": result["updated_at"],
            }
            write_json(manifest_path, manifest)
            result["run_registry"] = _register_run(root, result, write=True)
    else:
        result["run_registry"] = _register_run(root, result, write=False)
    return result


def _replacement_rows(replacements: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for alias, canonical in replacements:
        alias_text = str(alias or "").strip()
        canonical_text = str(canonical or "").strip()
        key = (alias_text.lower(), canonical_text)
        if not alias_text or not canonical_text or key in seen:
            continue
        seen.add(key)
        rows.append({"alias": alias_text, "canonical": canonical_text})
    rows.sort(key=lambda row: (-len(row["alias"]), row["alias"].lower(), row["canonical"]))
    return rows


def _documents(
    root: Path,
    manifest: dict[str, Any],
    candidates: list[tuple[str, str]],
    *,
    text_keys: set[str],
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, fallback in candidates:
        raw = str(manifest.get(key) or fallback).strip()
        if not raw:
            continue
        path = _bundle_path(root, raw)
        if not path.exists():
            continue
        resolved = str(path.resolve()).lower()
        if resolved in seen:
            continue
        seen.add(resolved)
        text = _read_document_text(path, text_keys=text_keys)
        if not text.strip():
            continue
        docs.append({"key": key, "path": str(path), "text": text, "characters": len(text)})
    return docs


def _read_document_text(path: Path, *, text_keys: set[str]) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = read_json(path)
        except Exception:
            return ""
        return "\n".join(_collect_text(data, text_keys=text_keys))
    try:
        return path.read_text(encoding="utf-8-sig")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""


def _collect_text(value: Any, *, text_keys: set[str]) -> list[str]:
    rows: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and str(key) in text_keys:
                rows.append(item)
            else:
                rows.extend(_collect_text(item, text_keys=text_keys))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_collect_text(item, text_keys=text_keys))
    return rows



def _count_documents(docs: list[dict[str, Any]], replacements: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in replacements:
        alias = row["alias"]
        counts[alias] = {}
        for doc in docs:
            count = _count_alias(str(doc.get("text") or ""), alias)
            if count:
                counts[alias][str(doc.get("key") or "unknown")] = count
    return counts


def _term_rows(
    replacements: list[dict[str, Any]],
    source_counts: dict[str, dict[str, int]],
    output_counts: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in replacements:
        alias = row["alias"]
        source_by_doc = source_counts.get(alias, {})
        output_by_doc = output_counts.get(alias, {})
        source_total = sum(source_by_doc.values())
        output_total = sum(output_by_doc.values())
        rows.append(
            {
                "alias": alias,
                "canonical": row["canonical"],
                "source_alias_count": source_total,
                "output_alias_count": output_total,
                "source_by_document": source_by_doc,
                "output_by_document": output_by_doc,
                "resolved_in_outputs": output_total == 0,
                "had_source_alias": source_total > 0,
            }
        )
    rows.sort(key=lambda item: (-int(item["output_alias_count"]), -int(item["source_alias_count"]), str(item["alias"]).lower()))
    return rows


def _count_alias(text: str, alias: str) -> int:
    if not text or not alias:
        return 0
    escaped = r"\s+".join(re.escape(part) for part in alias.split() if part) if len(alias.split()) > 1 else re.escape(alias)
    if _ascii_edge(alias):
        pattern = rf"(?<![A-Za-z0-9_-]){escaped}(?![A-Za-z0-9_-])"
    else:
        pattern = escaped
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _ascii_edge(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped and re.match(r"[A-Za-z0-9_-]", stripped[0]) and re.search(r"[A-Za-z0-9_-]$", stripped))


def _status(
    *,
    replacements: list[dict[str, Any]],
    source_total: int,
    output_total: int,
    final_residual_total: int,
    corrected_docs_present: bool,
    smart_summary_present: bool,
) -> str:
    if not replacements:
        return "no_glossary_terms"
    if source_total <= 0:
        return "no_source_aliases"
    if not corrected_docs_present:
        return "needs_transcript_source_arbitration"
    if not smart_summary_present:
        return "needs_export"
    if final_residual_total > 0:
        return "residual_aliases_in_final_exports"
    if output_total > 0:
        return "passed_with_nonfinal_residuals"
    return "passed"


def _next_actions(status: str) -> list[str]:
    if status == "no_glossary_terms":
        return ["Run term-arbitration-codex and import reviewed decisions before measuring correction impact."]
    if status == "needs_transcript_source_arbitration":
        return ["Run transcript-source-arbitration so reviewed glossary terms can create source-arbitrated transcript outputs."]
    if status == "needs_export":
        return ["Run generate-smart-summary-with-codex and export-knowledge-note to refresh final human-readable exports."]
    if status == "residual_aliases_in_final_exports":
        return ["Inspect term-correction-impact-report.md, rerun transcript-source-arbitration and export-knowledge-note, or review ambiguous residual aliases manually."]
    return []


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Term Correction Impact Report",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Replacement pairs: `{result.get('replacement_count')}`",
        f"- Source alias total: `{result.get('source_alias_total')}`",
        f"- Output alias total: `{result.get('output_alias_total')}`",
        f"- Final export alias total: `{result.get('final_export_alias_total')}`",
        f"- Reduction rate: `{result.get('reduction_rate')}`",
        f"- Final clean rate: `{result.get('final_clean_rate')}`",
        "",
        "## Documents",
        "",
        "| Type | Key | Path | Characters |",
        "| --- | --- | --- | ---: |",
    ]
    for doc in result.get("source_documents") or []:
        lines.append(f"| source | `{doc.get('key')}` | `{doc.get('path')}` | {doc.get('characters', 0)} |")
    for doc in result.get("output_documents") or []:
        lines.append(f"| output | `{doc.get('key')}` | `{doc.get('path')}` | {doc.get('characters', 0)} |")
    lines.extend(["", "## Terms", "", "| Alias | Canonical | Source count | Output count | Final status |", "| --- | --- | ---: | ---: | --- |"])
    for row in result.get("terms") or []:
        final_status = "resolved" if row.get("resolved_in_outputs") else "residual"
        lines.append(
            f"| {_md(row.get('alias'))} | {_md(row.get('canonical'))} | {row.get('source_alias_count', 0)} | {row.get('output_alias_count', 0)} | `{final_status}` |"
        )
    next_actions = result.get("next_actions") if isinstance(result.get("next_actions"), list) else []
    if next_actions:
        lines.extend(["", "## Next Actions", ""])
        lines.extend(f"- {action}" for action in next_actions)
    return "\n".join(lines).rstrip() + "\n"


def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    status = "completed" if result.get("ok") else "needs_review"
    failed_items = []
    if not result.get("ok"):
        failed_items.append({"reason": result.get("status"), "detail": ", ".join(result.get("next_actions") or [])})
    return register_bundle_run(
        root,
        run_type="term_correction_impact_report",
        run_id="term-correction-impact-report",
        status=status,
        title="Term correction impact report",
        summary=f"status={result.get('status')}; final_residual={result.get('final_export_alias_total')}.",
        inputs={"bundle_dir": str(root)},
        parameters={"min_confidence": result.get("min_confidence")},
        artifacts=[
            {"key": "json", "path": root / "term-correction-impact-report.json"},
            {"key": "markdown", "path": root / "term-correction-impact-report.md"},
            {"key": "mcp_args", "path": root / "mcp-term-correction-impact-report.args.json"},
        ],
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 term-correction-impact-report '{root}'",
        next_actions=result.get("next_actions") if isinstance(result.get("next_actions"), list) else [],
        operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {},
        write=write,
    )


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
