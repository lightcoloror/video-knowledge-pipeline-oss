from __future__ import annotations

from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .knowledge_note_export import export_knowledge_note
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .smart_summary_codex import generate_smart_summary_with_codex
from .smart_summary_input_pack import build_smart_summary_input_pack
from .storage import bundle_write_lock, read_json, write_json
from .term_arbitration_codex import build_term_arbitration_codex_pack, validate_term_arbitration_codex_result
from .term_correction_impact import term_correction_impact_report
from .transcript_source_arbitration import arbitrate_transcript_sources

SCHEMA = "video_knowledge_pipeline.term_correction_closure.v1"


def run_term_correction_closure(
    bundle_dir: str | Path,
    *,
    accept_draft: bool = False,
    input_json: str | Path | None = None,
    max_terms: int = 60,
    term_min_confidence: float = 0.88,
    transcript_min_confidence: float = 0.72,
    generate_codex_summary: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Run the local terminology correction closure chain.

    This is a local orchestration layer. It does not call online LLMs and does
    not mutate raw ASR/subtitle sources. With ``input_json`` it imports a Codex/LLM
    reviewed JSON or Markdown response. With ``accept_draft`` it accepts only
    high-confidence Codex-substitute draft term decisions; ambiguous terms stay
    review-only.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    title = str(manifest.get("title") or root.name)

    validation_result = _skipped_validation(root)
    if input_json:
        validation_result = validate_term_arbitration_codex_result(
            root,
            input_json=input_json,
            min_confidence=term_min_confidence,
            write=write,
        )
        if not validation_result.get("ok"):
            return _validation_failed_result(
                root,
                title=title,
                validation_result=validation_result,
                accept_draft=accept_draft,
                input_json=input_json,
                max_terms=max_terms,
                term_min_confidence=term_min_confidence,
                transcript_min_confidence=transcript_min_confidence,
                generate_codex_summary=generate_codex_summary,
                write=write,
            )

    term_result = build_term_arbitration_codex_pack(
        root,
        input_json=input_json,
        max_terms=max_terms,
        min_confidence=term_min_confidence,
        accept_draft=accept_draft,
        write=write,
    )
    glossary_path = root / "term-arbitration-glossary.json"
    glossary_ready = _glossary_has_terms(glossary_path)
    transcript_result = arbitrate_transcript_sources(
        root,
        glossary_json=glossary_path if glossary_ready else None,
        min_confidence=transcript_min_confidence,
        promote=True,
        write=write,
    )

    first_input_pack = build_smart_summary_input_pack(root, title=title, write=write)
    first_export = export_knowledge_note(root, title=title, write=write)
    codex_summary = generate_smart_summary_with_codex(root, write=write) if generate_codex_summary else _skipped_codex_summary(root)
    second_export = export_knowledge_note(root, title=title, write=write)
    impact = term_correction_impact_report(root, min_confidence=term_min_confidence, write=write)
    final_input_pack = build_smart_summary_input_pack(root, title=title, write=write)
    final_export = export_knowledge_note(root, title=title, write=write)
    final_codex_summary = _codex_summary_with_final_quality(codex_summary, final_export)

    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": title,
        "status": _closure_status(term_result, transcript_result, impact, final_codex_summary),
        "accept_draft": bool(accept_draft),
        "input_json": str(Path(input_json).expanduser().resolve()) if input_json else "",
        "term_validation_status": validation_result.get("status"),
        "accepted_validation_decisions": validation_result.get("accepted_decision_count", 0),
        "rejected_validation_decisions": validation_result.get("rejected_decision_count", 0),
        "semantic_review_status": ((term_result.get("llm_semantic_arbitration") or {}).get("review_status") if isinstance(term_result.get("llm_semantic_arbitration"), dict) else ""),
        "local_only": True,
        "no_cloud_call": True,
        "updated_at": now_iso(),
        "steps": {
            "term_arbitration_codex_validation": _step_summary(validation_result),
            "term_arbitration_codex": _step_summary(term_result),
            "transcript_source_arbitration": _step_summary(transcript_result),
            "smart_summary_input_pack_initial": _step_summary(first_input_pack),
            "export_initial": _step_summary(first_export),
            "smart_summary_codex": _step_summary(codex_summary),
            "export_after_codex": _step_summary(second_export),
            "term_correction_impact": _step_summary(impact),
            "smart_summary_input_pack_final": _step_summary(final_input_pack),
            "export_final": _step_summary(final_export),
        },
        "artifacts": {
            "report_json": str(root / "term-correction-closure.json"),
            "report_markdown": str(root / "term-correction-closure.md"),
            "term_glossary": str(glossary_path),
            "term_result": str(root / "term-arbitration-codex-result.json"),
            "term_validation": str(root / "term-arbitration-codex-validation.json"),
            "term_validation_markdown": str(root / "term-arbitration-codex-validation.md"),
            "source_arbitrated_transcript": str(root / "source-arbitrated-transcript.json"),
            "term_impact_report": str(root / "term-correction-impact-report.json"),
            "smart_summary": str(root / "exports" / "smart-summary.md"),
            "smart_summary_prompt": str(root / "exports" / "smart-summary-codex-prompt.md"),
            "smart_summary_input_pack": str(root / "exports" / "smart-summary-input-pack.md"),
        },
        "next_actions": _next_actions(term_result, transcript_result, impact, final_codex_summary),
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "no_download": True,
            "does_not_modify_raw_sources": True,
            "accept_draft_only_high_confidence": bool(accept_draft),
            "codex_or_llm_input_json": str(Path(input_json).expanduser().resolve()) if input_json else "",
            "codex_or_llm_input_validated": bool(input_json and validation_result.get("ok")),
            "ambiguous_terms_remain_review_only": True,
        },
        "write": bool(write),
    }
    run_artifact = _register_run(root, result, write=write)
    result["run_registry"] = run_artifact
    if write:
        with bundle_write_lock(root, operation="term_correction_closure", timeout_seconds=1.0):
            write_json(root / "term-correction-closure.json", result)
            (root / "term-correction-closure.md").write_text(_render_report(result), encoding="utf-8")
            manifest = _read_manifest(root)
            manifest["term_correction_closure_json"] = "term-correction-closure.json"
            manifest["term_correction_closure_markdown"] = "term-correction-closure.md"
            manifest["mcp_term_correction_closure_args"] = "mcp-term-correction-closure.args.json"
            manifest["term_correction_closure_summary"] = {
                "status": result["status"],
                "accept_draft": bool(accept_draft),
                "input_json": str(Path(input_json).expanduser().resolve()) if input_json else "",
                "semantic_review_status": result.get("semantic_review_status"),
                "term_validation_status": result.get("term_validation_status"),
                "accepted_validation_decisions": result.get("accepted_validation_decisions", 0),
                "rejected_validation_decisions": result.get("rejected_validation_decisions", 0),
                "term_status": term_result.get("status"),
                "transcript_status": transcript_result.get("status"),
                "impact_status": impact.get("status"),
                "updated_at": result["updated_at"],
            }
            write_json(root / "manifest.json", manifest)
            write_json(
                root / "mcp-term-correction-closure.args.json",
                {
                    "bundle_dir": str(root),
                    "accept_draft": bool(accept_draft),
                    "input_json": str(Path(input_json).expanduser().resolve()) if input_json else "",
                    "max_terms": int(max_terms or 0),
                    "term_min_confidence": float(term_min_confidence),
                    "transcript_min_confidence": float(transcript_min_confidence),
                    "generate_codex_summary": bool(generate_codex_summary),
                    "write": True,
                },
            )
    return result


def _validation_failed_result(
    root: Path,
    *,
    title: str,
    validation_result: dict[str, Any],
    accept_draft: bool,
    input_json: str | Path | None,
    max_terms: int,
    term_min_confidence: float,
    transcript_min_confidence: float,
    generate_codex_summary: bool,
    write: bool,
) -> dict[str, Any]:
    input_path = str(Path(input_json).expanduser().resolve()) if input_json else ""
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": title,
        "status": "needs_term_review",
        "accept_draft": bool(accept_draft),
        "input_json": input_path,
        "term_validation_status": validation_result.get("status"),
        "accepted_validation_decisions": validation_result.get("accepted_decision_count", 0),
        "rejected_validation_decisions": validation_result.get("rejected_decision_count", 0),
        "semantic_review_status": "codex_validation_failed",
        "local_only": True,
        "no_cloud_call": True,
        "updated_at": now_iso(),
        "steps": {
            "term_arbitration_codex_validation": _step_summary(validation_result),
            "term_arbitration_codex": {"status": "skipped_validation_failed", "ok": False},
            "transcript_source_arbitration": {"status": "skipped_validation_failed", "ok": False},
            "smart_summary_input_pack_initial": {"status": "skipped_validation_failed", "ok": False},
            "export_initial": {"status": "skipped_validation_failed", "ok": False},
            "smart_summary_codex": {"status": "skipped_validation_failed", "ok": False},
            "export_after_codex": {"status": "skipped_validation_failed", "ok": False},
            "term_correction_impact": {"status": "skipped_validation_failed", "ok": False},
            "smart_summary_input_pack_final": {"status": "skipped_validation_failed", "ok": False},
            "export_final": {"status": "skipped_validation_failed", "ok": False},
        },
        "artifacts": {
            "report_json": str(root / "term-correction-closure.json"),
            "report_markdown": str(root / "term-correction-closure.md"),
            "term_validation": str(root / "term-arbitration-codex-validation.json"),
            "term_validation_markdown": str(root / "term-arbitration-codex-validation.md"),
            "term_glossary": str(root / "term-arbitration-glossary.json"),
            "term_result": str(root / "term-arbitration-codex-result.json"),
            "source_arbitrated_transcript": str(root / "source-arbitrated-transcript.json"),
        },
        "next_actions": _validation_next_actions(validation_result),
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "no_download": True,
            "does_not_modify_raw_sources": True,
            "accept_draft_only_high_confidence": bool(accept_draft),
            "codex_or_llm_input_json": input_path,
            "codex_or_llm_input_validated": False,
            "no_glossary_write_after_failed_validation": True,
            "no_transcript_write_after_failed_validation": True,
            "ambiguous_terms_remain_review_only": True,
        },
        "write": bool(write),
    }
    run_artifact = _register_run(root, result, write=write)
    result["run_registry"] = run_artifact
    if write:
        with bundle_write_lock(root, operation="term_correction_closure", timeout_seconds=1.0):
            write_json(root / "term-correction-closure.json", result)
            (root / "term-correction-closure.md").write_text(_render_report(result), encoding="utf-8")
            manifest = _read_manifest(root)
            manifest["term_correction_closure_json"] = "term-correction-closure.json"
            manifest["term_correction_closure_markdown"] = "term-correction-closure.md"
            manifest["mcp_term_correction_closure_args"] = "mcp-term-correction-closure.args.json"
            manifest["term_correction_closure_summary"] = {
                "status": result["status"],
                "accept_draft": bool(accept_draft),
                "input_json": input_path,
                "semantic_review_status": result.get("semantic_review_status"),
                "term_validation_status": result.get("term_validation_status"),
                "accepted_validation_decisions": result.get("accepted_validation_decisions", 0),
                "rejected_validation_decisions": result.get("rejected_validation_decisions", 0),
                "term_status": "skipped_validation_failed",
                "transcript_status": "skipped_validation_failed",
                "impact_status": "skipped_validation_failed",
                "updated_at": result["updated_at"],
            }
            write_json(root / "manifest.json", manifest)
            write_json(
                root / "mcp-term-correction-closure.args.json",
                {
                    "bundle_dir": str(root),
                    "accept_draft": bool(accept_draft),
                    "input_json": input_path,
                    "max_terms": int(max_terms or 0),
                    "term_min_confidence": float(term_min_confidence),
                    "transcript_min_confidence": float(transcript_min_confidence),
                    "generate_codex_summary": bool(generate_codex_summary),
                    "write": True,
                },
            )
    return result


def _skipped_validation(root: Path) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.term_arbitration_codex_validation.v1",
        "bundle_dir": str(root),
        "status": "skipped_no_input_json",
        "ok": True,
        "decision_count": 0,
        "accepted_decision_count": 0,
        "rejected_decision_count": 0,
        "next_actions": [],
        "operator_boundary": {"preview_only": True, "no_cloud_call": True},
        "updated_at": now_iso(),
        "write": False,
    }


def _validation_next_actions(validation_result: dict[str, Any]) -> list[str]:
    actions = [str(value) for value in (validation_result.get("next_actions") or []) if str(value)]
    status = str(validation_result.get("status") or "")
    if status == "no_accepted_decisions":
        actions.append("Keep low-confidence tool-name decisions in review, or rerun Codex semantic arbitration with stronger ASR/OCR/visual evidence.")
    elif status == "invalid":
        actions.append("Fix the Codex response so it contains parseable JSON with a decisions array, then rerun term-correction-closure --input-json.")
    else:
        actions.append("Resolve the term-arbitration Codex validation issue before updating glossary or corrected transcript.")
    return _unique(actions)

def _codex_summary_with_final_quality(codex_summary: dict[str, Any], final_export: dict[str, Any]) -> dict[str, Any]:
    merged = dict(codex_summary) if isinstance(codex_summary, dict) else {}
    quality = final_export.get("smart_summary_quality") if isinstance(final_export, dict) and isinstance(final_export.get("smart_summary_quality"), dict) else {}
    if quality:
        merged["quality"] = quality
    return merged


def _closure_status(term_result: dict[str, Any], transcript_result: dict[str, Any], impact: dict[str, Any], codex_summary: dict[str, Any]) -> str:
    if term_result.get("status") == "draft_ready":
        return "needs_term_review"
    if not transcript_result.get("ok"):
        return "needs_transcript_source"
    if not impact.get("ok"):
        return "needs_retry"
    quality = codex_summary.get("quality") if isinstance(codex_summary.get("quality"), dict) else {}
    if quality and not quality.get("passed"):
        return "needs_smart_summary_fix"
    return "completed"


def _step_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    summary = {
        "status": payload.get("status") or payload.get("quality_status") or ("ok" if payload.get("ok") else "unknown"),
        "ok": payload.get("ok"),
    }
    for key in (
        "candidate_count",
        "decision_count",
        "accepted_decision_count",
        "rejected_decision_count",
        "parse_error",
        "changed_segments",
        "replacement_count",
        "final_export_alias_total",
        "smart_summary_path",
        "smart_summary_codex_path",
        "prompt_path",
    ):
        if key in payload:
            summary[key] = payload.get(key)
    if isinstance(payload.get("summary"), dict):
        for key in ("changed_segments", "review_segments", "high_confidence_term_replacements"):
            if key in payload["summary"]:
                summary[key] = payload["summary"].get(key)
    return summary


def _next_actions(term_result: dict[str, Any], transcript_result: dict[str, Any], impact: dict[str, Any], codex_summary: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if term_result.get("status") == "draft_ready":
        actions.append("Review term-arbitration-codex-prompt.md with Codex, then rerun term-correction-closure with --input-json <codex-response.md>; use --accept-draft only for high-confidence rule draft replacements.")
    if not transcript_result.get("ok"):
        actions.append("Provide or rerun ASR/subtitle sources, then rerun transcript-source-arbitration.")
    if not impact.get("ok"):
        actions.extend([str(value) for value in (impact.get("next_actions") or []) if str(value)])
        actions.append("Rerun term-correction-closure after the final readable exports are regenerated.")
    quality = codex_summary.get("quality") if isinstance(codex_summary.get("quality"), dict) else {}
    if quality and not quality.get("passed"):
        actions.append("Open smart-summary-codex-status.md and fix smart-summary.codex.md before final export.")
    return _unique(actions)


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Term Correction Closure",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Accept draft: `{result.get('accept_draft')}`",
        f"- Input JSON/Markdown: `{result.get('input_json', '')}`",
        f"- Semantic review status: `{result.get('semantic_review_status', '')}`",
        f"- Term validation status: `{result.get('term_validation_status', '')}`",
        f"- Accepted validation decisions: `{result.get('accepted_validation_decisions', 0)}`",
        f"- Rejected validation decisions: `{result.get('rejected_validation_decisions', 0)}`",
        f"- Local only: `{result.get('local_only')}`",
        f"- No cloud call: `{result.get('no_cloud_call')}`",
        "",
        "## Steps",
        "",
        "| Step | Status | Key result |",
        "| --- | --- | --- |",
    ]
    for key, step in (result.get("steps") or {}).items():
        lines.append(f"| `{key}` | `{step.get('status')}` | {_md(_step_key_result(step))} |")
    lines.extend(["", "## Artifacts", "", "| Artifact | Path |", "| --- | --- |"])
    for key, path in (result.get("artifacts") or {}).items():
        lines.append(f"| `{key}` | `{path}` |")
    next_actions = result.get("next_actions") if isinstance(result.get("next_actions"), list) else []
    if next_actions:
        lines.extend(["", "## Next Actions", ""])
        lines.extend([f"- {action}" for action in next_actions])
    lines.extend([
        "",
        "## Boundary",
        "",
        "- 本闭环只调用本地已有 ASR/字幕/术语/导出模块，不调用云模型。",
        "- `--input-json` 可导入 Codex/LLM reviewed JSON 或包含 JSON 的 Markdown 回复。",
        "- `--input-json` 会先经过 validation；未通过时不会写入术语词典或纠正版转写。",
        "- `--accept-draft` 只接受高置信 draft，疑难项仍保留人工/Codex review。",
        "- 原始 ASR、字幕、视频和截图不会被覆盖。",
    ])
    return "\n".join(lines).rstrip() + "\n"

def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    failed_items = []
    for key, step in (result.get("steps") or {}).items():
        status = str(step.get("status") or "")
        if status in {"needs_term_review", "needs_retry", "needs_transcript_source", "needs_smart_summary_fix", "failed", "error", "invalid", "no_accepted_decisions", "skipped_validation_failed", "no_transcript_sources"}:
            failed_items.append({"index": key, "reason": status, "detail": _step_key_result(step)})
    return register_bundle_run(
        root,
        run_type="term_correction_closure",
        run_id="term-correction-closure",
        status=result.get("status") or "unknown",
        title="Term correction closure",
        summary=f"status={result.get('status')}; accept_draft={result.get('accept_draft')}",
        inputs={"bundle_dir": str(root), "input_json": result.get("input_json", "")},
        parameters={"accept_draft": bool(result.get("accept_draft")), "semantic_review_status": result.get("semantic_review_status", "")},
        artifacts=[
            {"key": "report_json", "path": root / "term-correction-closure.json"},
            {"key": "report_markdown", "path": root / "term-correction-closure.md"},
            {"key": "term_glossary", "path": root / "term-arbitration-glossary.json"},
            {"key": "term_validation", "path": root / "term-arbitration-codex-validation.json"},
            {"key": "term_impact", "path": root / "term-correction-impact-report.md"},
            {"key": "smart_summary", "path": root / "exports" / "smart-summary.md"},
        ],
        failed_items=failed_items,
        retry_command=_retry_command(root, result),
        next_actions=result.get("next_actions") if isinstance(result.get("next_actions"), list) else [],
        operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {},
        write=write,
    )



def _retry_command(root: Path, result: dict[str, Any]) -> str:
    base = f".\\scripts\\video-knowledge.ps1 term-correction-closure {_ps_quote(str(root))}"
    input_json = str(result.get("input_json") or "").strip()
    if input_json:
        return f"{base} --input-json {_ps_quote(input_json)}"
    if result.get("accept_draft"):
        return f"{base} --accept-draft"
    return base

def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    return manifest


def _glossary_has_terms(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except Exception:
        return False
    return isinstance(data, dict) and any(isinstance(row, dict) for row in (data.get("terms") or []))


def _skipped_codex_summary(root: Path) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.smart_summary_codex_status.v1",
        "bundle_dir": str(root),
        "status": "skipped",
        "quality": {},
        "updated_at": now_iso(),
    }


def _step_key_result(step: dict[str, Any]) -> str:
    parts = []
    for key in ("candidate_count", "decision_count", "accepted_decision_count", "rejected_decision_count", "parse_error", "changed_segments", "replacement_count", "final_export_alias_total", "smart_summary_path", "smart_summary_codex_path"):
        if key in step and step.get(key) not in (None, ""):
            parts.append(f"{key}={step.get(key)}")
    return "; ".join(parts) or str(step.get("ok") if "ok" in step else "")


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")



def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
