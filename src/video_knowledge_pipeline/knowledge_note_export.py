from __future__ import annotations

import mimetypes
import json
import hashlib
import re
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .file_hash import sha256_file as _file_sha256
from .artifact_freshness import build_dependency_snapshot
from .content_asset_status import _semantic_asset_gate
from .content_scope import (
    content_scope_projection_summary,
    load_content_scope,
)
from .transcript_projection import project_transcript_cues
from .final_reading_note import render_final_reading_note
from .reader_export_receipt import build_reader_export_receipt, receipt_matches_reader_files
from .long_video_memory_pack import build_long_video_memory_pack
from .models import now_iso
from .production_artifact_gate import (
    evaluate_production_artifact_gate,
    transcript_completeness_status,
)
from .run_artifact_registry import RUN_REGISTRY_MANIFEST_FIELDS, register_bundle_run
from .smart_summary_codex import (
    _smart_summary_content_scope_gate,
    generate_smart_summary_with_codex,
    smart_summary_quality_check,
)
from .smart_summary_chapters import attach_content_candidate_links
from .smart_summary_input_pack import build_smart_summary_input_pack
from .storage import bundle_write_lock, read_json, write_json
from .storage import read_json_object_or_empty as _read_optional_mapping
from .term_correction_status import term_correction_status
from .term_text import apply_high_confidence_term_replacements, is_high_confidence_term_candidate
from .transcript import format_timestamp, parse_transcript
from .transcript_quality_gate import run_transcript_quality_gate
from .transcript_speakers import speaker_display_name, speaker_label_map
from .transcript_semantic_correction import transcript_semantic_correction_status
from .transcript_sidecar import ensure_review_transcript_sidecar
from .visual_structure import reconcile_ebook_pipeline_checkpoints
from .visual_evidence import (
    evidence_index_sets,
    human_review_accepted as _shared_human_review_accepted,
    temporal_understanding_value,
    visual_understanding_value,
)


EXPORT_SCHEMA = "lecture_knowledge_note_export.v1"
BENCHMARK_SCRIPT_SCHEMA = "video_knowledge_pipeline.benchmark_script.v1"
BENCHMARK_STRATA: tuple[tuple[str, float, float], ...] = (
    ("opening", 0.0, 0.12),
    ("early", 0.12, 0.35),
    ("middle", 0.35, 0.65),
    ("late", 0.65, 0.85),
    ("conclusion", 0.85, 1.01),
)


def export_knowledge_note(
    bundle_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    title: str = "",
    include_timeline: bool = True,
    include_full_transcript: bool = True,
    run_transcript_evidence_check: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    from .transcript_semantic_transaction import summary_transaction_status
    pending = summary_transaction_status(root)
    if not pending['passed']:
        raise RuntimeError(f'pending_artifact_transaction:{pending}')
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")
    # Reuse the OCR item's atomic checkpoint before any export reads coverage.
    # This is local reconciliation only: it never starts OCR or a model call.
    ebook_checkpoint_reconciliation = reconcile_ebook_pipeline_checkpoints(
        root, write=write
    )
    transcript_evidence_check = _safe_transcript_evidence_check(root, write=write) if run_transcript_evidence_check else {
        "status": "skipped",
        "ok": True,
        "reason": "disabled_by_caller",
    }
    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline_data, list):
        raise ValueError("timeline.json must be a JSON array")
    timeline = [item for item in timeline_data if isinstance(item, dict)]
    from .media_applicability import project_timeline_for_media, resolve_media_applicability
    applicability = resolve_media_applicability(root, manifest, timeline)
    manifest["media_applicability"] = applicability
    timeline = project_timeline_for_media(timeline, applicability)
    # Bind stable evidence before rendering, not after a concurrent edit.
    render_inputs = {str(Path(row['path']).relative_to(root)): _file_sha256(Path(row['path']))
                     for row in _knowledge_export_dependency_inputs(root, manifest)}
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "exports"
    note_path = out_dir / "knowledge-note.md"
    transcript_path = out_dir / "full-transcript.md"
    full_body_path = out_dir / "full-body.md"
    smart_summary_path = out_dir / "smart-summary.md"
    smart_summary_prompt_path = out_dir / "smart-summary-codex-prompt.md"
    audit_path = out_dir / "extraction-audit.md"
    key_segments_path = out_dir / "key-segments.md"
    short_video_scripts_path = out_dir / "short-video-script-drafts.md"
    benchmark_script_json_path = out_dir / "benchmark-script.json"
    benchmark_script_markdown_path = out_dir / "benchmark-script.md"
    highlight_posts_path = out_dir / "highlight-post-drafts.md"
    content_candidate_pack_path = out_dir / "content-candidate-pack.json"
    content_candidate_pack_markdown_path = out_dir / "content-candidate-pack.md"
    material_card_path = out_dir / "content-material-card.json"
    material_card_markdown_path = out_dir / "content-material-card.md"
    summary_path = out_dir / "export-summary.json"
    transcript_projection_audit_path = out_dir / "transcript-projection-audit.json"
    note_title = title or str(manifest.get("title") or root.name)
    term_correction = term_correction_status(root)
    transcript_semantic_correction = transcript_semantic_correction_status(root, write=False)
    transcript_sidecar = ensure_review_transcript_sidecar(root, manifest, timeline, title=note_title, write=write)
    readable_timeline = _timeline_with_canonical_transcript(root, manifest, timeline)
    summary = _build_summary(manifest, readable_timeline)
    transcript_quality_gate = _safe_transcript_quality_gate(root, write=write)
    production_artifact_gate = evaluate_production_artifact_gate(
        root,
        artifact_kind="knowledge_note_export",
        write=write,
    )
    speaker_review = _speaker_review(root)
    if transcript_quality_gate.get("status") not in {"unavailable", "error"}:
        manifest["transcript_quality_gate_summary"] = {
            "status": transcript_quality_gate.get("status"),
            "ok": bool(transcript_quality_gate.get("ok")),
            "fail_count": int(transcript_quality_gate.get("fail_count") or 0),
            "warning_count": int(transcript_quality_gate.get("warning_count") or 0),
            "source_path": str(transcript_quality_gate.get("source_path") or ""),
            "punctuation_density_per_1000_chars": transcript_quality_gate.get("punctuation_density_per_1000_chars", ""),
            "updated_at": transcript_quality_gate.get("updated_at") or now_iso(),
        }
    note_markdown = ""
    transcript_markdown = _render_full_transcript(note_title, timeline, bundle_dir=root, manifest=manifest, sidecar=transcript_sidecar, transcript_quality_gate=transcript_quality_gate)
    full_body_markdown = _render_full_body(
        note_title,
        timeline,
        bundle_dir=root,
        manifest=manifest,
        sidecar=transcript_sidecar,
    )
    transcript_projection_audit = _build_transcript_projection_audit(
        root,
        manifest,
        transcript_sidecar,
        timeline,
    )
    smart_summary_input_pack = build_smart_summary_input_pack(root, title=note_title, write=write)
    long_video_memory_pack = build_long_video_memory_pack(root, write=write)
    smart_summary_prompt_markdown = _render_smart_summary_codex_prompt(note_title, root, manifest, readable_timeline, summary, transcript_sidecar, smart_summary_input_pack)
    smart_summary_codex_result = _ensure_smart_summary_codex_summary(root, smart_summary_prompt_path, smart_summary_prompt_markdown, write=write)
    smart_summary_markdown = _render_llm_summary_required(note_title, root)
    existing_smart_summary = _existing_codex_smart_summary(root)
    existing_summary_scope_gate = (
        _smart_summary_content_scope_gate(
            root,
            existing_smart_summary,
            existing_smart_summary.read_text(encoding="utf-8-sig"),
        )
        if existing_smart_summary
        else {"status": "not_available", "passed": False}
    )
    if existing_smart_summary and existing_summary_scope_gate.get("passed"):
        # Intent: keep the reader export bound to the canonical LLM summary.
        # Decision: export the complete canonical body even when its independent
        # quality gate still requires review.
        # Reason: quality/publication state must not replace an existing summary
        # with a misleading ``needs_llm_summary`` placeholder.
        # Evidence: ``_existing_codex_smart_summary`` already rejects drafts that
        # lack an accepted final/section-rewrite marker.
        # Effective scope: reader-facing copies only; quality status remains
        # fail-closed and is still computed below.
        smart_summary_markdown = existing_smart_summary.read_text(encoding="utf-8-sig")
    if not production_artifact_gate.get("formal_generation_allowed"):
        smart_summary_markdown = _artifact_review_watermark(
            smart_summary_markdown,
            artifact_label="智能总结机器草稿",
            gate=production_artifact_gate,
        )
    note_markdown = render_final_reading_note(
        note_title,
        smart_summary_markdown=smart_summary_markdown,
        full_body_markdown=full_body_markdown,
        transcript_markdown=transcript_markdown,
        timeline=readable_timeline,
        content_type=_reader_content_type(manifest),
        participant_count=_reader_participant_count(
            manifest,
            transcript_quality_gate,
            speaker_review,
        ),
    )
    audit_markdown = _render_extraction_audit(root, note_title, manifest, timeline, summary, term_correction)
    content_assets = _content_assets_index(
        note_path=note_path,
        transcript_path=transcript_path,
        full_body_path=full_body_path,
        smart_summary_path=smart_summary_path,
        smart_summary_prompt_path=smart_summary_prompt_path,
        audit_path=audit_path,
        key_segments_path=key_segments_path,
        short_video_scripts_path=short_video_scripts_path,
        benchmark_script_json_path=benchmark_script_json_path,
        benchmark_script_markdown_path=benchmark_script_markdown_path,
        highlight_posts_path=highlight_posts_path,
        content_candidate_pack_path=content_candidate_pack_path,
        content_candidate_pack_markdown_path=content_candidate_pack_markdown_path,
        material_card_path=material_card_path,
        material_card_markdown_path=material_card_markdown_path,
    )
    material_card = _build_content_material_card(
        bundle_dir=root,
        title=note_title,
        manifest=manifest,
        timeline=readable_timeline,
        summary=summary,
        content_assets=content_assets,
        term_correction=term_correction,
        transcript_semantic_correction=transcript_semantic_correction,
    )
    key_segments_markdown = _render_key_segments(root, note_title, readable_timeline)
    short_video_scripts_markdown = _render_short_video_script_drafts(root, note_title, readable_timeline)
    smart_summary_quality_preview = smart_summary_quality_check(
        root,
        summary_path=_quality_summary_path(existing_smart_summary, smart_summary_path),
        require_codex=True,
        write=False,
    )
    benchmark_script = _build_benchmark_script(
        root,
        note_title,
        readable_timeline,
        production_artifact_gate=production_artifact_gate,
        smart_summary_quality=smart_summary_quality_preview,
    )
    benchmark_script_markdown = _render_benchmark_script_markdown(benchmark_script)
    highlight_posts_markdown = _render_highlight_post_drafts(root, note_title, readable_timeline)
    content_candidate_pack = _build_content_candidate_pack(root, note_title, readable_timeline, output_dir=out_dir, term_correction=term_correction, transcript_semantic_correction=transcript_semantic_correction)
    chapter_candidate_links = attach_content_candidate_links(root, content_candidate_pack, write=write)
    content_candidate_pack["summary_chapter_link_status"] = chapter_candidate_links
    content_candidate_pack_markdown = _render_content_candidate_pack_markdown(content_candidate_pack)
    material_card_markdown = _render_content_material_card_markdown(material_card)
    result = {
        "schema": EXPORT_SCHEMA,
        "exported_at": now_iso(),
        "bundle_dir": str(root),
        "output_dir": str(out_dir),
        "note_path": str(note_path),
        "full_transcript_path": str(transcript_path),
        "full_body_path": str(full_body_path),
        "smart_summary_path": str(smart_summary_path),
        "smart_summary_prompt_path": str(smart_summary_prompt_path),
        "smart_summary_input_pack_path": str(out_dir / "smart-summary-input-pack.md"),
        "long_video_memory_pack_path": str(out_dir / "long-video-memory-pack.md"),
        "extraction_audit_path": str(audit_path),
        "summary_path": str(summary_path),
        "transcript_projection_audit_path": str(transcript_projection_audit_path),
        "transcript_projection_audit": transcript_projection_audit,
        "content_material_card_path": str(material_card_path),
        "content_material_card_markdown_path": str(material_card_markdown_path),
        "content_candidate_pack_path": str(content_candidate_pack_path),
        "content_candidate_pack_markdown_path": str(content_candidate_pack_markdown_path),
        "benchmark_script_json_path": str(benchmark_script_json_path),
        "benchmark_script_markdown_path": str(benchmark_script_markdown_path),
        "benchmark_script": benchmark_script,
        "content_assets": content_assets,
        "content_material_card": material_card,
        "content_candidate_pack": content_candidate_pack,
        "summary_chapter_content_candidate_links": chapter_candidate_links,
        "smart_summary_input_pack": smart_summary_input_pack,
        "long_video_memory_pack": long_video_memory_pack,
        "smart_summary_codex": smart_summary_codex_result,
        "existing_smart_summary_scope_gate": existing_summary_scope_gate,
        "production_artifact_gate": production_artifact_gate,
        "term_correction": term_correction,
        "transcript_semantic_correction": transcript_semantic_correction,
        "transcript_evidence_correction_pipeline": transcript_evidence_check,
        "ebook_checkpoint_reconciliation": ebook_checkpoint_reconciliation,
        "transcript_quality_gate": transcript_quality_gate,
        "summary": summary,
        "write_status": "planned" if not write else "pending",
        "quality_status": str(production_artifact_gate.get("status") or "unknown"),
        "quality_ready": bool(production_artifact_gate.get("formal_generation_allowed")),
        "status": (
            "planned_quality_ready"
            if production_artifact_gate.get("formal_generation_allowed")
            else "planned_quality_blocked"
        ),
        "ok": False,
        "write": write,
    }
    if write:
        with bundle_write_lock(root, operation="export_knowledge_note"):
            from .transcript_semantic_transaction import commit_transcript_semantic_artifacts, _profile_targets
            from .courseware_material import safe_path
            from .canonical_json import canonical_json_sha256
            journalled = out_dir == root / 'exports'
            payloads: dict[str, bytes] = {}
            expected = dict(render_inputs)
            if journalled:
                for name in _profile_targets('knowledge_export'):
                    path = safe_path(root, name, exists=False)
                    expected[name] = _file_sha256(path) if path.is_file() else ''
            def emit_text(path, value):
                if journalled:
                    payloads[str(path)] = value.encode('utf-8')
                else:
                    from .storage import write_text_atomic
                    write_text_atomic(path, value)
            def emit_json(path, value):
                emit_text(path, json.dumps(value, ensure_ascii=False, indent=2))
            out_dir.mkdir(parents=True, exist_ok=True)
            emit_text(note_path, note_markdown)
            emit_text(transcript_path, transcript_markdown)
            emit_text(full_body_path, full_body_markdown)
            emit_json(transcript_projection_audit_path, transcript_projection_audit)
            emit_text(smart_summary_path, smart_summary_markdown)
            emit_text(smart_summary_prompt_path, smart_summary_prompt_markdown)
            emit_text(audit_path, audit_markdown)
            emit_text(key_segments_path, key_segments_markdown)
            emit_text(short_video_scripts_path, short_video_scripts_markdown)
            emit_text(highlight_posts_path, highlight_posts_markdown)
            emit_json(content_candidate_pack_path, content_candidate_pack)
            emit_text(content_candidate_pack_markdown_path, content_candidate_pack_markdown)
            emit_json(material_card_path, material_card)
            emit_text(material_card_markdown_path, material_card_markdown)
            smart_summary_quality = smart_summary_quality_check(
                root,
                summary_path=_quality_summary_path(existing_smart_summary, smart_summary_path),
                require_codex=True,
                write=not journalled,
            )
            if journalled:
                from .smart_summary_codex import _render_quality_markdown
                emit_json(root / 'exports/smart-summary-quality.json', smart_summary_quality)
                emit_text(root / 'exports/smart-summary-quality.md', _render_quality_markdown(smart_summary_quality))
            smart_summary_final_status = _smart_summary_final_status(smart_summary_quality)
            result["smart_summary_final_status"] = smart_summary_final_status
            result["smart_summary_publication_boundary"] = _smart_summary_publication_boundary(smart_summary_final_status)
            benchmark_script = _build_benchmark_script(
                root,
                note_title,
                readable_timeline,
                production_artifact_gate=production_artifact_gate,
                smart_summary_quality=smart_summary_quality,
            )
            benchmark_script_markdown = _render_benchmark_script_markdown(benchmark_script)
            emit_json(benchmark_script_json_path, benchmark_script)
            emit_text(benchmark_script_markdown_path, benchmark_script_markdown)
            result["benchmark_script"] = benchmark_script
            term_correction = term_correction_status(root)
            note_markdown = render_final_reading_note(
                note_title,
                smart_summary_markdown=smart_summary_markdown,
                full_body_markdown=full_body_markdown,
                transcript_markdown=transcript_markdown,
                timeline=readable_timeline,
                content_type=_reader_content_type(manifest),
                participant_count=_reader_participant_count(
                    manifest,
                    transcript_quality_gate,
                    speaker_review,
                ),
            )
            audit_markdown = _render_extraction_audit(root, note_title, manifest, timeline, summary, term_correction)
            emit_text(note_path, note_markdown)
            emit_text(audit_path, audit_markdown)
            result["smart_summary_quality"] = smart_summary_quality
            result["smart_summary_final_status"] = result.get("smart_summary_final_status", "unknown_not_checked")
            result["smart_summary_publication_boundary"] = result.get("smart_summary_publication_boundary", {})
            result["term_correction"] = term_correction
            reader_export_receipt_path = out_dir / "reader-export-receipt.json"
            emit_json(
                reader_export_receipt_path,
                build_reader_export_receipt(
                    canonical_transcript=_canonical_transcript_path(root, manifest),
                    full_transcript=transcript_path,
                    full_body=full_body_path,
                    reading_note=note_path,
                    reader_payloads=payloads if journalled else None,
                ),
            )
            result["reader_export_receipt_path"] = str(reader_export_receipt_path)
            canonical_integrity = canonical_export_integrity_status(root, reader_payloads=payloads if journalled else None)
            result["canonical_transcript_integrity"] = canonical_integrity
            dependency_snapshot = build_dependency_snapshot(
                root,
                subject="knowledge-note-export",
                inputs=_knowledge_export_dependency_inputs(root, manifest),
                source_run_id="knowledge-note-export",
                producer_schema=EXPORT_SCHEMA,
            )
            dependency_snapshot_path = out_dir / "knowledge-export-dependency-snapshot.json"
            emit_json(dependency_snapshot_path, dependency_snapshot)
            result["dependency_snapshot"] = dependency_snapshot
            result["write_status"] = (
                "exported"
                if canonical_integrity.get("passed")
                else "blocked_canonical_export_mismatch"
            )
            result["quality_status"] = str(
                production_artifact_gate.get("status") or "unknown"
            )
            result["quality_ready"] = bool(
                production_artifact_gate.get("formal_generation_allowed")
            )
            result["status"] = _export_status(
                write_status=result["write_status"],
                quality_ready=result["quality_ready"],
            )
            result["ok"] = bool(
                result["write_status"] == "exported" and result["quality_ready"]
            )
            freshness_path = root / "knowledge-export-freshness.json"
            freshness_receipt = {
                "schema": "video_knowledge_pipeline.knowledge_export_freshness_receipt.v1",
                "status": "fresh",
                "reason": "knowledge_export_dependency_snapshot_refreshed",
                "snapshot_sha256": str(dependency_snapshot.get("snapshot_sha256") or ""),
                "write_status": result["write_status"],
                "quality_status": result["quality_status"],
                "quality_ready": result["quality_ready"],
                "freshness_is_not_quality_approval": True,
                "written_at": now_iso(),
            }
            emit_json(freshness_path, freshness_receipt)
            result["knowledge_export_freshness"] = "fresh"
            result["knowledge_export_freshness_path"] = str(freshness_path)
            result["dependency_snapshot_path"] = str(dependency_snapshot_path)
            result['export_transaction_scope'] = 'bundle_reader_projection' if journalled else 'legacy_external_directory_not_transactional'
            emit_json(summary_path, result)
            manifest["knowledge_note_export"] = {
                "schema": EXPORT_SCHEMA,
                "exported_at": result["exported_at"],
                "note_path": str(note_path),
                "full_transcript_path": str(transcript_path),
                "full_body_path": str(full_body_path),
                "smart_summary_path": str(smart_summary_path),
                "smart_summary_prompt_path": str(smart_summary_prompt_path),
                "smart_summary_input_pack_path": str(out_dir / "smart-summary-input-pack.md"),
                "long_video_memory_pack_path": str(out_dir / "long-video-memory-pack.md"),
                "extraction_audit_path": str(audit_path),
                "summary_path": str(summary_path),
                "transcript_projection_audit_path": str(transcript_projection_audit_path),
                "transcript_projection_audit": transcript_projection_audit,
                "content_material_card_path": str(material_card_path),
                "content_material_card_markdown_path": str(material_card_markdown_path),
                "content_candidate_pack_path": str(content_candidate_pack_path),
                "content_candidate_pack_markdown_path": str(content_candidate_pack_markdown_path),
                "benchmark_script_json_path": str(benchmark_script_json_path),
                "benchmark_script_markdown_path": str(benchmark_script_markdown_path),
                "benchmark_script": benchmark_script,
                "content_assets": content_assets,
                "content_material_card": material_card,
                "content_candidate_pack": content_candidate_pack,
                "smart_summary_quality": smart_summary_quality,
                "smart_summary_final_status": result.get("smart_summary_final_status", "unknown_not_checked"),
                "smart_summary_publication_boundary": result.get("smart_summary_publication_boundary", {}),
                "smart_summary_input_pack": smart_summary_input_pack,
                "long_video_memory_pack": long_video_memory_pack,
                "smart_summary_codex": smart_summary_codex_result,
                "existing_smart_summary_scope_gate": existing_summary_scope_gate,
                "term_correction": term_correction,
                "transcript_semantic_correction": transcript_semantic_correction,
                "transcript_evidence_correction_pipeline": transcript_evidence_check,
                "transcript_quality_gate": transcript_quality_gate,
                "canonical_transcript_integrity": canonical_integrity,
                "summary": summary,
                "dependency_snapshot": dependency_snapshot,
                "dependency_snapshot_path": str(dependency_snapshot_path),
            }
            manifest["smart_summary_quality"] = smart_summary_quality
            manifest["smart_summary_final_status"] = result.get("smart_summary_final_status", "unknown_not_checked")
            manifest["smart_summary_publication_boundary"] = result.get("smart_summary_publication_boundary", {})
            manifest["term_correction_status"] = term_correction
            manifest["transcript_semantic_correction_status"] = transcript_semantic_correction
            manifest["transcript_evidence_correction_pipeline_summary"] = _compact_transcript_evidence_check(transcript_evidence_check)
            manifest["transcript_quality_gate_summary"] = manifest.get("transcript_quality_gate_summary", {}) or {
                "status": transcript_quality_gate.get("status"),
                "ok": bool(transcript_quality_gate.get("ok")),
                "fail_count": int(transcript_quality_gate.get("fail_count") or 0),
                "warning_count": int(transcript_quality_gate.get("warning_count") or 0),
                "source_path": str(transcript_quality_gate.get("source_path") or ""),
                "punctuation_density_per_1000_chars": transcript_quality_gate.get("punctuation_density_per_1000_chars", ""),
                "updated_at": transcript_quality_gate.get("updated_at") or now_iso(),
            }
            manifest["content_assets"] = content_assets
            manifest["content_material_card"] = material_card
            manifest["content_candidate_pack"] = content_candidate_pack
            manifest["content_candidate_pack_json"] = "exports/content-candidate-pack.json"
            manifest["content_candidate_pack_markdown"] = "exports/content-candidate-pack.md"
            manifest["benchmark_script_json"] = "exports/benchmark-script.json"
            manifest["benchmark_script_markdown"] = "exports/benchmark-script.md"
            manifest["knowledge_note_markdown"] = str(note_path)
            manifest["knowledge_note_transcript_markdown"] = str(transcript_path)
            manifest["knowledge_note_full_body_markdown"] = str(full_body_path)
            manifest["knowledge_note_smart_summary_markdown"] = str(smart_summary_path)
            manifest["knowledge_note_smart_summary_codex_prompt_markdown"] = str(smart_summary_prompt_path)
            manifest["knowledge_note_smart_summary_input_pack_markdown"] = str(out_dir / "smart-summary-input-pack.md")
            manifest["knowledge_note_long_video_memory_pack_markdown"] = str(out_dir / "long-video-memory-pack.md")
            manifest["smart_summary_input_pack"] = "exports/smart-summary-input-pack.json"
            manifest["smart_summary_input_pack_markdown"] = "exports/smart-summary-input-pack.md"
            manifest["long_video_memory_pack"] = "exports/long-video-memory-pack.json"
            manifest["long_video_memory_pack_markdown"] = "exports/long-video-memory-pack.md"
            manifest["mcp_build_smart_summary_input_pack_args"] = "mcp-build-smart-summary-input-pack.args.json"
            manifest["mcp_long_video_memory_pack_args"] = "mcp-long-video-memory-pack.args.json"
            manifest["knowledge_note_extraction_audit_markdown"] = str(audit_path)
            manifest["transcript_projection_audit_json"] = str(transcript_projection_audit_path)
            manifest["knowledge_export_dependency_snapshot"] = str(dependency_snapshot_path)
            # Register pointers in the same exact-byte commit, not afterward.
            # The shared registry contract remains the sole field definition.
            manifest.update(RUN_REGISTRY_MANIFEST_FIELDS)
            emit_json(manifest_path, manifest)
            if journalled:
                current_evidence_paths = {str(Path(row['path']).relative_to(root))
                    for row in _knowledge_export_dependency_inputs(root, manifest)}
                if current_evidence_paths != set(render_inputs):
                    raise RuntimeError('export_dependency_set_drift: re-render from current evidence')
                artifacts = {str(Path(name).relative_to(root)).replace('\\', '/'): raw for name, raw in payloads.items()}
                result['export_transaction'] = commit_transcript_semantic_artifacts(root,
                    artifacts=artifacts, expected_inputs=expected,
                    operation_identity={'pack_sha256': canonical_json_sha256({k: hashlib.sha256(v).hexdigest() for k, v in artifacts.items()})},
                    profile='knowledge_export')
    try:
        result["run_registry"] = _register_export_run(root, result, write=write)
    except Exception as exc:
        if not result.get('export_transaction'):
            raise
        result.update(status='exported_registry_pending', ok=False,
            next_actions=[f'Export projection committed; repair run registry without rewriting source evidence: {exc}'])
    return result


def _export_status(*, write_status: str, quality_ready: bool) -> str:
    if write_status != "exported":
        return write_status
    return "exported_quality_ready" if quality_ready else "exported_quality_blocked"


def _knowledge_export_dependency_inputs(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"role": "timeline", "path": root / "timeline.json"}]
    from .courseware_material import build_courseware_material
    material = build_courseware_material(root, write=False)
    for reference in material.get("dependencies", []):
        rows.append({"role": "courseware_material", "path": root / reference["path"]})
    optional = [
        ("review_notes", root / "review-notes.json"),
        ("semantic_closure", root / "transcript-semantic-correction-closure.json"),
        ("content_scope", root / "content-scope.json"),
        ("courseware_independent_review", root / "courseware/independent-review.json"),
    ]
    canonical = _canonical_transcript_path(root, manifest)
    if canonical is not None:
        optional.append(("canonical_transcript", canonical))
    selected_transcript = _full_transcript_source_path(root, manifest, {})
    if selected_transcript is not None and selected_transcript != canonical:
        optional.append(("reader_transcript_source", selected_transcript))
    for role, path in optional:
        if path.is_file():
            rows.append({"role": role, "path": path})
    return rows


def _smart_summary_final_status(quality: dict[str, Any]) -> str:
    if not isinstance(quality, dict) or not quality:
        return "unknown_not_checked"
    if bool(quality.get("passed")):
        return "final"
    checks = {str(row.get("key") or ""): bool(row.get("passed")) for row in quality.get("checks") or [] if isinstance(row, dict)}
    if checks.get("codex_final") is False:
        return "needs_llm_summary"
    semantic_gate = quality.get("transcript_semantic_correction_gate") if isinstance(quality.get("transcript_semantic_correction_gate"), dict) else {}
    if semantic_gate and not bool(semantic_gate.get("passed")):
        return "draft_needs_semantic_correction"
    return "draft_quality_failed"


def _smart_summary_publication_boundary(status: str) -> dict[str, Any]:
    final = status == "final"
    return {
        "smart_summary_status": status,
        "review_required": not final,
        "publication_allowed": False,
        "allowed_as_fact": False,
        "allowed_as_inspiration": final,
        "reason": "smart summary passed quality gates but still requires human publication review" if final else ("LLM-generated smart summary is required; rule-based composition is not a final summary" if status == "needs_llm_summary" else "smart summary is a draft until semantic correction and quality gates pass"),
    }

def _safe_transcript_evidence_check(root: Path, *, write: bool) -> dict[str, Any]:
    validation_path = root / "transcript-semantic-correction-validation.json"
    closure_path = root / "transcript-semantic-correction-closure.json"
    if validation_path.exists() and closure_path.exists():
        return {
            "schema": "video_knowledge_pipeline.transcript_evidence_correction_pipeline.reused.v1",
            "bundle_dir": str(root),
            "status": "skipped_validated_semantic_artifacts_present",
            "ok": True,
            "reason": "semantic validation and closure already exist; status and artifact identity gates validate freshness",
            "validation_json": str(validation_path),
            "closure_json": str(closure_path),
            "updated_at": now_iso(),
        }
    try:
        from .transcript_evidence_correction_pipeline import run_transcript_evidence_correction_pipeline

        result = run_transcript_evidence_correction_pipeline(
            root,
            execute_llm=False,
            use_agent_substitute=False,
            agent_name="export_knowledge_note",
            run_postprocess=False,
            run_source_arbitration=False,
            run_readable_llm=False,
            execute_readable_llm=False,
            auto_apply_high_confidence=False,
            materialise_corrected_alias=False,
            run_agent_readable_rewrite=False,
            refresh_exports=False,
            write=write,
        )
        return _compact_transcript_evidence_check(result)
    except FileNotFoundError as exc:
        return {
            "schema": "video_knowledge_pipeline.transcript_evidence_correction_pipeline.unavailable.v1",
            "bundle_dir": str(root),
            "status": "unavailable",
            "ok": False,
            "reason": str(exc),
            "updated_at": now_iso(),
        }
    except Exception as exc:  # pragma: no cover - defensive export should still produce reviewable files.
        return {
            "schema": "video_knowledge_pipeline.transcript_evidence_correction_pipeline.error.v1",
            "bundle_dir": str(root),
            "status": "error",
            "ok": False,
            "reason": f"Inspect transcript-evidence-correction-pipeline failure: {exc}",
            "updated_at": now_iso(),
        }


def _compact_transcript_evidence_check(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"status": "missing", "ok": False}
    semantic_pack = result.get("semantic_pack") if isinstance(result.get("semantic_pack"), dict) else {}
    evidence_conflict = result.get("evidence_conflict_index") if isinstance(result.get("evidence_conflict_index"), dict) else {}
    quality_gate = result.get("transcript_quality_gate") if isinstance(result.get("transcript_quality_gate"), dict) else {}
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    return {
        "schema": result.get("schema", "video_knowledge_pipeline.transcript_evidence_correction_pipeline.summary.v1"),
        "status": result.get("status", "unknown"),
        "ok": bool(result.get("ok", False)),
        "semantic_candidate_count": int(semantic_pack.get("candidate_count") or semantic_pack.get("total_candidates") or 0),
        "llm_arbitration_count": int(evidence_conflict.get("llm_arbitration_count") or 0),
        "transcript_quality_gate_status": quality_gate.get("status", ""),
        "transcript_quality_gate_ok": bool(quality_gate.get("ok", False)),
        "corrected_transcript_json": str(artifacts.get("corrected_transcript_json") or ""),
        "pipeline_json": str(artifacts.get("pipeline_json") or ""),
        "updated_at": result.get("updated_at") or now_iso(),
    }

def _safe_transcript_quality_gate(root: Path, *, write: bool) -> dict[str, Any]:
    try:
        return run_transcript_quality_gate(root, write=write)
    except FileNotFoundError as exc:
        return {
            "schema": "video_knowledge_pipeline.transcript_quality_gate.unavailable.v1",
            "bundle_dir": str(root),
            "status": "unavailable",
            "ok": False,
            "fail_count": 0,
            "warning_count": 0,
            "source_path": "",
            "next_actions": [str(exc)],
            "updated_at": now_iso(),
        }
    except Exception as exc:  # pragma: no cover - defensive export should still produce reviewable files.
        return {
            "schema": "video_knowledge_pipeline.transcript_quality_gate.error.v1",
            "bundle_dir": str(root),
            "status": "error",
            "ok": False,
            "fail_count": 1,
            "warning_count": 0,
            "source_path": "",
            "next_actions": [f"Inspect transcript-quality-gate failure: {exc}"],
            "updated_at": now_iso(),
        }


def _register_export_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    artifacts = [
        {"key": "knowledge_note", "path": result.get("note_path", "")},
        {"key": "full_transcript", "path": result.get("full_transcript_path", "")},
        {"key": "full_body", "path": result.get("full_body_path", "")},
        {"key": "smart_summary", "path": result.get("smart_summary_path", "")},
        {"key": "smart_summary_prompt", "path": result.get("smart_summary_prompt_path", "")},
        {"key": "smart_summary_input_pack", "path": result.get("smart_summary_input_pack_path", "")},
        {"key": "long_video_memory_pack", "path": result.get("long_video_memory_pack_path", "")},
        {"key": "extraction_audit", "path": result.get("extraction_audit_path", "")},
        {"key": "export_summary", "path": result.get("summary_path", "")},
        {"key": "content_material_card", "path": result.get("content_material_card_path", "")},
        {"key": "content_material_card_markdown", "path": result.get("content_material_card_markdown_path", "")},
        {"key": "content_candidate_pack", "path": result.get("content_candidate_pack_path", "")},
        {"key": "content_candidate_pack_markdown", "path": result.get("content_candidate_pack_markdown_path", "")},
        {"key": "benchmark_script_json", "path": result.get("benchmark_script_json_path", "")},
        {"key": "benchmark_script_markdown", "path": result.get("benchmark_script_markdown_path", "")},
    ]
    failed_items: list[dict[str, Any]] = []
    if write:
        for artifact in artifacts:
            raw_path = str(artifact.get("path") or "")
            if raw_path and not Path(raw_path).exists():
                failed_items.append(
                    {
                        "id": artifact.get("key"),
                        "reason": "missing_export_artifact",
                        "detail": raw_path,
                    }
                )
    candidate_pack = result.get("content_candidate_pack") if isinstance(result.get("content_candidate_pack"), dict) else {}
    candidates = candidate_pack.get("candidates") if isinstance(candidate_pack.get("candidates"), list) else []
    if not candidates:
        failed_items.append(
            {
                "id": "content_candidate_pack",
                "reason": "content_candidate_missing",
                "detail": "No reusable content candidates were generated from the timeline.",
            }
        )
    chapter_link_status = result.get("summary_chapter_content_candidate_links") if isinstance(result.get("summary_chapter_content_candidate_links"), dict) else {}
    linked_count = int(chapter_link_status.get("linked_candidate_count") or 0)
    if candidates and (not chapter_link_status.get("exists") or linked_count <= 0):
        failed_items.append(
            {
                "id": "content_candidate_chapter_refs",
                "reason": "content_candidate_chapter_refs_missing",
                "detail": "Content candidates were exported, but they are not linked back to smart-summary chapters.",
            }
        )
    if not bool(result.get("quality_ready")):
        failed_items.append(
            {
                "id": "production_artifact_gate",
                "reason": "quality_gate_blocked",
                "detail": str(result.get("quality_status") or "quality_not_ready"),
            }
        )
    benchmark_script = result.get("benchmark_script") if isinstance(result.get("benchmark_script"), dict) else {}
    benchmark_coverage = benchmark_script.get("coverage") if isinstance(benchmark_script.get("coverage"), dict) else {}
    if any(str(item.get("reason")) == "missing_export_artifact" for item in failed_items) or any(str(item.get("reason")) == "content_candidate_missing" for item in failed_items):
        status = "needs_input"
    elif failed_items:
        status = "needs_review"
    else:
        status = "completed"
    return register_bundle_run(
        root,
        run_type="knowledge_note_export",
        run_id="knowledge-note-export",
        status=status,
        title="Knowledge note export",
        summary=f"Exported knowledge-note artifacts with {len(candidates)} content candidates; {len(failed_items)} item(s) need follow-up.",
        inputs={
            "timeline": str(root / "timeline.json"),
            "manifest": str(root / "manifest.json"),
            "smart_summary_chapters": str(root / "exports" / "smart-summary-chapters.json"),
        },
        parameters={
            "write": bool(write),
            "candidate_count": len(candidates),
            "linked_content_candidate_count": linked_count,
            "chapter_links_available": bool(chapter_link_status.get("exists")),
            "write_status": str(result.get("write_status") or "unknown"),
            "quality_status": str(result.get("quality_status") or "unknown"),
            "quality_ready": bool(result.get("quality_ready")),
            "benchmark_script_status": str(benchmark_script.get("status") or "missing"),
            "benchmark_coverage_passed": bool(benchmark_coverage.get("passed")),
        },
        artifacts=artifacts,
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 export-knowledge-note {_ps_quote(str(root))}",
        next_actions=_export_run_next_actions(status, failed_items),
        operator_boundary={
            "local_only": True,
            "no_cloud_call": True,
            "no_download": True,
            "no_auto_publish": True,
            "review_required_before_reuse": True,
            "purpose": "Register final human-readable export artifacts for UI/MCP retry and content handoff.",
        },
        write=write,
    )


def _export_run_next_actions(status: str, failed_items: list[dict[str, Any]]) -> list[str]:
    reasons = {str(item.get("reason") or "") for item in failed_items}
    if "missing_export_artifact" in reasons:
        return ["Inspect export-summary.json and rerun export-knowledge-note for the bundle."]
    if "content_candidate_missing" in reasons:
        return ["Inspect timeline evidence and rerun export-knowledge-note after ASR/OCR/vision evidence exists."]
    if "content_candidate_chapter_refs_missing" in reasons:
        return ["Run build-smart-summary-chapters before export-knowledge-note to link content candidates to smart-summary chapters."]
    if "quality_gate_blocked" in reasons:
        return ["Files were written for review; inspect production-artifact-gate.json before treating them as production-ready."]
    if status == "completed":
        return ["Open exports/knowledge-note.md, full-transcript.md, smart-summary.md, and content-candidate-pack.md for human review."]
    return ["Review the failed items and rerun export-knowledge-note after fixing the inputs."]


def _content_assets_index(
    *,
    note_path: Path,
    transcript_path: Path,
    full_body_path: Path,
    smart_summary_path: Path,
    smart_summary_prompt_path: Path,
    audit_path: Path,
    key_segments_path: Path,
    short_video_scripts_path: Path,
    benchmark_script_json_path: Path,
    benchmark_script_markdown_path: Path,
    highlight_posts_path: Path,
    content_candidate_pack_path: Path,
    content_candidate_pack_markdown_path: Path,
    material_card_path: Path,
    material_card_markdown_path: Path,
) -> dict[str, Any]:
    source_paths = {
        "summary_path": str(note_path),
        "smart_summary_path": str(smart_summary_path),
        "smart_summary_prompt_path": str(smart_summary_prompt_path),
        "timeline_path": str(transcript_path),
        "full_body_path": str(full_body_path),
        "audit_path": str(audit_path),
        "key_segments_path": str(key_segments_path),
        "short_video_script_drafts_path": str(short_video_scripts_path),
        "benchmark_script_json_path": str(benchmark_script_json_path),
        "benchmark_script_markdown_path": str(benchmark_script_markdown_path),
        "highlight_post_drafts_path": str(highlight_posts_path),
        "content_candidate_pack_path": str(content_candidate_pack_path),
        "content_candidate_pack_markdown_path": str(content_candidate_pack_markdown_path),
        "content_material_card_path": str(material_card_path),
        "content_material_card_markdown_path": str(material_card_markdown_path),
    }
    material_card_contract = _material_card_contract(source_paths=source_paths)
    return {
        "schema": "video_knowledge_pipeline.content_assets.v1",
        "review_required": True,
        "publication_allowed": False,
        **source_paths,
        "material_card_contract": material_card_contract,
        "consumer_rules": _content_asset_consumer_rules(),
        "human_confirmation_required": _human_confirmation_required_actions(),
    }

def _build_content_material_card(
    *,
    bundle_dir: Path,
    title: str,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    summary: dict[str, Any],
    content_assets: dict[str, Any],
    term_correction: dict[str, Any] | None = None,
    transcript_semantic_correction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = content_assets.get("material_card_contract") if isinstance(content_assets.get("material_card_contract"), dict) else {}
    mapping = contract.get("field_mapping") if isinstance(contract.get("field_mapping"), dict) else {}
    source_path = mapping.get("source_path") if isinstance(mapping.get("source_path"), dict) else {}
    material_id = _material_id(bundle_dir, manifest)
    first_evidence = _first_evidence_paths(bundle_dir, timeline, limit=8)
    semantic_snapshot = _transcript_semantic_correction_material_snapshot(transcript_semantic_correction or {})
    semantic_gate = semantic_snapshot.get("asset_gate") if isinstance(semantic_snapshot.get("asset_gate"), dict) else {}
    semantic_ready = bool(semantic_gate.get("passed"))
    return {
        "schema": "self_media_material_card.v1",
        "created_at": now_iso(),
        "material_id": material_id,
        "title": title,
        "source_path": source_path,
        "source_type": "video",
        "source_fact_status": "ai_extracted_needs_review",
        "evidence_tier": "timestamped_video_evidence",
        "privacy_level": "unknown_review_required",
        "desensitized": False,
        "compliance_risk": "needs_review",
        "fact_check_status": "needs_review",
        "target_layer": ["content_asset_pool", "circle_of_friends_inspiration"],
        "publish_surface": ["content_assets", "draft_only_after_review"],
        "content_stage": "evidence",
        "cta_type": "",
        "crm_followup_needed": False,
        "owner_thread": "video-knowledge-pipeline",
        "next_action": "human_review_then_route_to_content_assets" if semantic_ready else "finish_transcript_semantic_correction_before_handoff",
        "blocked_reason": "publication_not_allowed_without_human_review" if semantic_ready else "semantic_correction_not_passed",
        "review_required": True,
        "publication_allowed": False,
        "allowed_as_inspiration": semantic_ready,
        "allowed_as_fact": False,
        "circle_of_friends_status": "needs_review_inspiration" if semantic_ready else "semantic_correction_required",
        "content_assets": content_assets,
        "term_correction": _term_correction_material_snapshot(term_correction or {}),
        "transcript_semantic_correction": semantic_snapshot,
        "summary": summary,
        "evidence_paths": first_evidence,
        "human_confirmation_required": _human_confirmation_required_actions(),
        "must_fact_check_before_claiming": _content_asset_consumer_rules()["circle_of_friends"]["must_not_use_as_confirmed_fact"],
    }


def _term_correction_material_snapshot(status: dict[str, Any]) -> dict[str, Any]:
    accepted_terms = _accepted_term_labels_for_audit(status)
    return {
        "status": str(status.get("status") or "missing"),
        "term_validation_status": str(status.get("term_validation_status") or "missing"),
        "accepted_validation_decisions": int(status.get("accepted_validation_decisions") or 0),
        "rejected_validation_decisions": int(status.get("rejected_validation_decisions") or 0),
        "accepted_term_count": int(status.get("accepted_term_count") or 0),
        "accepted_terms": accepted_terms[:30],
        "source_arbitrated_transcript_exists": bool(status.get("source_arbitrated_transcript_exists")),
        "final_export_alias_total": int(status.get("final_export_alias_total") or 0),
        "semantic_review_evidence": _term_correction_material_evidence(status),
    }


def _transcript_semantic_correction_material_snapshot(status: dict[str, Any]) -> dict[str, Any]:
    gate = _semantic_asset_gate(status if isinstance(status, dict) else {})
    artifacts = status.get("artifacts") if isinstance(status.get("artifacts"), dict) else {}
    artifact_keys = (
        "pack_json",
        "pack_markdown",
        "validation_json",
        "validation_markdown",
        "closure_json",
        "closure_markdown",
        "impact_report_json",
        "impact_report_markdown",
        "readable_impact_json",
        "readable_impact_markdown",
        "summary_impact_json",
        "summary_impact_markdown",
        "corrected_transcript_json",
        "status_json",
        "status_markdown",
    )
    return {
        "status": str(status.get("status") or "missing"),
        "ok": bool(status.get("ok")),
        "candidate_count": int(status.get("candidate_count") or 0),
        "accepted_decision_count": int(status.get("accepted_decision_count") or 0),
        "review_required_count": int(status.get("review_required_count") or 0),
        "final_residual_error_total": int(status.get("final_residual_error_total") or 0),
        "readable_impact_status": str(status.get("readable_impact_status") or "missing"),
        "readable_required_residual_total": int(status.get("readable_required_residual_total") or 0),
        "summary_impact_status": str(status.get("summary_impact_status") or "missing"),
        "summary_impact_ok": bool(status.get("summary_impact_ok")),
        "summary_absorption_rate": float(status.get("summary_absorption_rate") or 0.0),
        "summary_residual_original_total": int(status.get("summary_residual_original_total") or 0),
        "closure_status": str(status.get("closure_status") or "missing"),
        "closure_applied_correction_count": int(status.get("closure_applied_correction_count") or 0),
        "corrected_transcript_exists": bool(status.get("corrected_transcript_exists")),
        "next_action_key": str(status.get("next_action_key") or "none"),
        "candidate_type_counts": status.get("candidate_type_counts", {}) if isinstance(status.get("candidate_type_counts"), dict) else {},
        "risk_level_counts": status.get("risk_level_counts", {}) if isinstance(status.get("risk_level_counts"), dict) else {},
        "evidence_source_counts": status.get("evidence_source_counts", {}) if isinstance(status.get("evidence_source_counts"), dict) else {},
        "asset_gate": gate,
        "evidence": {key: str(artifacts.get(key) or "") for key in artifact_keys if str(artifacts.get(key) or "").strip()},
    }


def _term_correction_material_evidence(status: dict[str, Any]) -> dict[str, str]:
    artifacts = status.get("artifacts") if isinstance(status.get("artifacts"), dict) else {}
    keys = ("term_validation_markdown", "glossary_json", "source_arbitrated_transcript_json", "impact_report_markdown", "closure_markdown")
    return {key: str(artifacts.get(key) or "") for key in keys if str(artifacts.get(key) or "").strip()}

def _render_content_material_card_markdown(card: dict[str, Any]) -> str:
    source_path = card.get("source_path") if isinstance(card.get("source_path"), dict) else {}
    semantic = card.get("transcript_semantic_correction") if isinstance(card.get("transcript_semantic_correction"), dict) else {}
    semantic_gate = semantic.get("asset_gate") if isinstance(semantic.get("asset_gate"), dict) else {}
    lines = [
        f"# {card.get('title') or card.get('material_id') or '视频素材卡'}",
        "",
        f"- Created: `{card.get('created_at') or now_iso()}`",
        f"- Material ID: `{card.get('material_id') or ''}`",
        f"- Source type: `{card.get('source_type') or 'video'}`",
        f"- Content stage: `{card.get('content_stage') or ''}`",
        f"- Source fact status: `{card.get('source_fact_status') or ''}`",
        f"- Evidence tier: `{card.get('evidence_tier') or ''}`",
        f"- Review required: `{str(bool(card.get('review_required'))).lower()}`",
        f"- Publication allowed: `{str(bool(card.get('publication_allowed'))).lower()}`",
        f"- Allowed as inspiration: `{str(bool(card.get('allowed_as_inspiration'))).lower()}`",
        f"- Allowed as fact: `{str(bool(card.get('allowed_as_fact'))).lower()}`",
        f"- Circle-of-friends status: `{card.get('circle_of_friends_status') or ''}`",
        f"- Term correction status: `{(card.get('term_correction') if isinstance(card.get('term_correction'), dict) else {}).get('status', 'missing')}`",
        f"- Codex term validation: `{(card.get('term_correction') if isinstance(card.get('term_correction'), dict) else {}).get('term_validation_status', 'missing')}`",
        f"- Term validation accepted/rejected: `{int((card.get('term_correction') if isinstance(card.get('term_correction'), dict) else {}).get('accepted_validation_decisions') or 0)}/{int((card.get('term_correction') if isinstance(card.get('term_correction'), dict) else {}).get('rejected_validation_decisions') or 0)}`",
        f"- Transcript semantic correction status: `{semantic.get('status', 'missing')}`",
        f"- Transcript semantic correction gate: `{semantic_gate.get('status', 'missing')}`",
        f"- Transcript semantic correction next action: `{semantic.get('next_action_key', 'none')}`",
        "",
        "## Source Paths",
        "",
    ]
    if source_path:
        for key, value in source_path.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("（暂无 source_path。）")
    lines.extend(["", "## Required Human Confirmation", ""])
    for action in card.get("human_confirmation_required") or []:
        lines.append(f"- `{action}`")
    lines.extend(["", "## Must Fact Check Before Claiming", ""])
    for item in card.get("must_fact_check_before_claiming") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Evidence Paths", ""])
    for path in card.get("evidence_paths") or []:
        lines.append(f"- `{path}`")
    if not card.get("evidence_paths"):
        lines.append("（暂无证据路径。）")
    return "\n".join(lines).rstrip() + "\n"


def _material_id(bundle_dir: Path, manifest: dict[str, Any]) -> str:
    for key in ("material_id", "video_id", "id", "bundle_id"):
        value = _text(manifest.get(key))
        if value:
            return value
    sources = manifest.get("sources") if isinstance(manifest.get("sources"), list) else []
    for source in sources:
        if isinstance(source, dict):
            value = _text(source.get("video_id") or source.get("id") or source.get("source_id"))
            if value:
                return value
    return f"vkp-{bundle_dir.name}"


def _first_evidence_paths(bundle_dir: Path, timeline: list[dict[str, Any]], *, limit: int) -> list[str]:
    paths: list[str] = []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        paths.extend(
            _evidence_paths(
                bundle_dir,
                item,
                visual=_item_visual_understanding(item),
                temporal=_item_temporal_understanding(item),
            )
        )
        if len(paths) >= limit:
            break
    return _dedupe(paths)[:limit]


def _material_card_contract(*, source_paths: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": "self_media_material_card.v1",
        "field_mapping": {
            "material_id": "video_knowledge_bundle_id_or_generated_asset_id",
            "source_path": source_paths,
            "source_type": "video",
            "source_fact_status": "ai_extracted_needs_review",
            "evidence_tier": "timestamped_video_evidence",
            "privacy_level": "unknown_review_required",
            "desensitized": False,
            "compliance_risk": "needs_review",
            "fact_check_status": "needs_review",
            "target_layer": ["content_asset_pool", "circle_of_friends_inspiration"],
            "publish_surface": ["content_assets", "draft_only_after_review"],
            "content_stage": "evidence",
            "cta_type": "",
            "crm_followup_needed": False,
            "owner_thread": "video-knowledge-pipeline",
            "next_action": "human_review_then_route_to_content_assets",
            "blocked_reason": "publication_not_allowed_without_human_review",
        },
        "allowed_as_inspiration": True,
        "allowed_as_fact": False,
    }


def _content_asset_consumer_rules() -> dict[str, Any]:
    return {
        "circle_of_friends": {
            "allowed_status": "needs_review_inspiration",
            "draft_only": True,
            "may_use": [
                "topic_angle",
                "question_prompt",
                "analogy",
                "structure",
                "timestamped_quote_for_review",
            ],
            "must_not_use_as_confirmed_fact": [
                "customer_story",
                "medical_claim",
                "insurance_product_or_compliance_claim",
                "investment_or_income_claim",
                "platform_rule_or_tool_ranking_claim",
                "time_sensitive_market_or_policy_claim",
            ],
        },
        "content_assets": {
            "allowed_stage": "evidence",
            "requires_source_path": True,
            "requires_timestamp_or_evidence_path": True,
        },
    }


def _human_confirmation_required_actions() -> list[str]:
    return [
        "download_or_account_authorized_access",
        "cloud_asr_or_cloud_vision_execution",
        "fact_check_before_claiming_truth",
        "privacy_desensitization_before_customer_related_use",
        "compliance_review_before_insurance_or_medical_use",
        "publish_or_send_to_any_external_surface",
        "writeback_to_logseq_or_obsidian_canonical_notes",
    ]


def _render_key_segments(bundle_dir: Path, title: str, timeline: list[dict[str, Any]]) -> str:
    lines = [
        f"# {title} - 关键片段候选",
        "",
        f"- Created: `{now_iso()}`",
        "- Review required: `true`",
        "- Publication allowed: `false`",
        "",
        "> 这些片段是内容资产候选，不是已确认选题。每条都保留时间段和证据路径，发布前必须人工审核。",
        "",
    ]
    candidates = _content_candidate_items(timeline)
    if not candidates:
        lines.append("（暂无可生成的关键片段候选。）")
    for rank, item in enumerate(candidates[:12], start=1):
        visual = _transcript_visual_note(item)
        evidence = _evidence_paths(bundle_dir, item, visual=_item_visual_understanding(item), temporal=_item_temporal_understanding(item))
        lines.extend(
            [
                f"## {rank}. {_time_range(item)}",
                "",
                f"- Timeline index: `{item.get('index') or rank}`",
                f"- Route: `{item.get('visual_route') or 'unknown'}`",
                f"- Evidence: `{'; '.join(evidence[:6]) if evidence else 'none'}`",
                "",
                "### 说了什么",
                "",
                _truncate(_item_transcript(item), 320) or "（暂无转写。）",
                "",
                "### 演示了什么",
                "",
                visual or "（暂无可靠画面理解。）",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_short_video_script_drafts(bundle_dir: Path, title: str, timeline: list[dict[str, Any]]) -> str:
    lines = [
        f"# {title} - 短视频脚本草稿候选",
        "",
        f"- Created: `{now_iso()}`",
        "- Review required: `true`",
        "- Publication allowed: `false`",
        "- Selection: `timeline_stratified`",
        "",
    ]
    candidates = _content_candidate_items(timeline)
    duration = _timeline_duration(timeline)
    if not candidates:
        lines.append("（暂无可生成的短视频脚本草稿。）")
    for rank, item in enumerate(candidates[:6], start=1):
        transcript = _truncate(_item_transcript(item), 220)
        evidence = _evidence_paths(bundle_dir, item, visual=_item_visual_understanding(item), temporal=_item_temporal_understanding(item))
        lines.extend(
            [
                f"## 草稿 {rank}: {_time_range(item)}",
                "",
                f"- Timeline stratum: `{_benchmark_stratum(item, duration)}`",
                f"- Evidence: `{'; '.join(evidence[:4]) if evidence else 'none'}`",
                "- Status: `draft_for_human_review`",
                "",
                "### 开头",
                "",
                transcript or "这里有一个需要人工提炼的观点。",
                "",
                "### 主体",
                "",
                "- 复述原视频观点，避免添加未验证信息。",
                "- 结合该片段画面/演示说明，保留必要证据截图。",
                "- 明确这是来自视频片段的理解，不替代原视频。",
                "",
                "### 结尾",
                "",
                "- 留一个具体问题，引导人工判断是否值得继续加工。",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_highlight_post_drafts(bundle_dir: Path, title: str, timeline: list[dict[str, Any]]) -> str:
    lines = [
        f"# {title} - 精华帖素材草稿",
        "",
        f"- Created: `{now_iso()}`",
        "- Review required: `true`",
        "- Publication allowed: `false`",
        "",
    ]
    candidates = _content_candidate_items(timeline)
    if not candidates:
        lines.append("（暂无可生成的精华帖素材。）")
    for rank, item in enumerate(candidates[:8], start=1):
        evidence = _evidence_paths(bundle_dir, item, visual=_item_visual_understanding(item), temporal=_item_temporal_understanding(item))
        lines.extend(
            [
                f"## 素材 {rank}: {_time_range(item)}",
                "",
                f"- Timeline index: `{item.get('index') or rank}`",
                f"- Evidence: `{'; '.join(evidence[:4]) if evidence else 'none'}`",
                "",
                "### 可用观点",
                "",
                f"- {_truncate(_item_transcript(item), 260) or '待人工补充观点。'}",
                "",
                "### 画面/演示补充",
                "",
                _transcript_visual_note(item) or "（暂无可靠画面理解。）",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _build_content_candidate_pack(bundle_dir: Path, title: str, timeline: list[dict[str, Any]], *, output_dir: Path | None = None, term_correction: dict[str, Any] | None = None, transcript_semantic_correction: dict[str, Any] | None = None) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    citation_rows_by_index = _content_candidate_citations_by_timeline_index(output_dir or bundle_dir / "exports")
    term_snapshot = _term_correction_material_snapshot(term_correction or {})
    semantic_snapshot = _transcript_semantic_correction_material_snapshot(transcript_semantic_correction or {})
    semantic_gate = semantic_snapshot.get("asset_gate") if isinstance(semantic_snapshot.get("asset_gate"), dict) else {}
    semantic_ready = bool(semantic_gate.get("passed"))
    for rank, item in enumerate(_content_candidate_items(timeline)[:16], start=1):
        transcript = _item_transcript(item)
        visual_note = _transcript_visual_note(item)
        visual = _item_visual_understanding(item)
        temporal = _item_temporal_understanding(item)
        timeline_index = int(item.get("index") or rank)
        evidence_citations = citation_rows_by_index.get(timeline_index, [])
        summary_chapter_refs = _content_candidate_chapter_refs(evidence_citations)
        evidence = _merge_candidate_evidence_paths(_evidence_paths(bundle_dir, item, visual=visual, temporal=temporal), evidence_citations)
        start = _float_seconds(item.get("start"))
        end = _float_seconds(item.get("end"))
        candidate_types = _content_candidate_types(item, transcript, visual_note)
        reusable_quote = _extract_reusable_quote(transcript)
        viewpoint = _candidate_viewpoint(transcript, visual_note)
        case_or_example = _candidate_case_or_example(transcript, visual_note)
        candidates.append(
            {
                "id": f"candidate-{rank:03d}",
                "timeline_index": timeline_index,
                "time_range": _time_range(item),
                "start": start,
                "end": end,
                "candidate_types": candidate_types,
                "viewpoint": viewpoint,
                "case_or_example": case_or_example,
                "reusable_quote": reusable_quote,
                "short_video_script_draft": {
                    "hook": _candidate_hook(transcript, visual_note),
                    "body": viewpoint,
                    "evidence_note": _truncate(visual_note, 240) or "待人工复核画面证据。",
                    "cta": "把这一段和原视频时间戳一起人工复核后，再决定是否继续改写。",
                },
                "highlight_post_seed": {
                    "title_seed": _candidate_title_seed(transcript, candidate_types),
                    "core_point": viewpoint,
                    "supporting_evidence": _truncate(case_or_example or visual_note, 260),
                },
                "evidence_paths": evidence[:8],
                "evidence_citations": evidence_citations,
                "summary_chapter_refs": summary_chapter_refs,
                "summary_chapter_ref_count": len(summary_chapter_refs),
                "citation_digest_status": "ready" if evidence_citations else "not_available",
                "source_fact_status": "ai_extracted_needs_review",
                "fact_check_status": "needs_review",
                "privacy_level": "unknown_review_required",
                "review_required": True,
                "publication_allowed": False,
                "allowed_as_fact": False,
                "allowed_as_inspiration": semantic_ready,
                "term_correction_status": term_snapshot.get("status", "missing"),
                "term_validation_status": term_snapshot.get("term_validation_status", "missing"),
                "accepted_term_count": term_snapshot.get("accepted_term_count", 0),
                "semantic_correction_status": semantic_snapshot.get("status", "missing"),
                "semantic_correction_gate_status": semantic_gate.get("status", "missing"),
                "semantic_correction_review_count": semantic_snapshot.get("review_required_count", 0),
                "semantic_correction_summary_impact_status": semantic_snapshot.get("summary_impact_status", "missing"),
            }
        )
    return {
        "schema": "video_knowledge_pipeline.content_candidate_pack.v1",
        "created_at": now_iso(),
        "title": title,
        "bundle_dir": str(bundle_dir),
        "candidate_count": len(candidates),
        "citation_digest_candidate_count": sum(1 for candidate in candidates if candidate.get("evidence_citations")),
        "term_correction": term_snapshot,
        "transcript_semantic_correction": semantic_snapshot,
        "review_required": True,
        "publication_allowed": False,
        "allowed_as_fact": False,
        "allowed_as_inspiration": semantic_ready,
        "operator_boundary": {
            "not_a_publishable_draft": True,
            "fact_check_required": True,
            "human_review_required_before_reuse": True,
        },
        "candidates": candidates,
    }

def _content_candidate_citations_by_timeline_index(output_dir: Path) -> dict[int, list[dict[str, Any]]]:
    chapters_path = output_dir / "smart-summary-chapters.json"
    if not chapters_path.exists():
        return {}
    try:
        payload = read_json(chapters_path)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    chapters = payload.get("chapters") if isinstance(payload.get("chapters"), list) else []
    rows_by_index: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[int, str, str, str]] = set()
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        for citation in chapter.get("citation_digest") or []:
            if not isinstance(citation, dict):
                continue
            indexes = _citation_timeline_indexes(citation)
            if not indexes:
                continue
            row = _compact_content_candidate_citation(citation)
            if row:
                row["summary_chapter_ref"] = _content_candidate_chapter_ref(chapter)
            if not row:
                continue
            for index in indexes:
                from .canonical_json import canonical_json_sha256
                key = (index, canonical_json_sha256(row))
                if key in seen:
                    continue
                seen.add(key)
                rows_by_index.setdefault(index, []).append(row)
    return rows_by_index


def _candidate_ref_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

def _content_candidate_chapter_ref(chapter: dict[str, Any]) -> dict[str, Any]:
    index = _candidate_ref_int(chapter.get("index") or chapter.get("chapter_index"))
    return {
        "chapter_index": index,
        "chapter_title": _text(chapter.get("title")),
        "chapter_time_range": f"{chapter.get('start_time') or ''} - {chapter.get('end_time') or ''}".strip(" -"),
        "chapter_start": _float_seconds(chapter.get("start")),
        "chapter_end": _float_seconds(chapter.get("end")),
    }


def _content_candidate_chapter_refs(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[int] = set()
    for citation in citations:
        ref = citation.get("summary_chapter_ref") if isinstance(citation.get("summary_chapter_ref"), dict) else {}
        index = _candidate_ref_int(ref.get("chapter_index"))
        if index <= 0 or index in seen:
            continue
        seen.add(index)
        refs.append(ref)
    return refs

def _citation_timeline_indexes(citation: dict[str, Any]) -> list[int]:
    raw = citation.get("timeline_indexes")
    values = raw if isinstance(raw, list) else ([raw] if raw is not None else [])
    indexes: list[int] = []
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index not in indexes:
            indexes.append(index)
    return indexes


def _compact_content_candidate_citation(citation: dict[str, Any]) -> dict[str, Any]:
    from .citation_digest import evidence_level
    text = _text(citation.get("text") or citation.get("summary") or citation.get("content"))
    source_type = _text(citation.get("source_type") or citation.get("type") or "unknown") or "unknown"
    time = _text(citation.get("time") or citation.get("time_range") or "")
    timeline_indexes = _citation_timeline_indexes(citation)
    evidence_paths = [str(path) for path in citation.get("evidence_paths") or [] if str(path)] if isinstance(citation.get("evidence_paths"), list) else []
    if not text and not evidence_paths:
        return {}
    return {
        **citation,
        "evidence_level": evidence_level(citation),
        "source_type": source_type,
        "time": time,
        "timeline_indexes": timeline_indexes,
        "text": text,
        "evidence_paths": evidence_paths,
    }


def _merge_candidate_evidence_paths(paths: list[str], citations: list[dict[str, Any]]) -> list[str]:
    merged: list[str] = []
    for path in paths:
        value = str(path)
        if value and value not in merged:
            merged.append(value)
    for citation in citations:
        for path in citation.get("evidence_paths") or []:
            value = str(path)
            if value and value not in merged:
                merged.append(value)
    return merged

def _render_content_candidate_pack_markdown(pack: dict[str, Any]) -> str:
    lines = [
        f"# {pack.get('title') or '视频内容'} - 内容素材候选包",
        "",
        f"- Created: `{pack.get('created_at') or now_iso()}`",
        f"- Candidate count: `{pack.get('candidate_count') or 0}`",
        f"- Candidates with citation digest: `{pack.get('citation_digest_candidate_count') or 0}`",
        f"- Term correction status: `{(pack.get('term_correction') if isinstance(pack.get('term_correction'), dict) else {}).get('status', 'missing')}`",
        f"- Codex term validation: `{(pack.get('term_correction') if isinstance(pack.get('term_correction'), dict) else {}).get('term_validation_status', 'missing')}`",
        f"- Term validation accepted/rejected: `{int((pack.get('term_correction') if isinstance(pack.get('term_correction'), dict) else {}).get('accepted_validation_decisions') or 0)}/{int((pack.get('term_correction') if isinstance(pack.get('term_correction'), dict) else {}).get('rejected_validation_decisions') or 0)}`",
        f"- Transcript semantic correction status: `{((pack.get('transcript_semantic_correction') if isinstance(pack.get('transcript_semantic_correction'), dict) else {}).get('status', 'missing'))}`",
        f"- Transcript semantic correction gate: `{(((pack.get('transcript_semantic_correction') if isinstance(pack.get('transcript_semantic_correction'), dict) else {}).get('asset_gate') if isinstance((pack.get('transcript_semantic_correction') if isinstance(pack.get('transcript_semantic_correction'), dict) else {}).get('asset_gate'), dict) else {}).get('status', 'missing'))}`",
        "- Review required: `true`",
        "- Publication allowed: `false`",
        "- Allowed as fact: `false`",
        "- Allowed as inspiration: `true`",
        "",
        "> 这个文件只是一篮子可复用素材候选，不是发布稿。引用观点、案例、数字、工具名之前必须回看原视频和证据路径。",
        "",
    ]
    candidates = pack.get("candidates") if isinstance(pack.get("candidates"), list) else []
    if not candidates:
        lines.append("（暂无内容素材候选。）")
        return "\n".join(lines).rstrip() + "\n"
    lines.extend(["## 候选总览", "", "| ID | 时间 | 类型 | 观点种子 |", "| --- | --- | --- | --- |"])
    for candidate in candidates:
        types = ", ".join(str(value) for value in candidate.get("candidate_types") or [])
        lines.append(
            "| `{id}` | `{time}` | {types} | {viewpoint} |".format(
                id=_table_cell(str(candidate.get("id") or "")),
                time=_table_cell(str(candidate.get("time_range") or "")),
                types=_table_cell(types),
                viewpoint=_table_cell(_truncate(str(candidate.get("viewpoint") or ""), 96)),
            )
        )
    for candidate in candidates:
        evidence = candidate.get("evidence_paths") if isinstance(candidate.get("evidence_paths"), list) else []
        draft = candidate.get("short_video_script_draft") if isinstance(candidate.get("short_video_script_draft"), dict) else {}
        post = candidate.get("highlight_post_seed") if isinstance(candidate.get("highlight_post_seed"), dict) else {}
        citations = candidate.get("evidence_citations") if isinstance(candidate.get("evidence_citations"), list) else []
        lines.extend(
            [
                "",
                f"## {candidate.get('id')} `{candidate.get('time_range')}`",
                "",
                f"- Timeline index: `{candidate.get('timeline_index')}`",
                f"- Candidate types: `{', '.join(str(value) for value in candidate.get('candidate_types') or [])}`",
                f"- Summary chapters: `{_candidate_summary_chapter_label(candidate)}`",
                f"- Term correction status: `{candidate.get('term_correction_status') or 'missing'}`",
                f"- Codex term validation: `{candidate.get('term_validation_status') or 'missing'}`",
                "- Review required: `true`",
                "- Publication allowed: `false`",
                "- Allowed as fact: `false`",
                "",
                "### 观点种子",
                "",
                str(candidate.get("viewpoint") or "待人工提炼。"),
                "",
                "### 案例/演示线索",
                "",
                str(candidate.get("case_or_example") or "暂无可靠案例线索。"),
                "",
                "### 可复用表达",
                "",
                str(candidate.get("reusable_quote") or "暂无可直接复用表达。"),
                "",
                "### 短视频脚本种子",
                "",
                f"- Hook: {draft.get('hook') or '待人工补充。'}",
                f"- Body: {draft.get('body') or '待人工补充。'}",
                f"- Evidence note: {draft.get('evidence_note') or '待人工复核。'}",
                f"- CTA: {draft.get('cta') or '待人工补充。'}",
                "",
                "### 精华帖种子",
                "",
                f"- Title seed: {post.get('title_seed') or '待人工补充。'}",
                f"- Core point: {post.get('core_point') or '待人工补充。'}",
                f"- Supporting evidence: {post.get('supporting_evidence') or '待人工复核。'}",
                "",
                "### 证据引用 / Citation Digest",
                "",
                *_candidate_citation_lines(citations),
                "",
                "### 证据路径",
                "",
            ]
        )
        if evidence:
            lines.extend(f"- `{path}`" for path in evidence[:8])
        else:
            lines.append("- `none`")
    return "\n".join(lines).rstrip() + "\n"




def _candidate_summary_chapter_label(candidate: dict[str, Any]) -> str:
    refs = candidate.get("summary_chapter_refs") if isinstance(candidate.get("summary_chapter_refs"), list) else []
    labels = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        index = ref.get("chapter_index")
        title = _text(ref.get("chapter_title"))
        if index:
            labels.append(f"{index}: {title}" if title else str(index))
    return ", ".join(labels) if labels else "not_linked"

def _candidate_citation_lines(citations: list[dict[str, Any]]) -> list[str]:
    if not citations:
        return ["- `not_available`：尚未生成章节级 Citation Digest，需先运行 `build-smart-summary-chapters`。"]
    from .citation_digest import preview_note, evidence_level
    lines: list[str] = [preview_note(min(8, len(citations)), len(citations))]
    for citation in citations[:8]:
        level = evidence_level(citation)
        source_type = _text(citation.get("source_type") or "unknown") + "; evidence_level=" + level
        time = _text(citation.get("time") or "") or "unknown_time"
        text = _truncate(_text(citation.get("text") or ""), 140)
        if level == "inferred" and not any(term in text for term in ("可能", "推断", "疑似", "似乎", "看起来")):
            text = "推断：可能" + text
        indexes = ",".join(str(value) for value in citation.get("timeline_indexes") or []) or "unknown"
        paths = citation.get("evidence_paths") if isinstance(citation.get("evidence_paths"), list) else []
        path_note = f"；evidence={'; '.join(str(path) for path in paths[:3])}" if paths else ""
        lines.append(f"- `{time}` `{source_type}` {text}；timeline={indexes}{path_note}")
    return lines

def _content_candidate_types(item: dict[str, Any], transcript: str, visual_note: str) -> list[str]:
    text = " ".join([transcript, visual_note, str(item.get("visual_route") or "")])
    types: list[str] = []
    if any(word in text for word in ["案例", "比如", "例如", "举例", "故事", "客户", "成交"]):
        types.append("case")
    if any(word in text for word in ["步骤", "流程", "动作", "方法", "原则", "清单", "问题链"]):
        types.append("method")
    if any(word in text for word in ["演示", "展示", "界面", "按钮", "屏幕", "图表", "操作"]):
        types.append("visual_explainer")
    if any(word in text for word in ["观点", "核心", "总结", "本质", "关键", "为什么"]):
        types.append("viewpoint")
    if len(transcript) >= 40 and "viewpoint" not in types:
        types.append("viewpoint")
    return types or ["clip_seed"]


def _candidate_viewpoint(transcript: str, visual_note: str) -> str:
    if transcript:
        return _truncate(transcript, 260)
    if visual_note:
        return _truncate(visual_note.replace("\n", "；"), 260)
    return "待人工提炼观点。"


def _candidate_case_or_example(transcript: str, visual_note: str) -> str:
    text = transcript or visual_note
    if not text:
        return ""
    markers = ["比如", "例如", "案例", "客户", "演示", "展示"]
    for marker in markers:
        pos = text.find(marker)
        if pos >= 0:
            return _truncate(text[pos:], 240)
    return _truncate(visual_note.replace("\n", "；"), 220) if visual_note else ""


def _extract_reusable_quote(transcript: str) -> str:
    text = _text(transcript)
    if not text:
        return ""
    pieces = re.split(r"[。！？!?]\s*", text)
    candidates = [piece.strip() for piece in pieces if 14 <= len(piece.strip()) <= 90]
    if candidates:
        return candidates[0]
    return _truncate(text, 90)


def _candidate_hook(transcript: str, visual_note: str) -> str:
    quote = _extract_reusable_quote(transcript)
    if quote:
        return quote
    if visual_note:
        return _truncate(visual_note.replace("\n", "；"), 90)
    return "这一段可能适合作为内容开头，但需要人工复核。"


def _candidate_title_seed(transcript: str, candidate_types: list[str]) -> str:
    if "method" in candidate_types:
        return "把这一段方法拆成可执行清单"
    if "case" in candidate_types:
        return "这个案例里真正值得复用的点"
    if "visual_explainer" in candidate_types:
        return "看懂这个演示背后的操作逻辑"
    quote = _extract_reusable_quote(transcript)
    return _truncate(quote, 42) if quote else "从这段视频里提炼一个可讨论观点"

def _content_candidate_items(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for fallback_index, item in enumerate(timeline, start=1):
        transcript = _item_transcript(item)
        if not transcript and not (_item_visual_understanding(item) or _item_temporal_understanding(item) or _item_visual_text(item)):
            continue
        score = 0
        score += min(len(transcript), 240) // 40
        if _item_visual_understanding(item):
            score += 3
        if _item_temporal_understanding(item):
            score += 4
        if _item_visual_text(item):
            score += 2
        if item.get("quality_issues"):
            score -= 1
        scored.append((score, fallback_index, item))
    ranked = [item for _, _, item in sorted(scored, key=lambda row: (-row[0], row[1]))]
    duration = _timeline_duration(timeline)
    if duration <= 0 or len(ranked) <= 1:
        return ranked
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for stratum, _, _ in BENCHMARK_STRATA:
        candidate = next(
            (
                item
                for item in ranked
                if id(item) not in selected_ids
                and _benchmark_stratum(item, duration) == stratum
            ),
            None,
        )
        if candidate is None:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
    selected.extend(item for item in ranked if id(item) not in selected_ids)
    return selected


def _coverage_for_renderer(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    coverage = manifest.get("knowledge_coverage") if isinstance(manifest.get("knowledge_coverage"), dict) else {}
    coverage_path = root / "knowledge-coverage.json"
    if coverage_path.exists():
        try:
            loaded = read_json(coverage_path)
        except Exception:
            loaded = {}
        if isinstance(loaded, dict):
            coverage = loaded
    return coverage if isinstance(coverage, dict) else {}


def _build_summary(manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts: dict[str, int] = {}
    for item in timeline:
        route = str(item.get("visual_route") or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
    missing_visual = [
        int(item.get("index") or index)
        for index, item in enumerate(timeline, start=1)
        if _needs_visual_understanding(item)
    ]
    missing_structure = [
        int(item.get("index") or index)
        for index, item in enumerate(timeline, start=1)
        if _needs_document_structure(item)
    ]
    evidence_indexes = evidence_index_sets(timeline)
    return {
        "timeline_items": len(timeline),
        "route_counts": route_counts,
        "items_with_transcript": sum(1 for item in timeline if _text(item.get("transcript"))),
        "items_with_visual_text": sum(1 for item in timeline if _item_visual_text(item) or _human_keep_image(item)),
        "items_with_structured_visual": sum(1 for item in timeline if _structured_entries(item) or _human_structured_fallback(item)),
        "items_with_visual_understanding": len(evidence_indexes["visual_export_consumable"]),
        "items_with_temporal_understanding": len(evidence_indexes["temporal_export_consumable"]),
        "items_with_model_visual_understanding": len(evidence_indexes["visual_model_complete"]),
        "items_with_model_temporal_understanding": len(evidence_indexes["temporal_model_complete"]),
        "visual_evidence_indexes": evidence_indexes,
        "document_visual_missing_structure": missing_structure,
        "visual_understanding_missing": missing_visual,
        "coverage_status": (manifest.get("knowledge_coverage") or {}).get("status") if isinstance(manifest.get("knowledge_coverage"), dict) else "",
    }


def _render_note(
    bundle_dir: Path,
    title: str,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    include_timeline: bool,
    include_full_transcript: bool,
) -> str:
    lines = [
        "---",
        "type: video-knowledge-note",
        f'title: "{_frontmatter_escape(title)}"',
        f'created: "{now_iso()}"',
        "status: draft",
        "tags: [video-knowledge, lecture-note]",
        "---",
        "",
        f"# {title}",
        "",
        "> 这是知识类视频的全量整理稿，不是摘要。缺失项会保留在“待补齐”区，避免把未处理的信息误当成不存在。",
        "",
        "## 视频概要",
        "",
        _auto_overview(title, manifest, timeline, summary),
        "",
        "## 覆盖情况",
        "",
        *_coverage_table_lines(summary),
        "",
        "### 画面类型分布",
        "",
        *_route_table_lines(summary),
        "",
        "## 来源与证据",
        "",
        f"- Bundle: `{bundle_dir}`",
        f"- Review UI: `{bundle_dir / 'review.html'}`",
        f"- Timeline JSON: `{bundle_dir / 'timeline.json'}`",
        f"- Knowledge coverage: `{bundle_dir / 'knowledge-coverage.md'}`",
    ]
    source_lines = _source_lines(manifest)
    if source_lines:
        lines.extend(["", "### 原始来源", ""])
    lines.extend(source_lines)
    lines.extend(["", "## 知识结构", ""])
    lines.extend(_chapter_lines(bundle_dir, timeline))
    lines.extend(["", "## 关键概念与论点", ""])
    lines.extend(_concept_argument_lines(timeline))
    lines.extend(["", "## 图文与视觉信息", ""])
    visual_lines = _visual_knowledge_lines(bundle_dir, timeline, include_temporal=False)
    lines.extend(visual_lines or ["（当前还没有可导出的图文结构化或单帧视觉理解结果。）"])
    lines.extend(["", "## 连续演示与操作变化", ""])
    temporal_lines = _temporal_knowledge_lines(bundle_dir, timeline)
    lines.extend(temporal_lines or ["（当前还没有可导出的连续片段理解结果。）"])
    lines.extend(["", "## 表格、公式、代码与必须保留的图片", ""])
    lines.extend(_retained_media_lines(bundle_dir, timeline))
    if include_timeline:
        lines.extend(["", "## 逐字稿与演示记录", ""])
        if include_full_transcript:
            lines.extend([f"- 完整逐字稿另见：`{bundle_dir / 'exports' / 'full-transcript.md'}`", ""])
        for index, item in enumerate(timeline, start=1):
            lines.extend(_timeline_script_lines(bundle_dir, item, fallback_index=index))
    if include_full_transcript:
        lines.extend(["", "## 完整逐字稿（纯文本）", ""])
        lines.extend(_full_transcript_lines(timeline))
    lines.extend(["", "## 未解决缺口", ""])
    lines.extend(_render_gap_list("图文截图/表格/代码/公式待解析", summary.get("document_visual_missing_structure") or []))
    lines.extend(_render_gap_list("视频视觉理解待补齐", summary.get("visual_understanding_missing") or []))
    lines.extend(["", "## 证据索引", ""])
    lines.extend(_evidence_index_lines(bundle_dir, timeline))
    return "\n".join(lines).rstrip() + "\n"


def _render_full_transcript(
    title: str,
    timeline: list[dict[str, Any]],
    *,
    bundle_dir: Path | None = None,
    manifest: dict[str, Any] | None = None,
    sidecar: dict[str, Any] | None = None,
    transcript_quality_gate: dict[str, Any] | None = None,
) -> str:
    completeness = transcript_completeness_status(bundle_dir) if bundle_dir else {"passed": True}
    source_path = _full_transcript_source_path(bundle_dir, manifest or {}, sidecar or {}) if bundle_dir else None
    if source_path:
        try:
            cues = parse_transcript(source_path)
        except Exception:
            cues = []
        if cues:
            content_scope = load_content_scope(bundle_dir, transcript_path=source_path)
            cues, projection_report = project_transcript_cues(
                cues,
                content_scope,
                projection="full_transcript",
            )
            lines = [f"# {title} - 原始转录", ""]
            if not completeness.get("passed"):
                lines.extend(
                    [
                        "> ⚠️ 状态：`review-required`。这是机器逐字稿，不是人工校对或发布批准版本；转录质量门未通过。",
                        f"> 阻断原因：`transcript_quality:{completeness.get('status')}`。",
                        "",
                    ]
                )
            speaker_notice = _speaker_review_notice(_speaker_review(bundle_dir))
            if speaker_notice:
                lines.extend([f"> {speaker_notice}", ""])
            omitted = int(projection_report.get("projection_omitted_count") or 0)
            review_gaps = int(
                (projection_report.get("disposition_counts") or {}).get("review_gap") or 0
            )
            if omitted or review_gaps:
                lines.extend(
                    [
                        "> 投影审计：可读逐字稿仅清理有确定证据的空白、纯标点或分窗重叠；"
                        f"清理 `{omitted}` 条，待复核 `{review_gaps}` 条。原始 ASR 文件未修改。",
                        "",
                    ]
                )
            lines.extend(_full_transcript_lines_from_cues(cues))
            return "\n".join(lines).rstrip() + "\n"
    lines = [
        f"# {title} - 原始转录",
        "",
        *(
            [
                "> ⚠️ 状态：`review-required`。这是机器逐字稿，不是人工校对或发布批准版本；转录质量门未通过。",
                f"> 阻断原因：`transcript_quality:{completeness.get('status')}`。",
                "",
            ]
            if not completeness.get("passed")
            else []
        ),
        "（未找到可用逐字稿，以下为时间线摘录。）",
        "",
    ]
    lines.extend(_full_transcript_lines(timeline))
    return "\n".join(lines).rstrip() + "\n"


def _build_transcript_projection_audit(
    bundle_dir: Path,
    manifest: dict[str, Any],
    sidecar: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    """Record reader-projection decisions without copying or mutating source ASR."""

    source_path = _full_transcript_source_path(bundle_dir, manifest, sidecar)
    cues: list[Any] = []
    parse_error = ""
    if source_path:
        try:
            cues = parse_transcript(source_path)
        except Exception as exc:
            parse_error = str(exc)
    if not cues:
        cues = _timeline_cues(timeline)
    scope = load_content_scope(bundle_dir, transcript_path=source_path)
    projections: dict[str, Any] = {}
    for projection in ("full_transcript", "continuous_full_body", "smart_summary"):
        _projected, report = project_transcript_cues(cues, scope, projection=projection)
        projections[projection] = content_scope_projection_summary(
            scope,
            projection=projection,
            cue_report=report,
        )
    return {
        "schema": "video_knowledge_pipeline.transcript_projection_audit.v1",
        "created_at": now_iso(),
        "bundle_dir": str(bundle_dir),
        "source_path": str(source_path or ""),
        "source_sha256": _file_sha256(source_path) if source_path and source_path.is_file() else "",
        "source_cue_count": len(cues),
        "source_parse_error": parse_error,
        "content_scope_path": str(scope.get("path") or ""),
        "content_scope_sha256": str(scope.get("content_scope_sha256") or ""),
        "dispositions": [
            "retained_semantic",
            "excluded_confirmed",
            "normalized_disfluency",
            "review_gap",
        ],
        "projections": projections,
        "evidence_boundary": {
            "source_transcript_modified": False,
            "raw_asr_preserved": True,
            "reader_outputs_are_rebuildable_projections": True,
            "low_confidence_alone_authorizes_deletion": False,
        },
    }


def _build_benchmark_script(
    bundle_dir: Path,
    title: str,
    timeline: list[dict[str, Any]],
    *,
    production_artifact_gate: dict[str, Any],
    smart_summary_quality: dict[str, Any],
) -> dict[str, Any]:
    """Build an evidence-bound benchmark outline without claiming final copy.

    The selector intentionally reserves one slot per timeline stratum before
    filling remaining slots by quality score.  This prevents a dense opening
    from crowding the middle and conclusion out of long-video script evidence.
    """

    duration = _timeline_duration(timeline)
    available = _content_candidate_items(timeline)
    selected = sorted(
        available[:6],
        key=lambda item: (
            _float_seconds(item.get("start")),
            _float_seconds(item.get("end")),
            int(item.get("index") or 0),
        ),
    )
    beats: list[dict[str, Any]] = []
    timeline_path = bundle_dir / "timeline.json"
    for order, item in enumerate(selected, start=1):
        transcript = _item_transcript(item)
        visual_note = _transcript_visual_note(item)
        paths = _evidence_paths(
            bundle_dir,
            item,
            visual=_item_visual_understanding(item),
            temporal=_item_temporal_understanding(item),
        )
        timeline_index = int(item.get("index") or order)
        stratum = _benchmark_stratum(item, duration)
        time_range = _time_range(item)
        evidence_links = [
            {
                "kind": "timeline",
                "path": str(timeline_path),
                "timeline_index": timeline_index,
                "time_range": time_range,
            }
        ]
        evidence_links.extend(
            {
                "kind": "visual_or_media",
                "path": path,
                "timeline_index": timeline_index,
                "time_range": time_range,
            }
            for path in paths[:8]
        )
        beats.append(
            {
                "order": order,
                "stratum": stratum,
                "planned_role": _benchmark_role(stratum),
                "timeline_index": timeline_index,
                "start": _float_seconds(item.get("start")),
                "end": _float_seconds(item.get("end")),
                "time_range": time_range,
                "observed_excerpt": _truncate(transcript, 320) or "（该时间段没有可靠转写。）",
                "visual_expression": _truncate(visual_note, 360),
                "evidence_links": evidence_links,
                "quality_issues": [str(value) for value in item.get("quality_issues") or []],
            }
        )

    selected_strata = _dedupe(
        beat["stratum"] for beat in beats if beat.get("stratum") != "unknown"
    )
    available_strata = _dedupe(
        _benchmark_stratum(item, duration)
        for item in available
        if _benchmark_stratum(item, duration) != "unknown"
    )
    latest_evidence_seconds = max((float(beat["end"]) for beat in beats), default=0.0)
    latest_evidence_ratio = latest_evidence_seconds / duration if duration > 0 else 0.0
    is_long_video = duration > 120.0
    coverage_blocking_reasons: list[str] = []
    if not beats:
        coverage_blocking_reasons.append("no_evidence_candidates")
    if duration <= 0:
        coverage_blocking_reasons.append("duration_unknown")
    if is_long_video and len(selected_strata) < 4:
        coverage_blocking_reasons.append("long_video_requires_four_timeline_strata")
    if is_long_video and latest_evidence_ratio < 0.8:
        coverage_blocking_reasons.append("long_video_latest_evidence_before_eighty_percent")
    coverage_passed = not coverage_blocking_reasons

    production_passed = bool(production_artifact_gate.get("formal_generation_allowed"))
    summary_passed = bool(smart_summary_quality.get("passed"))
    source_quality_passed = production_passed and summary_passed
    quality_blocking_reasons: list[str] = []
    if not production_passed:
        quality_blocking_reasons.append(
            f"production_artifact_gate:{production_artifact_gate.get('status') or 'not_ready'}"
        )
        quality_blocking_reasons.extend(
            str(value) for value in production_artifact_gate.get("blocking_reasons") or []
        )
    if not summary_passed:
        quality_blocking_reasons.append(
            f"smart_summary_quality:{smart_summary_quality.get('status') or 'not_ready'}"
        )

    deliverable_status = (
        "needs_review" if coverage_passed and source_quality_passed else "draft"
    )
    hook_beat = next(
        (beat for beat in beats if beat["stratum"] == "opening"),
        beats[0] if beats else {},
    )
    cta_beat = next(
        (beat for beat in reversed(beats) if beat["stratum"] == "conclusion"),
        beats[-1] if beats else {},
    )
    visual_beats = [beat for beat in beats if beat.get("visual_expression")]
    evidence_links = [
        link
        for beat in beats
        for link in beat.get("evidence_links") or []
        if isinstance(link, dict)
    ]
    risk_notes = [
        "逐字稿、画面理解和数字事实必须回到时间戳证据复核；不得把模型补写当作原视频事实。",
        "原创改写必须替换表达、案例和叙事顺序，不逐字复制来源内容。",
        "发布前仍需人工检查隐私、版权、医疗/保险/法律/财务等高风险表述。",
    ]
    risk_notes.extend(
        f"覆盖门未通过：{reason}" for reason in coverage_blocking_reasons
    )
    risk_notes.extend(f"质量门未通过：{reason}" for reason in quality_blocking_reasons)

    return {
        "schema": BENCHMARK_SCRIPT_SCHEMA,
        "created_at": now_iso(),
        "title": title,
        "bundle_dir": str(bundle_dir),
        "status": deliverable_status,
        "review_required": True,
        "publication_allowed": False,
        "final_allowed": False,
        "status_reason": (
            "timeline_coverage_and_source_quality_ready_for_human_review"
            if deliverable_status == "needs_review"
            else "coverage_or_source_quality_incomplete"
        ),
        "selection": {
            "strategy": "timeline_stratified_then_global_score",
            "strata": [name for name, _, _ in BENCHMARK_STRATA],
            "available_candidate_count": len(available),
            "selected_candidate_count": len(beats),
            "available_strata": available_strata,
            "selected_strata": selected_strata,
        },
        "coverage": {
            "duration_seconds": round(duration, 3),
            "is_long_video": is_long_video,
            "required_strata_count": 4 if is_long_video else 1,
            "selected_strata_count": len(selected_strata),
            "latest_evidence_seconds": round(latest_evidence_seconds, 3),
            "latest_evidence_ratio": round(latest_evidence_ratio, 6),
            "required_latest_evidence_ratio": 0.8 if is_long_video else None,
            "passed": coverage_passed,
            "blocking_reasons": coverage_blocking_reasons,
        },
        "quality": {
            "passed": source_quality_passed,
            "production_artifact_gate_status": str(
                production_artifact_gate.get("status") or "not_ready"
            ),
            "production_artifact_gate_passed": production_passed,
            "smart_summary_quality_status": str(
                smart_summary_quality.get("status") or "not_ready"
            ),
            "smart_summary_quality_passed": summary_passed,
            "blocking_reasons": _dedupe(quality_blocking_reasons),
        },
        "structure_analysis": [
            {
                "order": beat["order"],
                "stratum": beat["stratum"],
                "planned_role": beat["planned_role"],
                "timeline_index": beat["timeline_index"],
                "time_range": beat["time_range"],
                "observed_excerpt": beat["observed_excerpt"],
                "evidence_links": beat["evidence_links"],
            }
            for beat in beats
        ],
        "hook": {
            "status": "source_candidate_needs_original_rewrite",
            "draft": _candidate_hook(
                str(hook_beat.get("observed_excerpt") or ""),
                str(hook_beat.get("visual_expression") or ""),
            )
            if hook_beat
            else "",
            "timeline_index": hook_beat.get("timeline_index"),
            "time_range": hook_beat.get("time_range", ""),
            "evidence_links": hook_beat.get("evidence_links", []),
        },
        "argument_rhythm": {
            "sequence": [beat["stratum"] for beat in beats],
            "beat_count": len(beats),
            "beats": beats,
        },
        "visual_expression": {
            "status": "ready_for_review" if visual_beats else "needs_input",
            "covered_beat_count": len(visual_beats),
            "items": [
                {
                    "timeline_index": beat["timeline_index"],
                    "time_range": beat["time_range"],
                    "stratum": beat["stratum"],
                    "observation": beat["visual_expression"],
                    "evidence_links": beat["evidence_links"],
                }
                for beat in visual_beats
            ],
        },
        "cta": {
            "status": "source_candidate_needs_original_rewrite",
            "source_excerpt": str(cta_beat.get("observed_excerpt") or ""),
            "instruction": "结合自己的受众与下一步行动重新设计 CTA，不把结尾片段自动认定为原作者 CTA。",
            "timeline_index": cta_beat.get("timeline_index"),
            "time_range": cta_beat.get("time_range", ""),
            "evidence_links": cta_beat.get("evidence_links", []),
        },
        "original_rewrite_outline": {
            "status": "draft",
            "hook_instruction": "保留来源问题意识，用自己的措辞、经历或案例重新写开场。",
            "beat_instructions": [
                {
                    "stratum": beat["stratum"],
                    "time_range": beat["time_range"],
                    "instruction": f"围绕“{_truncate(beat['observed_excerpt'], 80)}”提炼一个原创信息点，并回链原时间戳复核。",
                }
                for beat in beats
            ],
            "cta_instruction": "只在完成事实与风险复核后，加入符合自身账号定位的具体行动邀请。",
        },
        "originality_notes": [
            "只复用结构、问题意识和证据组织方式，不复制原句或身份化案例。",
            "新增观点必须标注为改写者判断，不能冒充来源事实。",
            "保留所有时间戳和证据路径，便于逐段回看。",
        ],
        "risk_notes": risk_notes,
        "timestamp_evidence_links": evidence_links,
    }


def _render_benchmark_script_markdown(script: dict[str, Any]) -> str:
    coverage = script.get("coverage") if isinstance(script.get("coverage"), dict) else {}
    quality = script.get("quality") if isinstance(script.get("quality"), dict) else {}
    lines = [
        f"# {script.get('title') or '视频'} - 对标脚本证据稿",
        "",
        f"- Status: `{script.get('status')}`",
        "- Review required: `true`",
        "- Publication allowed: `false`",
        "- Final allowed: `false`",
        f"- Selection: `{script.get('selection', {}).get('strategy', '')}`",
        f"- Timeline coverage: `{coverage.get('passed')}`; strata `{coverage.get('selected_strata_count', 0)}` / `{coverage.get('required_strata_count', 0)}`; latest `{float(coverage.get('latest_evidence_ratio') or 0):.1%}`",
        f"- Source quality: `{quality.get('passed')}`",
        "",
        "> 这是证据绑定的原创改写草稿，不是可直接发布的成稿。质量门或覆盖门未通过时只能继续补证据和人工复核。",
        "",
        "## 结构拆解",
        "",
    ]
    structures = script.get("structure_analysis") if isinstance(script.get("structure_analysis"), list) else []
    if not structures:
        lines.append("（暂无可用时间线证据。）")
    for row in structures:
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                f"### {row.get('order')}. {row.get('stratum')} · {row.get('time_range')}",
                "",
                f"- 结构角色：{row.get('planned_role')}",
                f"- 来源摘录：{row.get('observed_excerpt')}",
                *_benchmark_evidence_markdown(row.get("evidence_links")),
                "",
            ]
        )
    hook = script.get("hook") if isinstance(script.get("hook"), dict) else {}
    lines.extend(
        [
            "## 开头钩子",
            "",
            str(hook.get("draft") or "（待补充。）"),
            "",
            f"- 状态：`{hook.get('status') or 'needs_input'}`",
            f"- 时间戳：`{hook.get('time_range') or ''}`",
            *_benchmark_evidence_markdown(hook.get("evidence_links")),
            "",
            "## 论证节奏 / Beats",
            "",
        ]
    )
    rhythm = script.get("argument_rhythm") if isinstance(script.get("argument_rhythm"), dict) else {}
    for beat in rhythm.get("beats") or []:
        if not isinstance(beat, dict):
            continue
        lines.extend(
            [
                f"### Beat {beat.get('order')} · {beat.get('stratum')} · {beat.get('time_range')}",
                "",
                f"- 作用：{beat.get('planned_role')}",
                f"- 来源线索：{beat.get('observed_excerpt')}",
                f"- 画面线索：{beat.get('visual_expression') or '未提取到可靠视觉表达，需人工补齐。'}",
                *_benchmark_evidence_markdown(beat.get("evidence_links")),
                "",
            ]
        )
    visual = script.get("visual_expression") if isinstance(script.get("visual_expression"), dict) else {}
    lines.extend(["## 视觉表达", "", f"- 状态：`{visual.get('status') or 'needs_input'}`"])
    for item in visual.get("items") or []:
        if not isinstance(item, dict):
            continue
        lines.extend(
            [
                f"- `{item.get('time_range')}` [{item.get('stratum')}] {item.get('observation')}",
                *_benchmark_evidence_markdown(item.get("evidence_links"), indent="  "),
            ]
        )
    if not visual.get("items"):
        lines.append("- 未提取到可可靠复用的视觉说明；不得凭空补写镜头设计。")
    cta = script.get("cta") if isinstance(script.get("cta"), dict) else {}
    lines.extend(
        [
            "",
            "## CTA",
            "",
            f"- 来源结尾线索：{cta.get('source_excerpt') or '（暂无。）'}",
            f"- 改写要求：{cta.get('instruction') or ''}",
            f"- 时间戳：`{cta.get('time_range') or ''}`",
            *_benchmark_evidence_markdown(cta.get("evidence_links")),
            "",
            "## 原创改写框架",
            "",
        ]
    )
    rewrite = script.get("original_rewrite_outline") if isinstance(script.get("original_rewrite_outline"), dict) else {}
    lines.append(f"- 开场：{rewrite.get('hook_instruction') or ''}")
    for item in rewrite.get("beat_instructions") or []:
        if isinstance(item, dict):
            lines.append(
                f"- `{item.get('time_range')}` [{item.get('stratum')}] {item.get('instruction')}"
            )
    lines.extend(
        [
            f"- 结尾：{rewrite.get('cta_instruction') or ''}",
            "",
            "## 原创性说明",
            "",
            *[f"- {value}" for value in script.get("originality_notes") or []],
            "",
            "## 风险与待复核项",
            "",
            *[f"- {value}" for value in script.get("risk_notes") or []],
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _benchmark_evidence_markdown(value: Any, *, indent: str = "") -> list[str]:
    links = value if isinstance(value, list) else []
    rows: list[str] = []
    for index, link in enumerate(links, start=1):
        if not isinstance(link, dict):
            continue
        path = str(link.get("path") or "").strip()
        if not path:
            continue
        target = path.replace("\\", "/").replace(">", "%3E")
        rows.append(
            f"{indent}- 证据 {index}: [{link.get('kind') or 'source'}](<{target}>) · `{link.get('time_range') or ''}`"
        )
    return rows


def _benchmark_role(stratum: str) -> str:
    return {
        "opening": "开场、问题设置与观看承诺候选",
        "early": "背景、定义与前置条件候选",
        "middle": "核心论证、方法或案例展开候选",
        "late": "反证、收束或行动框架候选",
        "conclusion": "结论、复盘与 CTA 候选",
    }.get(stratum, "待人工判断的结构位置")


def _benchmark_stratum(item: dict[str, Any], duration: float) -> str:
    if duration <= 0:
        return "unknown"
    ratio = min(1.0, _float_seconds(item.get("start")) / duration)
    for name, lower, upper in BENCHMARK_STRATA:
        if lower <= ratio < upper:
            return name
    return "conclusion"


def _render_full_body(
    title: str,
    timeline: list[dict[str, Any]],
    *,
    bundle_dir: Path | None = None,
    manifest: dict[str, Any] | None = None,
    sidecar: dict[str, Any] | None = None,
) -> str:
    """Render a continuous, verbatim reading projection of transcript cues.

    Intent: provide a reader-facing full body in addition to the timestamped
    transcript and Smart Summary.
    Decision: reuse the same transcript source selection as the full transcript,
    remove cue metadata, and only aggregate whitespace-delimited paragraphs.
    Reason: a second LLM rewrite would risk omissions and factual drift.
    Evidence: the A option approved for the 2026-08-22 continuous-body design.
    Effective scope: presentation only; source transcript files and cue content
    are never modified.
    """

    completeness = transcript_completeness_status(bundle_dir) if bundle_dir else {"passed": True}
    cues: list[Any] = []
    source_path = _full_transcript_source_path(bundle_dir, manifest or {}, sidecar or {}) if bundle_dir else None
    if source_path:
        try:
            cues = parse_transcript(source_path)
        except Exception:
            cues = []
    if not cues:
        cues = _timeline_cues(timeline)
    content_scope = load_content_scope(bundle_dir, transcript_path=source_path) if bundle_dir else {
        "status": "not_configured",
        "excluded_intervals": [],
    }
    cues, cue_scope_report = project_transcript_cues(
        cues,
        content_scope,
        projection="continuous_full_body",
    )
    scope_summary = content_scope_projection_summary(
        content_scope,
        projection="continuous_full_body",
        cue_report=cue_scope_report,
    )
    lines = [f"# {title} - 正文", ""]
    if not completeness.get("passed"):
        lines.extend(
            [
                "> ⚠️ 状态：`review-required`。这是机器逐字稿的连续阅读版，不是人工校对或发布批准版本；转录质量门未通过。",
                f"> 阻断原因：`transcript_quality:{completeness.get('status')}`。",
                "",
            ]
        )
    if scope_summary.get("active"):
        ranges = ", ".join(
            f"{format_timestamp(_float_seconds(row.get('start')))}–{format_timestamp(_float_seconds(row.get('end')))} ({row.get('reason')})"
            for row in scope_summary.get("excluded_intervals") or []
            if isinstance(row, dict)
        )
        lines.extend(
            [
                f"> 内容范围：已排除 `{ranges}`；被排除内容仍完整保留在原始逐字稿和中间证据中。",
                "",
            ]
        )
    lines.extend(_continuous_body_lines_from_cues(cues))
    return "\n\n".join(line for line in lines if line != "").rstrip() + "\n"


def _continuous_body_lines_from_cues(
    cues: list[Any],
    *,
    target_chars: int = 600,
) -> list[str]:
    """Merge cue text into paragraphs without rewriting or reordering it."""

    paragraphs: list[str] = []
    current: list[str] = []
    current_chars = 0
    threshold = max(1, int(target_chars))
    for cue in cues:
        text = _text(getattr(cue, "text", ""))
        if not text:
            continue
        current.append(text)
        current_chars += len(text) + (1 if len(current) > 1 else 0)
        if current_chars >= threshold:
            paragraphs.append(" ".join(current))
            current = []
            current_chars = 0
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs or ["（无可用正文。）"]


def _artifact_review_watermark(text: str, *, artifact_label: str, gate: dict[str, Any]) -> str:
    marker = "> ⚠️ 状态：`review-required`。"
    if marker in text:
        return text
    reasons = ", ".join(str(value) for value in gate.get("blocking_reasons") or []) or "quality_review_required"
    lines = str(text or "").splitlines()
    insert_at = 1 if lines and lines[0].startswith("#") else 0
    watermark = [
        "",
        f"{marker} 这是{artifact_label}，不可视为正式成品或发布依据。",
        f"> 阻断原因：`{reasons}`。",
        "",
    ]
    lines[insert_at:insert_at] = watermark
    return "\n".join(lines).rstrip() + "\n"


def _full_transcript_quality_gate_lines(gate: dict[str, Any] | None) -> list[str]:
    if not isinstance(gate, dict) or not gate:
        return ["- Transcript quality gate: `not_run`"]
    status = str(gate.get("status") or "unknown")
    return [
        f"- Transcript quality gate: `{status}`",
        f"- Transcript quality OK: `{str(bool(gate.get('ok'))).lower()}`",
        f"- Transcript quality fail/warning: `{int(gate.get('fail_count') or 0)}` / `{int(gate.get('warning_count') or 0)}`",
        f"- Transcript punctuation density: `{gate.get('punctuation_density_per_1000_chars', '')}` / 1000 chars",
    ]


def _full_transcript_source_path(bundle_dir: Path, manifest: dict[str, Any], sidecar: dict[str, Any]) -> Path | None:
    candidates: list[str] = []
    for key in (
        "source_arbitrated_transcript_json",
        "human_corrected_transcript_json",
        "llm_readable_transcript_json",
        "agent_readable_transcript_json",
        "readable_transcript_json",
        "llm_corrected_transcript_json",
        "corrected_transcript_json",
        "source_arbitrated_transcript_srt",
        "human_corrected_transcript_srt",
        "llm_readable_transcript_srt",
        "agent_readable_transcript_srt",
        "llm_corrected_transcript_srt",
        "corrected_transcript_srt",
    ):
        value = manifest.get(key)
        if value:
            candidates.append(str(value))
    candidates.extend([
        "source-arbitrated-transcript.json",
        "source-arbitrated-transcript.srt",
        "human-corrected-transcript.json",
        "human-corrected-transcript.srt",
        "llm-readable-transcript.json",
        "llm-readable-transcript.srt",
        "agent-readable-transcript.json",
        "agent-readable-transcript.srt",
        "readable-transcript.json",
        "llm-corrected-transcript.json",
        "llm-corrected-transcript.srt",
        "corrected-transcript.json",
        "corrected-transcript.srt",
    ])
    for key in ("json_path", "path", "source_path"):
        value = sidecar.get(key)
        if value:
            candidates.append(str(value))
    for key in (
        "normalized_transcript_json",
        "source_transcript",
        "transcript_path",
        "transcript_json",
        "normalized_transcript_srt",
        "transcript_srt",
    ):
        value = manifest.get(key)
        if value:
            candidates.append(str(value))
    candidates.extend(["normalized-transcript.json", "normalized-transcript.srt", "transcript.json", "transcript.srt"])
    for value in candidates:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = bundle_dir / path
        if path.exists() and "timeline-transcript" not in path.name.lower():
            return path.resolve()
    return None

def _full_transcript_source_label(sidecar: dict[str, Any], source_path: Path) -> str:
    name = source_path.name.lower()
    if "llm-readable" in name:
        return "llm_readable_transcript"
    if "agent-readable" in name:
        return "agent_readable_transcript"
    if "readable" in name:
        return "readable_transcript"
    if "source-arbitrated" in name:
        return "source_arbitrated_transcript"
    if "llm-corrected" in name:
        return "llm_corrected_transcript"
    if "human-corrected" in name:
        return "human_corrected_transcript"
    if "corrected" in name:
        return "corrected_transcript"
    if "normalized" in name:
        return "normalized_asr"
    source = str(sidecar.get("source") or "").strip()
    if source:
        return source
    return "asr"


def canonical_export_integrity_status(bundle_dir: str | Path, *, reader_payloads: dict[str, bytes] | None = None) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    validation_basis = 'prospective_reader_bytes' if reader_payloads is not None else 'stored_reader_bytes'
    if reader_payloads is None:
        from .transcript_semantic_transaction import summary_transaction_status
        pending = summary_transaction_status(root)
        if not pending['passed']:
            return {'status': 'blocked_export_transaction', 'passed': False,
                    'issues': [{'key': 'pending_artifact_transaction', 'detail': str(pending)}]}
    def exists(path):
        return str(path) in reader_payloads if reader_payloads is not None and path.name in {
            'full-transcript.md', 'full-body.md', 'knowledge-note.md', 'reader-export-receipt.json'} else path.exists()
    def content(path):
        return reader_payloads[str(path)].decode('utf-8-sig') if reader_payloads is not None and str(path) in reader_payloads else path.read_text(encoding='utf-8-sig')
    def digest(path):
        return hashlib.sha256(reader_payloads[str(path)]).hexdigest() if reader_payloads is not None and str(path) in reader_payloads else _file_sha256(path)
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        manifest = {}
    canonical = _canonical_transcript_path(root, manifest)
    if canonical is None:
        return {
            "status": "not_applicable_pre_arbitration",
            "validation_basis": validation_basis,
            "passed": True,
            "canonical_path": "",
            "issues": [],
        }
    canonical_hash = _file_sha256(canonical)
    full_path = root / "exports" / "full-transcript.md"
    full_body_path = root / "exports" / "full-body.md"
    knowledge_note_path = root / "exports" / "knowledge-note.md"
    pack_path = root / "exports" / "smart-summary-input-pack.json"
    receipt_path = root / "exports" / "reader-export-receipt.json"
    full_text = content(full_path) if exists(full_path) else ""
    full_body_text = content(full_body_path) if exists(full_body_path) else ""
    knowledge_note_text = content(knowledge_note_path) if exists(knowledge_note_path) else ""
    pack = read_json(pack_path) if pack_path.exists() else {}
    receipt = json.loads(content(receipt_path)) if exists(receipt_path) else {}
    if not isinstance(pack, dict):
        pack = {}
    if not isinstance(receipt, dict):
        receipt = {}
    receipt_matches = (
        exists(full_path)
        and exists(full_body_path)
        and exists(knowledge_note_path)
        and receipt_matches_reader_files(
            receipt,
            canonical_transcript=canonical,
            full_transcript=full_path,
            full_body=full_body_path,
            reading_note=knowledge_note_path,
            reader_payloads=reader_payloads,
        )
    )
    full_header_hash = ""
    match = re.search(r"Canonical source SHA-256: `([0-9a-f]{64})`", full_text)
    if match:
        full_header_hash = match.group(1)
    full_body_header_hash = ""
    body_match = re.search(r"Canonical source SHA-256: `([0-9a-f]{64})`", full_body_text)
    if body_match:
        full_body_header_hash = body_match.group(1)
    knowledge_note_hash = ""
    note_match = re.search(r"Canonical transcript SHA-256: `([0-9a-f]{64})`", knowledge_note_text)
    if note_match:
        knowledge_note_hash = note_match.group(1)
    if receipt_matches:
        full_header_hash = canonical_hash
        full_body_header_hash = canonical_hash
        knowledge_note_hash = canonical_hash
    pack_source = Path(str(pack.get("transcript_source") or "")) if pack else Path()
    if pack_source and not pack_source.is_absolute():
        pack_source = root / pack_source
    try:
        pack_source_resolved = pack_source.resolve() if str(pack_source) else None
    except OSError:
        pack_source_resolved = None
    issues: list[dict[str, str]] = []
    if not exists(full_path):
        issues.append({"key": "full_transcript_missing", "detail": str(full_path)})
    elif full_header_hash != canonical_hash:
        issues.append({"key": "full_transcript_canonical_hash_mismatch", "detail": full_header_hash})
    if not exists(full_body_path):
        issues.append({"key": "full_body_missing", "detail": str(full_body_path)})
    elif full_body_header_hash != canonical_hash:
        issues.append({"key": "full_body_canonical_hash_mismatch", "detail": full_body_header_hash})
    if not exists(knowledge_note_path):
        issues.append({"key": "knowledge_note_missing", "detail": str(knowledge_note_path)})
    elif knowledge_note_hash != canonical_hash:
        issues.append({"key": "knowledge_note_canonical_hash_mismatch", "detail": knowledge_note_hash})
    if not pack_path.exists():
        issues.append({"key": "smart_summary_input_pack_missing", "detail": str(pack_path)})
    elif pack_source_resolved != canonical.resolve():
        issues.append({"key": "smart_summary_source_path_mismatch", "detail": str(pack_source_resolved or "")})
    if str(pack.get("transcript_source_sha256") or "") != canonical_hash:
        issues.append(
            {
                "key": "smart_summary_source_hash_mismatch",
                "detail": str(pack.get("transcript_source_sha256") or ""),
            }
        )
    return {
        "status": "passed" if not issues else "blocked_canonical_export_mismatch",
        "validation_basis": validation_basis,
        "passed": not issues,
        "canonical_path": str(canonical),
        "canonical_sha256": canonical_hash,
        "full_transcript_path": str(full_path),
        "full_transcript_sha256": digest(full_path) if exists(full_path) else "",
        "full_body_path": str(full_body_path),
        "full_body_sha256": digest(full_body_path) if exists(full_body_path) else "",
        "knowledge_note_path": str(knowledge_note_path),
        "knowledge_note_sha256": digest(knowledge_note_path) if exists(knowledge_note_path) else "",
        "smart_summary_input_pack_path": str(pack_path),
        "issues": issues,
    }


def _timeline_with_canonical_transcript(
    root: Path,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay canonical speech on timeline copies used by readable exports."""

    canonical_path = _canonical_transcript_path(root, manifest)
    if canonical_path is None:
        return [dict(row) for row in timeline]
    payload = read_json(canonical_path)
    segments = [
        dict(row)
        for row in payload.get("segments") or []
        if isinstance(row, dict)
    ] if isinstance(payload, dict) else []
    if not segments:
        return [dict(row) for row in timeline]
    canonical_hash = _file_sha256(canonical_path)
    result: list[dict[str, Any]] = []
    for item in timeline:
        row = dict(item)
        start = _float_seconds(row.get("start"))
        end = max(start, _float_seconds(row.get("end")))
        selected: list[str] = []
        for segment in segments:
            segment_start = _float_seconds(segment.get("start"))
            segment_end = max(segment_start, _float_seconds(segment.get("end")))
            overlap = min(end, segment_end) - max(start, segment_start)
            if overlap <= 0:
                continue
            text = _text(segment.get("text") or segment.get("corrected_text"))
            if text and (not selected or selected[-1] != text):
                selected.append(text)
        if selected:
            existing = _text(row.get("transcript") or row.get("original_transcript"))
            if existing and not _text(row.get("original_transcript")):
                row["original_transcript"] = existing
            row["corrected_transcript"] = " ".join(selected)
            row["canonical_transcript_source"] = str(canonical_path)
            row["canonical_transcript_sha256"] = canonical_hash
        result.append(row)
    return result


def _bind_knowledge_note_to_canonical(
    root: Path,
    manifest: dict[str, Any],
    markdown: str,
) -> str:
    canonical = _canonical_transcript_path(root, manifest)
    if canonical is None:
        return markdown
    marker = f"> Canonical transcript SHA-256: `{_file_sha256(canonical)}`"
    if marker in markdown:
        return markdown
    anchor = "> 这是知识类视频的结构化整理稿。主文档只保留可读知识与证据入口；完整逐字稿和逐项审计见对应导出文件。"
    if anchor in markdown:
        return markdown.replace(anchor, f"{anchor}\n>\n{marker}", 1)
    return f"{marker}\n\n{markdown}"


def _canonical_transcript_path(root: Path, manifest: dict[str, Any]) -> Path | None:
    values = [
        str(manifest.get("source_arbitrated_transcript_json") or ""),
        "source-arbitrated-transcript.json",
    ]
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            return path.resolve()
    return None


def _full_transcript_lines_from_cues(cues: list[Any], *, source_label: str = "", arbitration_status: str = "") -> list[str]:
    lines: list[str] = []
    labels = speaker_label_map(cues)
    for cue in cues:
        text = _text(getattr(cue, "text", ""))
        if not text:
            continue
        start = format_timestamp(float(getattr(cue, "start", 0.0) or 0.0))
        end = format_timestamp(float(getattr(cue, "end", 0.0) or 0.0))
        speaker = speaker_display_name(cue, labels)
        lines.extend([
            f"### {start} - {end}",
            "",
            *([f"**{speaker}**", ""] if speaker else []),
            text,
            "",
        ])
    return lines or ["（无转写。）"]


def _full_transcript_arbitration_status(source_label: str) -> str:
    label = str(source_label or "").strip().lower()
    if label in {"llm_readable_transcript", "agent_readable_transcript", "readable_transcript"}:
        return "readable_after_correction_or_postprocess"
    if label in {"source_arbitrated_transcript", "llm_corrected_transcript", "human_corrected_transcript"}:
        return "arbitrated_or_reviewed"
    if label == "corrected_transcript":
        return "corrected_postprocess_or_arbitrated"
    if label == "normalized_asr":
        return "raw_asr_fallback_not_arbitrated"
    if label == "timeline_fallback":
        return "timeline_fallback_not_arbitrated"
    return "unknown"


def _render_llm_summary_required(title: str, bundle_dir: Path) -> str:
    return "\n".join(
        [
            f"# {title} - 智能总结",
            "",
            "生成状态：needs_llm_summary。",
            "",
            "本文件当前不包含规则拼接的总结正文。VKP 已准备纠正版逐字稿、语义章节和视觉证据包；必须由 Codex、其他 agent 或已配置的在线文本大模型完成章节级总结后，才会写入最终智能总结。",
            "",
            "## 下一步",
            "",
            f"- 章节级 LLM：.\\scripts\\video-knowledge.ps1 run-smart-summary-section-llm-rewrite '{bundle_dir}' --auto-from-profile",
            f"- Codex/agent 手工交接：读取 {bundle_dir / 'exports' / 'smart-summary-codex-prompt.md'}，生成带 codex_final 或 codex_llm_rewrite_final 标记的 Markdown。",
            f"- 安装结果：.\\scripts\\video-knowledge.ps1 generate-smart-summary-with-codex '{bundle_dir}' --input-md '<LLM 输出路径>'",
            "",
            "规则仅用于证据准备、事实保护和最低质量验收，不负责生成总结正文。",
            "",
        ]
    )

def _render_smart_summary(
    title: str,
    bundle_dir: Path,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    summary: dict[str, Any],
    sidecar: dict[str, Any],
) -> str:
    codex_ready = _existing_codex_smart_summary(bundle_dir)
    if codex_ready:
        return codex_ready.read_text(encoding="utf-8-sig")
    cues, transcript_source, transcript_note = _smart_summary_cues(bundle_dir, manifest, sidecar, timeline)
    segments = _smart_summary_segments(cues, target_segments=10)
    duration = _smart_summary_duration(cues, timeline, manifest)
    source_path = _smart_source_path(manifest)
    visual_status = _smart_visual_status(summary)
    lines = [
        f"# {title} - 智能总结",
        "",
        "生成方式：`codex_assisted_draft`。这是给 Codex/LLM 复写前可直接阅读的本地草稿；只有存在带 `codex_final` 或 `codex_llm_rewrite_final` 标记的成品总结时，导出器才会采用它。",
        "",
        "## 基本信息",
        "",
        f"- 视频名：{title}",
        f"- 时长：`{_format_time(duration)}`",
        f"- 处理时间：`{now_iso()}`",
        f"- 来源路径：`{source_path or '(unknown)'}`",
        f"- 转写来源：`{transcript_source}`",
        f"- 转写覆盖：`{len(cues)}` cues，覆盖至 `{_format_time(duration)}`",
        f"- 视觉证据状态：{visual_status}",
        "",
        "## 一句话概览",
        "",
        _smart_one_sentence_overview(title, cues, segments),
        "",
        "## 核心主题 / 课程主线",
        "",
    ]
    lines.extend(_smart_mainline_lines(cues, segments))
    lines.extend(["", "## 分段总结", ""])
    lines.extend(_smart_segment_lines(segments))
    lines.extend(["", "## 关键观点 / 方法论", ""])
    lines.extend(_smart_key_point_lines(cues))
    lines.extend(["", "## 可执行动作清单", ""])
    lines.extend(_smart_action_lines(cues))
    lines.extend(["", "## 高频话术 / 可复用表达", ""])
    lines.extend(_smart_reusable_expression_lines(cues))
    lines.extend(["", "## 待复核点 / 低置信内容", ""])
    lines.extend(_smart_review_lines(summary, transcript_note, visual_status))
    lines.extend(
        [
            "",
            "## 相关产物",
            "",
            f"- 证据审计笔记：`{bundle_dir / 'exports' / 'knowledge-note.md'}`",
            f"- 完整逐字稿：`{bundle_dir / 'exports' / 'full-transcript.md'}`",
            f"- Codex 改写提示：`{bundle_dir / 'exports' / 'smart-summary-codex-prompt.md'}`",
            f"- 总结输入包：`{bundle_dir / 'exports' / 'smart-summary-input-pack.md'}`",
            f"- 长视频记忆包：`{bundle_dir / 'exports' / 'long-video-memory-pack.md'}`",
            f"- 提取审计：`{bundle_dir / 'exports' / 'extraction-audit.md'}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _smart_summary_candidate_ready_for_refresh_install(quality: dict[str, Any]) -> bool:
    """Allow installing a good Codex/LLM summary candidate when the only blocker is final summary impact refresh.

    The semantic-correction quality gate checks the final `smart-summary.md`. During
    export, that creates a small cycle: a fresh `smart-summary.codex.md` cannot be
    installed until `smart-summary.md` proves it absorbed the correction, but the
    proof cannot happen until the candidate is installed. This helper breaks that
    cycle while keeping all other quality checks strict.
    """

    checks = quality.get("checks") if isinstance(quality, dict) else []
    if not isinstance(checks, list) or not checks:
        return False
    failing = [str(row.get("key") or "") for row in checks if isinstance(row, dict) and not row.get("passed")]
    if failing != ["transcript_semantic_correction_impact"]:
        return False
    gate = quality.get("transcript_semantic_correction_gate") if isinstance(quality, dict) else {}
    if not isinstance(gate, dict):
        return False
    if str(gate.get("status") or "") != "summary_impact_needs_fix":
        return False
    if int(gate.get("final_residual_error_total") or 0) != 0:
        return False
    if str(gate.get("readable_impact_status") or "") != "passed":
        return False
    return True
def _render_smart_summary_codex_prompt(
    title: str,
    bundle_dir: Path,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    summary: dict[str, Any],
    sidecar: dict[str, Any],
    smart_summary_input_pack: dict[str, Any] | None = None,
) -> str:
    cues, transcript_source, transcript_note = _smart_summary_cues(bundle_dir, manifest, sidecar, timeline)
    segments = _smart_summary_segments(cues, target_segments=12)
    duration = _smart_summary_duration(cues, timeline, manifest)
    visual_status = _smart_visual_status(summary)
    lines = [
        f"# Codex Smart Summary Prompt: {title}",
        "",
        "请你作为 Codex，在本地优先读取同 bundle 的 `exports/long-video-memory-pack.md` 和 `exports/smart-summary-input-pack.md`，并参考 `exports/full-transcript.md`、`timeline.json`、`manifest.json`，生成或改写 `exports/smart-summary.md`。不要上传音频、视频、字幕或私有路径；如需在线/云端 LLM，必须复用同一证据包并遵守当前 provider/preflight 边界。",
        "",
        "## 输出目标",
        "",
        "生成一份对标得到大脑 `data.note.content` 的智能总结。它不是证据审计，不替代 `knowledge-note.md`；它应该是用户可以直接阅读、复制到笔记系统、用于复习和复用的成品总结。",
        "",
        "## 必须包含的结构",
        "",
        "1. 标题",
        "2. 基本信息：视频名、时长、处理时间、来源路径",
        "3. 一句话概览",
        "4. 核心主题/课程主线",
        "5. 分段总结，带时间范围",
        "6. 关键观点/方法论",
        "7. 可执行动作清单",
        "8. 高频话术/可复用表达",
        "9. 待复核点/低置信内容，放在末尾",
        "",
        "## 质量边界",
        "",
        "- 必须覆盖完整视频时长内所有未被 `content-scope` 排除的正文，不得只覆盖抽帧 timeline；排除区间不得进入正文或总结。",
        "- `full-transcript.md` 仅用于追溯原始证据；若它与 `smart-summary-input-pack` 的内容范围冲突，以后者的投影范围为准。",
        "- `projection_status=review_gap` 的转写不能作为事实，不得猜补听不清、串音或疑似静音幻觉。",
        "- 优先使用完整 ASR transcript。",
        "- 可以引用 OCR、frame、visual evidence，但未执行多模态时只能标注为待复核。",
        "- 不要机械列出每个 ASR 片段。",
        "- 时间戳用于导航，不要让格式压过内容。",
        "- 不要把低置信视觉内容写成确定事实。",
        "- 忠实还原音视频中说话人的原意；本任务不负责判断其陈述在外部世界是否真实。",
        "- 对主观判断、产品评价和经验结论使用明确归因，不因未做外部核验就删除原意。",
        "- 只有音频不清、来源冲突或模型新增内容进入待复核；不得补造原始材料没有的信息。",
        "",
        "## 当前材料概况",
        "",
        f"- Bundle: `{bundle_dir}`",
        f"- Title: `{title}`",
        f"- Duration: `{_format_time(duration)}`",
        f"- Transcript source: `{transcript_source}`",
        f"- Transcript cues: `{len(cues)}`",
        f"- Transcript note: `{transcript_note}`",
        f"- Content scope: `{(smart_summary_input_pack or {}).get('content_scope', {}).get('status', 'not_configured') if isinstance((smart_summary_input_pack or {}).get('content_scope', {}), dict) else 'not_configured'}`",
        f"- Content-scope excluded cues: `{(smart_summary_input_pack or {}).get('content_scope', {}).get('excluded_transcript_cue_count', 0) if isinstance((smart_summary_input_pack or {}).get('content_scope', {}), dict) else 0}`",
        f"- Transcript projection review gaps: `{len((smart_summary_input_pack or {}).get('transcript_projection_review_gaps') or [])}`",
        f"- Visual evidence status: {visual_status}",
        f"- Long video memory pack: `{bundle_dir / 'exports' / 'long-video-memory-pack.md'}`",
        f"- Smart summary input pack: `{bundle_dir / 'exports' / 'smart-summary-input-pack.md'}`",
        f"- Timeline items: `{summary.get('timeline_items', len(timeline))}`",
        f"- Missing visual understanding: `{len(summary.get('visual_understanding_missing') or [])}`",
        f"- Missing document structure: `{len(summary.get('document_visual_missing_structure') or [])}`",
        "",
        "## Terminology / Tool Name Arbitration",
        "",
    ]
    lines.extend(_smart_summary_term_arbitration_prompt_lines(smart_summary_input_pack))
    lines.extend(
        [
            "",
            "## ASR 分段导航（请用作章节覆盖检查，不要照抄成流水账）",
            "",
        ]
    )
    for segment in segments:
        lines.extend(
            [
                f"### {_format_time(segment['start'])} - {_format_time(segment['end'])}",
                "",
                _truncate(segment["text"], 900),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _smart_summary_term_arbitration_prompt_lines(input_pack: dict[str, Any] | None) -> list[str]:
    if not isinstance(input_pack, dict):
        return [
            "- Status: `missing_input_pack`",
            "- Guidance: 未读取到 `smart-summary-input-pack`；不要凭空纠正工具名，把疑似错词放入待复核点。",
        ]
    arbitration = input_pack.get("term_arbitration_codex") if isinstance(input_pack.get("term_arbitration_codex"), dict) else {}
    gate = input_pack.get("term_correction_impact_gate") if isinstance(input_pack.get("term_correction_impact_gate"), dict) else {}
    transcript_arbitration = input_pack.get("transcript_arbitration") if isinstance(input_pack.get("transcript_arbitration"), dict) else {}
    quality_notes = [str(note) for note in (input_pack.get("quality_notes") or []) if str(note)]
    lines = [
        "Codex 在生成最终智能总结时必须把术语/工具名当成语义判断问题处理：结合 ASR、自带字幕、OCR/ebook、画面理解、打标器标签和上下文，而不是只做字符串相似替换。",
        "",
        f"- Codex term arbitration status: `{arbitration.get('status', 'missing')}`",
        f"- Candidate count: `{arbitration.get('candidate_count', 0)}`",
        f"- Imported decisions: `{arbitration.get('imported_decision_count', 0)}`",
        f"- Accepted decisions: `{arbitration.get('accepted_decision_count', 0)}`",
        f"- Glossary terms: `{arbitration.get('glossary_term_count', 0)}`",
        f"- Codex review required: `{arbitration.get('codex_review_required', False)}`",
        f"- Ready for transcript arbitration: `{arbitration.get('ready_for_transcript_arbitration', False)}`",
        f"- Transcript arbitration status: `{transcript_arbitration.get('status', 'missing')}`",
        f"- Term correction impact gate: `{gate.get('status', 'not_required')}`",
        f"- Impact gate passed: `{gate.get('passed', False)}`",
        f"- Final export alias total: `{gate.get('final_export_alias_total', 0)}`",
        "",
        "使用规则：",
        "- `codex_review_required=True` 时，不能把候选纠错写成确定事实；只能写入“待复核点/低置信内容”。",
        "- `Ready for transcript arbitration=True` 但字幕仲裁或影响报告未通过时，优先提醒先跑 transcript-source-arbitration 和 term-correction-impact-report。",
        "- `Final export alias total > 0` 时，最终总结必须保留术语风险提示，不要假装错词已清除。",
        "- 只有已导入且通过影响门禁的高置信术语，才可以在标题、章节、关键观点和动作清单中按 canonical term 使用。",
    ]
    if quality_notes:
        lines.extend(["", "Input pack quality notes:"])
        lines.extend([f"- {note}" for note in quality_notes[:8]])
    next_actions = [str(action) for action in (arbitration.get("next_actions") or []) if str(action)]
    gate_actions = [str(action) for action in (gate.get("next_actions") or []) if str(action)]
    if next_actions or gate_actions:
        lines.extend(["", "Suggested next actions before final wording:"])
        lines.extend([f"- {action}" for action in (next_actions + gate_actions)[:8]])
    return lines

def _ensure_smart_summary_codex_summary(root: Path, prompt_path: Path, prompt_markdown: str, *, write: bool) -> dict[str, Any]:
    existing = _existing_codex_smart_summary(root)
    if existing:
        return {"status": "existing", "smart_summary_codex_path": str(existing), "generated": False, "write": write}
    if not write:
        return {"status": "skipped_write_false", "smart_summary_codex_path": str(root / "exports" / "smart-summary.codex.md"), "generated": False, "write": write}
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_markdown, encoding="utf-8")
    result = generate_smart_summary_with_codex(root, write=True)
    result["generated"] = True
    return result

def _quality_summary_path(existing: Path | None, rendered_export: Path) -> Path:
    """Validate the canonical candidate, never the compatibility copy written for readers."""
    if existing and existing.is_file() and existing.stat().st_size > 0:
        return existing
    return rendered_export

def _existing_codex_smart_summary(bundle_dir: Path) -> Path | None:
    for path in (bundle_dir / "smart-summary.codex.md", bundle_dir / "codex-smart-summary.md", bundle_dir / "exports" / "smart-summary.codex.md", bundle_dir / "exports" / "smart-summary.llm.md"):
        if not (path.exists() and path.is_file() and path.stat().st_size > 0):
            continue
        text = path.read_text(encoding="utf-8-sig")
        if _looks_like_final_llm_smart_summary(text):
            return path
    return None


def _looks_like_final_llm_smart_summary(text: str) -> bool:
    from .smart_summary_handoff import WATERMARK
    if WATERMARK in text:
        return False
    return any(marker in text for marker in ("codex_final", "codex_llm_rewrite_final", "online_llm_section_rewrite", "生成方式：`codex_final`", "生成方式：`codex_llm_rewrite_final`", "生成方式：`online_llm_section_rewrite`", "section_staged_apply", "codex_first_llm_substitute"))


def _smart_summary_cues(
    bundle_dir: Path,
    manifest: dict[str, Any],
    sidecar: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> tuple[list[Any], str, str]:
    source_path = _full_transcript_source_path(bundle_dir, manifest, sidecar)
    if source_path:
        try:
            cues = parse_transcript(source_path)
        except Exception as exc:
            fallback = _timeline_cues(timeline)
            scoped, note = _content_scoped_summary_cues(bundle_dir, fallback, transcript_path=None)
            return scoped, "timeline_fallback", f"ASR parse failed: {exc}; {note}"
        if cues:
            scoped, note = _content_scoped_summary_cues(bundle_dir, cues, transcript_path=source_path)
            return scoped, _full_transcript_source_label(sidecar, source_path), f"完整 ASR sidecar: {source_path}; {note}"
        fallback = _timeline_cues(timeline)
        scoped, note = _content_scoped_summary_cues(bundle_dir, fallback, transcript_path=None)
        return scoped, "timeline_fallback", f"ASR sidecar empty: {source_path}; {note}"
    fallback = _timeline_cues(timeline)
    scoped, note = _content_scoped_summary_cues(bundle_dir, fallback, transcript_path=None)
    return scoped, "timeline_fallback", f"ASR sidecar not found; used timeline fallback; {note}"


def _content_scoped_summary_cues(
    bundle_dir: Path,
    cues: list[Any],
    *,
    transcript_path: Path | None,
) -> tuple[list[Any], str]:
    scope = load_content_scope(bundle_dir, transcript_path=transcript_path)
    scoped, report = project_transcript_cues(cues, scope, projection="smart_summary")
    if not scope.get("active"):
        return scoped, "content scope not configured"
    return (
        scoped,
        f"content projection omitted {report.get('projection_omitted_count', 0)} cues "
        f"({report.get('excluded_count', 0)} operator-confirmed scope cues); "
        "excluded evidence remains in the full transcript",
    )


def _timeline_cues(timeline: list[dict[str, Any]]) -> list[Any]:
    rows = []
    for item in timeline:
        text = _item_transcript(item)
        if not text:
            continue
        rows.append(type("SmartCue", (), {"start": _float_seconds(item.get("start")), "end": _float_seconds(item.get("end")), "text": text})())
    return rows


def _smart_summary_segments(cues: list[Any], *, target_segments: int = 10) -> list[dict[str, Any]]:
    cleaned = [cue for cue in cues if _text(getattr(cue, "text", ""))]
    if not cleaned:
        return []
    start = min(_float_seconds(getattr(cue, "start", 0.0)) for cue in cleaned)
    end = max(_float_seconds(getattr(cue, "end", 0.0)) for cue in cleaned)
    duration = max(1.0, end - start)
    bucket = max(300.0, duration / max(1, target_segments))
    segments: list[dict[str, Any]] = []
    current: list[Any] = []
    current_start = _float_seconds(getattr(cleaned[0], "start", 0.0))
    for cue in cleaned:
        cue_start = _float_seconds(getattr(cue, "start", 0.0))
        if current and cue_start - current_start >= bucket:
            segments.append(_smart_segment_from_cues(current))
            current = []
            current_start = cue_start
        current.append(cue)
    if current:
        segments.append(_smart_segment_from_cues(current))
    return segments


def _smart_segment_from_cues(cues: list[Any]) -> dict[str, Any]:
    text = _compact_text(" ".join(_text(getattr(cue, "text", "")) for cue in cues))
    return {
        "start": _float_seconds(getattr(cues[0], "start", 0.0)),
        "end": max(_float_seconds(getattr(cue, "end", 0.0)) for cue in cues),
        "text": text,
        "keywords": _keyword_candidates(text, limit=8),
        "representative": _representative_sentences(text, limit=2),
    }


def _smart_summary_duration(cues: list[Any], timeline: list[dict[str, Any]], manifest: dict[str, Any]) -> float:
    values = [_float_seconds(getattr(cue, "end", 0.0)) for cue in cues]
    values.extend(_float_seconds(item.get("end")) for item in timeline)
    for key in ("duration_seconds", "duration"):
        if manifest.get(key):
            values.append(_float_seconds(manifest.get(key)))
    return max(values or [0.0])


def _smart_source_path(manifest: dict[str, Any]) -> str:
    for key in ("media_path", "source_video_path", "video_path", "local_media_path"):
        value = _text(manifest.get(key))
        if value:
            return value
    sources = manifest.get("sources") if isinstance(manifest.get("sources"), list) else []
    for source in sources:
        if isinstance(source, dict):
            value = _text(source.get("path") or source.get("local_media_path") or source.get("url"))
            if value:
                return value
    return ""


def _reader_content_type(manifest: dict[str, Any]) -> str:
    """Resolve reader metadata by reusing manifest identity and stdlib MIME data.

    Intent: distinguish recordings from videos without another probe/runtime.
    Decision: explicit manifest content type wins; otherwise reuse the existing
    source-path resolver and Python ``mimetypes``.
    Reason: final-note rendering must not duplicate FFprobe or media ingestion.
    Evidence: ``_smart_source_path`` is already the exporter's canonical source
    locator.
    Effective scope: reader-facing label only.
    """

    for key in ("content_type", "recording_type", "expected_content_type"):
        value = _text(manifest.get(key))
        if value:
            return value
    mime_type, _encoding = mimetypes.guess_type(_smart_source_path(manifest))
    if str(mime_type or "").startswith("audio/"):
        return "录音整理"
    if str(mime_type or "").startswith("video/"):
        return "视频整理"
    return "音视频整理"


def _reader_participant_count(
    manifest: dict[str, Any],
    transcript_quality_gate: dict[str, Any],
    speaker_review: dict[str, Any] | None = None,
) -> int:
    """Prefer a human-confirmed participant count, then machine evidence.

    Intent: avoid presenting chunk-local diarization clusters as real people.
    Decision: accept ``speaker-review.json`` only when its status explicitly
    confirms the count; otherwise reuse the diarization gate and declarations.
    Reason: long recordings may restart clustering for every ASR chunk.
    Evidence: five chunk-local labels were observed while the user confirmed
    three primary speakers.
    Effective scope: final-note metadata only; roles/names remain unmodified.
    """

    review = speaker_review if isinstance(speaker_review, dict) else {}
    if str(review.get("status") or "") == "human_confirmed_count":
        try:
            confirmed = int(review.get("confirmed_participant_count") or 0)
        except (TypeError, ValueError):
            confirmed = 0
        if confirmed > 0:
            return confirmed

    speaker_gate = (
        transcript_quality_gate.get("speaker_diarization")
        if isinstance(transcript_quality_gate.get("speaker_diarization"), dict)
        else {}
    )
    requirements = (
        manifest.get("transcript_requirements")
        if isinstance(manifest.get("transcript_requirements"), dict)
        else {}
    )
    candidates = [
        speaker_gate.get("distinct_speaker_count"),
        manifest.get("participant_count"),
        manifest.get("expected_speaker_count"),
        requirements.get("expected_speaker_count"),
    ]
    for value in candidates:
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return 0


def _speaker_review(root: Path) -> dict[str, Any]:
    path = root / "speaker-review.json"
    if not path.exists():
        return {}
    try:
        review = read_json(path)
    except Exception:
        return {}
    if not isinstance(review, dict):
        return {}
    if review.get("schema") != "video_knowledge_pipeline.speaker_review.v1":
        return {}
    return review


def _speaker_review_notice(review: dict[str, Any]) -> str:
    if str(review.get("status") or "") != "human_confirmed_count":
        return ""
    try:
        confirmed = int(review.get("confirmed_participant_count") or 0)
        observed = int(review.get("observed_cluster_count") or 0)
    except (TypeError, ValueError):
        return ""
    if confirmed <= 0 or observed <= confirmed:
        return ""
    return (
        f"说话人说明：人工确认实际有 {confirmed} 位主要说话人；当前逐字稿中的 "
        f"{observed} 个‘说话人’编号是分块声纹聚类标签，跨块身份尚未统一，"
        "不代表真实参与人数。"
    )


def _smart_visual_status(summary: dict[str, Any]) -> str:
    visual = int(summary.get("items_with_visual_understanding") or 0)
    temporal = int(summary.get("items_with_temporal_understanding") or 0)
    structured = int(summary.get("items_with_structured_visual") or 0)
    if visual or temporal or structured:
        return f"已有部分视觉证据：单帧 {visual}，连续片段 {temporal}，图文结构 {structured}；未覆盖部分仍需复核。"
    return "视觉证据未执行或未可靠提取；本总结主要依据 ASR，画面细节需复核。"


def _smart_one_sentence_overview(title: str, cues: list[Any], segments: list[dict[str, Any]]) -> str:
    if segments:
        keywords = _keyword_candidates(" ".join(segment["text"] for segment in segments[:3]), limit=5)
        if keywords:
            return f"这段视频围绕“{title}”展开，主线集中在{ '、'.join(keywords[:4]) }等内容，并通过连续讲解给出理解框架和执行提示。"
    first = _text(getattr(cues[0], "text", "")) if cues else ""
    return _truncate(first or f"这是一段关于 {title} 的知识讲解视频。", 160)


def _smart_mainline_lines(cues: list[Any], segments: list[dict[str, Any]]) -> list[str]:
    if not segments:
        return ["（暂无可用转写，无法生成课程主线。）"]
    lines = []
    for idx, segment in enumerate(segments[:8], start=1):
        keywords = "、".join(segment.get("keywords") or []) or "本段主题"
        lines.append(f"{idx}. `{_format_time(segment['start'])} - {_format_time(segment['end'])}`：{keywords}。")
    return lines


def _smart_segment_lines(segments: list[dict[str, Any]]) -> list[str]:
    if not segments:
        return ["（暂无可用转写。）"]
    lines = []
    for segment in segments:
        lines.extend([f"### {_format_time(segment['start'])} - {_format_time(segment['end'])}", ""])
        reps = segment.get("representative") or []
        if reps:
            for sentence in reps:
                lines.append(f"- {sentence}")
        else:
            lines.append(f"- {_truncate(segment.get('text', ''), 220)}")
        if segment.get("keywords"):
            lines.append(f"- 关键词：{ '、'.join(segment['keywords']) }")
        lines.append("")
    return lines


def _smart_key_point_lines(cues: list[Any]) -> list[str]:
    terms = ["关键", "核心", "原则", "方法", "逻辑", "重点", "本质", "问题", "原因", "结果", "总结", "所以", "一定", "不要", "必须"]
    return _selected_sentence_lines(cues, terms, fallback="（未自动提取到明确的方法论句子，建议 Codex 复写时从完整逐字稿中归纳。）", limit=12)


def _smart_action_lines(cues: list[Any]) -> list[str]:
    terms = ["要", "先", "再", "然后", "步骤", "动作", "执行", "练习", "准备", "记录", "复盘", "沟通", "提问", "确认", "解决"]
    return _selected_sentence_lines(cues, terms, fallback="（未自动提取到明确行动项，建议人工/Codex 根据课程主线补充。）", limit=12)


def _smart_reusable_expression_lines(cues: list[Any]) -> list[str]:
    terms = ["比如", "你可以", "如果", "怎么", "为什么", "其实", "不是", "而是", "这就是", "换句话说"]
    return _selected_sentence_lines(cues, terms, fallback="（未自动提取到高频话术，建议 Codex 根据逐字稿二次提炼。）", limit=10, quote=True)


def _smart_review_lines(summary: dict[str, Any], transcript_note: str, visual_status: str) -> list[str]:
    lines = [f"- 转写依据：{transcript_note}", f"- 视觉状态：{visual_status}"]
    visual_missing = summary.get("visual_understanding_missing") if isinstance(summary.get("visual_understanding_missing"), list) else []
    structure_missing = summary.get("document_visual_missing_structure") if isinstance(summary.get("document_visual_missing_structure"), list) else []
    if visual_missing:
        lines.append(f"- 仍有 `{len(visual_missing)}` 个片段缺少可靠视觉理解，涉及 timeline index: `{', '.join(str(v) for v in visual_missing[:30])}`。")
    if structure_missing:
        lines.append(f"- 仍有 `{len(structure_missing)}` 个图文/表格/代码/公式片段缺少结构化解析，涉及 timeline index: `{', '.join(str(v) for v in structure_missing[:30])}`。")
    if len(lines) == 2:
        lines.append("- 暂无额外自动复核点；仍建议人工抽样核对关键术语和数字。")
    return lines


def _selected_sentence_lines(cues: list[Any], terms: list[str], *, fallback: str, limit: int, quote: bool = False) -> list[str]:
    sentences = []
    for cue in cues:
        start = _float_seconds(getattr(cue, "start", 0.0))
        for sentence in _split_sentences(_text(getattr(cue, "text", ""))):
            if len(sentence) < 8:
                continue
            if any(term in sentence for term in terms):
                line = f"`{_format_time(start)}` {sentence}"
                sentences.append(f"- “{line}”" if quote else f"- {line}")
            if len(sentences) >= limit:
                return sentences
    return sentences or [fallback]


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？!?])\s*|[\n\r]+", _compact_text(text)) if part.strip()]


def _representative_sentences(text: str, *, limit: int) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    scored = []
    for sentence in sentences:
        score = min(len(sentence), 120)
        if any(term in sentence for term in ("关键", "核心", "所以", "但是", "方法", "步骤", "问题", "原则", "一定", "不要")):
            score += 80
        if re.search(r"\d", sentence):
            score += 20
        scored.append((score, sentence))
    scored.sort(key=lambda item: (-item[0], sentences.index(item[1])))
    picked = []
    for _, sentence in scored:
        if sentence not in picked:
            picked.append(sentence)
        if len(picked) >= limit:
            break
    return picked


def _keyword_candidates(text: str, *, limit: int) -> list[str]:
    stop = {"这个", "那个", "就是", "然后", "因为", "所以", "如果", "我们", "你们", "他们", "一个", "什么", "还是", "不是", "没有", "可以", "进行", "需要", "时候", "现在", "这里", "大家", "自己"}
    counts: dict[str, int] = {}
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{2,}|[\u4e00-\u9fff]{2,6}", text):
        if token in stop or len(token) < 2:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [token for token, _ in ranked[:limit]]


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", _text(text)).strip()

def _render_extraction_audit(
    bundle_dir: Path,
    title: str,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    summary: dict[str, Any],
    term_correction: dict[str, Any] | None = None,
) -> str:
    review_import = manifest.get("review_notes_last_import") if isinstance(manifest.get("review_notes_last_import"), dict) else {}
    acceptance = _read_optional_mapping(bundle_dir / "acceptance-check.json")
    acceptance_action = acceptance.get("next_action") if isinstance(acceptance.get("next_action"), dict) else {}
    review_lifecycle = manifest.get("review_lifecycle") if isinstance(manifest.get("review_lifecycle"), dict) else {}
    visual_missing = summary.get("visual_understanding_missing") if isinstance(summary.get("visual_understanding_missing"), list) else []
    structure_missing = summary.get("document_visual_missing_structure") if isinstance(summary.get("document_visual_missing_structure"), list) else []
    lines = [
        "---",
        "type: video-knowledge-extraction-audit",
        f'title: "{_frontmatter_escape(title)}"',
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# {title} - 提取审计",
        "",
        "这份文件用于判断视频知识提取是否还有漏项。它记录覆盖率、缺口、人工审核状态和证据路径，不替代最终知识笔记。",
        "",
        "## 1. 总览",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| Bundle | `{bundle_dir}` |",
        f"| 最终验收状态 | `{acceptance.get('status') or 'unknown'}` |",
        f"| 下一步动作 | `{acceptance_action.get('key') or 'unknown'}` |",
        f"| 人工复核状态 | `{review_lifecycle.get('state') or manifest.get('review_status') or 'unknown'}` |",
        f"| 时间线片段 | `{summary.get('timeline_items', 0)}` |",
        f"| 覆盖审计状态 | `{summary.get('coverage_status') or 'unknown'}` |",
        f"| 有转写 | `{summary.get('items_with_transcript', 0)}` |",
        f"| 有画面文字/OCR 或保留图片 | `{summary.get('items_with_visual_text', 0)}` |",
        f"| 有图文结构化或人工兜底 | `{summary.get('items_with_structured_visual', 0)}` |",
        f"| 有单帧视觉理解或人工验收 | `{summary.get('items_with_visual_understanding', 0)}` |",
        f"| 有连续片段理解或人工验收 | `{summary.get('items_with_temporal_understanding', 0)}` |",
        f"| 图文结构缺口 | `{len(structure_missing)}` |",
        f"| 视觉理解缺口 | `{len(visual_missing)}` |",
        f"| 最近导入审核条数 | `{review_import.get('updated', 0)}` |",
        "",
        "## 2. 术语与工具名纠错闭环",
        "",
        *_term_correction_audit_lines(term_correction or {}),
        "",
        "## 3. 画面类型分布",
        "",
        *_route_table_lines(summary),
        "",
        "## 4. 缺口索引",
        "",
        *_render_gap_list("图文截图/表格/代码/公式待解析", structure_missing),
        "",
        *_render_gap_list("视频视觉理解待补齐", visual_missing),
        "",
        "## 5. 人工审核状态",
        "",
    ]
    lines.extend(_review_audit_lines(timeline, review_import))
    lines.extend(["", "## 6. OCR 裁剪证据", ""])
    lines.extend(_ocr_crop_audit_lines(timeline))
    lines.extend(["", "## 7. 逐片段审计表", ""])
    lines.extend(_timeline_audit_table_lines(bundle_dir, timeline))
    lines.extend(["", "## 8. 产物路径", ""])
    lines.extend(
        [
            f"- Knowledge note: `{bundle_dir / 'exports' / 'knowledge-note.md'}`",
            f"- Full transcript: `{bundle_dir / 'exports' / 'full-transcript.md'}`",
            f"- Coverage report: `{bundle_dir / 'knowledge-coverage.md'}`",
            f"- Acceptance report: `{bundle_dir / 'acceptance-check.md'}`",
            f"- Review session: `{bundle_dir / 'review-session.md'}`",
            f"- Review template: `{bundle_dir / 'review-notes.template.json'}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _term_correction_audit_lines(status: dict[str, Any]) -> list[str]:
    if not status:
        return ["- 暂无术语纠错闭环状态。"]
    artifacts = status.get("artifacts") if isinstance(status.get("artifacts"), dict) else {}
    lines = [
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 闭环状态 | `{status.get('status') or 'unknown'}` |",
        f"| Codex语义预检 | `{status.get('term_validation_status') or 'missing'}` |",
        f"| 预检接受/拒绝 | `{int(status.get('accepted_validation_decisions') or 0)}/{int(status.get('rejected_validation_decisions') or 0)}` |",
        f"| 已接受术语/工具名 | `{int(status.get('accepted_term_count') or 0)}` |",
        f"| 纠正版转写参与导出 | `{_yes_no(bool(status.get('source_arbitrated_transcript_exists')))}` |",
        f"| 最终导出错词残留 | `{int(status.get('final_export_alias_total') or 0)}` |",
        f"| 智能总结质量门禁 | `{_yes_no(bool(status.get('smart_summary_quality_passed')))}` |",
        f"| 影响检查状态 | `{status.get('impact_status') or 'missing'}` |",
    ]
    next_action = str(status.get("next_action_key") or "").strip()
    if next_action:
        lines.append(f"| 推荐下一步 | `{next_action}` |")
    artifact_rows = []
    for label, key in (
        ("Codex语义预检", "term_validation_markdown"),
        ("术语仲裁词典", "glossary_json"),
        ("纠正版转写", "source_arbitrated_transcript_json"),
        ("影响报告", "impact_report_markdown"),
        ("闭环报告", "closure_markdown"),
    ):
        raw = str(artifacts.get(key) or "").strip()
        if raw:
            artifact_rows.append(f"- {label}: `{raw}`")
    accepted_terms = _accepted_term_labels_for_audit(status)
    if accepted_terms:
        lines.extend(["", "- 已接受术语/工具名：" + "、".join(accepted_terms[:30])])
    if artifact_rows:
        lines.extend(["", *artifact_rows])
    return lines


def _accepted_term_labels_for_audit(status: dict[str, Any]) -> list[str]:
    rows = status.get("accepted_terms") if isinstance(status.get("accepted_terms"), list) else []
    labels: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        canonical = str(row.get("canonical_term") or "").strip()
        if canonical and canonical not in labels:
            labels.append(canonical)
    return labels

def _review_audit_lines(timeline: list[dict[str, Any]], review_import: dict[str, Any]) -> list[str]:
    reviewed = [item for item in timeline if _human_review_accepted(item)]
    keep_image = [item for item in timeline if _human_keep_image(item)]
    corrected_visual = [item for item in timeline if _mapping(item.get("human_corrected_visual_understanding")) or _mapping(_human_review(item).get("corrected_visual_understanding"))]
    corrected_temporal = [
        item
        for item in timeline
        if _mapping(item.get("human_corrected_temporal_visual_understanding")) or _mapping(_human_review(item).get("corrected_temporal_visual_understanding"))
    ]
    lines = [
        "| 项目 | 数量/状态 |",
        "| --- | ---: |",
        f"| 已接受人工审核片段 | {len(reviewed)} |",
        f"| 人工保留图片片段 | {len(keep_image)} |",
        f"| 人工修正单帧视觉理解 | {len(corrected_visual)} |",
        f"| 人工修正连续片段理解 | {len(corrected_temporal)} |",
        f"| 最近导入 updated | {review_import.get('updated', 0)} |",
    ]
    if review_import:
        lines.extend(
            [
                "",
                f"- Review JSON: `{review_import.get('review_json', '')}`",
                f"- Updated indexes: `{', '.join(str(value) for value in review_import.get('updated_indexes') or [])}`",
            ]
        )
    return lines


def _timeline_audit_table_lines(bundle_dir: Path, timeline: list[dict[str, Any]]) -> list[str]:
    if not timeline:
        return ["（无时间线。）"]
    lines = ["| Index | 时间段 | 路由 | 转写 | 图文 | 单帧理解 | 连续理解 | 人审 | 缺口/风险 | 证据 |", "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for fallback_index, item in enumerate(timeline, start=1):
        index = int(item.get("index") or fallback_index)
        issues = ", ".join(str(value) for value in item.get("quality_issues") or [])
        evidence = "; ".join(_evidence_paths(bundle_dir, item, visual=_item_visual_understanding(item), temporal=_item_temporal_understanding(item))[:4])
        lines.append(
            "| {index} | `{time}` | `{route}` | {speech} | {visual_text} | {visual} | {temporal} | {review} | {issues} | {evidence} |".format(
                index=index,
                time=_time_range(item),
                route=_table_cell(str(item.get("visual_route") or "unknown")),
                speech=_yes_no(bool(_text(item.get("transcript") or item.get("original_transcript")))),
                visual_text=_yes_no(bool(_item_visual_text(item) or _structured_entries(item) or _human_keep_image(item))),
                visual=_yes_no(bool(_item_visual_understanding(item))),
                temporal=_yes_no(bool(_item_temporal_understanding(item))),
                review=_yes_no(_human_review_accepted(item)),
                issues=_table_cell(issues or "无"),
                evidence=_table_cell(evidence or "无"),
            )
        )
    return lines


def _yes_no(value: bool) -> str:
    return "是" if value else "否"


def _full_transcript_lines(timeline: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    previous = ""
    for index, item in enumerate(timeline, start=1):
        text = _item_transcript(item)
        if not text:
            continue
        if text == previous:
            continue
        previous = text
        visual_note = _transcript_visual_note(item)
        lines.extend([f"### {_time_range(item)}", "", "#### 说了什么", "", text, ""])
        term_lines = _term_resolution_lines(item)
        if term_lines:
            lines.extend(["#### 术语仲裁", "", *term_lines, ""])
        fallback = "（画面未可靠提取；请查看审计表和证据帧。）" if _expects_visual_note(item) else "（这一段暂无画面/演示记录。）"
        lines.extend(["#### 演示了什么", "", visual_note or fallback, ""])
    return lines or ["（无转写。）"]


def _ocr_crop_audit_lines(timeline: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for fallback_index, item in enumerate(timeline, start=1):
        recovery = item.get("screen_text_recovery") if isinstance(item.get("screen_text_recovery"), dict) else {}
        paths = [str(path) for path in recovery.get("crop_paths") or [] if str(path)]
        if not paths:
            continue
        rows.append(
            "| {index} | `{time}` | {count} | {paths} |".format(
                index=int(item.get("index") or fallback_index),
                time=_time_range(item),
                count=len(paths),
                paths=_table_cell("; ".join(paths[:12])),
            )
        )
    if not rows:
        return ["（暂无 OCR 裁剪证据。）"]
    return ["| Index | 时间段 | 裁剪数 | 裁剪证据路径 |", "| ---: | --- | ---: | --- |", *rows]


def _transcript_visual_note(item: dict[str, Any]) -> str:
    parts = []
    visual_text = _item_visual_text(item)
    if visual_text:
        parts.append("画面文字/OCR：" + _truncate(visual_text, 160))
    tagger_tags = _item_tagger_tags(item)
    if tagger_tags:
        parts.append("RAM++ 候选标签（未人工确认）：" + "、".join(tagger_tags))
    visual = _item_visual_understanding(item)
    if visual:
        compact = _compact_mapping(visual, keys=["actions", "interface_state", "instructor_focus", "non_text_information", "keep_image_reason"])
        if compact:
            parts.append("单帧视觉：" + compact)
    temporal = _item_temporal_understanding(item)
    if temporal:
        compact = _compact_mapping(temporal, keys=["event_sequence", "state_changes", "operation_steps", "causal_links"])
        if compact:
            parts.append("连续片段：" + compact)
    review = _human_review(item)
    if review:
        status = _text(item.get("review_status") or review.get("status"))
        comment = _text(review.get("comment") or review.get("notes"))
        parts.append("人工审核：" + "；".join(part for part in [status, comment] if part))
    return "\n".join(f"- {part}" for part in parts if part)


def _expects_visual_note(item: dict[str, Any]) -> bool:
    route = str(item.get("visual_route") or "")
    if route in {"semantic_frame", "temporal_sequence", "mixed", "document_visual"}:
        return True
    issues = {str(value) for value in item.get("quality_issues") or []}
    return bool(
        issues
        & {
            "missing_visual_text",
            "missing_visual_understanding",
            "semantic_frame_without_analysis",
            "temporal_sequence_without_analysis",
            "structured_visual_without_structure",
            "ocr_text_empty",
            "screen_text_low_confidence",
        }
    )


def _chapter_lines(bundle_dir: Path, timeline: list[dict[str, Any]]) -> list[str]:
    chapters = _build_chapters(timeline)
    lines: list[str] = []
    for number, chapter in enumerate(chapters, start=1):
        items = chapter["items"]
        first = items[0]
        last = items[-1]
        lines.extend([f"### {number}. {_time_range({'start': first.get('start'), 'end': last.get('end')})}", ""])
        lines.extend(["#### 说了什么", ""])
        lines.extend(_chapter_transcript_lines(items))
        lines.extend(["", "#### 演示了什么", ""])
        lines.extend(_chapter_demo_lines(items))
        lines.extend(["", "#### 屏幕/图表信息", ""])
        lines.extend(_chapter_screen_lines(items))
        evidence = _dedupe(path for item in items for path in _evidence_paths(bundle_dir, item, visual=_item_visual_understanding(item), temporal=_item_temporal_understanding(item)))
        lines.extend(["", "#### 证据", ""])
        lines.extend([f"- `{path}`" for path in evidence[:8]] or ["- 暂无证据帧"])
        gaps = _dedupe(issue for item in items for issue in (str(value) for value in item.get("quality_issues") or []))
        lines.extend(["", "#### 缺口", ""])
        lines.extend([f"- `{gap}`" for gap in gaps[:12]] or ["- 无"])
        lines.append("")
    return lines or ["（暂无可组织的时间线片段。）"]


def _build_chapters(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    previous_route = ""
    previous_end = 0.0
    for item in timeline:
        route = str(item.get("visual_route") or "unknown")
        start = _float_seconds(item.get("start"))
        gap = start - previous_end if current else 0.0
        should_split = bool(current) and (route != previous_route or gap > 30 or len(current) >= 8)
        if should_split:
            chapters.append({"items": current})
            current = []
        current.append(item)
        previous_route = route
        previous_end = _float_seconds(item.get("end"))
    if current:
        chapters.append({"items": current})
    return chapters


def _chapter_transcript_lines(items: list[dict[str, Any]]) -> list[str]:
    texts = []
    previous = ""
    for item in items:
        text = _truncate(_item_transcript(item), 220)
        if text and text != previous:
            texts.append(f"- `{int(item.get('index') or 0)}` {text}")
            previous = text
    return texts or ["- 暂无转写。"]


def _chapter_demo_lines(items: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in items:
        index = int(item.get("index") or 0)
        visual = _item_visual_understanding(item)
        temporal = _item_temporal_understanding(item)
        demo = _compact_mapping(visual, keys=["actions", "interface_state", "instructor_focus"]) or _compact_mapping(
            temporal, keys=["event_sequence", "operation_steps", "state_changes"]
        )
        if demo:
            lines.append(f"- `{index}` {demo}")
    return lines or ["- 暂无演示信息；需要视觉理解或人工标注补齐。"]


def _chapter_screen_lines(items: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in items:
        index = int(item.get("index") or 0)
        visual_text = _truncate(_item_visual_text(item), 220)
        structured = _structured_entries(item)
        visual = _item_visual_understanding(item)
        screen = visual_text or _compact_mapping(visual, keys=["objects", "spatial_relations", "non_text_information"])
        if not screen and _human_keep_image(item):
            screen = "人工确认：该帧保留为图片证据，未强行降维为 OCR 文本。"
        if screen:
            lines.append(f"- `{index}` {screen}")
        for entry in structured[:2]:
            markdown = _truncate(_meaningful_visual_text(entry.get("markdown"), item), 260)
            if markdown:
                lines.append(f"- `{index}` 图文结构化：{markdown}")
    return lines or ["- 暂无屏幕文字、图表、公式或代码结构化结果。"]


def _concept_argument_lines(timeline: list[dict[str, Any]]) -> list[str]:
    rows: list[tuple[int, str, str, str]] = []
    for item in timeline:
        index = int(item.get("index") or 0)
        transcript = _truncate(_item_transcript(item), 160)
        visual = _item_visual_understanding(item)
        temporal = _item_temporal_understanding(item)
        visual_hint = _compact_mapping(
            visual,
            keys=["objects", "actions", "interface_state", "spatial_relations", "instructor_focus", "non_text_information"],
        )
        temporal_hint = _compact_mapping(temporal, keys=["event_sequence", "state_changes", "operation_steps", "causal_links"])
        if not transcript and not visual_hint and not temporal_hint:
            continue
        argument = transcript or visual_hint or temporal_hint
        support = visual_hint or temporal_hint or _item_visual_text(item) or "仅有转写，视觉证据待补齐。"
        issues = ", ".join(str(issue) for issue in item.get("quality_issues") or [])
        if issues:
            support = f"{support}；缺口：{issues}"
        rows.append((index, _time_range(item), argument, support))
    if not rows:
        return ["（暂无可抽取的概念或论点；需要转写、视觉理解或人工标注补齐。）"]
    lines = ["| Index | 时间段 | 概念/论点线索 | 证据与风险 |", "| ---: | --- | --- | --- |"]
    for index, time_range, argument, support in rows[:80]:
        lines.append(f"| {index} | `{time_range}` | {_table_cell(argument)} | {_table_cell(support)} |")
    return lines


def _visual_knowledge_lines(bundle_dir: Path, timeline: list[dict[str, Any]], *, include_temporal: bool = True) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(timeline, start=1):
        blocks = []
        visual = _item_visual_understanding(item)
        temporal = _item_temporal_understanding(item)
        structured = _structured_entries(item)
        if visual:
            blocks.extend(_mapping_bullets("单帧视觉理解", visual, keys=["objects", "actions", "interface_state", "spatial_relations", "instructor_focus", "non_text_information", "keep_image_reason", "confidence"]))
        if include_temporal and temporal:
            blocks.extend(_mapping_bullets("连续片段理解", temporal, keys=["event_sequence", "state_changes", "operation_steps", "causal_links", "possible_missing_points", "confidence"]))
        if structured:
            blocks.append("#### 图文结构化")
            for entry in structured:
                markdown = _meaningful_visual_text(entry.get("markdown"), item)
                if markdown:
                    blocks.extend(["", markdown, ""])
        if not blocks:
            continue
        lines.extend([f"### {int(item.get('index') or index)}. {_time_range(item)}", ""])
        lines.extend(blocks)
        evidence = _evidence_paths(bundle_dir, item, visual=visual, temporal=temporal)
        if evidence:
            lines.extend(["#### 证据截图", ""])
            lines.extend(f"- `{path}`" for path in evidence)
        lines.append("")
    return lines


def _retained_media_lines(bundle_dir: Path, timeline: list[dict[str, Any]]) -> list[str]:
    rows: list[tuple[int, str, str, str, str]] = []
    for fallback_index, item in enumerate(timeline, start=1):
        index = int(item.get("index") or fallback_index)
        material_types = [str(value) for value in item.get("material_types") or [] if str(value).strip()]
        structured = _structured_entries(item)
        keep_image = _human_keep_image(item)
        if not material_types and not structured and not keep_image:
            continue
        content_parts: list[str] = []
        for entry in structured[:3]:
            markdown = _truncate(_meaningful_visual_text(entry.get("markdown") or entry.get("text"), item), 180)
            if markdown:
                content_parts.append(markdown)
        if keep_image:
            content_parts.append("人工确认保留图片证据，未强行降维为文字。")
        if not content_parts:
            content_parts.append("需要保留或补齐图文结构化结果。")
        evidence = "; ".join(_evidence_paths(bundle_dir, item, visual=_item_visual_understanding(item), temporal=_item_temporal_understanding(item))[:4])
        rows.append((index, _time_range(item), ", ".join(material_types) or "未标记", "；".join(content_parts), evidence or "暂无证据路径"))
    if not rows:
        return ["（暂无表格、公式、代码或人工保留图片记录。）"]
    lines = ["| Index | 时间段 | 类型 | 内容/保留理由 | 证据 |", "| ---: | --- | --- | --- | --- |"]
    for index, time_range, material, content, evidence in rows:
        lines.append(f"| {index} | `{time_range}` | {_table_cell(material)} | {_table_cell(content)} | {_table_cell(evidence)} |")
    return lines


def _temporal_knowledge_lines(bundle_dir: Path, timeline: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(timeline, start=1):
        temporal = _item_temporal_understanding(item)
        if not temporal:
            continue
        lines.extend([f"### {int(item.get('index') or index)}. {_time_range(item)}", ""])
        lines.extend(_mapping_bullets("连续片段理解", temporal, keys=["event_sequence", "state_changes", "operation_steps", "causal_links", "possible_missing_points", "confidence"]))
        evidence = _evidence_paths(bundle_dir, item, visual={}, temporal=temporal)
        if evidence:
            lines.extend(["#### 证据帧", ""])
            lines.extend(f"- `{path}`" for path in evidence[:12])
        lines.append("")
    return lines


def _evidence_index_lines(bundle_dir: Path, timeline: list[dict[str, Any]]) -> list[str]:
    rows: list[tuple[int, str, str, str]] = []
    for fallback_index, item in enumerate(timeline, start=1):
        index = int(item.get("index") or fallback_index)
        paths = _evidence_paths(bundle_dir, item, visual=_item_visual_understanding(item), temporal=_item_temporal_understanding(item))
        if not paths:
            continue
        rows.append((index, _time_range(item), str(item.get("visual_route") or "unknown"), "; ".join(paths[:8])))
    if not rows:
        return ["（暂无证据帧路径。）"]
    lines = ["| Index | 时间段 | 路由 | 证据路径 |", "| ---: | --- | --- | --- |"]
    for index, time_range, route, evidence in rows:
        lines.append(f"| {index} | `{time_range}` | `{route}` | {_table_cell(evidence)} |")
    return lines


def _timeline_item_lines(bundle_dir: Path, item: dict[str, Any], *, fallback_index: int) -> list[str]:
    index = int(item.get("index") or fallback_index)
    lines = [
        f"### {index}. {_time_range(item)}",
        "",
        f"- 路由：`{item.get('visual_route') or 'unknown'}`",
        f"- 材料类型：`{', '.join(str(value) for value in item.get('material_types') or [])}`",
    ]
    issues = [str(issue) for issue in item.get("quality_issues") or []]
    if issues:
        lines.append(f"- 缺口/风险：`{', '.join(issues)}`")
    transcript = _item_transcript(item)
    if transcript:
        lines.extend(["", "#### 口语/字幕", "", transcript])
    visual_text = _item_visual_text(item)
    if visual_text:
        lines.extend(["", "#### 画面文字/OCR", "", visual_text])
    structured = _structured_entries(item)
    if structured:
        lines.extend(["", "#### 图文结构化", ""])
        for entry in structured:
            markdown = _meaningful_visual_text(entry.get("markdown"), item)
            if markdown:
                lines.extend([markdown, ""])
    visual = _item_visual_understanding(item)
    if visual:
        lines.extend(["", *_mapping_bullets("单帧视觉理解", visual, keys=["objects", "actions", "interface_state", "spatial_relations", "instructor_focus", "non_text_information", "keep_image_reason", "confidence"])])
    temporal = _item_temporal_understanding(item)
    if temporal:
        lines.extend(["", *_mapping_bullets("连续片段理解", temporal, keys=["event_sequence", "state_changes", "operation_steps", "causal_links", "possible_missing_points", "confidence"])])
    term_lines = _term_resolution_lines(item)
    if term_lines:
        lines.extend(["", "#### 术语仲裁", "", *term_lines])
    evidence = _evidence_paths(bundle_dir, item, visual=visual, temporal=temporal)
    if evidence:
        lines.extend(["", "#### 证据", ""])
        lines.extend(f"- `{path}`" for path in evidence[:12])
    lines.append("")
    return lines


def _timeline_script_lines(bundle_dir: Path, item: dict[str, Any], *, fallback_index: int) -> list[str]:
    index = int(item.get("index") or fallback_index)
    route = str(item.get("visual_route") or "unknown")
    issues = [str(issue) for issue in item.get("quality_issues") or []]
    transcript = _item_transcript(item)
    visual = _item_visual_understanding(item)
    temporal = _item_temporal_understanding(item)
    visual_text = _item_visual_text(item)
    structured = _structured_entries(item)
    evidence = _evidence_paths(bundle_dir, item, visual=visual, temporal=temporal)
    human_review = _mapping(item.get("human_review"))
    lines = [
        f"### {index}. {_time_range(item)}",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        f"| 画面路由 | `{route}` / {_route_label(route)} |",
        f"| 材料类型 | `{', '.join(str(value) for value in item.get('material_types') or []) or '未标记'}` |",
        f"| 缺口/风险 | `{', '.join(issues) if issues else '无'}` |",
        "",
        "#### 说了什么",
        "",
        transcript or "（这一段暂无转写。）",
        "",
    ]
    if human_review:
        lines.extend(["#### 人工审核", ""])
        lines.extend(_human_review_lines(human_review, item))
        lines.append("")
    term_lines = _term_resolution_lines(item)
    if term_lines:
        lines.extend(["#### 术语仲裁", ""])
        lines.extend(term_lines)
        lines.append("")
    lines.extend(["#### 演示了什么", ""])
    lines.extend(_visual_description_lines(item=item, visual=visual, temporal=temporal, visual_text=visual_text, structured=structured))
    if evidence:
        lines.extend(["", "#### 证据截图/帧", ""])
        lines.extend(f"- `{path}`" for path in evidence[:12])
    lines.append("")
    return lines


def _human_review_lines(review: dict[str, Any], item: dict[str, Any]) -> list[str]:
    lines = [f"- 状态：`{review.get('status') or item.get('review_status') or ''}`"]
    tags = review.get("tags") if isinstance(review.get("tags"), list) else []
    if tags:
        lines.append(f"- 标签：{', '.join(str(tag) for tag in tags)}")
    comment = _text(review.get("comment") or review.get("notes"))
    if comment:
        lines.append(f"- 说明：{comment}")
    corrected_transcript = _text(item.get("human_corrected_transcript") or review.get("corrected_transcript"))
    if corrected_transcript:
        lines.append(f"- 修正转写：{corrected_transcript}")
    corrected_visual_text = _text(item.get("human_corrected_visual_text") or review.get("corrected_visual_text"))
    if corrected_visual_text:
        lines.append(f"- 修正画面文字：{corrected_visual_text}")
    corrected_visual = _mapping(item.get("human_corrected_visual_understanding") or review.get("corrected_visual_understanding"))
    if corrected_visual:
        compact = _compact_mapping(
            corrected_visual,
            keys=["objects", "actions", "interface_state", "spatial_relations", "instructor_focus", "non_text_information", "keep_image_reason"],
        )
        if compact:
            lines.append(f"- 修正视觉理解：{compact}")
    corrected_temporal = _mapping(item.get("human_corrected_temporal_visual_understanding") or review.get("corrected_temporal_visual_understanding"))
    if corrected_temporal:
        compact = _compact_mapping(
            corrected_temporal,
            keys=["event_sequence", "state_changes", "operation_steps", "causal_links", "possible_missing_points"],
        )
        if compact:
            lines.append(f"- 修正连续片段理解：{compact}")
    return lines


def _visual_description_lines(
    *,
    item: dict[str, Any],
    visual: dict[str, Any],
    temporal: dict[str, Any],
    visual_text: str,
    structured: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    if visual_text:
        lines.extend(["- **画面文字/OCR**：" + visual_text])
    elif _human_keep_image(item):
        lines.extend(["- **人工保留图片**：该帧经人工确认保留为图片证据，未强行降维为文字。"])
    if structured:
        lines.append("- **图文结构化**：")
        for entry in structured:
            markdown = _meaningful_visual_text(entry.get("markdown"), item)
            if markdown:
                lines.append("")
                lines.append(markdown)
    tagger_tags = _item_tagger_tags(item)
    if tagger_tags:
        lines.append("- **RAM++ 候选标签（未人工确认）**：" + "、".join(tagger_tags))
    if visual:
        compact = _compact_mapping(
            visual,
            keys=["objects", "actions", "interface_state", "spatial_relations", "instructor_focus", "non_text_information", "keep_image_reason"],
        )
        if compact:
            lines.append("- **单帧视觉理解**：" + compact)
    if temporal:
        compact = _compact_mapping(
            temporal,
            keys=["event_sequence", "state_changes", "operation_steps", "causal_links", "possible_missing_points"],
        )
        if compact:
            lines.append("- **连续片段理解**：" + compact)
    return lines or ["（这一段还没有完成画面理解；需要 OCR/图文解析或多模态视觉补齐。）"]


def _term_resolution_lines(item: dict[str, Any]) -> list[str]:
    rows = item.get("term_candidates") if isinstance(item.get("term_candidates"), list) else []
    if not rows:
        return []
    lines: list[str] = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        canonical = _text(row.get("canonical_term"))
        raw_mentions = row.get("raw_mentions") if isinstance(row.get("raw_mentions"), list) else []
        raw_text = ", ".join(_text(value) for value in raw_mentions if _text(value))
        confidence = row.get("confidence")
        review = "已自动采用" if is_high_confidence_term_candidate(row) else ("需人工复核" if row.get("needs_human_review") else "可暂用")
        source = _text(row.get("evidence_source"))
        parts = [f"建议：`{canonical or '未定'}`"]
        if raw_text:
            parts.append(f"原始候选：{raw_text}")
        if confidence is not None:
            parts.append(f"置信度：`{confidence}`")
        if source:
            parts.append(f"证据来源：`{source}`")
        parts.append(f"状态：{review}")
        lines.append("- " + "；".join(parts))
    return lines


def _compact_mapping(value: dict[str, Any], *, keys: list[str]) -> str:
    labels = {
        "objects": "对象",
        "actions": "动作",
        "interface_state": "界面状态",
        "spatial_relations": "空间关系",
        "instructor_focus": "讲师强调",
        "non_text_information": "非文字信息",
        "keep_image_reason": "保留截图理由",
        "event_sequence": "事件",
        "state_changes": "变化",
        "operation_steps": "步骤",
        "causal_links": "因果",
        "possible_missing_points": "可能遗漏",
    }
    parts = []
    for key in keys:
        rendered = _render_value(value.get(key))
        if rendered:
            parts.append(f"{labels.get(key, key)}：{rendered}")
    return "；".join(parts)


def _mapping_bullets(title: str, value: dict[str, Any], *, keys: list[str]) -> list[str]:
    lines = [f"#### {title}", ""]
    labels = {
        "objects": "对象",
        "actions": "动作",
        "interface_state": "界面状态",
        "spatial_relations": "空间关系",
        "instructor_focus": "讲师强调",
        "non_text_information": "非文字信息",
        "keep_image_reason": "保留截图理由",
        "confidence": "置信度",
        "event_sequence": "事件序列",
        "state_changes": "状态变化",
        "operation_steps": "操作步骤",
        "causal_links": "前后因果",
        "possible_missing_points": "可能遗漏点",
    }
    for key in keys:
        if key not in value:
            continue
        rendered = _render_value(value.get(key))
        if not rendered:
            continue
        lines.append(f"- **{labels.get(key, key)}**：{rendered}")
    return lines


def _render_value(value: Any) -> str:
    if isinstance(value, list):
        parts = [_text(item) for item in value if _text(item)]
        return "；".join(parts)
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_text(val)}" for key, val in value.items() if _text(val))
    return _text(value)


def _table_cell(value: str) -> str:
    return _text(value).replace("|", "\\|").replace("\n", "<br>")


def _render_gap_list(title: str, indexes: list[Any]) -> list[str]:
    lines = [f"### {title}", ""]
    values = [str(value) for value in indexes]
    if not values:
        lines.append("- 无")
    else:
        lines.append("- " + ", ".join(values[:80]))
    return lines


def _coverage_table_lines(summary: dict[str, Any]) -> list[str]:
    rows = [
        ("时间线片段", summary.get("timeline_items", 0)),
        ("有转写", summary.get("items_with_transcript", 0)),
        ("有画面文字/OCR", summary.get("items_with_visual_text", 0)),
        ("有图文结构化", summary.get("items_with_structured_visual", 0)),
        ("有单帧视觉理解", summary.get("items_with_visual_understanding", 0)),
        ("有连续片段理解", summary.get("items_with_temporal_understanding", 0)),
        ("覆盖审计状态", summary.get("coverage_status") or "unknown"),
    ]
    lines = ["| 项目 | 数值 |", "| --- | --- |"]
    lines.extend(f"| {label} | `{value}` |" for label, value in rows)
    return lines


def _route_table_lines(summary: dict[str, Any]) -> list[str]:
    route_counts = summary.get("route_counts") if isinstance(summary.get("route_counts"), dict) else {}
    if not route_counts:
        return ["（暂无画面路由结果。）"]
    lines = ["| 路由 | 含义 | 数量 |", "| --- | --- | ---: |"]
    for route, count in sorted(route_counts.items()):
        lines.append(f"| `{route}` | {_route_label(str(route))} | {count} |")
    return lines


def _auto_overview(title: str, manifest: dict[str, Any], timeline: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    first_text = ""
    for item in timeline:
        first_text = _item_transcript(item)
        if first_text:
            break
    duration = _format_time(_timeline_duration(timeline))
    source_hint = ", ".join(_plain_source_names(manifest)) or "本地视频"
    excerpt = _truncate(first_text, 180) if first_text else "暂无可用转写。"
    return (
        f"本视频《{title}》来自 {source_hint}，已整理为 `{summary.get('timeline_items', 0)}` 个时间线片段，"
        f"视频时间覆盖约 `{duration}`。当前稿件基于 ASR 转写、画面路由和已完成的视觉理解自动生成；"
        f"概要线索：{excerpt}"
    )


def _route_label(route: str) -> str:
    return {
        "document_visual": "图文型画面，适合 OCR/版面/表格/公式/代码解析",
        "semantic_frame": "非纯文字画面，适合多模态单帧理解",
        "temporal_sequence": "连续变化画面，适合多帧事件/操作链理解",
        "mixed": "混合画面，需要图文解析和多模态理解共同处理",
        "unknown": "未分类",
    }.get(route, "未分类")


def _timeline_duration(timeline: list[dict[str, Any]]) -> float:
    values = []
    for item in timeline:
        try:
            values.append(float(item.get("end") or 0))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0.0


def _plain_source_names(manifest: dict[str, Any]) -> list[str]:
    names = []
    for source in manifest.get("sources") or []:
        if isinstance(source, dict):
            name = _text(source.get("title") or source.get("id") or source.get("source_id") or source.get("path") or source.get("url"))
            if name:
                names.append(name)
    return names


def _truncate(value: str, limit: int) -> str:
    value = _text(value)
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _source_lines(manifest: dict[str, Any]) -> list[str]:
    lines = []
    for source in manifest.get("sources") or []:
        if not isinstance(source, dict):
            continue
        label = _text(source.get("title") or source.get("id") or source.get("source_id"))
        path = _text(source.get("path") or source.get("url") or source.get("local_media_path"))
        if label or path:
            lines.append(f"- {label or 'source'} `{path}`")
    if not lines:
        for source in manifest.get("source_records") or []:
            if isinstance(source, dict):
                lines.append(f"- {_text(source.get('title') or source.get('id'))} `{_text(source.get('path') or source.get('url'))}`")
    return lines


def _evidence_paths(bundle_dir: Path, item: dict[str, Any], *, visual: dict[str, Any] | None = None, temporal: dict[str, Any] | None = None) -> list[str]:
    paths: list[str] = []
    for source in (visual or {}).get("evidence_frame_paths") or []:
        paths.append(str(source))
    for source in (temporal or {}).get("evidence_frame_paths") or []:
        paths.append(str(source))
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            raw = _text(asset.get("path") or asset.get("source"))
            if raw:
                path = Path(raw)
                paths.append(str(path if path.is_absolute() else bundle_dir / path))
    for raw in item.get("frame_paths") or []:
        if raw:
            path = Path(str(raw))
            paths.append(str(path if path.is_absolute() else bundle_dir / path))
    recovery = item.get("screen_text_recovery") if isinstance(item.get("screen_text_recovery"), dict) else {}
    for raw in recovery.get("crop_paths") or []:
        if raw:
            path = Path(str(raw))
            paths.append(str(path if path.is_absolute() else bundle_dir / path))
    return _dedupe(paths)


def _time_range(item: dict[str, Any]) -> str:
    return f"{_format_time(item.get('start'))} - {_format_time(item.get('end'))}"


def _format_time(value: Any) -> str:
    try:
        seconds = max(0.0, float(value))
    except (TypeError, ValueError):
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{sec:02d}.{ms:03d}"


def _float_seconds(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _needs_visual_understanding(item: dict[str, Any]) -> bool:
    if _human_review_accepted(item):
        return False
    route = str(item.get("visual_route") or "")
    if route in {"semantic_frame", "mixed"} and not _item_visual_understanding(item):
        return True
    if route in {"temporal_sequence", "mixed"} and not _item_temporal_understanding(item):
        return True
    return False


def _needs_document_structure(item: dict[str, Any]) -> bool:
    route = str(item.get("visual_route") or "")
    material_types = {str(value) for value in item.get("material_types") or []}
    needs = route in {"document_visual", "mixed"} or bool(material_types & {"table", "formula", "code"})
    return needs and not (_structured_entries(item) or _human_structured_fallback(item))


def _structured_entries(item: dict[str, Any]) -> list[dict[str, Any]]:
    value = item.get("structured_visual")
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict) and _meaningful_visual_text(entry.get("markdown") or entry.get("text"), item)]


def _item_transcript(item: dict[str, Any]) -> str:
    """Project explicit human review before canonical/machine transcript text.

    Machine and canonical artifacts remain unchanged; this is only the
    reader-facing projection of an already persisted human review decision.
    """

    corrected = _text(
        item.get("human_corrected_transcript")
        or _human_review(item).get("corrected_transcript")
        or item.get("corrected_transcript")
    )
    if corrected:
        return corrected
    return apply_high_confidence_term_replacements(_text(item.get("transcript") or item.get("original_transcript")), item)


def _item_visual_text(item: dict[str, Any]) -> str:
    value = (
        _meaningful_visual_text(item.get("human_corrected_visual_text"), item)
        or _meaningful_visual_text(_human_review(item).get("corrected_visual_text"), item)
        or _meaningful_visual_text(item.get("visual_text") or item.get("original_visual_text"), item)
    )
    return apply_high_confidence_term_replacements(value, item)


def _item_visual_understanding(item: dict[str, Any]) -> dict[str, Any]:
    return visual_understanding_value(item)


def _item_tagger_tags(item: dict[str, Any], *, limit: int = 12) -> list[str]:
    values = item.get("tagger_tags")
    if not isinstance(values, list):
        integrated = item.get("integrated_visual")
        integrated = integrated if isinstance(integrated, dict) else {}
        values = integrated.get("tagger_tags")
    if not isinstance(values, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = _text(value).replace("\r", " ").replace("\n", " ").strip()
        key = tag.casefold()
        if not tag or key in seen:
            continue
        seen.add(key)
        tags.append(_truncate(tag, 80))
        if len(tags) >= max(1, int(limit)):
            break
    return tags


def _item_temporal_understanding(item: dict[str, Any]) -> dict[str, Any]:
    return temporal_understanding_value(item)


def _human_review(item: dict[str, Any]) -> dict[str, Any]:
    return item.get("human_review") if isinstance(item.get("human_review"), dict) else {}


def _human_review_accepted(item: dict[str, Any]) -> bool:
    return _shared_human_review_accepted(item)


def _human_keep_image(item: dict[str, Any]) -> bool:
    return bool(item.get("human_keep_image") or _human_review(item).get("keep_image"))


def _human_structured_fallback(item: dict[str, Any]) -> bool:
    material_types = {str(value) for value in item.get("material_types") or []}
    return bool(material_types & {"table", "formula", "code"}) and _human_review_accepted(item) and (
        _human_keep_image(item) or bool(_item_visual_text(item)) or bool(_item_visual_understanding(item))
    )


def _meaningful_visual_text(value: Any, item: dict[str, Any]) -> str:
    text = _text(value)
    if not text:
        return ""
    stems = _frame_stems(item)
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->") and "source:" in stripped.lower():
            continue
        if stripped.startswith("# ") and stripped[2:].strip() in stems:
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip()


def _frame_stems(item: dict[str, Any]) -> set[str]:
    stems: set[str] = set()
    for key in ("frame_paths", "evidence_paths"):
        values = item.get(key)
        if isinstance(values, list):
            for value in values:
                text = str(value or "").strip()
                if text:
                    stems.add(Path(text).stem)
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            text = str(asset.get("path") or asset.get("source") or "").strip()
            if text:
                stems.add(Path(text).stem)
    for entry in item.get("structured_visual") or []:
        if isinstance(entry, dict):
            for key in ("image_path", "source"):
                text = str(entry.get(key) or "").strip()
                if text and Path(text).suffix:
                    stems.add(Path(text).stem)
    return {stem for stem in stems if stem}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) and value else {}


def _text(value: Any) -> str:
    return str(value or "").strip()



def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _frontmatter_escape(value: str) -> str:
    return value.replace('"', '\\"')
