from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.bundle_next import bundle_advance, bundle_next_action
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.knowledge_coverage import (
    audit_knowledge_coverage,
    build_knowledge_coverage,
)
from video_knowledge_pipeline.timeline_alignment_audit import timeline_alignment_audit


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    _write_json(
        root / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "normalized_transcript_json": "normalized-transcript.json",
            "transcript_json": "normalized-transcript.json",
            # This is intentionally the wrong args file for normalize_asr_output.
            # The binding route must never reuse it.
            "mcp_refresh_args": "mcp-refresh-lecture-review.args.json",
        },
    )
    _write_json(
        root / "timeline.json",
        [
            {"index": 1, "start": 0.0, "end": 4.0, "transcript": ""},
            {"index": 2, "start": 4.0, "end": 8.0, "transcript": ""},
        ],
    )
    _write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0.2, "end": 3.8, "text": "第一段合成转写。"},
                {"start": 4.2, "end": 7.8, "text": "第二段合成转写。"},
            ]
        },
    )
    _write_json(
        root / "mcp-refresh-lecture-review.args.json",
        {"root": str(root.parent), "webui_output_dir": str(root)},
    )
    return root


def test_existing_normalized_transcript_routes_to_binding_refresh_not_asr(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    timeline_before = (bundle / "timeline.json").read_bytes()
    transcript_before = sha256_file(bundle / "normalized-transcript.json")
    audit_knowledge_coverage(bundle, write=True)

    pending = bundle_next_action(bundle, refresh=True)

    assert pending["next_action"]["key"] == "speech"
    assert pending["next_action"]["mcp_tool"] == "timeline_alignment_audit"
    assert pending["next_action"]["mcp_args_path"].endswith(
        "mcp-timeline-alignment-audit.args.json"
    )
    assert Path(pending["next_action"]["mcp_args_path"]).is_file()
    assert "normalize_asr_output" not in pending["next_action"]["command"]

    advanced = bundle_advance(bundle, execute=False, refresh_outputs=False)

    assert advanced["advanced"] is True
    assert advanced["before"]["next_action"]["key"] == "speech"
    assert advanced["after"]["next_action"]["key"] != "speech"
    assert (bundle / "timeline.json").read_bytes() == timeline_before
    assert sha256_file(bundle / "normalized-transcript.json") == transcript_before
    coverage = json.loads((bundle / "knowledge-coverage.json").read_text(encoding="utf-8"))
    assert coverage["canonical_transcript_aligned_items"] == 2


def test_transcript_hash_drift_invalidates_alignment_even_when_mtime_is_not_newer(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    first = timeline_alignment_audit(bundle, write=True)
    audit_path = bundle / "timeline-alignment-audit.json"
    audit_mtime = audit_path.stat().st_mtime_ns

    _write_json(
        bundle / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0.2, "end": 3.8, "text": "内容已经变化。"},
                {"start": 4.2, "end": 7.8, "text": "第二段也变化。"},
            ]
        },
    )
    # A copied/restored file can preserve an older mtime. Identity must still
    # be decided by the content hash, not by timestamp ordering alone.
    import os

    os.utime(bundle / "normalized-transcript.json", ns=(audit_mtime, audit_mtime))
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    coverage = build_knowledge_coverage(manifest, timeline, bundle_dir=bundle)

    assert first["transcript_sha256"] != sha256_file(
        bundle / "normalized-transcript.json"
    )
    assert coverage["canonical_transcript_aligned_items"] == 0
    assert coverage["next_action"]["mcp_tool"] == "timeline_alignment_audit"


def test_empty_ocr_plan_yields_to_pending_local_visual_understanding(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "visual-bundle"
    bundle.mkdir()
    _write_json(
        bundle / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "mcp_multimodal_frame_analysis_args": "mcp-run-multimodal-frame-analysis.args.json",
            "knowledge_coverage": {
                "status": "blocked",
                "next_action": {
                    "key": "screen_text",
                    "label": "屏幕文字",
                    "mcp_tool": "run_visual_structure_plan",
                    "mcp_args_path": "mcp-run-visual-structure.args.json",
                    "hint": "需要屏幕文字证据",
                },
                "blockers": [{"key": "screen_text"}],
                "weak_channels": [],
            },
            "visual_structure": {
                "last_run": {
                    "total_candidates": 0,
                    "selected_indexes": [],
                    "updated": 0,
                },
                "ebook_pipeline_results": [],
            },
        },
    )
    _write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": 4.0,
                "transcript": "已有转写",
                "visual_route": "semantic_frame",
                "visual_understanding": {},
            }
        ],
    )
    _write_json(
        bundle / "mcp-run-multimodal-frame-analysis.args.json",
        {"bundle_dir": str(bundle), "execute": False},
    )

    pending = bundle_next_action(bundle, refresh=False)

    assert pending["next_action"]["key"] == "multimodal_frame_analysis"
    assert pending["next_action"]["mcp_tool"] == "run_multimodal_frame_analysis"
    assert pending["next_action"]["human_required"] is False
    assert pending["next_action"]["evidence_boundary"] == (
        "visual_understanding_does_not_close_screen_text"
    )
