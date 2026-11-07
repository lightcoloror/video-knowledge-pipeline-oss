from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.review_session import apply_review_notes_to_bundle, prepare_review_session, review_closure_status


def test_review_statuses_distinguish_known_gap_keep_image_and_rerun_ocr(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "review_notes": "review-notes.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {"index": 1, "start": 0, "end": 1, "visual_route": "semantic_frame", "assets": [{"path": str(frame)}], "quality_issues": ["missing_visual_text"]},
                {"index": 2, "start": 1, "end": 2, "visual_route": "semantic_frame", "assets": [{"path": str(frame)}], "quality_issues": ["screen_text_low_confidence"]},
                {"index": 3, "start": 2, "end": 3, "visual_route": "document_visual", "assets": [{"path": str(frame)}], "quality_issues": ["ocr_text_empty"]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "review-notes.json").write_text(
        json.dumps(
            {
                "reviews": [
                    {"timeline_index": 1, "status": "accepted_known_gap", "comment": "small text cannot be recovered", "evidence_frame_paths": [str(frame)]},
                    {"timeline_index": 2, "status": "needs_rerun_ocr", "comment": "crop again", "evidence_frame_paths": [str(frame)]},
                    {"timeline_index": 3, "status": "keep_image", "comment": "must keep screenshot", "evidence_frame_paths": [str(frame)]},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = apply_review_notes_to_bundle(bundle)

    assert result["updated_indexes"] == [1, 2, 3]
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[0]["review_status"] == "accepted_known_gap"
    assert timeline[0]["needs_human_review"] is False
    assert "missing_visual_text" not in timeline[0]["quality_issues"]
    assert timeline[1]["review_status"] == "needs_rerun_ocr"
    assert timeline[1]["needs_human_review"] is True
    assert "screen_text_low_confidence" in timeline[1]["quality_issues"]
    assert timeline[2]["review_status"] == "keep_image"
    assert timeline[2]["human_keep_image"] is True
    assert result["post_apply_refresh"]["knowledge_note"].endswith("knowledge-note.md")


def test_prepare_review_session_suggests_screen_text_statuses(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {"index": 1, "start": 0, "end": 1, "visual_route": "semantic_frame", "quality_issues": ["screen_text_low_confidence"]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    session = prepare_review_session(bundle, refresh=False, limit=0)
    template = json.loads(Path(session["review_notes_template_path"]).read_text(encoding="utf-8"))

    assert template["reviews"][0]["suggested_status"] == "corrected_visual_text"
    guide = Path(session["review_fill_guide_path"]).read_text(encoding="utf-8")
    assert "accepted_known_gap" in guide
    assert "needs_rerun_ocr" in guide


def test_prepare_review_session_writes_review_pack_and_closure_status(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    crops = bundle / "ocr-crops"
    assets.mkdir(parents=True)
    crops.mkdir()
    frame = assets / "frame.jpg"
    crop = crops / "frame-crop.jpg"
    frame.write_bytes(b"fake image")
    crop.write_bytes(b"fake crop")
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 1,
                    "transcript": "这里有小字需要补。",
                    "visual_route": "document_visual",
                    "assets": [{"path": str(frame)}],
                    "quality_issues": ["missing_visual_text"],
                    "screen_text_recovery": {"crop_paths": [str(crop)]},
                },
                {
                    "index": 2,
                    "start": 1,
                    "end": 2,
                    "visual_route": "semantic_frame",
                    "review_status": "accepted_known_gap",
                    "quality_issues": ["missing_visual_text"],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    session = prepare_review_session(bundle, refresh=False, limit=0)
    pack = json.loads((bundle / "review-pack.json").read_text(encoding="utf-8"))
    todo = json.loads((bundle / "review-notes.todo.json").read_text(encoding="utf-8"))
    closure = json.loads((bundle / "review-closure-status.json").read_text(encoding="utf-8"))

    assert Path(session["review_pack_path"]).exists()
    assert Path(session["review_closure_status_path"]).exists()
    assert pack["summary"]["open_targets"] == 1
    assert pack["groups"][0]["key"] == "missing_visual_text"
    assert str(crop) in pack["groups"][0]["items"][0]["evidence_paths"]
    assert todo["reviews"][0]["timeline_index"] == 1
    assert closure["summary"]["open"] == 1
    assert closure["summary"]["closed"] == 1


def test_review_closure_status_counts_imported_invalid_and_next_batch(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "review_notes": "review-notes.json", "review_notes_last_import": {"updated": 2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {"index": 1, "start": 0, "end": 1, "visual_route": "semantic_frame", "quality_issues": ["semantic_frame_without_analysis"]},
                {"index": 2, "start": 1, "end": 2, "review_status": "keep_image"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "review-notes.json").write_text(
        json.dumps({"reviews": [{"timeline_index": 1, "status": "accepted"}, {"timeline_index": 1, "status": "accepted"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    status = review_closure_status(bundle)

    assert status["summary"]["open"] == 1
    assert status["summary"]["closed"] == 1
    assert status["summary"]["imported"] == 2
    assert status["summary"]["invalid"] == 1
    assert "prepare-review-session" in status["next_batch"]["command"]
    assert (bundle / "review-closure-status.md").exists()

# Moved from test_video_pipeline_smoke.py during Phase 10 split.

import json
from pathlib import Path

from video_knowledge_pipeline.acceptance_check import acceptance_check
from video_knowledge_pipeline.acceptance_run import run_acceptance_bundle, run_acceptance_run
from video_knowledge_pipeline.asr_adapter import normalize_asr_output
from video_knowledge_pipeline.asr_environment import asr_environment_status
from video_knowledge_pipeline.asr_execution import asr_smoke, run_asr_plan
from video_knowledge_pipeline.asr_runner import plan_asr_run
from video_knowledge_pipeline.batch_run import batch_video_knowledge_run
from video_knowledge_pipeline.bundle_next import bundle_advance, bundle_advance_log, bundle_advance_queue, bundle_next_action
from video_knowledge_pipeline.bundle_status import bundle_status_report, controlled_execution_check
from video_knowledge_pipeline.cli import audit_bundle_mcp_args, build_parser, main as cli_main, resolve_mcp_args_path, run_mcp_call
from video_knowledge_pipeline.config import config_status, resolve_vision_execution_profile, service_url, vision_execution_profile
from video_knowledge_pipeline.controlled_execution_smoke import controlled_execution_smoke
from video_knowledge_pipeline.knowledge_coverage import build_knowledge_coverage
from video_knowledge_pipeline.knowledge_note_export import export_knowledge_note
from video_knowledge_pipeline.lecture_package import render_lecture_review_html
from video_knowledge_pipeline.local_video_run import prepare_local_video_run
from video_knowledge_pipeline.local_vlm_server_adapter import local_vlm_adapter_plan
from video_knowledge_pipeline.ocr_backfill import run_ocr_backfill
from video_knowledge_pipeline.multimodal_frame_analyzer import (
    _normalise_visual_understanding,
    run_multimodal_frame_analysis,
    vision_analysis_apply_restore,
    vision_analysis_restore_plan,
    vision_analysis_run_log,
)
from video_knowledge_pipeline.peepshow_adapter import attach_peepshow_output_to_bundle
from video_knowledge_pipeline.review_session import apply_review_notes_to_bundle, prepare_review_session, review_closure_status, validate_review_notes_for_bundle
from video_knowledge_pipeline.source_artifacts import build_source_artifact_index, summarize_manifest_source_artifacts
from video_knowledge_pipeline.storage import bundle_write_lock, write_json
from video_knowledge_pipeline.temporal_frame_groups import run_temporal_frame_groups
from video_knowledge_pipeline.temporal_visual_analyzer import _normalise_temporal_understanding, run_temporal_visual_analysis
from video_knowledge_pipeline.transcript_resegment import resegment_transcript
from video_knowledge_pipeline.vision_acceptance import vision_acceptance_plan
from video_knowledge_pipeline.video_frame_router import run_video_frame_router
from video_knowledge_pipeline.video_source import prepare_video_source
from video_knowledge_pipeline.vision_api import parse_model_json, resolve_provider_config, test_vision_provider as run_vision_provider_test
from video_knowledge_pipeline.vision_environment import vision_environment_status
from video_knowledge_pipeline.vision_preflight import vision_execution_preflight
from video_knowledge_pipeline.vision_provider_smoke import rank_vision_providers, vision_provider_matrix, vision_provider_smoke
from video_knowledge_pipeline.webui_bridge import export_webui_bundle, refresh_bundle_review_html
import video_knowledge_pipeline.visual_structure as visual_structure
from video_knowledge_pipeline.visual_structure import run_visual_structure_plan



def test_refresh_lecture_review_cli_contract_matches_bundle_readme() -> None:
    args = build_parser().parse_args(
        [
            "refresh-lecture-review",
            "project",
            "bundle/review-notes.json",
            "--webui-output-dir",
            "bundle",
            "--target",
            "bilinote",
            "--allow-blocked-export",
        ]
    )

    assert args.command == "refresh-lecture-review"
    assert args.project == "project"
    assert args.review_json == "bundle/review-notes.json"
    assert args.webui_output_dir == "bundle"
    assert args.target == "bilinote"
    assert args.allow_blocked_export is True


def test_apply_review_notes_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "apply-review-notes",
            "bundle",
            "--review-json",
            "bundle/review-notes.json",
            "--no-write",
        ]
    )

    assert args.command == "apply-review-notes"
    assert args.bundle_dir == "bundle"
    assert args.review_json == "bundle/review-notes.json"
    assert args.no_write is True


def test_validate_review_notes_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "validate-review-notes",
            "bundle",
            "--review-json",
            "bundle/review-notes.json",
        ]
    )

    assert args.command == "validate-review-notes"
    assert args.bundle_dir == "bundle"
    assert args.review_json == "bundle/review-notes.json"


def test_prepare_review_session_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "prepare-review-session",
            "bundle",
            "--no-refresh",
            "--limit",
            "80",
            "--offset",
            "10",
            "--reason",
            "semantic_frame_without_analysis",
            "--group-by",
            "suggested_status",
            "--include-closed",
            "--output-prefix",
            "review-pack-semantic",
        ]
    )

    assert args.command == "prepare-review-session"
    assert args.bundle_dir == "bundle"
    assert args.no_refresh is True
    assert args.limit == 80
    assert args.offset == 10
    assert args.reason == "semantic_frame_without_analysis"
    assert args.group_by == "suggested_status"
    assert args.include_closed is True
    assert args.output_prefix == "review-pack-semantic"


def test_review_closure_status_cli_contract() -> None:
    args = build_parser().parse_args(["review-closure-status", "bundle", "--no-write"])

    assert args.command == "review-closure-status"
    assert args.bundle_dir == "bundle"
    assert args.no_write is True


def test_apply_review_notes_roundtrip_preserves_machine_outputs_and_exports(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    manifest = {
        "schema": "lecture_webui_bundle.v1",
        "title": "review roundtrip",
        "review_notes": "review-notes.json",
    }
    timeline = [
        {
            "index": 1,
            "start": 0.0,
            "end": 5.0,
            "transcript": "机器转写原文",
            "visual_route": "semantic_frame",
            "frame_paths": [str(frame)],
            "visual_understanding": {},
            "quality_issues": ["needs_human_review", "semantic_frame_without_analysis", "missing_visual_understanding"],
        }
    ]
    review_notes = {
        "schema": "lecture_review_notes.v1",
        "reviews": [
            {
                "timeline_index": 1,
                "status": "accepted",
                "tags": ["人工确认"],
                "comment": "人工确认这一帧已经足够。",
                "corrected_transcript": "人工修正转写",
                "corrected_visual_text": "人工补充画面文字",
                "corrected_visual_understanding": {
                    "objects": ["界面"],
                    "actions": ["演示"],
                    "evidence_frame_paths": [str(frame)],
                },
            }
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    (bundle / "review-notes.json").write_text(json.dumps(review_notes, ensure_ascii=False), encoding="utf-8")

    result = apply_review_notes_to_bundle(bundle)

    assert result["updated_indexes"] == [1]
    updated_timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    item = updated_timeline[0]
    assert item["transcript"] == "机器转写原文"
    assert item["visual_understanding"] == {}
    assert item["human_corrected_transcript"] == "人工修正转写"
    assert item["human_corrected_visual_text"] == "人工补充画面文字"
    assert item["human_corrected_visual_understanding"]["objects"] == ["界面"]
    assert item["review_status"] == "accepted"
    assert item["needs_human_review"] is False
    assert "semantic_frame_without_analysis" not in item["quality_issues"]
    assert result["post_apply_refresh"]["review_closure_status"].endswith("review-closure-status.md")
    assert (bundle / "review-closure-status.json").exists()
    coverage = json.loads((bundle / "knowledge-coverage.json").read_text(encoding="utf-8"))
    assert coverage["items_with_visual_understanding"] == 1
    readiness = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))["review_readiness"]
    assert readiness["counts"]["reviewed_items"] == 1

    export = export_knowledge_note(bundle, title="review roundtrip")
    note = Path(export["note_path"]).read_text(encoding="utf-8")
    assert "- 人工审核：accepted；人工确认这一帧已经足够。" in note
    assert "人工修正转写" in note
    assert "人工补充画面文字" in note


def test_review_notes_validation_skips_invalid_corrections_and_duplicates(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-review-validation"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "review_notes": "review-notes.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 1,
                    "transcript": "需要视觉理解",
                    "visual_route": "semantic_frame",
                    "assets": [{"path": str(frame)}],
                    "quality_issues": ["semantic_frame_without_analysis"],
                },
                {
                    "index": 2,
                    "start": 1,
                    "end": 2,
                    "transcript": "需要连续理解",
                    "visual_route": "temporal_sequence",
                    "assets": [{"path": str(frame)}],
                    "quality_issues": ["temporal_sequence_without_analysis"],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "review-notes.json").write_text(
        json.dumps(
            {
                "schema": "lecture_review_notes.v1",
                "reviews": [
                    {
                        "timeline_index": 1,
                        "status": "corrected_visual_understanding",
                        "comment": "缺少 required corrected_visual_understanding，应跳过。",
                    },
                    {
                        "timeline_index": 2,
                        "status": "corrected_temporal_visual_understanding",
                        "corrected_temporal_visual_understanding": {
                            "event_sequence": ["先展示界面，再切换状态"],
                            "state_changes": ["界面状态变化"],
                            "evidence_frame_paths": [str(frame)],
                        },
                    },
                    {
                        "timeline_index": 2,
                        "status": "accepted",
                        "comment": "重复索引，应跳过。",
                    },
                    {
                        "timeline_index": 999,
                        "status": "accepted",
                        "comment": "未知索引，应跳过。",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    validation = validate_review_notes_for_bundle(bundle)
    result = apply_review_notes_to_bundle(bundle)
    updated = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))

    assert validation["status"] == "has_errors"
    assert validation["error_count"] == 3
    assert {error["key"] for error in validation["errors"]} == {
        "missing_corrected_visual_understanding",
        "duplicate_timeline_index",
        "timeline_index_not_found",
    }
    assert result["validation"]["status"] == "has_errors"
    assert result["updated_indexes"] == [2]
    assert [row["reason"] for row in result["skipped"]] == ["validation_error", "validation_error", "validation_error"]
    assert "human_corrected_visual_understanding" not in updated[0]
    assert updated[1]["human_corrected_temporal_visual_understanding"]["event_sequence"]


def test_review_notes_keep_image_resolves_ocr_structure_and_temporal_gaps(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "016_0000240000ms.jpg"
    frame.write_bytes(b"fake image")
    manifest = {
        "schema": "lecture_webui_bundle.v1",
        "title": "keep image review",
        "review_notes": "review-notes.json",
        "visual_structure": {
            "last_run": {
                "ebook_pipeline_total": 1,
                "ebook_pipeline_succeeded": 0,
                "ebook_pipeline_blockers": {"ocr_text_empty": 1},
            },
            "ebook_pipeline_results": [{"index": 16, "ok": False, "blocker": "ocr_text_empty"}],
        },
    }
    timeline = [
        {
            "index": 16,
            "start": 24.0,
            "end": 29.0,
            "transcript": "这里展示一个表格，但 OCR 没识别出来。",
            "visual_route": "mixed",
            "material_types": ["table"],
            "frame_paths": [str(frame)],
            "assets": [{"path": str(frame), "copied": "true"}],
            "visual_text": f"# {frame.stem}\n\n<!-- source: {frame} -->",
            "quality_issues": [
                "needs_human_review",
                "missing_visual_text",
                "ocr_text_empty",
                "structured_visual_without_structure",
                "semantic_frame_without_analysis",
                "temporal_sequence_without_analysis",
                "missing_visual_understanding",
            ],
        }
    ]
    review_notes = {
        "schema": "lecture_review_notes.v1",
        "reviews": [
            {
                "timeline_index": 16,
                "status": "keep_image",
                "tags": ["保留截图", "OCR空结果已审核"],
                "comment": "这一帧主要保留为图片证据，不强行 OCR。",
                "corrected_temporal_visual_understanding": {
                    "event_sequence": ["画面停留在表格/文档页，讲解其结构"],
                    "state_changes": ["无明显动态变化"],
                    "operation_steps": ["人工确认保留截图"],
                    "evidence_frame_paths": [str(frame)],
                },
            }
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    (bundle / "review-notes.json").write_text(json.dumps(review_notes, ensure_ascii=False), encoding="utf-8")

    result = apply_review_notes_to_bundle(bundle)
    updated = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))[0]
    coverage = json.loads((bundle / "knowledge-coverage.json").read_text(encoding="utf-8"))
    channels = {channel["key"]: channel for channel in coverage["channels"]}
    readiness = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))["review_readiness"]
    export = export_knowledge_note(bundle, title="keep image review")
    note = Path(export["note_path"]).read_text(encoding="utf-8")

    assert result["updated_indexes"] == [16]
    assert updated["review_status"] == "keep_image"
    assert updated["human_keep_image"] is True
    assert updated["human_corrected_temporal_visual_understanding"]["event_sequence"]
    assert "ocr_text_empty" not in updated["quality_issues"]
    assert "structured_visual_without_structure" not in updated["quality_issues"]
    assert channels["screen_text"]["blocker_count"] == 0
    assert channels["structured_visual"]["covered_count"] == 1
    assert channels["structured_visual"]["blocker_count"] == 0
    assert channels["semantic_frame_understanding"]["covered_count"] == 1
    assert channels["temporal_visual_understanding"]["covered_count"] == 1
    assert readiness["counts"]["pending_review"] == 0
    assert readiness["counts"]["pending_structured"] == 0
    assert readiness["counts"]["visual_analysis_gap_items"] == 0
    assert "- 人工审核：keep_image；这一帧主要保留为图片证据，不强行 OCR。" in note
    assert "- 连续片段：事件：" in note
    assert "图文截图/表格/代码/公式待解析" not in note


def test_apply_review_notes_no_write_does_not_create_missing_review_file(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")

    result = apply_review_notes_to_bundle(bundle, write=False)

    assert result["write"] is False
    assert result["review_count"] == 0
    assert not (bundle / "review-notes.json").exists()


def test_direct_multimodal_preview_uses_unified_provider(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "video-knowledge-pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.config.v1",
                "services": {
                    "review_webui": {"type": "static_file", "entrypoint": "webui-bundle/review.html"},
                    "ebook_markdown_pipeline_http": {"host": "127.0.0.1", "port": 9876, "path": "/call"},
                    "mcp": {"transport": "stdio"},
                },
                "vision_execution": {"provider": "gemini", "model": "gemini-2.5-flash"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))
    monkeypatch.setenv("VKP_MODEL_API_SETTINGS_PATH", str(tmp_path / "missing-model-api-settings.json"))
    monkeypatch.delenv("LECTURE_VISION_PROVIDER", raising=False)
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [{"index": 1, "start": 0, "end": 1, "visual_route": "semantic_frame", "frame_paths": [str(frame)]}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    preview = run_multimodal_frame_analysis(bundle, limit=1)

    assert preview["summary"]["provider"]["provider"] == "gemini"
    assert preview["summary"]["provider"]["model"] == "gemini-2.5-flash"


def test_asr_smoke_preview_writes_local_report(tmp_path: Path) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    result = asr_smoke(media, output_dir=tmp_path / "smoke", duration_seconds=7, execute=False)

    assert result["status"] == "preview"
    assert result["execute"] is False
    assert result["duration_seconds"] == 7
    assert result["clip_command"][0]
    assert "Audio stays on this machine" in result["privacy"]
    assert Path(result["json_path"]).exists()
    assert Path(result["report_path"]).exists()


def test_local_vlm_adapter_plan_lists_reviewed_repos() -> None:
    plan = local_vlm_adapter_plan()

    assert plan["ok"] is True
    names = [repo["name"] for repo in plan["repos"]]
    assert any("Qwen" in name for name in names)
    assert any("InternVL" in name for name in names)
    assert any("LLaVA" in name for name in names)
    assert all(repo["adapter_contract"]["evidence_required"] for repo in plan["repos"])


def test_review_html_surfaces_controlled_vision_execution_entrypoints() -> None:
    package = {
        "title": "controlled vision",
        "coverage": {"timeline_items": 2},
        "sources": [],
        "review_artifacts": {
            "media_path": "C:/videos/lesson.mp4",
            "review_notes_template": "review-notes.template.json",
            "review_notes": "review-notes.json",
            "review_fill_guide": "review-fill-guide.md",
            "task_console": "task-console.html",
            "review_pack": "review-pack.md",
            "review_closure_status": "review-closure-status.md",
            "mcp_apply_review_notes_args": "mcp-apply-review-notes.args.json",
        },
        "timeline": [
            {
                "visual_route": "semantic_frame",
                "needs_human_review": True,
                "quality_issues": ["missing_visual_text", "semantic_frame_without_analysis"],
                "assets": [{"path": "frame.jpg"}],
                "frame_paths": ["frame.jpg"],
            },
            {
                "visual_route": "temporal_sequence",
                "quality_issues": ["temporal_sequence_without_analysis"],
                "assets": [{"path": "frame.jpg"}],
                "temporal_frame_paths": ["frame-a.jpg", "frame-b.jpg"],
            },
        ],
    }

    html = render_lecture_review_html(package)

    assert "视觉执行 preflight" in html
    assert "可控执行状态" in html
    assert "本地执行演练" in html
    assert "mcp-vision-execution-preflight.args.json" in html
    assert "mcp-controlled-execution-check.args.json" in html
    assert "mcp-controlled-execution-smoke.args.json" in html
    assert "mcp-acceptance-check.args.json" in html
    assert "acceptance-check.md" in html
    assert "review-session.md" in html
    assert "review-notes.template.json" in html
    assert "review-notes.json" in html
    assert "review-fill-guide.md" in html
    assert "task-console.html" in html
    assert "打开任务控制台" in html
    assert "任务控制台" in html
    assert "review-pack.md" in html
    assert "review-closure-status.md" in html
    assert "复核进度" in html
    assert "mcp-apply-review-notes.args.json" in html
    assert "exports/full-transcript.md" in html
    assert "exports/extraction-audit.md" in html
    assert "mcp-run-multimodal-frame-analysis-confirmed.args.json" in html
    assert "mcp-run-temporal-visual-analysis-confirmed.args.json" in html
    assert "mcp-call vision_execution_preflight" in html
    assert "mcp-call controlled_execution_check" in html
    assert "mcp-call controlled_execution_smoke" in html
    assert 'data-review-filter="needs-human-review"' in html
    assert 'data-review-filter="missing-visual-text"' in html
    assert 'data-review-filter="low-confidence"' in html
    assert 'data-review-filter="keep-image"' in html
    assert 'data-review-filter="corrected"' in html
    assert 'data-visual-route="semantic_frame"' in html
    assert 'data-review-status="pending"' in html
    assert "质量缺口" in html
    assert "关键帧" in html
    assert "corrected_visual_text" in html
    assert "corrected_visual_understanding" in html
    assert "corrected_temporal_visual_understanding" in html
    assert 'data-quick-status="accepted_known_gap"' in html
    assert 'data-quick-status="keep_image"' in html
    assert 'data-quick-status="needs_rerun_ocr"' in html
    assert 'data-quick-status="corrected_visual_text"' in html
    assert "setQuickReviewStatus" in html
    assert "review-json-draft" in html
    assert "copyReviewSnippet" in html
    assert "lecture_review_notes.v1" in html
    assert "视频审核" in html
    assert "review-video-player" in html
    assert "review-video-picker" in html
    assert "review-video-height-slider" in html
    assert "review-sidebar-width-slider" in html
    assert "setReviewVideoHeight" in html
    assert "setReviewSidebarWidth" in html
    assert "toggleReviewVideoWide" in html
    assert "--review-sidebar-width" in html
    assert "vkpReviewVideoHeight" in html
    assert "jumpToTimelineItem" in html
    assert "jumpToFirstVisibleReviewItem" in html
    assert "C:/videos/lesson.mp4" in html
    assert "选择本地视频文件" in html
    assert "jump-time" in html


def test_review_html_does_not_bind_long_asr_segment_to_short_visual_window() -> None:
    package = {
        "title": "long cue",
        "coverage": {},
        "sources": [],
        "review_artifacts": {},
        "review_transcript_segments": [
            {"start": 0.5, "end": 10.5, "text": "this speech continues beyond the short visual window"}
        ],
        "timeline": [
            {"index": 1, "start": 0, "end": 4, "transcript": "this speech continues beyond the short visual window"}
        ],
    }

    html = render_lecture_review_html(package)

    assert "jumpToTimelineItem(1, 0)" in html
    assert "time:visual_segment_start" in html
    assert "未找到可靠对齐的 ASR" in html
    assert "jumpToTimelineItem(1, 0.5)" not in html


def test_refresh_review_html_adds_acceptance_review_and_export_links_without_rebuild(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "existing bundle",
                "sources": [],
                "coverage": {"timeline_items": 1},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    original_timeline = [
        {
            "index": 1,
            "start": 1,
            "end": 3,
            "transcript": "保留当前时间线",
            "visual_route": "semantic_frame",
        }
    ]
    (bundle / "timeline.json").write_text(json.dumps(original_timeline, ensure_ascii=False), encoding="utf-8")
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0.75, "end": 1.8, "text": "保留当前时间线"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "review.html").write_text("<html>old</html>", encoding="utf-8")

    result = refresh_bundle_review_html(bundle)
    html = (bundle / "review.html").read_text(encoding="utf-8")
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    assert result["timeline_items"] == 1
    assert timeline == original_timeline
    assert manifest["review_html"] == "review.html"
    assert manifest["review_notes"] == "review-notes.json"
    assert manifest["review_notes_template"] == "review-notes.template.json"
    assert manifest["review_fill_guide"] == "review-fill-guide.md"
    assert manifest["mcp_apply_review_notes_args"] == "mcp-apply-review-notes.args.json"
    assert (bundle / "review-notes.template.json").exists()
    assert (bundle / "review-notes.json").exists()
    assert (bundle / "review-fill-guide.md").exists()
    assert (bundle / "mcp-apply-review-notes.args.json").exists()
    assert "acceptance-check.md" in html
    assert "review-session.md" in html
    assert "review-notes.template.json" in html
    assert "review-notes.json" in html
    assert "review-fill-guide.md" in html
    assert "mcp-apply-review-notes.args.json" in html
    assert "task-console.html" in html
    assert "exports/knowledge-note.md" in html
    assert "exports/full-transcript.md" in html
    assert "exports/extraction-audit.md" in html
    assert "jumpToTimelineItem(1, 0.75)" in html
    assert "time:asr_segment_start" in html


def test_refresh_review_html_preserves_existing_review_template(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    existing_template = {
        "schema": "lecture_review_notes.v1",
        "reviews": [{"timeline_index": 9, "status": "accepted", "comment": "open-only review pack row"}],
    }
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "review_notes_template": "review-notes.template.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps([{"index": 1, "start": 0, "end": 1}, {"index": 2, "start": 1, "end": 2}], ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "review.html").write_text("<html>old</html>", encoding="utf-8")
    (bundle / "review-notes.template.json").write_text(json.dumps(existing_template, ensure_ascii=False), encoding="utf-8")

    refresh_bundle_review_html(bundle)

    assert json.loads((bundle / "review-notes.template.json").read_text(encoding="utf-8")) == existing_template


def test_acceptance_run_writes_report_and_keeps_default_preview_safe(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    bundle = tmp_path / "run" / "webui-bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")

    import video_knowledge_pipeline.acceptance_run as acceptance_run

    calls: dict[str, object] = {}

    monkeypatch.setattr(
        acceptance_run,
        "prepare_local_video_run",
        lambda *args, **kwargs: {
            "schema": "video_knowledge_local_video_run.v1",
            "markdown_path": str(tmp_path / "run" / "video-knowledge-run.md"),
            "initial_bundle": {"status": "ok", "bundle_dir": str(bundle), "review_html": str(bundle / "review.html")},
        },
    )
    monkeypatch.setattr(acceptance_run, "run_video_frame_router", lambda bundle_dir: {"report_path": str(bundle / "router.md")})
    monkeypatch.setattr(acceptance_run, "run_visual_structure_plan", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "visual-structure-report.md")})

    def fake_multimodal(bundle_dir, **kwargs):
        calls["execute_vision"] = kwargs["execute"]
        return {"report_path": str(bundle / "multimodal-frame-analysis-report.md")}

    monkeypatch.setattr(acceptance_run, "run_multimodal_frame_analysis", fake_multimodal)
    monkeypatch.setattr(
        acceptance_run,
        "run_temporal_frame_groups",
        lambda bundle_dir, **kwargs: {"report_path": str(bundle / "temporal-frame-groups-report.md"), "summary": {"execute": kwargs["execute"]}},
    )
    monkeypatch.setattr(acceptance_run, "run_temporal_visual_analysis", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "temporal-visual-analysis-report.md")})
    monkeypatch.setattr(acceptance_run, "vision_acceptance_plan", lambda bundle_dir, **kwargs: {"report_path": str(bundle / "vision-acceptance-plan.md")})
    monkeypatch.setattr(acceptance_run, "audit_knowledge_coverage", lambda bundle_dir: {"coverage_markdown_path": str(bundle / "knowledge-coverage.md")})
    monkeypatch.setattr(acceptance_run, "export_knowledge_note", lambda bundle_dir, **kwargs: {"note_path": str(bundle / "exports" / "knowledge-note.md")})
    monkeypatch.setattr(acceptance_run, "bundle_status_report", lambda bundle_dir: {"report_path": str(bundle / "bundle-status.md")})

    result = run_acceptance_run(media, tmp_path / "run", title="课程测试")

    assert result["schema"] == "video_knowledge_acceptance_run.v1"
    assert result["summary"]["workflow_status"] == "ok"
    assert result["summary"]["status"] == "ready"
    assert calls["execute_vision"] is False
    assert Path(result["json_path"]).exists()
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "Video Knowledge Acceptance Run" in report
    assert "图文截图解析" in report
    assert "execute_vision" in report


def test_bundle_next_action_routes_ocr_empty_to_review_instead_of_rerunning_ocr(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "016_0000240000ms.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "mcp_review_session_args": "mcp-prepare-review-session.args.json",
                "mcp_multimodal_frame_analysis_args": "mcp-run-multimodal-frame-analysis.args.json",
                "visual_structure": {
                    "last_run": {
                        "ebook_pipeline_total": 1,
                        "ebook_pipeline_succeeded": 0,
                        "ebook_pipeline_blockers": {"ocr_text_empty": 1},
                    },
                    "ebook_pipeline_results": [{"index": 16, "ok": False, "blocker": "ocr_text_empty"}],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 16,
                    "start": 24,
                    "end": 26,
                    "transcript": "老师展示一个表格。",
                    "visual_route": "document_visual",
                    "material_types": ["table"],
                    "visual_text": f"# {frame.stem}\n\n<!-- source: {frame} -->",
                    "structured_visual": [{"markdown": f"# {frame.stem}\n\n<!-- source: {frame} -->", "image_path": str(frame)}],
                    "frame_paths": [str(frame)],
                    "assets": [{"path": str(frame), "copied": "true"}],
                    "quality_issues": ["needs_human_review"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = bundle_next_action(bundle)
    action = result["next_action"]

    assert result["status"] == "human_review_required"
    assert action["key"] == "ocr_text_empty_review"
    assert action["mcp_tool"] == "prepare_review_session"
    assert action["human_required"] is True
    assert action["fallback_mcp_tool"] == "run_multimodal_frame_analysis"


def test_prepare_review_session_surfaces_ocr_empty_targets_without_wrapper_excerpt(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "016_0000240000ms.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "ocr empty review",
                "review_notes": "review-notes.json",
                "mcp_review_session_args": "mcp-prepare-review-session.args.json",
                "mcp_multimodal_frame_analysis_args": "mcp-run-multimodal-frame-analysis.args.json",
                "visual_structure": {
                    "last_run": {
                        "ebook_pipeline_total": 1,
                        "ebook_pipeline_succeeded": 0,
                        "ebook_pipeline_blockers": {"ocr_text_empty": 1},
                    },
                    "ebook_pipeline_results": [
                        {
                            "index": 16,
                            "ok": False,
                            "blocker": "ocr_text_empty",
                            "frame_path": str(frame),
                            "artifact_path": str(bundle / "ebook-output" / "016.md"),
                        }
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 16,
                    "start": 24.0,
                    "end": 29.0,
                    "transcript": "这里讲一人公司的时间配置。",
                    "visual_route": "document_visual",
                    "material_types": ["slide"],
                    "frame_paths": [str(frame)],
                    "visual_text": "# 016_0000240000ms\n<!-- source: D:/tmp/016_0000240000ms.jpg -->\n",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    session = prepare_review_session(bundle, refresh=False)
    targets = session["review_targets"]["items"]
    target = targets[0]
    markdown = (bundle / "review-session.md").read_text(encoding="utf-8")

    assert target["index"] == 16
    assert "ocr_text_empty" in target["reasons"]
    assert target["suggested_filter"] == "ocr_empty"
    assert target["ebook_blocker"] == "ocr_text_empty"
    assert target["ebook_result"]["frame_path"] == str(frame)
    assert "ebook pipeline returned no meaningful text" in target["fallback_suggestion"]
    assert "这里讲一人公司的时间配置" in target["evidence_excerpt"]
    assert "016_0000240000ms" not in target["evidence_excerpt"]
    assert "### OCR Empty Targets" in markdown
    assert "ebook pipeline returned no meaningful text" in markdown


def test_prepare_review_session_writes_fillable_template_for_visual_gaps(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-review-template"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    temporal = assets / "temporal.jpg"
    temporal.write_bytes(b"fake image")
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "review template",
                "review_notes": "review-notes.json",
                "mcp_review_session_args": "mcp-prepare-review-session.args.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 2,
                    "transcript": "讲解软件界面",
                    "visual_route": "semantic_frame",
                    "assets": [{"path": "assets/frame.jpg"}],
                    "quality_issues": ["semantic_frame_without_analysis"],
                },
                {
                    "index": 2,
                    "start": 2,
                    "end": 4,
                    "transcript": "演示操作变化",
                    "visual_route": "temporal_sequence",
                    "assets": [{"path": "assets/temporal.jpg"}],
                    "temporal_frame_paths": ["assets/frame.jpg", "assets/temporal.jpg"],
                    "quality_issues": ["temporal_sequence_without_analysis"],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    session = prepare_review_session(bundle, refresh=True)
    template_path = Path(session["review_notes_template_path"])
    fill_guide_path = Path(session["review_fill_guide_path"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    fill_guide = fill_guide_path.read_text(encoding="utf-8")
    rows = template["reviews"]

    assert template_path.exists()
    assert fill_guide_path.exists()
    assert rows[0]["timeline_index"] == 2
    assert rows[0]["suggested_status"] == "corrected_temporal_visual_understanding"
    assert rows[0]["evidence_frame_paths"]
    assert rows[1]["timeline_index"] == 1
    assert rows[1]["suggested_status"] == "corrected_visual_understanding"
    assert "讲解软件界面" in rows[1]["transcript_excerpt"]
    assert "Review Notes 填写指南" in fill_guide
    assert "连续片段待补" in fill_guide
    assert "单帧视觉待补" in fill_guide
    assert "corrected_visual_understanding" in fill_guide
    assert "corrected_temporal_visual_understanding" in fill_guide
    assert "![evidence](" in fill_guide
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["review_session"] == "review-session.md"
    assert manifest["review_session_json"] == "review-session.json"
    assert manifest["review_notes_template"] == "review-notes.template.json"
    assert manifest["review_fill_guide"] == "review-fill-guide.md"
    assert manifest["mcp_prepare_review_session_args"] == "mcp-prepare-review-session.args.json"

    rows[0]["status"] = "corrected_temporal_visual_understanding"
    rows[0]["corrected_temporal_visual_understanding"] = {
        "event_sequence": ["打开设置面板"],
        "state_changes": ["界面从主页切换到设置"],
        "operation_steps": ["点击设置"],
        "evidence_frame_paths": rows[0]["evidence_frame_paths"],
    }
    rows[1]["status"] = "corrected_visual_understanding"
    rows[1]["corrected_visual_understanding"] = {
        "objects": ["软件界面"],
        "actions": ["展示控制台"],
        "interface_state": "控制台打开",
        "evidence_frame_paths": rows[1]["evidence_frame_paths"],
    }
    review_json = bundle / "review-notes.json"
    review_json.write_text(json.dumps(template, ensure_ascii=False), encoding="utf-8")

    result = apply_review_notes_to_bundle(bundle)
    coverage = json.loads((bundle / "knowledge-coverage.json").read_text(encoding="utf-8"))

    assert result["updated_indexes"] == [2, 1]
    assert coverage["semantic_frame_without_analysis"] == 0
    assert coverage["temporal_sequence_without_analysis"] == 0
    refreshed_session = prepare_review_session(bundle, refresh=True)
    assert refreshed_session["review_targets"]["total_open"] == 0
    assert json.loads((bundle / "review-notes.template.json").read_text(encoding="utf-8"))["reviews"] == []


def test_prepare_review_session_supports_limit_offset_and_reason_filter(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-review-slices"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    timeline = []
    for index in range(1, 6):
        route = "temporal_sequence" if index == 2 else "semantic_frame"
        issue = "temporal_sequence_without_analysis" if route == "temporal_sequence" else "semantic_frame_without_analysis"
        timeline.append(
            {
                "index": index,
                "start": index,
                "end": index + 1,
                "transcript": f"片段 {index}",
                "visual_route": route,
                "assets": [{"path": "assets/frame.jpg"}],
                "quality_issues": [issue],
            }
        )
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    semantic_page = prepare_review_session(bundle, refresh=True, limit=2, offset=1, reason="semantic_frame_without_analysis")
    targets = semantic_page["review_targets"]
    rows = json.loads((bundle / "review-notes.template.json").read_text(encoding="utf-8"))["reviews"]

    assert targets["total_open"] == 5
    assert targets["filtered_open"] == 4
    assert targets["listed_count"] == 2
    assert targets["offset"] == 1
    assert targets["limit"] == 2
    assert targets["reason_filter"] == "semantic_frame_without_analysis"
    assert [row["timeline_index"] for row in rows] == [3, 4]
    guide = (bundle / "review-fill-guide.md").read_text(encoding="utf-8")
    assert "Index 3" in guide
    assert "Index 4" in guide
    assert "Index 1" not in guide

    all_targets = prepare_review_session(bundle, refresh=True, limit=0)
    assert all_targets["review_targets"]["listed_count"] == 5
    all_rows = json.loads((bundle / "review-notes.template.json").read_text(encoding="utf-8"))["reviews"]
    assert len(all_rows) == 5
    args = json.loads((bundle / "mcp-prepare-review-session.args.json").read_text(encoding="utf-8"))
    assert args["limit"] == 0
    assert args["offset"] == 0




def test_tile_review_notes_accept_tile_corrections_as_visual_text(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-tile-review-corrections"
    tiles = bundle / "high-res-tiles"
    tiles.mkdir(parents=True)
    tile_path = tiles / "tile-0001.jpg"
    tile_path.write_bytes(b"fake tile")
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "review_notes": "review-notes.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 4,
                    "transcript": "这里需要核对屏幕小字。",
                    "visual_route": "document_visual",
                    "quality_issues": ["tile_result_needs_review", "missing_visual_text"],
                    "tile_review_targets": [
                        {
                            "tile_id": "tile-0001",
                            "confidence": 0.41,
                            "reasons": ["tile_result_low_confidence"],
                            "evidence_path": str(tile_path),
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    session = prepare_review_session(bundle, refresh=False, limit=0, group_by="reason")
    todo = json.loads((bundle / "review-notes.todo.json").read_text(encoding="utf-8"))
    item = session["review_targets"]["items"][0]

    assert item["suggested_status"] == "corrected_visual_text"
    assert todo["reviews"][0]["tile_corrections"][0]["tile_id"] == "tile-0001"
    assert todo["reviews"][0]["tile_corrections"][0]["status"] == "needs_review"
    assert todo["reviews"][0]["tile_corrections"][0]["evidence_path"] == str(tile_path)

    (bundle / "review-notes.json").write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "timeline_index": 1,
                        "status": "corrected_visual_text",
                        "tile_corrections": [
                            {
                                "tile_id": "tile-0001",
                                "status": "corrected",
                                "corrected_text": "屏幕小字：客户特点、成交原则。",
                                "comment": "人工核对 tile 后修正。",
                                "evidence_path": str(tile_path),
                            }
                        ],
                        "evidence_frame_paths": [str(tile_path)],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = apply_review_notes_to_bundle(bundle)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))

    assert result["updated_indexes"] == [1]
    assert timeline[0]["review_status"] == "corrected_visual_text"
    assert timeline[0]["needs_human_review"] is False
    assert "missing_visual_text" not in timeline[0]["quality_issues"]
    assert timeline[0]["human_corrected_visual_text"] == "[tile-0001] 屏幕小字：客户特点、成交原则。"
    assert timeline[0]["human_tile_corrections"][0]["corrected_text"] == "屏幕小字：客户特点、成交原则。"

def test_timeline_alignment_review_notes_can_correct_review_start(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-timeline-alignment-review"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "review.html").write_text("<html></html>", encoding="utf-8")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "review_notes": "review-notes.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 6.0,
                    "review_start": 4.0,
                    "review_start_source": "frame_time",
                    "transcript": "这里开始讲第一个关键动作。",
                    "visual_route": "semantic_frame",
                    "assets": [{"path": str(frame)}],
                    "quality_issues": [],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline-alignment-audit.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.timeline_alignment_audit.v1",
                "summary": {"items_with_issues": 1, "review_start_mismatch": 1},
                "items": [
                    {
                        "index": 1,
                        "start": 0.0,
                        "end": 6.0,
                        "issues": ["review_start_mismatch"],
                        "review_start": 4.0,
                        "review_start_source": "frame_time",
                        "asr_first_start": 1.25,
                        "frame_time": 4.0,
                        "asr_excerpt": "这里开始讲第一个关键动作。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    session = prepare_review_session(bundle, refresh=False, limit=0, group_by="reason")
    todo = json.loads((bundle / "review-notes.todo.json").read_text(encoding="utf-8"))
    pack_md = (bundle / "review-pack.md").read_text(encoding="utf-8")

    assert session["review_targets"]["total_open"] == 1
    assert session["review_targets"]["items"][0]["suggested_status"] == "corrected_review_start"
    assert todo["reviews"][0]["suggested_status"] == "corrected_review_start"
    assert todo["reviews"][0]["timeline_alignment"]["suggested_review_start"] == 1.25
    assert todo["reviews"][0]["corrected_review_start"] == ""
    assert "时间轴错位" in pack_md
    assert "suggested=1.25" in pack_md

    (bundle / "review-notes.json").write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "timeline_index": 1,
                        "status": "corrected_review_start",
                        "corrected_review_start": 1.25,
                        "comment": "人工核对视频后采用 ASR 起点。",
                        "evidence_frame_paths": [str(frame)],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = apply_review_notes_to_bundle(bundle)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))

    assert result["updated_indexes"] == [1]
    assert timeline[0]["review_status"] == "corrected_review_start"
    assert timeline[0]["needs_human_review"] is False
    assert timeline[0]["review_start"] == 1.25
    assert timeline[0]["review_start_source"] == "human_review_note"
    assert timeline[0]["human_corrected_review_start"] == 1.25

def test_acceptance_check_distinguishes_review_template_from_imported_notes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-acceptance-review-lifecycle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "review lifecycle",
                "review_notes_template": "review-notes.template.json",
                "review_session_json": "review-session.json",
                "vision_execution_preflight_json": "vision-execution-preflight.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 1,
                    "transcript": "讲解一个需要看画面的界面状态",
                    "visual_route": "semantic_frame",
                    "assets": [{"path": "assets/frame.jpg"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "vision-execution-preflight.json").write_text(
        json.dumps(
            {
                "ready_to_execute": False,
                "blockers": [{"key": "provider_health_failed"}],
                "provider_health": {"status": "provider_unreachable", "safe_to_execute": False},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "review-session.json").write_text(
        json.dumps(
            {
                "schema": "lecture_review_session.v1",
                "review_targets": {"total_open": 1, "listed_count": 1},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "review-notes.template.json").write_text(
        json.dumps(
            {
                "schema": "lecture_review_notes.v1",
                "reviews": [
                    {
                        "timeline_index": 1,
                        "suggested_status": "corrected_visual_understanding",
                        "evidence_frame_paths": ["assets/frame.jpg"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = acceptance_check(bundle, refresh=True)
    markdown = (bundle / "acceptance-check.md").read_text(encoding="utf-8")

    assert report["status"] == "provider_blocked"
    assert report["summary"]["review_state"] == "human_review_ready"
    assert report["summary"]["review_template_prepared"] is True
    assert report["summary"]["review_notes_imported"] is False
    assert report["review_lifecycle"]["review_targets_open"] == 1
    assert report["next_action"]["key"] == "provider_repair_or_apply_review_notes"
    assert report["next_action"]["mcp_tool"] == "vision_provider_matrix"
    assert report["next_action"]["fallback_mcp_tool"] == "apply_review_notes"
    assert "Review Lifecycle" in markdown
    assert "human_review_ready" in markdown
