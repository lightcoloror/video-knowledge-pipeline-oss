from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.batch_repair as batch_repair
from video_knowledge_pipeline.batch_repair import batch_repair_run
from video_knowledge_pipeline.cli import build_parser


def _write_summary(path: Path, bundle: Path, *, next_action_key: str, screen_text_status: str = "ok") -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_batch_acceptance_summary.v1",
                "workspace": str(path.parent),
                "items": [
                    {
                        "id": "lesson-001",
                        "title": "Lesson",
                        "bundle_dir": str(bundle),
                        "acceptance_status": "needs_review",
                        "screen_text_status": screen_text_status,
                        "next_action_key": next_action_key,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_batch_repair_preview_does_not_call_machine_tools(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    summary = tmp_path / "batch-acceptance-summary.json"
    _write_summary(summary, bundle, next_action_key="semantic_frame_understanding")

    monkeypatch.setattr(
        batch_repair,
        "_next",
        lambda *_args, **_kwargs: {
            "status": "needs_review",
            "next_action": {"key": "semantic_frame_understanding", "mcp_tool": "run_multimodal_frame_analysis"},
        },
    )
    monkeypatch.setattr(batch_repair, "run_screen_text_recovery", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OCR called")))
    monkeypatch.setattr(batch_repair, "bundle_advance_queue", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("vision called")))
    monkeypatch.setattr(batch_repair, "plan_asr_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ASR called")))

    result = batch_repair_run(summary, execute=False, allow_vision=False)

    assert result["summary"]["total"] == 1
    assert result["items"][0]["status"] == "blocked_not_allowed"
    assert Path(result["report_path"]).exists()
    assert Path(result["human_review_path"]).exists()


def test_batch_repair_skips_accepted_none(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    summary = tmp_path / "batch-acceptance-summary.json"
    _write_summary(summary, bundle, next_action_key="none")

    monkeypatch.setattr(batch_repair, "_next", lambda *_args, **_kwargs: {"status": "accepted_with_known_gaps", "next_action": {"key": "none"}})
    monkeypatch.setattr(batch_repair, "_refresh_bundle", lambda *_args, **_kwargs: {})

    result = batch_repair_run(summary)

    assert result["items"][0]["status"] == "skipped_completed"


def test_batch_repair_screen_text_execute_generates_human_review_when_empty(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    summary = tmp_path / "batch-acceptance-summary.json"
    _write_summary(summary, bundle, next_action_key="screen_text", screen_text_status="weak")

    monkeypatch.setattr(batch_repair, "_next", lambda *_args, **_kwargs: {"status": "accepted_with_known_gaps", "next_action": {"key": "none"}})
    monkeypatch.setattr(batch_repair, "_refresh_bundle", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        batch_repair,
        "run_screen_text_recovery",
        lambda *_args, **_kwargs: {"schema": "screen_text_recovery.v1", "ocr_summary": {"updated": 0}, "report_path": str(bundle / "screen-text-recovery.md")},
    )

    result = batch_repair_run(summary, execute=True, allow_ocr=True)

    assert result["items"][0]["status"] == "human_review_required"
    human_review = Path(result["human_review_path"]).read_text(encoding="utf-8")
    assert "screen_text" in human_review


def test_batch_repair_human_review_uses_review_pack_item_rows(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    summary = tmp_path / "batch-acceptance-summary.json"
    _write_summary(summary, bundle, next_action_key="human_review")
    (bundle / "review-pack.json").write_text(
        json.dumps(
            {
                "schema": "lecture_review_pack.v1",
                "groups": [
                    {
                        "key": "missing_visual_text",
                        "label": "缺屏幕文字",
                        "items": [
                            {
                                "index": 7,
                                "reasons": ["missing_visual_text"],
                                "suggested_action": "人工补齐或修正画面文字/OCR。",
                                "evidence_paths": ["ocr-crops/frame-7.jpg"],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        batch_repair,
        "_next",
        lambda *_args, **_kwargs: {
            "status": "needs_review",
            "next_action": {"key": "human_review", "label": "人工复核", "human_required": True, "reason": "screen text failed"},
        },
    )

    result = batch_repair_run(summary, execute=False)
    rows = result["human_review"]["items"]
    markdown = Path(result["human_review_path"]).read_text(encoding="utf-8")

    assert rows[0]["scope"] == "timeline:7"
    assert rows[0]["reason"] == "missing_visual_text"
    assert rows[0]["review_pack"].endswith("review-pack.json")
    assert "timeline:7" in markdown
    assert "ocr-crops/frame-7.jpg" in markdown


def test_batch_repair_cli_contract() -> None:
    args = build_parser().parse_args(["batch-repair-run", "batch-acceptance-summary.json", "--allow-ocr", "--execute", "--limit", "1"])

    assert args.command == "batch-repair-run"
    assert args.allow_ocr is True
    assert args.execute is True
    assert args.limit == 1
