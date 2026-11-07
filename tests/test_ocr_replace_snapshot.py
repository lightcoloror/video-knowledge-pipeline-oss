from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.ocr_backfill import run_ocr_backfill


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    rows = []
    for index, text in ((1, "stale first"), (2, "stale second")):
        frame = assets / f"frame-{index}.jpg"
        frame.write_bytes(f"synthetic-{index}".encode())
        rows.append(
            {
                "index": index,
                "start": index - 1,
                "end": index,
                "frame_paths": [str(frame)],
                "visual_text": text,
                "quality_issues": [],
            }
        )
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(rows), encoding="utf-8")
    return bundle


def _input(path: Path, items: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"schema": "lecture_ocr_backfill_input.v1", "items": items}), encoding="utf-8")
    return path


def test_replace_snapshot_updates_and_clears_authoritative_observations(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    input_json = _input(
        tmp_path / "snapshot.json",
        [
            {"index": 1, "text": "fresh first", "source": "reviewed_snapshot"},
            {"index": 2, "text": "", "source": "reviewed_snapshot"},
        ],
    )

    result = run_ocr_backfill(bundle, input_json=input_json, apply_mode="replace_snapshot")

    assert result["backfill"]["applied"] is True
    assert result["backfill"]["updated_indexes"] == [1]
    assert result["backfill"]["cleared_indexes"] == [2]
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[0]["visual_text"] == "fresh first"
    assert "visual_text" not in timeline[1]
    assert timeline[1]["original_visual_text"] == "stale second"
    receipt_path = Path(result["overwrite_receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "video_knowledge_pipeline.ocr_overwrite_receipt.v1"
    assert receipt["status"] == "applied"
    assert receipt["coverage"]["complete"] is True
    assert receipt["cleared_indexes"] == [2]
    assert receipt["rollback"]["source"] == "changes[].before_text"
    cleared_change = next(row for row in receipt["changes"] if row["index"] == 2)
    assert cleared_change["before_text"] == "stale second"
    assert cleared_change["after_text"] == ""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ocr_backfill"]["overwrite_receipt_path"] == str(receipt_path)
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "Apply mode: `replace_snapshot`" in report
    assert str(receipt_path) in report


def test_replace_snapshot_rejects_incomplete_coverage_without_mutation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    input_json = _input(
        tmp_path / "incomplete.json",
        [{"index": 1, "text": "fresh first", "source": "reviewed_snapshot"}],
    )

    result = run_ocr_backfill(bundle, input_json=input_json, apply_mode="replace_snapshot")

    assert result["backfill"]["applied"] is False
    assert result["backfill"]["coverage_complete"] is False
    assert result["status"] == "replace_snapshot_incomplete"
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert [row["visual_text"] for row in timeline] == ["stale first", "stale second"]
    receipt = json.loads(Path(result["overwrite_receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["status"] == "rejected_incomplete_snapshot"
    assert receipt["coverage"]["missing_indexes"] == [2]


def test_replace_snapshot_preserves_old_text_for_non_authoritative_failure(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    input_json = _input(
        tmp_path / "failed.json",
        [
            {"index": 1, "text": "fresh first", "source": "reviewed_snapshot"},
            {"index": 2, "source": "ocr-crops/frame-2.jpg", "notes": "screen_text_recovery crop OCR"},
        ],
    )

    result = run_ocr_backfill(bundle, input_json=input_json, apply_mode="replace_snapshot")

    assert result["backfill"]["applied"] is False
    assert result["backfill"]["coverage_complete"] is False
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[1]["visual_text"] == "stale second"
    assert 2 in result["backfill"]["preserved_indexes"]


def test_invalid_apply_mode_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    try:
        run_ocr_backfill(bundle, apply_mode="truncate")
    except ValueError as exc:
        assert "apply_mode" in str(exc)
    else:
        raise AssertionError("invalid apply_mode must fail")


def test_cli_accepts_replace_snapshot_mode() -> None:
    args = build_parser().parse_args(["run-ocr-backfill", "synthetic-bundle", "--apply-mode", "replace_snapshot"])

    assert args.apply_mode == "replace_snapshot"
