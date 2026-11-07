from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import video_knowledge_pipeline.visual_structure as visual_structure


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(path)


def _write_bundle(bundle: Path) -> Path:
    frame = bundle / "assets" / "frame.jpg"
    _write_image(frame)
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 2,
                    "visual_route": "document_visual",
                    "material_types": ["slide"],
                    "frame_paths": [str(frame)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return frame


def test_recovers_atomic_ebook_checkpoint_before_new_work(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-checkpoint-recovery"
    frame = _write_bundle(bundle)
    output_dir = (
        bundle / "visual-structure" / "timeline-0001" / "ebook_pipeline"
    )
    output_dir.mkdir(parents=True)
    visual_structure._write_ebook_result_checkpoint(
        output_dir,
        {
            "index": 1,
            "image_path": str(frame),
            "output_dir": str(output_dir),
            "ok": True,
            "artifact": {
                "path": str(output_dir / "output.md"),
                "artifact_type": "markdown",
                "text": "# 课程标题\n\n- 第一项要点\n- 第二项要点",
            },
            "ebook_quality": {
                "quality": "usable",
                "reason": "meaningful_text",
                "text_char_count": 20,
                "line_count": 3,
            },
            "meaningful_text_char_count": 20,
            "meaningful_line_count": 3,
        },
    )

    first = visual_structure.reconcile_ebook_pipeline_checkpoints(bundle)
    timeline_after_first = (bundle / "timeline.json").read_bytes()
    manifest_after_first = (bundle / "manifest.json").read_bytes()
    second = visual_structure.reconcile_ebook_pipeline_checkpoints(bundle)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))

    assert first["recovered_indexes"] == [1]
    assert second["recovered_indexes"] == []
    assert timeline[0]["ebook_pipeline_status"]["ok"] is True
    assert timeline[0]["visual_text"].startswith("# 课程标题")
    assert len(timeline[0]["structured_visual"]) == 1
    assert (bundle / "timeline.json").read_bytes() == timeline_after_first
    assert (bundle / "manifest.json").read_bytes() == manifest_after_first


def test_recovers_legacy_markdown_without_checkpoint(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-legacy-ocr-recovery"
    _write_bundle(bundle)
    output_dir = (
        bundle / "visual-structure" / "timeline-0001" / "ebook_pipeline"
    )
    output_dir.mkdir(parents=True)
    (output_dir / "output.md").write_text(
        "# 已完成课件页\n\n这里是中断前已经完成、但尚未注册的 OCR 正文。",
        encoding="utf-8",
    )

    result = visual_structure.reconcile_ebook_pipeline_checkpoints(bundle)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))

    assert result["recovered_indexes"] == [1]
    assert timeline[0]["ebook_pipeline_status"]["ok"] is True
    assert "已完成课件页" in timeline[0]["visual_text"]


def test_rejects_corrupt_or_out_of_range_checkpoint(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-invalid-checkpoint"
    _write_bundle(bundle)
    output_dir = (
        bundle / "visual-structure" / "timeline-0002" / "ebook_pipeline"
    )
    output_dir.mkdir(parents=True)
    (output_dir / "result-checkpoint.json").write_text(
        json.dumps(
            {
                "schema": visual_structure.EBOOK_CHECKPOINT_SCHEMA,
                "result": {"index": 2, "ok": True},
            }
        ),
        encoding="utf-8",
    )

    result = visual_structure.reconcile_ebook_pipeline_checkpoints(bundle)

    assert result["status"] == "degraded"
    assert result["invalid_count"] == 1
    assert result["recovered_count"] == 0
