from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.transcript_semantic_correction import (
    build_transcript_semantic_correction_pack,
)


def _write_bundle(
    tmp_path: Path,
    *,
    segments: list[dict],
    timeline: list[dict],
) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "semantic visual alignment fixture",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(root / "normalized-transcript.json", {"segments": segments})
    write_json(root / "timeline.json", timeline)
    return root


def _number_candidate(pack: dict) -> dict:
    return next(
        row
        for row in pack["candidates"]
        if row.get("correction_type") == "number"
    )


def _visual_evidence(candidate: dict) -> list[dict]:
    return [
        row
        for row in candidate.get("evidence") or []
        if row.get("source_type") == "visual_understanding"
    ]


def test_confirmed_rich_visual_schema_is_candidate_evidence_and_uses_time_not_index(
    tmp_path: Path,
) -> None:
    root = _write_bundle(
        tmp_path,
        segments=[{"start": 10.0, "end": 12.0, "text": "这里有1个方案"}],
        timeline=[
            {
                # This index aliases the zero-based ASR cue index, but its time is
                # unrelated. It must never be selected merely because both are 0.
                "index": 0,
                "start": 100.0,
                "end": 102.0,
                "frame_path": "frames/wrong.jpg",
                "visual_understanding": {
                    "evidence_level": "confirmed",
                    "objects": ["错误时间线对象"],
                },
            },
            {
                "index": 27,
                "start": 10.0,
                "end": 12.0,
                "frame_path": "frames/right.jpg",
                "visual_understanding": {
                    "evidence_level": "confirmed",
                    "objects": ["中意人寿产品说明"],
                    "actions": ["讲师指向方案一"],
                    "interface_state": "方案一被高亮",
                    "spatial_relations": ["标题位于页面顶部"],
                    "instructor_focus": "件均保费",
                },
            },
        ],
    )

    candidate = _number_candidate(
        build_transcript_semantic_correction_pack(root, write=False)
    )
    evidence = _visual_evidence(candidate)

    assert len(evidence) == 1
    row = evidence[0]
    assert row["timeline_index"] == 27
    assert row["evidence_level"] == "confirmed"
    assert row["timeline_alignment"]["basis"] == "time_overlap"
    assert row["timeline_alignment"]["timeline_index"] == 27
    assert "中意人寿产品说明" in row["text"]
    assert "讲师指向方案一" in row["text"]
    assert "方案一被高亮" in row["text"]
    assert "标题位于页面顶部" in row["text"]
    assert "件均保费" in row["text"]
    assert "错误时间线对象" not in row["text"]


def test_inferred_visual_is_qualified_but_unavailable_visual_is_not_evidence(
    tmp_path: Path,
) -> None:
    inferred_root = _write_bundle(
        tmp_path,
        segments=[{"start": 20.0, "end": 22.0, "text": "这里有2个页面"}],
        timeline=[
            {
                "index": 8,
                "start": 20.0,
                "end": 22.0,
                "frame_path": "frames/inferred.jpg",
                "visual_understanding": {
                    "evidence_level": "inferred",
                    "objects": ["安联人寿页面"],
                    "inference_basis": "页面标题局部可见",
                },
            }
        ],
    )

    inferred = _visual_evidence(
        _number_candidate(
            build_transcript_semantic_correction_pack(inferred_root, write=False)
        )
    )
    assert len(inferred) == 1
    assert inferred[0]["evidence_level"] == "inferred"
    assert inferred[0]["text"].startswith("推断：画面可能显示")
    assert "安联人寿页面" in inferred[0]["text"]
    assert "页面标题局部可见" in inferred[0]["text"]

    # Use a second tmp root because the helper intentionally creates a fixed
    # bundle name.
    unavailable_root = _write_bundle(
        tmp_path / "unavailable",
        segments=[{"start": 30.0, "end": 32.0, "text": "这里有3个页面"}],
        timeline=[
            {
                "index": 9,
                "start": 30.0,
                "end": 32.0,
                "frame_path": "frames/unavailable.jpg",
                "visual_understanding": {
                    "evidence_level": "unavailable",
                    "objects": ["不得作为事实的伪内容"],
                    "unavailable_reason": "frame unreadable",
                },
            }
        ],
    )
    unavailable = _number_candidate(
        build_transcript_semantic_correction_pack(unavailable_root, write=False)
    )
    assert _visual_evidence(unavailable) == []
    assert all(
        "不得作为事实的伪内容" not in str(row.get("text") or "")
        for row in unavailable.get("evidence") or []
    )


def test_explicit_shared_segment_identity_is_recorded_and_can_override_time(
    tmp_path: Path,
) -> None:
    root = _write_bundle(
        tmp_path,
        segments=[
            {
                "segment_id": "source-segment-42",
                "start": 40.0,
                "end": 42.0,
                "text": "这里有4个重点",
            }
        ],
        timeline=[
            {
                "index": 14,
                "segment_id": "source-segment-42",
                "start": 400.0,
                "end": 402.0,
                "frame_path": "frames/shared-id.jpg",
                "visual_understanding": {
                    "evidence_level": "confirmed",
                    "objects": ["共享片段身份匹配"],
                },
            }
        ],
    )

    evidence = _visual_evidence(
        _number_candidate(build_transcript_semantic_correction_pack(root, write=False))
    )

    assert len(evidence) == 1
    assert evidence[0]["timeline_alignment"]["basis"] == "shared_identity"
    assert evidence[0]["timeline_alignment"]["shared_ids"] == [
        "source-segment-42"
    ]
    assert "共享片段身份匹配" in evidence[0]["text"]


def test_explicit_confirmed_visual_without_frame_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    root = _write_bundle(
        tmp_path,
        segments=[{"start": 50.0, "end": 52.0, "text": "这里有5个重点"}],
        timeline=[
            {
                "index": 15,
                "start": 50.0,
                "end": 52.0,
                "visual_understanding": {
                    "evidence_level": "confirmed",
                    "objects": ["缺少帧证据的内容"],
                },
            }
        ],
    )

    candidate = _number_candidate(
        build_transcript_semantic_correction_pack(root, write=False)
    )

    assert _visual_evidence(candidate) == []


def test_zero_duration_time_fallback_is_bounded_and_records_rounding_window(tmp_path: Path) -> None:
    bounded_root = _write_bundle(
        tmp_path,
        segments=[{"start": 60.0, "end": 60.0, "text": "这里有6个重点"}],
        timeline=[
            {
                "index": 16,
                "start": 60.125,
                "end": 60.125,
                "frame_path": "frames/rounding.jpg",
                "visual_understanding": {
                    "evidence_level": "confirmed",
                    "objects": ["四分之一秒内时间舍入匹配"],
                },
            }
        ],
    )
    bounded = _visual_evidence(
        _number_candidate(
            build_transcript_semantic_correction_pack(bounded_root, write=False)
        )
    )
    assert len(bounded) == 1
    assert bounded[0]["timeline_alignment"]["basis"] == "bounded_time_window"
    assert bounded[0]["timeline_alignment"]["distance_seconds"] == 0.125

    far_root = _write_bundle(
        tmp_path / "far",
        segments=[{"start": 70.0, "end": 72.0, "text": "这里有7个重点"}],
        timeline=[
            {
                "index": 17,
                "start": 80.0,
                "end": 82.0,
                "frame_path": "frames/far.jpg",
                "visual_understanding": {
                    "evidence_level": "confirmed",
                    "objects": ["不得进行无界最近帧匹配"],
                },
            }
        ],
    )
    far_candidate = _number_candidate(
        build_transcript_semantic_correction_pack(far_root, write=False)
    )
    assert _visual_evidence(far_candidate) == []
    asr_evidence = next(
        row
        for row in far_candidate["evidence"]
        if row.get("source_type") == "asr_or_subtitle"
    )
    assert asr_evidence["timeline_alignment"]["status"] == "not_matched"
    assert asr_evidence["timeline_alignment"]["basis"] == "none"
