from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.transcript_reference_window import (
    export_transcript_reference_window,
)


def _speaker_transcript(path: Path) -> None:
    path.write_text(
        """说话人1 00:00:00
第一段
说话人2 00:00:10
第二段
说话人1 00:00:20
第三段
说话人2 00:00:40
第四段
""",
        encoding="utf-8",
    )


def test_reference_window_preserves_speaker_order_ids_and_clips_boundaries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reference.txt"
    output = tmp_path / "window.json"
    _speaker_transcript(source)

    receipt = export_transcript_reference_window(
        source,
        output,
        start_seconds=5,
        end_seconds=25,
        write=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "written"
    assert receipt["segment_count"] == 3
    assert receipt["speaker_evidence"]["speaker_count"] == 2
    assert [row["speaker"] for row in payload["segments"]] == [
        "说话人1",
        "说话人2",
        "说话人1",
    ]
    assert [row["segment_id"] for row in payload["segments"]] == [
        "segment-000001",
        "segment-000002",
        "segment-000003",
    ]
    assert [row["source_segment_ids"] for row in payload["segments"]] == [
        ["segment-000001"],
        ["segment-000002"],
        ["segment-000003"],
    ]
    assert [(row["start"], row["end"]) for row in payload["segments"]] == [
        (0.0, 5.0),
        (5.0, 15.0),
        (15.0, 20.0),
    ]
    assert all(
        row["transformations"][-1]["type"] == "evaluation_time_window_clip"
        for row in payload["segments"]
    )
    assert payload["policy"]["must_not_enter_prompt_hotwords_or_routing"] is True
    assert payload["policy"]["speaker_roles_are_not_inferred"] is True


def test_reference_window_preview_is_content_addressed_without_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reference.txt"
    output = tmp_path / "window.json"
    _speaker_transcript(source)

    receipt = export_transcript_reference_window(
        source,
        output,
        start_seconds=0,
        end_seconds=30,
        write=False,
    )

    assert receipt["status"] == "preview"
    assert receipt["artifact_written"] is False
    assert len(receipt["artifact_sha256"]) == 64
    assert not output.exists()


def test_reference_window_reuses_human_confirmed_correction_engine(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reference.txt"
    output = tmp_path / "window.json"
    corrections = tmp_path / "human-corrections.json"
    source.write_text(
        """说话人1 00:00:00
根排期来的嘛，会义纪要。
说话人2 00:00:10
发了一分材料。
""",
        encoding="utf-8",
    )
    corrections.write_text(
        json.dumps(
            {
                "source_sha256": sha256_file(source),
                "decisions": [
                    {
                        "candidate_id": "human-1",
                        "segment_index": 0,
                        "action": "replace",
                        "apply_scope": "segment",
                        "correction_type": "term",
                        "original_text": "根排期来的嘛",
                        "corrected_text": "根据排期来的嘛",
                        "human_confirmed": True,
                    },
                    {
                        "candidate_id": "human-2",
                        "segment_index": 0,
                        "action": "replace",
                        "apply_scope": "segment",
                        "correction_type": "term",
                        "original_text": "会义纪要",
                        "corrected_text": "会议纪要",
                        "human_confirmed": True,
                    },
                    {
                        "candidate_id": "human-3",
                        "segment_index": 1,
                        "action": "replace",
                        "apply_scope": "segment",
                        "correction_type": "term",
                        "original_text": "发了一分材料",
                        "corrected_text": "发了一份材料",
                        "human_confirmed": True,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    receipt = export_transcript_reference_window(
        source,
        output,
        start_seconds=0,
        end_seconds=15,
        human_corrections_json=corrections,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [row["text"] for row in payload["segments"]] == [
        "根据排期来的嘛，会议纪要。",
        "发了一份材料。",
    ]
    assert [row["speaker"] for row in payload["segments"]] == [
        "说话人1",
        "说话人2",
    ]
    assert receipt["human_corrections"]["applied_count"] == 3
    assert payload["segments"][0]["transformations"][0]["type"] == (
        "human_confirmed_source_fidelity_correction"
    )


def test_reference_window_rejects_unconfirmed_or_wrong_source_corrections(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reference.txt"
    output = tmp_path / "window.json"
    corrections = tmp_path / "human-corrections.json"
    _speaker_transcript(source)
    corrections.write_text(
        json.dumps(
            {
                "source_sha256": "0" * 64,
                "decisions": [
                    {
                        "segment_index": 0,
                        "action": "replace",
                        "original_text": "第一段",
                        "corrected_text": "修改",
                        "human_confirmed": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_sha256"):
        export_transcript_reference_window(
            source,
            output,
            start_seconds=0,
            end_seconds=20,
            human_corrections_json=corrections,
        )


def test_reference_window_rejects_unconfirmed_correction_on_matching_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reference.txt"
    corrections = tmp_path / "human-corrections.json"
    _speaker_transcript(source)
    corrections.write_text(
        json.dumps(
            {
                "source_sha256": sha256_file(source),
                "decisions": [
                    {
                        "segment_index": 0,
                        "action": "replace",
                        "original_text": "第一段",
                        "corrected_text": "修改",
                        "human_confirmed": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not human_confirmed"):
        export_transcript_reference_window(
            source,
            tmp_path / "window.json",
            start_seconds=0,
            end_seconds=20,
            human_corrections_json=corrections,
        )


@pytest.mark.parametrize(
    ("start_seconds", "end_seconds", "message"),
    [(-1, 10, "non-negative"), (10, 10, "greater than")],
)
def test_reference_window_rejects_invalid_bounds(
    tmp_path: Path,
    start_seconds: float,
    end_seconds: float,
    message: str,
) -> None:
    source = tmp_path / "reference.txt"
    _speaker_transcript(source)

    with pytest.raises(ValueError, match=message):
        export_transcript_reference_window(
            source,
            tmp_path / "window.json",
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
