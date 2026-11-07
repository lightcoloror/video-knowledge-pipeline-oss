from __future__ import annotations

import json
import hashlib
import subprocess
import time
from pathlib import Path

import pytest

from video_knowledge_pipeline.cli import audit_bundle_mcp_args, build_parser
from video_knowledge_pipeline.long_video_fast_segment import (
    apply_long_video_fast_segment_review,
    build_long_video_fast_segment_plan,
    render_long_video_fast_segment,
)
from video_knowledge_pipeline.storage import write_json


def _bundle(tmp_path: Path, *, visual_gap: bool = False) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    media = bundle / "source.mp4"
    media.write_bytes(b"synthetic-media-do-not-decode")
    transcript = bundle / "normalized-transcript.json"
    write_json(
        transcript,
        {
            "segments": [
                {"segment_id": "s1", "start": 0.0, "end": 5.0, "text": "今天介绍超长视频的第一部分。", "speaker": "speaker-1"},
                {"segment_id": "s2", "start": 15.0, "end": 20.0, "text": "等一下，我们先调试一下声音。", "speaker": "speaker-2"},
                {"segment_id": "s3", "start": 25.0, "end": 30.0, "text": "今天介绍超长视频的第一部分。", "speaker": "speaker-1"},
                {"segment_id": "s-smalltalk", "start": 31.0, "end": 34.0, "text": "今天天气怎么样，路上堵车吗？", "speaker": "speaker-2"},
                {"segment_id": "s-wait", "start": 35.0, "end": 37.0, "text": "请稍等一下，我们马上开始。", "speaker": "speaker-2"},
                {"segment_id": "s4", "start": 55.0, "end": 60.0, "text": "这里是最后的有效结论。", "speaker": "speaker-1"},
            ]
        },
    )
    vad = bundle / "silero-vad-candidate.json"
    write_json(
        vad,
        {
            "schema": "video_knowledge_pipeline.silero_vad_candidate.v1",
            "status": "completed",
            "segments": [
                {"start": 0.0, "end": 5.0},
                {"start": 15.0, "end": 20.0},
                {"start": 25.0, "end": 30.0},
                {"start": 31.0, "end": 34.0},
                {"start": 35.0, "end": 37.0},
                {"start": 55.0, "end": 60.0},
            ],
        },
    )
    write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "start": 5.0,
                "end": 15.0,
                "visual_text": "独有课件页" if visual_gap else "",
            }
        ],
    )
    write_json(
        bundle / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "title": "Long fixture",
            "source_video": "source.mp4",
            "normalized_transcript": "normalized-transcript.json",
            "silero_vad_candidate": "silero-vad-candidate.json",
        },
    )
    return bundle, media


def _confirm_all(bundle: Path, *, drop_first: bool = True) -> Path:
    path = bundle / "long-video-fast-segment-review.todo.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    for index, row in enumerate(value["decisions"]):
        row["decision"] = "drop" if drop_first and index == 0 else "keep"
    value["operator_confirmation"] = {
        "confirmed": True,
        "confirmed_by": "fixture-reviewer",
        "confirmed_at": "2026-08-18T16:00:00+08:00",
    }
    write_json(path, value)
    return path


def test_plan_combines_vad_transcript_visual_and_repeat_evidence(tmp_path: Path) -> None:
    bundle, media = _bundle(tmp_path, visual_gap=True)
    before = media.read_bytes()

    first = build_long_video_fast_segment_plan(bundle, write=True)
    second = build_long_video_fast_segment_plan(bundle, write=False)

    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["automatic_delete_allowed"] is False
    assert first["source_media_modified"] is False
    assert media.read_bytes() == before
    assert any(row["kind"] == "long_silence" for row in first["segments"])
    visual_gap = next(row for row in first["segments"] if row["start"] == 5.0)
    assert visual_gap["classification"] == "drop_review_required"
    assert "timeline:1" in visual_gap["evidence_ids"]
    assert any(row["kind"] == "technical_blank" for row in first["segments"])
    assert any(row["kind"] == "waiting_or_technical_setup" for row in first["segments"])
    assert any(row["kind"] == "non_information_smalltalk" for row in first["segments"])
    assert any(row["kind"] == "possible_retake_or_restatement" for row in first["segments"])
    assert (bundle / "exports/long-video-fast-segment-plan.md").is_file()
    assert (bundle / "long-video-fast-segment-review.todo.json").is_file()
    audit = audit_bundle_mcp_args(bundle)
    row = next(row for row in audit["rows"] if row["key"] == "mcp_long_video_fast_segment_plan_args")
    assert row["tool"] == "long_video_fast_segment_plan"
    assert row["ok"] is True


def test_plan_includes_hash_bound_vad_activity_blind_spots_as_review_only(
    tmp_path: Path,
) -> None:
    bundle, media = _bundle(tmp_path)
    vad = bundle / "silero-vad-candidate.json"
    activity_audit = bundle / "asr-vad-activity-audit.json"
    write_json(
        activity_audit,
        {
            "schema": "video_knowledge_pipeline.asr_vad_activity_audit.v1",
            "status": "review_required",
            "source_media": {
                "path": str(media),
                "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            },
            "vad_sha256": hashlib.sha256(vad.read_bytes()).hexdigest(),
            "candidate_gap_count": 1,
            "candidate_gaps": [
                {
                    "candidate_id": "audio-activity-gap-0001",
                    "start": 40.0,
                    "end": 50.0,
                    "candidate_only": True,
                    "reason": "non_silent_audio_without_vad_coverage",
                }
            ],
        },
    )

    result = build_long_video_fast_segment_plan(
        bundle,
        activity_audit_path=activity_audit,
        write=False,
    )

    candidate = next(
        row
        for row in result["segments"]
        if "activity-audit:audio-activity-gap-0001" in row["evidence_ids"]
    )
    assert candidate["start"] <= 40.0
    assert candidate["end"] >= 50.0
    assert candidate["classification"] == "drop_review_required"
    assert candidate["human_confirmation_required"] is True
    assert candidate["activity_audit_status"] == "non_silent_audio_without_vad_coverage"
    assert result["evidence"]["activity_audit"]["sha256"] == hashlib.sha256(
        activity_audit.read_bytes()
    ).hexdigest()
    assert result["operator_boundary"]["activity_audit_candidate_only"] is True


def test_plan_rejects_activity_audit_bound_to_other_media(tmp_path: Path) -> None:
    bundle, _media = _bundle(tmp_path)
    vad = bundle / "silero-vad-candidate.json"
    activity_audit = bundle / "asr-vad-activity-audit.json"
    write_json(
        activity_audit,
        {
            "schema": "video_knowledge_pipeline.asr_vad_activity_audit.v1",
            "status": "review_required",
            "source_media": {"sha256": "0" * 64},
            "vad_sha256": hashlib.sha256(vad.read_bytes()).hexdigest(),
            "candidate_gap_count": 0,
            "candidate_gaps": [],
        },
    )

    with pytest.raises(ValueError, match="source media does not match"):
        build_long_video_fast_segment_plan(
            bundle,
            activity_audit_path=activity_audit,
            write=False,
        )


def test_apply_requires_explicit_complete_review_and_preserves_speaker_turns(tmp_path: Path) -> None:
    bundle, media = _bundle(tmp_path)
    before = media.read_bytes()
    plan = build_long_video_fast_segment_plan(bundle, write=True)
    review_path = bundle / "long-video-fast-segment-review.todo.json"

    with pytest.raises(ValueError, match="explicit operator confirmation"):
        apply_long_video_fast_segment_review(bundle, review_path, write=False)

    review_path = _confirm_all(bundle)
    approved = apply_long_video_fast_segment_review(bundle, review_path, write=True)
    repeated = apply_long_video_fast_segment_review(bundle, review_path, write=False)

    assert approved["status"] == "ready_for_explicit_render"
    assert approved["operator_boundary"]["source_media_modified"] is False
    assert media.read_bytes() == before
    assert approved["keep_segments"]
    # The actual speaker turns remain in keep ranges unless the operator selected
    # their exact candidate; silence decisions never invade speech intervals.
    for start, end in ((0.0, 5.0), (15.0, 20.0), (25.0, 30.0), (31.0, 34.0), (35.0, 37.0), (55.0, 60.0)):
        assert any(row["start"] <= start and row["end"] >= end for row in approved["keep_segments"])
    assert plan["source_media"]["sha256"] == approved["source_media"]["sha256"]
    assert repeated["approved_sha256"] == approved["approved_sha256"]


def test_apply_fails_closed_when_media_or_plan_binding_changes(tmp_path: Path) -> None:
    bundle, media = _bundle(tmp_path)
    build_long_video_fast_segment_plan(bundle, write=True)
    review_path = _confirm_all(bundle)
    media.write_bytes(b"changed")

    with pytest.raises(ValueError, match="source media changed"):
        apply_long_video_fast_segment_review(bundle, review_path, write=False)


def test_render_defaults_to_preview_new_copy_and_never_launches_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, media = _bundle(tmp_path)
    build_long_video_fast_segment_plan(bundle, write=True)
    approved = apply_long_video_fast_segment_review(bundle, _confirm_all(bundle), write=True)
    monkeypatch.setattr("video_knowledge_pipeline.long_video_fast_segment.resolve_media_tool", lambda name: str(tmp_path / f"{name}.exe"))
    monkeypatch.setattr(
        "video_knowledge_pipeline.long_video_fast_segment.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preview must not execute ffmpeg")),
    )

    result = render_long_video_fast_segment(bundle, execute=False, write=True)

    assert result["status"] == "planned"
    assert result["operator_boundary"]["new_copy_only"] is True
    assert Path(result["output_path"]) != media
    assert not Path(result["output_path"]).exists()
    assert result["approved_edit"]["approved_sha256"] == approved["approved_sha256"]


def test_explicit_render_uses_shared_ffmpeg_receipt_and_writes_new_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, media = _bundle(tmp_path)
    build_long_video_fast_segment_plan(bundle, write=True)
    apply_long_video_fast_segment_review(bundle, _confirm_all(bundle), write=True)
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.write_bytes(b"ffmpeg")
    ffprobe.write_bytes(b"ffprobe")
    output = bundle / "exports" / "trimmed.mp4"

    monkeypatch.setattr(
        "video_knowledge_pipeline.long_video_fast_segment.resolve_media_tool",
        lambda name: str(ffmpeg if name == "ffmpeg" else ffprobe),
    )

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if Path(command[0]).name == "ffprobe.exe" and "stream=codec_type" in command:
            return subprocess.CompletedProcess(command, 0, '{"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}', "")
        if Path(command[0]).name == "ffprobe.exe":
            return subprocess.CompletedProcess(command, 0, "50.0\n", "")
        Path(command[-1]).write_bytes(b"rendered-new-copy")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("video_knowledge_pipeline.long_video_fast_segment.subprocess.run", fake_run)
    before = media.read_bytes()
    result = render_long_video_fast_segment(bundle, output_path=output, execute=True, write=True)

    assert result["status"] == "completed"
    assert output.read_bytes() == b"rendered-new-copy"
    assert media.read_bytes() == before
    assert result["ffmpeg_execution_receipt"]["outlet_id"] == "video_creation_pipeline.single_ffmpeg_outlet"
    assert result["qa"]["status"] == "passed"


def test_cli_exposes_plan_review_and_explicit_render() -> None:
    parser = build_parser()
    assert parser.parse_args(["long-video-fast-segment-plan", "bundle"]).command == "long-video-fast-segment-plan"
    assert parser.parse_args(["apply-long-video-fast-segment-review", "bundle", "review.json"]).command == "apply-long-video-fast-segment-review"
    render = parser.parse_args(["render-long-video-fast-segment", "bundle"])
    assert render.execute is False


def test_corrupt_transcript_fails_closed(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    (bundle / "normalized-transcript.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        build_long_video_fast_segment_plan(bundle, write=False)


def test_review_cannot_delete_the_entire_source(tmp_path: Path) -> None:
    bundle = tmp_path / "all-drop-bundle"
    bundle.mkdir()
    (bundle / "source.mp4").write_bytes(b"media")
    write_json(
        bundle / "normalized-transcript.json",
        {"segments": [{"segment_id": "only", "start": 0.0, "end": 60.0, "text": "现在调试一下声音。"}]},
    )
    write_json(bundle / "timeline.json", [])
    write_json(bundle / "manifest.json", {"source_video": "source.mp4", "normalized_transcript": "normalized-transcript.json"})
    build_long_video_fast_segment_plan(bundle, write=True)
    review = _confirm_all(bundle, drop_first=True)

    with pytest.raises(ValueError, match="delete the entire source"):
        apply_long_video_fast_segment_review(bundle, review, write=False)


def test_long_transcript_planning_stays_bounded_and_does_not_decode_media(tmp_path: Path) -> None:
    bundle = tmp_path / "large-bundle"
    bundle.mkdir()
    (bundle / "source.mp4").write_bytes(b"not-decoded")
    segments = [
        {
            "segment_id": f"segment-{index:05d}",
            "start": float(index * 2),
            "end": float(index * 2 + 1),
            "text": f"第{index}个不同的有效内容说明，包含唯一编号{index}。",
        }
        for index in range(2000)
    ]
    write_json(bundle / "normalized-transcript.json", {"segments": segments})
    write_json(bundle / "timeline.json", [])
    write_json(
        bundle / "manifest.json",
        {
            "source_video": "source.mp4",
            "normalized_transcript": "normalized-transcript.json",
        },
    )
    started = time.perf_counter()
    result = build_long_video_fast_segment_plan(bundle, long_silence_seconds=4.0, write=False)
    elapsed = time.perf_counter() - started

    assert result["duration_seconds"] == 3999.0
    assert elapsed < 5.0
    assert result["operator_boundary"]["external_provider_called"] is False
