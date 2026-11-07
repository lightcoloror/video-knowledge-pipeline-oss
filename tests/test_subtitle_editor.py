from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen  # noqa: TID251 - loopback-only Review Server test.

import pytest

from video_knowledge_pipeline.subtitle_editor import (
    apply_subtitle_review,
    build_subtitle_editor_projection,
    validate_subtitle_review,
)
from video_knowledge_pipeline.review_http import build_server
from video_knowledge_pipeline.subtitle_editor_ui import prepare_subtitle_editor


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bundle(root: Path, *, translated: bool = True) -> Path:
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    media = bundle / "interview.mp4"
    media.write_bytes(b"synthetic-media")
    _write_json(
        bundle / "manifest.json",
        {
            "title": "粤语采访",
            "media_path": str(media),
            "media_duration_seconds": 6.0,
            "normalized_transcript_json": "normalized-transcript.json",
            "mandarin_translated_transcript_json": "mandarin-translated-transcript.json",
        },
    )
    _write_json(
        bundle / "timeline.json",
        [
            {"index": 0, "start": 0.0, "end": 2.0, "transcript": "根据情况来的嘛"},
            {"index": 1, "start": 2.0, "end": 4.5, "transcript": "送了意外险"},
        ],
    )
    _write_json(
        bundle / "normalized-transcript.json",
        {
            "segments": [
                {
                    "segment_id": "segment-0001",
                    "source_segment_ids": ["raw-1"],
                    "start": 0.0,
                    "end": 2.0,
                    "text": "根据情况来的嘛",
                    "speaker_global_id": "speaker-global-001",
                    "words": [
                        {"word": "根据", "start": 0.0, "end": 0.5, "speaker": "speaker-global-001"},
                        {"word": "情况", "start": 0.5, "end": 1.2, "speaker": "speaker-global-001"},
                    ],
                },
                {
                    "segment_id": "segment-0002",
                    "source_segment_ids": ["raw-2"],
                    "start": 2.0,
                    "end": 4.5,
                    "text": "送了意外险",
                    "speaker_global_id": "speaker-global-002",
                },
            ]
        },
    )
    if translated:
        _write_json(
            bundle / "mandarin-translated-transcript.json",
            {
                "schema": "video_knowledge_pipeline.translated_transcript.v1",
                "segments": [
                    {"index": 0, "segment_id": "segment-0001", "source_text": "根据情况来的嘛", "text": "要根据情况决定"},
                    {"index": 1, "segment_id": "segment-0002", "source_text": "送了意外险", "text": "赠送了意外险"},
                ],
            },
        )
    return bundle


def _notes(projection: dict[str, object]) -> dict[str, object]:
    segments = json.loads(json.dumps(projection["segments"], ensure_ascii=False))
    segments[0]["source_text"] = "根据实际情况决定"
    segments[0]["mandarin_text"] = "需要根据实际情况决定"
    segments[0]["end_ms"] = 1900
    segments[1]["start_ms"] = 1900
    return {
        "schema": "video_knowledge_pipeline.subtitle_review_notes.v1",
        "projection_sha256": projection["projection_sha256"],
        "source_sha256": projection["source_sha256"],
        "segments": segments,
        "gap_remove": {"schema": "moy.asr.gap_remove.v1", "gaps": []},
        "human_confirmed": True,
    }


def test_projection_is_idempotent_and_preserves_dual_tracks_speakers_and_words(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    first = build_subtitle_editor_projection(bundle, write=False)
    second = build_subtitle_editor_projection(bundle, write=False)

    assert first == second
    assert first["schema"] == "video_knowledge_pipeline.subtitle_editor_projection.v1"
    assert first["tracks"]["source"]["language"] == "yue"
    assert first["tracks"]["mandarin"]["language"] == "zh-CN"
    assert first["segments"][0]["start_ms"] == 0
    assert first["segments"][0]["end_ms"] == 2000
    assert first["segments"][0]["speaker_global_id"] == "speaker-global-001"
    assert first["segments"][0]["mandarin_text"] == "要根据情况决定"
    assert first["segments"][0]["words"][1]["end_ms"] == 1200
    assert first["media_duration_ms"] == 6000


def test_projection_keeps_missing_translation_explicit(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, translated=False)

    result = build_subtitle_editor_projection(bundle, write=False)

    assert result["tracks"]["mandarin"]["status"] == "missing"
    assert result["segments"][0]["mandarin_text"] == ""


def test_validate_rejects_route_drift_unknown_source_and_cross_speaker_merge(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    projection = build_subtitle_editor_projection(bundle, write=False)
    notes = _notes(projection)
    notes["projection_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="projection_sha256"):
        validate_subtitle_review(bundle, notes)

    notes = _notes(projection)
    notes["segments"][0]["source_segment_ids"] = ["missing-source"]
    notes["segments"][0]["source_lineage_ids"] = ["missing-source"]
    with pytest.raises(ValueError, match="unknown source lineage ID"):
        validate_subtitle_review(bundle, notes)

    notes = _notes(projection)
    notes["segments"] = [
        {
            **notes["segments"][0],
            "segment_id": "human-merge",
            "source_segment_ids": ["raw-1", "raw-2"],
            "source_lineage_ids": ["raw-1", "raw-2"],
            "start_ms": 0,
            "end_ms": 4500,
        }
    ]
    with pytest.raises(ValueError, match="different speakers"):
        validate_subtitle_review(bundle, notes)


def test_validate_rejects_duplicate_id_out_of_media_bounds_and_damaged_json(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    projection = build_subtitle_editor_projection(bundle, write=False)

    duplicate = _notes(projection)
    duplicate["segments"][1]["segment_id"] = duplicate["segments"][0]["segment_id"]
    with pytest.raises(ValueError, match="missing or duplicate"):
        validate_subtitle_review(bundle, duplicate)

    out_of_bounds = _notes(projection)
    out_of_bounds["segments"][1]["end_ms"] = 6001
    with pytest.raises(ValueError, match="invalid or out-of-order timing"):
        validate_subtitle_review(bundle, out_of_bounds)

    damaged = bundle / "damaged-review.json"
    damaged.write_text("{not-json", encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        validate_subtitle_review(bundle, damaged)


def test_prepare_reuses_registered_source_package_media_without_scanning(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    media_path = manifest.pop("media_path")
    source_package = bundle.parent / "lecture-packages" / "lecture-package.json"
    _write_json(
        source_package,
        {
            "schema": "video_knowledge_pipeline.lecture_package.v1",
            "sources": [{"video_id": "video-1", "path": media_path}],
        },
    )
    manifest["source_package"] = str(source_package)
    manifest["sources"] = []
    _write_json(bundle / "manifest.json", manifest)

    result = prepare_subtitle_editor(bundle, write=False)

    assert result["ok"] is True
    assert result["timing_review_status"] == "ready"
    assert result["write"] is False


def test_projection_scopes_duplicate_chunk_ids_and_requires_overlap_review(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, translated=False)
    transcript_path = bundle / "normalized-transcript.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript["segments"][1].update(
        {
            "segment_id": "segment-0001",
            "source_segment_ids": ["raw-1"],
            "start": 1.5,
            "end": 4.5,
            "speaker_global_id": "speaker-global-001",
        }
    )
    _write_json(transcript_path, transcript)

    projection = build_subtitle_editor_projection(bundle, write=False)

    assert projection["timing_review"]["status"] == "needs_review"
    assert projection["timing_review"]["overlap_count"] == 1
    assert [row["segment_id"] for row in projection["segments"]] == [
        "segment-0001::occurrence-001",
        "segment-0001::occurrence-002",
    ]
    assert [row["source_lineage_ids"][0] for row in projection["segments"]] == [
        "raw-1::occurrence-001",
        "raw-1::occurrence-002",
    ]
    unresolved = {
        "schema": "video_knowledge_pipeline.subtitle_review_notes.v1",
        "projection_sha256": projection["projection_sha256"],
        "source_sha256": projection["source_sha256"],
        "segments": projection["segments"],
        "human_confirmed": True,
    }
    with pytest.raises(ValueError, match="invalid or out-of-order timing"):
        validate_subtitle_review(bundle, unresolved)

    resolved = json.loads(json.dumps(unresolved, ensure_ascii=False))
    resolved["segments"][0]["end_ms"] = 1500
    resolved["segments"][1]["start_ms"] = 1500
    validated = validate_subtitle_review(bundle, resolved)
    assert validated["ok"] is True
    assert validated["summary"]["timing_changes"] == 1

    merged = json.loads(json.dumps(unresolved, ensure_ascii=False))
    merged["segments"] = [
        {
            **merged["segments"][0],
            "segment_id": "human-merge:duplicate-provider-ids",
            "source_segment_ids": ["raw-1", "raw-1"],
            "source_lineage_ids": [
                "raw-1::occurrence-001",
                "raw-1::occurrence-002",
            ],
            "start_ms": 0,
            "end_ms": 4500,
        }
    ]
    assert validate_subtitle_review(bundle, merged)["ok"] is True


def test_apply_writes_reviewed_sidecars_without_mutating_sources(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    source_paths = [
        bundle / "normalized-transcript.json",
        bundle / "timeline.json",
        bundle / "mandarin-translated-transcript.json",
    ]
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    projection = build_subtitle_editor_projection(bundle, write=True)
    notes_path = bundle / "subtitle-review-notes.json"
    _write_json(notes_path, _notes(projection))

    result = apply_subtitle_review(bundle, review_json=notes_path, write=True)

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert (bundle / "human-corrected-transcript.json").is_file()
    assert (bundle / "human-reviewed-subtitle-track.json").is_file()
    assert (bundle / "human-reviewed-source.srt").is_file()
    assert (bundle / "human-reviewed-mandarin.srt").is_file()
    assert (bundle / "human-reviewed-source.vtt").is_file()
    assert (bundle / "human-reviewed-mandarin.ass").is_file()
    assert (bundle / "human-reviewed-subtitle.otio.json").is_file()
    assert (bundle / "human-reviewed-kept-ranges.json").is_file()
    assert (bundle / "human-reviewed.ffconcat").is_file()
    assert (bundle / "subtitle-review-apply-receipt.json").is_file()
    corrected = json.loads((bundle / "human-corrected-transcript.json").read_text(encoding="utf-8"))
    assert corrected["segments"][0]["raw_text"] == "根据情况来的嘛"
    assert corrected["segments"][0]["corrected_text"] == "根据实际情况决定"
    assert corrected["segments"][0]["changed"] is True
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    assert after == before
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["human_reviewed_subtitle_track_json"] == "human-reviewed-subtitle-track.json"
    assert manifest["smart_summary_status"] == "stale_after_subtitle_review"


def test_apply_requires_explicit_human_confirmation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    projection = build_subtitle_editor_projection(bundle, write=False)
    notes = _notes(projection)
    notes["human_confirmed"] = False

    with pytest.raises(ValueError, match="human_confirmed"):
        validate_subtitle_review(bundle, notes)


def test_apply_missing_translation_does_not_claim_or_export_complete_mandarin(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, translated=False)
    projection = build_subtitle_editor_projection(bundle, write=False)
    notes = _notes(projection)
    for row in notes["segments"]:
        row["mandarin_text"] = ""

    result = apply_subtitle_review(bundle, review_json=notes, write=True)

    assert result["translation_status"] == "incomplete"
    assert not (bundle / "human-reviewed-mandarin.srt").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert "human_reviewed_mandarin_srt" not in manifest


def test_gap_plan_requires_explicit_removed_true(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    projection = build_subtitle_editor_projection(bundle, write=False)
    notes = _notes(projection)
    notes["gap_remove"] = {
        "schema": "moy.asr.gap_remove.v1",
        "gaps": [
            {"start_ms": 100, "end_ms": 300},
            {"start_ms": 1000, "end_ms": 1200, "removed": True},
        ],
    }

    apply_subtitle_review(bundle, review_json=notes, write=True)

    kept = json.loads((bundle / "human-reviewed-kept-ranges.json").read_text(encoding="utf-8"))
    assert kept["ranges"] == [
        {"start_ms": 0, "end_ms": 1000},
        {"start_ms": 1200, "end_ms": 4500},
    ]


def test_prepared_page_reuses_full_shell_and_adds_vkp_dual_track_boundary(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    result = prepare_subtitle_editor(bundle, write=True)
    page = Path(result["html_path"]).read_text(encoding="utf-8")

    assert "VKP 双轨字幕审核" in page
    assert "vkp-mandarin-text" in page
    assert "保存到 VKP" in page
    assert "waveform-pane" in page
    assert "gap-remove-manage" in page
    assert "auto-merge-manage" in page
    assert "speaker-global-001" in page
    assert "approvedStickerOnly" in page
    assert "Qwen" not in page
    assert "Soniox API" not in page


def test_loopback_subtitle_editor_validates_and_applies_review(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    projection = build_subtitle_editor_projection(bundle, write=False)
    server = build_server(bundle, port=0, csrf_token="subtitle-token", refresh=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        with urlopen(base + "/subtitle-editor", timeout=10) as response:
            page = response.read().decode("utf-8")
        assert "VKP 双轨字幕审核" in page
        assert "subtitle-token" in page
        body = json.dumps(
            {
                "bundle_revision": projection["bundle_revision"],
                "subtitle_review_notes": _notes(projection),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-VKP-Review-Token": "subtitle-token",
            "Origin": base,
        }
        validate_request = Request(
            base + "/api/subtitle-editor/validate",
            method="POST",
            data=body,
            headers=headers,
        )
        with urlopen(validate_request, timeout=10) as response:
            validated = json.loads(response.read().decode("utf-8"))
        apply_request = Request(
            base + "/api/subtitle-editor/apply",
            method="POST",
            data=body,
            headers=headers,
        )
        with urlopen(apply_request, timeout=10) as response:
            applied = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert validated["ok"] is True
    assert validated["status"] == "validated"
    assert applied["ok"] is True
    assert (bundle / "subtitle-review-apply-receipt.json").is_file()
