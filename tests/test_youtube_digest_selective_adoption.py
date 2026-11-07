from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen  # noqa: TID251 - loopback-only Review Server test.

import pytest

from video_knowledge_pipeline.review_http import build_server
from video_knowledge_pipeline.smart_summary_chapters import (
    evaluate_chapter_timeline_coverage,
)
from video_knowledge_pipeline.smart_summary_codex import (
    _summary_timeline_anchor_quality,
)
from video_knowledge_pipeline.subtitle_editor import (
    apply_subtitle_review,
    build_subtitle_editor_projection,
    subtitle_translation_slice,
    validate_subtitle_review,
)
from video_knowledge_pipeline.subtitle_editor_ui import render_subtitle_editor_page


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bundle(root: Path, *, count: int = 6) -> Path:
    bundle = root / "youtube-digest-adoption-bundle"
    bundle.mkdir(parents=True)
    media = bundle / "synthetic.mp4"
    media.write_bytes(b"synthetic-media")
    segments = []
    translations = []
    for index in range(count):
        start = index * 2.0
        segment_id = f"segment-{index + 1:04d}"
        segments.append(
            {
                "segment_id": segment_id,
                "source_segment_ids": [f"raw-{index + 1}"],
                "start": start,
                "end": start + 2.0,
                "text": f"粤语原文第{index + 1}段",
                "speaker_global_id": f"speaker-global-{index % 2 + 1:03d}",
            }
        )
        translations.append(
            {
                "index": index,
                "segment_id": segment_id,
                "source_text": f"粤语原文第{index + 1}段",
                "text": f"普通话译文第{index + 1}段",
            }
        )
    _write_json(
        bundle / "manifest.json",
        {
            "title": "合成双语采访",
            "media_path": str(media),
            "media_duration_seconds": count * 2.0,
            "normalized_transcript_json": "normalized-transcript.json",
            "mandarin_translated_transcript_json": "mandarin-translated-transcript.json",
        },
    )
    _write_json(bundle / "timeline.json", segments)
    _write_json(bundle / "normalized-transcript.json", {"segments": segments})
    _write_json(
        bundle / "mandarin-translated-transcript.json",
        {
            "schema": "video_knowledge_pipeline.translated_transcript.v1",
            "status": "completed",
            "source_sha256": "1" * 64,
            "route_id": "local-reviewed-route",
            "route_revision": "2" * 64,
            "segments": translations,
        },
    )
    return bundle


def _review(projection: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "video_knowledge_pipeline.subtitle_review_notes.v1",
        "projection_sha256": projection["projection_sha256"],
        "source_sha256": projection["source_sha256"],
        "segments": json.loads(json.dumps(projection["segments"], ensure_ascii=False)),
        "human_confirmed": True,
    }


def test_stable_id_translation_batches_and_slice_are_bounded_and_idempotent(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    first = build_subtitle_editor_projection(bundle, write=False)
    second = build_subtitle_editor_projection(bundle, write=False)

    assert first == second
    loading = first["translation_loading"]
    assert loading["mode"] == "viewport_lazy_existing_sidecar"
    assert loading["batch_max_segments"] == 4
    assert [len(row["segment_ids"]) for row in loading["batches"]] == [4, 2]
    assert loading["provider_execution"] is False
    result = subtitle_translation_slice(
        bundle,
        projection_sha256=first["projection_sha256"],
        segment_ids=["segment-0002", "segment-0003"],
        generation=7,
    )
    assert result["generation"] == 7
    assert [row["segment_id"] for row in result["segments"]] == [
        "segment-0002",
        "segment-0003",
    ]
    assert [row["text"] for row in result["segments"]] == [
        "普通话译文第2段",
        "普通话译文第3段",
    ]
    assert result["operator_boundary"]["provider_execution"] is False

    with pytest.raises(ValueError, match="projection_sha256"):
        subtitle_translation_slice(
            bundle,
            projection_sha256="0" * 64,
            segment_ids=["segment-0001"],
        )
    with pytest.raises(ValueError, match="1-4"):
        subtitle_translation_slice(
            bundle,
            projection_sha256=first["projection_sha256"],
            segment_ids=[f"segment-{index + 1:04d}" for index in range(5)],
        )


def test_lazy_page_uses_modes_intersection_observer_and_generation_without_inline_translation(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    projection = build_subtitle_editor_projection(bundle, write=False)

    lazy_page = render_subtitle_editor_page(
        bundle,
        projection=projection,
        csrf_token="synthetic-token",
        lazy_translation=True,
    )
    static_page = render_subtitle_editor_page(
        bundle,
        projection=projection,
        lazy_translation=False,
    )

    assert "普通话译文第1段" not in lazy_page
    assert "普通话译文第1段" in static_page
    assert "IntersectionObserver" in lazy_page
    assert "translationGeneration" in lazy_page
    assert 'data-vkp-transcript-mode="original"' in lazy_page
    assert 'data-vkp-transcript-mode="mandarin"' in lazy_page
    assert 'data-vkp-transcript-mode="bilingual"' in lazy_page
    assert "vkp-note-original" in lazy_page
    assert "vkp-note-polished" in lazy_page


def test_loopback_translation_slice_echoes_generation_and_never_executes_provider(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    projection = build_subtitle_editor_projection(bundle, write=False)
    server = build_server(bundle, port=0, csrf_token="slice-token", refresh=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    query = urlencode(
        [
            ("projection_sha256", projection["projection_sha256"]),
            ("generation", "12"),
            ("segment_id", "segment-0001"),
            ("segment_id", "segment-0002"),
        ]
    )
    try:
        with urlopen(
            f"http://{host}:{port}/api/subtitle-editor/translations?{query}",
            timeout=10,
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["generation"] == 12
    assert len(result["segments"]) == 2
    assert result["operator_boundary"]["provider_execution"] is False


def test_timestamp_notes_preserve_original_quote_and_write_derived_sidecar(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, count=2)
    source = bundle / "normalized-transcript.json"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    projection = build_subtitle_editor_projection(bundle, write=False)
    review = _review(projection)
    review["timestamp_notes"] = [
        {
            "note_id": "note:segment-0001",
            "segment_id": "segment-0001",
            "timestamp_ms": 0,
            "original_quote": "粤语原文第1段",
            "polished_quote": "普通话润色摘录",
            "note_text": "这是人工时间戳笔记",
        }
    ]

    validated = validate_subtitle_review(bundle, review)
    assert validated["summary"]["timestamp_note_count"] == 1
    apply_subtitle_review(bundle, review_json=review, write=True)

    sidecar = json.loads(
        (bundle / "human-reviewed-timestamp-notes.json").read_text(encoding="utf-8")
    )
    assert sidecar["notes"][0]["original_quote"] == "粤语原文第1段"
    assert sidecar["notes"][0]["polished_quote"] == "普通话润色摘录"
    assert sidecar["notes"][0]["provenance"]["polished_quote_is_derived"] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before

    invalid = _review(projection)
    invalid["timestamp_notes"] = [
        {
            "note_id": "note:invalid",
            "segment_id": "segment-0001",
            "timestamp_ms": 0,
            "original_quote": "模型编造的原话",
            "note_text": "不能通过",
        }
    ]
    with pytest.raises(ValueError, match="not bound to source evidence"):
        validate_subtitle_review(bundle, invalid)


def test_chapter_and_final_summary_timeline_gates_cover_first_last_and_range() -> None:
    transcript = [
        {"start": 0.0, "end": 25.0},
        {"start": 25.0, "end": 75.0},
        {"start": 75.0, "end": 100.0},
    ]
    passed = evaluate_chapter_timeline_coverage(
        [
            {"start": 0.0, "end": 40.0},
            {"start": 40.0, "end": 100.0},
        ],
        transcript,
        duration_seconds=100.0,
    )
    assert passed["passed"] is True

    late_only = evaluate_chapter_timeline_coverage(
        [{"start": 40.0, "end": 80.0}],
        transcript,
        duration_seconds=100.0,
    )
    failed_keys = {
        row["key"] for row in late_only["checks"] if not row["passed"]
    }
    assert "first_quarter_covered" in failed_keys

    out_of_range = evaluate_chapter_timeline_coverage(
        [{"start": 0.0, "end": 110.0}],
        transcript,
        duration_seconds=100.0,
    )
    assert out_of_range["passed"] is False
    assert _summary_timeline_anchor_quality([5.0, 80.0], 100.0)[
        "first_last_passed"
    ] is True
    final_invalid = _summary_timeline_anchor_quality([5.0, 110.0], 100.0)
    assert final_invalid["timestamp_range_passed"] is False
