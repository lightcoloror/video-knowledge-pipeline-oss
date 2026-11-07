from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import media_applicability as media
from video_knowledge_pipeline.knowledge_coverage import build_knowledge_coverage
from video_knowledge_pipeline.storage import write_json


@pytest.fixture(autouse=True)
def clear_probe_cache():
    media._probe_bound_bytes.cache_clear()
    yield
    media._probe_bound_bytes.cache_clear()


def bundle(tmp_path, monkeypatch, *, suffix=".mp3", audio=1, video=0):
    source = tmp_path / ("source" + suffix)
    source.write_bytes(b"synthetic media bytes")
    manifest = {"sources": [{"path": str(source)}]}
    timeline = [{"index": 1, "start": 0, "end": 3, "transcript": "speech",
                 "material_types": ["text", "table"], "frame_paths": [],
                 "quality_issues": ["missing_frame", "missing_ocr", "missing_visual_text",
                                    "structured_visual_without_structure", "needs_human_review"]}]
    monkeypatch.setattr(media, "_probe_with_ffprobe", lambda p: {
        "audio_stream_count": audio, "video_stream_count": video, "duration_seconds": 3})
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "timeline.json", timeline)
    return manifest, timeline


@pytest.mark.parametrize("suffix", [".mp3", ".m4a", ".mp4"])
def test_verified_audio_exempts_only_visual_obligations(tmp_path, monkeypatch, suffix):
    manifest, timeline = bundle(tmp_path, monkeypatch, suffix=suffix)
    before = json.dumps(timeline, sort_keys=True)
    report = build_knowledge_coverage(manifest, timeline, bundle_dir=tmp_path)
    for row in report["channels"]:
        if row["key"] in media.VISUAL_CHANNELS:
            assert row["status"] == "not_applicable"
            assert row["expected_count"] == row["covered_count"] == row["blocker_count"] == 0
    assert report["media_applicability"]["source_bound"] is True
    inventory = report["ocr_execution"]
    assert inventory["status"] == "not_applicable"
    assert inventory["summary"]["executed"] == 0
    assert inventory["summary"]["execution_complete"] is None
    assert json.dumps(timeline, sort_keys=True) == before
    projected = media.project_timeline_for_media(timeline, report["media_applicability"])
    assert projected[0]["quality_issues"] == ["needs_human_review"]
    assert not (tmp_path / "media-applicability.json").exists()


@pytest.mark.parametrize("mode", ["video_without_frames", "probe_failed", "unknown_streams", "no_sources"])
def test_video_and_unverified_inputs_keep_visual_obligations(tmp_path, monkeypatch, mode):
    manifest, timeline = bundle(tmp_path, monkeypatch, video=1)
    if mode == "probe_failed":
        def fail(p):
            raise RuntimeError("synthetic probe failure")
        monkeypatch.setattr(media, "_probe_with_ffprobe", fail)
    elif mode == "unknown_streams":
        monkeypatch.setattr(media, "_probe_with_ffprobe", lambda p: {"duration_seconds": 3})
    elif mode == "no_sources":
        manifest["sources"] = []
        write_json(tmp_path / "manifest.json", manifest)
    report = build_knowledge_coverage(manifest, timeline, bundle_dir=tmp_path)
    assert report["media_applicability"]["visual_required"] is True
    assert any(c["status"] == "blocked" for c in report["channels"] if c["key"] in media.VISUAL_CHANNELS)
    assert report["ocr_execution"] is None  # Existing video execution status is not fabricated.


@pytest.mark.parametrize("material", ["frame", "assets_only", "companion", "declared_missing_companion", "panorama"])
def test_audio_does_not_drop_supplementary_visual_materials(tmp_path, monkeypatch, material):
    manifest, timeline = bundle(tmp_path, monkeypatch)
    if material == "frame":
        timeline[0]["frame_paths"] = ["user-slide.png"]
    elif material == 'assets_only':
        timeline[0]['assets'] = [{'path': 'user-slide.png'}]
    elif material == "companion":
        manifest["visual_materials"] = ["slides.pdf"]
    elif material == "panorama":
        timeline[0]["panorama_visual_understanding"] = {"sessions": []}
    else:
        manifest["companion_courseware_text"] = "exports/companion-courseware-text.json"
    result = media.resolve_media_applicability(tmp_path, manifest, timeline)
    assert result["media_kind"] == "audio_only"
    assert result["visual_status"] == "required"
    assert media.audio_only_ocr_status(result) is None


def test_relative_media_source_remains_audio_only_after_repair_refresh(tmp_path, monkeypatch):
    from video_knowledge_pipeline.repair_status import refresh_bundle_repair_status
    manifest, timeline = bundle(tmp_path, monkeypatch)
    manifest['sources'][0]['path'] = 'source.mp3'
    write_json(tmp_path / 'manifest.json', manifest)
    result = refresh_bundle_repair_status(tmp_path)
    assert result['repair_status']['media_applicability']['visual_status'] == 'not_applicable'
    updated = json.loads((tmp_path / 'manifest.json').read_text('utf-8'))
    assert updated['review_readiness']['media_applicability']['visual_status'] == 'not_applicable'
    assert all(row['status'] == 'not_applicable' for row in result['repair_status']['items']
               if row['key'] != 'time_gap_recapture')


def test_source_change_invalidates_bound_audio_exemption(tmp_path, monkeypatch):
    manifest, timeline = bundle(tmp_path, monkeypatch)
    source = Path(manifest["sources"][0]["path"])
    manifest["sources"][0]["sha256"] = media.sha256_file(source)
    old = media.resolve_media_applicability(tmp_path, manifest, timeline, write=True)
    assert old["visual_status"] == "not_applicable"
    source.write_bytes(b"different media")
    current = media.resolve_media_applicability(tmp_path, manifest, timeline)
    assert current["visual_status"] == "required"
    assert current["sources"][0]["error"] == "source_sha256_mismatch"


def test_stale_manifest_flag_cannot_bypass_probe(tmp_path, monkeypatch):
    manifest, timeline = bundle(tmp_path, monkeypatch, video=1)
    manifest["media_applicability"] = {"visual_status": "not_applicable", "media_kind": "audio_only"}
    assert media.resolve_media_applicability(tmp_path, manifest, timeline)["visual_required"]


def test_video_probe_distinguishes_cover_art_and_real_video(tmp_path, monkeypatch):
    from video_knowledge_pipeline import video
    from types import SimpleNamespace
    monkeypatch.setattr(video, "resolve_media_tool", lambda name: "ffprobe")
    payload = {"streams": [{"codec_type": "audio"}, {"codec_type": "video", "disposition": {"attached_pic": 1}}],
               "format": {"duration": "12"}}
    monkeypatch.setattr(video.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload)))
    result = video._probe_with_ffprobe(tmp_path / "audio.m4a")
    assert result["audio_stream_count"] == 1
    assert result["video_stream_count"] == 0
    assert result["attached_picture_count"] == 1
    payload["streams"].append({"codec_type": "video", "width": 100, "height": 100})
    assert video._probe_with_ffprobe(tmp_path / "video.mp4")["video_stream_count"] == 1


def test_control_and_readiness_use_same_applicability(tmp_path, monkeypatch):
    from video_knowledge_pipeline.bundle_status import _controlled_execution_summary
    from video_knowledge_pipeline.bundle_readiness import build_bundle_readiness
    from video_knowledge_pipeline.repair_status import build_repair_status
    manifest, timeline = bundle(tmp_path, monkeypatch)
    manifest["repair_status"] = build_repair_status(manifest, timeline)
    ready = build_bundle_readiness(manifest, timeline, bundle_dir=tmp_path)
    assert ready["counts"]["frame_gap_items"] == 0
    assert ready["counts"]["structure_gap_items"] == 0
    assert _controlled_execution_summary(tmp_path, manifest)["status"] == "not_applicable"
    assert all(r["status"] == "not_applicable" for r in manifest["repair_status"]["items"] if r["key"] != "time_gap_recapture")
    from video_knowledge_pipeline.review_session import _review_targets
    targets = _review_targets(timeline, readiness=ready, manifest=manifest, bundle_dir=tmp_path)
    assert targets['media_applicability']['visual_required'] is False
    assert targets['total_open'] == 1  # Generic human-review uncertainty stays.
    assert 'frame_gap' not in targets['by_reason']
    assert 'missing_visual_text' not in targets['by_reason']
    assert 'pending_structured' not in targets['by_reason']


def test_probe_timeout_keeps_visual_obligations(tmp_path, monkeypatch):
    from subprocess import TimeoutExpired
    manifest, timeline = bundle(tmp_path, monkeypatch)
    def timeout(path):
        raise TimeoutExpired('ffprobe', 30)
    monkeypatch.setattr(media, '_probe_with_ffprobe', timeout)
    result = media.resolve_media_applicability(tmp_path, manifest, timeline)
    assert result['status'] == 'unknown'
    assert result['visual_required'] is True
    assert 'timed out' in result['sources'][0]['error']


@pytest.mark.parametrize('invalid_stream', [None, 'unknown', {}, {'codec_type': 'video', 'disposition': []},
    {'codec_type': 'video', 'disposition': {'attached_pic': '0'}}])
def test_invalid_probe_inventory_cannot_hide_a_video_stream(tmp_path, monkeypatch, invalid_stream):
    from types import SimpleNamespace
    from video_knowledge_pipeline import video
    monkeypatch.setattr(video, 'resolve_media_tool', lambda name: 'ffprobe')
    payload = {'streams': [{'codec_type': 'audio'}, invalid_stream], 'format': {'duration': '12'}}
    monkeypatch.setattr(video.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload)))
    with pytest.raises(ValueError):
        video._probe_with_ffprobe(tmp_path / 'audio.mp3')


@pytest.mark.parametrize('suffix,video_streams', [('.mp3', 0), ('.m4a', 0), ('.mp4', 0), ('.mp4', 1)])
def test_new_media_preparation_skips_frame_work_only_for_verified_audio(tmp_path, monkeypatch, suffix, video_streams):
    from video_knowledge_pipeline import orchestrator, video
    manifest, _ = bundle(tmp_path, monkeypatch, suffix=suffix, video=video_streams)
    monkeypatch.setattr(video, '_probe_with_ffprobe', media._probe_with_ffprobe)
    called = []
    monkeypatch.setattr(orchestrator, 'extract_segment_frames', lambda *a: called.append('frames'))
    monkeypatch.setattr(orchestrator, 'pyscenedetect_timepoints', lambda *a, **kw: ([], {}))
    result = orchestrator.add_video(tmp_path / 'prepared', manifest['sources'][0]['path'], topic='synthetic')
    assert bool(called) is bool(video_streams)
    assert result['sampling_plan']['media_capability']['source_bound'] is True
    assert Path(result['segments_path']).is_file()


def test_summary_review_and_acceptance_share_audio_applicability(tmp_path, monkeypatch):
    from video_knowledge_pipeline.acceptance_check import acceptance_check
    from video_knowledge_pipeline.production_artifact_gate import evaluate_production_artifact_gate
    from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
    from video_knowledge_pipeline.webui_bridge import refresh_bundle_review_html
    manifest, timeline = bundle(tmp_path, monkeypatch)
    manifest.update(title='synthetic audio', normalized_transcript_json='normalized-transcript.json')
    write_json(tmp_path / 'manifest.json', manifest)
    write_json(tmp_path / 'normalized-transcript.json', {'segments': [{'start': 0, 'end': 3, 'text': '待核词。'}]})
    write_json(tmp_path / 'transcript-quality-gate.json', {
        'schema': 'video_knowledge_pipeline.transcript_quality_gate.v1', 'status': 'failed',
        'ok': False, 'fail_count': 1, 'source_completeness': {'status': 'failed'},
    })
    pack = build_smart_summary_input_pack(tmp_path, write=False)
    assert pack['media_applicability']['visual_status'] == 'not_applicable'
    assert pack['visual_digest']['status'] == 'not_applicable'
    refresh_bundle_review_html(tmp_path, write=True)
    html = (tmp_path / 'review.html').read_text('utf-8')
    assert 'OCR 不适用' in html
    assert json.loads((tmp_path / 'timeline.json').read_text('utf-8')) == timeline
    gate = evaluate_production_artifact_gate(tmp_path, discover_prior_reviews=False, write=False)
    assert gate['media_applicability']['visual_status'] == 'not_applicable'
    assert gate['formal_generation_allowed'] is False
    result = acceptance_check(tmp_path, refresh=True, write=False)
    assert result['status'] == 'incomplete'
    assert result['summary']['frames'] == result['summary']['screen_text'] == 'not_applicable'
    assert result['summary']['ocr_execution_summary']['required'] == 0
    assert any(row['key'] == 'transcript_quality_failed' for row in result['blockers'])
