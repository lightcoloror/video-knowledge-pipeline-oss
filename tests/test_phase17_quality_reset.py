from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline.cli import _mcp_callables

from video_knowledge_pipeline.asr_consensus import build_asr_consensus
from video_knowledge_pipeline.asr_runner import plan_asr_run
from video_knowledge_pipeline.quality_benchmark import _aligned_sample_window, _align_window_to_vad_segments, _asr_disagreement, _baseline_asr_text, _merged_review_seed, _timestamp_errors, build_quality_benchmark, run_quality_benchmark
from video_knowledge_pipeline.quality_console import export_quality_console
from video_knowledge_pipeline.semantic_chapter_plan import build_semantic_chapter_plan
from video_knowledge_pipeline.transcript_evidence_correction_pipeline import _quality_profile_execution


def _write_bundle(root: Path, *, duration: int = 7200) -> Path:
    root.mkdir(parents=True)
    segments = []
    timeline = []
    for index, start in enumerate(range(0, duration, 600), start=1):
        end = min(duration, start + 590)
        text = f"第{index}部分讨论客户沟通方法。接下来进入案例和执行步骤。"
        segments.append({"index": index, "start": start, "end": end, "text": text})
        timeline.append({"index": index, "start": start, "end": end, "transcript": text, "tags": ["重点", "案例"]})
    (root / "corrected-transcript.json").write_text(json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8")
    (root / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": "Phase 17 fixture", "corrected_transcript_json": "corrected-transcript.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def test_phase17_evidence_conflict_index_is_available_to_mcp_audit() -> None:
    assert "evidence_conflict_index" in _mcp_callables()


def test_asr_consensus_preserves_independent_hypotheses(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle", duration=1200)
    primary = bundle / "primary.json"
    secondary = bundle / "secondary.json"
    primary.write_text(json.dumps({"segments": [{"start": 0, "end": 10, "text": "使用 Playwright MCP 操作浏览器"}, {"start": 10, "end": 20, "text": "最后检查结果"}]}, ensure_ascii=False), encoding="utf-8")
    secondary.write_text(json.dumps({"segments": [{"start": 0, "end": 10, "text": "使用 Playwright client 操作浏览器"}, {"start": 10, "end": 20, "text": "最后检查结果"}]}, ensure_ascii=False), encoding="utf-8")

    result = build_asr_consensus(bundle, primary_transcript=primary, secondary_transcript=secondary, write=True)

    assert result["conflict_count"] == 1
    assert result["counts"]["agreement"] == 1
    assert result["operator_boundary"]["does_not_promote_secondary"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["asr_primary_transcript"] == str(primary.resolve())
    assert manifest["asr_secondary_transcript"] == str(secondary.resolve())


def test_semantic_chapter_plan_covers_long_video_and_builds_parts(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "long-bundle", duration=7200)

    result = build_semantic_chapter_plan(bundle, chapter_mode="semantic", write=True)

    assert result["ok"] is True
    assert result["duration_seconds"] >= 7190
    assert result["part_count"] >= 2
    assert result["chapters"][0]["start"] == 0
    assert result["chapters"][-1]["end"] >= 7190
    assert max(row["end"] - row["start"] for row in result["chapters"]) <= 900


def test_quality_benchmark_requires_human_reference_and_reports_metrics(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "benchmark-bundle", duration=1200)
    output = tmp_path / "benchmark"
    manifest = build_quality_benchmark(output, bundle_dirs=[bundle], samples_per_bundle=2, sample_seconds=60, write=True)
    assert manifest["sample_count"] == 2
    assert manifest["candidate_variant"] == "qwen3_asr_1_7b"
    assert len(manifest["summary_blind_review"]["items"]) == 1

    manifest_path = output / "quality-benchmark-manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = data["samples"][0]
    first["reference_text"] = "客户沟通方法，接下来进入案例和执行步骤。"
    second = data["samples"][1]
    second["reference_text"] = "客户沟通方法，接下来进入案例和执行步骤。"
    data["window_strategy"] = "asr_vad_sentence_aligned_v1"
    for sample in data["samples"]:
        sample["window_strategy"] = "asr_vad_sentence_aligned_v1"
        sample["boundary_alignment"] = {"ready": True, "start_aligned": True, "end_aligned": True, "status": "aligned"}
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    result = run_quality_benchmark(manifest_path, write=True)
    assert result["reference_ready_count"] == 2
    assert "corrected_transcript" in result["variants"]
    assert result["status"] == "needs_candidate_variant"
    assert result["candidate_available_count"] == 0
    assert result["candidate_required_count"] == 2
    assert "candidate_variant_missing" in {row["key"] for row in result["quality_blockers"]}
    assert "execute_quality_benchmark_candidate_variant" in result["next_actions"]
    assert result["acceptance"]["model_switch_allowed"] is False
    assert result["acceptance"]["decision"] == "keep_current_default"
    assert (output / "quality-benchmark.html").exists()


def test_quality_benchmark_prefills_punctuation_only_draft_from_fine_grained_asr(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "punctuated-review-draft", duration=60)
    raw = bundle / "normalized-transcript.json"
    raw.write_text(
        json.dumps(
            {
                "provider": "sensevoice",
                "segments": [
                    {"start": 0, "end": 3, "text": "大家好"},
                    {"start": 3, "end": 6, "text": "今天我们讨论获客"},
                    {"start": 6, "end": 9, "text": "然后看具体案例"},
                    {"start": 9, "end": 12, "text": "最后总结方法"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_quality_benchmark(
        tmp_path / "punctuated-review-output",
        bundle_dirs=[bundle],
        samples_per_bundle=1,
        sample_seconds=60,
        write=True,
    )

    sample = result["samples"][0]
    assert sample["asr_draft_source"] == str(raw.resolve())
    assert "，" in sample["asr_draft_text"]
    assert sample["asr_draft_text"].endswith("。")
    assert sample["reference_text"] == ""


def test_quality_benchmark_keeps_raw_postprocessed_and_corrected_variants_distinct(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "benchmark-stages", duration=1200)
    raw = bundle / "normalized-transcript.json"
    postprocessed = bundle / "postprocessed-transcript.json"
    raw.write_text(json.dumps({"provider": "sensevoice", "segments": [{"start": 0, "end": 590, "text": "原始识别"}]}, ensure_ascii=False), encoding="utf-8")
    postprocessed.write_text(json.dumps({"provider": "local_asr_postprocess", "segments": [{"start": 0, "end": 590, "text": "原始识别。"}]}, ensure_ascii=False), encoding="utf-8")

    result = build_quality_benchmark(tmp_path / "benchmark-stages-output", bundle_dirs=[bundle], samples_per_bundle=1, write=False)

    sample = result["samples"][0]
    assert sample["source_transcript"] == str(postprocessed.resolve())
    assert sample["variants"]["sensevoice_raw"] == str(raw.resolve())
    assert sample["variants"]["sensevoice_full_punc"] == str(postprocessed.resolve())
    assert sample["variants"]["corrected_transcript"] == str((bundle / "corrected-transcript.json").resolve())

def test_quality_benchmark_rejects_probably_partial_human_reference(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "benchmark-partial-reference", duration=1200)
    output = tmp_path / "benchmark-partial-output"
    build_quality_benchmark(output, bundle_dirs=[bundle], samples_per_bundle=1, write=True)
    manifest_path = output / "quality-benchmark-manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["samples"][0]["reference_text"] = "只标了一小段"
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    result = run_quality_benchmark(manifest_path, write=False)

    sample = result["samples"][0]
    assert sample["reference_ready"] is False
    assert sample["reference_completeness"]["reason"] == "reference_probably_partial"
    assert result["status"] == "needs_human_reference"

def test_quality_benchmark_trusts_completed_review_only_for_same_aligned_window(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "benchmark-completed-aligned-reference", duration=1200)
    output = tmp_path / "benchmark-completed-aligned-output"
    build_quality_benchmark(output, bundle_dirs=[bundle], samples_per_bundle=1, write=True)
    manifest_path = output / "quality-benchmark-manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = data["samples"][0]
    sample["reference_text"] = "短句。"
    sample["human_review_status"] = "completed"
    sample["reference_start_seconds"] = sample["start_seconds"]
    sample["reference_end_seconds"] = sample["end_seconds"]
    sample["window_strategy"] = "asr_vad_sentence_aligned_v1"
    sample["boundary_alignment"] = {"ready": True, "start_aligned": True, "end_aligned": True}
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    completed = run_quality_benchmark(manifest_path, write=False)

    assert completed["reference_ready_count"] == 1
    assert completed["samples"][0]["reference_completeness"]["reason"] == "completed_aligned_human_review"
    assert completed["samples"][0]["reference_completeness"]["reviewed_window_matches"] is True
    assert completed["reference_policy"]["legacy_reference_used_for_scoring"] is False

    data["samples"][0]["reference_end_seconds"] = float(sample["end_seconds"]) - 10.0
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    mismatched = run_quality_benchmark(manifest_path, write=False)

    assert mismatched["reference_ready_count"] == 0
    assert mismatched["samples"][0]["reference_completeness"]["reason"] == "reference_probably_partial"
    assert mismatched["samples"][0]["reference_completeness"]["reviewed_window_matches"] is False

def test_quality_benchmark_does_not_score_text_only_asr_as_bad_timestamps() -> None:
    text_only = [SimpleNamespace(start=42.0, end=42.0, text="只有文本，没有对齐结果")]
    sample = {"reference_start_seconds": 42.0, "reference_end_seconds": 102.0}

    result = _timestamp_errors(text_only, sample)

    assert result["available"] is False
    assert result["status"] == "asr_text_only_alignment_not_run"
    assert result["median"] is None
    assert result["p95"] is None

def test_quality_benchmark_aligns_window_to_complete_utterance_boundaries() -> None:
    cues = [
        SimpleNamespace(start=0.0, end=18.0, text="前一句。"),
        SimpleNamespace(start=18.0, end=33.0, text="这是一段连续发言的上半"),
        SimpleNamespace(start=33.0, end=62.0, text="这是连续发言的下半。"),
        SimpleNamespace(start=62.0, end=80.0, text="下一句。"),
    ]

    start, end = _aligned_sample_window(cues, center=40.0, duration=80.0, sample_seconds=20.0)

    assert start == 18.0
    assert end == 62.0


def test_asr_disagreement_normalizes_equivalent_chinese_and_arabic_amounts() -> None:
    result = _asr_disagreement(
        "客户保费是两百万元，复购率较高。",
        "客户保费是200万元，复购率较高。",
    )

    assert result["available"] is True
    assert result["review_priority"] == "low"
    assert result["number_conflicts"] == []


def test_asr_disagreement_normalizes_spoken_chinese_year_digits() -> None:
    result = _asr_disagreement(
        "这个案例发生在二四年。",
        "这个案例发生在24年。",
    )

    assert result["review_priority"] == "low"
    assert result["number_conflicts"] == []


def test_asr_disagreement_prioritizes_real_number_conflicts_without_human_reference() -> None:
    result = _asr_disagreement(
        "客户保费是200万元，复购率较高。",
        "客户保费是300万元，复购率较高。",
    )

    assert result["available"] is True
    assert result["review_priority"] == "high"
    assert result["number_conflicts"] == ["200万元", "300万元"]


def test_baseline_asr_text_reuses_source_bound_draft_when_variant_is_absent() -> None:
    assert _baseline_asr_text({}, {"asr_draft_text": "已有 SenseVoice 粤语草稿"}) == "已有 SenseVoice 粤语草稿"
    assert (
        _baseline_asr_text(
            {"sensevoice_full_punc": "独立重跑结果"},
            {"asr_draft_text": "已有草稿"},
        )
        == "独立重跑结果"
    )


def test_quality_benchmark_compares_qwen_with_source_bound_asr_draft(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    qwen = tmp_path / "qwen.json"
    source.write_text(
        json.dumps(
            {"segments": [{"start": 0.0, "end": 5.0, "text": "治疗需要二十八日。"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    qwen.write_text(
        json.dumps(
            {"segments": [{"start": 0.0, "end": 5.0, "text": "治疗需要二十日。"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "quality-benchmark-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.quality_benchmark_manifest.v1",
                "window_strategy": "asr_vad_sentence_aligned_v1",
                "candidate_variant": "qwen3_asr_1_7b",
                "samples": [
                    {
                        "sample_id": "detail-001",
                        "start_seconds": 0.0,
                        "end_seconds": 5.0,
                        "source_transcript": str(source),
                        "asr_draft_text": "治疗需要二十八日。",
                        "asr_draft_source": str(source),
                        "boundary_alignment": {"ready": True},
                        "human_review_status": "asr_prefilled_todo",
                        "variants": {"qwen3_asr_1_7b": str(qwen)},
                    }
                ],
                "summary_blind_review": {"required": False, "items": []},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_quality_benchmark(manifest, write=False)

    disagreement = result["samples"][0]["asr_disagreement"]
    assert disagreement["available"] is True
    assert disagreement["review_priority"] == "high"
    assert set(disagreement["number_conflicts"]) == {"二十八日", "二十日"}
    assert result["asr_disagreement_summary"]["compared_count"] == 1

def test_quality_benchmark_prefers_real_vad_boundaries_over_estimated_asr_chunks() -> None:
    segments = [
        {"start": 12.0, "end": 28.0},
        {"start": 31.0, "end": 74.0},
        {"start": 79.0, "end": 92.0},
    ]

    start, end, metadata = _align_window_to_vad_segments(
        segments,
        start=35.0,
        end=65.0,
        duration=120.0,
    )

    assert start == 30.8
    assert end == 74.2
    assert metadata["ready"] is True
    assert metadata["source"] == "funasr_fsmn_vad"

def test_quality_benchmark_merges_legacy_correction_with_new_asr_boundaries() -> None:
    result = _merged_review_seed(
        "大家好，今天分享客户经营的方法。",
        "开场提醒。大家好。今天分享客户经营的方法。最后做总结。",
    )

    assert result["source"] == "legacy_reference_plus_boundary_asr"
    assert result["merge_ready"] is True
    assert result["text"].startswith("开场提醒。")
    assert "大家好，今天分享客户经营的方法。" in result["text"]
    assert result["text"].endswith("最后做总结。")


def test_quality_benchmark_does_not_merge_unrelated_legacy_text() -> None:
    result = _merged_review_seed("人工确认的保险术语", "完全无关的片段级识别结果")

    assert result["source"] == "current_clip_asr_with_legacy_unmerged"
    assert result["merge_ready"] is False
    assert result["text"] == "完全无关的片段级识别结果"


def test_quality_benchmark_preserves_legacy_annotations_but_blocks_model_switch(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "legacy-window-bundle", duration=1200)
    output = tmp_path / "legacy-window-output"
    build_quality_benchmark(output, bundle_dirs=[bundle], samples_per_bundle=1, write=True)
    manifest_path = output / "quality-benchmark-manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.pop("window_strategy", None)
    data.pop("window_alignment_ready", None)
    sample = data["samples"][0]
    sample.pop("window_strategy", None)
    sample.pop("boundary_alignment", None)
    sample["start_seconds"] = float(sample["start_seconds"]) + 5.0
    sample["end_seconds"] = float(sample["end_seconds"]) - 5.0
    sample["reference_text"] = "人工已按被截断的音频完成标注"
    sample["human_review_status"] = "completed"
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    result = run_quality_benchmark(manifest_path, write=False)

    assert result["reference_ready_count"] == 1
    assert result["samples"][0]["reference_completeness"]["reason"] == "legacy_fixed_window_reference"
    assert result["status"] == "needs_boundary_aligned_rebuild"
    assert result["window_alignment_ready"] is False
    assert result["acceptance"]["model_switch_allowed"] is False

    migrated = build_quality_benchmark(
        tmp_path / "aligned-with-legacy",
        bundle_dirs=[bundle],
        samples_per_bundle=1,
        legacy_reference_manifest=manifest_path,
        write=True,
    )
    migrated_sample = migrated["samples"][0]
    assert migrated["legacy_reference_count"] == 1
    assert migrated_sample["legacy_reference_text"] == "人工已按被截断的音频完成标注"
    assert migrated_sample["human_review_status"] == "needs_boundary_extension"
    review_page = (tmp_path / "aligned-with-legacy" / "quality-benchmark-review.html").read_text(encoding="utf-8")
    assert "旧人工稿与当前 ASR 对齐不足，已用当前完整片段 ASR 预填" in review_page
    assert "人工已按被截断的音频完成标注" in review_page


def test_quality_benchmark_generates_unique_playable_human_review_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _write_bundle(tmp_path / "first-bundle", duration=1200)
    second = _write_bundle(tmp_path / "second-bundle", duration=1200)
    first_media = tmp_path / "first.mp4"
    second_media = tmp_path / "second.mp4"
    first_media.write_bytes(b"media")
    second_media.write_bytes(b"media")

    def fake_clip(media: Path, output: Path, *, start: float, end: float) -> str:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFFfixture")
        return ""

    monkeypatch.setattr("video_knowledge_pipeline.quality_benchmark._write_audio_clip", fake_clip)
    output = tmp_path / "review-pack"
    result = build_quality_benchmark(
        output,
        bundle_dirs=[first, second],
        media_paths=[first_media, second_media],
        samples_per_bundle=2,
        sample_seconds=60,
        execute_clips=True,
        write=True,
    )

    assert result["sample_count"] == 4
    assert result["audio_clip_count"] == 4
    assert len({row["sample_id"] for row in result["samples"]}) == 4
    assert all(row["asr_draft_text"] for row in result["samples"])
    assert all(row["reference_text"] == "" for row in result["samples"])
    assert all(row["human_review_status"] == "asr_prefilled_todo" for row in result["samples"])
    page = (output / "quality-benchmark-review.html").read_text(encoding="utf-8")
    assert page.count("<audio controls") == 4
    assert page.count('class="video-jump"') == 4
    assert 'id="sourceVideo"' in page
    assert "下载已审核基准清单" in page
    assert "已用当前音频片段的 ASR 结果预填" in page
    assert "const pauseAudios=" in page
    assert 'video.addEventListener("play",()=>pauseAudios())' in page
    assert 'audio.addEventListener("play",()=>{video.pause();pauseAudios(audio);})' in page


def test_quality_profile_does_not_export_without_permission(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "profile-bundle", duration=1200)
    result, runtime = _quality_profile_execution(
        bundle,
        profile_name="quality",
        provider_config={"provider": "openai_compatible", "base_url": "https://example.invalid/v1", "model": "test", "api_key": "runtime-only"},
        candidate_count=2,
        readable_max_prompt_chars=9000,
        explicit_execute=False,
    )
    assert runtime and runtime.get("api_key") == "runtime-only"
    assert result["status"] == "data_export_not_allowed"
    assert result["auto_execute"] is False
    assert "api_key" not in result["provider"]


def test_qwen3_plan_uses_official_runner_and_five_minute_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"fixture")
    monkeypatch.setattr("video_knowledge_pipeline.asr_runner._module_available_in_python", lambda module, python: module == "qwen_asr")
    monkeypatch.setattr("video_knowledge_pipeline.asr_runner._model_ready", lambda **kwargs: {"ready": False, "status": "not_cached"})
    monkeypatch.setattr("video_knowledge_pipeline.asr_runner._local_qwen_model_path", lambda preset: None)
    monkeypatch.setattr("video_knowledge_pipeline.asr_runner._local_qwen_aligner_path", lambda: None)

    result = plan_asr_run(
        project,
        media,
        preset="qwen3-asr-1.7b",
        language="yue",
        hotword="重疾险 免赔额",
    )

    assert result["runner"] == "qwen3_asr_python"
    assert "Qwen/Qwen3-ASR-1.7B" in result["command"]
    assert result["command"][result["command"].index("--chunk-seconds") + 1] == "300"
    assert result["command"][result["command"].index("--language") + 1] == "yue"
    assert result["command"][result["command"].index("--context") + 1] == "重疾险 免赔额"
    assert result["available"] is True


def test_quality_console_is_static_and_links_existing_surfaces(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "quality-ui", duration=1200)

    result = export_quality_console(bundle, write=True)

    assert result["operator_boundary"]["static_read_only_ui"] is True
    page = (bundle / "quality-console.html").read_text(encoding="utf-8")
    assert "task-console.html" in page
    assert "video-workbench.html" in page
    assert "copyCommand" in page
