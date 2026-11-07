from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.quality_benchmark_variants import execute_quality_benchmark_variants


def _manifest(root: Path) -> Path:
    root.mkdir()
    clip = root / "sample.wav"
    clip.write_bytes(b"RIFF")
    path = root / "quality-benchmark-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.quality_benchmark_manifest.v1",
                "samples": [
                    {
                        "sample_id": "sample-01",
                        "audio_clip_path": str(clip),
                        "reference_text": "人工内容",
                        "human_review_status": "completed",
                        "variants": {
                            "sensevoice_raw": "",
                            "sensevoice_full_punc": "",
                            "qwen3_asr_1_7b": "",
                            "qwen3_asr_0_6b": "",
                            "fun_asr_nano": "",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _plan_builder(workspace: Path, clip: Path, **kwargs):
    workspace.mkdir(parents=True, exist_ok=True)
    plan = workspace / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    return {
        "plan_path": str(plan),
        "local_asr_device": "cuda",
        "model_ready": {"ready": True},
        "preset": kwargs["preset"],
    }


def test_execute_variants_preview_does_not_run_models(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "benchmark")

    def fail_executor(*args, **kwargs):
        raise AssertionError("preview must not execute ASR")

    result = execute_quality_benchmark_variants(
        manifest,
        variants=["qwen3_asr_1_7b"],
        execute=False,
        plan_builder=_plan_builder,
        plan_executor=fail_executor,
    )

    assert result["status"] == "planned"
    assert result["status_counts"] == {"planned": 1}
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["samples"][0]["reference_text"] == "人工内容"
    assert saved["samples"][0]["human_review_status"] == "completed"
    assert saved["samples"][0]["variants"]["qwen3_asr_1_7b"] == ""


def test_execute_variants_writes_normalized_paths_and_resumes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "benchmark")
    calls: list[str] = []

    def executor(plan_path: str, **kwargs):
        calls.append(plan_path)
        output = Path(plan_path).parent / "normalized-transcript.json"
        output.write_text(
            json.dumps({"segments": [{"start": 0, "end": 1, "text": "识别结果"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        return {"status": "ok", "normalized": {"json_path": str(output)}, "stderr": ""}

    first = execute_quality_benchmark_variants(
        manifest,
        variants=["qwen3_asr_1_7b"],
        execute=True,
        plan_builder=_plan_builder,
        plan_executor=executor,
    )
    second = execute_quality_benchmark_variants(
        manifest,
        variants=["qwen3_asr_1_7b"],
        execute=True,
        plan_builder=_plan_builder,
        plan_executor=executor,
    )

    assert first["status"] == "completed"
    assert first["runs"][0]["device"] == "cuda"
    assert len(calls) == 1
    assert second["status_counts"] == {"skipped_existing": 1}
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert Path(saved["samples"][0]["variants"]["qwen3_asr_1_7b"]).exists()


def test_execute_variants_restores_source_timeline_offset(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "benchmark")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["samples"][0]["start_seconds"] = 120.0
    payload["samples"][0]["end_seconds"] = 180.0
    manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def executor(plan_path: str, **kwargs):
        output = Path(plan_path).parent / "normalized-transcript.json"
        output.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "start": 1.0,
                            "end": 2.0,
                            "text": "识别结果",
                            "words": [{"start": 1.0, "end": 1.5, "text": "识别"}],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {"status": "ok", "normalized": {"json_path": str(output)}, "stderr": ""}

    execute_quality_benchmark_variants(
        manifest,
        variants=["qwen3_asr_1_7b"],
        execute=True,
        plan_builder=_plan_builder,
        plan_executor=executor,
    )

    saved = json.loads(manifest.read_text(encoding="utf-8"))
    transcript = Path(saved["samples"][0]["variants"]["qwen3_asr_1_7b"])
    shifted = json.loads(transcript.read_text(encoding="utf-8"))
    assert shifted["segments"][0]["start"] == 121.0
    assert shifted["segments"][0]["end"] == 122.0
    assert shifted["segments"][0]["words"][0]["start"] == 121.0
    assert shifted["benchmark_time_offset_seconds"] == 120.0


def test_failed_qwen_1_7b_does_not_silently_run_0_6b(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "benchmark")

    def executor(plan_path: str, **kwargs):
        return {"status": "asr_model_not_ready", "normalized": None, "stderr": "model missing"}

    result = execute_quality_benchmark_variants(
        manifest,
        variants=["qwen3_asr_1_7b"],
        execute=True,
        plan_builder=_plan_builder,
        plan_executor=executor,
    )

    assert result["status"] == "blocked_or_failed"
    assert [row["variant"] for row in result["runs"]] == ["qwen3_asr_1_7b"]
    assert result["runs"][0]["status"] == "asr_model_not_ready"
    assert result["operator_boundary"]["qwen_0_6b_fallback_is_explicit"] is True


def test_execute_variants_no_write_preview_does_not_create_plans(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "benchmark")

    def fail_plan(*args, **kwargs):
        raise AssertionError("no-write preview must not create ASR plans")

    result = execute_quality_benchmark_variants(
        manifest,
        variants=["qwen3_asr_1_7b"],
        execute=False,
        write=False,
        plan_builder=fail_plan,
    )

    assert result["status"] == "planned"
    assert result["status_counts"] == {"planned_no_write": 1}
    assert not (manifest.parent / "variant-runs").exists()

def test_sensevoice_clip_result_prefills_review_text_without_overwriting_completed_reference(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "benchmark-prefill")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    sample["reference_text"] = ""
    sample["human_review_status"] = "asr_prefilled_todo"
    manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def executor(plan_path: str, **kwargs):
        output = Path(plan_path).parent / "normalized-transcript.json"
        output.write_text(
            json.dumps({"segments": [{"start": 0, "end": 4, "text": "这是片段级识别结果。"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        return {"status": "ok", "normalized": {"json_path": str(output)}, "stderr": ""}

    execute_quality_benchmark_variants(
        manifest,
        variants=["sensevoice_full_punc"],
        execute=True,
        plan_builder=_plan_builder,
        plan_executor=executor,
    )

    saved = json.loads(manifest.read_text(encoding="utf-8"))
    sample = saved["samples"][0]
    assert sample["asr_draft_text"] == "这是片段级识别结果。"
    assert sample["draft_source"] == "sample_clip_sensevoice_full_punc"
    assert Path(sample["asr_draft_source"]).exists()


def test_qwen_benchmark_variant_disables_inline_forced_alignment(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "benchmark-qwen-plan")
    captured: list[dict] = []

    def builder(workspace: Path, clip: Path, **kwargs):
        captured.append(kwargs)
        return _plan_builder(workspace, clip, **kwargs)

    execute_quality_benchmark_variants(
        manifest,
        variants=["qwen3_asr_1_7b"],
        execute=False,
        plan_builder=builder,
    )

    assert captured[0]["qwen_timestamps"] is False


def test_qwen_benchmark_uses_sample_language_and_evidence_context(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "benchmark-qwen-cantonese")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    sample["asr_language"] = "yue"
    sample["context_hotwords"] = ["重疾险", "免赔额"]
    manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    captured: list[dict] = []

    def builder(workspace: Path, clip: Path, **kwargs):
        captured.append(kwargs)
        return _plan_builder(workspace, clip, **kwargs)

    execute_quality_benchmark_variants(
        manifest,
        variants=["qwen3_asr_1_7b"],
        execute=False,
        plan_builder=builder,
    )

    assert captured[0]["language"] == "yue"
    assert captured[0]["hotword"] == "重疾险 免赔额"


def test_quality_benchmark_build_persists_deduplicated_context_hotwords(tmp_path: Path) -> None:
    from video_knowledge_pipeline.quality_benchmark import build_quality_benchmark

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    transcript = bundle / "transcript.json"
    transcript.write_text(
        json.dumps(
            {"segments": [{"index": 1, "start": 0, "end": 5, "text": "粤语采访测试。"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps({"corrected_transcript_json": str(transcript)}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = build_quality_benchmark(
        tmp_path / "benchmark",
        bundle_dirs=[bundle],
        samples_per_bundle=1,
        sample_seconds=5,
        asr_language="yue",
        asr_context_hotwords=["质子治疗", "保险理赔", "质子治疗", ""],
    )

    assert result["context_hotwords"] == ["质子治疗", "保险理赔"]
    assert result["samples"][0]["context_hotwords"] == ["质子治疗", "保险理赔"]


def test_contextual_paraformer_variants_keep_no_hotword_and_evidence_hotword_independent(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "benchmark-contextual")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    sample["context_hotwords"] = ["明亚保险", "小红书"]
    manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    captured: list[dict] = []

    def builder(workspace: Path, clip: Path, **kwargs):
        captured.append(kwargs)
        return _plan_builder(workspace, clip, **kwargs)

    execute_quality_benchmark_variants(
        manifest,
        variants=["contextual_paraformer_no_hotword", "contextual_paraformer_hotword"],
        execute=False,
        plan_builder=builder,
    )

    assert captured[0]["preset"] == "contextual-paraformer"
    assert captured[0]["hotword"] == ""
    assert captured[1]["preset"] == "contextual-paraformer"
    assert captured[1]["hotword"] == "明亚保险 小红书"
    assert all("reference_text" not in kwargs for kwargs in captured)
