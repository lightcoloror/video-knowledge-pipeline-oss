from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_response_quality import assess_asr_response
from video_knowledge_pipeline.asr_response_quality_cli import main as quality_cli_main
from video_knowledge_pipeline.model_connector_consent import (
    create_model_connector_consent,
    validate_model_connector_consent,
)
from video_knowledge_pipeline.trusted_model_connector import (
    execute_consented_model_task,
)


PROVIDER = {
    "provider": "custom_openai_compatible_asr",
    "base_url": "https://asr.example/v1",
    "model": "whisper-test",
}


def test_short_hotword_matches_are_not_asr_prompt_leak() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 2,
                    "text": "alpha beta",
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0,
                }
            ]
        },
        asr_prompt="Context terms: alpha, beta, gamma, delta, epsilon, zeta.",
    )

    assert result["status"] == "passed"
    assert result["failed_chunks"] == []


def test_long_asr_prompt_echo_is_blocked() -> None:
    prompt = "Context terms alpha beta gamma delta epsilon zeta eta theta."
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 4,
                    "text": prompt,
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0,
                }
            ]
        },
        asr_prompt=prompt,
    )

    assert result["status"] == "degraded"
    assert result["failed_chunks"][0]["reasons"] == ["asr_prompt_leak"]


def _audio(root: Path) -> Path:
    path = root / "sample.wav"
    path.write_bytes(b"RIFF-fake-wave")
    return path


def test_verbose_asr_quality_flags_instruction_leak_and_builds_vad_retry() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "id": 4,
                    "start": 123.82,
                    "end": 153.8,
                    "text": "请逐字转写整个字。",
                    "avg_logprob": -1.0868256,
                    "compression_ratio": 0.75,
                    "no_speech_prob": 0,
                },
                {
                    "id": 5,
                    "start": 153.8,
                    "end": 160.0,
                    "text": "这里继续讲方案制作原则。",
                    "avg_logprob": -0.08,
                    "compression_ratio": 1.35,
                    "no_speech_prob": 0.01,
                },
            ]
        },
        task_instructions="请逐字转写整段中文知识视频音频，保留分段时间戳。",
        vad_intervals=[{"start": 122.5, "end": 154.2}],
        media_duration_seconds=200,
    )

    assert result["status"] == "degraded"
    assert result["quality_gate_passed"] is False
    assert result["failed_segment_count"] == 1
    assert "task_instruction_leak" in result["failed_chunks"][0]["reasons"]
    assert result["segments"][0]["avg_logprob"] == -1.0868256
    assert result["segments"][0]["compression_ratio"] == 0.75
    assert result["segments"][0]["no_speech_prob"] == 0
    retry = result["retry_plan"]
    assert retry["requires_new_exact_consent"] is True
    assert retry["silent_provider_fallback_allowed"] is False
    assert retry["windows"][0]["alignment_source"] == "vad_boundary"
    assert retry["windows"][0]["start"] == 121.0
    assert retry["windows"][0]["end"] == 155.7
    assert result["preservation"]["flagged_original_text_preserved"] is True


def test_low_logprob_alone_requests_review_without_erasing_text() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 3,
                    "text": "明亚保险方案",
                    "avg_logprob": -0.9,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0,
                }
            ]
        }
    )

    assert result["status"] == "review_required"
    assert result["quality_gate_passed"] is False
    assert result["failed_chunks"] == []
    assert result["review_chunks"][0]["original_text"] == "明亚保险方案"

def test_long_sparse_review_segment_builds_targeted_retry_without_erasing_text() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "id": 368,
                    "start": 919.359,
                    "end": 949.359,
                    "text": "\u6210\u529f\u5730\u52fe\u8d77\u4e86\u5ba2\u6237\u7684\u5174\u8d77\u70b9\uff0c",
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0,
                }
            ]
        },
        media_duration_seconds=1000,
    )

    assert result["status"] == "review_required"
    assert result["quality_gate_passed"] is False
    assert result["failed_segment_count"] == 0
    assert result["review_segment_count"] == 1
    assert result["review_chunks"][0]["original_text"] == "\u6210\u529f\u5730\u52fe\u8d77\u4e86\u5ba2\u6237\u7684\u5174\u8d77\u70b9\uff0c"
    retry = result["retry_plan"]
    assert retry["status"] == "authorization_required"
    assert retry["requires_new_exact_consent"] is True
    assert retry["windows"] == [
        {
            "retry_id": "retry-0001",
            "source_segment_ids": ["368"],
            "start": 917.859,
            "end": 950.859,
            "alignment_source": "provider_segment_boundary",
            "reasons": ["low_text_density"],
            "snippet_artifact_status": "not_created",
        }
    ]


def test_low_logprob_review_is_not_production_qualified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    consent = create_model_connector_consent(
        tmp_path,
        task="cloud_asr",
        artifact_paths=[_audio(tmp_path)],
        provider_config=PROVIDER,
        instructions="Audit-only transcription instructions.",
        asr_prompt="domain term",
        max_estimated_cost_usd=0.1,
        max_retries_per_call=0,
        confirm_data_export=True,
    )

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        lambda *args, **kwargs: {
            "ok": True,
            "status": "completed",
            "content": "domain term transcript",
            "raw_response": {
                "segments": [
                    {
                        "start": 0,
                        "end": 2,
                        "text": "domain term transcript",
                        "avg_logprob": -0.9,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0,
                    }
                ]
            },
        },
    )

    result = execute_consented_model_task(
        consent["consent_path"],
        provider_config=PROVIDER,
        write=False,
    )

    assert result["ok"] is True
    assert result["status"] == "review_required"
    assert result["transport_ok"] is True
    assert result["quality_gate_passed"] is False
    assert result["production_qualified"] is False
    quality = result["model_result"]["asr_quality"]
    assert quality["failed_segment_count"] == 0
    assert quality["review_segment_count"] == 1


def test_missing_verbose_segments_fails_closed() -> None:
    result = assess_asr_response({"text": "只有整段文本，没有分段元数据"})

    assert result["status"] == "failed"
    assert result["quality_gate_passed"] is False
    assert result["segment_count"] == 0
    assert result["response_issues"][0]["key"] == "verbose_segments_missing"
    assert result["retry_plan"]["windows"] == []


def test_quality_gate_unwraps_saved_connector_execution() -> None:
    result = assess_asr_response(
        {
            "model_result": {
                "runtime_result": {
                    "raw_output": {
                        "segments": [
                            {
                                "start": 0,
                                "end": 2,
                                "text": "usable transcript",
                                "avg_logprob": -0.1,
                                "compression_ratio": 1.1,
                                "no_speech_prob": 0,
                            }
                        ]
                    }
                }
            }
        }
    )

    assert result["status"] == "passed"
    assert result["segment_count"] == 1


def test_quality_cli_reads_audit_instructions_from_saved_consent(
    tmp_path: Path,
) -> None:
    consent_path = tmp_path / "consent.json"
    consent_path.write_text(
        '{"instructions":"never emit this audit instruction","asr_prompt":"domain term"}',
        encoding="utf-8",
    )
    execution_path = tmp_path / "connector-execution.json"
    execution_path.write_text(
        """{
          "consent_path": "%s",
          "model_result": {
            "runtime_result": {
              "raw_output": {
                "segments": [{
                  "start": 0,
                  "end": 2,
                  "text": "never emit this audit instruction",
                  "avg_logprob": -0.1,
                  "compression_ratio": 1.1,
                  "no_speech_prob": 0
                }]
              }
            }
          }
        }"""
        % str(consent_path).replace("\\", "\\\\"),
        encoding="utf-8",
    )
    output_path = tmp_path / "quality.json"

    assert quality_cli_main([str(execution_path), str(output_path)]) == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["status"] == "degraded"
    assert report["failed_chunks"][0]["reasons"] == ["task_instruction_leak"]
    assert report["input_context"]["instruction_source"] == "connector_consent"
    assert report["input_context"]["task_instructions_sha256"]
    assert report["input_context"]["asr_prompt_sha256"]
    assert report["input_context"]["plaintext_instructions_persisted"] is False
    assert "never emit" not in str(report["input_context"])


def test_asr_consent_separates_audit_instructions_from_provider_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    consent = create_model_connector_consent(
        tmp_path,
        task="cloud_asr",
        artifact_paths=[_audio(tmp_path)],
        provider_config=PROVIDER,
        instructions="逐字转写整段音频并输出时间戳。",
        asr_prompt="明亚保险 领航计划 明亚APP",
        max_estimated_cost_usd=0.1,
        max_retries_per_call=0,
        confirm_data_export=True,
    )
    captured: dict[str, object] = {}

    def fake_call(task: str, **kwargs: object) -> dict[str, object]:
        captured.update({"task": task, **kwargs})
        return {
            "ok": True,
            "status": "completed",
            "content": "明亚保险领航计划",
            "raw_response": {
                "segments": [
                    {
                        "start": 0,
                        "end": 2,
                        "text": "明亚保险领航计划",
                        "avg_logprob": -0.1,
                        "compression_ratio": 1.1,
                        "no_speech_prob": 0,
                    }
                ]
            },
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        fake_call,
    )
    result = execute_consented_model_task(
        consent["consent_path"], provider_config=PROVIDER, write=False
    )
    status = validate_model_connector_consent(
        consent["consent_path"], provider_config=PROVIDER, expected_calls=0
    )

    assert consent["instruction_transport"] == "audit_only"
    assert consent["asr_prompt_transport"] == "provider_audio_prompt"
    assert captured["prompt"] == "明亚保险 领航计划 明亚APP"
    assert "逐字转写" not in str(captured["prompt"])
    assert captured["max_retries"] == 0
    assert consent["scope"]["max_retries_per_call"] == 0
    assert status["scope"]["max_retries_per_call"] == 0
    assert result["status"] == "completed"
    assert result["production_qualified"] is True
    assert result["model_result"]["asr_quality"]["status"] == "passed"
    assert status["instructions"] == "逐字转写整段音频并输出时间戳。"
    assert status["asr_prompt"] == "明亚保险 领航计划 明亚APP"


def test_legacy_v2_asr_consent_without_asr_prompt_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    consent = create_model_connector_consent(
        tmp_path,
        task="cloud_asr",
        artifact_paths=[_audio(tmp_path)],
        provider_config=PROVIDER,
        instructions="这段旧任务说明绝不能成为 ASR prompt。",
        max_estimated_cost_usd=0.1,
        confirm_data_export=True,
    )
    captured: dict[str, object] = {}

    def fake_call(task: str, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "ok": True,
            "status": "completed",
            "content": "正常正文",
            "raw_response": {
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "text": "正常正文",
                        "avg_logprob": -0.1,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0,
                    }
                ]
            },
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        fake_call,
    )
    execute_consented_model_task(
        consent["consent_path"], provider_config=PROVIDER, write=False
    )

    assert captured["prompt"] == ""


def test_asr_instruction_leak_marks_connector_degraded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    consent = create_model_connector_consent(
        tmp_path,
        task="cloud_asr",
        artifact_paths=[_audio(tmp_path)],
        provider_config=PROVIDER,
        instructions="中间字段可用英文，识别正文必须为中文。",
        asr_prompt="明亚保险",
        max_estimated_cost_usd=0.1,
        confirm_data_export=True,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        lambda *args, **kwargs: {
            "ok": True,
            "status": "completed",
            "content": "中间字段可用英文字幕。",
            "raw_response": {
                "segments": [
                    {
                        "start": 608.446,
                        "end": 649.409,
                        "text": "中间字段可用英文字幕。",
                        "avg_logprob": -0.96464723,
                        "compression_ratio": 0.75,
                        "no_speech_prob": 0,
                    }
                ]
            },
        },
    )

    result = execute_consented_model_task(
        consent["consent_path"], provider_config=PROVIDER, write=False
    )

    assert result["ok"] is True
    assert result["status"] == "degraded"
    assert result["transport_ok"] is True
    assert result["quality_gate_passed"] is False
    assert result["production_qualified"] is False
    quality = result["model_result"]["asr_quality"]
    assert quality["failed_segment_count"] == 1
    assert quality["retry_plan"]["requires_new_exact_consent"] is True


def test_asr_prompt_is_rejected_for_non_asr_task(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    with pytest.raises(ValueError, match="ASR tasks"):
        create_model_connector_consent(
            tmp_path,
            task="smart_summary_rewrite",
            artifact_paths=[source],
            provider_config={
                "provider": "custom_openai_compatible",
                "base_url": "https://text.example/v1",
                "model": "text-test",
            },
            asr_prompt="not allowed",
            max_estimated_cost_usd=0.1,
            confirm_data_export=True,
        )


def test_vad_speech_without_transcript_builds_exact_gap_retry() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "id": 1,
                    "start": 0,
                    "end": 10,
                    "text": "第一段语音已经完整识别。",
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0,
                }
            ]
        },
        vad_intervals=[{"start": 0, "end": 10}, {"start": 20, "end": 30}],
        media_duration_seconds=35,
    )

    assert result["status"] == "degraded"
    assert result["quality_gate_passed"] is False
    assert result["failed_segment_count"] == 0
    assert result["coverage_gap_count"] == 1
    assert result["speech_coverage"]["coverage_ratio"] == 0.5
    assert result["response_issues"][-1]["key"] == "missing_speech_coverage"
    assert result["retry_plan"]["windows"] == [
        {
            "retry_id": "retry-0001",
            "source_segment_ids": ["vad-gap-0001"],
            "start": 18.5,
            "end": 31.5,
            "alignment_source": "vad_coverage_gap",
            "reasons": ["missing_speech_coverage"],
            "snippet_artifact_status": "not_created",
        }
    ]


def test_vad_speech_with_full_transcript_coverage_passes() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 10,
                    "text": "完整覆盖这一段语音内容。",
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0,
                }
            ]
        },
        vad_intervals=[{"start": 0, "end": 10}],
    )

    assert result["status"] == "passed"
    assert result["speech_coverage"]["status"] == "passed"
    assert result["coverage_gap_count"] == 0


def test_missing_verbose_segments_with_vad_can_retry_all_speech() -> None:
    result = assess_asr_response(
        {"text": "只有整段文本"},
        vad_intervals=[{"start": 5, "end": 12}],
        media_duration_seconds=20,
    )

    assert result["status"] == "failed"
    assert result["coverage_gap_count"] == 1
    assert [row["key"] for row in result["response_issues"]] == [
        "verbose_segments_missing",
        "missing_speech_coverage",
    ]
    assert result["retry_plan"]["windows"][0]["start"] == 3.5
    assert result["retry_plan"]["windows"][0]["end"] == 13.5


def test_word_timestamps_expose_internal_gap_hidden_by_segment_bounds() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 20,
                    "text": "前半段文本后半段文本",
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0,
                    "words": [
                        {"word": "前半段文本", "start": 0, "end": 5},
                        {"word": "后半段文本", "start": 15, "end": 20},
                    ],
                }
            ]
        },
        vad_intervals=[{"start": 0, "end": 20}],
        media_duration_seconds=20,
    )

    assert result["status"] == "degraded"
    assert result["speech_coverage"]["evidence"] == {
        "word_timestamp_segment_count": 1,
        "segment_bounds_fallback_count": 0,
        "incomplete_word_timestamp_segment_count": 0,
        "word_timestamp_preferred_when_complete": True,
    }
    assert result["speech_coverage"]["gaps"] == [
        {
            "segment_id": "vad-gap-0001",
            "position": 0,
            "start": 5.0,
            "end": 15.0,
            "duration_seconds": 10.0,
            "reasons": ["missing_speech_coverage"],
            "original_text": "",
            "preserve_original_text": True,
        }
    ]
    assert result["retry_plan"]["windows"][0]["start"] == 3.5
    assert result["retry_plan"]["windows"][0]["end"] == 16.5


def test_segment_bounds_remain_explicit_fallback_without_word_timestamps() -> None:
    result = assess_asr_response(
        {"segments": [{"start": 0, "end": 10, "text": "完整覆盖这一段语音内容。"}]},
        vad_intervals=[{"start": 0, "end": 10}],
    )

    assert result["status"] == "passed"
    assert result["speech_coverage"]["evidence"] == {
        "word_timestamp_segment_count": 0,
        "segment_bounds_fallback_count": 1,
        "incomplete_word_timestamp_segment_count": 0,
        "word_timestamp_preferred_when_complete": True,
    }

def test_incomplete_word_timestamps_fall_back_to_segment_bounds() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 10,
                    "text": "完整覆盖这一段语音内容。",
                    "words": [{"word": "完整", "start": 0, "end": 1}],
                }
            ]
        },
        vad_intervals=[{"start": 0, "end": 10}],
    )

    assert result["status"] == "passed"
    assert result["segments"][0]["coverage_evidence"] == "segment_bounds"
    assert (
        result["segments"][0]["coverage_evidence_reason"]
        == "word_timestamp_text_incomplete"
    )
    assert result["speech_coverage"]["evidence"] == {
        "word_timestamp_segment_count": 0,
        "segment_bounds_fallback_count": 1,
        "incomplete_word_timestamp_segment_count": 1,
        "word_timestamp_preferred_when_complete": True,
    }

def test_whisper_word_anomaly_creates_review_and_targeted_retry_candidate() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 3,
                    "text": "甲乙丙",
                    "words": [
                        {"word": "甲", "start": 0, "end": 1, "probability": 0.1},
                        {"word": "乙", "start": 1, "end": 2, "probability": 0.1},
                        {"word": "丙", "start": 2, "end": 3, "probability": 0.1},
                    ],
                }
            ]
        }
    )

    segment = result["segments"][0]
    assert result["status"] == "review_required"
    assert result["quality_gate_passed"] is False
    assert segment["word_anomaly_evidence"]["status"] == "anomaly"
    assert segment["word_anomaly_evidence"]["score"] == 3.0
    assert [row["key"] for row in segment["issues"]] == ["whisper_word_anomaly"]
    assert result["review_chunks"][0]["reasons"] == ["whisper_word_anomaly"]
    assert result["retry_plan"]["status"] == "authorization_required"
    assert result["retry_plan"]["windows"][0]["source_segment_ids"] == ["segment-0001"]
    assert result["quality_signal_sources"][0]["algorithm"] == (
        "word_anomaly_score/is_segment_anomaly"
    )


def test_whisper_word_anomaly_passes_well_timed_high_probability_words() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 3,
                    "text": "甲乙丙",
                    "words": [
                        {"word": "甲", "start": 0, "end": 1, "score": 0.9},
                        {"word": "乙", "start": 1, "end": 2, "score": 0.9},
                        {"word": "丙", "start": 2, "end": 3, "score": 0.9},
                    ],
                }
            ]
        }
    )

    assert result["status"] == "passed"
    assert result["quality_gate_passed"] is True
    assert result["segments"][0]["word_anomaly_evidence"]["status"] == "passed"
    assert result["segments"][0]["word_anomaly_evidence"]["score"] == 0.0


def test_whisper_word_anomaly_does_not_treat_missing_confidence_as_zero() -> None:
    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 3,
                    "text": "甲乙丙",
                    "words": [
                        {"word": "甲", "start": 0, "end": 1},
                        {"word": "乙", "start": 1, "end": 2},
                        {"word": "丙", "start": 2, "end": 3},
                    ],
                }
            ]
        }
    )

    evidence = result["segments"][0]["word_anomaly_evidence"]
    assert result["status"] == "passed"
    assert result["quality_gate_passed"] is True
    assert evidence["status"] == "not_evaluated"
    assert evidence["reason"] == "word_confidence_or_timing_incomplete"


def test_estimated_chunk_timing_does_not_create_false_density_retry() -> None:
    """Character-proportional navigation time is not measured speech time.

    Intent: lock the false-positive fix for text-only FunASR chunks.
    Decision: preserve density as diagnostic data but do not create a retry
    from density alone when the segment carries timing-estimation provenance.
    Reason: silence inside a fixed chunk would otherwise look like missing ASR.
    Evidence: the 2026-07-24 production bundle's review count fell from 320 to
    zero after applying its recorded chunk offsets.
    Effective scope: estimated-timing segments only.
    """

    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 300,
                    "text": "测试测试1234。",
                    "transformations": [
                        {
                            "type": "timing_estimation",
                            "method": "character_proportional_within_source_window",
                        }
                    ],
                }
            ]
        }
    )

    assert result["status"] == "passed"
    assert result["segments"][0]["timing_estimated"] is True
    assert result["segments"][0]["text_density_chars_per_second"] < 0.5
    assert result["segments"][0]["issues"] == []
    assert result["retry_plan"]["windows"] == []


def test_estimated_chunk_uses_vad_speech_seconds_to_detect_real_sparse_text() -> None:
    """A speech-heavy coarse chunk with almost no text remains reviewable."""

    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 300,
                    "text": "一",
                    "transformations": [
                        {
                            "type": "timing_estimation",
                            "method": "character_proportional_within_source_window",
                            "source_window_start": 0,
                            "source_window_end": 300,
                        }
                    ],
                }
            ]
        },
        vad_intervals=[{"start": 0, "end": 300}],
        media_duration_seconds=300,
    )

    coarse = result["coarse_timing_density"]
    assert result["status"] == "review_required"
    assert coarse["status"] == "review_required"
    assert coarse["review_window_count"] == 1
    assert coarse["windows"][0]["vad_speech_seconds"] == 300
    assert result["retry_plan"]["windows"][0]["reasons"] == [
        "coarse_timing_low_speech_text_density"
    ]


def test_estimated_chunk_vad_density_passes_when_text_matches_speech_duration() -> None:
    """Silence in the source window must not dilute actual speaking density."""

    result = assess_asr_response(
        {
            "segments": [
                {
                    "start": 0,
                    "end": 300,
                    "text": "测试测试1234。",
                    "transformations": [
                        {
                            "type": "timing_estimation",
                            "method": "character_proportional_within_source_window",
                            "source_window_start": 0,
                            "source_window_end": 300,
                        }
                    ],
                }
            ]
        },
        vad_intervals=[{"start": 24, "end": 30}],
        media_duration_seconds=300,
    )

    coarse = result["coarse_timing_density"]
    assert result["status"] == "passed"
    assert coarse["status"] == "passed"
    assert coarse["review_window_count"] == 0
    assert coarse["windows"][0]["vad_speech_seconds"] == 6
    assert coarse["windows"][0]["text_chars_per_vad_speech_second"] > 0.5
    assert result["retry_plan"]["windows"] == []
