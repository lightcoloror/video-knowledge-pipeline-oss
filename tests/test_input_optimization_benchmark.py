from __future__ import annotations

import json
import sys
from pathlib import Path

from video_knowledge_pipeline.input_optimization_benchmark import (
    combine_input_optimization_benchmarks,
    compare_asr_input_optimization,
    compare_semantic_input_optimization,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return path


def _report(
    *,
    task: str,
    model: str,
    uploaded_bytes: int,
    content: str = "",
    prompt_tokens: int | None = None,
    quality_gate_passed: bool = True,
    asr_quality: dict | None = None,
) -> dict:
    usage = {"prompt_tokens": prompt_tokens} if prompt_tokens is not None else {}
    return {
        "ok": True,
        "status": "completed",
        "task": task,
        "contract_ok": True,
        "quality_gate_passed": quality_gate_passed,
        "route": {
            "route_id": f"route-{task}",
            "route_revision": "revision-1",
            "deployments": [{"provider": "fixture", "model": model}],
        },
        "upload_manifest": {"total_bytes": uploaded_bytes},
        "model_result": {
            "content": content,
            "asr_quality": asr_quality or {},
            "runtime_result": {"status": "completed", "latency_ms": 10, "usage": usage},
        },
    }


def test_asr_comparison_reuses_stability_gate_and_does_not_promote(
    tmp_path: Path,
) -> None:
    baseline_audio = tmp_path / "64k.mp3"
    optimized_audio = tmp_path / "32k.mp3"
    baseline_audio.write_bytes(b"a" * 1000)
    optimized_audio.write_bytes(b"a" * 500)
    transcript = {
        "segments": [
            {"id": "1", "start": 0, "end": 2, "text": "Mingya insurance plan"},
            {"id": "2", "start": 2, "end": 4, "text": "customer needs analysis"},
        ]
    }
    baseline_transcript = _write(tmp_path / "baseline.json", transcript)
    optimized_transcript = _write(tmp_path / "optimized.json", transcript)
    quality = {
        "status": "passed",
        "segment_count": 2,
        "passed_segment_count": 2,
        "review_segment_count": 0,
        "failed_segment_count": 0,
    }
    baseline_report = _write(
        tmp_path / "baseline-report.json",
        _report(
            task="cloud_asr", model="whisper", uploaded_bytes=1000, asr_quality=quality
        ),
    )
    optimized_report = _write(
        tmp_path / "optimized-report.json",
        _report(
            task="cloud_asr", model="whisper", uploaded_bytes=500, asr_quality=quality
        ),
    )
    result = compare_asr_input_optimization(
        baseline_audio_path=baseline_audio,
        optimized_audio_path=optimized_audio,
        baseline_transcript_path=baseline_transcript,
        optimized_transcript_path=optimized_transcript,
        baseline_execution_report_path=baseline_report,
        optimized_execution_report_path=optimized_report,
        critical_terms=["Mingya", "customer"],
    )
    assert result["status"] == "passed"
    assert result["input_bytes"]["reduction_ratio"] == 0.5
    assert result["transcript_promoted"] is False
    assert result["provider_calls_made"] == 0


def test_semantic_comparison_uses_manifest_and_provider_usage(tmp_path: Path) -> None:
    candidate = {
        "candidate_id": "c-1",
        "original_text": "cell",
        "evidence_ids": ["asr-1", "ocr-1"],
    }
    pack = _write(
        tmp_path / "compact.json",
        {
            "candidates": [candidate],
            "candidate_selection": {"deferred_candidate_count": 4},
        },
    )
    decision = {
        "candidate_id": "c-1",
        "action": "replace",
        "original_text": "cell",
        "corrected_text": "Excel",
        "evidence_ids": ["ocr-1"],
    }
    content = json.dumps({"decisions": [decision]})
    baseline = _write(
        tmp_path / "baseline-report.json",
        _report(
            task="transcript_semantic_correction",
            model="gemini",
            uploaded_bytes=1000,
            content=content,
            prompt_tokens=1000,
            quality_gate_passed=False,
        ),
    )
    optimized = _write(
        tmp_path / "optimized-report.json",
        _report(
            task="transcript_semantic_correction",
            model="gemini",
            uploaded_bytes=300,
            content=content,
            prompt_tokens=250,
        ),
    )
    result = compare_semantic_input_optimization(
        optimized_pack_path=pack,
        baseline_execution_report_path=baseline,
        optimized_execution_report_path=optimized,
    )
    assert result["status"] == "passed"
    assert result["input_bytes"]["baseline"] == 1000
    assert result["prompt_tokens"]["reduction_ratio"] == 0.75
    assert result["candidate_accounting"]["deferred_low_evidence"] == 4
    assert result["corrections_applied"] is False


def test_semantic_comparison_rejects_mutated_candidates(tmp_path: Path) -> None:
    pack = _write(
        tmp_path / "compact.json",
        {
            "candidates": [
                {
                    "candidate_id": "c-1",
                    "original_text": "original",
                    "evidence_ids": ["e-1"],
                }
            ]
        },
    )
    bad_content = json.dumps(
        {
            "decisions": [
                {
                    "candidate_id": "c-1",
                    "original_text": "mutated",
                    "evidence_ids": ["invented"],
                }
            ]
        }
    )
    baseline = _write(
        tmp_path / "baseline.json",
        _report(
            task="transcript_semantic_correction",
            model="gemini",
            uploaded_bytes=1000,
            prompt_tokens=1000,
        ),
    )
    optimized = _write(
        tmp_path / "optimized.json",
        _report(
            task="transcript_semantic_correction",
            model="gemini",
            uploaded_bytes=300,
            content=bad_content,
            prompt_tokens=250,
        ),
    )
    result = compare_semantic_input_optimization(
        optimized_pack_path=pack,
        baseline_execution_report_path=baseline,
        optimized_execution_report_path=optimized,
    )
    assert result["status"] == "failed"
    assert result["gates"]["original_text_preserved"] is False
    assert result["gates"]["evidence_ids_bound_to_pack"] is False


def test_combined_result_requires_both_benchmarks() -> None:
    result = combine_input_optimization_benchmarks(
        {"status": "passed"}, {"status": "failed"}
    )
    assert result["status"] == "failed"
    assert result["production_recommendation"] == "keep_existing_defaults"


def test_final_cli_reads_report_paths_and_writes_failed_summary(tmp_path) -> None:
    from video_knowledge_pipeline import input_optimization_benchmark

    asr_path = tmp_path / "asr.json"
    semantic_path = tmp_path / "semantic.json"
    output_path = tmp_path / "final.json"
    asr_path.write_text('{"status":"failed"}', encoding="utf-8")
    semantic_path.write_text('{"status":"passed"}', encoding="utf-8")

    exit_code = input_optimization_benchmark.main(
        ["final", str(asr_path), str(semantic_path), str(output_path)]
    )

    assert exit_code == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["results"]["asr"]["status"] == "failed"
    assert payload["results"]["semantic"]["status"] == "passed"


def test_consent_status_without_provider_config_does_not_invent_default(
    monkeypatch,
) -> None:
    from video_knowledge_pipeline import model_connector_consent

    captured = {}

    def fake_validate(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return {"status": "active"}

    monkeypatch.setattr(
        model_connector_consent, "validate_model_connector_consent", fake_validate
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["model_connector_consent", "status", "pending-consent.json"],
    )

    assert model_connector_consent.main() == 0
    assert captured["provider_config"] is None
    assert captured["route_snapshot"] is None
