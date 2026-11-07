from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_adapter import normalize_asr_output


def _connector_execution(*, transport_ok: bool = True) -> dict:
    return {
        "schema": "video_knowledge_pipeline.trusted_model_connector.v1",
        "task": "cloud_asr",
        "ok": True,
        "status": "review_required",
        "transport_ok": transport_ok,
        "contract_ok": True,
        "quality_gate_passed": False,
        "production_qualified": False,
        "model_result": {
            "model_type": "asr",
            "runtime_result": {
                "consent_id": "consent-1",
                "route_id": "route-asr",
                "route_revision": "revision-1",
                "provider": "groq",
                "deployment": "whisper-large-v3-turbo",
                "raw_output": {
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 1.2, "text": "第一段"},
                        {"id": 2, "start": 1.2, "end": 2.4, "text": ""},
                        {"id": 3, "start": 2.4, "end": 3.6, "text": "第三段"},
                    ]
                },
                "asr_quality": {
                    "status": "review_required",
                    "quality_gate_passed": False,
                    "segment_count": 3,
                    "passed_segment_count": 2,
                    "review_segment_count": 1,
                    "failed_segment_count": 0,
                    "retry_plan": {
                        "status": "not_needed",
                        "requires_new_exact_consent": False,
                    },
                },
            },
        },
    }


def test_normalize_trusted_connector_asr_execution_preserves_review_provenance(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "connector-execution.json"
    raw.write_text(
        json.dumps(_connector_execution(), ensure_ascii=False),
        encoding="utf-8",
    )

    result = normalize_asr_output(
        tmp_path / "workspace",
        raw,
        provider="openai",
        title="测试",
    )

    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert result["segment_count"] == 3
    assert [segment["text"] for segment in payload["segments"]] == [
        "第一段",
        "",
        "第三段",
    ]
    assert payload["segments"][1]["segment_id"] == "2"
    assert payload["segments"][0]["segment_id"] == "0"
    assert payload["segments"][1]["start"] == 1.2
    assert payload["segments"][1]["end"] == 2.4
    assert payload["segments"][1]["empty_text_preserved"] is True
    assert payload["source_execution"]["status"] == "review_required"
    assert payload["source_execution"]["consent_id"] == "consent-1"
    assert payload["source_execution"]["asr_quality"] == {
        "status": "review_required",
        "segment_count": 3,
        "passed_segment_count": 2,
        "review_segment_count": 1,
        "failed_segment_count": 0,
        "retry_status": "not_needed",
        "requires_new_exact_consent": False,
        "review_chunks": [],
        "failed_chunks": [],
    }


def test_normalize_trusted_connector_asr_execution_rejects_failed_transport(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "connector-execution.json"
    raw.write_text(
        json.dumps(
            _connector_execution(transport_ok=False),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="transport validation failed"):
        normalize_asr_output(
            tmp_path / "workspace",
            raw,
            provider="openai",
            title="测试",
        )

def test_normalize_word_only_asr_reuses_aligned_word_segmentation(
    tmp_path: Path,
) -> None:
    payload = _connector_execution()
    payload["model_result"]["runtime_result"]["raw_output"] = {
        "text": "前半段后半段",
        "words": [
            {"word": "前半段", "start": 0.0, "end": 1.0},
            {"word": "后半段", "start": 2.0, "end": 3.0},
        ],
    }
    raw = tmp_path / "word-only-connector-execution.json"
    raw.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = normalize_asr_output(
        tmp_path / "workspace",
        raw,
        provider="openai",
        title="词级测试",
    )

    normalized = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert result["segment_count"] == 2
    assert [row["text"] for row in normalized["segments"]] == ["前半段", "后半段"]
    assert normalized["segments"][0]["metadata"]["alignment"] == "word_level"
    assert normalized["segments"][0]["metadata"]["words"] == [
        {"word": "前半段", "start": 0.0, "end": 1.0}
    ]
