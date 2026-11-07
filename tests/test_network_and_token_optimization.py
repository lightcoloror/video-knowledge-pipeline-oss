from __future__ import annotations

import io
import json
from pathlib import Path

from video_knowledge_pipeline.cloud_asr import prepare_cloud_asr_audio
from video_knowledge_pipeline.model_runtime_client import _read_gateway_response
from video_knowledge_pipeline.transcript_semantic_correction import (
    PACK_SCHEMA,
    _build_transcript_semantic_gateway_pack,
)
from video_knowledge_pipeline.trusted_model_connector import (
    EXECUTION_RECEIPT_SCHEMA,
    compact_model_execution_receipt,
)


class _Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "application/json"}


def test_gateway_response_reports_exact_loopback_payload_bytes() -> None:
    raw = json.dumps({"ok": True, "content": "结果"}, ensure_ascii=False).encode("utf-8")
    payload, headers, status, payload_bytes = _read_gateway_response(
        _Response(raw), streamed=False
    )

    assert payload["content"] == "结果"
    assert headers["Content-Type"] == "application/json"
    assert status == 200
    assert payload_bytes == len(raw)


def test_compact_execution_receipt_omits_content_and_sums_network_bytes(
    tmp_path: Path,
) -> None:
    report = tmp_path / "connector-execution.json"
    report.write_text('{"large":"local only"}', encoding="utf-8")
    result = {
        "ok": True,
        "status": "completed",
        "execution_id": "run-1",
        "task": "online_ocr",
        "consent_id": "consent-1",
        "route": {
            "route_id": "route-1",
            "route_revision": "rev-1",
            "virtual_model": "vkp-remote-ocr-test",
        },
        "model_result": {
            "content": "x" * 10000,
            "calls": [
                {
                    "network_accounting": {
                        "gateway_request_bytes": 120,
                        "gateway_response_bytes": 40,
                        "source_artifact_bytes": 90,
                    }
                },
                {
                    "network_accounting": {
                        "gateway_request_bytes": 80,
                        "gateway_response_bytes": 20,
                        "source_artifact_bytes": 60,
                    }
                },
            ],
        },
        "artifacts": {"execution_report": str(report)},
    }

    receipt = compact_model_execution_receipt(result)

    assert receipt["schema"] == EXECUTION_RECEIPT_SCHEMA
    assert receipt["content_returned_over_mcp"] is False
    assert "content" not in receipt
    assert receipt["network_accounting"] == {
        "scope": "vkp_to_loopback_gateway_payload",
        "call_count": 2,
        "gateway_request_bytes": 200,
        "gateway_response_bytes": 60,
        "source_artifact_bytes": 150,
        "provider_wire_bytes_exact": False,
    }
    assert receipt["execution_report_bytes"] == report.stat().st_size
    assert len(json.dumps(receipt, ensure_ascii=False)) < receipt["full_result_json_bytes"]


def test_semantic_gateway_pack_reuses_prioritised_compaction(tmp_path: Path) -> None:
    source_pack = tmp_path / "transcript-semantic-correction-pack.json"
    candidate = {
        "candidate_id": "c-1",
        "segment_index": 7,
        "start": 10.0,
        "end": 12.0,
        "time_range": "00:00:10.000 - 00:00:12.000",
        "timeline_indexes": [2],
        "correction_type": "term",
        "original_text": "cell",
        "suggested_text": "Excel",
        "context_text": "上下文" * 2000,
        "evidence_ids": [f"e-{index}" for index in range(8)],
        "evidence": [
            {"evidence_id": f"e-{index}", "source_type": "ocr", "text": "证据" * 500}
            for index in range(8)
        ],
        "unneeded_full_payload": "z" * 20000,
    }
    source = {"schema": PACK_SCHEMA, "title": "test", "candidates": [candidate]}
    source_pack.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    compact = _build_transcript_semantic_gateway_pack(
        root=tmp_path,
        pack_path=source_pack,
        pack=source,
        candidates=[candidate],
        selection_summary={"strategy": "source_conflict_first"},
    )

    row = compact["candidates"][0]
    assert compact["schema"] == PACK_SCHEMA
    assert compact["pack_profile"] == "gateway_compact_v1"
    assert row["segment_index"] == 7
    assert len(row["context_text"]) == 800
    assert len(row["evidence"]) == 5
    assert "unneeded_full_payload" not in row
    assert len(json.dumps(compact, ensure_ascii=False)) < source_pack.stat().st_size


def test_cloud_asr_audio_candidate_is_local_preview(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"fixture-media")

    result = prepare_cloud_asr_audio(media, execute=False)

    assert result["status"] == "planned"
    assert result["network_call"] is False
    assert result["bitrate_kbps"] == 32
    assert result["sample_rate_hz"] == 16000
    assert result["channels"] == 1
    assert result["command"][-3:-1] == ["-b:a", "32k"]
    assert not Path(result["output_path"]).exists()


def test_cloud_asr_audio_candidate_attaches_ffmpeg_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    media = tmp_path / "source.wav"
    media.write_bytes(b"source-audio")
    target = tmp_path / "candidate.mp3"
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"ffmpeg")
    bundle = tmp_path / "bundle"

    class Completed:
        returncode = 0
        stderr = ""

    def fake_run(command, **kwargs):
        assert command[0] == str(ffmpeg)
        target.write_bytes(b"candidate-audio")
        return Completed()

    monkeypatch.setattr(
        "video_knowledge_pipeline.cloud_asr.resolve_media_tool",
        lambda name: str(ffmpeg) if name == "ffmpeg" else None,
    )
    monkeypatch.setattr("video_knowledge_pipeline.cloud_asr.subprocess.run", fake_run)

    result = prepare_cloud_asr_audio(
        media,
        output_path=target,
        execute=True,
        receipt_bundle_dir=bundle,
    )

    receipt_path = (
        bundle
        / "exports/media-execution/ffmpeg-cloud-asr-audio-execution-receipt.json"
    )
    assert receipt_path.is_file()
    assert result["ffmpeg_execution_receipt"]["path"] == str(receipt_path.resolve())
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["operation"] == "transcode"
    assert receipt["outputs"][0]["role"] == "cloud_asr_audio_candidate"
    assert receipt["execution_boundary"]["second_ffmpeg_orchestrator_created"] is False
