from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.asr_adapter import read_asr_cues, read_asr_segment_dicts
from video_knowledge_pipeline.asr_runner import _model_ready
from video_knowledge_pipeline.qwen3_asr_python_runner import _language, _torch_dtype


def test_qwen3_asr_raw_output_is_detected_and_normalized(tmp_path: Path) -> None:
    source = tmp_path / "qwen3.json"
    source.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.qwen3_asr_raw_output.v1",
                "provider": "qwen3-asr",
                "segments": [
                    {
                        "start": 12.25,
                        "end": 18.5,
                        "text": "使用 Qwen3-ASR 生成独立识别假设。",
                        "words": [{"start": 12.25, "end": 12.8, "text": "使用"}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cues = read_asr_cues(source)

    assert len(cues) == 1
    assert cues[0].start == 12.25
    assert cues[0].end == 18.5
    assert "Qwen3-ASR" in cues[0].text

def test_qwen3_asr_explicit_cpu_bfloat16_dtype() -> None:
    class TorchStub:
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"

    assert _torch_dtype("bfloat16", "cpu", TorchStub) == "bfloat16"
    assert _torch_dtype("auto", "cpu", TorchStub) == "float32"
    assert _torch_dtype("auto", "cuda", TorchStub) == "bfloat16"


def test_qwen3_asr_maps_vkp_yue_alias_to_canonical_cantonese() -> None:
    assert _language("yue") == "Cantonese"
    assert _language("zh-yue") == "Cantonese"
    assert _language("Cantonese") == "Cantonese"


def test_qwen3_asr_readiness_rejects_index_without_weight_shards(tmp_path: Path) -> None:
    snapshot = tmp_path / "incomplete"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"encoder": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )

    result = _model_ready(preset="qwen3-asr-1.7b", model=str(snapshot))

    assert result["ready"] is False
    assert result["status"] == "incomplete_cache"
    assert result["incomplete_cache_matches"][0]["status"] == "incomplete_weight_shards"


def test_qwen3_asr_readiness_accepts_complete_indexed_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "complete"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    shard = snapshot / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"offline-fixture-weight")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"encoder": shard.name}}),
        encoding="utf-8",
    )

    result = _model_ready(preset="qwen3-asr-1.7b", model=str(snapshot))

    assert result["ready"] is True
    assert result["cache_matches"] == [str(snapshot.resolve())]


def test_qwen3_chunk_results_use_forced_alignment_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "qwen3-chunks.json"
    source.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.qwen3_asr_raw_output.v1",
                "provider": "qwen3-asr",
                "results": [
                    {
                        "chunk_index": 0,
                        "chunk_offset_seconds": 0.0,
                        "text": "欢迎来到课程。第二句话。",
                        "timestamps": [
                            {"text": "欢", "start": 0.4, "end": 0.5},
                            {"text": "迎", "start": 0.5, "end": 0.7},
                            {"text": "来", "start": 2.0, "end": 2.1},
                            {"text": "到", "start": 2.1, "end": 2.2},
                        ],
                    },
                    {
                        "chunk_index": 1,
                        "chunk_offset_seconds": 300.0,
                        "text": "后半段",
                        "timestamps": [
                            {"text": "后", "start": 300.2, "end": 300.3},
                            {"text": "半", "start": 300.3, "end": 300.4},
                            {"text": "段", "start": 300.4, "end": 300.5},
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cues = read_asr_cues(source)
    segments = read_asr_segment_dicts(source)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0.4, 0.7, "欢迎"),
        (2.0, 2.2, "来到"),
        (300.2, 300.5, "后半段"),
    ]
    assert segments[0]["metadata"]["alignment"] == "word_level"
    assert segments[0]["metadata"]["word_count"] == 2
