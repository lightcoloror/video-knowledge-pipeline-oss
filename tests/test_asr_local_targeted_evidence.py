from __future__ import annotations

import hashlib
from pathlib import Path

from video_knowledge_pipeline.asr_local_targeted_evidence import (
    build_local_targeted_asr_evidence,
)
from video_knowledge_pipeline.cli import main
from video_knowledge_pipeline.storage import read_json, write_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_chunked_sensevoice_reuses_provider_timing_and_clip_identity(tmp_path):
    manifests,outputs=_fixture(tmp_path)
    original=read_json(outputs[0])
    write_json(outputs[0],{'schema':'video_knowledge_funasr_chunked_raw_output.v1',
        'provider':'sensevoice','model':'iic/SenseVoiceSmall','device':'cuda:0',
        'input':original['input_path'],'duration_seconds':4,'status':'completed','ok':True,
        'canonical_complete':True,'failed_chunk_count':0,'unresolved_chunk_indexes':[],
        'result':[{'text':'保留原文','sentence_info':[{'text':'保留原文','start':1000,'end':2000}]}]})
    result=build_local_targeted_asr_evidence(manifests,outputs)
    assert result['ok'] is True
    row=result['segments'][0]
    assert (row['start'],row['end'],row['text'])==(11,12,'保留原文')


def test_failed_funasr_text_cannot_be_relabelled_completed(tmp_path):
    manifests,outputs=_fixture(tmp_path)
    original=read_json(outputs[0])
    write_json(outputs[0],{'schema':'video_knowledge_funasr_raw_output.v1',
        'provider':'sensevoice','model':'iic/SenseVoiceSmall','device':'cuda:0',
        'input':original['input_path'],'duration_seconds':4,'status':'failed','ok':False,
        'result':[{'text':'残留的旧文本'}]})
    result=build_local_targeted_asr_evidence(manifests,outputs)
    assert result['ok'] is False
    assert result['failed_output_count']==1


def _fixture(tmp_path: Path, *, device: str = "cuda:0") -> tuple[list[Path], list[Path]]:
    manifests: list[Path] = []
    outputs: list[Path] = []
    for index, (start, end, text) in enumerate(
        ((10.0, 14.0, "个性化需求"), (30.0, 35.0, "腾讯会议")),
        start=1,
    ):
        clip = tmp_path / f"clip-{index}.wav"
        clip.write_bytes(f"audio-{index}".encode())
        manifest = tmp_path / f"manifest-{index}.json"
        write_json(
            manifest,
            {
                "schema": "video_knowledge_pipeline.asr_retry_snippets.v1",
                "artifacts": [
                    {
                        "retry_id": f"retry-{index:04d}",
                        "source_segment_ids": [index + 10],
                        "start": start,
                        "end": end,
                        "duration_seconds": end - start,
                        "path": str(clip),
                        "status": "completed",
                        "sha256": _sha256(clip),
                        "bytes": clip.stat().st_size,
                    }
                ],
            },
        )
        raw = tmp_path / f"raw-{index}.json"
        write_json(
            raw,
            {
                "schema": "video_knowledge_pipeline.qwen3_asr_raw_output.v1",
                "provider": "qwen3-asr",
                "model": "C:/models/Qwen3-ASR-1.7B",
                "device": device,
                "input_path": str(clip),
                "ok": True,
                "usable": True,
                "status": "completed",
                "failed_chunk_count": 0,
                "segments": [
                    {
                        "segment_id": f"local-{index}",
                        "start": 0.5,
                        "end": end - start - 0.25,
                        "text": text,
                    }
                ],
            },
        )
        manifests.append(manifest)
        outputs.append(raw)
    return manifests, outputs


def test_build_local_targeted_asr_evidence_shifts_verified_gpu_timestamps(
    tmp_path: Path,
) -> None:
    manifests, outputs = _fixture(tmp_path)
    destination = tmp_path / "targeted-evidence.json"

    result = build_local_targeted_asr_evidence(
        manifests,
        outputs,
        output_json=destination,
        write=True,
    )

    assert result["status"] == "completed"
    assert result["ok"] is True
    assert result["candidate_only"] is True
    assert result["completed_window_count"] == 2
    assert result["missing_window_count"] == 0
    assert result["failed_output_count"] == 0
    assert result["segments"][0]["start"] == 10.5
    assert result["segments"][0]["end"] == 13.75
    assert result["segments"][1]["start"] == 30.5
    assert result["segments"][1]["end"] == 34.75
    assert all(row["device"] == "cuda:0" for row in result["segments"])
    assert result["operator_boundary"]["canonical_transcript_modified"] is False
    assert destination.is_file()
    assert read_json(destination)["segments"][1]["text"] == "腾讯会议"


def test_coarse_qwen_window_precision_survives_global_time_projection(tmp_path):
    manifests, outputs = _fixture(tmp_path)
    raw = read_json(outputs[0])
    raw['segments'][0]['timestamp_precision'] = 'audio_window_only'
    write_json(outputs[0], raw)
    result = build_local_targeted_asr_evidence(manifests, outputs, write=False)
    assert result['segments'][0]['timestamp_precision'] == 'audio_window_only'
    assert result['segments'][0]['start'] == 10.5


def test_build_local_targeted_asr_evidence_rejects_cpu_when_gpu_required(
    tmp_path: Path,
) -> None:
    manifests, outputs = _fixture(tmp_path, device="cpu")

    result = build_local_targeted_asr_evidence(manifests, outputs)

    assert result["status"] == "failed"
    assert result["ok"] is False
    assert result["failed_output_count"] == 2
    assert result["missing_window_count"] == 2
    assert "did not run on CUDA" in result["failed_outputs"][0]["error"]


def test_local_targeted_asr_evidence_cli_writes_candidate_pack(
    tmp_path: Path,
    capsys,
) -> None:
    manifests, outputs = _fixture(tmp_path)
    destination = tmp_path / "cli-targeted-evidence.json"
    argv = ["asr-local-targeted-evidence", str(destination)]
    for manifest in manifests:
        argv.extend(["--snippet-manifest", str(manifest)])
    for output in outputs:
        argv.extend(["--raw-output", str(output)])
    argv.append("--write")

    exit_code = main(argv)

    assert exit_code == 0
    assert destination.is_file()
    assert read_json(destination)["status"] == "completed"
    assert '"candidate_only": true' in capsys.readouterr().out
