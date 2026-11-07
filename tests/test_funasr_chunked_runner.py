from __future__ import annotations

import json
import subprocess
from video_knowledge_pipeline.asr_execution import _upgrade_funasr_command_to_chunked
from pathlib import Path

import video_knowledge_pipeline.funasr_chunked_runner as runner


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _chunked_media(tmp_path: Path) -> tuple[Path, list[Path]]:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"video")
    chunks = [tmp_path / "chunk-0000.wav", tmp_path / "chunk-0001.wav"]
    for path in chunks:
        path.write_bytes(b"audio")
    return media, chunks


def test_chunked_funasr_preserves_success_and_resumes_only_failed_chunk(tmp_path: Path, monkeypatch) -> None:
    media, chunks = _chunked_media(tmp_path)
    output = tmp_path / "raw-asr-output.json"
    monkeypatch.setattr(runner, "_audio_chunks", lambda *_args, **_kwargs: chunks)
    first_calls: list[list[str]] = []

    def first_run(command, **_kwargs):
        command = list(command)
        first_calls.append(command)
        source = Path(_option(command, "--input"))
        if source.name == "chunk-0001.wav":
            return subprocess.CompletedProcess(command, 1, "", "CUDA out of memory")
        Path(_option(command, "--output")).write_text(
            json.dumps({"result": [{"text": "第一段", "sentence_info": [{"text": "第一段", "start": 0, "end": 1000}]}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", first_run)
    first = runner.run_funasr_chunked(
        input_path=str(media), output_path=str(output), provider="sensevoice", model="iic/SenseVoiceSmall", chunk_seconds=300
    )

    assert first["status"] == "degraded"
    assert first["successful_chunk_indexes"] == [0]
    assert first["failed_chunks"][0]["chunk_index"] == 1
    assert "--chunk-indexes" in first["retry_commands"][0]["command"]
    assert len([command for command in first_calls if "--input" in command]) == 2

    second_calls: list[list[str]] = []

    def second_run(command, **_kwargs):
        command = list(command)
        second_calls.append(command)
        assert Path(_option(command, "--input")).name == "chunk-0001.wav"
        Path(_option(command, "--output")).write_text(
            json.dumps({"result": [{"text": "第二段", "sentence_info": [{"text": "第二段", "start": 0, "end": 1000}]}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", second_run)
    second = runner.run_funasr_chunked(
        input_path=str(media), output_path=str(output), provider="sensevoice", model="iic/SenseVoiceSmall", chunk_seconds=300
    )

    assert second["status"] == "completed"
    assert len([command for command in second_calls if "--input" in command]) == 1
    assert second["successful_chunk_indexes"] == [0, 1]
    assert second["result"][1]["sentence_info"][0]["start"] == 300000.0
    checkpoint = json.loads(Path(second["checkpoint_path"]).read_text(encoding="utf-8"))
    assert checkpoint["successful_chunk_indexes"] == [0, 1]
    assert checkpoint["schema"] == "video_knowledge_pipeline.funasr_chunked_asr_checkpoint.v2"
    assert len(checkpoint["execution_contract_revision"]) == 64

    changed_contract_calls: list[list[str]] = []

    def changed_contract_run(command, **_kwargs):
        command = list(command)
        changed_contract_calls.append(command)
        Path(_option(command, "--output")).write_text(
            json.dumps(
                {
                    "result": [
                        {
                            "text": "参数改变后的内容",
                            "sentence_info": [
                                {"text": "参数改变后的内容", "start": 0, "end": 1000}
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", changed_contract_run)
    changed = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="iic/SenseVoiceSmall",
        chunk_seconds=300,
        batch_size_s=30,
    )

    assert changed["status"] == "completed"
    assert changed["resumed_from_checkpoint"] is False
    assert len([command for command in changed_contract_calls if "--input" in command]) == 2
    assert changed["execution_contract_revision"] != checkpoint["execution_contract_revision"]


def test_chunked_funasr_maps_swapped_local_speakers_and_hides_centers(
    tmp_path: Path, monkeypatch
) -> None:
    media, chunks = _chunked_media(tmp_path)
    output = tmp_path / "raw-asr-output.json"
    monkeypatch.setattr(runner, "_audio_chunks", lambda *_args, **_kwargs: chunks)

    def fake_run(command, **_kwargs):
        command = list(command)
        source = Path(_option(command, "--input"))
        if source.name == "chunk-0000.wav":
            centers = [
                {"local_speaker_id": "0", "center": [1.0, 0.0]},
                {"local_speaker_id": "1", "center": [0.0, 1.0]},
            ]
            sentences = [
                {"text": "甲", "start": 0, "end": 1000, "spk": 0},
                {"text": "乙", "start": 1000, "end": 2000, "spk": 1},
            ]
        else:
            centers = [
                {"local_speaker_id": "0", "center": [0.01, 0.99]},
                {"local_speaker_id": "1", "center": [0.99, 0.01]},
            ]
            sentences = [
                {"text": "乙", "start": 0, "end": 1000, "spk": 0},
                {"text": "甲", "start": 1000, "end": 2000, "spk": 1},
            ]
        Path(_option(command, "--output")).write_text(
            json.dumps(
                {
                    "result": [
                        {
                            "text": "".join(row["text"] for row in sentences),
                            "sentence_info": sentences,
                            "_speaker_embedding_centers": centers,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="iic/SenseVoiceSmall",
        spk_model="cam++",
        chunk_seconds=300,
    )

    assert result["speaker_global_alignment"]["global_speaker_count"] == 2
    assert result["chunk_results"][0]["sentence_info"][0]["speaker_global_id"] == "speaker-global-001"
    assert result["chunk_results"][1]["sentence_info"][1]["speaker_global_id"] == "speaker-global-001"
    assert "_speaker_embedding_centers" not in json.dumps(result["chunk_results"])
    private_path = Path(result["speaker_global_alignment"]["artifacts"]["private_path"])
    assert private_path.exists()
    assert json.loads(private_path.read_text(encoding="utf-8"))["biometric_data"] is True
    checkpoint = json.loads(Path(result["checkpoint_path"]).read_text(encoding="utf-8"))
    assert "_speaker_embedding_centers" in json.dumps(checkpoint)


def test_rebuild_from_checkpoint_skips_media_probe_and_child_execution(
    tmp_path: Path, monkeypatch
) -> None:
    media, chunks = _chunked_media(tmp_path)
    output = tmp_path / "raw-asr-output.json"
    monkeypatch.setattr(runner, "_audio_chunks", lambda *_args, **_kwargs: chunks[:1])

    def first_run(command, **_kwargs):
        command = list(command)
        Path(_option(command, "--output")).write_text(
            json.dumps(
                {
                    "result": [
                        {
                            "text": "检查点正文",
                            "sentence_info": [
                                {
                                    "text": "检查点正文",
                                    "start": 0,
                                    "end": 1000,
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", first_run)
    first = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="iic/SenseVoiceSmall",
        chunk_seconds=300,
    )
    assert first["status"] == "completed"
    output.write_text("corrupted parent output", encoding="utf-8")
    media.unlink()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("checkpoint rebuild must not probe media or run a child")

    monkeypatch.setattr(runner, "_media_duration_seconds", forbidden)
    monkeypatch.setattr(runner, "_audio_chunks", forbidden)
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    rebuilt = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="iic/SenseVoiceSmall",
        chunk_seconds=300,
        rebuild_from_checkpoint=True,
    )

    assert rebuilt["status"] == "completed"
    assert rebuilt["resumed_from_checkpoint"] is True
    assert rebuilt["source_freshness"] == "not_revalidated_checkpoint_only"
    assert rebuilt["result"][0]["text"] == "检查点正文"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "completed"


def test_chunked_runner_keeps_existing_plan_parameters_for_each_child(tmp_path: Path, monkeypatch) -> None:
    media, chunks = _chunked_media(tmp_path)
    output = tmp_path / "raw-asr-output.json"
    monkeypatch.setattr(runner, "_audio_chunks", lambda *_args, **_kwargs: chunks[:1])
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = list(command)
        captured.append(command)
        Path(_option(command, "--output")).write_text(json.dumps({"result": [{"text": "内容"}]}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_funasr_chunked(
        input_path=str(media), output_path=str(output), provider="sensevoice", model="model", batch_size_s=10,
        vad_max_single_segment_time_ms=15000, device="cpu", chunk_seconds=120,
    )

    assert result["status"] == "completed"
    child = next(command for command in captured if "--batch-size-s" in command)
    assert _option(child, "--batch-size-s") == "10"
    assert _option(child, "--vad-max-single-segment-time-ms") == "15000"
    assert _option(child, "--device") == "cpu"

def test_existing_funasr_plan_is_upgraded_to_resumable_chunk_runner() -> None:
    command = ["python", "-m", "video_knowledge_pipeline.funasr_python_runner", "--device", "cuda"]
    upgraded = _upgrade_funasr_command_to_chunked(
        command,
        {"runner": "funasr_python", "preset": "sensevoice"},
    )

    assert "video_knowledge_pipeline.funasr_chunked_runner" in upgraded
    assert _option(upgraded, "--chunk-seconds") == "300"
    assert _option(upgraded, "--chunk-overlap-seconds") == "5"
    assert command[-1] == "cuda"

def test_untimed_funasr_chunks_preserve_absolute_media_offsets(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_adapter import read_asr_cues

    source = tmp_path / "raw-asr-output.json"
    source.write_text(
        json.dumps(
            {
                "provider": "sensevoice",
                "duration_seconds": 550.0,
                "chunk_seconds": 300,
                "result": [
                    {"key": "chunk-0000", "text": "first chunk", "chunk_offset_seconds": 0.0},
                    {"key": "chunk-0001", "text": "second chunk", "chunk_offset_seconds": 300.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    cues = read_asr_cues(source, provider="sensevoice")

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0.0, 300.0, "first chunk"),
        (300.0, 550.0, "second chunk"),
    ]
    assert all(
        cue.transformations
        == [
            {
                "type": "timing_estimation",
                "method": "character_proportional_within_source_window",
                "precision": "coarse",
                "source_window_start": cue.start if cue.start in {0.0, 300.0} else None,
                "source_window_end": 300.0 if cue.start == 0.0 else 550.0,
                "source_record_index": 0 if cue.start == 0.0 else 1,
            }
        ]
        for cue in cues
    )


def test_legacy_untimed_funasr_result_keeps_full_duration_behavior(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_adapter import read_asr_cues

    source = tmp_path / "legacy-raw-asr-output.json"
    source.write_text(
        json.dumps(
            {
                "provider": "sensevoice",
                "duration_seconds": 42.0,
                "result": [{"text": "legacy untimed result"}],
            }
        ),
        encoding="utf-8",
    )

    cues = read_asr_cues(source, provider="sensevoice")

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0.0, 42.0, "legacy untimed result"),
    ]


def test_chunk_runtime_summary_reads_only_child_telemetry(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "chunk-0000.json").write_text(
        json.dumps(
            {
                "runtime_metrics": {
                    "device": "cuda",
                    "elapsed_seconds": 12.5,
                    "cuda_peak_memory_allocated_mib": 2048.0,
                    "cuda_peak_memory_reserved_mib": 2304.0,
                },
                "result": [{"text": "must not be copied"}],
            }
        ),
        encoding="utf-8",
    )
    (chunks / "chunk-0001.json").write_text(
        json.dumps(
            {
                "runtime_metrics": {
                    "device": "cuda",
                    "elapsed_seconds": 10.0,
                    "cuda_peak_memory_allocated_mib": 2560.0,
                    "cuda_peak_memory_reserved_mib": 2816.0,
                }
            }
        ),
        encoding="utf-8",
    )

    summary = runner._chunk_runtime_summary(chunks, [0, 1, 2])

    assert summary["status"] == "available"
    assert summary["measured_chunk_count"] == 2
    assert summary["missing_chunk_indexes"] == [2]
    assert summary["total_child_elapsed_seconds"] == 22.5
    assert summary["max_cuda_peak_memory_allocated_mib"] == 2560.0
    assert summary["max_cuda_peak_memory_reserved_mib"] == 2816.0
    assert all("result" not in row for row in summary["chunks"])

def test_cli_summary_omits_full_transcript_payload() -> None:
    result = {
        "status": "completed",
        "ok": True,
        "usable": True,
        "quality_status": "completed",
        "output_path": "raw.json",
        "report_path": "report.json",
        "successful_chunk_count": 3,
        "failed_chunk_count": 0,
        "unresolved_chunk_indexes": [],
        "progress": {"progress_json": "progress.json"},
        "result": [{"text": "very large transcript"}],
        "chunk_results": [{"payload": "large"}],
    }

    summary = runner._cli_result_summary(result)

    assert summary["schema"] == "video_knowledge_pipeline.funasr_chunked_runner_result.v1"
    assert summary["successful_chunk_count"] == 3
    assert "result" not in summary
    assert "chunk_results" not in summary

def test_overlap_canonicalization_keeps_one_core_owner() -> None:
    manifest = {
        "strategy": {"overlap_seconds": 5.0},
        "chunks": [
            {
                "index": 0,
                "start_seconds": 0.0,
                "end_seconds": 45.0,
                "core_start_seconds": 0.0,
                "core_end_seconds": 40.0,
            },
            {
                "index": 1,
                "start_seconds": 35.0,
                "end_seconds": 80.0,
                "core_start_seconds": 40.0,
                "core_end_seconds": 80.0,
            },
        ],
    }
    records = [
        {
            "chunk_index": 0,
            "record_index": 0,
            "text": "第一段边界句",
            "sentence_info": [
                {"text": "第一段", "start": 0, "end": 39000},
                {"text": "边界句", "start": 39000, "end": 43000},
            ],
        },
        {
            "chunk_index": 1,
            "record_index": 0,
            "text": "边界句第二段",
            "sentence_info": [
                {"text": "边界句", "start": 39000, "end": 43000},
                {"text": "第二段", "start": 43000, "end": 80000},
            ],
        },
    ]

    canonical, report = runner._canonicalize_overlap_records(records, manifest)

    assert [row["text"] for row in canonical] == ["第一段", "边界句第二段"]
    assert report["status"] == "completed"
    assert report["excluded_padding_sentence_count"] == 1
    assert report["boundary_review_required_count"] == 0
    assert report["boundaries"][0]["local_agreement"]["common_prefix"] == "边界句"
    assert "chunk_index" not in canonical[0]


def test_timed_overlap_accepts_boundary_lcs_when_sentence_split_differs() -> None:
    manifest = {
        "strategy": {"overlap_seconds": 5.0},
        "chunks": [
            {
                "index": 0,
                "start_seconds": 0.0,
                "end_seconds": 45.0,
                "core_start_seconds": 0.0,
                "core_end_seconds": 40.0,
            },
            {
                "index": 1,
                "start_seconds": 35.0,
                "end_seconds": 80.0,
                "core_start_seconds": 40.0,
                "core_end_seconds": 80.0,
            },
        ],
    }
    records = [
        {
            "chunk_index": 0,
            "record_index": 0,
            "text": "前文。方案做好以后先约客户开腾讯会议。",
            "sentence_info": [
                {"text": "前文。", "start": 0, "end": 35000},
                {"text": "方案做好以后先约客户开腾讯会议。", "start": 35000, "end": 45000},
            ],
        },
        {
            "chunk_index": 1,
            "record_index": 0,
            "text": "先约客户开腾讯会议。接着讲解产品。",
            "sentence_info": [
                {"text": "先约客户开腾讯会议。", "start": 35000, "end": 39000},
                {"text": "接着讲解产品。", "start": 39000, "end": 80000},
            ],
        },
    ]

    canonical, report = runner._canonicalize_overlap_records(records, manifest)

    assert report["boundaries"][0]["local_agreement"]["agreement_over_shorter"] < 0.2
    assert report["boundaries"][0]["boundary_lcs"]["automatic_merge_allowed"] is True
    assert report["boundaries"][0]["requires_human_review"] is False
    assert report["status"] == "completed"
    assert [row["text"] for row in canonical] == ["前文。", "接着讲解产品。"]
def test_timed_overlap_crops_long_sentence_to_character_timestamp_owner() -> None:
    manifest = {
        "strategy": {"overlap_seconds": 5.0},
        "chunks": [
            {
                "index": 0,
                "start_seconds": 0.0,
                "end_seconds": 45.0,
                "core_start_seconds": 0.0,
                "core_end_seconds": 40.0,
            },
            {
                "index": 1,
                "start_seconds": 35.0,
                "end_seconds": 80.0,
                "core_start_seconds": 40.0,
                "core_end_seconds": 80.0,
            },
        ],
    }
    records = [
        {
            "chunk_index": 0,
            "record_index": 0,
            "text": "重复边界",
            "sentence_info": [
                {
                    "text": "重复边界",
                    "start": 38000,
                    "end": 42000,
                    "timestamp": [
                        [38000, 39000],
                        [39000, 40000],
                        [40000, 41000],
                        [41000, 42000],
                    ],
                }
            ],
        },
        {
            "chunk_index": 1,
            "record_index": 0,
            "text": "重复边界后文",
            "sentence_info": [
                {
                    "text": "重复边界后文",
                    "sentence": "重复边界后文",
                    "start": 38000,
                    "end": 44000,
                    "timestamp": [
                        [38000, 39000],
                        [39000, 40000],
                        [40000, 41000],
                        [41000, 42000],
                        [42000, 43000],
                        [43000, 44000],
                    ],
                }
            ],
        },
    ]

    canonical, report = runner._canonicalize_overlap_records(records, manifest)

    assert [row["text"] for row in canonical] == ["重复", "边界后文"]
    assert canonical[0]["sentence_info"][0]["end"] == 40000.0
    assert canonical[1]["sentence_info"][0]["start"] == 40000.0
    assert canonical[1]["sentence_info"][0]["sentence"] == "边界后文"
    assert report["excluded_padding_sentence_count"] == 2


def test_untimed_overlap_uses_boundary_lcs_and_preserves_raw_records() -> None:
    manifest = {
        "strategy": {"overlap_seconds": 5.0},
        "chunks": [
            {
                "index": 0,
                "start_seconds": 0.0,
                "end_seconds": 45.0,
                "core_start_seconds": 0.0,
                "core_end_seconds": 40.0,
            },
            {
                "index": 1,
                "start_seconds": 35.0,
                "end_seconds": 80.0,
                "core_start_seconds": 40.0,
                "core_end_seconds": 80.0,
            },
        ],
    }
    records = [
        {
            "chunk_index": 0,
            "record_index": 0,
            "text": "前文。方案做好以后先约客户开腾讯会议。",
        },
        {
            "chunk_index": 1,
            "record_index": 0,
            "text": "先约客户开腾讯会议。接着讲解产品。",
        },
    ]

    canonical, report = runner._canonicalize_overlap_records(records, manifest)

    assert [row["text"] for row in canonical] == [
        "前文。方案做好以后先约客户开腾讯会议。",
        "接着讲解产品。",
    ]
    assert records[1]["text"] == "先约客户开腾讯会议。接着讲解产品。"
    assert report["status"] == "completed"
    assert report["untimed_deduplicated_boundary_count"] == 1
    assert report["boundaries"][0]["boundary_lcs"]["automatic_merge_allowed"] is True
    assert canonical[1]["overlap_deduplication"]["raw_chunk_preserved"] is True


def test_untimed_overlap_without_boundary_match_requires_review() -> None:
    manifest = {
        "strategy": {"overlap_seconds": 5.0},
        "chunks": [
            {
                "index": 0,
                "start_seconds": 0.0,
                "end_seconds": 45.0,
                "core_start_seconds": 0.0,
                "core_end_seconds": 40.0,
            },
            {
                "index": 1,
                "start_seconds": 35.0,
                "end_seconds": 80.0,
                "core_start_seconds": 40.0,
                "core_end_seconds": 80.0,
            },
        ],
    }

    canonical, report = runner._canonicalize_overlap_records(
        [
            {"chunk_index": 0, "record_index": 0, "text": "保险方案设计原则"},
            {"chunk_index": 1, "record_index": 0, "text": "客户服务案例开始"},
        ],
        manifest,
    )

    assert [row["text"] for row in canonical] == [
        "保险方案设计原则",
        "客户服务案例开始",
    ]
    assert report["status"] == "review_required"
    assert report["untimed_deduplicated_boundary_count"] == 0
    assert report["boundary_review_required_count"] == 1


def test_single_untimed_chunk_has_no_overlap_boundary_to_review() -> None:
    manifest = {
        "strategy": {"overlap_seconds": 5.0},
        "chunks": [
            {
                "index": 0,
                "start_seconds": 0.0,
                "end_seconds": 45.0,
                "core_start_seconds": 0.0,
                "core_end_seconds": 40.0,
            }
        ],
    }

    canonical, report = runner._canonicalize_overlap_records(
        [{"chunk_index": 0, "record_index": 0, "text": "无时间戳正文"}],
        manifest,
    )

    assert canonical[0]["text"] == "无时间戳正文"
    assert report["status"] == "completed"
    assert report["untimed_record_count"] == 1
    assert report["untimed_deduplicated_boundary_count"] == 0
