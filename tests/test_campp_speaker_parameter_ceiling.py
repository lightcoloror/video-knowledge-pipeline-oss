from __future__ import annotations

import sys
from pathlib import Path

from video_knowledge_pipeline import (
    asr_ab_compare,
    funasr_chunked_runner,
    funasr_python_runner,
)
from video_knowledge_pipeline.asr_ab_plan import plan_asr_ab_sample


def test_funasr_reuses_upstream_campp_ceiling_parameters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "dialogue.wav"
    media.write_bytes(b"audio")
    captured: dict[str, object] = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            captured["model_kwargs"] = kwargs

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return [{"text": "你好"}]

    monkeypatch.setitem(
        sys.modules,
        "funasr",
        type("FakeFunASR", (), {"AutoModel": FakeAutoModel}),
    )
    monkeypatch.setattr(funasr_python_runner, "_select_device", lambda _device: "cuda")

    result = funasr_python_runner.run_funasr(
        input_path=str(media),
        output_path=str(tmp_path / "raw.json"),
        provider="sensevoice",
        model="iic/SenseVoiceSmall",
        spk_model="cam++",
        speaker_merge_threshold=0.74,
        preset_speaker_count=2,
        device="cuda",
    )

    assert result["ok"] is True
    assert captured["model_kwargs"]["spk_kwargs"] == {
        "cb_kwargs": {"merge_thr": 0.74}
    }
    assert captured["generate_kwargs"]["preset_spk_num"] == 2


def test_ab_plan_marks_known_two_speaker_variant_evaluation_only(
    tmp_path: Path,
) -> None:
    media = tmp_path / "dialogue.wav"
    media.write_bytes(b"audio")

    plan = plan_asr_ab_sample(tmp_path / "workspace", media, write=False)
    variant = {row["key"]: row for row in plan["variants"]}[
        "sensevoice_full_punc_campp_oracle_2"
    ]

    assert variant["role"] == "local_speaker_diagnostic_upper_bound"
    assert variant["command"][variant["command"].index("--preset-speaker-count") + 1] == "2"
    assert variant["full_mode"]["preset_speaker_count"] == 2
    assert variant["operator_boundary"] == {
        "does_not_promote_any_transcript": True,
        "speaker_labels_are_anonymous_candidates": True,
        "speaker_roles_are_not_inferred": True,
        "requires_prepared_local_speaker_model": True,
        "gpu_only": True,
        "evaluation_only_known_speaker_count": True,
        "known_speaker_count": 2,
        "must_not_become_automatic_production_route": True,
    }


def test_oracle_count_diagnostic_cannot_satisfy_production_speaker_gate() -> None:
    rows = [
        {
            "key": "sensevoice_full_punc_campp_oracle_2",
            "status": "ok",
            "metrics": {
                "segment_count": 3,
                "speaker_labeled_segment_count": 3,
                "speaker_count": 2,
            },
            "operator_boundary": {"evaluation_only_known_speaker_count": True},
        }
    ]
    requirement = {"required": True, "min_speaker_count": 2}

    gates = asr_ab_compare._gates(rows, speaker_requirement=requirement)
    decision = asr_ab_compare._decision(rows, speaker_requirement=requirement)

    assert gates["speaker_ready_variants"] == []
    assert gates["speaker_diagnostic_variants"] == [
        "sensevoice_full_punc_campp_oracle_2"
    ]
    assert gates["speaker_requirement_met"] is False
    assert decision["production_recommendation"] == (
        "blocked_until_speaker_diarization_ready"
    )


def test_invalid_campp_controls_fail_before_model_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "dialogue.wav"
    media.write_bytes(b"audio")

    class MustNotLoad:
        def __init__(self, **_kwargs):
            raise AssertionError("invalid controls must fail before model load")

    monkeypatch.setitem(
        sys.modules,
        "funasr",
        type("FakeFunASR", (), {"AutoModel": MustNotLoad}),
    )

    result = funasr_python_runner.run_funasr(
        input_path=str(media),
        output_path=str(tmp_path / "raw.json"),
        provider="sensevoice",
        model="iic/SenseVoiceSmall",
        spk_model="cam++",
        speaker_merge_threshold=1.1,
        preset_speaker_count=0,
    )

    assert result["ok"] is False
    assert result["error"] == "speaker_merge_threshold must be in (0, 1]"


def test_chunked_wrapper_forwards_upstream_campp_ceiling_parameters() -> None:
    command = funasr_chunked_runner._child_command(
        chunk=Path("chunk.wav"),
        output=Path("chunk.json"),
        provider="sensevoice",
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        speaker_merge_threshold=0.74,
        preset_speaker_count=2,
        language="zh",
        hotword="",
        batch_size_s=60,
        use_itn=True,
        merge_vad=True,
        merge_length_s=15,
        vad_max_single_segment_time_ms=30000,
        device="cuda",
    )

    assert command[command.index("--speaker-merge-threshold") + 1] == "0.74"
    assert command[command.index("--preset-speaker-count") + 1] == "2"
