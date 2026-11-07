from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.asr_adapter import normalize_asr_output, render_srt
from video_knowledge_pipeline.knowledge_note_export import _full_transcript_lines_from_cues
from video_knowledge_pipeline.models import TranscriptCue
from video_knowledge_pipeline.smart_summary_section_llm import _section_prompt
from video_knowledge_pipeline.transcript import parse_transcript
from video_knowledge_pipeline.transcript_postprocess import postprocess_asr_transcript
from video_knowledge_pipeline.transcript_quality_gate import run_transcript_quality_gate
from video_knowledge_pipeline.transcript_speakers import (
    speaker_label_map,
    speaker_payload,
)
from video_knowledge_pipeline.transcript_semantic_correction import (
    DOMAIN_SEMANTIC_REVIEW_ONLY_VARIANTS,
    _apply_decisions_to_cues,
    _domain_semantic_suspect_matches,
    transcript_semantic_correction_model_instructions,
)


def _write_transcript(path: Path, segments: list[dict]) -> None:
    path.write_text(
        json.dumps({"segments": segments}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_json_and_moss_normalization_preserve_speaker_contract(tmp_path: Path) -> None:
    source = tmp_path / "segments.json"
    _write_transcript(
        source,
        [
            {
                "id": "seg-1",
                "start": 0,
                "end": 1,
                "text": "根据排期来的嘛",
                "speaker": "S01",
                "speaker_role": "主持人",
            },
            {
                "id": "seg-2",
                "start": 1,
                "end": 2,
                "text": "什么都没有",
                "speaker": "S02",
            },
        ],
    )

    result = normalize_asr_output(
        tmp_path / "workspace",
        source,
        provider="moss-transcribe-diarize",
    )
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    cues = parse_transcript(result["json_path"])

    assert payload["segments"][0]["speaker"] == "S01"
    assert payload["segments"][0]["speaker_label"] == "主持人（说话人1）"
    assert payload["segments"][1]["speaker_label"] == "说话人2"
    assert cues[0].speaker == "S01"
    assert cues[0].speaker_role == "主持人"
    assert cues[1].speaker == "S02"


def test_numeric_zero_speaker_cluster_is_preserved_and_labeled() -> None:
    """Guard the real CAM++ ``spk: 0`` / ``spk: 1`` output contract.

    Intent: retain both anonymous clusters from a two-person recording.
    Decision: test numeric IDs before any ASR adapter serializes them.
    Reason: Python truthiness previously erased cluster zero.
    Evidence: the local CAM++ GPU trial contained numeric cluster IDs 0 and 1.
    Effective scope: metadata normalization only; labels remain anonymous.
    """

    cues = [
        {"start": 0, "end": 1, "text": "第一句", "spk": 0},
        {"start": 1, "end": 2, "text": "第二句", "spk": 1},
    ]

    labels = speaker_label_map(cues)

    assert labels == {"0": "说话人1", "1": "说话人2"}
    assert speaker_payload(cues[0], labels) == {
        "speaker": "0",
        "speaker_label": "说话人1",
    }
    assert speaker_payload(cues[1], labels) == {
        "speaker": "1",
        "speaker_label": "说话人2",
    }


def test_funasr_numeric_speaker_clusters_survive_normalize_and_parse(
    tmp_path: Path,
) -> None:
    """Exercise the full raw FunASR -> normalized JSON -> reader path.

    Intent: prevent a lower-level fix from being undone by an adapter/parser.
    Decision: use the upstream ``sentence_info`` shape with integer spk IDs.
    Reason: the real CAM++ output reaches VKP through exactly this contract.
    Evidence: FunASR 1.3.30 produced 168 sentence rows with spk 0 and spk 1.
    Effective scope: offline normalization compatibility only.
    """

    source = tmp_path / "funasr.json"
    source.write_text(
        json.dumps(
            {
                "result": [
                    {
                        "sentence_info": [
                            {"start": 0, "end": 1000, "text": "第一句", "spk": 0},
                            {"start": 1000, "end": 2000, "text": "第二句", "spk": 1},
                        ]
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = normalize_asr_output(tmp_path / "workspace", source, provider="funasr")
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    parsed = parse_transcript(result["json_path"])

    assert [row["speaker"] for row in payload["segments"]] == ["0", "1"]
    assert [row["speaker_label"] for row in payload["segments"]] == [
        "说话人1",
        "说话人2",
    ]
    assert [cue.speaker for cue in parsed] == ["0", "1"]


def test_srt_roundtrip_keeps_anonymous_speaker_labels(tmp_path: Path) -> None:
    cues = [
        TranscriptCue(start=0, end=1, text="你好", speaker="S01"),
        TranscriptCue(start=1, end=2, text="开始吧", speaker="S02"),
    ]
    path = tmp_path / "dialogue.srt"
    path.write_text(render_srt(cues), encoding="utf-8")

    parsed = parse_transcript(path)

    assert [cue.speaker for cue in parsed] == ["说话人1", "说话人2"]
    assert [cue.text for cue in parsed] == ["你好", "开始吧"]


def test_postprocess_preserves_speakers_and_never_merges_across_them(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = bundle / "normalized-transcript.json"
    _write_transcript(
        source,
        [
            {"id": "a", "start": 0, "end": 1, "text": "第一句", "speaker": "S01"},
            {"id": "b", "start": 1, "end": 2, "text": "第二句", "speaker": "S01"},
            {"id": "c", "start": 2, "end": 3, "text": "客户回答", "speaker": "S02"},
        ],
    )

    result = postprocess_asr_transcript(
        bundle,
        input_path=source,
        target_seconds=30,
        max_chars=500,
        segment_policy="readable_merge",
        set_corrected=False,
        write=True,
    )
    payload = json.loads(
        (bundle / "postprocessed-transcript.json").read_text(encoding="utf-8")
    )

    assert result["postprocessed_segment_count"] == 2
    assert [row["speaker"] for row in payload["segments"]] == ["S01", "S02"]
    assert payload["segments"][0]["source_segment_ids"] == ["a", "b"]
    assert payload["segments"][1]["source_segment_ids"] == ["c"]


def test_quality_gate_fails_closed_when_dialogue_requires_diarization(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "missing-speakers"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "duration_seconds": 2,
                "corrected_transcript_json": "corrected-transcript.json",
                "transcript_requirements": {
                    "speaker_diarization_required": True,
                    "expected_speaker_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_transcript(
        bundle / "corrected-transcript.json",
        [
            {"start": 0, "end": 1, "text": "你好。"},
            {"start": 1, "end": 2, "text": "开始。"},
        ],
    )

    result = run_transcript_quality_gate(
        bundle,
        min_punctuation_per_1000=0,
        max_punctuation_per_1000=1000,
        write=False,
    )

    assert result["speaker_diarization"]["status"] == "speaker_diarization_required"
    assert result["speaker_diarization"]["distinct_speaker_count"] == 0
    assert "speaker_diarization_required" in {
        row["kind"] for row in result["issues"]
    }


def test_quality_gate_accepts_two_fully_labeled_speakers(tmp_path: Path) -> None:
    bundle = tmp_path / "two-speakers"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"duration_seconds": 2}),
        encoding="utf-8",
    )
    transcript = bundle / "dialogue.json"
    _write_transcript(
        transcript,
        [
            {"start": 0, "end": 1, "text": "你好。", "speaker": "S01"},
            {"start": 1, "end": 2, "text": "开始。", "speaker": "S02"},
        ],
    )

    result = run_transcript_quality_gate(
        bundle,
        input_path=transcript,
        require_speaker_diarization=True,
        min_speaker_count=2,
        min_punctuation_per_1000=0,
        max_punctuation_per_1000=1000,
        write=False,
    )

    assert result["speaker_diarization"]["passed"] is True
    assert result["speaker_diarization"]["segment_label_coverage"] == 1.0
    assert result["speaker_diarization"]["distinct_speaker_count"] == 2
    assert "speaker_diarization_required" not in {
        row["kind"] for row in result["issues"]
    }


def test_human_confirmed_source_corrections_keep_speaker_identity() -> None:
    cue = TranscriptCue(
        start=0,
        end=5,
        text="根排期来的嘛，会义纪要，发了一分材料，星合系统。",
        speaker="S01",
    )
    pairs = [
        ("根排期来的嘛", "根据排期来的嘛"),
        ("会义纪要", "会议纪要"),
        ("发了一分材料", "发了一份材料"),
        ("星合系统", "星河系统"),
    ]
    decisions = [
        {
            "candidate_id": f"human-{index}",
            "segment_index": 0,
            "action": "replace",
            "correction_type": "term",
            "original_text": original,
            "corrected_text": corrected,
            "confidence": 1.0,
            "rationale": "user confirmed against the recording",
            "human_confirmed": True,
        }
        for index, (original, corrected) in enumerate(pairs, start=1)
    ]

    segments, applied = _apply_decisions_to_cues([cue], decisions)

    assert segments[0]["text"] == "根据排期来的嘛，会议纪要，发了一份材料，星河系统。"
    assert segments[0]["speaker"] == "S01"
    assert segments[0]["speaker_label"] == "说话人1"
    assert len(applied) == 4
    for original, _corrected in pairs:
        assert original not in DOMAIN_SEMANTIC_REVIEW_ONLY_VARIANTS
        assert not _domain_semantic_suspect_matches(original)


def test_summary_prompts_define_source_fidelity_not_world_fact_judgment(
    tmp_path: Path,
) -> None:
    prompt = _section_prompt(
        tmp_path,
        {
            "start_time": "00:00:00",
            "end_time": "00:01:00",
            "title": "产品观点",
            "evidence": {"summary_sentences": ["讲者表示目前条款设计最优。"]},
        },
        max_prompt_chars=4000,
    )
    correction_instructions = transcript_semantic_correction_model_instructions()

    assert "目标不是判断其观点在外部世界是否真实" in prompt
    assert "讲者表示/客户认为" in prompt
    assert "source fidelity" in correction_instructions
    assert "external-world uncertainty alone is not a correction reason" in correction_instructions


def test_final_transcript_reader_output_includes_speaker_labels() -> None:
    lines = _full_transcript_lines_from_cues(
        [
            TranscriptCue(start=0, end=1, text="你好", speaker="S01"),
            TranscriptCue(start=1, end=2, text="你好", speaker="S02"),
        ]
    )
    text = "\n".join(lines)

    assert "**说话人1**" in text
    assert "**说话人2**" in text
