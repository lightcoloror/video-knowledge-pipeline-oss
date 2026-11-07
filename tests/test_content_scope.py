from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from video_knowledge_pipeline.content_scope import (
    CONTENT_SCOPE_SCHEMA,
    CONTENT_SCOPE_SCHEMA_V1,
    apply_content_scope_to_cues,
    load_content_scope,
)
from video_knowledge_pipeline.knowledge_note_export import (
    _build_transcript_projection_audit,
    _render_full_body,
    _render_full_transcript,
    export_knowledge_note,
)
from video_knowledge_pipeline.models import TranscriptCue
from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
from video_knowledge_pipeline.smart_summary_codex import _smart_summary_content_scope_gate
from video_knowledge_pipeline.transcript_projection import (
    classify_whisper_quality_signals,
    project_transcript_cues,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bundle_with_warmup(tmp_path: Path) -> tuple[Path, dict[str, object], list[dict[str, object]]]:
    root = tmp_path / "bundle"
    root.mkdir()
    transcript_path = root / "normalized-transcript.json"
    _write_json(
        transcript_path,
        {
            "segments": [
                {"id": "warmup-1", "start": 0.0, "end": 9.5, "text": "穿过黑夜寻找黎明方向。"},
                {"id": "meeting-1", "start": 10.0, "end": 14.0, "text": "大家早上好，正式分享现在开始。"},
                {"id": "meeting-2", "start": 14.2, "end": 20.0, "text": "今天讲三个实操步骤。"},
            ]
        },
    )
    transcript_sha256 = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "title": "合成早会",
        "normalized_transcript_json": "normalized-transcript.json",
    }
    timeline: list[dict[str, object]] = [
        {"index": 1, "start": 0.0, "end": 9.5, "transcript": "穿过黑夜寻找黎明方向。"},
        {"index": 2, "start": 10.0, "end": 14.0, "transcript": "大家早上好，正式分享现在开始。"},
        {"index": 3, "start": 14.2, "end": 20.0, "transcript": "今天讲三个实操步骤。"},
    ]
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "timeline.json", timeline)
    _write_json(
        root / "content-scope.json",
        {
            "schema": CONTENT_SCOPE_SCHEMA,
            "status": "confirmed",
            "transcript_source": "normalized-transcript.json",
            "transcript_source_sha256": transcript_sha256,
            "excluded_intervals": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "reason": "warmup_music",
                    "category": "warmup_music",
                    "disposition": "excluded_confirmed",
                    "decision": "operator_confirmed",
                    "exclude_from": ["continuous_full_body", "smart_summary"],
                    "retain_in": ["intermediate_evidence", "full_transcript"],
                }
            ],
        },
    )
    return root, manifest, timeline


def test_warmup_is_excluded_from_full_body_and_summary_but_retained_in_transcript(tmp_path: Path) -> None:
    root, manifest, timeline = _bundle_with_warmup(tmp_path)

    full_body = _render_full_body("合成早会", timeline, bundle_dir=root, manifest=manifest, sidecar={})
    full_transcript = _render_full_transcript(
        "合成早会",
        timeline,
        bundle_dir=root,
        manifest=manifest,
        sidecar={},
    )
    summary_pack = build_smart_summary_input_pack(root, title="合成早会", write=False)
    projection_audit = _build_transcript_projection_audit(root, manifest, {}, timeline)
    summary_text = " ".join(str(row.get("raw_text") or "") for row in summary_pack["transcript_segments"])

    assert "穿过黑夜" not in full_body
    assert "穿过黑夜" not in summary_text
    assert "穿过黑夜" in full_transcript
    assert "正式分享现在开始" in full_body
    assert "正式分享现在开始" in summary_text
    assert summary_pack["content_scope"]["excluded_transcript_cue_count"] == 1
    assert summary_pack["content_scope"]["excluded_intervals"][0]["reason"] == "warmup_music"
    assert projection_audit["evidence_boundary"]["source_transcript_modified"] is False
    assert projection_audit["projections"]["smart_summary"]["disposition_counts"]["excluded_confirmed"] == 1


def test_content_scope_hash_drift_fails_closed(tmp_path: Path) -> None:
    root, _manifest, _timeline = _bundle_with_warmup(tmp_path)
    scope_path = root / "content-scope.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["transcript_source_sha256"] = "0" * 64
    _write_json(scope_path, scope)

    with pytest.raises(ValueError, match="transcript_source_sha256"):
        load_content_scope(root, transcript_path=root / "normalized-transcript.json")


def test_partial_boundary_cue_is_retained_and_reported(tmp_path: Path) -> None:
    root, _manifest, _timeline = _bundle_with_warmup(tmp_path)
    scope = load_content_scope(root, transcript_path=root / "normalized-transcript.json")
    cues = [
        TranscriptCue(start=9.0, end=11.0, text="跨越暖场和正式开场的原句", segment_id="boundary"),
        TranscriptCue(start=11.0, end=12.0, text="正式内容", segment_id="content"),
    ]

    kept, report = apply_content_scope_to_cues(cues, scope, projection="smart_summary")

    assert [cue.segment_id for cue in kept] == ["boundary", "content"]
    assert report["excluded_count"] == 0
    assert report["partial_overlap_count"] == 1


def test_v1_scope_remains_readable_without_bulk_migration(tmp_path: Path) -> None:
    root, _manifest, _timeline = _bundle_with_warmup(tmp_path)
    scope_path = root / "content-scope.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["schema"] = CONTENT_SCOPE_SCHEMA_V1
    scope["excluded_intervals"][0].pop("category")
    scope["excluded_intervals"][0].pop("disposition")
    _write_json(scope_path, scope)

    loaded = load_content_scope(root, transcript_path=root / "normalized-transcript.json")

    assert loaded["schema"] == CONTENT_SCOPE_SCHEMA_V1
    assert loaded["excluded_intervals"][0]["category"] == "warmup_music"
    assert loaded["excluded_intervals"][0]["disposition"] == "excluded_confirmed"


def test_body_replacement_compresses_host_intro_without_leaking_to_summary(tmp_path: Path) -> None:
    root, manifest, _timeline = _bundle_with_warmup(tmp_path)
    transcript_path = root / "normalized-transcript.json"
    _write_json(
        transcript_path,
        {
            "segments": [
                {"id": "warmup", "start": 0.0, "end": 9.5, "text": "暖场歌词。"},
                {"id": "test", "start": 10.0, "end": 19.0, "text": "能看到画面吗，请签到。"},
                {"id": "host", "start": 20.0, "end": 25.0, "text": "我是主持人秋叶，先聊聊新学期。"},
                {"id": "agenda", "start": 26.0, "end": 30.0, "text": "今天先讲医疗险的三个问题。"},
            ]
        },
    )
    manifest["normalized_transcript_json"] = "normalized-transcript.json"
    _write_json(root / "manifest.json", manifest)
    transcript_hash = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
    _write_json(
        root / "content-scope.json",
        {
            "schema": CONTENT_SCOPE_SCHEMA,
            "status": "confirmed",
            "transcript_source": "normalized-transcript.json",
            "transcript_source_sha256": transcript_hash,
            "excluded_intervals": [
                {
                    "interval_id": "technical-check",
                    "start": 10.0,
                    "end": 20.0,
                    "reason": "microphone_screen_and_sign_in",
                    "category": "technical_check",
                    "disposition": "excluded_confirmed",
                    "decision": "operator_confirmed",
                    "exclude_from": ["continuous_full_body", "smart_summary"],
                    "retain_in": ["full_transcript", "intermediate_evidence"],
                },
                {
                    "interval_id": "host-intro",
                    "start": 20.0,
                    "end": 26.0,
                    "reason": "host_intro_and_small_talk",
                    "category": "small_talk",
                    "disposition": "excluded_confirmed",
                    "decision": "operator_confirmed",
                    "exclude_from": ["continuous_full_body", "smart_summary"],
                    "retain_in": ["full_transcript", "intermediate_evidence"],
                    "replacement_by_projection": {
                        "continuous_full_body": "早会正式开始。主持人是秋叶。"
                    },
                },
            ],
        },
    )
    timeline = [
        {"index": 1, "start": 0.0, "end": 9.5, "transcript": "暖场歌词。"},
        {"index": 2, "start": 10.0, "end": 19.0, "transcript": "能看到画面吗，请签到。"},
        {"index": 3, "start": 20.0, "end": 25.0, "transcript": "我是主持人秋叶，先聊聊新学期。"},
        {"index": 4, "start": 26.0, "end": 30.0, "transcript": "今天先讲医疗险的三个问题。"},
    ]
    _write_json(root / "timeline.json", timeline)

    full_body = _render_full_body("合成早会", timeline, bundle_dir=root, manifest=manifest, sidecar={})
    full_transcript = _render_full_transcript(
        "合成早会", timeline, bundle_dir=root, manifest=manifest, sidecar={}
    )
    summary_pack = build_smart_summary_input_pack(root, title="合成早会", write=False)
    summary_text = " ".join(str(row.get("raw_text") or "") for row in summary_pack["transcript_segments"])

    assert "能看到画面吗" not in full_body
    assert "先聊聊新学期" not in full_body
    assert "早会正式开始。主持人是秋叶。" in full_body
    assert "今天先讲医疗险" in full_body
    assert "能看到画面吗" not in summary_text
    assert "先聊聊新学期" not in summary_text
    assert "早会正式开始" not in summary_text
    assert "能看到画面吗" in full_transcript
    assert "先聊聊新学期" in full_transcript
    assert summary_pack["content_scope"]["replacement_count"] == 0


def test_projection_only_auto_removes_high_certainty_noise() -> None:
    cues = [
        TranscriptCue(start=0, end=1, text="", segment_id="empty"),
        TranscriptCue(start=1, end=2, text="？！", segment_id="punctuation"),
        TranscriptCue(start=2, end=3, text="呃", segment_id="safe-filler"),
        TranscriptCue(start=3, end=4, text="嗯", segment_id="ambiguous-yes"),
        TranscriptCue(start=4, end=5, text="好", segment_id="ambiguous-ok"),
        TranscriptCue(start=5, end=6, text="对", segment_id="ambiguous-right"),
        TranscriptCue(start=6, end=7, text="不是不能赔，除外责任要核对。", segment_id="negation"),
        TranscriptCue(start=7, end=8, text="保额是 100 万，日期是 8 月 31 日。", segment_id="numbers"),
    ]

    projected, report = project_transcript_cues(
        cues,
        {"schema": CONTENT_SCOPE_SCHEMA, "status": "not_configured", "excluded_intervals": []},
        projection="smart_summary",
    )

    assert [cue.segment_id for cue in projected] == [
        "ambiguous-yes",
        "ambiguous-ok",
        "ambiguous-right",
        "negation",
        "numbers",
    ]
    assert report["projection_omitted_count"] == 3
    assert report["disposition_counts"] == {
        "retained_semantic": 5,
        "excluded_confirmed": 2,
        "normalized_disfluency": 1,
        "review_gap": 0,
    }


def test_duplicate_requires_overlap_or_shared_source_identity() -> None:
    cues = [
        TranscriptCue(start=0.0, end=2.0, text="这是分窗边界重复。", segment_id="a"),
        TranscriptCue(start=1.8, end=3.0, text="这是分窗边界重复。", segment_id="b"),
        TranscriptCue(start=10.0, end=11.0, text="这是分窗边界重复。", segment_id="c"),
    ]

    projected, report = project_transcript_cues(
        cues,
        {"schema": CONTENT_SCOPE_SCHEMA, "status": "not_configured", "excluded_intervals": []},
        projection="continuous_full_body",
    )

    assert [cue.segment_id for cue in projected] == ["a", "c"]
    assert report["disposition_counts"]["normalized_disfluency"] == 1


def test_whisper_scores_create_review_gaps_not_deletion_authority() -> None:
    silence_candidate = classify_whisper_quality_signals(
        {"no_speech_prob": 0.8, "avg_logprob": -1.2, "compression_ratio": 1.0}
    )
    compression_candidate = classify_whisper_quality_signals(
        {"no_speech_prob": 0.1, "avg_logprob": -0.5, "compression_ratio": 2.8}
    )
    cues = [
        TranscriptCue(
            start=0,
            end=1,
            text="静音上疑似生成的句子。",
            segment_id="silence-candidate",
            metadata={"no_speech_prob": 0.8, "avg_logprob": -1.2},
        ),
        TranscriptCue(
            start=1,
            end=2,
            text="真实低分语音不能直接删除。",
            segment_id="compression-review",
            metadata={"compression_ratio": 2.8},
        ),
    ]

    summary, report = project_transcript_cues(
        cues,
        {"schema": CONTENT_SCOPE_SCHEMA, "status": "not_configured", "excluded_intervals": []},
        projection="smart_summary",
    )
    transcript, _ = project_transcript_cues(
        cues,
        {"schema": CONTENT_SCOPE_SCHEMA, "status": "not_configured", "excluded_intervals": []},
        projection="full_transcript",
    )

    assert silence_candidate.disposition == "review_gap"
    assert silence_candidate.omission_allowed is False
    assert compression_candidate.disposition == "review_gap"
    assert [cue.segment_id for cue in summary] == ["compression-review"]
    assert [cue.segment_id for cue in transcript] == ["silence-candidate", "compression-review"]
    assert report["disposition_counts"]["review_gap"] == 2


def test_confirmed_hallucination_can_leave_readable_transcript_but_not_raw_source(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    transcript_path = root / "normalized-transcript.json"
    _write_json(
        transcript_path,
        {
            "segments": [
                {"id": "hallucination", "start": 0, "end": 1, "text": "字幕组制作。"},
                {"id": "speech", "start": 2, "end": 3, "text": "正式内容。"},
            ]
        },
    )
    source_hash = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
    _write_json(root / "manifest.json", {"normalized_transcript_json": "normalized-transcript.json"})
    _write_json(root / "timeline.json", [])
    _write_json(
        root / "content-scope.json",
        {
            "schema": CONTENT_SCOPE_SCHEMA,
            "status": "confirmed",
            "transcript_source": "normalized-transcript.json",
            "transcript_source_sha256": source_hash,
            "excluded_intervals": [
                {
                    "interval_id": "confirmed-hallucination",
                    "start": 0,
                    "end": 1.5,
                    "reason": "confirmed_silence_hallucination",
                    "category": "asr_hallucination",
                    "disposition": "excluded_confirmed",
                    "decision": "operator_confirmed",
                    "exclude_from": ["full_transcript", "continuous_full_body", "smart_summary"],
                    "retain_in": ["raw_asr", "correction_audit"],
                }
            ],
        },
    )

    rendered = _render_full_transcript("合成", [], bundle_dir=root, manifest={}, sidecar={})

    assert "字幕组制作" not in rendered
    assert "正式内容" in rendered
    assert hashlib.sha256(transcript_path.read_bytes()).hexdigest() == source_hash


def test_smart_summary_quality_gate_rejects_stale_or_leaked_scope_content(tmp_path: Path) -> None:
    root, _manifest, _timeline = _bundle_with_warmup(tmp_path)
    exports = root / "exports"
    exports.mkdir()
    scope = load_content_scope(root, transcript_path=root / "normalized-transcript.json")
    _write_json(
        exports / "smart-summary-input-pack.json",
        {"content_scope": {"content_scope_sha256": scope["content_scope_sha256"]}},
    )
    summary_path = exports / "smart-summary.codex.md"
    summary_path.write_text("# 总结\n\n穿过黑夜寻找黎明方向。\n", encoding="utf-8")

    leaked = _smart_summary_content_scope_gate(
        root,
        summary_path,
        summary_path.read_text(encoding="utf-8"),
    )
    summary_path.write_text("# 总结\n\n正式分享包含三个实操步骤。\n", encoding="utf-8")
    clean = _smart_summary_content_scope_gate(
        root,
        summary_path,
        summary_path.read_text(encoding="utf-8"),
    )

    assert leaked["passed"] is False
    assert leaked["forbidden_phrase_hits"]
    assert clean["passed"] is True


def test_export_does_not_adopt_stale_codex_summary_that_leaks_excluded_content(tmp_path: Path) -> None:
    root, _manifest, _timeline = _bundle_with_warmup(tmp_path)
    exports = root / "exports"
    exports.mkdir()
    candidate = exports / "smart-summary.codex.md"
    candidate.write_text(
        "# 合成早会 - 智能总结\n\n生成方式：`codex_final`。\n\n穿过黑夜寻找黎明方向。\n",
        encoding="utf-8",
    )
    scope_path = root / "content-scope.json"
    candidate_mtime = candidate.stat().st_mtime_ns
    os.utime(scope_path, ns=(candidate_mtime + 1_000_000_000, candidate_mtime + 1_000_000_000))

    result = export_knowledge_note(
        root,
        title="合成早会",
        run_transcript_evidence_check=False,
    )
    installed = (exports / "smart-summary.md").read_text(encoding="utf-8")

    assert result["existing_smart_summary_scope_gate"]["passed"] is False
    assert "穿过黑夜" not in installed
    assert "needs_llm_summary" in installed
