from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline import dual_asr_workflow as workflow
from video_knowledge_pipeline.asr_consensus import build_asr_consensus
from video_knowledge_pipeline.asr_comparison_alignment import align_groups, locate_primary_span
from video_knowledge_pipeline.asr_diff_adjudication import build_asr_diff_adjudication, apply_asr_diff_adjudication
from video_knowledge_pipeline.asr_variant_evaluation import compare_asr_variants
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json


def transcript(root, name, rows):
    path = root / name
    write_json(path, {"segments": [{"start": a, "end": b, "text": t} for a, b, t in rows]})
    return path


def compare(root, left, right, **kwargs):
    p = transcript(root, "primary.json", left)
    s = transcript(root, "secondary.json", right)
    return build_asr_consensus(root, primary_transcript=p, secondary_transcript=s, **kwargs)


def test_many_to_many_preserves_original_segments_and_no_false_conflict(tmp_path):
    r = compare(tmp_path, [(0, 4, "大家好"), (4, 10, "今天讨论模型")],
                [(0, 3, "大家"), (3, 8, "好今天讨论"), (8, 10, "模型")])
    assert r["counts"]["agreement"] == 1
    assert r["windows"][0]["primary_indices"] == [0, 1]
    assert r["windows"][0]["secondary_indices"] == [0, 1, 2]
    assert r["review_audit"]["agreement_samples"]
    assert r["review_audit"]["coverage"]["status"] == "not_assessed"
    assert r["quality_verified"] is False


@pytest.mark.parametrize("left,right,flag", [
    ("会议讨论结束之后我们同意支付100万元", "会议讨论结束之后我们不同意支付100万元", "negation"),
    ("本次合同的最终金额经过审核是100万元", "本次合同的最终金额经过审核是101万元", "number_or_amount"),
    ("本次利率为1.5%", "本次利率为15%", "number_or_amount"),
    ("最终变化-10%", "最终变化+10%", "number_or_amount"),
    ("最后支付￥100", "最后支付$100", "currency"),
    ("由张三在会议结束之后提交这份正式材料", "由张山在会议结束之后提交这份正式材料", "lexical_change"),
    ("会议日期是2026-09-17", "会议日期是2026-09-18", "date_or_time"),
])
def test_critical_changes_are_conflicts_even_with_high_similarity(tmp_path, left, right, flag):
    r = compare(tmp_path, [(0, 10, left)], [(0, 10, right)])
    w = r["windows"][0]
    assert w["status"] == "conflict"
    assert flag in w["critical_differences"]
    assert w["character_differences"]
    pack = build_asr_diff_adjudication(tmp_path, write=False)
    assert pack["difference_count"] >= 1


def test_multisegment_patch_targets_second_source_not_first(tmp_path):
    compare(tmp_path, [(0, 4, "大家好"), (4, 10, "金额是100元")], [(0, 10, "大家好金额是200元")])
    pack = build_asr_diff_adjudication(tmp_path)
    diff = pack["differences"][0]
    assert diff["primary_segment_index"] == 1
    assert diff["primary_char_start"] == 3
    decision = tmp_path / "decisions.json"
    write_json(decision, {"rows": [{"diff_id": diff["diff_id"],
        "choice": "A" if diff["candidate_a_text"] == "200" else "B",
        "confidence": .99, "evidence_refs": ["local-audio-confirmation"]}]})
    r = apply_asr_diff_adjudication(tmp_path, decisions_json=decision)
    assert r["summary"]["applied_patches"] == 1
    assert read_json(tmp_path / "asr-consensus-patched-transcript.json")["segments"][1]["text"] == "金额是200元"
    assert read_json(tmp_path / "primary.json")["segments"][1]["text"] == "金额是100元"


def test_cross_segment_and_boundary_insertion_cannot_target_segment_zero():
    spans = [{"index": 0, "start": 0, "end": 3}, {"index": 1, "start": 3, "end": 7}]
    assert locate_primary_span(spans, 2, 5) is None
    assert locate_primary_span(spans, 3, 3) is None


def test_independent_common_gap_and_real_overlap_survive_agreement(tmp_path):
    media = tmp_path / "audio.wav"
    media.write_bytes(b"fixture only")
    activity = tmp_path / "activity.json"
    diar = tmp_path / "diar.json"
    write_json(activity, {"source_media_sha256": sha256_file(media), "speech_intervals": [{"start": 0, "end": 20}]})
    write_json(diar, {"source_media_sha256": sha256_file(media), "segments": [
        {"start": 0, "end": 6, "speaker": "A"}, {"start": 1, "end": 3, "speaker": "A"},
        {"start": 4, "end": 8, "speaker": "B"}]})
    r = compare(tmp_path, [(0, 10, "一样的文字")], [(0, 10, "一样的文字")], media_path=media,
                speech_activity_json=activity, diarization_json=diar)
    audit = r["review_audit"]
    assert audit["coverage"]["both"]["uncovered_seconds"] == 10
    assert audit["overlap"]["intervals"] == [{"start": 4., "end": 6.}]
    assert audit["independent_risk_windows"]
    assert r["windows"][0]["review_required"]
    write_json(activity, {"source_media_sha256": "wrong", "speech_intervals": []})
    with pytest.raises(ValueError, match="different audio"):
        compare(tmp_path, [(0, 10, "字")], [(0, 10, "字")], media_path=media, speech_activity_json=activity)


def test_exclusive_diarization_cannot_claim_overlap_checked(tmp_path):
    diar = tmp_path / "diar.json"
    write_json(diar, {"exclusive": True, "segments": []})
    r = compare(tmp_path, [(0, 10, "字")], [(0, 10, "字")], diarization_json=diar)
    assert r["review_audit"]["overlap"]["status"] == "not_assessed"


def test_preview_never_calls_models_and_reports_secondary_pending(tmp_path, monkeypatch):
    p = transcript(tmp_path, "primary.json", [(0, 10, "字")])
    monkeypatch.setattr(workflow, "plan_asr_run", lambda *a, **k: pytest.fail("model planning in preview"))
    r = workflow.run_dual_asr_workflow(tmp_path, primary_transcript=p, write=False)
    assert r["status"] == "secondary_pending"
    assert r["dual_full_transcription_complete"] is False
    assert "--execute" in r["resume_argv"]


@pytest.mark.parametrize("write", [False, True])
def test_supplied_secondary_updates_final_audit_and_retains_primary_risks(tmp_path, monkeypatch, write):
    media = tmp_path / "media.wav"
    media.write_bytes(b"fixture")
    activity = tmp_path / "activity.json"
    write_json(activity, {"source_media_sha256": sha256_file(media),
                          "speech_intervals": [{"start": 0, "end": 12}]})
    p = transcript(tmp_path, "primary.json", [(0, 4, "前半段内容")])
    s = transcript(tmp_path, "secondary.json", [(4, 10, "后半段内容")])
    monkeypatch.setattr(workflow, "plan_asr_run", lambda *a, **k: pytest.fail("model planning with supplied evidence"))
    r = workflow.run_dual_asr_workflow(tmp_path, media_path=media, primary_transcript=p,
        secondary_transcript=s, speech_activity_json=activity, secondary_timestamps="window", write=write)

    assert r["status"] == "dual_inputs_compared"
    assert r["review_audit"] == r["consensus"]["review_audit"]
    assert r["primary_review_audit"]["coverage"]["secondary"]["covered_seconds"] == 0
    assert r["primary_review_audit"]["coverage"]["both"]["covered_seconds"] == 4
    coverage = r["review_audit"]["coverage"]
    assert coverage["secondary"]["covered_seconds"] == 6
    assert coverage["both"]["covered_seconds"] == 10
    assert coverage["both"]["uncovered_seconds"] == 2
    assert r["review_audit"]["independent_risk_windows"]
    assert r["quality_verified"] is False
    assert r["dual_full_transcription_complete"] is False
    if write:
        for path in (tmp_path / "dual-asr-review.json",
                     tmp_path / "dual-asr-runs" / r["execution_id"][:20] / "state.json"):
            saved = read_json(path)
            assert saved["review_audit"] == r["review_audit"]
            assert saved["primary_review_audit"] == r["primary_review_audit"]
    else:
        assert not (tmp_path / "dual-asr-review.json").exists()


@pytest.mark.parametrize("write", [False, True])
def test_secondary_pending_retains_single_model_audit(tmp_path, monkeypatch, write):
    media = tmp_path / "media.wav"
    media.write_bytes(b"fixture")
    activity = tmp_path / "activity.json"
    write_json(activity, {"source_media_sha256": sha256_file(media),
                          "speech_intervals": [{"start": 0, "end": 12}]})
    p = transcript(tmp_path, "primary.json", [(0, 4, "已有单路内容")])
    monkeypatch.setattr(workflow, "plan_asr_run", lambda *a, **k: pytest.fail("model planning in preview"))
    r = workflow.run_dual_asr_workflow(tmp_path, media_path=media, primary_transcript=p,
        speech_activity_json=activity, secondary_timestamps="window", write=write)
    assert r["status"] == "secondary_pending"
    assert r["review_audit"] == r["primary_review_audit"]
    assert r["review_audit"]["coverage"]["secondary"]["covered_seconds"] == 0
    assert r["review_audit"]["coverage"]["both"]["uncovered_seconds"] == 8
    assert "consensus" not in r
    assert not r["dual_full_transcription_complete"]


def test_important_auto_advances_resumes_and_does_not_fallback(tmp_path, monkeypatch):
    media = tmp_path / "media.wav"
    media.write_bytes(b"fixture")
    calls = []
    monkeypatch.setattr(workflow, "probe_video", lambda *a: SimpleNamespace(duration_seconds=10))
    monkeypatch.setattr(workflow, "probe_audio_silence", lambda *a, **k: {"ok": True, "activity_intervals": [{"start": 0, "end": 10}]})
    def plan(root, media, **kw):
        calls.append(kw["preset"])
        return {"plan_path": str(root / "plan.json")}
    def run(plan_path, **kw):
        root = Path(plan_path).parent
        p = transcript(root, "normalized.json", [(0, 10, "同样的内容")])
        return {"status": "ok", "normalized": {"json_path": str(p)}}
    monkeypatch.setattr(workflow, "plan_asr_run", plan)
    monkeypatch.setattr(workflow, "run_asr_plan", run)
    r = workflow.run_dual_asr_workflow(tmp_path, media_path=media, execute=True)
    assert r["status"] == "dual_model_completed"
    assert calls == ["sensevoice", "qwen3-asr-1.7b"]
    r = workflow.run_dual_asr_workflow(tmp_path, media_path=media, execute=True)
    assert calls == ["sensevoice", "qwen3-asr-1.7b"]
    assert r["stages"]["primary"]["reused"]
    # Changed output cannot be silently reused.
    write_json(Path(r["secondary_transcript"]), {"segments": [{"start": 0, "end": 10, "text": "changed"}]})
    workflow.run_dual_asr_workflow(tmp_path, media_path=media, execute=True)
    assert calls[-1] == "qwen3-asr-1.7b" and len(calls) == 3


def test_secondary_failure_never_claims_dual_completion(tmp_path, monkeypatch):
    media = tmp_path / "media.wav"
    media.write_bytes(b"fixture")
    p = transcript(tmp_path, "primary.json", [(0, 10, "字")])
    monkeypatch.setattr(workflow, "probe_video", lambda *a: (_ for _ in ()).throw(ValueError("probe unavailable")))
    monkeypatch.setattr(workflow, "plan_asr_run", lambda *a, **k: {"plan_path": "failed-plan.json"})
    monkeypatch.setattr(workflow, "run_asr_plan", lambda *a, **k: {"status": "asr_model_not_ready"})
    r = workflow.run_dual_asr_workflow(tmp_path, media_path=media, primary_transcript=p, execute=True)
    assert r["status"] == "secondary_failed"
    assert r["stages"]["primary"]["status"] == "supplied"
    assert not r["dual_full_transcription_complete"]
    assert r["review_audit"] == r["primary_review_audit"]
    assert "consensus" not in r


def test_daily_selection_bounded_with_explicit_remainder():
    cues = [SimpleNamespace(start=0, end=200, text="金额100元")]
    r = workflow.daily_windows(cues, [], limit=2)
    assert len(r["windows"]) == 2
    assert r["remaining_windows"]
    assert all(w["end"]-w["start"] <= 30 for w in r["windows"]+r["remaining_windows"])


def test_three_variants_need_reviewed_reference_and_report_worse_integration(tmp_path):
    reference = tmp_path / "reference.json"
    write_json(reference, {"samples": [{"start": 0, "end": 10, "reference_text": "付款100元", "reference_status": "draft"}]})
    p = transcript(tmp_path, "primary.json", [(0, 10, "付款100元")])
    s = transcript(tmp_path, "secondary.json", [(0, 10, "付款100元")])
    f = transcript(tmp_path, "integrated.json", [(0, 10, "付款200元")])
    assert compare_asr_variants(reference, primary=p, secondary=s, integrated=f)["status"] == "blocked_reference_missing"
    data = read_json(reference)
    data["samples"][0]["reference_status"] = "human_verified"
    write_json(reference, data)
    r = compare_asr_variants(reference, primary=p, secondary=s, integrated=f)
    assert r["variants"]["primary"]["mean_sample_cer"] == 0
    assert r["variants"]["integrated"]["mean_sample_cer"] > 0
    assert r["model_switch_allowed"] is False


def test_quality_pipeline_does_not_polish_without_second_model(tmp_path, monkeypatch):
    from video_knowledge_pipeline import transcript_evidence_correction_pipeline as pipeline
    p = transcript(tmp_path, "primary.json", [(0, 10, "字")])
    write_json(tmp_path / "manifest.json", {"normalized_transcript_json": p.name})
    monkeypatch.setattr(pipeline, "postprocess_asr_transcript", lambda *a, **k: pytest.fail("promoted before dual ASR"))
    r = pipeline.run_transcript_evidence_correction_pipeline(tmp_path, quality_profile="quality", write=False)
    assert r["status"] == "secondary_pending"
    assert r["canonical_transcript_modified"] is False


def test_invalid_times_cannot_be_silently_aligned():
    with pytest.raises(ValueError):
        align_groups([SimpleNamespace(start=0, end=float("nan"))], [])


def test_daily_unchecked_ranges_are_not_false_omission_conflicts(tmp_path):
    r = compare(tmp_path, [(0, 20, "一整段长文字"), (30, 40, "没有复核的内容")],
                [(5, 10, "长文字")], secondary_scope="targeted")
    assert r["conflict_count"] == 0
    assert r["counts"]["not_comparable_partial_scope"] == 2
    assert r["status"] == "completed_partial_comparison"


def test_daily_uses_existing_targeted_runner_and_does_not_claim_full_review(tmp_path, monkeypatch):
    media = tmp_path / "media.wav"
    media.write_bytes(b"fixture")
    p = transcript(tmp_path, "primary.json", [(0, 10, "付款100元"), (10, 60, "继续讲话")])
    monkeypatch.setattr(workflow, "probe_video", lambda *a: SimpleNamespace(duration_seconds=60))
    monkeypatch.setattr(workflow, "probe_audio_silence", lambda *a, **k: {"ok": True, "activity_intervals": [{"start": 0, "end": 60}]})
    calls = []
    def targeted(root, **kwargs):
        calls.append(kwargs)
        plan = read_json(Path(kwargs["input_plan"]))
        assert len(plan["retry_plan"]["windows"]) == 1
        assert kwargs["preset"] == "qwen3-asr-1.7b"
        s = transcript(root, "evidence.json", [(0, 10, "付款100元")])
        return {"status": "completed", "artifacts": {"evidence_json": str(s)}}
    monkeypatch.setattr(workflow, "run_local_targeted_asr_evidence", targeted)
    monkeypatch.setattr(workflow, "plan_asr_run", lambda *a, **k: pytest.fail("daily must not rerun full secondary"))
    r = workflow.run_dual_asr_workflow(tmp_path, primary_transcript=p, media_path=media,
                                        mode="daily", execute=True, max_windows=1)
    assert r["status"] == "targeted_review_partial"
    assert not r["ok"] and not r["dual_full_transcription_complete"]
    assert calls and r["targeted_selection"]["remaining_windows"]
    assert r["review_audit"] == r["consensus"]["review_audit"]
    assert r["primary_review_audit"]["coverage"]["secondary"]["covered_seconds"] == 0
    assert r["review_audit"]["coverage"]["secondary"]["covered_seconds"] == 10
    assert int(r["resume_argv"][r["resume_argv"].index("--max-windows") + 1]) > 1


def test_cli_and_mcp_accept_new_controls():
    from video_knowledge_pipeline.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["dual-asr-review", "bundle", "--mode", "daily", "--execute"])
    assert args.mode == "daily" and args.execute
    args = parser.parse_args(["transcript-evidence-correction-pipeline", "bundle", "--asr-mode", "important", "--execute-local-asr"])
    assert args.execute_local_asr and args.asr_mode == "important"
    import ast
    from video_knowledge_pipeline import mcp_server
    ast.parse(Path(mcp_server.__file__).read_text(encoding="utf-8"))


def test_benchmark_rejects_different_sample_boundaries(tmp_path):
    reference = tmp_path / "reference.json"
    write_json(reference, {"samples": [{"start": 1, "end": 4, "reference_text": "中间文字", "reference_status": "human_verified"}]})
    p = transcript(tmp_path, "p.json", [(0, 5, "开头中间文字结尾")])
    r = compare_asr_variants(reference, primary=p, secondary=p, integrated=p)
    assert r["status"] == "blocked_incompatible_sample_boundaries"
    assert r["reviewed_sample_count"] == 0
    assert r["variants"]["primary"]["mean_sample_cer"] is None


def test_real_local_audio_probe_detects_both_transcripts_missing_tail(tmp_path):
    import math
    import struct
    import wave
    from video_knowledge_pipeline.media_tools import resolve_media_tool
    if not resolve_media_tool("ffmpeg") or not resolve_media_tool("ffprobe"):
        pytest.skip("local FFmpeg unavailable")
    media = tmp_path / "synthetic-tone.wav"
    with wave.open(str(media), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"".join(struct.pack("<h", int(12000 * math.sin(2*math.pi*440*i/16000)) if i < 16000 or i >= 32000 else 0) for i in range(48000)))
    p = transcript(tmp_path, "primary.json", [(0, 1, "占位文字")])
    s = transcript(tmp_path, "secondary.json", [(0, 1, "占位文字")])
    r = workflow.run_dual_asr_workflow(tmp_path, media_path=media, primary_transcript=p, secondary_transcript=s, execute=True)
    assert r["status"] == "dual_inputs_compared"
    audit = r["consensus"]["review_audit"]
    assert audit["coverage"]["evidence_kind"] == "non_silent_audio_not_proven_speech"
    assert audit["coverage"]["both"]["uncovered_seconds"] > .9
    assert not r["dual_full_transcription_complete"]


def test_mcp_registration_forwards_local_controls(tmp_path, monkeypatch):
    from mcp.server import fastmcp
    from video_knowledge_pipeline import mcp_server
    registered = {}
    class FakeServer:
        def __init__(self, *a, **kw): pass
        def tool(self, *a, **kw):
            def register(func):
                registered[func.__name__] = func
                return func
            return register
        def run(self, *a, **kw): pass
    monkeypatch.setattr(fastmcp, "FastMCP", FakeServer)
    mcp_server.main()
    p = transcript(tmp_path, "primary.json", [(0, 1, "字")])
    r = registered["dual_asr_review"](str(tmp_path), primary_transcript=str(p), mode="daily", write=False)
    assert r["status"] == "secondary_pending" and r["mode"] == "daily"
    assert "compare_asr_variants" in registered
