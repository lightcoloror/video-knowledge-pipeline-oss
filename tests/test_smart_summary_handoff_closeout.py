"""Eight handoff/closeout acceptance classes; synthetic inputs and no providers."""
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import smart_summary_codex as codex
from video_knowledge_pipeline import smart_summary_handoff as handoff
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.workflow_completion import workflow_completion_status

NATIVE_PREPARE = codex.prepare_smart_summary_llm_rewrite
NATIVE_REFRESH = codex._refresh_smart_summary_input_pack
NATIVE_CHAPTERS = codex.build_smart_summary_chapter_pack


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    from video_knowledge_pipeline import media_applicability, production_artifact_gate
    root = tmp_path / "bundle"
    root.mkdir()
    media = root / "input.mp3"
    media.write_bytes(b"synthetic stream fixture, not real audio")
    transcript = root / "corrected-transcript.json"
    write_json(transcript, [{"start": 0, "end": 20, "text": "合成内容，先检查输入，然后处理并核对结果。"}])
    write_json(root / "manifest.json", {"title": "Synthetic handoff",
        "sources": [{"path": str(media), "sha256": sha256_file(media)}],
        "corrected_transcript_json": "corrected-transcript.json",
        "source_arbitrated_transcript_json": "corrected-transcript.json"})
    write_json(root / "timeline.json", [{"index": 1, "start": 0, "end": 20,
        "transcript": "合成内容，先检查输入，然后处理并核对结果。", "frame_paths": [],
        "quality_issues": ["missing_frame", "missing_ocr"]}])
    write_json(root / "transcript-quality-gate.json", {"status": "passed", "ok": True,
        "fail_count": 0, "source_path": str(transcript),
        "source_completeness": {"applicable": True, "speech_completeness_verified": True}})
    media_applicability._probe_bound_bytes.cache_clear()
    monkeypatch.setattr(media_applicability, "_probe_with_ffprobe", lambda *_: {
        "audio_stream_count": 1, "video_stream_count": 0, "duration_seconds": 20})
    actual_gate = production_artifact_gate.evaluate_production_artifact_gate
    monkeypatch.setattr(codex, "evaluate_production_artifact_gate",
        lambda root, **kw: actual_gate(root, discover_prior_reviews=False, **kw))

    def prepare(root, **kwargs):
        root = Path(root)
        prompt = root / "exports/smart-summary-llm-rewrite-pack.md"
        if kwargs.get("write", True):
            write_json(root / "exports/smart-summary-input-pack.json", {
                "transcript_source": str(transcript), "transcript_source_sha256": sha256_file(transcript)})
            prompt.write_text("Read the bound synthetic transcript; write a candidate with unresolved items.\n", encoding="utf-8")
        return {"prompt_path": str(prompt), "status": "prepared"}

    monkeypatch.setattr(codex, "_refresh_smart_summary_input_pack", prepare)
    monkeypatch.setattr(codex, "prepare_smart_summary_llm_rewrite", prepare)
    monkeypatch.setattr(codex, "build_smart_summary_chapter_pack", lambda *a, **k: {"chapters": []})
    monkeypatch.setattr(codex, "call_openai_compatible_text", lambda **k: pytest.fail("unexpected provider call"))
    yield root
    media_applicability._probe_bound_bytes.cache_clear()


def candidate(root, prepared):
    path = Path(prepared["handoff"]["candidate_output_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!-- codex_final -->\n# 合成总结\n\n已根据输入检查，待核项必须继续保留。\n", encoding="utf-8")
    return path


def test_01_local_connection_failure_exposes_codex_handoff(bundle, monkeypatch):
    monkeypatch.setattr(codex, "resolve_text_provider_config", lambda *_: {
        "provider": "local_vlm", "base_url": "http://127.0.0.1:1234/v1", "model": "synthetic"})
    calls = []
    def unavailable(**kw):
        calls.append(1)
        raise ConnectionError("synthetic local connection refusal")
    monkeypatch.setattr(codex, "call_openai_compatible_text", unavailable)
    result = codex.run_smart_summary_llm_rewrite(bundle, execute=True)
    assert calls == [1]  # Mock only; no fallback network/model invocation.
    assert result["workflow_status"] == "awaiting-agent"
    assert result["handoff"]["operator_boundary"]["model_called"] is False
    followup = codex.generate_smart_summary_with_codex(bundle, agent_available=True)
    assert followup["workflow_status"] == "awaiting-agent"
    assert "--handoff-id" in followup["handoff"]["submit_command"]


def test_02_unattended_resume_is_persistent_idempotent_and_not_complete(bundle):
    first = codex.generate_smart_summary_with_codex(bundle)
    paths = list(bundle.rglob("*"))
    before = {str(p): p.read_bytes() for p in paths if p.is_file()}
    second = codex.generate_smart_summary_with_codex(bundle)
    assert first["workflow_status"] == second["workflow_status"] == "awaiting-agent"
    assert second["reused"] and not second["ok"]
    assert {str(p): p.read_bytes() for p in bundle.rglob("*") if p.is_file()} == before
    assert not (bundle / "exports/smart-summary.codex.md").exists()


def test_03_needs_llm_rewrite_is_a_candidate_writing_task(bundle):
    result = codex.generate_smart_summary_with_codex(bundle, agent_available=True)
    assert result["status"] == "needs_llm_rewrite"
    assert result["workflow_status"] == "awaiting-agent" and not result["ok"]
    assert Path(result["handoff"]["prompt_path"]).is_file()
    assert "--input-md" in result["handoff"]["submit_command"]
    assert not Path(result["handoff"]["candidate_output_path"]).exists()
    runs = read_json(bundle / "run-artifact-registry.json")["runs"]
    assert any(r["status"] == "awaiting-agent" for r in runs)


def test_04_admitted_candidate_uses_real_transaction_and_reader_export(bundle, monkeypatch):
    prepared = codex.generate_smart_summary_with_codex(bundle)
    source = candidate(bundle, prepared)
    # Isolate prose quality from the admission/transaction/export contract.
    monkeypatch.setattr(codex, "smart_summary_quality_check", lambda *a, **k: {
        "schema": codex.QUALITY_SCHEMA, "passed": True, "status": "passed", "checks": []})
    result = codex.generate_smart_summary_with_codex(bundle, input_md=source,
        handoff_id=prepared["handoff"]["handoff_id"])
    assert result["ok"] and result["installation_transaction"]["profile"] == "summary_install"
    assert (bundle / "exports/smart-summary.codex.md").read_bytes() == source.read_bytes()
    assert read_json(bundle / "exports/smart-summary-quality.json")["passed"] is True
    from video_knowledge_pipeline import knowledge_note_export
    monkeypatch.setattr(knowledge_note_export, "run_transcript_quality_gate",
        lambda *a, **k: read_json(bundle / "transcript-quality-gate.json"))
    secondary = bundle / "secondary.json"
    secondary.write_bytes((bundle / "corrected-transcript.json").read_bytes())
    write_json(bundle / "dual-asr-review.json", {"status": "dual_inputs_compared", "ok": True,
        "stages": {role: {"status": "supplied", "path": str(path), "sha256": sha256_file(path)}
                   for role, path in (("primary", bundle / "corrected-transcript.json"), ("secondary", secondary))},
        "consensus": {"review_required_count": 0}})
    from video_knowledge_pipeline import quality_finalize
    monkeypatch.setattr(quality_finalize, "run_transcript_quality_gate",
        lambda *a, **k: read_json(bundle / "transcript-quality-gate.json"))
    monkeypatch.setattr(quality_finalize, "smart_summary_quality_check", codex.smart_summary_quality_check)
    monkeypatch.setattr(quality_finalize, "build_semantic_chapter_plan",
        lambda *a, **k: pytest.fail("installed prose must not restart generation"))
    finalized = quality_finalize.finalize_quality_outputs(bundle, execute_llm=True)
    assert finalized["status"] == "completed" and finalized["ok"]
    assert (bundle / "exports/full-transcript.md").stat().st_size > 0
    assert (bundle / "exports/smart-summary.md").stat().st_size > 0
    completion = workflow_completion_status(bundle)
    assert completion["completed"], completion["remaining_stages"]


def test_05_failed_gate_registers_watermarked_candidate_and_retains_formal_bytes(bundle):
    prepared = codex.generate_smart_summary_with_codex(bundle)
    source = candidate(bundle, prepared)
    write_json(bundle / "transcript-quality-gate.json", {"status": "failed", "ok": False,
        "fail_count": 1, "issues": [{"key": "boundary", "severity": "fail"}]})
    exports = bundle / "exports"
    (exports / "smart-summary.md").write_bytes(b"old formal export\n")
    result = codex.generate_smart_summary_with_codex(bundle, input_md=source,
        handoff_id=prepared["handoff"]["handoff_id"])
    assert result["status"] == "blocked_by_production_artifact_gate"
    assert result["workflow_status"] == "review-required" and not result["ok"]
    state = result["candidate"]
    assert handoff.WATERMARK in (bundle / state["review_candidate_path"]).read_text("utf-8")
    assert "transcript_quality:failed" in state["remaining_review_items"]
    assert {"key": "boundary", "severity": "fail"} in state["remaining_review_items"]
    assert (exports / "smart-summary.md").read_bytes() == b"old formal export\n"
    assert not (exports / "smart-summary.codex.md").exists()
    assert any(r["status"] == "review-required" for r in read_json(bundle / "run-artifact-registry.json")["runs"])


def test_06_exit_zero_with_bad_quality_cannot_complete(bundle):
    write_json(bundle / "asr-run.json", {"exit_code": 0, "status": "completed"})
    write_json(bundle / "transcript-quality-gate.json", {"status": "failed", "ok": False, "fail_count": 1})
    result = workflow_completion_status(bundle)
    assert not result["completed"]
    stages = {s["key"]: s for s in result["stages"]}
    assert stages["primary_transcript"]["status"] == "review-required"
    assert stages["summary_validation"]["status"] == "blocked"
    assert len(result["stages"]) == 8


@pytest.mark.parametrize("video", [False, True])
def test_07_audio_exempts_frames_but_video_keeps_requirements(bundle, monkeypatch, video):
    from video_knowledge_pipeline import media_applicability as media
    from video_knowledge_pipeline.repair_status import refresh_bundle_repair_status
    media._probe_bound_bytes.cache_clear()
    monkeypatch.setattr(media, "_probe_with_ffprobe", lambda *_: {
        "audio_stream_count": 1, "video_stream_count": int(video), "duration_seconds": 20})
    result = refresh_bundle_repair_status(bundle)
    visual = result["repair_status"]["media_applicability"]
    assert (visual["visual_status"] == "not_applicable") is (not video)
    completion = workflow_completion_status(bundle)
    stage = next(s for s in completion["stages"] if s["key"] == "visual_evidence")
    assert (stage["status"] == "not-applicable") is (not video)
    if not video:
        assert all(r["status"] == "not_applicable" for r in result["repair_status"]["items"]
                   if r["key"] != "time_gap_recapture")


@pytest.mark.parametrize("changed", ["corrected-transcript.json", "exports/smart-summary-input-pack.json"])
def test_08_changed_inputs_cannot_promote_old_candidate(bundle, monkeypatch, changed):
    prepared = codex.generate_smart_summary_with_codex(bundle)
    source = candidate(bundle, prepared)
    write_json(bundle / changed, {"changed": True})
    monkeypatch.setattr(codex, "_install_summary_candidate", lambda *a, **k: pytest.fail("stale install"))
    result = codex.generate_smart_summary_with_codex(bundle, input_md=source,
        handoff_id=prepared["handoff"]["handoff_id"])
    assert result["status"] == "blocked_stale_candidate" and not result["ok"]
    assert not (bundle / "exports/smart-summary.codex.md").exists()


def test_registered_candidate_tampering_is_not_promotable(bundle):
    prepared = codex.generate_smart_summary_with_codex(bundle)
    source = candidate(bundle, prepared)
    write_json(bundle / "transcript-quality-gate.json", {"status": "failed"})
    result = codex.generate_smart_summary_with_codex(bundle, input_md=source)
    raw = bundle / result["candidate"]["raw_candidate_path"]
    raw.write_text("tampered", encoding="utf-8")
    assert codex.generate_smart_summary_with_codex(bundle, input_md=raw)["status"] == "blocked_stale_candidate"


def test_refresh_never_rebinds_old_installed_summary(bundle, monkeypatch):
    prepared = codex.generate_smart_summary_with_codex(bundle)
    source = candidate(bundle, prepared)
    monkeypatch.setattr(codex, "smart_summary_quality_check", lambda *a, **k: {
        "passed": True, "status": "passed", "checks": []})
    codex.generate_smart_summary_with_codex(bundle, input_md=source)
    snapshot = bundle / "exports/smart-summary-dependency-snapshot.json"
    before = snapshot.read_bytes()
    write_json(bundle / "corrected-transcript.json", [{"start": 0, "end": 20, "text": "输入已改变"}])
    codex.generate_smart_summary_with_codex(bundle)
    assert snapshot.read_bytes() == before
    assert codex._smart_summary_evidence_freshness_gate(bundle)["passed"] is False


def test_timestamp_only_refresh_is_not_progress(bundle, monkeypatch):
    from video_knowledge_pipeline import autopilot
    before = autopilot._action_fingerprint(bundle, {"status": "repair_pending", "key": "repair"})
    def churn(*args, **kwargs):
        m = read_json(bundle / "manifest.json")
        m["updated_at"] = "new timestamp only"
        write_json(bundle / "manifest.json", m)
        return {"advanced": True, "status": "completed"}
    monkeypatch.setattr(autopilot, "bundle_advance", churn)
    assert not autopilot._run_deterministic_action(bundle, {"status": "repair_pending", "key": "repair"})
    assert autopilot._action_fingerprint(bundle, {"status": "repair_pending", "key": "repair"}) == before


def test_background_autopilot_persists_agent_wait_without_executor(bundle, monkeypatch, tmp_path):
    from video_knowledge_pipeline import autopilot
    monkeypatch.setattr(autopilot, "resource_snapshot", lambda **k: autopilot.ResourceSnapshot(
        "synthetic", 1, 16 * 1024**3, .5, None, None, False))
    monkeypatch.setattr(autopilot, "local_vision_provider_snapshot", lambda: {"ready": False})
    action = {"status": "awaiting-agent", "kind": "smart_summary_handoff", "key": "smart_summary_handoff"}
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *a, **k: {"next_action": action})
    started = autopilot.autopilot_start(bundle, runner="background", job_root=tmp_path / "jobs")
    run = lambda: autopilot.autopilot_next(started["job_id"], job_root=tmp_path / "jobs",
        background_executor=lambda *a: pytest.fail("must not execute an Agent handoff unattended"))
    first, second = run(), run()
    assert first["status"] == second["status"] == "awaiting-agent"
    assert first["summary_handoff"]["handoff_id"] == second["summary_handoff"]["handoff_id"]
    assert not second["completion"]["completed"]


def test_no_write_candidate_registration_is_read_only(bundle):
    source = bundle / "candidate.md"
    source.write_text("synthetic unreviewed candidate", encoding="utf-8")
    write_json(bundle / "transcript-quality-gate.json", {"status": "failed"})
    before = {str(p): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    codex.generate_smart_summary_with_codex(bundle, input_md=source, write=False)
    assert {str(p): p.read_bytes() for p in bundle.rglob("*") if p.is_file()} == before


def test_native_preparation_has_matching_nonformal_prompt_and_idempotent_binding(bundle, monkeypatch):
    monkeypatch.setattr(codex, "prepare_smart_summary_llm_rewrite", NATIVE_PREPARE)
    monkeypatch.setattr(codex, "_refresh_smart_summary_input_pack", NATIVE_REFRESH)
    monkeypatch.setattr(codex, "build_smart_summary_chapter_pack", NATIVE_CHAPTERS)
    result = codex.generate_smart_summary_with_codex(bundle, agent_available=True)
    assert result["workflow_status"] == "awaiting-agent", result
    state = result["handoff"]
    output = Path(state["candidate_output_path"])
    assert output.is_relative_to(bundle / "runs") and output.parent.is_dir()
    assert str(output) in Path(state["prompt_path"]).read_text("utf-8")
    assert handoff.handoff_status(bundle)["freshness"] == "fresh"
    repeated = codex.generate_smart_summary_with_codex(bundle)
    assert repeated["reused"] and repeated["handoff"]["handoff_id"] == state["handoff_id"]
    assert not (bundle / "exports/smart-summary.codex.md").exists()


def test_bound_input_change_during_quality_check_aborts_real_install_transaction(bundle, monkeypatch):
    prepared = codex.generate_smart_summary_with_codex(bundle)
    source = candidate(bundle, prepared)
    def quality_race(*a, **k):
        write_json(bundle / "exports/smart-summary-input-pack.json", {"changed": "during check"})
        return {"passed": True, "status": "passed", "checks": []}
    monkeypatch.setattr(codex, "smart_summary_quality_check", quality_race)
    with pytest.raises(RuntimeError, match="semantic_transaction_input_drift:exports/smart-summary-input-pack.json"):
        codex.generate_smart_summary_with_codex(bundle, input_md=source,
            handoff_id=prepared["handoff"]["handoff_id"])
    assert not (bundle / "exports/smart-summary.codex.md").exists()


def test_repeated_timestamp_churn_records_exact_no_progress_action(bundle, monkeypatch, tmp_path):
    from video_knowledge_pipeline import autopilot
    monkeypatch.setattr(autopilot, "resource_snapshot", lambda **k: autopilot.ResourceSnapshot(
        "synthetic", 1, 16 * 1024**3, .5, None, None, False))
    monkeypatch.setattr(autopilot, "local_vision_provider_snapshot", lambda: {"ready": False})
    action = {"status": "repair_pending", "key": "synthetic", "command": "synthetic-repair --inspect"}
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *a, **k: {"next_action": action})
    calls = []
    def churn(*a, **k):
        calls.append(1)
        manifest = read_json(bundle / "manifest.json")
        manifest["updated_at"] = str(len(calls))
        write_json(bundle / "manifest.json", manifest)
        return {"advanced": True}
    monkeypatch.setattr(autopilot, "bundle_advance", churn)
    started = autopilot.autopilot_start(bundle, job_root=tmp_path / "jobs")
    result = autopilot.autopilot_next(started["job_id"], job_root=tmp_path / "jobs")
    assert result["no_progress"]["attempts"] == len(calls) == 2
    assert result["no_progress"]["next_action"] == action["command"]
    assert not result["completion"]["completed"]
    autopilot.autopilot_next(started["job_id"], job_root=tmp_path / "jobs")
    assert len(calls) == 2


def test_old_acceptance_without_stage_proof_does_not_rerun_media_or_replace_reports(bundle, monkeypatch, tmp_path):
    from video_knowledge_pipeline import batch_run
    receipt = bundle / "acceptance-check.json"
    write_json(receipt, {"status": "accepted", "next_action": {"key": "done"}})
    before = receipt.read_bytes()
    for name in ("run_acceptance_run", "run_acceptance_bundle"):
        monkeypatch.setattr(batch_run, name, lambda *a, **k: pytest.fail("no whole media rerun"))
    batch = tmp_path / "batch.json"
    write_json(batch, {"workspace": str(tmp_path / "batch"), "items": [{"id": "fixture", "bundle_dir": str(bundle)}]})
    result = batch_run.batch_video_knowledge_run(batch, resume=True)
    assert result["summary"]["completed_or_skipped"] == 0
    assert result["items"][0]["action"] == "inspect_incomplete_completion"
    assert receipt.read_bytes() == before


def test_watermark_is_not_a_formal_summary_even_when_final_marker_is_present():
    from video_knowledge_pipeline.knowledge_note_export import _looks_like_final_llm_smart_summary
    assert not _looks_like_final_llm_smart_summary(handoff.WATERMARK + "\n<!-- codex_final -->\n# Draft")


def test_malformed_dual_stage_is_blocked_instead_of_crashing_status(bundle):
    write_json(bundle / "dual-asr-review.json", {"status": "dual_model_completed", "ok": True,
                                               "stages": ["invalid"], "consensus": {"review_required_count": 0}})
    report = workflow_completion_status(bundle)
    assert "secondary_asr_and_differences" in report["remaining_stages"]


def test_autopilot_run_yields_agent_wait_without_spinning_and_legacy_draft_can_resume(bundle, monkeypatch, tmp_path):
    from video_knowledge_pipeline import autopilot
    monkeypatch.setattr(autopilot, "resource_snapshot", lambda **k: autopilot.ResourceSnapshot(
        "synthetic", 1, 16 * 1024**3, .5, None, None, False))
    monkeypatch.setattr(autopilot, "local_vision_provider_snapshot", lambda: {"ready": False})
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *a, **k: {"next_action": {
        "status": "awaiting-agent", "kind": "smart_summary_handoff", "key": "smart_summary_handoff"}})
    finished = autopilot.autopilot_run(bundle, runner="background", job_root=tmp_path / "jobs",
        background_executor=lambda *a: pytest.fail("no background summary call"))
    assert finished["status"] == "awaiting-agent"
    path = Path(finished["paths"]["job"])
    old = read_json(path)
    old.update(state="complete_draft", completion={"completed": True, "status": "completed"})
    old.pop("summary_handoff", None)
    write_json(path, old)
    status = autopilot.autopilot_status(finished["job_id"], job_root=tmp_path / "jobs")
    assert not status["completion"]["completed"]
    resumed = autopilot.autopilot_next(finished["job_id"], job_root=tmp_path / "jobs")
    assert resumed["status"] == "awaiting-agent" and resumed["summary_handoff"]["handoff_id"]
