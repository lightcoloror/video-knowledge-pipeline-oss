from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from video_knowledge_pipeline.cli import build_parser, main
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.small_model_video_closure import (
    VideoClosureError,
    finalize_video_closure,
    resume_video_closure,
    run_video_closure,
    video_closure_doctor,
    video_closure_status,
)


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "small-model-video-closure"


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "synthetic-bundle"
    shutil.copytree(FIXTURE, bundle)
    return bundle


def _transcript_path(bundle: Path) -> Path:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    return bundle / manifest["normalized_transcript"]


def _mvp_root(tmp_path: Path) -> Path:
    root = tmp_path / "mvp-video-pipeline"
    bridge = root / "src" / "video_creation_pipeline" / "vkp_bridge.py"
    cli = root / "src" / "video_creation_pipeline" / "cli.py"
    schema = root / "tests" / "fixtures" / "content-video-brief-v1" / "upstream-canonical" / "content-video-brief.v1.schema.json"
    bridge.parent.mkdir(parents=True)
    schema.parent.mkdir(parents=True)
    bridge.write_text("# synthetic bridge contract\n", encoding="utf-8")
    cli.write_text("# synthetic cli contract\n", encoding="utf-8")
    schema.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": [
                    "schema_version",
                    "brief_id",
                    "content_id",
                    "mode",
                    "title",
                    "thesis",
                    "audience",
                    "target_duration_seconds",
                    "chapters",
                    "scenes",
                    "review_gate",
                    "provenance",
                ],
                "properties": {
                    "schema_version": {"const": "content-video-brief.v1"},
                    "mode": {"const": "draft_only"},
                    "chapters": {"type": "array"},
                    "scenes": {"type": "array"},
                    "review_gate": {"type": "object"},
                    "provenance": {"type": "object"},
                },
                "additionalProperties": True,
            }
        ),
        encoding="utf-8",
    )
    return root


def _fake_edit_job(bundle: Path) -> dict:
    return {
        "schema": "video_creation_pipeline.vkp_edit_job.v3",
        "source_bundle": {"path": str(bundle)},
        "readiness": {"status": "ready_for_human_selection", "blockers": []},
        "candidate_clips": [{"candidate_id": "timeline-0001", "selection_status": "candidate_only", "human_confirmed": False, "source_interval": {"start": 0.0, "end": 8.0}}],
        "operator_boundary": {"automatic_publish": False, "bridge_executes_ffmpeg": False, "bridge_executes_asr": False},
        "remote_capabilities": {"execute": False, "automatic_publish": False, "upload": {"allowed": False, "automatic": False}},
    }


def test_doctor_is_compact_read_only_and_exposes_stable_commands(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    mvp_root = _mvp_root(tmp_path)
    before = {path: sha256_file(path) for path in bundle.rglob("*") if path.is_file()}

    result = video_closure_doctor(bundle, mvp_root=mvp_root)

    after = {path: sha256_file(path) for path in bundle.rglob("*") if path.is_file()}
    assert result["schema"] == "video_knowledge_pipeline.video_closure_doctor.v1"
    assert result["ready_to_run"] is True
    assert result["auto_retry_count"] == 0
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 8 * 1024
    assert set(result["commands"]) == {"run", "resume", "status", "doctor"}
    assert before == after


def test_positive_closure_is_plan_first_candidate_only_and_review_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    mvp_root = _mvp_root(tmp_path)
    output = tmp_path / "closure-run"
    source_before = (bundle / "source.synthetic-video.fixture").read_bytes()
    monkeypatch.setattr("video_knowledge_pipeline.small_model_video_closure._invoke_mvp_bridge", lambda root, selected_bundle, target: _fake_edit_job(selected_bundle))

    result = run_video_closure(bundle, output_dir=output, mvp_root=mvp_root)

    assert result["status"] == "review_required"
    assert result["mode"] == "candidate_only"
    assert result["completed_stages"] == ["evidence", "segment_plan", "content_brief", "candidate_handoff", "review_packet"]
    assert result["auto_retry_count"] == 0
    assert (bundle / "source.synthetic-video.fixture").read_bytes() == source_before
    plan = json.loads((output / "artifacts" / "segment-plan.json").read_text(encoding="utf-8"))
    assert plan["editing_authorized"] is False
    assert plan["plan_precedes_editing"] is True
    assert plan["original_media_overwrite_allowed"] is False
    assert plan["window_plan"]
    assert any(row["default_action"] == "keep" for row in plan["low_confidence_items"])
    assert all(row["status"] == "review_required" for row in plan["human_chapter_boundaries"])
    handoff = json.loads((output / "artifacts" / "candidate-editing-handoff.json").read_text(encoding="utf-8"))
    assert handoff["mvp_contract"] == "video_creation_pipeline.vkp_edit_job.v3"
    assert handoff["candidate_only"] is True
    assert handoff["execute"] is False
    packet = json.loads((output / "artifacts" / "review-packet.json").read_text(encoding="utf-8"))
    assert packet["status"] == "review_required"
    assert packet["accepted"] is False
    assert packet["published"] is False


def test_pause_resume_is_explicit_hash_bound_and_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    mvp_root = _mvp_root(tmp_path)
    output = tmp_path / "closure-run"
    monkeypatch.setattr("video_knowledge_pipeline.small_model_video_closure._invoke_mvp_bridge", lambda root, selected_bundle, target: _fake_edit_job(selected_bundle))

    paused = run_video_closure(bundle, output_dir=output, mvp_root=mvp_root, pause_after="content_brief")
    assert paused["status"] == "paused"
    assert paused["next_stage"] == "candidate_handoff"
    with pytest.raises(VideoClosureError, match="explicit_resume_required"):
        run_video_closure(bundle, output_dir=output, mvp_root=mvp_root)

    resumed = resume_video_closure(output, mvp_root=mvp_root)
    hashes = {path.name: sha256_file(path) for path in (output / "artifacts").glob("*.json")}
    repeated = resume_video_closure(output, mvp_root=mvp_root)

    assert resumed["status"] == "review_required"
    assert repeated["status"] == "review_required"
    assert repeated["idempotent_no_op"] is True
    assert hashes == {path.name: sha256_file(path) for path in (output / "artifacts").glob("*.json")}


@pytest.mark.parametrize(("mutation", "error_code"), [("corrupt_media", "source_media_hash_mismatch"), ("corrupt_transcript", "corrupt_transcript"), ("timestamp_drift", "timestamp_outside_media"), ("missing_evidence", "missing_evidence")])
def test_bad_inputs_fail_closed_with_zero_retry(tmp_path: Path, mutation: str, error_code: str) -> None:
    bundle = _bundle(tmp_path)
    output = tmp_path / "closure-run"
    if mutation == "corrupt_media":
        (bundle / "source.synthetic-video.fixture").write_text("damaged", encoding="utf-8")
    elif mutation == "corrupt_transcript":
        _transcript_path(bundle).write_text("not-json", encoding="utf-8")
    elif mutation == "timestamp_drift":
        timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
        timeline[-1]["end"] = 24.01
        (bundle / "timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    else:
        (bundle / "timeline.json").write_text("[]", encoding="utf-8")

    with pytest.raises(VideoClosureError, match=error_code):
        run_video_closure(bundle, output_dir=output)

    failures = list((output / "failure-receipts").glob("*.json"))
    assert len(failures) == 1
    receipt = json.loads(failures[0].read_text(encoding="utf-8"))
    assert receipt["error_code"] == error_code
    assert receipt["auto_retry_count"] == 0
    assert receipt["status"] == "blocked"


def test_long_fixture_is_split_into_bounded_checkpoint_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    mvp_root = _mvp_root(tmp_path)
    media = bundle / "source.synthetic-video.fixture"
    media.write_bytes(b"VKP_SYNTHETIC_MEDIA_FIXTURE_V1\nduration_seconds=901.000\nfixture_only=true\n")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["source_video_sha256"] = sha256_file(media)
    manifest["media"]["duration_seconds"] = 901.0
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    transcript_path = _transcript_path(bundle)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    for row, bounds in zip(transcript["segments"], ((0.0, 301.0), (301.0, 602.0), (602.0, 901.0)), strict=True):
        row["start"], row["end"] = bounds
    for row, bounds in zip(timeline, ((0.0, 301.0), (301.0, 602.0), (602.0, 901.0)), strict=True):
        row["start"], row["end"] = bounds
    transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
    (bundle / "timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    monkeypatch.setattr("video_knowledge_pipeline.small_model_video_closure._invoke_mvp_bridge", lambda root, selected_bundle, target: _fake_edit_job(selected_bundle))

    run_video_closure(bundle, output_dir=tmp_path / "long-run", mvp_root=mvp_root)
    plan = json.loads((tmp_path / "long-run" / "artifacts" / "segment-plan.json").read_text(encoding="utf-8"))

    assert len(plan["window_plan"]) >= 3
    assert all(row["checkpoint_status"] == "planned" for row in plan["window_plan"])
    assert max(row["end"] - row["start"] for row in plan["window_plan"]) <= 301.0


def test_contract_drift_early_finalize_and_execution_request_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    mvp_root = _mvp_root(tmp_path)
    output = tmp_path / "closure-run"
    monkeypatch.setattr("video_knowledge_pipeline.small_model_video_closure._invoke_mvp_bridge", lambda root, selected_bundle, target: _fake_edit_job(selected_bundle))
    paused = run_video_closure(bundle, output_dir=output, mvp_root=mvp_root, pause_after="segment_plan")
    assert paused["status"] == "paused"

    with pytest.raises(VideoClosureError, match="premature_finalize"):
        finalize_video_closure(output)

    schema = mvp_root / "tests" / "fixtures" / "content-video-brief-v1" / "upstream-canonical" / "content-video-brief.v1.schema.json"
    schema.write_text(schema.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(VideoClosureError, match="downstream_contract_drift"):
        resume_video_closure(output, mvp_root=mvp_root)

    fresh = _bundle(tmp_path / "unauthorized")
    with pytest.raises(VideoClosureError, match="real_execution_not_authorized"):
        run_video_closure(fresh, output_dir=tmp_path / "unauthorized-run", execution_mode="render")


def test_cli_exposes_short_stable_video_closure_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["video-closure-doctor"]).command == "video-closure-doctor"
    assert parser.parse_args(["video-closure-run", "bundle"]).command == "video-closure-run"
    assert parser.parse_args(["video-closure-resume", "run"]).command == "video-closure-resume"
    assert parser.parse_args(["video-closure-status", "run"]).command == "video-closure-status"


def test_powershell_wrapper_routes_closure_before_provider_configuration() -> None:
    wrapper = (ROOT / "scripts" / "video-knowledge.ps1").read_text(encoding="utf-8")

    runtime_selection = wrapper.index('$venvPython = Join-Path $root ".venv\\Scripts\\python.exe"')
    closure_dispatch = wrapper.index('$offlineProjectCommands = @(')
    provider_env_parsing = wrapper.index('foreach ($envFile in @(')
    assert runtime_selection < closure_dispatch < provider_env_parsing
    assert '"video-closure-run"' in wrapper[closure_dispatch:provider_env_parsing]
    assert '& $pythonCommand -m video_knowledge_pipeline.cli @args' in wrapper[closure_dispatch:provider_env_parsing]
    assert 'PYTHONDONTWRITEBYTECODE' in wrapper[closure_dispatch:provider_env_parsing]


def test_cli_failure_is_compact_json_and_non_retrying(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = _bundle(tmp_path)
    _transcript_path(bundle).write_text("not-json", encoding="utf-8")

    exit_code = main(["video-closure-run", str(bundle), "--output-dir", str(tmp_path / "failed-run")])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["error_code"] == "corrupt_transcript"
    assert payload["auto_retry_count"] == 0


def test_failure_receipt_write_error_does_not_mask_primary_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        "video_knowledge_pipeline.small_model_video_closure._record_failure",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("synthetic receipt ACL")),
    )

    with pytest.raises(VideoClosureError, match="real_execution_not_authorized"):
        run_video_closure(bundle, output_dir=tmp_path / "failed-run", execution_mode="render")


def test_status_refuses_tampered_stage_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    mvp_root = _mvp_root(tmp_path)
    output = tmp_path / "closure-run"
    monkeypatch.setattr("video_knowledge_pipeline.small_model_video_closure._invoke_mvp_bridge", lambda root, selected_bundle, target: _fake_edit_job(selected_bundle))
    run_video_closure(bundle, output_dir=output, mvp_root=mvp_root)
    evidence = output / "artifacts" / "evidence-pack.json"
    evidence.write_text(evidence.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(VideoClosureError, match="stage_artifact_drift"):
        video_closure_status(output)


def test_completed_run_refuses_stale_input_on_status_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    mvp_root = _mvp_root(tmp_path)
    output = tmp_path / "closure-run"
    monkeypatch.setattr("video_knowledge_pipeline.small_model_video_closure._invoke_mvp_bridge", lambda root, selected_bundle, target: _fake_edit_job(selected_bundle))
    run_video_closure(bundle, output_dir=output, mvp_root=mvp_root)
    transcript_path = _transcript_path(bundle)
    transcript_path.write_text(transcript_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(VideoClosureError, match="input_drift"):
        video_closure_status(output)
    with pytest.raises(VideoClosureError, match="input_drift"):
        resume_video_closure(output, mvp_root=mvp_root)
