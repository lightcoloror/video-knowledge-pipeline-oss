from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import video_knowledge_pipeline.personal_vlog_runtime as runtime
from video_knowledge_pipeline.personal_vlog import (
    PersonalVlogDriftError,
    build_asset_evidence,
    load_personal_vlog_project,
)
from video_knowledge_pipeline.personal_vlog_handoff import (
    MVP_V4_SCHEMA,
    build_mvp_v4_candidate_handoff,
    validate_mvp_v4_candidate_handoff,
)
from video_knowledge_pipeline.personal_vlog_otio import (
    export_recipe_otio,
    opentimelineio_status,
)
from video_knowledge_pipeline.personal_vlog_runtime import (
    STAGES,
    personal_vlog_doctor,
    personal_vlog_prepare,
    personal_vlog_resume,
    personal_vlog_run,
    personal_vlog_status,
)


def _file(root: Path, name: str, payload: bytes, seconds: int) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    stamp = 1_700_010_000 + seconds
    os.utime(path, (stamp, stamp))
    return path


def _prepared(tmp_path: Path) -> tuple[Path, Path, dict]:
    event = tmp_path / "synthetic-event"
    project = tmp_path / "vlog-project"
    event.mkdir()
    _file(event, "photo.JPG", b"not-a-real-image-but-bounded", 1)
    _file(event, "clip.MP4", b"not-a-real-video-but-bounded", 2)
    _file(event, "sound.WAV", b"not-a-real-audio-but-bounded", 3)
    _file(event, "broken.MOV", b"broken-media", 4)
    manifest = personal_vlog_prepare(event, project)
    return event, project, manifest


def test_doctor_is_local_only_and_declares_optional_gaps():
    doctor = personal_vlog_doctor()

    assert doctor["profile"] == "personal-vlog-v1"
    assert doctor["ready_to_prepare"] is True
    assert doctor["ready_for_candidate_recipe"] is (not doctor["blockers"])
    assert doctor["ready_for_full_local_evidence"] is (
        not doctor["blockers"] and not doctor["full_local_evidence_gaps"]
    )
    assert doctor["policy"] == {
        "local_first": True,
        "remote_fallback": False,
        "automatic_model_download": False,
        "source_mutation": False,
        "automatic_render": False,
        "automatic_upload": False,
        "automatic_publish": False,
    }
    assert doctor["otio"]["serialization_fallback"] is False


def test_one_command_synthetic_run_builds_three_recipes_handoffs_and_review(tmp_path: Path):
    _event, project, _manifest = _prepared(tmp_path)

    result = personal_vlog_run(project, presets="all")

    assert result["status"] == "complete_draft"
    assert result["review_required"] is True
    assert result["accepted"] is False
    assert result["completed_stages"] == list(STAGES)
    run_dir = Path(result["run_dir"])
    for preset in ("highlight", "standard", "long"):
        recipe = json.loads((run_dir / f"edit-recipe-{preset}.json").read_text("utf-8"))
        handoff = json.loads((run_dir / f"mvp-v4-handoff-{preset}.json").read_text("utf-8"))
        assert recipe["output_mode"] == "candidate_only"
        assert handoff["schema"] == MVP_V4_SCHEMA
        validate_mvp_v4_candidate_handoff(handoff)
    packet = json.loads((run_dir / "review-packet.json").read_text("utf-8"))
    markdown = (run_dir / "review-packet.md").read_text("utf-8")
    workbench = json.loads((run_dir / "personal-vlog-workbench.json").read_text("utf-8"))
    workbench_html = (run_dir / "personal-vlog-workbench.html").read_text("utf-8")
    assert packet["status"] == "complete_draft"
    assert packet["privacy"]["source_paths_exported"] is False
    assert str(project) not in markdown
    assert str(_event) not in markdown
    assert "自动渲染、上传、发布：否" in markdown
    assert workbench["surface"] == "video-workbench:personal-vlog-v1"
    assert workbench["source_jump_available"] is True
    assert workbench["actions"]["render"] == "not_available_in_vkp"
    assert "未入选不代表删除" in workbench_html
    assert "data-vkp-surface=\"personal-vlog-v1\"" in workbench_html
    evidence = json.loads((run_dir / "asset-evidence.json").read_text("utf-8"))
    assert any(
        "media_probe_failed" in row.get("limitations", [])
        for row in evidence["assets"]
    )
    assert any(
        "filmed_v1_child_bundle_not_completed" in row.get("limitations", [])
        for row in evidence["assets"]
        if row["media_type"] == "video"
    )


def test_video_analysis_is_shared_across_candidate_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _event, project, manifest = _prepared(tmp_path)
    video = next(asset for asset in manifest["assets"] if asset["media_type"] == "video")
    calls: list[dict[str, object]] = []

    def fake_prepare(source, output_dir, **kwargs):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        transcript = output / "normalized-transcript.json"
        transcript.write_text(json.dumps({"segments": [{"text": "完整本地分析转写"}]}), encoding="utf-8")
        calls.append({"source": str(source), "output_dir": str(output), **kwargs})
        return {
            "ok": True,
            "json_path": str(output / "video-knowledge-run.json"),
            "transcript_path": str(transcript),
            "initial_bundle": {"status": "ok", "bundle_dir": str(output / "webui-bundle")},
        }

    monkeypatch.setattr(runtime, "prepare_local_video_run", fake_prepare)
    monkeypatch.setattr(
        runtime,
        "_probe_assets",
        lambda _project: {str(video["asset_id"]): {"probe_status": "ok", "technical_quality": 0.6}},
    )
    monkeypatch.setattr(
        runtime,
        "_probe_av",
        lambda _path, _executable, _kind: {
            "probe_status": "ok",
            "duration_seconds": 12.0,
            "width": 1080,
            "height": 1920,
            "technical_quality": 0.6,
        },
    )

    shared = runtime._run_shared_analysis(
        load_personal_vlog_project(project), tmp_path / "shared-analysis-run"
    )

    assert calls and calls[0]["execute_asr"] is True
    assert calls[0]["build_initial_bundle"] is True
    assert calls[0]["copy_media"] is False
    assert shared["preset_independent"] is True
    assert shared["assets"][0]["child_bundle_status"] == "completed"
    evidence = build_asset_evidence(
        load_personal_vlog_project(project),
        imported=runtime._merge_shared_analysis(
            {str(video["asset_id"]): {"probe_status": "ok", "technical_quality": 0.6}},
            shared,
        ),
    )
    analyzed = next(row for row in evidence["assets"] if row["asset_id"] == video["asset_id"])
    assert analyzed["child_bundle_status"] == "completed"
    assert analyzed["semantic"]["asr_excerpt"] == "完整本地分析转写"


def test_pause_resume_is_explicit_idempotent_and_skips_completed_stages(tmp_path: Path):
    _event, project, _manifest = _prepared(tmp_path)
    paused = personal_vlog_run(project, stop_after="evidence")

    assert paused["status"] == "paused"
    assert paused["completed_stages"] == ["inventory", "shared_analysis", "evidence"]
    run_id = str(paused["vlog_run_id"])
    evidence_path = Path(paused["run_dir"]) / "asset-evidence.json"
    first_hash = evidence_path.read_bytes()

    resumed = personal_vlog_resume(project, run_id)
    repeated = personal_vlog_resume(project, run_id)

    assert resumed["status"] == "complete_draft"
    assert repeated["status"] == "complete_draft"
    assert first_hash == evidence_path.read_bytes()
    assert resumed["automatic_retry_count"] == 0
    assert personal_vlog_status(project, run_id)["status"] == "complete_draft"


def test_injected_failure_is_receipted_and_explicit_resume_completes(tmp_path: Path):
    _event, project, _manifest = _prepared(tmp_path)

    failed = personal_vlog_run(project, fail_after_stage="event_plan")

    assert failed["status"] == "failed"
    assert failed["automatic_retry_count"] == 0
    assert failed["completed_stages"] == ["inventory", "shared_analysis", "evidence", "event_plan"]
    assert failed["recovery_command"].startswith("personal-vlog-resume ")
    recovered = personal_vlog_resume(project, str(failed["vlog_run_id"]))
    assert recovered["status"] == "complete_draft"
    assert recovered["completed_stages"] == list(STAGES)


def test_resume_rejects_project_manifest_and_source_drift(tmp_path: Path):
    event, project, _manifest = _prepared(tmp_path)
    paused = personal_vlog_run(project, stop_after="inventory")
    run_id = str(paused["vlog_run_id"])
    (event / "photo.JPG").write_bytes(b"source-drift")

    with pytest.raises(PersonalVlogDriftError, match="source_hash_drift"):
        personal_vlog_resume(project, run_id)


def test_resume_rejects_completed_recipe_and_receipt_drift(tmp_path: Path):
    _event, project, _manifest = _prepared(tmp_path)
    paused = personal_vlog_run(project, presets="highlight", stop_after="recipes")
    run_id = str(paused["vlog_run_id"])
    run_dir = Path(paused["run_dir"])
    recipe_path = run_dir / "edit-recipe-highlight.json"
    recipe = json.loads(recipe_path.read_text("utf-8"))
    recipe["clips"] = []
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")

    with pytest.raises(PersonalVlogDriftError, match="artifact_hash_drift"):
        personal_vlog_resume(project, run_id)

    second_root = tmp_path / "second"
    second_root.mkdir()
    _event2, project2, _manifest2 = _prepared(second_root)
    paused2 = personal_vlog_run(project2, stop_after="evidence")
    receipt_path = Path(paused2["run_dir"]) / "receipts" / "02-evidence.json"
    receipt_path.write_text("{}", encoding="utf-8")

    with pytest.raises(PersonalVlogDriftError, match="receipt_hash_drift"):
        personal_vlog_resume(project2, str(paused2["vlog_run_id"]))


def test_handoff_is_multi_source_path_redacted_and_hash_bound(tmp_path: Path):
    _event, project_dir, manifest = _prepared(tmp_path)
    result = personal_vlog_run(project_dir, presets="highlight")
    run_dir = Path(result["run_dir"])
    recipe = json.loads((run_dir / "edit-recipe-highlight.json").read_text("utf-8"))

    handoff = build_mvp_v4_candidate_handoff(manifest, recipe)
    raw = json.dumps(handoff, ensure_ascii=False)

    assert handoff["schema"] == MVP_V4_SCHEMA
    assert handoff["recipe"]["recipe_sha256"] == recipe["recipe_sha256"]
    assert len(handoff["source_assets"]) >= 3
    assert '"source_path":' not in raw
    assert str(project_dir) not in raw
    assert all(row["status"] == "candidate_only" for row in handoff["candidate_clips"])


def test_otio_uses_official_api_or_returns_explicit_degraded_receipt(tmp_path: Path):
    _event, project, _manifest = _prepared(tmp_path)
    result = personal_vlog_run(project, presets="highlight")
    recipe = json.loads(
        (Path(result["run_dir"]) / "edit-recipe-highlight.json").read_text("utf-8")
    )
    status = opentimelineio_status()
    output = tmp_path / "candidate.otio"

    receipt = export_recipe_otio(recipe, output)

    if status["available"]:
        assert receipt["round_trip_verified"] is True
        assert output.is_file()
        assert str(project) not in output.read_text("utf-8")
    else:
        assert receipt["status"] == "degraded"
        assert receipt["serialization_fallback"] is False
        assert receipt["review_gap"].startswith("opentimelineio_not_installed")
        assert not output.exists()
