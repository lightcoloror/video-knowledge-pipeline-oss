from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from video_knowledge_pipeline import autopilot
from video_knowledge_pipeline.autopilot import (
    SUBMISSION_SCHEMA,
    autopilot_next,
    autopilot_start,
    autopilot_status,
    autopilot_submit,
)
from video_knowledge_pipeline.bundle_next import bundle_next_action
from video_knowledge_pipeline.panorama_analysis import PANORAMA_ANALYSIS_PLAN_SCHEMA
from video_knowledge_pipeline.panorama_projection import (
    PANORAMA_PROJECTION_PLAN_SCHEMA,
    PANORAMA_VIEWPORT_PACK_SCHEMA,
    viewport_pack_sha256,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, *, ready_profile: bool = True, temporal: bool = False) -> Path:
    root = tmp_path / "panorama-bundle"
    root.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic-panorama-source")
    profile = {
        "schema": "video_knowledge_pipeline.panorama_profile.v1",
        "projection": "equirectangular_mono" if ready_profile else "unknown",
        "stereo_mode": "mono",
        "input_fov": None,
        "lens_order": "",
        "projection_source": "explicit_override" if ready_profile else "unresolved",
        "metadata_confidence": "unavailable",
        "metadata_evidence": [],
        "aspect_ratio_hint": "",
        "issues": [] if ready_profile else [{"key": "projection_required", "message": "confirm projection"}],
        "ready": ready_profile,
        "profile_sha256": "a" * 64,
    }
    manifest = {
        "schema": "lecture_webui_bundle.v1",
        "panorama_profile": profile,
    }
    timeline = [
        {
            "index": 1,
            "start": 0.0,
            "end": 10.0,
            "visual_route": "mixed",
            "video_key": str(source),
            "video_duration_seconds": 10.0,
            "transcript": "synthetic speech remains available",
        }
    ]
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "timeline.json", timeline)
    if not ready_profile:
        return root

    run_root = root / "runs" / "panorama-fixture"
    projection_plan = {
        "schema": PANORAMA_PROJECTION_PLAN_SCHEMA,
        "run_id": "fixture",
        "bundle_dir": str(root),
        "profile": profile,
        "selected_indexes": [1],
        "plan_sha256": "b" * 64,
    }
    _write_json(run_root / "projection-plan.json", projection_plan)
    directions = {
        "front": (0.0, 0.0),
        "right": (90.0, 0.0),
        "back": (180.0, 0.0),
        "left": (-90.0, 0.0),
        "up": (0.0, 90.0),
        "down": (0.0, -90.0),
    }
    viewports: list[dict[str, object]] = []
    for sample in range(1, 6):
        for ordinal, (direction, (yaw, pitch)) in enumerate(directions.items()):
            path = run_root / "viewports" / "0001" / f"sample-{sample:02d}" / f"{direction}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (sample * 10, ordinal * 20, 30)).save(path)
            viewports.append(
                {
                    "viewport_id": f"vp-1-{sample}-{direction}",
                    "timeline_index": 1,
                    "sample_index": sample,
                    "timestamp_seconds": float(sample),
                    "direction": direction,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": 0.0,
                    "h_fov": 90.0,
                    "v_fov": 90.0,
                    "path": path.relative_to(root).as_posix(),
                    "source_sha256": "c" * 64,
                    "filter": "v360=input=e:output=flat",
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    pack = {
        "schema": PANORAMA_VIEWPORT_PACK_SCHEMA,
        "run_id": "fixture",
        "profile": profile,
        "plan_sha256": projection_plan["plan_sha256"],
        "manifest_sha256": _sha256(root / "manifest.json"),
        "timeline_sha256": _sha256(root / "timeline.json"),
        "source_sha256": "c" * 64,
        "selected_indexes": [1],
        "segment_count": 1,
        "sample_count": 5,
        "frame_count": 5,
        "face_size": 16,
        "viewport_count": len(viewports),
        "viewports": viewports,
        "geometric_coverage_sha256": "d" * 64,
    }
    pack["task_pack_sha256"] = viewport_pack_sha256(pack)
    _write_json(run_root / "viewport-pack.json", pack)

    tasks: list[dict[str, object]] = [
        {
            "task_id": "panorama-semantic-000001-003",
            "task_type": "panorama_six_view_semantic",
            "timeline_index": 1,
            "sample_index": 3,
            "frame_paths": [
                str((root / str(row["path"])).resolve())
                for row in viewports
                if row["sample_index"] == 3
            ],
            "route": {"status": "ready", "route": "codex_session", "unattended": False},
        }
    ]
    if temporal:
        tasks.append(
            {
                "task_id": "panorama-temporal-000001-right",
                "task_type": "panorama_temporal_direction",
                "timeline_index": 1,
                "direction": "right",
                "change_score": 0.5,
                "frame_paths": [
                    str((root / str(row["path"])).resolve())
                    for row in viewports
                    if row["direction"] == "right"
                ],
                "representative_frame_paths": [],
                "route": {"status": "ready", "route": "codex_session", "unattended": False},
            }
        )
    analysis = {
        "schema": PANORAMA_ANALYSIS_PLAN_SCHEMA,
        "status": "ready",
        "bundle_dir": str(root),
        "viewport_pack_path": str(run_root / "viewport-pack.json"),
        "viewport_pack_sha256": pack["task_pack_sha256"],
        "selected_indexes": [1],
        "temporal_required_indexes": [1] if temporal else [],
        "semantic_task_count": 1,
        "temporal_task_count": int(temporal),
        "tasks": tasks,
        "review_gaps": [],
        "plan_sha256": "e" * 64,
    }
    analysis_path = root / "runs" / "panorama-analysis-fixture" / "analysis-plan.json"
    _write_json(analysis_path, analysis)
    manifest.update(
        {
            "panorama_projection_plan": {
                "run_id": "fixture",
                "plan_sha256": projection_plan["plan_sha256"],
                "path": run_root.relative_to(root).as_posix() + "/projection-plan.json",
            },
            "panorama_analysis_plan": {
                "selected_indexes": [1],
                "temporal_required_indexes": [1] if temporal else [],
                "plan_sha256": analysis["plan_sha256"],
                "path": analysis_path.relative_to(root).as_posix(),
            },
        }
    )
    _write_json(root / "manifest.json", manifest)
    return root


def _fill_result(work_item: dict[str, object], *, unavailable: bool = False) -> dict[str, object]:
    template_ref = next(
        row for row in work_item["input_refs"] if isinstance(row, dict) and row.get("role") == "result_template_path"
    )
    result = json.loads(Path(str(template_ref["path"])).read_text(encoding="utf-8"))
    result["model_actual"] = "gpt-5.6-luna"
    for row in result["items"]:
        row["evidence_level"] = "unavailable" if unavailable else "confirmed"
        row["unavailable_reason"] = "synthetic backend gap" if unavailable else ""
        row["observed_frame_ids"] = list(row["expected_frame_ids"])
        if row["kind"] == "semantic":
            row["visual_understanding"].update(
                {
                    "objects": ["synthetic panorama"],
                    "confidence": 0.8,
                    "evidence_frame_paths": list(row["frame_paths"]),
                }
            )
            for observation in row.get("panorama_viewport_observations") or []:
                observation["observed"] = True
                observation["observation"] = "synthetic view"
        else:
            observations = [
                {"frame_id": frame_id, "observation": "synthetic temporal frame"}
                for frame_id in row["expected_frame_ids"]
            ]
            row["temporal_visual_understanding"].update(
                {
                    "event_sequence": ["synthetic change"],
                    "observed_frame_count": len(observations),
                    "observed_frame_ids": list(row["expected_frame_ids"]),
                    "per_frame_observations": observations,
                    "evidence_frame_paths": list(row["frame_paths"]),
                    "confidence": 0.7,
                }
            )
    return result


def _submission(work_item: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    return {
        "schema": SUBMISSION_SCHEMA,
        "job_id": work_item["job_id"],
        "bundle_identity_sha256": work_item["bundle_identity_sha256"],
        "state_revision": work_item["state_revision"],
        "task_id": work_item["task_id"],
        "lease_id": work_item["lease_id"],
        "result": result,
    }


def test_bundle_next_blocks_ambiguous_projection_before_any_model_action(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, ready_profile=False)

    result = bundle_next_action(bundle, refresh=False)

    assert result["next_action"]["key"] == "panorama_projection_confirmation"
    assert result["next_action"]["kind"] == "panorama_blocker"
    assert result["next_action"]["human_required"] is True


def test_autopilot_leases_exactly_one_panorama_task_and_resume_does_not_duplicate_it(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    job_root = tmp_path / "jobs"
    started = autopilot_start(bundle, job_root=job_root)

    first = autopilot_next(started["job_id"], job_root=job_root)
    repeated = autopilot_next(started["job_id"], job_root=job_root)

    assert first["status"] == "waiting_model"
    assert first["work_item"]["task_type"] == "codex_vision_session"
    assert first["work_item"]["action"]["key"] == "panorama-semantic-000001-003"
    assert repeated["work_item"]["task_id"] == first["work_item"]["task_id"]
    assert autopilot_status(started["job_id"], job_root=job_root)["budget_usage"]["model_calls"] == 0


def test_autopilot_imports_semantic_then_one_directional_temporal_task(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, temporal=True)
    job_root = tmp_path / "jobs"
    started = autopilot_start(bundle, job_root=job_root)

    semantic = autopilot_next(started["job_id"], job_root=job_root)["work_item"]
    receipt = autopilot_submit(
        started["job_id"],
        submission=_submission(semantic, _fill_result(semantic)),
        job_root=job_root,
    )
    temporal = autopilot_next(started["job_id"], job_root=job_root)["work_item"]

    assert receipt["status"] == "succeeded"
    assert temporal["action"]["key"] == "panorama-temporal-000001-right"
    assert len([row for row in temporal["input_refs"] if row["role"].startswith("panorama_frame_")]) == 5

    temporal_receipt = autopilot_submit(
        started["job_id"],
        submission=_submission(temporal, _fill_result(temporal)),
        job_root=job_root,
    )
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert temporal_receipt["status"] == "succeeded"
    assert timeline[0]["panorama_temporal_understanding"]["tracks"][0]["direction"] == "right"


def test_background_local_executor_cannot_take_over_panorama_session(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    job_root = tmp_path / "jobs"

    def forbidden_executor(work_item):
        raise AssertionError("interactive panorama was sent to the flat background executor")

    monkeypatch.setattr(autopilot, "_local_background_ready", lambda job: True)
    started = autopilot_start(bundle, runner="background", execution_route="local-only", job_root=job_root)
    result = autopilot_next(started["job_id"], job_root=job_root, background_executor=forbidden_executor)

    assert result["status"] == "waiting_model"
    assert result["work_item"]["task_type"] == "codex_vision_session"
    assert len([row for row in result["work_item"]["input_refs"] if row["role"].startswith("panorama_frame_")]) == 6
    assert autopilot_status(started["job_id"], job_root=job_root)["budget_usage"]["model_calls"] == 0


def test_unavailable_panorama_backend_becomes_gap_then_flat_outputs_continue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = _bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    plan_path = bundle / manifest["panorama_analysis_plan"]["path"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["status"] = "review_gap"
    plan["tasks"][0]["route"] = {
        "status": "unavailable",
        "route": "review_gap",
        "reason": "no backend",
    }
    plan["review_gaps"] = [{"index": 1, "key": "panorama_visual_backend_unavailable"}]
    _write_json(plan_path, plan)
    original = bundle_next_action

    def next_action(root: Path, *, refresh: bool = True, include_panorama: bool = True):
        result = original(root, refresh=False, include_panorama=include_panorama)
        if result["next_action"].get("kind") != "panorama_review_gap":
            return {
                "status": "ready",
                "next_action": {
                    "status": "ready",
                    "kind": "export",
                    "key": "refresh_review_outputs",
                    "mcp_tool": "refresh_lecture_review_outputs",
                },
            }
        return result

    monkeypatch.setattr(autopilot, "bundle_next_action", next_action)
    started = autopilot_start(bundle, job_root=tmp_path / "jobs")

    finished = autopilot_next(started["job_id"], job_root=tmp_path / "jobs")

    assert finished["state"] == "complete_draft"
    assert "panorama_visual_backend_unavailable" in {row["key"] for row in finished["review_gaps"]}
    assert finished["budget_usage"]["model_calls"] == 0
    assert json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))[0]["transcript"]
