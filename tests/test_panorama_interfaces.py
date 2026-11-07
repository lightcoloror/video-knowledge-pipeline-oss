from __future__ import annotations

import inspect
import json
from pathlib import Path

from video_knowledge_pipeline.panorama_cli import (
    panorama_prepare,
    panorama_run,
    panorama_status,
    prepare_main,
)
from video_knowledge_pipeline.panorama_mcp import (
    panorama_doctor_tool,
    panorama_import_tool,
    panorama_prepare_tool,
    panorama_restore_tool,
    panorama_run_tool,
    panorama_status_tool,
)


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "bundle with spaces"
    root.mkdir()
    source = tmp_path / "source with spaces.mp4"
    source.write_bytes(b"synthetic panorama")
    (root / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}), encoding="utf-8")
    (root / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 10.0,
                    "visual_route": "mixed",
                    "video_key": str(source),
                    "video_duration_seconds": 10.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    return root, source


def test_prepare_no_write_is_powershell_path_safe_and_does_not_create_artifacts(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)

    result = panorama_prepare(
        bundle,
        projection="equirectangular_mono",
        indexes=[1],
        frame_count=5,
        limit=1,
        face_size=64,
        run_id="no-write",
        write=False,
    )

    assert result["write_performed"] is False
    assert "bundle with spaces" in result["bundle_dir"]
    assert not Path(result["plan_path"]).exists()
    assert result["viewport_count"] == 30


def test_prepare_cli_accepts_comma_indexes_and_outputs_json(tmp_path: Path, capsys: object) -> None:
    bundle, _ = _bundle(tmp_path)

    prepare_main(
        [
            str(bundle),
            "--projection",
            "equirectangular_mono",
            "--indexes",
            "1",
            "--frame-count",
            "5",
            "--limit",
            "1",
            "--face-size",
            "64",
            "--run-id",
            "cli",
            "--no-write",
        ]
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["run_id"] == "cli"
    assert output["selected_indexes"] == [1]


def test_prepare_run_status_are_short_stable_interfaces(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    prepared = panorama_prepare(
        bundle,
        projection="equirectangular_mono",
        frame_count=5,
        limit=1,
        face_size=64,
        run_id="stable",
        write=True,
    )

    run = panorama_run(bundle, run_id="stable", execute_projection=False)
    status = panorama_status(bundle, run_id="stable")

    assert Path(prepared["plan_path"]).is_file()
    assert run["status"] == "planned"
    assert run["remote_calls_made"] == 0
    assert run["next_action"]["key"] == "execute_projection"
    assert status["status"] == "projection_pending"
    assert "--execute-projection" in status["recovery_command"]


def test_panorama_mcp_surface_matches_six_public_commands_without_provider_calls() -> None:
    tools = {
        "panorama_doctor": panorama_doctor_tool,
        "panorama_prepare": panorama_prepare_tool,
        "panorama_run": panorama_run_tool,
        "panorama_import": panorama_import_tool,
        "panorama_status": panorama_status_tool,
        "panorama_restore": panorama_restore_tool,
    }

    assert set(tools) == {
        "panorama_doctor",
        "panorama_prepare",
        "panorama_run",
        "panorama_import",
        "panorama_status",
        "panorama_restore",
    }
    assert "execute_projection" in inspect.signature(panorama_run_tool).parameters
    assert "input_json" in inspect.signature(panorama_import_tool).parameters
    assert "confirm_run_id" in inspect.signature(panorama_restore_tool).parameters


def test_panorama_discovery_registers_only_real_local_trial_interfaces() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "agent-tool-manifest.v1.json").read_text(encoding="utf-8"))
    discovery = (root / "AGENT_DISCOVERY.md").read_text(encoding="utf-8")
    commands = {row.get("command") for row in manifest["interfaces"]["canonical"]}

    assert manifest["manifest_version"] == "1.2.0"
    assert "panorama_projection_analysis_local_trial" in manifest["capabilities"]
    assert {
        "panorama-doctor <media-or-bundle>",
        "panorama-prepare <bundle> --projection <type>",
        "panorama-run <bundle> --run-id <id> [--execute-projection]",
        "panorama-import <bundle> --run-id <id> --input-json <result.json>",
        "panorama-status <bundle> [--run-id <id>]",
        "panorama-restore <bundle> --run-id <id> [--execute --confirm-run-id <id>]",
    } <= commands
    assert any(row.get("command") == "video-knowledge-panorama-mcp" for row in manifest["interfaces"]["mcp"])
    assert "FFmpeg `v360` is the only production projection backend" in discovery
    assert "candidate_only" in discovery and "review_required" in discovery
