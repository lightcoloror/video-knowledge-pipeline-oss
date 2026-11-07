from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.video_workbench import export_video_workbench


def test_creative_contract_cli_front_doors() -> None:
    parser = build_parser()
    generation = parser.parse_args(
        [
            "import-generation-contracts",
            "bundle",
            "--task",
            "task.json",
            "--receipt",
            "receipt.json",
            "--validation",
            "validation.json",
            "--source-root",
            "contracts",
        ]
    )
    assert generation.command == "import-generation-contracts"
    assert generation.source_root == ["contracts"]
    previs = parser.parse_args(
        [
            "import-previs-candidate",
            "bundle",
            "--scene",
            "scene.json",
            "--capture-manifest",
            "captures.json",
            "--validation",
            "validation.json",
        ]
    )
    assert previs.command == "import-previs-candidate"


def test_workbench_reads_generation_and_previs_candidate_cards(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "title": "fixture",
            "generation_contract_import_json": "exports/generation-contract-import.json",
            "previs_candidate_evidence_json": "exports/previs-candidate-evidence.json",
        },
    )
    write_json(root / "timeline.json", [])
    write_json(
        exports / "generation-contract-import.json",
        {
            "status": "completed",
            "generator": "hyperframes",
            "required_capability": "hyperframes_render",
            "capability_preflight": {"ready": True, "probe_method": "registered_tool_health_check"},
            "technical_verification": {"probe_method": "ffprobe-json"},
            "visual_verification": {"inspected": True, "representative_frames": [{"path": "frame.png"}]},
        },
    )
    write_json(
        exports / "previs-candidate-evidence.json",
        {
            "status": "needs_review",
            "scene": {"scene_id": "scene-1"},
            "cameras": [{"camera_id": "camera-main"}],
            "captures": [{"capture_id": "capture-1"}],
            "authority_boundary": {"synthetic": True, "observed_video_fact": False},
        },
    )

    result = export_video_workbench(root, write=False)

    assert result["creative_workflow"]["generation"]["generator"] == "hyperframes"
    assert result["creative_workflow"]["generation"]["visual_verification"]["inspected"] is True
    assert result["creative_workflow"]["previs"]["camera_count"] == 1
    assert result["creative_workflow"]["previs"]["authority_boundary"]["observed_video_fact"] is False
