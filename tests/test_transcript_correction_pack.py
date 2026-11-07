import json
from pathlib import Path

from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.transcript_correction_pack import build_transcript_correction_pack


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(
        bundle / "manifest.json",
        {
            "title": "浏览器自动化横评",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(
        bundle / "normalized-transcript.json",
        {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "brother mc p"},
                {"start": 3.0, "end": 5.0, "text": "play right client"},
            ]
        },
    )
    return bundle


def test_transcript_correction_pack_preview_writes_messages_and_manifest(tmp_path: Path):
    bundle = _bundle(tmp_path)

    result = build_transcript_correction_pack(bundle, write=True)

    assert result["status"] == "planned"
    assert result["segment_count"] == 2
    assert result["run_registry"]["run_type"] == "transcript_correction_pack"
    assert result["run_registry"]["status"] == "needs_execution"
    assert result["run_registry"]["parameters"]["segment_count"] == 2
    assert (bundle / "runs" / "transcript-correction-pack" / "run.json").exists()
    assert (bundle / "exports" / "transcript-correction-pack.json").exists()
    assert (bundle / "exports" / "transcript-correction-llm-messages.json").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["transcript_correction_pack_json"] == "exports/transcript-correction-pack.json"
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    run_types = {row["run_type"]: row["status"] for row in registry["runs"]}
    assert run_types["transcript_correction_pack"] == "needs_execution"


def test_transcript_correction_pack_imports_json_without_overwriting_term_resolution(tmp_path: Path):
    bundle = _bundle(tmp_path)
    correction = tmp_path / "correction.json"
    write_json(
        correction,
        {
            "segments": [
                {"index": 0, "text": "Browser MCP"},
                {"index": 1, "text": "Playwright client"},
            ]
        },
    )

    result = build_transcript_correction_pack(bundle, input_json=correction, write=True)

    assert result["status"] == "imported"
    assert result["correction_summary"]["corrected_segments"] == 2
    assert result["run_registry"]["status"] == "completed"
    assert result["run_registry"]["failed_items"] == []
    artifact_keys = {row["key"] for row in result["run_registry"]["artifacts"]}
    assert "corrected_json" in artifact_keys
    assert "pack_json" in artifact_keys
    assert (bundle / "llm-corrected-transcript.json").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm_corrected_transcript_json"] == "llm-corrected-transcript.json"
    assert "corrected_transcript_json" not in manifest
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    run_types = {row["run_type"]: row["status"] for row in registry["runs"]}
    assert run_types["transcript_correction_pack"] == "completed"
