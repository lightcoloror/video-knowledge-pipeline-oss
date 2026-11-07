"""Synthetic-only readiness and completion regressions for the existing parser lane."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import visual_structure as vs
from video_knowledge_pipeline.storage import write_json


def _candidate(index: int, *, recovered: bool = False) -> dict:
    return {"index": index, "start": index - 1, "end": index,
            "visual_route": "document_visual", "material_types": ["slide"],
            "visual_text": "Synthetic courseware: alpha has a detailed definition." if recovered else ""}


def _bundle(tmp_path: Path, count: int = 2) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    write_json(root / "timeline.json", [_candidate(i) for i in range(1, count + 1)])
    return root


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(vs, "tool_source_review_root", lambda: tmp_path / "sources")
    monkeypatch.setattr(vs, "workspace_root", lambda: tmp_path / "workspace")
    monkeypatch.setattr(vs.shutil, "which", lambda _name: None)


def test_source_checkout_is_not_an_executable_backend(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    (tmp_path / "sources" / "docling").mkdir(parents=True)
    row = next(item for item in vs._tool_statuses() if item["name"] == "docling")
    assert row["available"] is False
    assert row["local_root_exists"] is True
    assert row["runtime_status"] == "command_missing"
    assert row["ready_to_execute"] is False


def test_command_discovery_does_not_claim_models_or_runtime_validated(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    command = str(tmp_path / "docling.exe")
    monkeypatch.setattr(vs.shutil, "which", lambda name: command if name == "docling" else None)
    row = next(item for item in vs._tool_statuses() if item["name"] == "docling")
    assert row["available"] is True  # Backward-compatible command presence, not readiness.
    assert row["command_available"] is True
    assert row["runtime_status"] == "not_probed"
    assert row["ready_to_execute"] is False
    assert row["model_status"] == "not_assessed"
    assert row["probe_executed"] is False
    assert "runtime_not_probed" in row["blockers"]


def test_unverified_handoff_cannot_recommend_direct_execution_or_upload() -> None:
    steps = vs._visual_structure_next_steps([{"name": "docling", "available": True}], "import-command")
    assert steps[0]["action_code"] == "verify_local_parser_runtime"
    assert steps[0]["execution_allowed"] is False
    assert "commercial" not in json.dumps(steps)
    assert "Run an available" not in json.dumps(steps)
    assert steps[-1]["command"] == "import-command"


@pytest.mark.parametrize("recovered,updated,expected", [
    (0, 1, "needs_structure"), (1, 1, "partially_structured"),
    (2, 0, "structured"), (2, 1, "structured"),
])
def test_completion_uses_actual_selected_evidence_not_update_count(recovered, updated, expected) -> None:
    candidates = [_candidate(i, recovered=i <= recovered) for i in (1, 2)]
    assert vs._visual_structure_handoff_status({"total_candidates": 2, "updated": updated}, candidates) == expected


def test_mismatched_summary_count_fails_closed() -> None:
    assert vs._visual_structure_handoff_status({"total_candidates": 1, "updated": 1}, []) == "needs_structure"


def test_partial_import_refreshes_selected_rows_and_keeps_remaining_work(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    root = _bundle(tmp_path)
    source = tmp_path / "import.json"
    write_json(source, {"items": [{"index": 1, "type": "slide", "markdown": _candidate(1, recovered=True)["visual_text"]}]})
    result = vs.run_visual_structure_plan(root, input_json=source)
    handoff = json.loads(Path(result["handoff_json_path"]).read_text(encoding="utf-8"))
    run = json.loads((root / "runs" / "visual-structure-ebook" / "run.json").read_text(encoding="utf-8"))
    assert result["summary"]["updated"] == 1
    assert result["items"][0]["visual_text"]
    assert handoff["status"] == "partially_structured"
    assert handoff["unresolved_indexes"] == [2]
    assert handoff["completion_scope"] == "selected_candidates_only"
    assert result["summary"]["structured_count"] == 1
    assert result["summary"]["unresolved_indexes"] == [2]
    assert run["status"] == "needs_execution"
    assert "--indexes '2'" in run["retry_command"]


def test_completed_selected_batch_does_not_claim_whole_bundle_complete(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    root = _bundle(tmp_path)
    source = tmp_path / "import.json"
    write_json(source, {"items": [{"index": 1, "type": "slide", "markdown": _candidate(1, recovered=True)["visual_text"]}]})
    result = vs.run_visual_structure_plan(root, input_json=source, indexes=[1])
    handoff = json.loads(Path(result["handoff_json_path"]).read_text(encoding="utf-8"))
    assert handoff["status"] == "structured"
    assert handoff["completion_scope"] == "selected_candidates_only"
    assert result["summary"]["unselected_candidate_count"] == 1
    assert result["summary"]["bundle_completeness"] == "not_assessed"


def test_attempt_with_no_results_cannot_report_completed(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    root = _bundle(tmp_path)
    monkeypatch.setattr(vs, "_run_ebook_pipeline_candidates", lambda *_args, **_kwargs: [])
    result = vs.run_visual_structure_plan(root, execute_ebook_pipeline=True)
    run = json.loads((root / "runs" / "visual-structure-ebook" / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "needs_retry"
    assert result["summary"]["unresolved_indexes"] == [1, 2]


def test_recovered_evidence_is_complete_without_a_new_write(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    root = _bundle(tmp_path)
    write_json(root / "timeline.json", [_candidate(i, recovered=True) for i in (1, 2)])
    result = vs.run_visual_structure_plan(root)
    run = json.loads((root / "runs" / "visual-structure-ebook" / "run.json").read_text(encoding="utf-8"))
    assert result["summary"]["updated"] == 0
    assert run["status"] == "completed"
    assert run["retry_command"] == ""
    handoff = json.loads(Path(result["handoff_json_path"]).read_text(encoding="utf-8"))
    assert handoff["next_steps"][0]["action_code"] == "review_selected_evidence"


def test_rendered_readiness_is_explicit(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    root = _bundle(tmp_path)
    result = vs.run_visual_structure_plan(root)
    markdown = Path(result["handoff_path"]).read_text(encoding="utf-8")
    assert "Runtime status" in markdown
    assert "Ready to execute" in markdown
    assert "not_assessed" in markdown
    assert "selected_candidates_only" in markdown


@pytest.mark.parametrize("file_content", [None, "Different stored artifact content"])
def test_service_response_without_matching_local_artifact_is_still_rejected(tmp_path: Path, file_content) -> None:
    from PIL import Image

    root = _bundle(tmp_path, count=1)
    frame = root / "frame.png"
    Image.new("RGB", (32, 32), "white").save(frame)
    write_json(root / "timeline.json", [{**_candidate(1), "frame_paths": [str(frame)]}])
    identity = vs._ebook_input_identity(root, index=1, image_path=str(frame))
    artifact = root / "visual-structure" / "timeline-0001" / "ebook_pipeline" / "output.md"
    if file_content is not None:
        artifact.parent.mkdir(parents=True)
        artifact.write_text(file_content, encoding="utf-8")
    before = (root / "timeline.json").read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        vs._ebook_result_binding(root, {"index": 1, "image_path": str(frame),
            "artifact": {"path": str(artifact), "text": "Claimed extracted courseware content."}}, identity)
    assert (root / "timeline.json").read_bytes() == before
