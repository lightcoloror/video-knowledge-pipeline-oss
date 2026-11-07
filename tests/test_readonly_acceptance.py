"""No-write acceptance must inspect evidence without filesystem mutations."""
from __future__ import annotations

import builtins
import io
import json
import os
from pathlib import Path

import pytest

from video_knowledge_pipeline.acceptance_check import acceptance_check
from video_knowledge_pipeline.bundle_next import bundle_next_action
from video_knowledge_pipeline.bundle_status import bundle_status_report, controlled_execution_check
from video_knowledge_pipeline.cli import main as cli_main
from video_knowledge_pipeline.storage import read_json, write_json


def _bundle(path: Path, *, route: str = "speech") -> Path:
    path.mkdir()
    manifest = {
        "schema": "lecture_webui_bundle.v1", "title": "Synthetic read-only fixture",
        "source_package": "", "review_readiness": {"ready": True},
        "knowledge_coverage": {"status": "blocked", "next_action": {
            "key": "speech" if route == "speech" else "semantic_visual_understanding",
            "mcp_tool": "timeline_alignment_audit" if route == "speech" else "run_multimodal_frame_analysis",
        }},
    }
    if route == "panorama":
        manifest["panorama_profile"] = {
            "schema": "video_knowledge_pipeline.panorama_profile.v1", "ready": True,
            "projection": "equirectangular",
        }
    if route == "correction":
        manifest["knowledge_coverage"] = {"status": "passed"}
    write_json(path / "manifest.json", manifest)
    write_json(path / "timeline.json", [{"index": 1, "start": 0, "end": 3,
        "transcript": "合成的待核内容。", "visual_route": "semantic_frame",
        "quality_issues": ["missing_visual_text"], "frame_paths": []}])
    write_json(path / "source-arbitrated-transcript.json", {"segments": [
        {"start": 0, "end": 3, "text": "合成的待核内容。"}]})
    write_json(path / "mcp-bundle-next-action.args.json", {"sentinel": "keep existing bytes"})
    if route == "provider":
        write_json(path / "vision-execution-preflight.json", {
            "provider_health": {"safe_to_execute": False},
            "interactive_session": {"ready_to_prepare": True},
            "selected_indexes": {"semantic": [1]},
        })
    return path


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def deny_bundle_mutations(monkeypatch):
    """Deny attempts, not just observe final content after a hidden rewrite."""
    def guard(root: Path):
        denied = []

        def inside(path):
            return isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve().is_relative_to(root.resolve())

        def wrap_open(original):
            def checked(file, mode="r", *args, **kwargs):
                if inside(file) and any(c in str(mode) for c in "wax+"):
                    denied.append(("open", str(file)))
                    raise PermissionError("No-write test: output attempted inside source Bundle")
                return original(file, mode, *args, **kwargs)
            return checked

        monkeypatch.setattr(builtins, "open", wrap_open(builtins.open))
        monkeypatch.setattr(io, "open", wrap_open(io.open))
        for name in ("mkdir", "remove", "unlink", "rmdir", "rename", "replace"):
            original = getattr(os, name)

            def checked(*args, _name=name, _original=original, **kwargs):
                targets = args[:2] if _name in {"rename", "replace"} else args[:1]
                if any(inside(p) for p in targets):
                    denied.append((_name, str(targets)))
                    raise PermissionError("No-write test: filesystem mutation attempted")
                return _original(*args, **kwargs)

            monkeypatch.setattr(os, name, checked)
        return denied
    return guard


@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("entry", ["acceptance", "status", "controlled", "cli"])
def test_no_write_acceptance_chain_never_attempts_bundle_mutations(
    tmp_path, deny_bundle_mutations, refresh, entry, capsys,
):
    bundle = _bundle(tmp_path / "bundle")
    before = _snapshot(bundle)
    denied = deny_bundle_mutations(bundle)
    if entry == "acceptance":
        result = acceptance_check(bundle, refresh=refresh, write=False)
        assert result["status"] == "incomplete"
        assert result["blockers"]
    elif entry == "status":
        result = bundle_status_report(bundle, refresh=refresh, write=False)
        assert result["completion_status"] != "completed"
    elif entry == "controlled":
        result = controlled_execution_check(bundle, refresh=refresh, write=False)
        assert result["ready_for_real_vision_execution"] is False
    else:
        argv = ["acceptance-check", str(bundle), "--no-write", "--verbose"]
        if not refresh:
            argv.append("--no-refresh")
        assert cli_main(argv) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "incomplete" and result["blockers"]
    assert denied == []
    assert _snapshot(bundle) == before


@pytest.mark.parametrize("route,expected", [
    ("speech", "coverage_blocked"),
    ("semantic", "coverage_blocked"),
    ("provider", "awaiting-agent"),
    ("panorama", "panorama_pending"),
    ("correction", "semantic_correction_pending"),
])
def test_readonly_next_action_routes_keep_recommendations_without_args_writes(
    tmp_path, deny_bundle_mutations, route, expected,
):
    bundle = _bundle(tmp_path / "bundle", route=route)
    before = _snapshot(bundle)
    denied = deny_bundle_mutations(bundle)
    result = bundle_next_action(bundle, refresh=False, write=False)
    assert result["status"] == expected
    assert result["mcp_args_written"] is False
    assert result["next_action"]["mcp_tool"]
    if route == "provider":
        # The existing summary handoff keeps priority over human-required
        # provider recovery; do not change that policy for read-only mode.
        assert result["next_action"]["upstream_next_action"]["status"] == "interactive_session_ready"
    assert denied == [] and _snapshot(bundle) == before


def test_readonly_refresh_uses_current_evidence_instead_of_stale_ready_manifest(
    tmp_path, deny_bundle_mutations,
):
    bundle = _bundle(tmp_path / "bundle")
    manifest = read_json(bundle / "manifest.json")
    manifest["knowledge_coverage"] = {"status": "passed"}
    write_json(bundle / "manifest.json", manifest)
    before = _snapshot(bundle)
    denied = deny_bundle_mutations(bundle)
    result = bundle_next_action(bundle, refresh=True, write=False)
    assert result["refreshed"] is True
    assert result["next_action"]["kind"] == "knowledge_coverage"
    assert result["next_action"]["coverage_status"] == "blocked"
    assert denied == [] and _snapshot(bundle) == before


def test_default_write_mode_still_materializes_arguments_and_acceptance_outputs(tmp_path):
    bundle = _bundle(tmp_path / "bundle")
    next_result = bundle_next_action(bundle, refresh=False)
    assert next_result["mcp_args_written"] is True
    assert read_json(Path(next_result["mcp_args_path"])) == {"bundle_dir": str(bundle.resolve()), "refresh": True}
    result = acceptance_check(bundle)
    assert result["status"] == "incomplete"
    assert read_json(bundle / "acceptance-check.json")["status"] == "incomplete"
    assert (bundle / "acceptance-check.md").is_file()
    assert (bundle / "mcp-acceptance-check.args.json").is_file()


def test_readonly_status_refresh_reports_live_coverage_in_compact_summary(tmp_path, deny_bundle_mutations):
    bundle = _bundle(tmp_path / "bundle")
    manifest = read_json(bundle / "manifest.json")
    manifest["knowledge_coverage"] = {"status": "passed"}
    write_json(bundle / "manifest.json", manifest)
    before = _snapshot(bundle)
    denied = deny_bundle_mutations(bundle)
    result = bundle_status_report(bundle, refresh=True, write=False)
    assert result["next_action"]["coverage_status"] == "blocked"
    assert result["knowledge_coverage"]["status"] == "blocked"
    assert result["knowledge_coverage"]["blocker_count"] > 0
    assert denied == [] and _snapshot(bundle) == before
