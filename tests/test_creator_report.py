from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from test_shot_breakdown import _bundle, _write_json
from video_knowledge_pipeline import creator_report as adapter
from video_knowledge_pipeline.cli import build_parser, main as cli_main
from video_knowledge_pipeline.creator_report_renderers import bundle_file, html_document
from video_knowledge_pipeline.creator_scriptbreak import export_prompt_pack, project_candidate
from video_knowledge_pipeline.video_workbench import export_video_workbench


BRIEF = {"product": "轻便水杯", "audience": "通勤者", "goal": "演示开盖", "resources": "手机、桌面、自然光"}


def bundle(tmp_path: Path) -> Path:
    root = _bundle(tmp_path)
    for image in (root / "frames").iterdir():
        Image.new("RGB", (160, 90), "green").save(image)
    return root


def test_candidate_projection_preserves_unknowns_and_is_read_only(tmp_path):
    root = bundle(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = adapter.export_creator_report(root, write=False)
    report = result["report"]
    assert report["candidate_only"] and not report["publication_allowed"]
    assert report["summary"] == "" and report["summary_status"] == "blocked"
    assert report["transcript_scope"] == "timeline_fragments"
    assert report["shots"][0]["facts"]["shot_type"] == "unknown"
    assert len(report["previews"]) == 2
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_real_md_html_exports_and_workbench_link(tmp_path):
    root = bundle(tmp_path)
    original = (root / "manifest.json").read_bytes(), (root / "timeline.json").read_bytes()
    result = adapter.export_creator_report(root, title="通勤 / 水杯", brief=BRIEF)
    assert result["status"] == "exported_candidate"
    assert {a["format"] for a in result["artifacts"]} == {"md", "html", "scriptbreak", "projection"}
    report_html = root / next(a["path"] for a in result["artifacts"] if a["format"] == "html")
    content = report_html.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in content and "00:00:05.000" in content
    assert str(root) not in content and "timeline-001" in content
    assert adapter.creator_report_status(root)["status"] == "fresh"
    assert original == ((root / "manifest.json").read_bytes(), (root / "timeline.json").read_bytes())
    workbench = export_video_workbench(root, write=False)
    card = next(a for a in workbench["artifacts"] if a["key"] == "creator_report_index")
    assert "creator-report/index.html" in card["href"]
    assert card["candidate_only"] and card["status"] == "fresh"


def test_report_becomes_stale_when_new_or_changed_evidence_arrives(tmp_path):
    root = bundle(tmp_path)
    adapter.export_creator_report(root)
    timeline = json.loads((root / "timeline.json").read_text(encoding="utf-8"))
    timeline[0]["transcript"] = "新增一条事实"
    _write_json(root / "timeline.json", timeline)
    assert "inputs_changed" in adapter.creator_report_status(root)["issues"]


def test_changed_export_is_detected_and_old_revisions_retained(tmp_path):
    root = bundle(tmp_path)
    first = adapter.export_creator_report(root)
    target = root / first["artifacts"][0]["path"]
    target.write_text("tampered", encoding="utf-8")
    assert "artifact_changed:md" in adapter.creator_report_status(root)["issues"]
    second = adapter.export_creator_report(root)
    assert first["artifacts"][0]["path"] != second["artifacts"][0]["path"]
    assert target.read_text(encoding="utf-8") == "tampered"


def test_renderer_failure_is_partial_and_retains_other_formats(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    real = adapter.export_format
    def render(markdown, target, **kwargs):
        if target.suffix == ".pdf":
            raise ImportError("optional PDF dependency unavailable")
        real(markdown, target, **kwargs)
    monkeypatch.setattr(adapter, "export_format", render)
    result = adapter.export_creator_report(root, formats=["html", "pdf"])
    assert result["status"] == "partial_failed"
    assert result["failures"][0]["format"] == "pdf"
    assert any(a["format"] == "html" for a in result["artifacts"])
    assert not result["publication_allowed"]


def test_change_during_export_does_not_advance_latest(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    first = adapter.export_creator_report(root)
    real = adapter.export_format
    def render(markdown, target, **kwargs):
        real(markdown, target, **kwargs)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest["title"] = "Changed title"
        _write_json(root / "manifest.json", manifest)
    monkeypatch.setattr(adapter, "export_format", render)
    result = adapter.export_creator_report(root)
    assert result["status"] == "stale_during_export"
    latest = json.loads((root / adapter.EXPORT_ROOT / "latest.json").read_text(encoding="utf-8"))
    assert latest["input_fingerprint"] == first["input_fingerprint"]


def test_unsafe_frames_and_html_are_not_embedded(tmp_path):
    root = bundle(tmp_path)
    outside = tmp_path / "private.png"
    Image.new("RGB", (20, 20)).save(outside)
    timeline = json.loads((root / "timeline.json").read_text(encoding="utf-8"))
    timeline[0]["frame_paths"] = [str(outside), "https://example.com/private.png"]
    timeline[0]["corrected_transcript"] = '<script>alert(1)</script> ![remote](https://example.com/a.png)'
    _write_json(root / "timeline.json", timeline)
    report = adapter.build_creator_report(root)
    assert "shot-0001" not in report["previews"]
    document = html_document(adapter.render_creator_markdown(report), report["title"])
    assert "<script>" not in document and 'src="https://' not in document
    assert report["preview_issues"]
    with pytest.raises(ValueError):
        bundle_file(root, outside)


def test_unvalidated_summary_does_not_leak_into_report_findings(tmp_path):
    root = bundle(tmp_path)
    (root / "exports" / "smart-summary.md").write_text("# 智能总结\n\n## 核心观点\n不可靠的旧总结秘密", encoding="utf-8")
    report = adapter.build_creator_report(root)
    assert report["summary"] == ""
    assert "不可靠的旧总结秘密" not in adapter.render_creator_markdown(report)


def test_validated_summary_does_not_authorize_a_different_old_summary(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    (root / "exports/smart-summary.md").write_text("# 总结\n\n## 核心主题\n旧文件不应授权", encoding="utf-8")
    accepted = root / "validated-summary.md"
    accepted.write_text("已经通过上游验收的新总结", encoding="utf-8")
    original_completion = adapter.workflow_completion_status(root)
    for stage in original_completion["stages"]:
        if stage["key"] == "summary_validation":
            stage["status"] = "completed"
        if stage["key"] == "smart_summary":
            stage["evidence"]["path"] = str(accepted)
    monkeypatch.setattr(adapter, "workflow_completion_status", lambda _, **kwargs: original_completion)
    report = adapter.build_creator_report(root)
    markdown = adapter.render_creator_markdown(report)
    assert "已经通过上游验收的新总结" in markdown
    assert "旧文件不应授权" not in markdown
    assert any(s["artifact_id"] == "validated-summary-001" for s in report["sources"])


def test_oversized_image_rejection_preserves_text_report(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    def oversized(*args, **kwargs):
        raise Image.DecompressionBombError("synthetic oversized image header")
    monkeypatch.setattr(Image, "open", oversized)
    report = adapter.build_creator_report(root)
    assert not report["previews"] and report["preview_issues"]
    assert "讲师介绍第一步" in adapter.render_creator_markdown(report)


@pytest.mark.parametrize("formats", [[], ["png"], ["md", "exe"]])
def test_invalid_formats_fail_before_writes(tmp_path, formats):
    root = bundle(tmp_path)
    with pytest.raises(ValueError):
        adapter.export_creator_report(root, formats=formats)
    assert not (root / adapter.EXPORT_ROOT).exists()


def test_scriptbreak_requires_brief_and_never_promotes_camera_defaults(tmp_path):
    report = adapter.build_creator_report(bundle(tmp_path))
    with pytest.raises(ValueError):
        project_candidate(report)
    report["brief"] = BRIEF
    project = project_candidate(report)
    assert project["app"] == "scriptbreak" and project["version"] == 2
    assert project["state"]["scenes"][0]["shots"][0]["size"] == ""
    assert "讲师介绍第一步" in project["state"]["scenes"][0]["shots"][0]["desc"]
    assert project["vkp_provenance"]["candidate_only"]


def test_scriptbreak_source_hash_mismatch_never_executes(tmp_path, monkeypatch):
    script = tmp_path / "script.mjs"
    script.write_text("throw Error('should not run')", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_prompt_pack(tmp_path / "candidate.scriptbreak", script, expected_sha256="0" * 64)


def test_cli_entry_uses_existing_bundle_without_execution(tmp_path, capsys):
    root = bundle(tmp_path)
    args = build_parser().parse_args(["export-creator-report", str(root), "--no-write"])
    assert args.formats == ["md", "html"]
    assert cli_main(["export-creator-report", str(root), "--no-write"]) == 0
    assert '"preview"' in capsys.readouterr().out
    assert not (root / adapter.EXPORT_ROOT).exists()


def test_invalid_receipt_fails_closed_without_crashing_workbench(tmp_path):
    root = bundle(tmp_path)
    _write_json(root / adapter.EXPORT_ROOT / "latest.json", {"schema": "wrong", "artifacts": [False]})
    assert adapter.creator_report_status(root)["status"] == "invalid"
    cards = export_video_workbench(root, write=False)["artifacts"]
    assert next(c for c in cards if c["key"] == "creator_report_index")["status"] == "invalid"


def test_cli_format_failure_exits_nonzero_with_diagnostic(tmp_path, monkeypatch, capsys):
    root = bundle(tmp_path)
    def unavailable(*args, **kwargs):
        raise ImportError("optional PDF dependency unavailable")
    monkeypatch.setattr(adapter, "export_format", unavailable)
    assert cli_main(["export-creator-report", str(root), "--formats", "pdf"]) == 1
    assert 'partial_failed' in capsys.readouterr().out


def test_canonical_transcript_is_used_and_source_removal_invalidates(tmp_path):
    root = bundle(tmp_path)
    source = root / "human-corrected-transcript.json"
    _write_json(source, {"segments": [{"start": 0, "end": 4, "text": "规范逐字稿", "speaker": "S1"}]})
    report = adapter.build_creator_report(root)
    assert report["transcript_scope"] == "canonical"
    assert report["transcript"][0]["text"] == "规范逐字稿"
    adapter.export_creator_report(root)
    source.unlink()
    assert adapter.creator_report_status(root)["status"] == "stale"


def test_mcp_tools_delegate_without_executable_path_surface(tmp_path, monkeypatch):
    import sys
    import types
    from video_knowledge_pipeline import mcp_server
    tools = {}
    class Server:
        def __init__(self, *args, **kwargs): pass
        def tool(self, *args, **kwargs):
            def add(function):
                tools[function.__name__] = function
                return function
            return add
        def run(self): pass
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", types.SimpleNamespace(FastMCP=Server))
    mcp_server.main()
    root = bundle(tmp_path)
    result = tools["export_creator_report"](str(root), write=False)
    assert result["status"] == "preview"
    assert tools["creator_report_status"](str(root))["status"] == "missing"
    import inspect
    assert "scriptbreak_script" not in inspect.signature(tools["export_creator_report"]).parameters


@pytest.mark.parametrize("field,value", [("path", ["not-a-path"]), ("format", {"bad": True}), ("sha256", None)])
def test_invalid_artifact_field_types_are_a_diagnostic_not_a_crash(tmp_path, field, value):
    root = bundle(tmp_path)
    adapter.export_creator_report(root)
    path = root / adapter.EXPORT_ROOT / "latest.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["artifacts"][0][field] = value
    _write_json(path, receipt)
    assert adapter.creator_report_status(root)["status"] == "invalid"


def test_disappearing_source_during_export_retains_receipt_and_previous_pointer(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    adapter.export_creator_report(root)
    latest = root / adapter.EXPORT_ROOT / "latest.json"
    before = latest.read_bytes()
    real = adapter.export_format
    def render(*args, **kwargs):
        real(*args, **kwargs)
        (root / "manifest.json").write_text("{incomplete", encoding="utf-8")
    monkeypatch.setattr(adapter, "export_format", render)
    result = adapter.export_creator_report(root)
    assert result["status"] == "stale_during_export"
    assert result["input_revalidation_error"]["error_type"] == "JSONDecodeError"
    assert latest.read_bytes() == before
    assert len(list((root / adapter.EXPORT_ROOT).glob("*/receipt.json"))) == 2
