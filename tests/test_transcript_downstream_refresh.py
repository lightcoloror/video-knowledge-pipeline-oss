from __future__ import annotations

from pathlib import Path

import video_knowledge_pipeline.knowledge_coverage as knowledge_coverage
import video_knowledge_pipeline.knowledge_note_export as knowledge_note_export
import video_knowledge_pipeline.quality_console as quality_console
import video_knowledge_pipeline.quality_finalize as quality_finalize
import video_knowledge_pipeline.smart_summary_codex as smart_summary_codex
import video_knowledge_pipeline.task_console as task_console
import video_knowledge_pipeline.transcript_quality_gate as transcript_quality_gate
import video_knowledge_pipeline.video_workbench as video_workbench
import video_knowledge_pipeline.webui_bridge as webui_bridge
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.transcript_downstream_refresh import (
    invalidate_smart_summary_for_transcript_change,
    refresh_transcript_downstream_outputs,
)


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    write_json(root / "manifest.json", {"title": "refresh fixture"})
    write_json(
        root / "source-arbitrated-transcript.json",
        {"segments": [{"start": 0, "end": 5, "text": "修复后的正文。"}]},
    )
    (exports / "smart-summary.md").write_text(
        "# 旧摘要\n\n生成方式：`codex_final`\n",
        encoding="utf-8",
    )
    return root


def _patch_stages(
    monkeypatch,
    calls: list[str],
    *,
    summary_passed: bool,
    fail_stage: str = "",
) -> None:
    def fake(name: str, payload: dict):
        def run(*args, **kwargs):
            del args, kwargs
            calls.append(name)
            if name == fail_stage:
                raise RuntimeError(f"{name} failed")
            return dict(payload)

        return run

    monkeypatch.setattr(
        knowledge_note_export,
        "export_knowledge_note",
        fake("export_knowledge_note", {"status": "exported", "ok": True}),
    )
    monkeypatch.setattr(
        quality_finalize,
        "finalize_quality_outputs",
        fake(
            "prepare_quality_outputs",
            {"status": "planned_section_llm", "ok": False},
        ),
    )
    monkeypatch.setattr(
        knowledge_coverage,
        "audit_knowledge_coverage",
        fake("audit_knowledge_coverage", {"status": "completed", "ok": True}),
    )
    monkeypatch.setattr(
        webui_bridge,
        "refresh_bundle_review_html",
        fake("refresh_review_html", {"status": "refreshed", "ok": True}),
    )
    monkeypatch.setattr(
        transcript_quality_gate,
        "run_transcript_quality_gate",
        fake(
            "final_transcript_quality_gate",
            {"status": "passed", "ok": True, "fail_count": 0},
        ),
    )
    monkeypatch.setattr(
        smart_summary_codex,
        "smart_summary_quality_check",
        fake(
            "final_smart_summary_quality_gate",
            {
                "status": "passed" if summary_passed else "failed",
                "passed": summary_passed,
            },
        ),
    )
    monkeypatch.setattr(
        quality_console,
        "export_quality_console",
        fake("export_quality_console", {"status": "ready", "ok": True}),
    )
    monkeypatch.setattr(
        task_console,
        "export_task_console",
        fake("export_task_console", {"status": "ready", "ok": True}),
    )
    monkeypatch.setattr(
        video_workbench,
        "export_video_workbench",
        fake("export_video_workbench", {"status": "ready", "ok": True}),
    )


def test_invalidation_archives_old_summary_and_fails_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    summary = root / "exports" / "smart-summary.md"
    old_hash = sha256_file(summary)

    result = invalidate_smart_summary_for_transcript_change(
        root,
        canonical_before_sha256="a" * 64,
        canonical_after_sha256="b" * 64,
        write=True,
    )
    gate = smart_summary_codex._smart_summary_invalidation_gate(root, summary)

    assert result["invalidated_summary_count"] == 1
    archive = Path(result["invalidated_summaries"][0]["archive_path"])
    assert archive.is_file()
    assert sha256_file(archive) == old_hash
    assert gate["passed"] is False
    assert gate["status"] == "summary_invalidated_after_transcript_update"
    manifest = read_json(root / "manifest.json")
    assert manifest["smart_summary_requires_regeneration"] is True

    summary.write_text("# 新摘要\n", encoding="utf-8")
    regenerated = smart_summary_codex._smart_summary_invalidation_gate(
        root,
        summary,
    )
    assert regenerated["passed"] is True
    assert regenerated["status"] == "regenerated_after_invalidation"


def test_refresh_rebuilds_in_order_and_requires_fresh_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _root(tmp_path)
    calls: list[str] = []
    _patch_stages(monkeypatch, calls, summary_passed=False)

    result = refresh_transcript_downstream_outputs(
        root,
        canonical_before_sha256="a" * 64,
        canonical_after_sha256="b" * 64,
        write=True,
    )

    assert calls == [
        "export_knowledge_note",
        "prepare_quality_outputs",
        "audit_knowledge_coverage",
        "refresh_review_html",
        "final_transcript_quality_gate",
        "final_smart_summary_quality_gate",
        "export_quality_console",
        "export_task_console",
        "export_video_workbench",
    ]
    assert result["status"] == "needs_summary_regeneration"
    assert result["local_refresh_completed"] is True
    assert result["transcript_quality_passed"] is True
    assert result["smart_summary_quality_passed"] is False
    assert result["full_pipeline_production_qualified"] is False
    assert (root / "transcript-downstream-refresh.json").is_file()


def test_refresh_completes_only_after_both_final_gates_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _root(tmp_path)
    calls: list[str] = []
    _patch_stages(monkeypatch, calls, summary_passed=True)

    result = refresh_transcript_downstream_outputs(
        root,
        canonical_before_sha256="a" * 64,
        canonical_after_sha256="b" * 64,
        write=False,
    )

    assert result["status"] == "completed"
    assert result["full_pipeline_production_qualified"] is True
    assert not (root / "exports" / "smart-summary-invalidation.json").exists()
    assert not (root / "transcript-downstream-refresh.json").exists()


def test_refresh_is_degraded_but_preserves_other_stage_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _root(tmp_path)
    calls: list[str] = []
    _patch_stages(
        monkeypatch,
        calls,
        summary_passed=True,
        fail_stage="audit_knowledge_coverage",
    )

    result = refresh_transcript_downstream_outputs(
        root,
        canonical_before_sha256="a" * 64,
        canonical_after_sha256="b" * 64,
        write=True,
    )

    assert result["status"] == "degraded"
    assert result["failed_steps"] == ["audit_knowledge_coverage"]
    assert "export_video_workbench" in calls
    assert result["full_pipeline_production_qualified"] is False
