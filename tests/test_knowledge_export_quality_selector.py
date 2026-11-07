from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.knowledge_note_export import _quality_summary_path


def test_export_quality_uses_existing_canonical_summary_not_readable_copy(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    canonical = exports / "smart-summary.codex.md"
    readable_copy = exports / "smart-summary.md"
    canonical.write_text("canonical", encoding="utf-8")
    readable_copy.write_text("readable copy", encoding="utf-8")

    assert _quality_summary_path(canonical, readable_copy) == canonical


def test_export_quality_falls_back_to_readable_copy_without_canonical_summary(tmp_path: Path) -> None:
    readable_copy = tmp_path / "smart-summary.md"

    assert _quality_summary_path(None, readable_copy) == readable_copy

def test_final_reader_export_uses_complete_canonical_summary_for_two_bundle_styles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import json

    import video_knowledge_pipeline.knowledge_note_export as knowledge_note_export
    from video_knowledge_pipeline.knowledge_note_export import export_knowledge_note

    def _unexpected_model_call(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("existing canonical summary must not trigger a model call")

    monkeypatch.setattr(
        knowledge_note_export,
        "generate_smart_summary_with_codex",
        _unexpected_model_call,
    )
    cases = (
        ("courseware-companion", "信任建立需要持续输出可验证的专业内容。"),
        ("national-morning-meeting", "早会内容按业绩、方法和行动计划组织。"),
    )
    for case_name, summary_detail in cases:
        bundle = tmp_path / case_name / "webui-bundle"
        exports = bundle / "exports"
        exports.mkdir(parents=True)
        title = case_name.replace("-", " ")
        transcript_text = f"{title} 的完整逐字稿内容。"
        (bundle / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "lecture_webui_bundle.v1",
                    "title": title,
                    "normalized_transcript_json": "normalized-transcript.json",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (bundle / "timeline.json").write_text(
            json.dumps(
                [{"index": 1, "start": 0, "end": 12, "transcript": transcript_text}],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (bundle / "normalized-transcript.json").write_text(
            json.dumps(
                {"segments": [{"start": 0, "end": 12, "text": transcript_text}]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        canonical_summary = "\n".join(
            [
                f"# {title} - 智能总结",
                "",
                "生成方式：`online_llm_section_rewrite` + `section_staged_apply`。",
                "",
                "## 一句话概览",
                "",
                summary_detail,
                "",
                "## 分段总结",
                "",
                "### 00:00:00 - 00:00:12",
                "",
                f"{summary_detail} 本段证据来自逐字稿。",
                "",
            ]
        )
        canonical_path = exports / "smart-summary.codex.md"
        canonical_path.write_text(canonical_summary, encoding="utf-8")
        final_dir = bundle.parent / "exports-final"

        result = export_knowledge_note(
            bundle,
            output_dir=final_dir,
            run_transcript_evidence_check=False,
        )

        exported_summary = Path(result["smart_summary_path"]).read_text(encoding="utf-8")
        reading_note = Path(result["note_path"]).read_text(encoding="utf-8")
        assert exported_summary == canonical_summary
        assert summary_detail in reading_note
        assert "（智能总结尚未生成" not in reading_note
        assert "needs_llm_summary" not in reading_note
        assert "  - 📑 智能总结" in reading_note
        assert "- 逐字稿" in reading_note
        assert reading_note.index("  - 📑 智能总结") < reading_note.index("- 逐字稿")
        assert not any(line.lstrip().startswith("#") for line in reading_note.splitlines())
        assert transcript_text in reading_note
