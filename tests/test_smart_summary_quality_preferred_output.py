from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.smart_summary_codex import smart_summary_quality_check


def test_quality_check_prefers_installed_codex_summary_by_default(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"normalized_transcript_json": "normalized-transcript.json"}),
        encoding="utf-8",
    )
    (root / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 10, "text": "课程内容"}]}),
        encoding="utf-8",
    )
    (root / "timeline.json").write_text("[]", encoding="utf-8")
    (exports / "smart-summary.md").write_text("legacy summary", encoding="utf-8")
    (exports / "smart-summary.codex.md").write_text(
        "生成方式：`codex_llm_rewrite_final`\n\n## 一句话概览\n课程内容",
        encoding="utf-8",
    )

    result = smart_summary_quality_check(root, write=False)

    assert result["summary_path"] == str(exports / "smart-summary.codex.md")
