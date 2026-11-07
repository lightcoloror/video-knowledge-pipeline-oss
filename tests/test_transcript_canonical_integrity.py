from __future__ import annotations

import hashlib
import json
from pathlib import Path

from video_knowledge_pipeline.knowledge_note_export import canonical_export_integrity_status


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_export_integrity_blocks_stale_transcript_exports(tmp_path: Path) -> None:
    canonical = tmp_path / "source-arbitrated-transcript.json"
    canonical.write_text('{"segments":[{"text":"正确逐字稿"}]}', encoding="utf-8")
    canonical_hash = _sha256(canonical)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"source_arbitrated_transcript_json": canonical.name}),
        encoding="utf-8",
    )
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "full-transcript.md").write_text(
        f"# 逐字稿\n\n- Canonical source SHA-256: `{canonical_hash}`\n",
        encoding="utf-8",
    )
    (exports / "full-body.md").write_text(
        f"# 正文\n\n- Canonical source SHA-256: `{canonical_hash}`\n",
        encoding="utf-8",
    )
    (exports / "knowledge-note.md").write_text(
        f"# 知识笔记\n\n> Canonical transcript SHA-256: `{canonical_hash}`\n",
        encoding="utf-8",
    )
    (exports / "smart-summary-input-pack.json").write_text(
        json.dumps(
            {
                "transcript_source": canonical.name,
                "transcript_source_sha256": canonical_hash,
            }
        ),
        encoding="utf-8",
    )

    initial = canonical_export_integrity_status(tmp_path)
    assert initial["status"] == "passed"
    assert initial["passed"] is True

    canonical.write_text('{"segments":[{"text":"人工修订后的逐字稿"}]}', encoding="utf-8")
    stale = canonical_export_integrity_status(tmp_path)
    assert stale["status"] == "blocked_canonical_export_mismatch"
    assert stale["passed"] is False
    assert {issue["key"] for issue in stale["issues"]} == {
        "full_transcript_canonical_hash_mismatch",
        "full_body_canonical_hash_mismatch",
        "knowledge_note_canonical_hash_mismatch",
        "smart_summary_source_hash_mismatch",
    }
