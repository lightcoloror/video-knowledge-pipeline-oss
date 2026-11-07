from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.markdown_text import markdown_table_cell


def test_markdown_table_cell_preserves_existing_contract() -> None:
    assert markdown_table_cell("  alpha|beta\ngamma  ") == "alpha\\|beta gamma"
    assert markdown_table_cell("") == ""


def test_markdown_table_cell_has_one_direct_owner() -> None:
    source_root = Path(__file__).parents[1] / "src" / "video_knowledge_pipeline"
    implementation = 'return value.replace("|", "\\\\|").replace("\\n", " ").strip()'
    owners = {
        path.name
        for path in source_root.glob("*.py")
        if implementation in path.read_text(encoding="utf-8")
    }

    assert owners == {"markdown_text.py"}
