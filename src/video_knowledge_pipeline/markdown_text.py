from __future__ import annotations


def markdown_table_cell(value: str) -> str:
    """Render the repository's established single-line Markdown table cell."""
    return value.replace("|", "\\|").replace("\n", " ").strip()
