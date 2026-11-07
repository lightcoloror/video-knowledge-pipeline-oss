from video_knowledge_pipeline.smart_summary_codex import (
    _local_chunk_summary_lines,
    _quality_number_evidence,
    _section_text,
    numbered_summary_heading,
)
from video_knowledge_pipeline.smart_summary_global_reduce import _normalise_markdown


def test_default_summary_headings_use_arabic_hierarchy():
    assert numbered_summary_heading("基本信息") == "## 1 基本信息"
    assert numbered_summary_heading("分段总结") == "## 4 分段总结"
    assert numbered_summary_heading("自定义条目", level=3, number="4.1.1") == "### 4.1.1 自定义条目"


def test_numbered_headings_are_read_by_quality_sections():
    text = """# 标题

## 1 基本信息

基本信息内容。

## 4 分段总结

### 4.1.1 开场

开场内容。

## 5 关键观点

- 5.1.1 关键内容。（00:00:01）
"""
    assert "基本信息内容" in _section_text(text, "## 基本信息")
    assert "4.1.1 开场" in _section_text(text, "## 分段总结")
    assert "5.1.1" in _section_text(text, "## 关键观点")


def test_structural_heading_numbers_are_not_factual_numbers():
    evidence = _quality_number_evidence(
        "## 1 基本信息\n### 4.1.1 开场\n"
        "3.1.1 核心主题\n- 5.1.1 关键观点\n"
        "- 99% 通过。（00:00:01）"
    )
    assert "percentage:99:%" in evidence
    assert all(
        marker not in key
        for key in evidence
        for marker in ("number:1:", "number:3.1:", "number:4:", "number:5.1:")
    )


def test_local_scaffold_uses_three_level_child_numbering():
    lines = _local_chunk_summary_lines([{"start": 0, "end": 5, "title": "开场"}])
    assert lines[0].startswith("### 4.1.1 ")
    assert "第 1 段" not in lines[0]


def test_global_reduce_preserves_numbered_required_headings():
    normalized = _normalise_markdown(
        "# 1 基本信息\n\n内容\n\n# 4 分段总结\n\n### 4.1.1 开场\n\n段落",
        title="测试",
    )
    assert "## 1 基本信息" in normalized
    assert "## 4 分段总结" in normalized
    assert "### 4.1.1 开场" in normalized
