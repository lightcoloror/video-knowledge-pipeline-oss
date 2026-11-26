from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

from video_knowledge_pipeline.subtitle_regression_guard import (
    load_canonical_corrections,
    apply_canonical_patches,
    check_text_regression,
    assert_no_text_regression,
)


def test_load_canonical_corrections_from_dict():
    raw = {"合成错词甲": "合成正词甲", "归类错词": "规范正词"}
    loaded = load_canonical_corrections(raw)
    assert loaded == raw


def test_load_canonical_corrections_from_diff_list():
    with tempfile.TemporaryDirectory() as td:
        diffs_file = Path(td) / "diffs.json"
        diffs_data = [
            {"tag": "replace", "draft": "合成错词甲", "input": "合成正词甲"},
            {"tag": "delete", "draft": "冗余", "input": ""},
            {"tag": "replace", "draft": "归类错词", "input": "规范正词"},
        ]
        diffs_file.write_text(json.dumps(diffs_data, ensure_ascii=False), encoding="utf-8")

        loaded = load_canonical_corrections(diffs_file)
        assert loaded == {"合成错词甲": "合成正词甲", "归类错词": "规范正词"}


def test_apply_canonical_patches_string():
    raw_text = "这是合成测试句子，包含合成错词甲，以及归类错词示例。"
    corrections = {"合成错词甲": "合成正词甲", "归类错词": "规范正词"}
    patched, count = apply_canonical_patches(raw_text, corrections)
    assert patched == "这是合成测试句子，包含合成正词甲，以及规范正词示例。"
    assert count == 2


def test_apply_canonical_patches_cues():
    cues = [
        {"start": 10.0, "end": 12.0, "text": "合成错词甲"},
        {"start": 12.0, "end": 14.0, "text": "正常通过"},
    ]
    corrections = {"合成错词甲": "合成正词甲"}
    patched_cues, count = apply_canonical_patches(cues, corrections)
    assert patched_cues[0]["text"] == "合成正词甲"
    assert patched_cues[1]["text"] == "正常通过"
    assert count == 1


def test_regression_guard_catches_typos():
    srt_content = (
        "1\n00:10:13,792 --> 00:10:15,006\n合成错词甲\n\n"
        "2\n00:10:16,000 --> 00:10:18,000\n正常字幕\n"
    )
    corrections = {"合成错词甲": "合成正词甲"}
    violations = check_text_regression(srt_content, corrections)
    assert len(violations) == 1
    assert violations[0]["typo"] == "合成错词甲"

    with pytest.raises(AssertionError) as exc_info:
        assert_no_text_regression(srt_content, corrections)
    assert "CRITICAL GATE FAILED" in str(exc_info.value)
    assert "合成错词甲" in str(exc_info.value)


def test_regression_guard_passes_when_clean():
    clean_srt = (
        "1\n00:10:13,792 --> 00:10:15,006\n合成正词甲\n\n"
        "2\n00:10:16,000 --> 00:10:18,000\n规范正词阶段\n"
    )
    corrections = {"合成错词甲": "合成正词甲", "归类错词": "规范正词"}
    violations = check_text_regression(clean_srt, corrections)
    assert violations == []
    assert_no_text_regression(clean_srt, corrections)
