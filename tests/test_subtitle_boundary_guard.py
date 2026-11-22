from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from video_knowledge_pipeline.subtitle_boundary_guard import (
    SubtitleBoundaryGuard,
    check_srt_file,
    export_srt,
    fix_srt_file,
    format_srt_time,
    parse_srt,
)


class SubtitleBoundaryGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = SubtitleBoundaryGuard()

    def test_measure_text_and_fallback(self):
        w_short, h_short = self.guard.measure_text("短文本")
        w_long, h_long = self.guard.measure_text("这是一条非常非常长的中文字幕文本超过了安全界限")
        self.assertGreater(w_long, w_short)
        self.assertGreater(h_short, 0)

    def test_audit_compliant_cue(self):
        cue = {
            "index": 1,
            "start": 0.0,
            "end": 2.0,
            "text": "安全字幕",
        }
        res = self.guard.audit_cue(cue)
        self.assertFalse(res["is_overflow"])
        self.assertEqual(res["overflow_amount_px"], 0)
        self.assertGreaterEqual(res["lines"][0]["left_margin_px"], 80)
        self.assertGreaterEqual(res["lines"][0]["right_margin_px"], 80)

    def test_audit_overflowing_cue(self):
        cue = {
            "index": 2,
            "start": 2.0,
            "end": 5.0,
            "text": "这是一条包含多个长分句的示例测试字幕，用于验证超宽文本触发边界溢出告警",
        }
        res = self.guard.audit_cue(cue)
        self.assertTrue(res["is_overflow"])
        self.assertGreater(res["overflow_amount_px"], 0)

    def test_repair_cue_punctuation_splitting(self):
        cue = {
            "index": 3,
            "start": 10.0,
            "end": 14.0,
            "text": "这是一段需要按中文标点折行的测试字幕，并且后续分句也应落在安全宽度内",
        }
        repaired = self.guard.repair_cue(cue)
        self.assertGreater(len(repaired), 0)
        for r in repaired:
            audit = self.guard.audit_cue(r)
            self.assertFalse(audit["is_overflow"], f"Line failed audit: {r['text']}")

    def test_repair_cue_proportional_timing(self):
        long_text = "这是一段非常非常长的口播实录内容，没有任何标点符号但是必须被强制切分以防止字幕超出整机画幅九百二十像素的绝对安全线"
        cue = {
            "index": 4,
            "start": 1.0,
            "end": 5.0,
            "text": long_text,
        }
        repaired = self.guard.repair_cue(cue)
        self.assertGreater(len(repaired), 1)
        self.assertAlmostEqual(repaired[0]["start"], 1.0, places=2)
        self.assertAlmostEqual(repaired[-1]["end"], 5.0, places=2)
        for i in range(len(repaired) - 1):
            self.assertAlmostEqual(repaired[i]["end"], repaired[i + 1]["start"], places=2)
            audit = self.guard.audit_cue(repaired[i])
            self.assertFalse(audit["is_overflow"])

    def test_srt_roundtrip_check_and_fix(self):
        sample_srt = """1
00:00:01,000 --> 00:00:03,000
用于测试边界检查的示例句

2
00:00:03,000 --> 00:00:08,000
我们在合成字幕测试中构造了一段很长的口播示例，包含多个连续分句与超宽字符用于验证自动折行修复

3
00:00:08,000 --> 00:00:10,000
正常合规短句
"""
        with tempfile.TemporaryDirectory() as td:
            in_file = Path(td) / "test_input.srt"
            out_file = Path(td) / "test_repaired.srt"
            in_file.write_text(sample_srt, encoding="utf-8")

            # Check original: should fail
            initial_audit = check_srt_file(in_file)
            self.assertFalse(initial_audit["passed"])
            self.assertGreater(initial_audit["violating_cues"], 0)

            # Fix
            fix_report = fix_srt_file(in_file, out_file)
            self.assertTrue(fix_report["passed"])
            self.assertEqual(fix_report["violating_cues"], 0)
            self.assertEqual(fix_report["compliance_rate"], "100.0%")

            # Re-read repaired file and check
            recheck = check_srt_file(out_file)
            self.assertTrue(recheck["passed"])
            self.assertEqual(recheck["violating_cues"], 0)


    def test_fallback_measurement_without_ttf_font(self):
        guard = SubtitleBoundaryGuard(font_path="__nonexistent_font__.ttf")
        w_short, h_short = guard.measure_text("短文本")
        w_long, h_long = guard.measure_text("这是一条用于验证无TTF字体回退测量的超长中文字幕文本")
        self.assertFalse(guard._using_truetype)
        self.assertGreater(w_long, w_short)
        self.assertEqual(h_short, guard.base_font_size)
        repaired = guard.repair_cue({
            "index": 5,
            "start": 0.0,
            "end": 4.0,
            "text": "这是一条用于验证无TTF字体回退测量的超长中文字幕文本",
        })
        self.assertGreater(len(repaired), 0)
        for item in repaired:
            self.assertFalse(guard.audit_cue(item)["is_overflow"])

if __name__ == "__main__":
    unittest.main()
