from __future__ import annotations

import unittest

from video_knowledge_pipeline.video_quality_guard import (
    SubtitlePacingLinter,
    VisualSafeZoneLinter,
    check_av_sync,
    check_disclosure_coverage,
)


class VideoQualityGuardTests(unittest.TestCase):
    def setUp(self):
        self.zone_linter = VisualSafeZoneLinter()
        self.pacing_linter = SubtitlePacingLinter()

    def test_safe_zone_critical_content_strictly_blocked(self):
        # A critical card placed on the right rail (x=900..1060, y=800..1200)
        # Heavy overlap with right interaction rail
        res = self.zone_linter.audit_element(
            element_id="card-evidence-table",
            bbox=(900, 800, 1060, 1200),
            is_critical=True,
            description="合成核心数据卡片",
        )
        self.assertFalse(res["passed"])
        self.assertTrue(res["blocking"])
        self.assertEqual(res["severity"], "BLOCKING_ERROR")

    def test_safe_zone_ordinary_content_generously_tolerated(self):
        # An ordinary subtitle or secondary badge partially entering the right boundary
        res = self.zone_linter.audit_element(
            element_id="badge-secondary-note",
            bbox=(880, 800, 1020, 900),
            is_critical=False,
            description="次要普通装饰或角标",
        )
        # Should NOT block, should pass pragmatically
        self.assertTrue(res["passed"])
        self.assertFalse(res["blocking"])

    def test_safe_zone_center_critical_content_passes(self):
        # A critical evidence card properly centered
        res = self.zone_linter.audit_element(
            element_id="card-centered-evidence",
            bbox=(100, 400, 800, 1200),
            is_critical=True,
            description="居中证物卡片",
        )
        self.assertTrue(res["passed"])
        self.assertFalse(res["blocking"])
        self.assertEqual(res["severity"], "PASS")

    def test_subtitle_pacing_audit(self):
        valid_cues = [
            {"index": 1, "start": 0.0, "end": 2.0, "text": "合成测试字幕第一行"},
            {"index": 2, "start": 2.0, "end": 4.5, "text": "合成测试字幕第二行演示"},
        ]
        res_valid = self.pacing_linter.audit_cues(valid_cues)
        self.assertTrue(res_valid["passed"])
        self.assertEqual(res_valid["pacing_violations_count"], 0)

        # Flashing cue (0.2s duration for 14 characters = 70 CPS)
        invalid_cues = [
            {"index": 3, "start": 5.0, "end": 5.2, "text": "这句话闪现过快根本无法看清"},
        ]
        res_invalid = self.pacing_linter.audit_cues(invalid_cues)
        self.assertFalse(res_invalid["passed"])
        self.assertEqual(res_invalid["pacing_violations_count"], 1)

    def test_av_sync_check(self):
        # 15ms delta at 30fps is ~0.45 frames (< 1 frame), should pass
        res_pass = check_av_sync(video_duration_s=120.000, audio_duration_s=120.015, fps=30.0)
        self.assertTrue(res_pass["passed"])
        self.assertLess(res_pass["drift_frames"], 1.0)

        # 300ms drift is 9 frames, should fail
        res_fail = check_av_sync(video_duration_s=120.000, audio_duration_s=120.300, fps=30.0)
        self.assertFalse(res_fail["passed"])

    def test_disclosure_coverage(self):
        ai_scenes = [
            (4.04, 11.33, "synthetic-ai-scene-01.png"),
            (127.20, 145.20, "synthetic-ai-scene-02.png"),
        ]
        ai_badges_full = [
            (4.04, 11.33),
            (127.20, 145.20),
        ]
        res_full = check_disclosure_coverage(ai_scenes, ai_badges_full)
        self.assertTrue(res_full["passed"])

        # Missing badge for second scene
        ai_badges_missing = [(4.04, 11.33)]
        res_missing = check_disclosure_coverage(ai_scenes, ai_badges_missing)
        self.assertFalse(res_missing["passed"])
        self.assertEqual(len(res_missing["uncovered_scenes"]), 1)


if __name__ == "__main__":
    unittest.main()
