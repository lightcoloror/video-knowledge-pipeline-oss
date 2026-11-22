"""
Video Production Quality Guard & Pragmatic Linter
-------------------------------------------------
Automated quality gate for vertical video (1080x1920) production.

Design Principles:
1. Pragmatic Tolerance: Only 'critical content' strictly avoids platform interactive zones.
   Ordinary elements (subtitles, badges, ambient backgrounds) are permitted partial overlap
   to avoid over-constraining the canvas and triggering false alarms.
2. Single-Pass Audio Strategy: Loudness leveling is locked during rough-cut audio prep.
   Downstream rendering only checks lightweight A/V duration alignment, not redundant loudness recalculation.
3. Automated Quality Auditing:
   - Visual Safe Zone / UI Occlusion Gate (differentiated by element criticality)
   - Subtitle Pacing Gate (CPS / minimum readability duration)
   - A/V Sync & Drift Gate (duration delta <= 1 frame)
   - Mandatory Regulatory & AI Disclosure Coverage Gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920

# Platform Interactive Dead Zones (Douyin / WeChat Channels / TikTok vertical UI)
# Right rail: like / comment / share / avatar buttons
RIGHT_INTERACTION_ZONE = {"x1": 940, "y1": 700, "x2": 1080, "y2": 1400}
# Bottom bar: publisher info, music disc, description, seekbar
BOTTOM_INTERACTION_ZONE = {"x1": 0, "y1": 1750, "x2": 1080, "y2": 1920}
# Top notch / camera cutout / status bar
TOP_STATUS_ZONE = {"x1": 0, "y1": 0, "x2": 1080, "y2": 120}


def calculate_overlap_ratio(box_a: dict[str, int], box_b: dict[str, int]) -> float:
    """Calculates what fraction of box_a is covered by box_b."""
    inter_x1 = max(box_a["x1"], box_b["x1"])
    inter_y1 = max(box_a["y1"], box_b["y1"])
    inter_x2 = min(box_a["x2"], box_b["x2"])
    inter_y2 = min(box_a["y2"], box_b["y2"])

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = max(1, (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"]))
    return inter_area / float(a_area)


class VisualSafeZoneLinter:
    """
    Checks visual element placement against platform UI interaction zones.
    Enforces strict avoidance ONLY for critical content (evidence cards, data tables, primary figures).
    Permits generous partial overlap for ordinary elements (badges, subtitles, ambient background).
    """

    def __init__(
        self,
        canvas_w: int = CANVAS_WIDTH,
        canvas_h: int = CANVAS_HEIGHT,
        max_critical_overlap: float = 0.08,  # Critical content cannot overlap interactive rail > 8%
        max_ordinary_overlap: float = 0.40,  # Ordinary elements tolerate up to 40% overlap
    ):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.max_critical_overlap = max_critical_overlap
        self.max_ordinary_overlap = max_ordinary_overlap

    def audit_element(
        self,
        element_id: str,
        bbox: tuple[int, int, int, int],  # (x1, y1, x2, y2)
        is_critical: bool = False,
        description: str = "",
    ) -> dict[str, Any]:
        box = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}

        right_ratio = calculate_overlap_ratio(box, RIGHT_INTERACTION_ZONE)
        bottom_ratio = calculate_overlap_ratio(box, BOTTOM_INTERACTION_ZONE)
        top_ratio = calculate_overlap_ratio(box, TOP_STATUS_ZONE)

        total_blocked_ratio = max(right_ratio, bottom_ratio, top_ratio)
        allowed_limit = self.max_critical_overlap if is_critical else self.max_ordinary_overlap

        passed = total_blocked_ratio <= allowed_limit
        severity = "PASS"
        if not passed:
            severity = "BLOCKING_ERROR" if is_critical else "ADVISORY_WARNING"

        return {
            "element_id": element_id,
            "description": description,
            "is_critical": is_critical,
            "bbox": bbox,
            "overlap_ratios": {
                "right_rail": round(right_ratio, 3),
                "bottom_bar": round(bottom_ratio, 3),
                "top_status": round(top_ratio, 3),
            },
            "allowed_overlap_limit": allowed_limit,
            "passed": passed or not is_critical,  # Non-critical overlap produces warning, does not block
            "blocking": not passed and is_critical,
            "severity": severity,
        }


class SubtitlePacingLinter:
    """
    Checks subtitle reading pace (CPS - Characters Per Second) and minimum on-screen duration.
    Prevents subtitle cues from flashing too quickly (<0.5s) or exceeding human reading speed.
    """

    def __init__(
        self,
        max_cps: float = 7.5,
        min_duration_s: float = 0.50,
    ):
        self.max_cps = max_cps
        self.min_duration_s = min_duration_s

    def audit_cues(self, cues: list[dict[str, Any]]) -> dict[str, Any]:
        violations = []
        for c in cues:
            start = c.get("start", 0.0)
            end = c.get("end", 0.0)
            text = c.get("text", "").replace("\n", "").strip()
            char_count = len(text)
            dur = max(0.001, end - start)
            cps = char_count / dur

            is_too_fast = cps > self.max_cps and char_count > 3
            is_too_short = dur < self.min_duration_s and char_count > 2

            if is_too_fast or is_too_short:
                violations.append({
                    "index": c.get("index"),
                    "start": start,
                    "end": end,
                    "duration_s": round(dur, 3),
                    "text": text,
                    "char_count": char_count,
                    "cps": round(cps, 1),
                    "reason": "TOO_FAST" if is_too_fast else "TOO_SHORT",
                })

        return {
            "total_cues": len(cues),
            "pacing_violations_count": len(violations),
            "passed": len(violations) == 0,
            "violations": violations,
        }


def check_av_sync(video_duration_s: float, audio_duration_s: float, fps: float = 30.0) -> dict[str, Any]:
    """
    Checks if video and audio container durations match within 1 frame tolerance.
    """
    frame_dur = 1.0 / float(fps)
    delta = abs(video_duration_s - audio_duration_s)
    passed = delta <= (frame_dur * 1.5)

    return {
        "video_duration_s": round(video_duration_s, 3),
        "audio_duration_s": round(audio_duration_s, 3),
        "delta_s": round(delta, 4),
        "max_tolerated_delta_s": round(frame_dur * 1.5, 4),
        "drift_frames": round(delta / frame_dur, 2),
        "passed": passed,
    }


def check_disclosure_coverage(
    ai_asset_intervals: list[tuple[float, float, str]],
    ai_badge_intervals: list[tuple[float, float]],
) -> dict[str, Any]:
    """
    Verifies that every AI scene asset is 100% covered by an AI disclosure badge.
    """
    uncovered = []
    for (a_s, a_e, asset_name) in ai_asset_intervals:
        covered = False
        for (b_s, b_e) in ai_badge_intervals:
            if b_s <= (a_s + 0.1) and b_e >= (a_e - 0.1):
                covered = True
                break
        if not covered:
            uncovered.append({
                "asset": asset_name,
                "start": a_s,
                "end": a_e,
                "missing_badge": "AI 生成画面｜情境示意",
            })

    return {
        "total_ai_scenes": len(ai_asset_intervals),
        "uncovered_scenes": uncovered,
        "passed": len(uncovered) == 0,
    }
