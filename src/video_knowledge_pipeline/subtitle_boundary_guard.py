"""
Subtitle Boundary Guard & Safe Margin Linter
--------------------------------------------
Automated quality gate for detecting and repairing subtitle boundary violations.
Ensures no subtitle text ever overflows the vertical video canvas (1080x1920).

Specifications:
- Canvas Width: 1080 px
- Safe Margin X: >= 80 px (symmetrical left and right)
- Max Safe Text Width: 920 px (1080 - 80*2)
- Base Font Size: 76 px (Microsoft YaHei Bold / CJK Bold)
- Max Chinese Characters per line: 11-12 chars (~836-912 px <= 920 px)

Core Capabilities:
1. `check`: Audits an SRT or cue list, reporting all overflow violations with timestamps and pixel metrics.
2. `fix`: Automatically repairs overflowing cues via:
   - Natural punctuation and max-length recursive wrapping (max 11-12 chars per line).
   - If line count > 2, seamlessly splits cue into consecutive temporal sub-cues based on duration.
   - 100% mathematical guarantee that EVERY rendered line is within 920px.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
DEFAULT_SAFE_MARGIN_X = 80
MAX_SAFE_WIDTH = CANVAS_WIDTH - DEFAULT_SAFE_MARGIN_X * 2  # 920 px
DEFAULT_FONT_SIZE = 76
MAX_CHARS_PER_LINE = 11

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


class SubtitleBoundaryGuard:
    def __init__(
        self,
        canvas_width: int = CANVAS_WIDTH,
        safe_margin_x: int = DEFAULT_SAFE_MARGIN_X,
        font_path: str | None = None,
        base_font_size: int = DEFAULT_FONT_SIZE,
    ):
        self.canvas_width = canvas_width
        self.safe_margin_x = safe_margin_x
        self.max_safe_width = canvas_width - safe_margin_x * 2
        self.font_path = font_path or self._resolve_default_font()
        self.base_font_size = base_font_size

        self._img = Image.new("RGB", (self.canvas_width, CANVAS_HEIGHT))
        self._draw = ImageDraw.Draw(self._img)
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}
        self._using_truetype = True

    def _resolve_default_font(self) -> str | None:
        for candidate in FONT_CANDIDATES:
            if os.path.exists(candidate):
                return candidate
        return None

    def get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._font_cache:
            if self.font_path and os.path.exists(self.font_path):
                try:
                    self._font_cache[size] = ImageFont.truetype(self.font_path, size)
                    self._using_truetype = True
                    return self._font_cache[size]
                except Exception:
                    pass
            self._font_cache[size] = ImageFont.load_default()
            self._using_truetype = False
        return self._font_cache[size]

    def measure_text(self, text: str, font_size: int | None = None) -> tuple[int, int]:
        size = font_size or self.base_font_size
        font = self.get_font(size)
        if self._using_truetype:
            bbox = self._draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            return w, h

        # Fallback accurate CJK text measurement when running on environments without TTF
        w = 0
        for char in text:
            # Full-width CJK characters are 1em wide, ASCII characters ~0.55em wide
            if ord(char) > 127:
                w += size
            else:
                w += int(round(size * 0.55))
        return w, size

    def audit_cue(self, cue: dict[str, Any], font_size: int | None = None) -> dict[str, Any]:
        text = cue.get("text", "").strip()
        lines = text.split("\n")
        size = font_size or self.base_font_size

        max_line_w = 0
        line_details = []
        is_overflow = False

        for l in lines:
            w, h = self.measure_text(l, size)
            left_margin = (self.canvas_width - w) // 2
            right_margin = self.canvas_width - w - left_margin
            overflow_px = max(0, w - self.max_safe_width)

            if overflow_px > 0 or left_margin < self.safe_margin_x or right_margin < self.safe_margin_x:
                is_overflow = True

            line_details.append({
                "line": l,
                "width_px": w,
                "height_px": h,
                "left_margin_px": left_margin,
                "right_margin_px": right_margin,
                "overflow_px": overflow_px,
            })
            if w > max_line_w:
                max_line_w = w

        return {
            "index": cue.get("index"),
            "start": cue.get("start"),
            "end": cue.get("end"),
            "text": text,
            "font_size": size,
            "max_line_width_px": max_line_w,
            "max_safe_width_px": self.max_safe_width,
            "is_overflow": is_overflow,
            "overflow_amount_px": max(0, max_line_w - self.max_safe_width),
            "lines": line_details,
        }

    def check_all(self, cues: list[dict[str, Any]], font_size: int | None = None) -> dict[str, Any]:
        violations = []
        clean_count = 0

        for cue in cues:
            res = self.audit_cue(cue, font_size)
            if res["is_overflow"]:
                violations.append(res)
            else:
                clean_count += 1

        pass_rate = (clean_count / len(cues) * 100) if cues else 100.0

        return {
            "schema": "vkp.subtitle-boundary-audit.v1",
            "passed": len(violations) == 0,
            "total_cues": len(cues),
            "compliant_cues": clean_count,
            "violating_cues": len(violations),
            "compliance_rate": f"{pass_rate:.1f}%",
            "canvas_width_px": self.canvas_width,
            "safe_margin_x_px": self.safe_margin_x,
            "max_safe_width_px": self.max_safe_width,
            "violations": violations,
        }

    def _split_into_safe_lines(self, text: str) -> list[str]:
        """Recursively split text so every single line is <= max_safe_width."""
        text = text.strip()
        if not text:
            return []

        w, _ = self.measure_text(text)
        if w <= self.max_safe_width:
            return [text]

        # Punctuation-aware splitting
        for punct in ["，", "、", "；", "。"]:
            if punct in text:
                idx = text.find(punct)
                left = text[: idx + 1]
                right = text[idx + 1 :]
                if left and right:
                    return self._split_into_safe_lines(left) + self._split_into_safe_lines(right)

        # No punctuation, split at safe character boundary
        split_point = min(MAX_CHARS_PER_LINE, max(1, len(text) // 2))
        left = text[:split_point]
        right = text[split_point:]
        return self._split_into_safe_lines(left) + self._split_into_safe_lines(right)

    def repair_cue(self, cue: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Guaranteed repair:
        1. Breaks input text into guaranteed safe lines (<= 920px).
        2. If 1 or 2 lines, formats as single cue with line break if 2 lines.
        3. If > 2 lines, splits into consecutive sub-cues with proportional time allocation.
        """
        audit = self.audit_cue(cue)
        if not audit["is_overflow"]:
            return [cue]

        raw_text = cue.get("text", "").replace("\r\n", " ").replace("\n", " ").strip()
        safe_lines = self._split_into_safe_lines(raw_text)

        start = cue.get("start", 0.0)
        end = cue.get("end", 0.0)
        dur = max(0.2, end - start)

        if len(safe_lines) <= 1:
            return [{
                "start": start,
                "end": end,
                "text": safe_lines[0] if safe_lines else "",
            }]

        if len(safe_lines) == 2:
            return [{
                "start": start,
                "end": end,
                "text": "\n".join(safe_lines),
            }]

        # Group lines into pairs
        paired_groups = []
        for i in range(0, len(safe_lines), 2):
            paired_groups.append(safe_lines[i : i + 2])

        total_chars = sum(len("".join(g)) for g in paired_groups)
        cursor = start
        result_cues = []

        for idx, grp in enumerate(paired_groups):
            grp_text = "\n".join(grp)
            grp_chars = len("".join(grp))
            fraction = (grp_chars / total_chars) if total_chars else (1.0 / len(paired_groups))
            grp_dur = dur * fraction

            grp_start = cursor
            grp_end = start + dur if idx == len(paired_groups) - 1 else cursor + grp_dur
            cursor = grp_end

            result_cues.append({
                "start": grp_start,
                "end": grp_end,
                "text": grp_text,
            })

        return result_cues

    def enforce_safe_cues(self, cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        safe_cues = []
        new_idx = 1
        for cue in cues:
            repaired = self.repair_cue(cue)
            for r in repaired:
                r_copy = dict(r)
                r_copy["index"] = new_idx
                safe_cues.append(r_copy)
                new_idx += 1
        return safe_cues


def parse_srt(srt_path: Path | str) -> list[dict[str, Any]]:
    path = Path(srt_path)
    content = path.read_text(encoding="utf-8").strip()
    blocks = re.split(r"\n\s*\n", content)
    cues = []

    def parse_time(ts: str) -> float:
        parts = ts.strip().split(":")
        hh = int(parts[0])
        mm = int(parts[1])
        ss, ms = parts[2].split(",")
        return hh * 3600 + mm * 60 + int(ss) + int(ms) / 1000.0

    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) >= 3:
            try:
                idx = int(lines[0])
                t_s, t_e = lines[1].split(" --> ")
                text = "\n".join(lines[2:]).strip()
                cues.append({
                    "index": idx,
                    "start": parse_time(t_s),
                    "end": parse_time(t_e),
                    "start_str": t_s,
                    "end_str": t_e,
                    "text": text,
                })
            except Exception:
                continue
    return cues


def format_srt_time(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        seconds += 1
        ms = 0
    total_seconds = int(seconds)
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def export_srt(cues: list[dict[str, Any]], out_path: Path | str) -> None:
    path = Path(out_path)
    lines = []
    for idx, c in enumerate(cues, 1):
        lines.append(str(idx))
        t_s = format_srt_time(c["start"])
        t_e = format_srt_time(c["end"])
        lines.append(f"{t_s} --> {t_e}")
        lines.append(c["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def check_srt_file(path: Path | str) -> dict[str, Any]:
    guard = SubtitleBoundaryGuard()
    cues = parse_srt(path)
    return guard.check_all(cues)


def fix_srt_file(in_path: Path | str, out_path: Path | str) -> dict[str, Any]:
    guard = SubtitleBoundaryGuard()
    cues = parse_srt(in_path)
    repaired_cues = guard.enforce_safe_cues(cues)
    export_srt(repaired_cues, out_path)
    return guard.check_all(repaired_cues)


def main():
    parser = argparse.ArgumentParser(description="Subtitle Boundary Guard CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check subtitle boundary compliance")
    check_parser.add_argument("srt", type=str, help="Path to SRT file")
    check_parser.add_argument("--json", type=str, help="Optional output JSON report path")

    fix_parser = subparsers.add_parser("fix", help="Fix subtitle boundary violations")
    fix_parser.add_argument("srt", type=str, help="Path to input SRT file")
    fix_parser.add_argument("--out", type=str, required=True, help="Path to output repaired SRT file")

    args = parser.parse_args()
    guard = SubtitleBoundaryGuard()

    if args.command == "check":
        srt_file = Path(args.srt)
        cues = parse_srt(srt_file)
        report = guard.check_all(cues)
        print("=" * 60)
        print(f"SUBTITLE BOUNDARY AUDIT: {srt_file.name}")
        print("=" * 60)
        print(f"Passed:            {report['passed']}")
        print(f"Total Cues:        {report['total_cues']}")
        print(f"Compliant:         {report['compliant_cues']}")
        print(f"Violations:        {report['violating_cues']}")
        print(f"Compliance Rate:   {report['compliance_rate']}")
        print(f"Max Safe Width:    {report['max_safe_width_px']} px (Margin >= {report['safe_margin_x_px']}px)")
        print("=" * 60)

        if report["violations"]:
            print("\nVIOLATION SAMPLES:")
            for v in report["violations"][:5]:
                print(f"  [Cue {v['index']:03d}] Width: {v['max_line_width_px']}px (Overflow +{v['overflow_amount_px']}px)")
                print(f"    Text: {v['text']}")

        if args.json:
            out_json = Path(args.json)
            out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nReport written to: {out_json.name}")

        sys.exit(0 if report["passed"] else 1)

    elif args.command == "fix":
        srt_file = Path(args.srt)
        cues = parse_srt(srt_file)
        safe_cues = guard.enforce_safe_cues(cues)
        out_file = Path(args.out)
        export_srt(safe_cues, out_file)
        print(f"Repaired {len(cues)} cues into {len(safe_cues)} boundary-safe cues.")
        print(f"Exported clean SRT to: {out_file.name}")


if __name__ == "__main__":
    main()
