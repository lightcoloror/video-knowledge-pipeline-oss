"""Read-only oral chapter/representative-frame view over existing evidence.

Chapters remain semantic navigation, never fabricated technical shot boundaries.
The existing summary chapter builder, timestamp parser and safe image adapter are
reused; this module only selects a bounded, deduplicated reading projection.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .creator_report_renderers import frame_preview
from .shot_breakdown import _frames
from .smart_summary_chapters import build_smart_summary_chapter_pack
from .transcript import parse_timestamp

_CLOCK = r"\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?"
_HEADING = re.compile(r"^###\s+(" + _CLOCK + r")\s*[-–—~至]\s*(" + _CLOCK + r")(.*)$")


def _finite(value: Any, default: float = 0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) and number >= 0 else default


def _reference_chapters(text: str, duration: float) -> list[dict]:
    rows, inside = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line[3:].strip() == "分段总结"
        match = _HEADING.match(line) if inside else None
        if not match:
            continue
        start, end = parse_timestamp(match[1]), parse_timestamp(match[2])
        if not 0 <= start < end <= duration + 2 or (rows and start < rows[-1]["end"]):
            return []
        rows.append({"start": start, "end": min(end, duration),
                     "title": match[3].strip(" ｜|·—-:") or "口播章节",
                     "boundary_source": "current_registered_summary"})
    if rows:
        # Existing hand-written reference headings may round fractional cue
        # boundaries to seconds. Preserve those declared bounds while extending
        # only navigation coverage across <=2s rounding gaps and the media tail.
        if rows[0]["start"] > 2 or duration - rows[-1]["end"] > 2:
            return []
        if any(right["start"] - left["end"] > 2 for left, right in zip(rows, rows[1:])):
            return []
        for index, row in enumerate(rows):
            row["declared_start"], row["declared_end"] = row["start"], row["end"]
            if index == 0:
                row["start"] = 0.0
            row["end"] = rows[index + 1]["start"] if index + 1 < len(rows) else duration
    return rows


def build_oral_chapters(root: Path, *, transcript: list[dict], timeline: list[dict],
                        visuals: list[dict], reference: dict, max_images: int = 2) -> dict:
    duration = max((_finite(row.get("end")) for row in [*transcript, *timeline]), default=0)
    chapters = _reference_chapters(str(reference.get("text") or ""), duration)
    chapter_error = ""
    if not chapters:
        try:
            existing = build_smart_summary_chapter_pack(root, write=False)
            chapters = [{"start": _finite(row.get("start")), "end": _finite(row.get("end")),
                         "title": ("口播章节" if re.match(r"^(?:objects|actions|interface_state|timeline)\s*[:=]", str(row.get("title") or ""), re.I)
                                   else str(row.get("title") or "口播章节")),
                         "boundary_source": "existing_semantic_chapter_builder"}
                        for row in existing.get("chapters", []) if isinstance(row, dict)
                        and _finite(row.get("end")) > _finite(row.get("start"))]
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            chapter_error = type(exc).__name__
    if not chapters and duration:
        chapters = [{"start": 0.0, "end": duration, "title": "口播内容",
                     "boundary_source": "single_reading_range_no_chapter_evidence"}]

    samples, rejected = [], []
    for position, source in enumerate(timeline):
        if not isinstance(source, dict):
            continue
        time = _finite(source.get("midpoint"), _finite(source.get("start"), -1))
        if time < 0 or time > duration:
            rejected.append({"timeline_index": position, "reason": "invalid_sample_time"})
            continue
        for raw in _frames(source):
            try:
                preview = frame_preview(root, raw)
                preview["time"] = time
                preview["timeline_index"] = position
                samples.append(preview)
                break
            except (OSError, ValueError, Warning) as exc:
                rejected.append({"timeline_index": position, "reason": type(exc).__name__})
    samples.sort(key=lambda row: (row["time"], row["timeline_index"]))
    by_index = {int(row["evidence_refs"][0].split(":")[1]): row for row in visuals
                if row.get("evidence_refs") and re.fullmatch(r"timeline:\d+", row["evidence_refs"][0])}
    previews, used = {}, set()
    for index, chapter in enumerate(chapters, 1):
        chapter.update(chapter_id=f"oral-chapter-{index:03d}", index=index, candidate_only=True,
                       human_confirmed=False, boundary_kind="semantic_chapter", images=[], observations=[],
                       screen_text_candidates=[])
        related = [r for r in samples if chapter["start"] <= r["time"] < chapter["end"]
                   or (index == len(chapters) and r["time"] == chapter["end"])]
        unique = [r for r in related if r["sha256"] not in used]
        # Spread picks across the available sample sequence, rather than repeating
        # one nearby representative for every unreviewed technical cut.
        picks = unique if len(unique) <= max_images else [unique[round(i * (len(unique)-1) / (max_images-1))]
                                                          for i in range(max_images)] if max_images > 1 else unique[:1]
        for sample in picks:
            if sample["sha256"] in used:
                continue
            used.add(sample["sha256"])
            key = f"oral-frame-{sample['timeline_index'] + 1:04d}"
            previews[key] = sample
            chapter["images"].append(key)
            visual = by_index.get(sample["timeline_index"], {})
            chapter["observations"].extend(visual.get("observations", []))
            for line in visual.get("ocr_lines", []):
                if line.get("text") and line["text"] not in {r["text"] for r in chapter["screen_text_candidates"]}:
                    chapter["screen_text_candidates"].append(line)
        chapter["sample_count"] = len(related)
        chapter["evidence_refs"] = [f"timeline:{r['timeline_index']}" for r in related]
    return {"chapters": chapters, "previews": previews, "summary": {
        "available_sample_count": len(samples), "rendered_unique_frames": len(previews),
        "unique_sample_hashes": len({r["sha256"] for r in samples}), "max_images_per_chapter": max_images,
        "technical_shots_used_as_chapters": False, "chapter_builder_error": chapter_error,
        "rejected_assets": rejected}}
