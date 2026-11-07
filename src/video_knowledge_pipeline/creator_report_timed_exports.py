"""Reuse native subtitle and CSV adapters for candidate-only report downloads."""
from __future__ import annotations

import csv
import io
import json
import math

from .asr_adapter import render_srt
from .models import TranscriptCue
from .transcript import format_timestamp


def render_report_srt(report: dict) -> tuple[str, dict]:
    cues, mapping = [], []
    for index, row in enumerate(report["transcript"]):
        start, end = float(row["start"]), float(row["end"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end:
            raise ValueError("invalid canonical subtitle time; no source mutation performed")
        entry = {"source_cue_index": row.get("source_cue_index", index),
                 "source_cue_id": row.get("source_cue_id", ""),
                 "source_segment_ids": row.get("source_segment_ids", []),
                 "start": start, "end": end, "srt_index": None, "status": "empty_gap"}
        if str(row.get("text") or "").strip():
            if end <= start:
                raise ValueError("nonempty SRT subtitle requires a positive duration; source preserved")
            # Native render_srt uses segment_id as its sequence when provided.
            # This export-only cue intentionally uses numeric SRT numbering;
            # the exact canonical identity stays in the adjacent source map.
            cues.append(TranscriptCue(start, end, str(row["text"]), speaker=row.get("speaker", "")))
            entry.update(srt_index=len(cues), status="exported_candidate")
        mapping.append(entry)
    if not cues:
        raise ValueError("no nonempty timed subtitles available for SRT export")
    return render_srt(cues), {"schema": "video_knowledge_pipeline.report_subtitle_source_map.v1",
        "candidate_only": True, "canonical_modified": False,
        "input_fingerprint": report["input_fingerprint"], "transcript_scope": report.get("transcript_scope", "unknown"),
        "rows": mapping}


def render_report_chapter_csv(report: dict) -> str:
    stream = io.StringIO(newline="")
    columns = ["chapter_id", "title", "start_time", "end_time", "boundary_kind", "boundary_source",
               "evidence_refs", "candidate_only", "human_confirmed"]
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in report.get("practical", {}).get("chapters", []):
        writer.writerow({"chapter_id": row["chapter_id"], "title": row["title"],
            "start_time": format_timestamp(row["start"]), "end_time": format_timestamp(row["end"]),
            "boundary_kind": "semantic_chapter", "boundary_source": row["boundary_source"],
            "evidence_refs": json.dumps(row["evidence_refs"], ensure_ascii=False),
            "candidate_only": True, "human_confirmed": False})
    return stream.getvalue()
