from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .asr_adapter import render_srt
from .local_media_progress import LocalMediaProgress, ProgressCallback
from .models import TranscriptCue, now_iso
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .transcript import parse_transcript
from .transcript_speakers import (
    cue_speaker,
    cue_speaker_role,
    speaker_label_map,
    speaker_payload,
)

SCHEMA = "video_knowledge_pipeline.asr_transcript_postprocess.v1"
QUESTION_MARKERS = ("是不是", "有没有", "怎么", "为什么", "哪一个", "哪一", "谁", "多少", "怎么办", "对不对", "好不好", "要不要", "能不能")
QUESTION_ENDINGS = ("吗", "么", "嘛")
SENTENCE_END = ("。", "！", "？", ".", "!", "?")
SPLIT_MARKERS = ("那", "好", "然后", "接下来", "具体", "其实", "同时", "另外", "首先", "其次", "第一个", "第二个", "第三个", "比如说", "所以")


def postprocess_asr_transcript(
    bundle_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    target_seconds: float = 18.0,
    max_chars: int = 180,
    punctuation_mode: str = "readable",
    segment_policy: str = "preserve",
    set_corrected: bool = True,
    write: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Turn raw ASR cues into a more readable transcript sidecar.

    This is a local, deterministic post-process. The default preserves every
    source segment boundary; readable merging/splitting requires the explicit
    readable_merge policy. It does not call ASR, LLM, vision, or downloaders.
    """

    root = Path(bundle_dir).expanduser().resolve()
    policy = _normalize_segment_policy(segment_policy)
    mode = _normalize_punctuation_mode(punctuation_mode)
    progress = (
        LocalMediaProgress(
            pipeline="local_transcript_postprocess",
            snapshot_path=root / "asr-transcript-postprocess-progress.json",
            events_path=root / "asr-transcript-postprocess-progress.jsonl",
            callback=progress_callback,
        )
        if write
        else None
    )
    if progress:
        progress.emit(stage="load", percent=0, message="Loading source transcript")
    source = _resolve_input(root, input_path)
    cues = parse_transcript(source)
    cues = _ensure_segment_identity(cues)
    target_seconds_value = max(2.0, float(target_seconds or 18.0))
    max_chars_value = max(40, int(max_chars or 180))
    if progress:
        progress.emit(
            stage="transform",
            percent=35,
            current_item=0,
            total_items=len(cues),
            message=f"Applying {policy} segment policy",
        )
    if policy == "preserve":
        expanded = list(cues)
        merged = _preserve_cues(cues, punctuation_mode=mode)
    else:
        expanded = _expand_long_cues(cues, target_seconds=target_seconds_value, max_chars=max_chars_value)
        merged = _merge_cues(
            expanded,
            target_seconds=target_seconds_value,
            max_chars=max_chars_value,
            punctuation_mode=mode,
        )
    transformation_records = _transformation_records(merged)
    speaker_labels = speaker_label_map(merged)
    segments = [
        {
            "id": cue.segment_id,
            "segment_id": cue.segment_id,
            "source_segment_ids": list(cue.source_segment_ids),
            "start": cue.start,
            "end": cue.end,
            "text": cue.text,
            "transformations": list(cue.transformations),
            **speaker_payload(cue, speaker_labels),
            "metadata": {
                **dict(cue.metadata),
                "source": "asr_postprocess",
                "input_path": str(source),
            },
        }
        for cue in merged
    ]
    payload = {
        "schema": "video_knowledge_pipeline.postprocessed_transcript.v1",
        "provider": "local_asr_postprocess",
        "mode": mode,
        "segment_policy": policy,
        "source_path": str(source),
        "created_at": now_iso(),
        "segments": segments,
        "transformations": transformation_records,
    }
    out_json = root / "postprocessed-transcript.json"
    out_srt = root / "postprocessed-transcript.srt"
    readable_json = root / "readable-transcript.json"
    readable_srt = root / "readable-transcript.srt"
    corrected_json = root / "corrected-transcript.json"
    corrected_srt = root / "corrected-transcript.srt"
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "input_path": str(source),
        "status": "completed" if merged else "empty",
        "source_segment_count": len(cues),
        "expanded_segment_count": len(expanded),
        "postprocessed_segment_count": len(merged),
        "target_seconds": float(target_seconds),
        "max_chars": int(max_chars),
        "punctuation_mode": mode,
        "segment_policy": policy,
        "transformations": transformation_records,
        "set_corrected": bool(set_corrected),
        "artifacts": {
            "json": str(out_json),
            "srt": str(out_srt),
            "readable_json": str(readable_json),
            "readable_srt": str(readable_srt),
            "transformations_json": str(root / "transcript-transformations.json"),
        },
        "operator_boundary": {"local_only": True, "no_cloud_call": True, "does_not_modify_raw_asr": True},
        "updated_at": now_iso(),
    }
    if write:
        if progress:
            progress.emit(
                stage="write",
                percent=80,
                current_item=len(merged),
                total_items=len(merged),
                message="Writing preserved transcript sidecars",
            )
        write_json(out_json, payload)
        out_srt.write_text(render_srt(merged), encoding="utf-8")
        write_json(readable_json, {**payload, "schema": "video_knowledge_pipeline.readable_transcript.v1"})
        write_json(
            root / "transcript-transformations.json",
            {
                "schema": "video_knowledge_pipeline.transcript_transformations.v1",
                "segment_policy": policy,
                "source_segment_count": len(cues),
                "output_segment_count": len(merged),
                "transformations": transformation_records,
            },
        )
        readable_srt.write_text(render_srt(merged), encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if not isinstance(manifest, dict):
            manifest = {}
        manifest["postprocessed_transcript_json"] = "postprocessed-transcript.json"
        manifest["postprocessed_transcript_srt"] = "postprocessed-transcript.srt"
        manifest["readable_transcript_json"] = "readable-transcript.json"
        manifest["readable_transcript_srt"] = "readable-transcript.srt"
        if set_corrected:
            write_json(corrected_json, payload)
            corrected_srt.write_text(render_srt(merged), encoding="utf-8")
            manifest["corrected_transcript_json"] = "corrected-transcript.json"
            manifest["corrected_transcript_srt"] = "corrected-transcript.srt"
            manifest["transcript_json"] = "corrected-transcript.json"
            manifest["transcript_srt"] = "corrected-transcript.srt"
            result["artifacts"]["corrected_json"] = str(corrected_json)
            result["artifacts"]["corrected_srt"] = str(corrected_srt)
        manifest["asr_transcript_postprocess"] = result
        manifest["mcp_postprocess_asr_transcript_args"] = "mcp-postprocess-asr-transcript.args.json"
        write_json(manifest_path, manifest)
        write_json(root / "asr-transcript-postprocess.json", result)
        (root / "asr-transcript-postprocess.md").write_text(_render_markdown(result), encoding="utf-8")
        write_json(
            root / "mcp-postprocess-asr-transcript.args.json",
            {
                "bundle_dir": str(root),
                "input_path": str(source),
                "target_seconds": target_seconds,
                "max_chars": max_chars,
                "punctuation_mode": mode,
                "segment_policy": policy,
                "set_corrected": set_corrected,
                "write": True,
            },
        )
        result["run_artifact"] = register_bundle_run(
            root,
            run_type="asr_transcript_postprocess",
            run_id="asr-transcript-postprocess",
            status="completed" if merged else "needs_input",
            title="ASR transcript punctuation and segmentation postprocess",
            summary=f"segments {len(cues)} -> {len(merged)}; set_corrected={set_corrected}",
            inputs={"input_path": str(source)},
            parameters={"target_seconds": target_seconds, "max_chars": max_chars, "punctuation_mode": mode, "set_corrected": set_corrected},
            artifacts=[{"key": key, "path": value} for key, value in result["artifacts"].items()],
            failed_items=[] if merged else [{"id": "transcript", "reason": "empty_transcript", "detail": str(source)}],
            retry_command=f".\\scripts\\video-knowledge.ps1 postprocess-asr-transcript '{root}'",
            next_actions=["Run export-knowledge-note so full-transcript.md and smart-summary inputs use corrected-transcript.json."],
            operator_boundary=result["operator_boundary"],
            write=True,
        )
        write_json(root / "asr-transcript-postprocess.json", result)
        (root / "asr-transcript-postprocess.md").write_text(_render_markdown(result), encoding="utf-8")
        if progress:
            terminal = "completed" if merged else "failed"
            progress.emit(
                stage="finalize",
                percent=100,
                current_item=len(merged),
                total_items=len(merged),
                message="Transcript postprocess completed" if merged else "Transcript postprocess produced no usable segments",
                status=terminal,
                output_paths=[out_json, readable_json],
                report_paths=[root / "asr-transcript-postprocess.md", root / "transcript-transformations.json"],
                details={"segment_policy": policy, "transformations": len(transformation_records)},
            )
            result["progress"] = progress.artifacts()
            write_json(root / "asr-transcript-postprocess.json", result)
    return result


def _resolve_input(root: Path, input_path: str | Path | None) -> Path:
    if input_path:
        path = Path(input_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        if path.exists():
            return path.resolve()
        return (root / path).resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        manifest = {}
    # This command is an ASR readability postprocess. Prefer raw/normalized ASR
    # sidecars even when a previous run has already promoted corrected-transcript.
    for key in ("normalized_transcript_json", "asr_transcript_json", "raw_transcript_json", "source_transcript", "transcript_path"):
        value = str(manifest.get(key) or "").strip()
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            if path.exists():
                return path.resolve()
    for name in ("normalized-transcript.json", "raw-asr-output.json", "transcript.json", "timeline-transcript.json"):
        path = root / name
        if path.exists():
            return path.resolve()
    for key in ("source_arbitrated_transcript_json", "llm_corrected_transcript_json", "human_corrected_transcript_json", "corrected_transcript_json", "transcript_json"):
        value = str(manifest.get(key) or "").strip()
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            if path.exists():
                return path.resolve()
    corrected = root / "corrected-transcript.json"
    if corrected.exists():
        return corrected.resolve()
    raise FileNotFoundError(f"no transcript found for bundle: {root}")


def _normalize_segment_policy(value: str) -> str:
    policy = str(value or "preserve").strip().lower().replace("-", "_")
    if policy in {"preserve", "strict_preserve"}:
        return "preserve"
    if policy in {"readable_merge", "merge", "legacy"}:
        return "readable_merge"
    raise ValueError("segment_policy must be preserve or readable_merge")


def _ensure_segment_identity(cues: list[TranscriptCue]) -> list[TranscriptCue]:
    rows: list[TranscriptCue] = []
    for index, cue in enumerate(cues, start=1):
        segment_id = str(cue.segment_id or f"segment-{index:06d}")
        source_ids = list(cue.source_segment_ids or [segment_id])
        rows.append(
            TranscriptCue(
                start=float(cue.start),
                end=float(cue.end),
                text=str(cue.text or ""),
                segment_id=segment_id,
                source_segment_ids=source_ids,
                transformations=list(cue.transformations),
                speaker=cue_speaker(cue),
                speaker_role=cue_speaker_role(cue),
                metadata=dict(cue.metadata),
            )
        )
    return rows


def _preserve_cues(cues: list[TranscriptCue], *, punctuation_mode: str) -> list[TranscriptCue]:
    rows: list[TranscriptCue] = []
    for cue in cues:
        source_text = str(cue.text or "")
        text = _punctuate(_clean_text(source_text), mode=punctuation_mode)
        transformations = list(cue.transformations)
        if text != source_text:
            transformations.append(
                {
                    "type": "text_cleanup",
                    "policy": "preserve",
                    "source_segment_ids": list(cue.source_segment_ids),
                    "boundary_changed": False,
                }
            )
        rows.append(
            TranscriptCue(
                start=cue.start,
                end=cue.end,
                text=text,
                segment_id=cue.segment_id,
                source_segment_ids=list(cue.source_segment_ids),
                transformations=transformations,
                speaker=cue_speaker(cue),
                speaker_role=cue_speaker_role(cue),
                metadata=dict(cue.metadata),
            )
        )
    return rows


def _transformation_records(cues: list[TranscriptCue]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue in cues:
        for transformation in cue.transformations:
            rows.append(
                {
                    "output_segment_id": cue.segment_id,
                    "source_segment_ids": list(cue.source_segment_ids),
                    **dict(transformation),
                }
            )
    return rows


def format_asr_review_draft(
    cues: list[TranscriptCue],
    *,
    start_seconds: float,
    end_seconds: float,
    target_seconds: float = 14.0,
    max_chars: int = 120,
) -> str:
    """Build a punctuation-only human-review draft from fine-grained ASR cues."""

    selected = [
        cue
        for cue in cues
        if max(float(start_seconds), float(cue.start)) <= min(float(end_seconds), float(cue.end))
    ]
    groups: list[list[TranscriptCue]] = []
    current: list[TranscriptCue] = []
    for cue in selected:
        text = _clean_text(cue.text)
        if not text:
            continue
        gap = max(0.0, float(cue.start) - float(current[-1].end)) if current else 0.0
        next_chars = sum(len(_clean_text(row.text)) for row in current) + len(text)
        duration = float(cue.end) - float(current[0].start) if current else 0.0
        should_flush = bool(current) and (
            _speaker_boundary_changed(current, cue)
            or gap >= 0.65
            or duration >= max(6.0, float(target_seconds))
            or next_chars > max(40, int(max_chars))
            or _strong_sentence_end(current[-1].text)
        )
        if should_flush:
            groups.append(current)
            current = []
        current.append(
            TranscriptCue(
                start=cue.start,
                end=cue.end,
                text=text,
                speaker=cue_speaker(cue),
                speaker_role=cue_speaker_role(cue),
                metadata=dict(cue.metadata),
            )
        )
    if current:
        groups.append(current)
    return "\n".join(
        _join_cues(group, punctuation_mode="readable").text
        for group in groups
        if group
    ).strip()


def _expand_long_cues(cues: list[TranscriptCue], *, target_seconds: float, max_chars: int) -> list[TranscriptCue]:
    expanded: list[TranscriptCue] = []
    for cue in cues:
        text = _clean_text(cue.text)
        duration = max(0.0, float(cue.end) - float(cue.start))
        if not text:
            continue
        if duration <= target_seconds * 1.35 and len(text) <= max_chars:
            expanded.append(
                TranscriptCue(
                    start=cue.start,
                    end=cue.end,
                    text=text,
                    segment_id=cue.segment_id,
                    source_segment_ids=list(cue.source_segment_ids),
                    transformations=list(cue.transformations),
                    speaker=cue_speaker(cue),
                    speaker_role=cue_speaker_role(cue),
                    metadata=dict(cue.metadata),
                )
            )
            continue
        desired_chunks = max(1, int(max(duration / max(1.0, target_seconds), len(text) / max(1, max_chars)) + 0.999))
        chunk_budget = max(50, min(max_chars, int((len(text) / desired_chunks) + 0.999)))
        chunks = _split_long_text(text, max_chars=chunk_budget)
        if len(chunks) <= 1:
            expanded.append(
                TranscriptCue(
                    start=cue.start,
                    end=cue.end,
                    text=text,
                    segment_id=cue.segment_id,
                    source_segment_ids=list(cue.source_segment_ids),
                    transformations=list(cue.transformations),
                    speaker=cue_speaker(cue),
                    speaker_role=cue_speaker_role(cue),
                    metadata=dict(cue.metadata),
                )
            )
            continue
        total_chars = max(1, sum(len(chunk) for chunk in chunks))
        cursor = float(cue.start)
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                end = float(cue.end)
            else:
                share = len(chunk) / total_chars
                end = min(float(cue.end), cursor + duration * share)
            expanded.append(
                TranscriptCue(
                    start=cursor,
                    end=end,
                    text=chunk,
                    segment_id=f"{cue.segment_id}-split-{index + 1:02d}",
                    source_segment_ids=list(cue.source_segment_ids),
                    transformations=list(cue.transformations)
                    + [
                        {
                            "type": "explicit_split",
                            "policy": "readable_merge",
                            "source_segment_ids": list(cue.source_segment_ids),
                            "part": index + 1,
                            "part_count": len(chunks),
                            "boundary_changed": True,
                        }
                    ],
                    speaker=cue_speaker(cue),
                    speaker_role=cue_speaker_role(cue),
                    metadata=dict(cue.metadata),
                )
            )
            cursor = end
    return expanded


def _split_long_text(text: str, *, max_chars: int) -> list[str]:
    value = _clean_text(text)
    if len(value) <= max_chars:
        return [value] if value else []
    chunks: list[str] = []
    start = 0
    soft_min = max(28, int(max_chars * 0.45))
    while start < len(value):
        hard_end = min(len(value), start + max_chars)
        if hard_end >= len(value):
            chunks.append(value[start:].strip())
            break
        window = value[start:hard_end]
        split_at = _best_split_offset(window, soft_min=soft_min)
        if split_at <= 0:
            split_at = len(window)
        chunk = value[start:start + split_at].strip()
        if chunk:
            chunks.append(chunk)
        start = start + split_at
    return [chunk for chunk in chunks if chunk]


def _best_split_offset(window: str, *, soft_min: int) -> int:
    punctuation_offsets = [window.rfind(mark) + 1 for mark in ("。", "？", "！", "；", ";", "，", ",") if window.rfind(mark) >= soft_min]
    if punctuation_offsets:
        return max(punctuation_offsets)
    marker_offsets: list[int] = []
    for marker in SPLIT_MARKERS:
        pos = window.rfind(marker)
        if pos >= soft_min:
            marker_offsets.append(pos)
    if marker_offsets:
        return max(marker_offsets)
    return len(window)

def _merge_cues(cues: list[TranscriptCue], *, target_seconds: float, max_chars: int, punctuation_mode: str) -> list[TranscriptCue]:
    rows: list[TranscriptCue] = []
    current: list[TranscriptCue] = []
    for cue in cues:
        text = _clean_text(cue.text)
        if not text:
            continue
        next_len = sum(len(_clean_text(row.text)) for row in current) + len(text)
        start = current[0].start if current else cue.start
        should_flush = bool(current) and (
            _speaker_boundary_changed(current, cue)
            or (cue.end - start) >= target_seconds
            or next_len > max_chars
            or _strong_sentence_end(current[-1].text)
        )
        if should_flush:
            rows.append(_join_cues(current, punctuation_mode=punctuation_mode))
            current = []
        current.append(
            TranscriptCue(
                start=cue.start,
                end=cue.end,
                text=text,
                segment_id=cue.segment_id,
                source_segment_ids=list(cue.source_segment_ids),
                transformations=list(cue.transformations),
                speaker=cue_speaker(cue),
                speaker_role=cue_speaker_role(cue),
                metadata=dict(cue.metadata),
            )
        )
    if current:
        rows.append(_join_cues(current, punctuation_mode=punctuation_mode))
    return rows


def _join_cues(cues: list[TranscriptCue], *, punctuation_mode: str) -> TranscriptCue:
    speakers = {cue_speaker(cue) for cue in cues if cue_speaker(cue)}
    if len(speakers) > 1:
        raise ValueError("readable_merge_crosses_speaker_boundary")
    roles = {cue_speaker_role(cue) for cue in cues if cue_speaker_role(cue)}
    start = cues[0].start
    end = max(cue.end for cue in cues)
    parts = [_clean_text(cue.text) for cue in cues if _clean_text(cue.text)]
    text = _join_parts_for_readability(parts, punctuation_mode=punctuation_mode)
    text = _punctuate(text, mode=punctuation_mode)
    source_ids = [
        source_id
        for cue in cues
        for source_id in (cue.source_segment_ids or [cue.segment_id])
        if source_id
    ]
    transformations = [
        dict(value)
        for cue in cues
        for value in cue.transformations
    ]
    if len(cues) > 1:
        transformations.append(
            {
                "type": "explicit_merge",
                "policy": "readable_merge",
                "source_segment_ids": source_ids,
                "boundary_changed": True,
            }
        )
    return TranscriptCue(
        start=start,
        end=end,
        text=text,
        segment_id=cues[0].segment_id if len(cues) == 1 else f"merge-{cues[0].segment_id}-{cues[-1].segment_id}",
        source_segment_ids=source_ids,
        transformations=transformations,
        speaker=next(iter(speakers), ""),
        speaker_role=next(iter(roles), "") if len(roles) == 1 else "",
        metadata=dict(cues[0].metadata),
    )


def _speaker_boundary_changed(current: list[TranscriptCue], cue: TranscriptCue) -> bool:
    """Keep explicit readable merges inside one diarization cluster."""

    if not current:
        return False
    previous = cue_speaker(current[-1])
    incoming = cue_speaker(cue)
    return bool((previous or incoming) and previous != incoming)


def _clean_text(text: str) -> str:
    value = re.sub(r"\s+", "", str(text or ""))
    value = re.sub(r"<\|[^>]+\|>", "", value)
    return value.strip()


def _normalize_punctuation_mode(mode: str) -> str:
    value = str(mode or "readable").strip().lower()
    if value in {"terminal", "conservative"}:
        return "conservative"
    if value in {"preserve", "none"}:
        return "preserve"
    return "readable"


def _join_parts_for_readability(parts: list[str], *, punctuation_mode: str) -> str:
    clean_parts = [_normalize_inline_punctuation(part) for part in parts if _normalize_inline_punctuation(part)]
    if not clean_parts:
        return ""
    if punctuation_mode != "readable" or len(clean_parts) == 1:
        return "".join(clean_parts)
    rows: list[str] = []
    for index, part in enumerate(clean_parts):
        value = part.rstrip("，,")
        if index < len(clean_parts) - 1 and not value.endswith(("。", "？", "！", "；", ";", "：", ":")):
            value += "，"
        rows.append(value)
    return "".join(rows)


def _punctuate(text: str, *, mode: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = _normalize_inline_punctuation(value)
    # Intent: make the documented preserve mode actually preserve terminal
    # punctuation instead of turning a source comma into the invalid `，。`.
    # Decision: return after safe full-width normalisation and before adding a
    # sentence terminator.  Readable/conservative modes keep their old policy.
    # Reason: Cantonese SenseVoice cues commonly end in commas; appending a full
    # stop produced hundreds of false quality warnings in real interview runs.
    # Evidence: the 2026-08-10 interview bundles reproduced `，。` in 281 cues
    # even when the operator selected `--punctuation-mode preserve`.
    # Effective scope: preserve mode text only; cue IDs, timestamps, speakers,
    # raw ASR and the other punctuation modes are unchanged.
    if mode == "preserve":
        return value
    if mode == "readable":
        value = _insert_readable_clause_commas(value)
    if any(mark in value for mark in SENTENCE_END):
        return value
    return value + ("？" if _looks_like_question(value) else "。")


def _normalize_inline_punctuation(text: str) -> str:
    value = str(text or "").strip()
    value = value.replace(",", "，").replace(";", "；").replace(":", "：")
    value = re.sub(r"[，,]{2,}", "，", value)
    value = re.sub(r"[。\.]{2,}", "。", value)
    value = re.sub(r"[？?]{2,}", "？", value)
    value = re.sub(r"[！!]{2,}", "！", value)
    return value


def _insert_readable_clause_commas(text: str) -> str:
    value = str(text or "")
    if not value or _punctuation_count(value) >= max(2, len(value) // 45):
        return value
    for marker in sorted(SPLIT_MARKERS, key=len, reverse=True):
        value = _insert_comma_before_marker(value, marker)
    return re.sub(r"，{2,}", "，", value).strip("，")


def _insert_comma_before_marker(text: str, marker: str) -> str:
    if not marker:
        return text
    pieces: list[str] = []
    cursor = 0
    for match in re.finditer(re.escape(marker), text):
        pos = match.start()
        if pos <= 10 or pos - cursor < 14:
            continue
        prev = text[pos - 1] if pos > 0 else ""
        if prev in "，。？！；：":
            continue
        pieces.append(text[cursor:pos] + "，")
        cursor = pos
    if not pieces:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)


def _punctuation_count(text: str) -> int:
    return sum(1 for char in str(text or "") if char in "，。？！；：,.!?;:")


def _looks_like_question(text: str) -> bool:
    value = _clean_text(text)
    tail = value[-24:]
    if value.endswith(QUESTION_ENDINGS):
        return True
    return any(marker in tail for marker in QUESTION_MARKERS)


def _strong_sentence_end(text: str) -> bool:
    value = str(text or "").strip()
    return bool(value.endswith(SENTENCE_END))


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ASR Transcript Postprocess",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Source segments: `{result.get('source_segment_count')}`",
        f"- Expanded segments: `{result.get('expanded_segment_count')}`",
        f"- Postprocessed segments: `{result.get('postprocessed_segment_count')}`",
        f"- Punctuation mode: `{result.get('punctuation_mode')}`",
        f"- Set corrected: `{result.get('set_corrected')}`",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in (result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines).rstrip() + "\n"
