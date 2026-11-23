from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from jsonschema import Draft202012Validator, FormatChecker

from .file_hash import sha256_file
from .media_tools import resolve_media_tool
from .review_http import ReviewHandler, _require_loopback_host
from .storage import read_json, write_json, write_text_atomic
from .targeted_review_ui import render_targeted_review_html
from .video import probe_video


PACK_SCHEMA = "vkp-targeted-review-pack.v1"
DECISIONS_SCHEMA = "vkp-targeted-review-decisions.v1"
PACK_FILENAME = "targeted-review-pack.json"
PACK_HASH_FIELD = "pack_payload_sha256"
HTML_FILENAME = "targeted-review.html"
DECISIONS_FILENAME = "targeted-review-decisions.json"
LANES = {"audio", "subtitle", "picture", "privacy", "compliance", "publication", "mixed"}
PLAYBACK_MODES = {"normal", "audio_only", "muted", "frame_step"}
DECISIONS = {"pending", "pass", "needs_rework", "uncertain"}


class TargetedReviewError(ValueError):
    """Fail-closed error for targeted human-review artifacts."""


def _now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _payload_sha256(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _schema(name: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(value: dict[str, Any], schema_name: str) -> None:
    validator = Draft202012Validator(_schema(schema_name), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].path)
        raise TargetedReviewError(f"schema_validation_failed:{schema_name}:{path}:{errors[0].message}")


def _resolve_source(base: Path, value: str) -> Path:
    path = Path(str(value or "").strip()).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise TargetedReviewError(f"review_source_missing:{path}")
    return path


def _float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TargetedReviewError(f"{label}_invalid") from error
    if result < 0 or result != result or result in {float("inf"), float("-inf")}:
        raise TargetedReviewError(f"{label}_invalid")
    return round(result, 3)


def _timecode(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _input_rows(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    schema_version = str(raw.get("schema_version") or raw.get("schema") or "")
    if schema_version == "self-media-video-review-plan.v1":
        inputs = raw.get("inputs") if isinstance(raw.get("inputs"), dict) else {}
        source = inputs.get("source_media") if isinstance(inputs.get("source_media"), dict) else {}
        rows = raw.get("human_review_requests")
        if not isinstance(rows, list):
            raise TargetedReviewError("human_review_requests_required")
        return [dict(row) for row in rows if isinstance(row, dict)], source, str(raw.get("mode") or "video_review")
    if schema_version == "self-media-human-review-request-pack.v1":
        rows = raw.get("requests")
        if not isinstance(rows, list):
            raise TargetedReviewError("review_requests_required")
        return [dict(row) for row in rows if isinstance(row, dict)], {}, str(raw.get("scope") or "targeted_review")
    raise TargetedReviewError(f"unsupported_review_request_schema:{schema_version or 'missing'}")


def _source_row(source_path: Path, expected_sha: str, source_id: str) -> dict[str, Any]:
    metadata = probe_video(source_path)
    actual_sha = str(metadata.sha256).lower()
    if actual_sha != expected_sha:
        raise TargetedReviewError("artifact_hash_drift")
    return {
        "source_id": source_id,
        "label": source_path.name,
        "path": str(source_path),
        "sha256": actual_sha,
        "duration_seconds": round(
            _playback_duration_seconds(source_path, float(metadata.duration_seconds)), 3
        ),
        "width": int(metadata.width) if metadata.width else None,
        "height": int(metadata.height) if metadata.height else None,
        "fps": round(float(metadata.fps or 30.0), 3),
        "video_codec": _video_codec_name(source_path),
    }


def _video_codec_name(path: Path) -> str:
    """Record the video codec so browser playback failures can be diagnosed."""
    ffprobe = resolve_media_tool("ffprobe")
    if not ffprobe:
        return "unknown"
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            return "unknown"
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") if isinstance(payload, dict) else None
        codec = streams[0].get("codec_name") if streams and isinstance(streams[0], dict) else None
        return str(codec or "unknown").strip().lower()
    except (OSError, subprocess.SubprocessError, TypeError, ValueError, json.JSONDecodeError):
        return "unknown"


def _playback_duration_seconds(path: Path, video_duration_seconds: float) -> float:
    """Return the complete playable duration without changing frame-extraction semantics.

    VKP's canonical video probe intentionally reports the decodable video-stream
    bound. A browser review must also hear a slightly longer audio/container tail,
    so this local projection uses the larger container duration when available.
    """
    fallback = max(0.0, float(video_duration_seconds or 0.0))
    ffprobe = resolve_media_tool("ffprobe")
    if not ffprobe:
        return fallback
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        container_duration = float((result.stdout or "").strip()) if result.returncode == 0 else 0.0
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        return fallback
    if container_duration != container_duration or container_duration in {
        float("inf"),
        float("-inf"),
    }:
        return fallback
    return max(fallback, container_duration)


def _audio_position_candidate(value: Any, *, duration: float, label: str) -> tuple[float, float] | None:
    """Return a usable positive interval; word candidates may be unavailable and fall back."""
    if not isinstance(value, dict):
        return None
    try:
        start = _float(value.get("start_seconds"), f"{label}_start_seconds")
        end = _float(value.get("end_seconds"), f"{label}_end_seconds")
    except TargetedReviewError:
        return None
    if not (start < end <= duration + 0.05):
        return None
    return start, end


def _resolve_audio_position(row: dict[str, Any], *, duration: float) -> dict[str, Any] | None:
    """Use word timing when valid, otherwise explicitly downgrade to a sentence/segment range."""
    raw = row.get("audio_time_position")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TargetedReviewError("audio_time_position_invalid")

    word_raw = raw.get("word")
    word = _audio_position_candidate(word_raw, duration=duration, label="word_time")
    segment_raw = raw.get("sentence_segment")
    segment = _audio_position_candidate(segment_raw, duration=duration, label="sentence_segment_time")
    if segment_raw is not None and segment is None:
        raise TargetedReviewError("audio_sentence_segment_range_invalid")
    if word is not None and segment is not None and not (
        segment[0] <= word[0] < word[1] <= segment[1]
    ):
        raise TargetedReviewError("audio_sentence_segment_does_not_contain_word")

    if word is not None:
        issue_start, issue_end = word
        granularity = "word"
        fallback_reason = None
    elif segment is not None:
        issue_start, issue_end = segment
        granularity = "sentence_segment"
        fallback_reason = (
            "word_timestamp_unavailable"
            if word_raw is None
            else "word_timestamp_invalid_or_zero_duration"
        )
    else:
        raise TargetedReviewError("audio_position_unavailable_without_sentence_segment")

    # Preserve a valid wider review context when it contains the chosen focus; otherwise
    # use the sentence segment (or the word interval when no sentence is available).
    playback_interval = segment or word
    if playback_interval is None:  # Guard the invariant if this resolver is changed later.
        raise TargetedReviewError("audio_position_unavailable")
    playback_start, playback_end = playback_interval
    try:
        context_start = _float(row.get("review_start_seconds"), "review_start_seconds")
        context_end = _float(row.get("review_end_seconds"), "review_end_seconds")
        if (
            context_start < context_end <= duration + 0.05
            and context_start <= issue_start
            and issue_end <= context_end
            and context_end - context_start <= 15.001
        ):
            playback_start, playback_end = context_start, context_end
    except TargetedReviewError:
        pass

    return {
        "issue_start_seconds": issue_start,
        "issue_end_seconds": issue_end,
        "review_start_seconds": playback_start,
        "review_end_seconds": playback_end,
        "time_positioning_granularity": granularity,
        "time_positioning_fallback_reason": fallback_reason,
    }


def prepare_targeted_review_pack(input_path: str | Path, output_dir: str | Path, *, project_identity: dict[str,Any] | None = None) -> dict[str, Any]:
    request_path = Path(input_path).expanduser().resolve()
    if not request_path.is_file():
        raise TargetedReviewError(f"review_request_pack_missing:{request_path}")
    raw = read_json(request_path)
    if not isinstance(raw, dict):
        raise TargetedReviewError("review_request_pack_must_be_object")
    rows, default_source, default_stage = _input_rows(raw)
    if not rows:
        raise TargetedReviewError("review_requests_empty")
    full_source_rows = [row for row in rows if row.get("full_source_review") is True]
    if full_source_rows and len(rows) != 1:
        raise TargetedReviewError("full_source_review_must_be_single_item")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_cache: dict[Path, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    seen_review_ids: set[str] = set()

    for row in rows:
        review_id = str(row.get("review_id") or "").strip()
        if not review_id or review_id in seen_review_ids:
            raise TargetedReviewError(f"review_id_missing_or_duplicate:{review_id or 'missing'}")
        seen_review_ids.add(review_id)
        source_value = str(row.get("source_path") or default_source.get("path") or "").strip()
        source_path = _resolve_source(request_path.parent, source_value)
        expected_sha = str(row.get("artifact_sha256") or default_source.get("sha256") or "").strip().lower()
        if len(expected_sha) != 64:
            raise TargetedReviewError(f"artifact_sha256_required:{review_id}")
        source = source_cache.get(source_path)
        if source is None:
            try:
                source = _source_row(source_path, expected_sha, f"source-{len(sources) + 1:03d}")
            except TargetedReviewError as error:
                raise TargetedReviewError(f"{error}:{review_id}") from error
            source_cache[source_path] = source
            sources.append(source)
        elif source["sha256"] != expected_sha:
            raise TargetedReviewError(f"artifact_hash_conflict:{review_id}")

        duration = float(source["duration_seconds"])
        lane = str(row.get("review_lane") or "mixed").strip()
        playback = str(row.get("playback_mode") or "normal").strip()
        if lane not in LANES:
            raise TargetedReviewError(f"review_lane_invalid:{review_id}:{lane}")
        if playback not in PLAYBACK_MODES:
            raise TargetedReviewError(f"playback_mode_invalid:{review_id}:{playback}")
        if row.get("audio_time_position") is not None and lane != "audio":
            raise TargetedReviewError(f"audio_time_position_requires_audio_lane:{review_id}")
        audio_position = (
            _resolve_audio_position(row, duration=duration)
            if row.get("audio_time_position") is not None
            else None
        )
        if audio_position is not None:
            start = float(audio_position["review_start_seconds"])
            end = float(audio_position["review_end_seconds"])
            issue_start = float(audio_position["issue_start_seconds"])
            issue_end = float(audio_position["issue_end_seconds"])
        else:
            start = _float(row.get("review_start_seconds"), "review_start_seconds")
            end = _float(row.get("review_end_seconds"), "review_end_seconds")
            issue_start = _float(row.get("issue_start_seconds", start), "issue_start_seconds")
            issue_end = _float(row.get("issue_end_seconds", end), "issue_end_seconds")
        if not (start < end <= duration + 0.05):
            raise TargetedReviewError(f"review_range_out_of_bounds:{review_id}")
        if not (start <= issue_start < issue_end <= end):
            raise TargetedReviewError(f"issue_range_outside_review_range:{review_id}")
        full_source_review = row.get("full_source_review") is True
        if full_source_review and not (
            start <= 0.001
            and issue_start <= 0.001
            and duration - end <= 0.05
            and duration - issue_end <= 0.05
        ):
            raise TargetedReviewError(f"full_source_review_must_cover_entire_source:{review_id}")
        if not full_source_review and lane != "publication" and end - start > 15.001:
            raise TargetedReviewError(f"targeted_review_range_exceeds_15_seconds:{review_id}")
        raw_subtitle_cues = row.get("subtitle_cues")
        subtitle_cues: list[dict[str, Any]] = []
        if raw_subtitle_cues is not None:
            if not isinstance(raw_subtitle_cues, list) or len(raw_subtitle_cues) > 200:
                raise TargetedReviewError(f"subtitle_cues_invalid:{review_id}")
            for index, raw_cue in enumerate(raw_subtitle_cues):
                if not isinstance(raw_cue, dict):
                    raise TargetedReviewError(f"subtitle_cue_invalid:{review_id}:{index}")
                cue_start = _float(raw_cue.get("start_seconds"), "subtitle_cue_start_seconds")
                cue_end = _float(raw_cue.get("end_seconds"), "subtitle_cue_end_seconds")
                cue_text = str(raw_cue.get("text") or "").strip()
                if not (start <= cue_start < cue_end <= end):
                    raise TargetedReviewError(f"subtitle_cue_outside_review_range:{review_id}:{index}")
                if not cue_text or len(cue_text) > 1000:
                    raise TargetedReviewError(f"subtitle_cue_text_invalid:{review_id}:{index}")
                presentation = {}
                for field, choices, default in (
                    ("layer", {"dialogue", "age", "context"}, "dialogue"),
                    ("speaker", {"child", "father", "neutral"}, "neutral"),
                    ("motion", {"none", "soft_pop", "bounce", "love"}, "none"),
                    ("timing_status", {"candidate", "user_confirmed"}, "candidate"),
                ):
                    if field in raw_cue:
                        value = raw_cue.get(field, default)
                        if not isinstance(value, str) or value not in choices:
                            raise TargetedReviewError(f"subtitle_cue_presentation_invalid:{review_id}:{index}:{field}")
                        presentation[field] = value
                if "font_size_1080" in raw_cue:
                    size = _float(raw_cue["font_size_1080"], "subtitle_font_size_1080")
                    if not 36 <= size <= 120:
                        raise TargetedReviewError(f"subtitle_cue_font_size_invalid:{review_id}:{index}")
                    presentation["font_size_1080"] = size
                if "emphasis_reason" in raw_cue:
                    reason = raw_cue['emphasis_reason']
                    if not isinstance(reason, str) or not reason.strip() or len(reason)>500:
                        raise TargetedReviewError(f"subtitle_emphasis_reason_invalid:{review_id}:{index}")
                    presentation['emphasis_reason'] = reason
                subtitle_cues.append(
                    {
                        "start_seconds": cue_start,
                        "end_seconds": cue_end,
                        "timecode_start": _timecode(cue_start),
                        "timecode_end": _timecode(cue_end),
                        "text": cue_text,
                        **presentation,
                    }
                )
        if str(row.get("review_access_mode") or "source_time_range") != "source_time_range":
            raise TargetedReviewError(f"independent_clip_not_supported_by_default:{review_id}")
        purpose = str(row.get("review_purpose") or "").strip()
        question = str(row.get("review_question") or purpose).strip()
        if not purpose or not question:
            raise TargetedReviewError(f"review_purpose_required:{review_id}")
        status = str(row.get("status") or "pending").strip()
        if status not in DECISIONS:
            status = "pending"
        raw_labels = row.get("decision_labels")
        decision_labels: dict[str, str] | None = None
        if raw_labels is not None:
            if not isinstance(raw_labels, dict):
                raise TargetedReviewError(f"decision_labels_invalid:{review_id}")
            decision_labels = {}
            for decision in ("pass", "needs_rework", "uncertain"):
                label = str(raw_labels.get(decision) or "").strip()
                if not label or len(label) > 80:
                    raise TargetedReviewError(f"decision_label_invalid:{review_id}:{decision}")
                decision_labels[decision] = label
        request = {
            "review_id": review_id,
            "source_id": source["source_id"],
            "workflow_stage": str(row.get("workflow_stage") or default_stage or lane).strip(),
            "review_lane": lane,
            "review_purpose": purpose,
            "review_question": question,
        }
        request["issue_start_seconds"] = issue_start
        request["issue_end_seconds"] = issue_end
        request["review_start_seconds"] = start
        request["review_end_seconds"] = end
        request["timecode_start"] = _timecode(start)
        request["timecode_end"] = _timecode(end)
        request["playback_mode"] = playback
        request["review_access_mode"] = "source_time_range"
        request["time_positioning_granularity"] = (
            audio_position["time_positioning_granularity"] if audio_position else "time_range"
        )
        if subtitle_cues:
            request["subtitle_cues"] = subtitle_cues
        if audio_position and audio_position["time_positioning_fallback_reason"]:
            request["time_positioning_fallback_reason"] = audio_position["time_positioning_fallback_reason"]
        request["decision_options"] = ["pass", "needs_rework", "uncertain"]
        request["status"] = status
        if decision_labels:
            request["decision_labels"] = decision_labels
        if row.get("reference_text"):
            request["reference_text"] = str(row.get("reference_text") or "").strip()
        requests.append(request)

    pack = {"schema_version": PACK_SCHEMA, "pack_id": f"targeted-{request_path.stem}"}
    pack["generated_at"] = _now()
    pack["source_request_path"] = str(request_path)
    pack["source_request_sha256"] = sha256_file(request_path)
    pack["sources"] = sources
    pack["requests"] = requests
    from .review_export_context import identity_for_preparation,write_bound_review_export_context
    identity=identity_for_preparation(raw,request_path.parent,output,project_identity)
    if identity is not None:pack['project_identity']=identity
    if raw.get("transcript_edit") is not None:
        from .targeted_transcript import normalize_transcript_source
        pack["transcript_edit"] = normalize_transcript_source(raw["transcript_edit"], sources)
    pack["publication_boundary"] = "manual_only"  # Human gate; never publication authority.
    pack["implementation_record"] = {
        "intent": "让所有基于视频时间轴的人工审核在同一个VKP界面完成。",
        "decision": "默认单项审核区间不超过15秒；明确声明全片审核时，只允许单条请求覆盖完整原片。音频优先使用有效词级时间；缺失、无效或零时长时降级到句段时间。字幕候选可按原片时间在播放器上同步叠显预览。审核只引用原视频，不创建独立审核视频。",
        "reason": "短区间适用于精确定位；字幕整片复核可通过唯一全源请求一次播放全片并自动显示时间同步字幕，避免拆碎审核和导出片段。",
        "evidence": ["VKP本地Range播放器", "VKP审核草稿与JSON写回", "现有精确时间范围审核合同"],
        "effective_scope": "音频、字幕、画面、隐私、合规、发布及返工复核。",
    }
    pack[PACK_HASH_FIELD] = _payload_sha256(pack, PACK_HASH_FIELD)
    _validate(pack, "targeted-review-pack.v1.schema.json")
    write_json(output / PACK_FILENAME, pack)
    export_context=write_bound_review_export_context(output,pack)
    write_text_atomic(output / HTML_FILENAME, render_targeted_review_html(pack,export_context))
    if not (output / DECISIONS_FILENAME).exists():
        receipt = compose_review_receipt(pack, "待填写", [])
        write_json(output / DECISIONS_FILENAME, receipt)
    return _preparation_receipt(output, pack)


def _preparation_receipt(output: Path, pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "prepared",
        "bundle_dir": str(output),
        "review_html": str(output / HTML_FILENAME),
        "pack_path": str(output / PACK_FILENAME),
        "decisions_path": str(output / DECISIONS_FILENAME),
        "source_count": len(pack["sources"]),
        "request_count": len(pack["requests"]),
        "lanes": sorted({row["review_lane"] for row in pack["requests"]}),
        PACK_HASH_FIELD: pack[PACK_HASH_FIELD],
        "media_copies_created": 0,
        "export_naming_status": "configured" if pack.get('project_identity') else "missing_project_identity",
    }


def _normalize_reviewer(value: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise TargetedReviewError("reviewer_required")
    return result


def compose_review_receipt(pack: dict[str, Any], reviewer: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    reviewer = _normalize_reviewer(reviewer)
    supplied = {str(row.get("review_id") or ""): row for row in rows if isinstance(row, dict)}
    expected_ids = [str(row["review_id"]) for row in pack.get("requests") or []]
    unknown = sorted(set(supplied) - set(expected_ids))
    if unknown:
        raise TargetedReviewError(f"unknown_review_id:{unknown[0]}")
    normalized: list[dict[str, str]] = []
    for review_id in expected_ids:
        row = supplied.get(review_id) or {}
        decision = str(row.get("decision") or "pending")
        note = str(row.get("note") or "").strip()
        if decision not in DECISIONS:
            raise TargetedReviewError(f"decision_invalid:{review_id}")
        if len(note) > 1000:
            raise TargetedReviewError(f"decision_note_too_long:{review_id}")
        normalized.append({"review_id": review_id, "decision": decision, "note": note})
    result = {
        "schema_version": DECISIONS_SCHEMA,
        "pack_payload_sha256": str(pack.get(PACK_HASH_FIELD) or ""),
        "saved_at": _now(),
        "reviewer": reviewer,
        "status": "complete" if all(row["decision"] != "pending" for row in normalized) else "in_progress",
        "decisions": normalized,
    }
    result["authority_boundary"] = {
        "manual_review_only": True,
        "media_modified": False,
        "automatic_publish": False,
        "customer_contact": False,
    }
    result["receipt_payload_sha256"] = _payload_sha256(result, "receipt_payload_sha256")
    _validate(result, "targeted-review-decisions.v1.schema.json")
    return result


def load_targeted_review_bundle(bundle_dir: str | Path, *, verify_sources: bool = True) -> tuple[Path, dict[str, Any]]:
    root = Path(bundle_dir).expanduser().resolve()
    pack_path = root / PACK_FILENAME
    if not pack_path.is_file() or not (root / HTML_FILENAME).is_file():
        raise TargetedReviewError(f"not_a_targeted_review_bundle:{root}")
    pack = read_json(pack_path)
    if not isinstance(pack, dict):
        raise TargetedReviewError("targeted_review_pack_invalid")
    _validate(pack, "targeted-review-pack.v1.schema.json")
    if pack.get(PACK_HASH_FIELD) != _payload_sha256(pack, PACK_HASH_FIELD):
        raise TargetedReviewError("targeted_review_pack_integrity_failed")
    if pack.get("transcript_edit") is not None:
        from .targeted_transcript import normalize_transcript_source
        if normalize_transcript_source(pack["transcript_edit"], pack["sources"]) != pack["transcript_edit"]:
            raise TargetedReviewError("targeted_transcript_projection_invalid")
    if verify_sources:
        for source in pack.get("sources") or []:
            path = Path(str(source.get("path") or "")).resolve()
            if not path.is_file() or sha256_file(path) != source.get("sha256"):
                raise TargetedReviewError(f"targeted_review_source_hash_drift:{source.get('source_id')}")
    return root, pack


def live_visual_review_state(review_dir: Path) -> dict[str, Any] | None:
    """Opt-in live projection; historical packs and their decisions stay immutable."""
    binding = review_dir / "review-live-source.json"
    if not binding.is_file():
        return None
    from .targeted_visual_evidence import run_targeted_visual_evidence
    from .ocr_inventory import build_ocr_inventory
    try:
        config = read_json(binding)
        if config.get("scope") != "visual_only":
            return None
        root = Path(config["bundle_dir"]).resolve()
        queue = run_targeted_visual_evidence(root, write=False)
        inventory = build_ocr_inventory(root, read_json(root / "timeline.json"))
        acceptance = read_json(root / "acceptance-check.json") if (root / "acceptance-check.json").is_file() else {}
        indexes = sorted(set(queue.get("human_review_indexes", []) + queue.get("unresolved_document_indexes", []) + queue.get("triage", {}).get("temporal_indexes", [])))
        revision = sha256_file(root / "timeline.json")
        return {"status": "no_candidates" if not indexes else "review_required", "indexes": indexes,
                "checked_at": _now(), "timeline_sha256": revision, "bundle_dir": str(root),
                "ocr_execution": inventory["summary"], "acceptance_status": acceptance.get("status", "unknown"),
                "historical_pack": True,
                "pack_evidence_current": revision == config.get("timeline_sha256")}
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "unavailable", "error": str(exc), "pack_evidence_current": False}


class TargetedReviewHandler(ReviewHandler):
    pack: dict[str, Any] = {}
    source_paths: dict[str, Path] = {}
    source_stats: dict[str, tuple[int, int]] = {}

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            return self._json_error(HTTPStatus.MISDIRECTED_REQUEST, "invalid Host header")
        parsed = urlsplit(self.path)
        if parsed.path in {"/targeted-program-preview", "/targeted-program-media", "/api/targeted-review/program/status"}:
            from .targeted_program_preview import load_program_preview, render_program_preview
            try:
                preview = load_program_preview(self.bundle_dir, self.pack)
                if preview is None:
                    return self._json_error(HTTPStatus.NOT_FOUND, "continuous preview is not bound")
            except (ValueError, OSError, KeyError, TypeError) as error:
                return self._json_error(HTTPStatus.CONFLICT, str(error))
            if parsed.path == "/targeted-program-media":
                return self._serve_file(preview["media_path"], allow_range=True)
            if parsed.path == "/api/targeted-review/program/status":
                return self._json_response({"ok": True, "duration_s": preview["duration_s"], "status": "continuous_candidate"})
            payload = render_program_preview(preview).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._security_headers("text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path in {"/", f"/{HTML_FILENAME}"}:
            return self._serve_targeted_html()
        if parsed.path == "/targeted-transcript-editor" and self.pack.get("transcript_edit"):
            from .targeted_transcript import render_transcript_editor
            try:
                text = render_transcript_editor(self.pack, self.bundle_dir, self.csrf_token)
            except (ValueError, OSError) as error:
                return self._json_error(HTTPStatus.CONFLICT, str(error))
            payload = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._security_headers("text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path.startswith("/targeted-transcript-assets/") and self.pack.get("transcript_edit"):
            from .targeted_transcript import ASSETS, ASSET_ROOT
            name = parsed.path.removeprefix("/targeted-transcript-assets/")
            if name in ASSETS and (ASSET_ROOT / name).is_file():
                return self._serve_file(ASSET_ROOT / name, allow_range=False)
            return self._json_error(HTTPStatus.NOT_FOUND, "unknown transcript asset")
        if parsed.path == "/api/targeted-review/transcript/status" and self.pack.get("transcript_edit"):
            from .targeted_transcript import load_transcript_draft
            return self._json_response({"ok": True, "draft": load_transcript_draft(self.bundle_dir, self.pack)})
        if parsed.path == "/api/targeted-review/status":
            decisions = read_json(self.bundle_dir / DECISIONS_FILENAME)
            return self._json_response({
                "ok": True,
                "service": "vkp-targeted-review",
                "loopback_only": True,
                PACK_HASH_FIELD: self.pack.get(PACK_HASH_FIELD),
                "decisions": decisions,
                "live_visual_state": live_visual_review_state(self.bundle_dir),
            })
        if parsed.path == "/targeted-media":
            source_id = str((parse_qs(parsed.query).get("source_id") or [""])[0])
            path = self.source_paths.get(source_id)
            if path is None:
                return self._json_error(HTTPStatus.NOT_FOUND, "unknown source_id")
            current_stat = path.stat()
            if (current_stat.st_size, current_stat.st_mtime_ns) != self.source_stats[source_id]:
                return self._json_error(HTTPStatus.CONFLICT, "source changed after review server start")
            return self._serve_file(path, allow_range=True)
        return self._json_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        route = urlsplit(self.path).path
        if route not in {"/api/targeted-review/save", "/api/targeted-review/transcript/save"}:
            return self._json_error(HTTPStatus.NOT_FOUND, "not found")
        if not self._prepare_mutation():
            return
        try:
            live = live_visual_review_state(self.bundle_dir)
            if live and (live.get("status") != "review_required" or not live.get("pack_evidence_current")):
                return self._json_error(HTTPStatus.CONFLICT, "Historical review pack: regenerate a current version before saving. Existing decisions are preserved.")
            request = self._read_json_body()
            if request.get(PACK_HASH_FIELD) != self.pack.get(PACK_HASH_FIELD):
                return self._json_error(HTTPStatus.CONFLICT, "review pack changed; reload before saving")
            if route == "/api/targeted-review/transcript/save":
                if not self.pack.get("transcript_edit"):
                    return self._json_error(HTTPStatus.NOT_FOUND, "transcript editor not enabled")
                from .targeted_transcript import save_transcript_draft, TranscriptDraftConflict
                try:
                    result = save_transcript_draft(self.bundle_dir, self.pack, request)
                except TranscriptDraftConflict as error:
                    return self._json_error(HTTPStatus.CONFLICT, str(error))
                return self._json_response({"ok": True, "status": "draft_only", "saved_at": result["saved_at"], "revision_sha256": result["revision_sha256"]})
            result = compose_review_receipt(
                self.pack,
                str(request.get("reviewer") or ""),
                request.get("decisions") if isinstance(request.get("decisions"), list) else [],
            )
            write_json(self.bundle_dir / DECISIONS_FILENAME, result)
            return self._json_response({"ok": True, "status": result["status"], "receipt": result})
        except (TargetedReviewError, OSError, TypeError, ValueError) as error:
            return self._json_error(HTTPStatus.BAD_REQUEST, str(error))

    def _serve_targeted_html(self) -> None:
        from .targeted_review_ui import render_targeted_review_html
        try:
            from .review_export_context import resolve_review_export_context
            export_context=resolve_review_export_context(self.pack,root=self.bundle_dir)
            text = render_targeted_review_html(self.pack, export_context)
            from .targeted_program_preview import FILENAME, add_program_panel
            if (self.bundle_dir / FILENAME).is_file() and self.pack.get("transcript_edit"):
                text = add_program_panel(text)
        except (ValueError, OSError) as error:
            return self._json_error(HTTPStatus.CONFLICT, str(error))
        config = json.dumps({
            "status_url": "/api/targeted-review/status",
            "save_url": "/api/targeted-review/save",
            "media_url": "/targeted-media",
            "token": self.csrf_token,
            PACK_HASH_FIELD: self.pack.get(PACK_HASH_FIELD),
        }, ensure_ascii=False).replace("</", "<\\/")
        bootstrap = f"<script>window.VKP_TARGETED_REVIEW_API={config};</script>"
        text = text.replace("</head>", bootstrap + "</head>", 1)
        payload = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._security_headers("text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _security_headers(self, content_type: str) -> None:
        if urlsplit(self.path).path not in {"/targeted-transcript-editor", "/targeted-program-preview"}:
            return super()._security_headers(content_type)
        # Only this own-origin component is embeddable in the canonical review page.
        # Other review routes retain the existing DENY / frame-ancestors 'none'.
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'self'; base-uri 'none'; form-action 'self'")


def build_targeted_review_server(
    bundle_dir: str | Path, *, host: str = "127.0.0.1", port: int = 0, csrf_token: str | None = None
) -> ThreadingHTTPServer:
    _require_loopback_host(host)
    root, pack = load_targeted_review_bundle(bundle_dir, verify_sources=True)

    class Handler(TargetedReviewHandler):
        pass

    Handler.bundle_dir = root
    Handler.csrf_token = str(csrf_token or secrets.token_urlsafe(32))
    Handler.pack = pack
    Handler.source_paths = {
        str(row["source_id"]): Path(str(row["path"])).resolve()
        for row in pack["sources"]
    }
    Handler.source_stats = {
        source_id: (path.stat().st_size, path.stat().st_mtime_ns)
        for source_id, path in Handler.source_paths.items()
    }
    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare or open VKP exact-range targeted review")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("input_json")
    prepare.add_argument("output_dir")
    prepare.add_argument('--topic-title')
    prepare.add_argument('--video-title')
    prepare.add_argument('--video-id')
    serve = sub.add_parser("serve")
    serve.add_argument("bundle_dir")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        explicit={k:getattr(args,k) for k in ('topic_title','video_title','video_id')} if any(getattr(args,k) for k in ('topic_title','video_title','video_id')) else None
        result = prepare_targeted_review_pack(args.input_json, args.output_dir,project_identity=explicit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    server = build_targeted_review_server(args.bundle_dir, host=args.host, port=args.port)
    address, port = server.server_address[:2]
    url = f"http://{address}:{port}/"
    print(f"VKP targeted review UI: {url}", flush=True)
    print("Loopback only. Original media is referenced, not copied or modified.", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
