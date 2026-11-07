from __future__ import annotations

import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import jsonschema

from .artifact_freshness import build_dependency_snapshot, validate_dependency_snapshot
from .asr_vad_activity_audit import SCHEMA as ACTIVITY_AUDIT_SCHEMA
from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .local_media_contracts import build_ffmpeg_execution_receipt
from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .models import TranscriptCue, now_iso
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic
from .technical_shot_detection import load_verified_technical_shots
from .transcript import format_timestamp, parse_transcript


PLAN_SCHEMA = "video_knowledge_pipeline.long_video_content_segmentation_plan.v1"
REVIEW_SCHEMA = "video_knowledge_pipeline.long_video_content_review_notes.v1"
APPROVED_SCHEMA = "video_knowledge_pipeline.long_video_content_approved_edit.v1"
RENDER_SCHEMA = "video_knowledge_pipeline.long_video_content_render_receipt.v1"

PLAN_PATH = "exports/long-video-fast-segment-plan.json"
PLAN_MARKDOWN_PATH = "exports/long-video-fast-segment-plan.md"
REVIEW_PATH = "long-video-fast-segment-review.todo.json"
MCP_PLAN_ARGS_PATH = "mcp-long-video-fast-segment-plan.args.json"
APPROVED_PATH = "exports/long-video-fast-segment-approved.json"
QA_PATH = "exports/long-video-fast-segment-render-qa.json"
RENDER_RECEIPT_PATH = "exports/long-video-fast-segment-render-receipt.json"

SEMANTIC_DROP_PATTERNS = {
    "technical_blank": (
        r"(?:调试|测试一下|试音|没声音|冇声|听得到吗|睇到吗)",
        r"(?:共享屏幕|share screen|网络卡|断网|重新连接|登录一下)",
    ),
    "waiting_or_technical_setup": (
        r"(?:稍等|等一下|等一会|等佢|等阵)",
        r"(?:准备开始|还没开始|未开始|马上开始|稍后开始)",
    ),
    "non_information_smalltalk": (
        r"(?:吃饭了吗|食咗饭未|今天天气|今日天气|路上堵车|塞车)",
        r"(?:最近怎么样|最近点样|好久不见|好耐冇见)",
    ),
}


def build_long_video_fast_segment_plan(
    bundle_dir: str | Path,
    *,
    media_path: str | Path | None = None,
    transcript_path: str | Path | None = None,
    vad_path: str | Path | None = None,
    activity_audit_path: str | Path | None = None,
    profile: str = "auto",
    long_silence_seconds: float = 4.0,
    edge_blank_seconds: float = 8.0,
    repeat_similarity: float = 0.92,
    write: bool = True,
) -> dict[str, Any]:
    """Build a review-only content-aware edit plan from existing VKP evidence.

    Intent: shorten very long videos without mistaking low-confidence content for
    disposable footage. Decision: combine the existing transcript, VAD, Timeline,
    and verified technical-shot projections, then emit candidates only. Reason:
    silence, retakes, waiting, and visual-only content require different evidence.
    Evidence: VKP's FunASR/Silero VAD, content_clip_boundary, technical shots, and
    videocut-kit-derived review invariants. Effective scope: derived review files;
    the source media, Timeline, transcript, and evidence are never modified.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    manifest = _read_object(manifest_path, "bundle manifest")
    media = _resolve_media(root, manifest, media_path)
    transcript = _resolve_transcript(root, manifest, transcript_path)
    cues = parse_transcript(transcript)
    if not cues or not any(cue.end > cue.start for cue in cues):
        raise ValueError("a timestamped transcript is required for safe long-video segmentation")
    vad = _resolve_optional(root, vad_path, _manifest_value(manifest, "silero_vad_candidate", "asr_vad_json"))
    vad_payload = _read_object(vad, "VAD evidence") if vad else {}
    activity_audit = _resolve_activity_audit(
        root,
        manifest,
        explicit=activity_audit_path,
        vad_path=vad,
    )
    activity_payload = (
        _validated_activity_audit(activity_audit, media_path=media, vad_path=vad)
        if activity_audit
        else {}
    )
    timeline_path = root / "timeline.json"
    timeline = read_json(timeline_path) if timeline_path.is_file() else []
    timeline_rows = [row for row in timeline if isinstance(row, dict)] if isinstance(timeline, list) else []
    shots, shot_provenance = load_verified_technical_shots(root)
    duration = _media_duration(cues, vad_payload, timeline_rows, shots)
    if duration <= 0:
        raise ValueError("media duration cannot be established from current bundle evidence")

    parameters = {
        "profile": _profile(profile, cues),
        "long_silence_seconds": _positive(long_silence_seconds, "long_silence_seconds"),
        "edge_blank_seconds": _positive(edge_blank_seconds, "edge_blank_seconds"),
        "repeat_similarity": _bounded(repeat_similarity, "repeat_similarity", 0.75, 1.0),
    }
    candidates: list[dict[str, Any]] = []
    candidates.extend(
        _silence_candidates(
            cues,
            vad_payload,
            duration=duration,
            timeline_rows=timeline_rows,
            shots=shots,
            threshold=parameters["long_silence_seconds"],
            edge_threshold=parameters["edge_blank_seconds"],
        )
    )
    candidates.extend(_idle_candidates(cues, profile=parameters["profile"]))
    candidates.extend(_repeat_candidates(cues, similarity=parameters["repeat_similarity"]))
    candidates.extend(_apply_activity_audit_evidence(candidates, activity_payload))
    candidates = _deduplicate_candidates(candidates, duration=duration)
    for index, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"long-video-drop-{index:05d}"
        row["start_time"] = format_timestamp(row["start"])
        row["end_time"] = format_timestamp(row["end"])
        row["preview"] = {
            "media_path": str(media),
            "start": row["start"],
            "end": row["end"],
            "pre_roll_seconds": 2.0,
            "post_roll_seconds": 2.0,
        }

    # The shared dependency snapshot intentionally accepts Bundle-local inputs.
    # External media/transcripts remain hash-bound below and are validated
    # explicitly at apply/render time rather than weakening that path boundary.
    inputs: list[dict[str, Any]] = []
    if transcript.is_relative_to(root):
        inputs.append({"role": "transcript", "path": transcript})
    if vad and vad.is_relative_to(root):
        inputs.append({"role": "vad", "path": vad})
    if activity_audit and activity_audit.is_relative_to(root):
        inputs.append({"role": "activity_audit", "path": activity_audit})
    if timeline_path.is_file():
        inputs.append({"role": "timeline", "path": timeline_path})
    shot_path = Path(str(shot_provenance.get("path") or ""))
    if shot_path.is_file() and shot_path.is_relative_to(root):
        inputs.append({"role": "technical_shots", "path": shot_path})
    snapshot = build_dependency_snapshot(
        root,
        subject="long-video-fast-segment",
        inputs=inputs,
        source_run_id="long-video-fast-segment-plan",
        producer_schema=PLAN_SCHEMA,
    )
    result: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "needs_human_review",
        "generated_at": now_iso(),
        "bundle_dir": str(root),
        "source_media": _artifact(media),
        "duration_seconds": round(duration, 6),
        "parameters": parameters,
        "evidence": {
            "transcript": _artifact(transcript),
            "vad": _artifact(vad) if vad else None,
            "activity_audit": _artifact(activity_audit) if activity_audit else None,
            "timeline": _artifact(timeline_path) if timeline_path.is_file() else None,
            "technical_shots": shot_provenance,
        },
        "dependency_snapshot": snapshot,
        "summary": {
            "candidate_count": len(candidates),
            "safe_candidate_count": sum(row["classification"] == "drop_safe_candidate" for row in candidates),
            "review_candidate_count": sum(row["classification"] == "drop_review_required" for row in candidates),
            "candidate_drop_seconds": round(sum(row["end"] - row["start"] for row in candidates), 6),
        },
        "segments": candidates,
        "review_required": True,
        "automatic_delete_allowed": False,
        "source_media_modified": False,
        "operator_boundary": {
            "local_only": True,
            "candidate_only": True,
            "human_confirmation_required": True,
            "low_confidence_defaults_to_keep": True,
            "source_media_modified": False,
            "automatic_publish": False,
            "external_provider_called": False,
            "activity_audit_candidate_only": True,
        },
        "artifacts": {
            "plan_json": str(root / PLAN_PATH),
            "plan_markdown": str(root / PLAN_MARKDOWN_PATH),
            "review_todo": str(root / REVIEW_PATH),
            "mcp_args": str(root / MCP_PLAN_ARGS_PATH),
        },
    }
    result["plan_sha256"] = _plan_sha(result)
    _validate_schema(result, "long-video-content-segmentation-plan.v1.schema.json")
    review = _review_template(result)
    _validate_schema(review, "long-video-content-review-notes.v1.schema.json")
    if write:
        with bundle_write_lock(root, operation="long_video_fast_segment_plan", timeout_seconds=1.0):
            write_json(root / PLAN_PATH, result)
            write_text_atomic(root / PLAN_MARKDOWN_PATH, _render_plan_markdown(result))
            write_json(root / REVIEW_PATH, review)
            write_json(
                root / MCP_PLAN_ARGS_PATH,
                {
                    "bundle_dir": str(root),
                    "media_path": str(media),
                    "transcript_path": str(transcript),
                    "vad_path": str(vad or ""),
                    "activity_audit_path": str(activity_audit or ""),
                    **parameters,
                    "write": True,
                },
            )
            manifest.update(
                {
                    "long_video_fast_segment_plan_json": PLAN_PATH,
                    "long_video_fast_segment_plan_markdown": PLAN_MARKDOWN_PATH,
                    "long_video_fast_segment_review_todo": REVIEW_PATH,
                    "mcp_long_video_fast_segment_plan_args": MCP_PLAN_ARGS_PATH,
                    "long_video_fast_segment_updated_at": result["generated_at"],
                }
            )
            write_json(manifest_path, manifest)
        register_bundle_run(
            root,
            run_type="long_video_fast_segment_plan",
            run_id="long-video-fast-segment-plan",
            status="needs_review",
            title="超长视频内容感知快速分段候选",
            summary=f"Generated {len(candidates)} review-only removal candidate(s).",
            inputs={"dependency_snapshot": snapshot},
            parameters=parameters,
            artifacts=[root / PLAN_PATH, root / PLAN_MARKDOWN_PATH, root / REVIEW_PATH],
            next_actions=["逐段预览并填写 review.todo.json；确认后运行 apply-long-video-fast-segment-review。"],
            operator_boundary=result["operator_boundary"],
            write=True,
        )
    return result


def apply_long_video_fast_segment_review(
    bundle_dir: str | Path,
    review_json: str | Path,
    *,
    plan_json: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Apply explicit operator decisions; undecided or uncertain ranges are kept."""

    root = Path(bundle_dir).expanduser().resolve()
    plan_path = _resolve_inside(root, plan_json or PLAN_PATH, "plan")
    review_path = _resolve_inside(root, review_json, "review notes")
    plan = _read_object(plan_path, "long-video plan")
    review = _read_object(review_path, "long-video review")
    _validate_schema(plan, "long-video-content-segmentation-plan.v1.schema.json")
    _validate_schema(review, "long-video-content-review-notes.v1.schema.json")
    if plan.get("plan_sha256") != _plan_sha(plan):
        raise ValueError("long-video plan integrity check failed")
    if review.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("review does not bind the current long-video plan")
    if review.get("source_media_sha256") != plan["source_media"]["sha256"]:
        raise ValueError("review does not bind the current source media")
    _verify_artifact(dict(plan["source_media"]), "source media")
    for label, artifact in (plan.get("evidence") or {}).items():
        if isinstance(artifact, dict) and {"path", "bytes", "sha256"}.issubset(artifact):
            _verify_artifact(artifact, str(label))
    freshness = validate_dependency_snapshot(root, dict(plan["dependency_snapshot"]))
    if freshness.get("status") != "fresh":
        raise ValueError(f"long-video plan inputs are not fresh: {freshness.get('status')}")
    confirmation = review.get("operator_confirmation") or {}
    if confirmation.get("confirmed") is not True or not str(confirmation.get("confirmed_by") or "").strip():
        raise ValueError("explicit operator confirmation is required")
    expected = {str(row["candidate_id"]): row for row in plan["segments"]}
    rows = review.get("decisions") or []
    actual = {str(row.get("candidate_id") or ""): row for row in rows if isinstance(row, dict)}
    if set(actual) != set(expected):
        raise ValueError("review decisions must cover every current candidate exactly once")
    if len(actual) != len(rows):
        raise ValueError("duplicate or invalid review decision")
    delete_ranges: list[tuple[float, float]] = []
    decisions: list[dict[str, Any]] = []
    for candidate_id, source in expected.items():
        decision = str(actual[candidate_id].get("decision") or "")
        if decision not in {"keep", "drop"}:
            raise ValueError(f"candidate {candidate_id} must be explicitly kept or dropped")
        if decision == "drop" and source.get("classification") not in {"drop_safe_candidate", "drop_review_required"}:
            raise ValueError(f"candidate {candidate_id} is not eligible for removal")
        row = {
            "id": candidate_id,
            "start": float(source["start"]),
            "end": float(source["end"]),
            "kind": str(source["kind"]),
            "action": "delete" if decision == "drop" else "keep",
            "confirmed": True,
            "source": "human_review",
            "reason": str(actual[candidate_id].get("note") or source["reason"]),
            "evidence_ids": list(source.get("evidence_ids") or []),
        }
        decisions.append(row)
        if decision == "drop":
            delete_ranges.append((row["start"], row["end"]))
    delete_ranges = _merge_ranges(delete_ranges)
    keep_ranges = _complement(delete_ranges, float(plan["duration_seconds"]))
    if not keep_ranges:
        raise ValueError("review would delete the entire source media")
    result: dict[str, Any] = {
        "schema": APPROVED_SCHEMA,
        "status": "ready_for_explicit_render",
        "created_at": now_iso(),
        "bundle_dir": str(root),
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path), "plan_sha256": plan["plan_sha256"]},
        "review": {"path": str(review_path), "sha256": sha256_file(review_path)},
        "source_media": dict(plan["source_media"]),
        "dependency_snapshot": dict(plan["dependency_snapshot"]),
        "freshness": freshness,
        "decisions": decisions,
        "delete_segments": [_range_row(start, end) for start, end in delete_ranges],
        "keep_segments": [_range_row(start, end) for start, end in keep_ranges],
        "summary": {
            "deleted_seconds": round(sum(end - start for start, end in delete_ranges), 6),
            "kept_seconds": round(sum(end - start for start, end in keep_ranges), 6),
            "delete_count": len(delete_ranges),
            "keep_count": len(keep_ranges),
        },
        "operator_confirmation": dict(confirmation),
        "operator_boundary": {
            "human_confirmed": True,
            "source_media_modified": False,
            "render_executed": False,
            "automatic_publish": False,
            "automatic_fallback": False,
        },
    }
    result["approved_sha256"] = _approved_sha(result)
    if write:
        with bundle_write_lock(root, operation="apply_long_video_fast_segment_review", timeout_seconds=1.0):
            write_json(root / APPROVED_PATH, result)
            write_json(root / "edit.decisions.json", decisions)
            write_json(root / "delete_segments.json", result["delete_segments"])
            write_json(root / "cut.segments.json", result["keep_segments"])
            manifest_path = root / "manifest.json"
            manifest = _read_object(manifest_path, "bundle manifest")
            manifest.update(
                {
                    "long_video_fast_segment_approved_json": APPROVED_PATH,
                    "long_video_fast_segment_edit_decisions": "edit.decisions.json",
                    "long_video_fast_segment_delete_segments": "delete_segments.json",
                    "long_video_fast_segment_keep_segments": "cut.segments.json",
                }
            )
            write_json(manifest_path, manifest)
        register_bundle_run(
            root,
            run_type="long_video_fast_segment_review",
            run_id=f"long-video-review-{result['approved_sha256'][:12]}",
            status="ready_for_explicit_render",
            title="超长视频分段人工确认",
            summary=f"Approved {len(delete_ranges)} removal range(s); source remains unchanged.",
            inputs={"plan_sha256": plan["plan_sha256"], "review_sha256": sha256_file(review_path)},
            parameters={"human_confirmed": True},
            artifacts=[root / APPROVED_PATH, root / "edit.decisions.json", root / "delete_segments.json", root / "cut.segments.json"],
            operator_boundary=result["operator_boundary"],
            write=True,
        )
    return result


def render_long_video_fast_segment(
    bundle_dir: str | Path,
    *,
    approved_json: str | Path | None = None,
    output_path: str | Path | None = None,
    execute: bool = False,
    timeout_seconds: int = 14400,
    write: bool = True,
) -> dict[str, Any]:
    """Preview or execute one exact, CPU FFmpeg render after human approval."""

    root = Path(bundle_dir).expanduser().resolve()
    approved_path = _resolve_inside(root, approved_json or APPROVED_PATH, "approved edit")
    approved = _read_object(approved_path, "approved edit")
    if approved.get("schema") != APPROVED_SCHEMA or approved.get("approved_sha256") != _approved_sha(approved):
        raise ValueError("approved edit integrity check failed")
    if approved.get("status") != "ready_for_explicit_render" or (approved.get("operator_confirmation") or {}).get("confirmed") is not True:
        raise ValueError("approved edit is not human-confirmed")
    freshness = validate_dependency_snapshot(root, dict(approved["dependency_snapshot"]))
    if freshness.get("status") != "fresh":
        raise ValueError(f"approved edit inputs are not fresh: {freshness.get('status')}")
    media = Path(str(approved["source_media"]["path"])).resolve()
    if not media.is_file() or sha256_file(media) != approved["source_media"]["sha256"]:
        raise ValueError("source media changed after approval")
    output = Path(output_path).expanduser().resolve() if output_path else root / "exports" / f"{media.stem}.content-trimmed.mp4"
    if output == media:
        raise ValueError("render output must never overwrite source media")
    if output.exists() and execute:
        raise FileExistsError(f"render output already exists; refusing overwrite: {output}")
    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available through the VKP shared media-tool resolver")
    keep = [(float(row["start"]), float(row["end"])) for row in approved["keep_segments"]]
    filter_path = root / "exports" / "long-video-fast-segment-filter.txt"
    filter_text = _filter_complex(keep)
    command = [
        str(Path(ffmpeg).resolve()), "-hide_banner", "-nostdin", "-n", "-i", str(media),
        "-filter_complex_script", str(filter_path), "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", str(output),
    ]
    result: dict[str, Any] = {
        "schema": RENDER_SCHEMA,
        "status": "planned" if not execute else "running",
        "created_at": now_iso(),
        "execute": bool(execute),
        "source_media": dict(approved["source_media"]),
        "approved_edit": {"path": str(approved_path), "sha256": sha256_file(approved_path), "approved_sha256": approved["approved_sha256"]},
        "output_path": str(output),
        "filter_script": {"path": str(filter_path), "sha256": _text_sha(filter_text)},
        "command": ["ffmpeg", *command[1:]],
        "qa": None,
        "ffmpeg_execution_receipt": None,
        "operator_boundary": {
            "source_media_modified": False,
            "new_copy_only": True,
            "human_confirmation_required": True,
            "human_confirmation_present": True,
            "automatic_fallback": False,
            "automatic_publish": False,
            "external_provider_called": False,
        },
    }
    if write:
        write_text_atomic(filter_path, filter_text)
    if not execute:
        result["render_sha256"] = _render_sha(result)
        if write:
            write_json(root / RENDER_RECEIPT_PATH, result)
            _update_render_manifest(root)
        return result
    if not write:
        raise ValueError("execute=True requires write=True for an auditable render")
    streams = _probe_streams(media)
    if not streams["video"] or not streams["audio"]:
        raise ValueError("the exact v1 render route requires one video and one audio stream; no silent stream fallback is allowed")
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(1, int(timeout_seconds)),
        check=False,
        env=local_tool_subprocess_env(),
    )
    if completed.returncode != 0:
        result.update({"status": "failed", "error": (completed.stderr or completed.stdout)[-4000:]})
        result["render_sha256"] = _render_sha(result)
        write_json(root / RENDER_RECEIPT_PATH, result)
        return result
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("ffmpeg returned success but the output is missing or empty")
    expected_duration = sum(end - start for start, end in keep)
    actual_duration = _probe_duration(output)
    qa = {
        "schema": "video_knowledge_pipeline.long_video_content_render_qa.v1",
        "status": "passed" if abs(actual_duration - expected_duration) <= max(0.75, len(keep) * 0.05) else "failed",
        "expected_duration_seconds": round(expected_duration, 6),
        "actual_duration_seconds": round(actual_duration, 6),
        "duration_delta_seconds": round(actual_duration - expected_duration, 6),
        "output": _artifact(output),
    }
    write_json(root / QA_PATH, qa)
    receipt = build_ffmpeg_execution_receipt(
        root,
        operation="render",
        ffmpeg_path=command[0],
        actual_argv=command,
        inputs=[{"role": "source_video", "path": media}, {"role": "approved_edit", "path": approved_path}, {"role": "filter_script", "path": filter_path}],
        outputs=[{"role": "rendered_video", "path": output}, {"role": "render_qa", "path": root / QA_PATH}],
        requested_backend="cpu",
        selected_backend="cpu",
        hardware_accelerated=False,
        fallback_used=False,
        allowed_roots=[root, media.parent, Path(command[0]).parent, output.parent],
        write=True,
    )
    result.update({"status": "completed" if qa["status"] == "passed" else "degraded", "qa": qa, "ffmpeg_execution_receipt": receipt})
    result["render_sha256"] = _render_sha(result)
    write_json(root / RENDER_RECEIPT_PATH, result)
    _update_render_manifest(root)
    return result


def _silence_candidates(
    cues: list[TranscriptCue], vad: dict[str, Any], *, duration: float,
    timeline_rows: list[dict[str, Any]], shots: list[dict[str, Any]], threshold: float,
    edge_threshold: float,
) -> list[dict[str, Any]]:
    intervals = _vad_intervals(vad) or [(cue.start, cue.end) for cue in cues if cue.text.strip()]
    merged = _merge_ranges(intervals)
    gaps = _complement(merged, duration)
    vad_confirmed = bool(vad) and str(vad.get("status") or "") == "completed"
    rows: list[dict[str, Any]] = []
    for start, end in gaps:
        edge = start <= 0.001 or end >= duration - 0.001
        if end - start < (edge_threshold if edge else threshold):
            continue
        visual = _visual_evidence(timeline_rows, shots, start, end)
        classification = "drop_safe_candidate" if vad_confirmed and not visual else "drop_review_required"
        kind = "intro_outro_blank" if edge else "long_silence"
        rows.append(_candidate(start, end, kind, classification, "verified_non_speech_gap" if vad_confirmed else "transcript_gap_requires_vad_review", [*visual, "vad" if vad_confirmed else "transcript_timing"]))
    return rows


def _idle_candidates(cues: list[TranscriptCue], *, profile: str) -> list[dict[str, Any]]:
    patterns = {
        kind: [re.compile(value, re.IGNORECASE) for value in values]
        for kind, values in SEMANTIC_DROP_PATTERNS.items()
    }
    rows = []
    for cue in cues:
        text = cue.text.strip()
        for kind, kind_patterns in patterns.items():
            if text and any(pattern.search(text) for pattern in kind_patterns):
                rows.append(_candidate(cue.start, cue.end, kind, "drop_review_required", f"profile={profile}; matched explicit {kind} language; semantic value still requires human review", [cue.segment_id or "transcript_segment"], excerpt=text))
                break
    return rows


def _repeat_candidates(cues: list[TranscriptCue], *, similarity: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recent: list[TranscriptCue] = []
    for cue in cues:
        current = _normalise_text(cue.text)
        if len(current) < 12:
            recent.append(cue)
            recent = recent[-12:]
            continue
        best: tuple[float, TranscriptCue] | None = None
        for previous in recent:
            prior = _normalise_text(previous.text)
            if len(prior) < 12:
                continue
            score = SequenceMatcher(None, prior, current, autojunk=False).ratio()
            if best is None or score > best[0]:
                best = (score, previous)
        if best and best[0] >= similarity:
            rows.append(_candidate(cue.start, cue.end, "possible_retake_or_restatement", "drop_review_required", f"near-duplicate transcript similarity={best[0]:.4f}; later take is never auto-selected", [cue.segment_id or "transcript_segment", best[1].segment_id or "earlier_transcript_segment"], excerpt=cue.text))
        recent.append(cue)
        recent = recent[-12:]
    return rows


def _candidate(start: float, end: float, kind: str, classification: str, reason: str, evidence_ids: list[str], *, excerpt: str = "") -> dict[str, Any]:
    return {"candidate_id": "", "start": round(max(0.0, start), 6), "end": round(max(start, end), 6), "kind": kind, "classification": classification, "recommended_action": "review_for_drop", "reason": reason, "evidence_ids": [value for value in evidence_ids if value], "transcript_excerpt": excerpt, "confidence": "high" if classification == "drop_safe_candidate" else "medium", "human_confirmation_required": True}


def _apply_activity_audit_evidence(
    candidates: list[dict[str, Any]],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    additions: list[dict[str, Any]] = []
    for position, gap in enumerate(payload.get("candidate_gaps") or [], start=1):
        if not isinstance(gap, dict):
            raise ValueError("ASR VAD activity audit candidate gap must be an object")
        start = float(gap.get("start") or 0.0)
        end = float(gap.get("end") or 0.0)
        if start < 0 or end <= start:
            raise ValueError("ASR VAD activity audit candidate gap has invalid bounds")
        candidate_id = str(gap.get("candidate_id") or f"audio-activity-gap-{position:04d}")
        evidence_id = f"activity-audit:{candidate_id}"
        overlapping = [
            row
            for row in candidates
            if min(float(row["end"]), end) > max(float(row["start"]), start)
        ]
        if overlapping:
            for row in overlapping:
                row["classification"] = "drop_review_required"
                row["confidence"] = "medium"
                row["activity_audit_status"] = "non_silent_audio_without_vad_coverage"
                row["evidence_ids"] = list(dict.fromkeys([*row.get("evidence_ids", []), evidence_id]))
                row["reason"] = (
                    str(row.get("reason") or "")
                    + "; non-silent audio exists without speech-VAD coverage; music/noise is possible"
                ).lstrip("; ")
            continue
        row = _candidate(
            start,
            end,
            "audio_activity_without_vad",
            "drop_review_required",
            "non-silent audio exists without speech-VAD coverage; music/noise is possible",
            [evidence_id],
        )
        row["activity_audit_status"] = "non_silent_audio_without_vad_coverage"
        additions.append(row)
    return additions


def _resolve_activity_audit(
    root: Path,
    manifest: dict[str, Any],
    *,
    explicit: str | Path | None,
    vad_path: Path | None,
) -> Path | None:
    discovered = _manifest_value(
        manifest,
        "asr_vad_activity_audit_json",
        "asr_vad_activity_audit",
    )
    if explicit not in (None, "") or discovered not in (None, ""):
        return _resolve_optional(root, explicit, discovered)
    if vad_path:
        adjacent = vad_path.with_name("asr-vad-activity-audit.json")
        if adjacent.is_file():
            return adjacent.resolve()
    return None


def _validated_activity_audit(
    path: Path,
    *,
    media_path: Path,
    vad_path: Path | None,
) -> dict[str, Any]:
    if vad_path is None:
        raise ValueError("ASR VAD activity audit requires VAD evidence")
    payload = _read_object(path, "ASR VAD activity audit")
    if payload.get("schema") != ACTIVITY_AUDIT_SCHEMA:
        raise ValueError("unsupported ASR VAD activity audit schema")
    if payload.get("status") not in {"passed", "review_required"}:
        raise ValueError("ASR VAD activity audit is not complete")
    source = payload.get("source_media")
    source = source if isinstance(source, dict) else {}
    if str(source.get("sha256") or "") != sha256_file(media_path):
        raise ValueError("ASR VAD activity audit source media does not match")
    if str(payload.get("vad_sha256") or "") != sha256_file(vad_path):
        raise ValueError("ASR VAD activity audit VAD evidence does not match")
    gaps = payload.get("candidate_gaps")
    if not isinstance(gaps, list):
        raise ValueError("ASR VAD activity audit candidate_gaps must be an array")
    return payload


def _deduplicate_candidates(rows: list[dict[str, Any]], *, duration: float) -> list[dict[str, Any]]:
    ordered = sorted((row for row in rows if row["end"] > row["start"]), key=lambda row: (row["start"], row["end"], row["kind"]))
    result: list[dict[str, Any]] = []
    for row in ordered:
        row["start"] = round(max(0.0, min(duration, row["start"])), 6)
        row["end"] = round(max(row["start"], min(duration, row["end"])), 6)
        if row["end"] <= row["start"]:
            continue
        if result and row["start"] < result[-1]["end"]:
            # Conflicting semantic reasons are not merged into a stronger claim.
            row["start"] = result[-1]["end"]
            if row["end"] <= row["start"]:
                continue
        result.append(row)
    return result


def _review_template(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "source_media_sha256": plan["source_media"]["sha256"],
        "decisions": [{"candidate_id": row["candidate_id"], "decision": "pending", "note": ""} for row in plan["segments"]],
        "operator_confirmation": {"confirmed": False, "confirmed_by": "", "confirmed_at": ""},
    }


def _render_plan_markdown(plan: dict[str, Any]) -> str:
    lines = ["# 超长视频快速分段审核计划", "", f"- 状态：`{plan['status']}`", f"- 原片：`{plan['source_media']['path']}`", f"- 时长：`{format_timestamp(plan['duration_seconds'])}`", f"- 候选：`{plan['summary']['candidate_count']}`", "- 说明：所有候选必须人工确认；证据不足默认保留；原片不会被修改。", "", "## 候选片段", ""]
    for row in plan["segments"]:
        lines.extend([f"- {row['candidate_id']} · {row['start_time']} - {row['end_time']}", f"  - 分类：`{row['classification']}` / `{row['kind']}`", f"  - 理由：{row['reason']}", f"  - 证据：`{', '.join(row['evidence_ids']) or 'missing'}`", f"  - 文本：{row['transcript_excerpt'] or '（无）'}"])
    return "\n".join(lines).rstrip() + "\n"


def _filter_complex(keep: list[tuple[float, float]]) -> str:
    parts: list[str] = []
    labels: list[str] = []
    for index, (start, end) in enumerate(keep):
        parts.append(f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS[v{index}]")
        parts.append(f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{index}]")
        labels.append(f"[v{index}][a{index}]")
    parts.append(f"{''.join(labels)}concat=n={len(keep)}:v=1:a=1[outv][outa]")
    return ";\n".join(parts) + "\n"


def _probe_duration(path: Path) -> float:
    ffprobe = resolve_media_tool("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required for render QA")
    completed = subprocess.run([str(ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False, env=local_tool_subprocess_env())
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe render QA failed: {(completed.stderr or completed.stdout)[-1000:]}")
    return float(completed.stdout.strip())


def _probe_streams(path: Path) -> dict[str, bool]:
    ffprobe = resolve_media_tool("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required for render stream preflight")
    completed = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        env=local_tool_subprocess_env(),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe stream preflight failed: {(completed.stderr or completed.stdout)[-1000:]}")
    payload = json.loads(completed.stdout or "{}")
    kinds = {str(row.get("codec_type") or "") for row in payload.get("streams") or [] if isinstance(row, dict)}
    return {"video": "video" in kinds, "audio": "audio" in kinds}


def _resolve_media(root: Path, manifest: dict[str, Any], explicit: str | Path | None) -> Path:
    values = [explicit, _manifest_value(manifest, "source_video", "video_path", "source_media", "media_path")]
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.resolve().is_file():
            return path.resolve()
    raise FileNotFoundError("source media not found; pass --media-path or register it in manifest.json")


def _resolve_transcript(root: Path, manifest: dict[str, Any], explicit: str | Path | None) -> Path:
    values = [explicit, _manifest_value(manifest, "human_corrected_transcript_json", "normalized_transcript", "full_transcript", "transcript_path"), "exports/human-corrected-transcript.json", "normalized-transcript.json", "exports/full-transcript.md"]
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.resolve().is_file():
            return path.resolve()
    raise FileNotFoundError("timestamped transcript not found; pass --transcript-path")


def _resolve_optional(root: Path, explicit: str | Path | None, discovered: Any) -> Path | None:
    for value in (explicit, discovered, "exports/silero-vad-candidate.json", "silero-vad-candidate.json"):
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.resolve().is_file():
            return path.resolve()
    return None


def _resolve_inside(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and not path.is_relative_to(root):
        raise ValueError(f"{label} must stay inside the bundle")
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _media_duration(cues: list[TranscriptCue], vad: dict[str, Any], timeline: list[dict[str, Any]], shots: list[dict[str, Any]]) -> float:
    values = [cue.end for cue in cues]
    values.extend(float(row.get("end") or row.get("end_seconds") or 0.0) for row in timeline)
    values.extend(float(row.get("end") or row.get("end_seconds") or 0.0) for row in shots)
    values.extend(end for _, end in _vad_intervals(vad))
    return max(values or [0.0])


def _vad_intervals(payload: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(row.get("start") or 0.0), float(row.get("end") or 0.0)) for row in payload.get("segments") or [] if isinstance(row, dict) and float(row.get("end") or 0.0) > float(row.get("start") or 0.0)]


def _visual_evidence(timeline: list[dict[str, Any]], shots: list[dict[str, Any]], start: float, end: float) -> list[str]:
    evidence: list[str] = []
    for row in timeline:
        row_start = float(row.get("start") or row.get("start_seconds") or 0.0)
        row_end = float(row.get("end") or row.get("end_seconds") or row_start)
        if row_end <= start or row_start >= end:
            continue
        if str(row.get("visual_text") or row.get("ocr_text") or "").strip() or row.get("temporal_visual_understanding"):
            evidence.append(f"timeline:{row.get('index', '')}")
    for row in shots:
        row_start = float(row.get("start") or row.get("start_seconds") or 0.0)
        row_end = float(row.get("end") or row.get("end_seconds") or row_start)
        if row_end > start and row_start < end:
            evidence.append(f"shot:{row.get('shot_id') or row.get('id') or ''}")
    return evidence


def _merge_ranges(rows: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[list[float]] = []
    for start, end in sorted((max(0.0, float(start)), max(0.0, float(end))) for start, end in rows if float(end) > float(start)):
        if result and start <= result[-1][1] + 0.001:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return [(round(start, 6), round(end, 6)) for start, end in result]


def _complement(rows: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    cursor = 0.0
    result: list[tuple[float, float]] = []
    for start, end in _merge_ranges(rows):
        start, end = max(0.0, min(duration, start)), max(0.0, min(duration, end))
        if start > cursor + 0.001:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration - 0.001:
        result.append((cursor, duration))
    return [(round(start, 6), round(end, 6)) for start, end in result if end - start > 0.001]


def _range_row(start: float, end: float) -> dict[str, Any]:
    return {"start": round(start, 6), "end": round(end, 6), "start_time": format_timestamp(start), "end_time": format_timestamp(end)}


def _artifact(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _verify_artifact(artifact: dict[str, Any], label: str) -> None:
    path = Path(str(artifact.get("path") or "")).resolve()
    if not path.is_file():
        raise ValueError(f"{label} is missing after planning")
    if path.stat().st_size != int(artifact.get("bytes") or -1) or sha256_file(path) != artifact.get("sha256"):
        raise ValueError(f"{label} changed after planning")


def _update_render_manifest(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = _read_object(manifest_path, "bundle manifest")
    manifest.update(
        {
            "long_video_fast_segment_render_receipt": RENDER_RECEIPT_PATH,
            "long_video_fast_segment_render_qa": QA_PATH,
        }
    )
    write_json(manifest_path, manifest)


def _profile(value: str, cues: list[TranscriptCue]) -> str:
    allowed = {"auto", "lecture", "interview", "meeting", "tutorial", "vlog"}
    if value not in allowed:
        raise ValueError(f"unsupported long-video profile: {value}")
    if value != "auto":
        return value
    speakers = {str(cue.speaker or "") for cue in cues if str(cue.speaker or "")}
    return "interview" if len(speakers) >= 2 else "lecture"


def _manifest_value(manifest: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if manifest.get(key):
            return manifest[key]
    return None


def _normalise_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.lower())


def _positive(value: float, label: str) -> float:
    if float(value) <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _bounded(value: float, label: str, minimum: float, maximum: float) -> float:
    value = float(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _payload_sha(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return canonical_json_sha256(payload)


def _plan_sha(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("plan_sha256", None)
    payload.pop("generated_at", None)
    snapshot = dict(payload.get("dependency_snapshot") or {})
    snapshot.pop("created_at", None)
    payload["dependency_snapshot"] = snapshot
    return canonical_json_sha256(payload)


def _approved_sha(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("approved_sha256", None)
    payload.pop("created_at", None)
    freshness = dict(payload.get("freshness") or {})
    freshness.pop("checked_at", None)
    payload["freshness"] = freshness
    return canonical_json_sha256(payload)


def _render_sha(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("render_sha256", None)
    payload.pop("created_at", None)
    return canonical_json_sha256(payload)


def _text_sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_schema(payload: dict[str, Any], filename: str) -> None:
    path = Path(__file__).with_name("schemas") / filename
    schema = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
