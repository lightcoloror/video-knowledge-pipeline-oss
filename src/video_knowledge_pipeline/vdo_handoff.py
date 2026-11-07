from __future__ import annotations

from .config import DEFAULT_LOCAL_FRAME_BUDGET, DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS, DEFAULT_LOCAL_FRAME_SAMPLING_MODE
import json
from pathlib import Path
from typing import Any

from .models import now_iso
from .page_metadata import import_page_metadata_payload
from .path_defaults import workspace_root

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv", ".ts"}
SIDECAR_KINDS = {
    ".json": "info_json",
    ".description": "description",
    ".vtt": "subtitle",
    ".srt": "subtitle",
    ".ass": "subtitle",
    ".jpg": "thumbnail",
    ".jpeg": "thumbnail",
    ".png": "thumbnail",
    ".webp": "thumbnail",
}


def vdo_handoff_plan(
    *,
    manifest_path: str | Path = "",
    summary_path: str | Path = "",
    review_checklist_path: str | Path = "",
    media_path: str | Path = "",
    host_root: str | Path = str(workspace_root()),
    container_root: str = "/mnt/used-by-codex",
    workspace: str | Path = "",
    title: str = "",
) -> dict[str, Any]:
    """Normalize VDO artifacts into a VKP-safe preview handoff.

    This function does not download, process, or inspect video content. It only
    reads local JSON artifacts and checks whether referenced files exist.
    """

    manifest = _read_json_object(manifest_path)
    summary = _read_json_object(summary_path)
    review = _read_json_object(review_checklist_path)
    resolved_media = _choose_media_path(media_path=media_path, summary=summary, review=review)
    source_url = _first_text(manifest.get("url"), summary.get("url"), manifest.get("source"))
    output_dir = _first_text(summary.get("output_dir"), manifest.get("output_dir"))
    effective_title = title or _title_from_artifacts(manifest, summary, resolved_media)
    review_block = _review_block(review, summary, resolved_media)
    can_ingest = _can_ingest(summary=summary, media_path=resolved_media, review_block=review_block)
    run_workspace = str(Path(workspace).expanduser().resolve()) if workspace else _default_workspace(resolved_media, effective_title)
    sidecars = _collect_sidecars(review=review, summary=summary, host_root=host_root, container_root=container_root)
    vdo_artifacts = _artifact_paths(
        manifest_path=manifest_path,
        summary_path=summary_path,
        review_checklist_path=review_checklist_path,
        summary=summary,
        review=review,
    )

    next_action = "run_openclaw_video_ingest" if can_ingest else "operator_review"
    if not resolved_media:
        next_action = "download_or_media_path_required"
    elif review_block["needs_human_review"]:
        next_action = "review_vdo_output_before_ingest"

    return {
        "schema": "video_knowledge_pipeline.vdo_handoff.v1",
        "created_at": now_iso(),
        "ok": can_ingest,
        "status": "ready_for_ingest" if can_ingest else "needs_review",
        "source_tool": "video-download-orchestrator",
        "task_id": _first_text(summary.get("task_id"), manifest.get("task_id")),
        "source_url": source_url,
        "canonical_url": source_url,
        "platform": _platform(manifest, summary, source_url),
        "title": effective_title,
        "selected_backend": _first_text(summary.get("selected_backend"), manifest.get("selected_backend")),
        "vdo_status": _first_text(summary.get("status"), review.get("status"), manifest.get("status"), default="unknown"),
        "media_path": str(Path(resolved_media).expanduser().resolve()) if resolved_media else "",
        "media_path_container": _to_container_path(resolved_media, host_root=host_root, container_root=container_root) if resolved_media else "",
        "output_dir": str(Path(output_dir).expanduser().resolve()) if output_dir else "",
        "sidecars": sidecars,
        "vdo_artifacts": vdo_artifacts,
        "review": review_block,
        "content_assets": {
            "review_required": True,
            "publication_allowed": False,
            "summary_path": "",
            "timeline_path": "",
            "audit_path": "",
            "key_segments_path": "",
            "short_video_script_drafts_path": "",
            "highlight_post_drafts_path": "",
            "material_card_contract": {
                "schema": "self_media_material_card.v1",
                "field_mapping": {
                    "material_id": "vdo_task_id_or_vkp_bundle_id",
                    "source_path": {
                        "vdo_manifest_path": str(Path(manifest_path).expanduser().resolve()) if manifest_path else "",
                        "vdo_summary_path": str(Path(summary_path).expanduser().resolve()) if summary_path else "",
                        "vdo_review_checklist_path": str(Path(review_checklist_path).expanduser().resolve()) if review_checklist_path else "",
                    },
                    "source_type": "video",
                    "source_fact_status": "download_artifact_needs_vkp_extraction",
                    "evidence_tier": "download_manifest_only",
                    "privacy_level": "unknown_review_required",
                    "desensitized": False,
                    "compliance_risk": "needs_review",
                    "fact_check_status": "needs_review",
                    "target_layer": ["content_asset_pool"],
                    "publish_surface": ["none_until_vkp_export"],
                    "content_stage": "candidate",
                    "cta_type": "",
                    "crm_followup_needed": False,
                    "owner_thread": "video-knowledge-pipeline",
                    "next_action": "ingest_with_vkp_after_vdo_review",
                    "blocked_reason": "no_structured_content_extracted_yet",
                },
                "allowed_as_inspiration": False,
                "allowed_as_fact": False,
            },
            "consumer_rules": {
                "circle_of_friends": {
                    "allowed_status": "not_allowed_until_vkp_export",
                    "draft_only": True,
                },
                "content_assets": {
                    "allowed_stage": "candidate",
                    "requires_source_path": True,
                    "requires_timestamp_or_evidence_path": False,
                },
            },
            "human_confirmation_required": [
                "download_or_account_authorized_access",
                "vdo_review_before_ingest",
                "cloud_asr_or_cloud_vision_execution",
                "publish_or_send_to_any_external_surface",
            ],
        },
        "ingestion": {
            "recommended_tool": "openclaw_video_ingest",
            "workspace": run_workspace,
            "title": effective_title,
            "execute_asr": False,
            "execute_vision": False,
            "max_frames": DEFAULT_LOCAL_FRAME_BUDGET,
            "sample_interval": DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
            "sample_mode": DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
            "next_action": next_action,
        },
        "safety": {
            "download_was_explicitly_confirmed": _download_was_confirmed(summary, manifest),
            "secrets_redacted": True,
            "no_cloud_execution_by_default": True,
            "content_assets_are_review_drafts": True,
            "operator_boundary": "VDO owns download. VKP owns ASR/OCR/frame understanding/knowledge exports.",
        },
        "next_actions": _next_actions(can_ingest=can_ingest, media_path=resolved_media, review_block=review_block),
    }


def ingest_vdo_handoff(
    *,
    handoff_path: str | Path = "",
    manifest_path: str | Path = "",
    summary_path: str | Path = "",
    review_checklist_path: str | Path = "",
    media_path: str | Path = "",
    host_root: str | Path = str(workspace_root()),
    container_root: str = "/mnt/used-by-codex",
    workspace: str | Path = "",
    title: str = "",
    execute: bool = False,
    max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
    sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
    sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
    prepare_runner: Any = None,
) -> dict[str, Any]:
    """Preview or execute VKP ingest from a reviewed VDO handoff."""

    handoff = _read_json_object(handoff_path) if handoff_path else {}
    if not handoff:
        handoff = vdo_handoff_plan(
            manifest_path=manifest_path,
            summary_path=summary_path,
            review_checklist_path=review_checklist_path,
            media_path=media_path,
            host_root=host_root,
            container_root=container_root,
            workspace=workspace,
            title=title,
        )
    media = str(media_path or handoff.get("media_path") or "").strip()
    ingestion = handoff.get("ingestion") if isinstance(handoff.get("ingestion"), dict) else {}
    run_workspace = str(workspace or ingestion.get("workspace") or "").strip()
    run_title = str(title or ingestion.get("title") or handoff.get("title") or "").strip()
    ready = bool(handoff.get("ok")) and str(handoff.get("status") or "") == "ready_for_ingest"
    if not ready:
        return {
            "schema": "video_knowledge_pipeline.vdo_handoff_ingest.v1",
            "created_at": now_iso(),
            "ok": False,
            "status": "operator_review_required",
            "execute": execute,
            "handoff": handoff,
            "media_path": media,
            "workspace": run_workspace,
            "next_actions": ["review_vdo_handoff", "fix_or_accept_vdo_download_before_ingest"],
            "operator_boundary": {
                "kind": "vdo_review_gate",
                "summary": "VDO handoff is not ready; VKP ingest is blocked.",
                "no_video_processing": True,
                "no_cloud_calls": True,
            },
        }
    if not execute:
        return {
            "schema": "video_knowledge_pipeline.vdo_handoff_ingest.v1",
            "created_at": now_iso(),
            "ok": True,
            "status": "preview_ready_for_ingest",
            "execute": False,
            "handoff": handoff,
            "media_path": media,
            "workspace": run_workspace,
            "next_actions": ["rerun_with_execute_to_prepare_vkp_bundle"],
            "operator_boundary": {
                "kind": "preview_only",
                "summary": "No video processing is performed until execute=true.",
                "no_video_processing": True,
                "no_cloud_calls": True,
            },
        }

    from .openclaw_integration import openclaw_video_ingest

    kwargs: dict[str, Any] = {}
    if prepare_runner is not None:
        kwargs["prepare_runner"] = prepare_runner
    ingest = openclaw_video_ingest(
        media,
        workspace=run_workspace,
        title=run_title,
        execute_asr=False,
        max_frames=max_frames,
        sample_interval=sample_interval,
        sample_mode=sample_mode,
        **kwargs,
    )
    result = {
        "schema": "video_knowledge_pipeline.vdo_handoff_ingest.v1",
        "created_at": now_iso(),
        "ok": bool(ingest.get("ok")),
        "status": "ingested" if ingest.get("ok") else "ingest_failed",
        "execute": True,
        "handoff": handoff,
        "media_path": media,
        "workspace": str(ingest.get("workspace") or run_workspace),
        "review_url_or_file": str(ingest.get("review_url_or_file") or ""),
        "ingest": ingest,
        "next_actions": ingest.get("next_actions") if isinstance(ingest.get("next_actions"), list) else ["inspect_vkp_ingest_result"],
        "operator_boundary": {
            "kind": "local_ingest_only",
            "summary": "VDO handoff was ready; VKP prepared a local review bundle without cloud vision execution.",
            "no_download": True,
            "cloud_vision_requires_separate_preflight": True,
        },
    }
    local_run = ingest.get("local_run") if isinstance(ingest.get("local_run"), dict) else {}
    initial_bundle = local_run.get("initial_bundle") if isinstance(local_run.get("initial_bundle"), dict) else {}
    bundle_dir = str(initial_bundle.get("bundle_dir") or "").strip()
    if bundle_dir and handoff:
        try:
            result["page_metadata_import"] = import_page_metadata_payload(
                bundle_dir,
                handoff,
                source_path=handoff_path or manifest_path or summary_path or None,
                write=True,
            )
        except (OSError, ValueError) as exc:
            result["page_metadata_import"] = {
                "ok": False,
                "status": "degraded_source_context_import_failed",
                "error": str(exc),
                "no_network_call": True,
            }
    else:
        result["page_metadata_import"] = {
            "ok": False,
            "status": "skipped_no_bundle_or_handoff",
            "no_network_call": True,
        }
    return result

def _read_json_object(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _choose_media_path(*, media_path: str | Path, summary: dict[str, Any], review: dict[str, Any]) -> str:
    explicit = str(media_path or "").strip()
    if explicit:
        return explicit
    candidates: list[str] = []
    for check in review.get("checks") if isinstance(review.get("checks"), list) else []:
        if not isinstance(check, dict):
            continue
        for file_row in check.get("files") if isinstance(check.get("files"), list) else []:
            if isinstance(file_row, dict):
                candidates.extend(_candidate_strings(file_row.get(key) for key in ("path", "file", "output_file")))
            elif isinstance(file_row, str):
                candidates.append(file_row)
    backend_result = summary.get("backend_result") if isinstance(summary.get("backend_result"), dict) else {}
    candidates.extend(_candidate_strings(backend_result.get(key) for key in ("output_file", "media_path", "path")))
    candidates.extend(_candidate_strings(summary.get(key) for key in ("media_path", "output_file")))
    ffprobe = summary.get("ffprobe") if isinstance(summary.get("ffprobe"), dict) else {}
    candidates.extend(_candidate_strings(ffprobe.get(key) for key in ("path", "file")))
    for value in candidates:
        if Path(value).suffix.lower() in MEDIA_EXTENSIONS:
            return value
    return ""


def _candidate_strings(values: Any) -> list[str]:
    result: list[str] = []
    if isinstance(values, str) or not hasattr(values, "__iter__"):
        iterable = [values]
    else:
        iterable = values
    for value in iterable:
        if isinstance(value, str) and value.strip():
            result.append(value.strip())
    return result


def _review_block(review: dict[str, Any], summary: dict[str, Any], media_path: str) -> dict[str, Any]:
    reasons: list[str] = []
    if not media_path:
        reasons.append("media_missing")
    elif not Path(media_path).expanduser().exists():
        reasons.append("media_path_not_found")
    if bool(review.get("needs_review")):
        reasons.append("vdo_needs_review")
    if bool(review.get("manual_review_required")):
        reasons.append("manual_review_required")
    if str(summary.get("status") or "").lower() not in {"", "finished", "ok", "success"}:
        reasons.append("vdo_status_not_finished")
    for check in review.get("checks") if isinstance(review.get("checks"), list) else []:
        if isinstance(check, dict) and str(check.get("status") or "").lower() in {"warning", "failed", "error"}:
            name = str(check.get("name") or "review_check")
            reasons.append(f"{name}_{check.get('status')}")
    return {
        "needs_human_review": bool(reasons),
        "manual_review_required": bool(review.get("manual_review_required")) or bool(reasons),
        "review_status": str(review.get("status") or "unknown"),
        "review_checklist_path": str(summary.get("review_checklist_path") or ""),
        "reasons": sorted(set(reasons)),
        "accepted_by_human": False,
    }


def _can_ingest(*, summary: dict[str, Any], media_path: str, review_block: dict[str, Any]) -> bool:
    if not media_path or not Path(media_path).expanduser().exists():
        return False
    if str(summary.get("status") or "").lower() not in {"finished", "ok", "success"}:
        return False
    return not bool(review_block.get("needs_human_review"))


def _collect_sidecars(*, review: dict[str, Any], summary: dict[str, Any], host_root: str | Path, container_root: str) -> list[dict[str, Any]]:
    candidates: list[str] = []
    for archive in summary.get("archive_files") if isinstance(summary.get("archive_files"), list) else []:
        if isinstance(archive, dict):
            candidates.extend(_candidate_strings(archive.get(key) for key in ("path", "file")))
        elif isinstance(archive, str):
            candidates.append(archive)
    for check in review.get("checks") if isinstance(review.get("checks"), list) else []:
        if not isinstance(check, dict):
            continue
        for file_row in check.get("files") if isinstance(check.get("files"), list) else []:
            if isinstance(file_row, dict):
                candidates.extend(_candidate_strings(file_row.get(key) for key in ("path", "file")))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        suffix = path.suffix.lower()
        if suffix in MEDIA_EXTENSIONS:
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.append(
            {
                "kind": SIDECAR_KINDS.get(suffix, "other"),
                "path": resolved,
                "path_container": _to_container_path(resolved, host_root=host_root, container_root=container_root),
                "language": "",
                "source": "video-download-orchestrator",
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def _artifact_paths(
    *,
    manifest_path: str | Path,
    summary_path: str | Path,
    review_checklist_path: str | Path,
    summary: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, str]:
    return {
        "manifest_path": _resolve_if_present(manifest_path),
        "report_path": _resolve_if_present(summary.get("report_path")),
        "summary_path": _resolve_if_present(summary_path),
        "review_checklist_path": _resolve_if_present(review_checklist_path or summary.get("review_checklist_path") or review.get("review_checklist_path")),
        "log_path": _resolve_if_present(summary.get("log_path")),
    }


def _resolve_if_present(path: Any) -> str:
    text = str(path or "").strip()
    return str(Path(text).expanduser().resolve()) if text else ""


def _to_container_path(path: str | Path, *, host_root: str | Path, container_root: str) -> str:
    if not path:
        return ""
    raw = str(Path(path).expanduser().resolve())
    host = str(Path(host_root).expanduser().resolve()).rstrip("\\/")
    container = str(container_root or "/mnt/used-by-codex").rstrip("/")
    lowered_raw = raw.lower().replace("/", "\\")
    lowered_host = host.lower().replace("/", "\\")
    if lowered_raw == lowered_host:
        return container
    prefix = lowered_host + "\\"
    if lowered_raw.startswith(prefix):
        relative = raw[len(host) :].lstrip("\\/").replace("\\", "/")
        return f"{container}/{relative}"
    return ""


def _first_text(*values: Any, default: str = "") -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _title_from_artifacts(manifest: dict[str, Any], summary: dict[str, Any], media_path: str) -> str:
    title = _first_text(manifest.get("title"), summary.get("title"))
    if title:
        return title
    if media_path:
        return Path(media_path).stem
    return "video"


def _platform(manifest: dict[str, Any], summary: dict[str, Any], source_url: str) -> str:
    route = manifest.get("route") if isinstance(manifest.get("route"), dict) else {}
    platform = _first_text(manifest.get("platform"), summary.get("platform"), route.get("platform"), route.get("kind"))
    if platform:
        return platform
    host = source_url.lower()
    if "bilibili" in host:
        return "bilibili"
    if "youtube" in host or "youtu.be" in host:
        return "youtube"
    if "feishu" in host or "larksuite" in host:
        return "feishu"
    return "unknown"


def _default_workspace(media_path: str, title: str) -> str:
    root = Path(__file__).resolve().parents[2] / "openclaw-runs"
    stem = Path(media_path).stem if media_path else title
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in stem).strip("-") or "video"
    return str((root / safe).resolve())


def _download_was_confirmed(summary: dict[str, Any], manifest: dict[str, Any]) -> bool:
    if str(summary.get("status") or "").lower() in {"finished", "ok", "success"}:
        return True
    command = summary.get("command")
    return bool(command) or str(manifest.get("status") or "").lower() in {"downloaded", "finished"}


def _next_actions(*, can_ingest: bool, media_path: str, review_block: dict[str, Any]) -> list[str]:
    if can_ingest:
        return ["run_openclaw_video_ingest", "keep_asr_and_vision_execution_explicit"]
    if not media_path:
        return ["inspect_vdo_download_result", "provide_existing_media_path_or_confirm_download"]
    if bool(review_block.get("needs_human_review")):
        return ["review_vdo_checklist", "accept_or_fix_download_before_vkp_ingest"]
    return ["inspect_vdo_handoff"]
