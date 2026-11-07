from __future__ import annotations

import json
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file as _sha256
from .storage import read_json, write_json
from .vision_execution_route import resolve_vision_task_execution_route
from .time_utils import parse_utc_datetime_or_none as _parse_datetime, utc_now_iso_seconds
from .vision_api import provider_runtime_diagnostics, resolve_provider_config

SCHEMA = "video_knowledge_pipeline.vision_export_consent.v1"
DEFAULT_FILENAME = "vision-export-consent.json"


def create_vision_export_consent(
    bundle_dir: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    semantic_indexes: list[int] | None = None,
    temporal_indexes: list[int] | None = None,
    max_calls: int | None = None,
    expires_hours: float = 24.0,
    image_max_edge: int = 512,
    image_jpeg_quality: int = 55,
    purpose: str = "targeted multimodal review",
    confirm_data_export: bool = False,
    output_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = _bundle_root(bundle_dir)
    semantic = _indexes(semantic_indexes)
    temporal = _indexes(temporal_indexes)
    requested_calls = len(semantic) + len(temporal)
    call_limit = int(max_calls) if max_calls is not None else requested_calls
    if requested_calls <= 0:
        raise ValueError("At least one semantic or temporal index is required")
    if call_limit < requested_calls:
        raise ValueError("max_calls cannot be lower than the number of authorised indexes")
    if expires_hours <= 0:
        raise ValueError("expires_hours must be positive")
    if image_max_edge <= 0:
        raise ValueError("image_max_edge must be positive so an agent cannot export original-resolution frames")
    if not 1 <= int(image_jpeg_quality) <= 100:
        raise ValueError("image_jpeg_quality must be between 1 and 100")

    cfg = resolve_provider_config(_resolve_consent_provider_config(provider_config, semantic_indexes=semantic, temporal_indexes=temporal))
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    consent_path = Path(output_path).expanduser().resolve() if output_path else root / DEFAULT_FILENAME
    confirmed = bool(confirm_data_export)
    payload = {
        "schema": SCHEMA,
        "consent_id": str(uuid4()),
        "status": "active" if confirmed else "confirmation_required",
        "user_confirmed_data_export": confirmed,
        "bundle": {
            "bundle_dir": str(root),
            "timeline_path": str(root / "timeline.json"),
            "timeline_sha256": _sha256(root / "timeline.json"),
            "export_evidence_sha256": _export_evidence_sha256(
                root,
                semantic_indexes=semantic,
                temporal_indexes=temporal,
            ),
        },
        "provider": _provider_identity(cfg),
        "scope": {
            "semantic_indexes": semantic,
            "temporal_indexes": temporal,
            "max_calls": call_limit,
            "image_max_edge": int(image_max_edge),
            "image_jpeg_quality": int(image_jpeg_quality),
            "allowed_data_types": [
                "selected_video_frames",
                "selected_transcript_context",
                "selected_ocr_context",
            ],
            "prohibited_data_types": ["full_video", "audio", "cookies", "credentials", "api_keys"],
            "purpose": str(purpose or "targeted multimodal review"),
        },
        "created_at": created_at.isoformat(),
        "expires_at": (created_at + timedelta(hours=float(expires_hours))).isoformat(),
        "operator_boundary": {
            "authorises_project_execution_gate": confirmed,
            "does_not_override_agent_platform_policy": True,
            "api_key_persisted": False,
            "selected_frames_only": True,
            "bulk_video_export_allowed": False,
        },
        "artifacts": {
            "consent_json": str(consent_path),
            "consent_markdown": str(consent_path.with_suffix(".md")),
        },
    }
    payload["consent_sha256"] = _payload_sha256(payload)
    if write:
        write_json(consent_path, payload)
        consent_path.with_suffix(".md").write_text(render_vision_export_consent_markdown(payload), encoding="utf-8")
    return payload


def vision_export_consent_status(
    bundle_dir: str | Path,
    *,
    consent_path: str | Path | None = None,
    provider_config: dict[str, Any] | None = None,
    semantic_indexes: list[int] | None = None,
    temporal_indexes: list[int] | None = None,
    expected_calls: int | None = None,
    image_max_edge: int = 512,
    image_jpeg_quality: int = 55,
) -> dict[str, Any]:
    root = _bundle_root(bundle_dir)
    path = Path(consent_path).expanduser().resolve() if consent_path else root / DEFAULT_FILENAME
    cfg = resolve_provider_config(_resolve_consent_provider_config(provider_config, semantic_indexes=_indexes(semantic_indexes), temporal_indexes=_indexes(temporal_indexes)))
    return validate_vision_export_consent(
        path,
        bundle_dir=root,
        provider_config=cfg,
        semantic_indexes=semantic_indexes,
        temporal_indexes=temporal_indexes,
        expected_calls=expected_calls,
        image_max_edge=image_max_edge,
        image_jpeg_quality=image_jpeg_quality,
    )


def validate_vision_export_consent(
    consent_path: str | Path,
    *,
    bundle_dir: str | Path,
    provider_config: dict[str, Any],
    semantic_indexes: list[int] | None = None,
    temporal_indexes: list[int] | None = None,
    expected_calls: int | None = None,
    image_max_edge: int = 512,
    image_jpeg_quality: int = 55,
) -> dict[str, Any]:
    root = _bundle_root(bundle_dir)
    path = Path(consent_path).expanduser().resolve()
    blockers: list[dict[str, str]] = []
    if not path.exists():
        return _status_result(path, "missing", [{"key": "consent_missing", "message": f"Consent file not found: {path}"}])
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _status_result(path, "invalid", [{"key": "consent_unreadable", "message": str(exc)}])
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return _status_result(path, "invalid", [{"key": "consent_schema_invalid", "message": "Unsupported vision export consent schema"}])

    if payload.get("status") != "active" or not payload.get("user_confirmed_data_export"):
        blockers.append({"key": "consent_not_active", "message": "The consent was not explicitly confirmed or is no longer active"})
    if str(payload.get("consent_sha256") or "") != _payload_sha256(payload):
        blockers.append({"key": "consent_integrity_failed", "message": "The consent payload changed after it was created"})

    bundle = payload.get("bundle") if isinstance(payload.get("bundle"), dict) else {}
    if _norm_path(bundle.get("bundle_dir")) != _norm_path(root):
        blockers.append({"key": "consent_bundle_mismatch", "message": "The consent belongs to a different bundle"})

    expected_provider = _provider_identity(resolve_provider_config(provider_config))
    authorised_provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    for key in ("provider", "model", "endpoint", "route_id", "route_revision", "virtual_model", "profile_id"):
        if str(authorised_provider.get(key) or "") != str(expected_provider.get(key) or ""):
            blockers.append({"key": f"consent_provider_{key}_mismatch", "message": f"Provider {key} differs from the consent"})

    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    authorised_semantic = _indexes(scope.get("semantic_indexes"))
    authorised_temporal = _indexes(scope.get("temporal_indexes"))
    expected_evidence_hash = str(bundle.get("export_evidence_sha256") or "")
    if expected_evidence_hash:
        current_evidence_hash = _export_evidence_sha256(
            root,
            semantic_indexes=authorised_semantic,
            temporal_indexes=authorised_temporal,
        )
        if expected_evidence_hash != current_evidence_hash:
            blockers.append(
                {
                    "key": "consent_export_evidence_changed",
                    "message": "Authorised frames or their transcript/OCR context changed after consent creation; create a new consent",
                }
            )
    elif str(bundle.get("timeline_sha256") or "") != _sha256(root / "timeline.json"):
        blockers.append(
            {
                "key": "consent_timeline_changed",
                "message": "Legacy consent no longer matches timeline.json; create a new consent",
            }
        )

    semantic = _indexes(semantic_indexes)
    temporal = _indexes(temporal_indexes)
    if not set(semantic).issubset(set(authorised_semantic)):
        blockers.append({"key": "consent_semantic_indexes_exceeded", "message": "Requested semantic indexes are outside the consent"})
    if not set(temporal).issubset(set(authorised_temporal)):
        blockers.append({"key": "consent_temporal_indexes_exceeded", "message": "Requested temporal indexes are outside the consent"})
    calls = int(expected_calls) if expected_calls is not None else len(semantic) + len(temporal)
    if calls > int(scope.get("max_calls") or 0):
        blockers.append({"key": "consent_call_limit_exceeded", "message": "Requested API calls exceed the consent limit"})

    max_edge = int(image_max_edge or 0)
    if max_edge <= 0 or max_edge > int(scope.get("image_max_edge") or 0):
        blockers.append({"key": "consent_image_resolution_exceeded", "message": "Agent export must use a positive max edge within the consent"})
    if int(image_jpeg_quality or 0) > int(scope.get("image_jpeg_quality") or 0):
        blockers.append({"key": "consent_image_quality_exceeded", "message": "JPEG quality is higher than the consent allows"})

    expires_at = _parse_datetime(payload.get("expires_at"))
    if expires_at is None or expires_at <= datetime.now(timezone.utc):
        blockers.append({"key": "consent_expired", "message": "The vision export consent expired"})

    status = "active" if not blockers else "blocked"
    result = _status_result(path, status, blockers)
    result.update(
        {
            "consent_id": str(payload.get("consent_id") or ""),
            "expires_at": str(payload.get("expires_at") or ""),
            "provider": authorised_provider,
            "scope": scope,
            "platform_policy_may_still_block": True,
        }
    )
    return result


def revoke_vision_export_consent(
    bundle_dir: str | Path,
    *,
    consent_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = _bundle_root(bundle_dir)
    path = Path(consent_path).expanduser().resolve() if consent_path else root / DEFAULT_FILENAME
    if not path.exists():
        return _status_result(path, "missing", [{"key": "consent_missing", "message": f"Consent file not found: {path}"}])
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return _status_result(path, "invalid", [{"key": "consent_schema_invalid", "message": "Unsupported vision export consent schema"}])
    payload["status"] = "revoked"
    payload["revoked_at"] = utc_now_iso_seconds()
    payload["consent_sha256"] = _payload_sha256(payload)
    if write:
        write_json(path, payload)
        path.with_suffix(".md").write_text(render_vision_export_consent_markdown(payload), encoding="utf-8")
    return {"schema": SCHEMA, "status": "revoked", "consent_path": str(path), "consent_id": payload.get("consent_id")}


def render_vision_export_consent_markdown(payload: dict[str, Any]) -> str:
    bundle = payload.get("bundle") if isinstance(payload.get("bundle"), dict) else {}
    provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    lines = [
        "# Vision Export Consent",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Consent ID: `{payload.get('consent_id', '')}`",
        f"- Bundle: `{bundle.get('bundle_dir', '')}`",
        f"- Timeline SHA-256: `{bundle.get('timeline_sha256', '')}`",
        f"- Export evidence SHA-256: `{bundle.get('export_evidence_sha256', '')}`",
        f"- Provider: `{provider.get('provider', '')}` / `{provider.get('model', '')}`",
        f"- Endpoint: `{provider.get('endpoint', '')}`",
        f"- Semantic indexes: `{scope.get('semantic_indexes', [])}`",
        f"- Temporal indexes: `{scope.get('temporal_indexes', [])}`",
        f"- Maximum calls: `{scope.get('max_calls', 0)}`",
        f"- Image limit: `{scope.get('image_max_edge', 0)}px`, JPEG quality `{scope.get('image_jpeg_quality', 0)}`",
        f"- Expires at: `{payload.get('expires_at', '')}`",
        "",
        "This authorises only the project execution gate. It cannot override an agent platform's external-data policy. API keys, audio, full video, cookies, and credentials are not included or authorised.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _bundle_root(bundle_dir: str | Path) -> Path:
    root = Path(bundle_dir).expanduser().resolve()
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    if not (root / "timeline.json").exists():
        raise FileNotFoundError(f"bundle missing timeline.json: {root}")
    return root


def _provider_identity(provider_config: dict[str, Any]) -> dict[str, str]:
    cfg = resolve_provider_config(provider_config)
    diagnostics = provider_runtime_diagnostics(cfg)
    endpoint = _normalise_endpoint(str(diagnostics.get("base_url") or cfg.get("base_url") or ""))
    return {
        "provider": str(cfg.get("provider") or ""),
        "model": str(cfg.get("model") or ""),
        "endpoint": endpoint,
        "route_id": str(cfg.get("route_id") or ""),
        "route_revision": str(cfg.get("route_revision") or ""),
        "virtual_model": str(cfg.get("virtual_model") or ""),
        "profile_id": str(cfg.get("profile_id") or ""),
    }


def vision_export_consent_image_limits(consent_path: str | Path) -> dict[str, int]:
    """Read only the approved image transformation ceiling from a consent."""
    path = Path(consent_path).expanduser().resolve()
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("vision export consent schema is invalid")
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    max_edge = int(scope.get("image_max_edge") or 0)
    jpeg_quality = int(scope.get("image_jpeg_quality") or 0)
    if max_edge <= 0 or not 1 <= jpeg_quality <= 100:
        raise ValueError("vision export consent has invalid image limits")
    return {"image_max_edge": max_edge, "image_jpeg_quality": jpeg_quality}


def _resolve_consent_provider_config(
    provider_config: dict[str, Any] | None,
    *,
    semantic_indexes: list[int],
    temporal_indexes: list[int],
) -> dict[str, Any]:
    if provider_config is not None:
        return dict(provider_config)
    if semantic_indexes and temporal_indexes:
        raise ValueError("Mixed semantic and temporal vision consent requires an explicit provider configuration")
    task = "temporal_sequence" if temporal_indexes else "semantic_frame"
    resolution = resolve_vision_task_execution_route(task)
    if resolution.get("legacy_fallback_blocked"):
        raise ValueError(f"No configured route-based {task} provider; legacy fallback is blocked")
    return dict(resolution.get("provider_config") or {})


def _normalise_endpoint(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value.rstrip("/")
    return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


def _indexes(values: Any) -> list[int]:
    if not isinstance(values, (list, tuple, set)):
        return []
    indexes: set[int] = set()
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index > 0:
            indexes.add(index)
    return sorted(indexes)


def _export_evidence_sha256(
    root: Path,
    *,
    semantic_indexes: list[int],
    temporal_indexes: list[int],
) -> str:
    timeline = read_json(root / "timeline.json")
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must contain a list")
    semantic = set(_indexes(semantic_indexes))
    temporal = set(_indexes(temporal_indexes))
    selected = semantic | temporal
    projection: list[dict[str, Any]] = []
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        index = int(item.get("index") or position)
        if index not in selected:
            continue
        frame_paths = _evidence_paths(item, temporal=index in temporal)
        projection.append(
            {
                "index": index,
                "start": item.get("start"),
                "end": item.get("end"),
                "visual_route": item.get("visual_route"),
                "transcript": item.get("transcript"),
                "visual_text": item.get("visual_text"),
                "frame_paths": frame_paths,
                "frame_sha256": [_optional_sha256(_resolve_evidence_path(root, value)) for value in frame_paths],
            }
        )
    payload = {
        "semantic_indexes": sorted(semantic),
        "temporal_indexes": sorted(temporal),
        "items": sorted(projection, key=lambda row: int(row["index"])),
    }
    return canonical_json_sha256(payload)



def _evidence_paths(item: dict[str, Any], *, temporal: bool) -> list[str]:
    key = "temporal_frame_paths" if temporal else "frame_paths"
    values = item.get(key) if isinstance(item.get(key), list) else []
    if temporal and not values:
        values = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
    return [str(value).strip() for value in values if str(value).strip()]


def _resolve_evidence_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _optional_sha256(path: Path) -> str:
    return _sha256(path) if path.is_file() else "missing"


def _payload_sha256(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("consent_sha256", None)
    return canonical_json_sha256(value)



def _norm_path(value: Any) -> str:
    return str(Path(str(value or "")).expanduser().resolve()).casefold()


def _status_result(path: Path, status: str, blockers: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.vision_export_consent_status.v1",
        "status": status,
        "valid": status == "active",
        "consent_path": str(path),
        "blockers": blockers,
        "next_actions": _next_actions(status, path),
        "operator_boundary": {
            "project_gate_only": True,
            "does_not_override_agent_platform_policy": True,
            "visible_powershell_fallback_allowed": True,
            "local_vlm_fallback_allowed": True,
        },
    }


def _next_actions(status: str, path: Path) -> list[str]:
    if status == "active":
        return ["Run vision-execution-preflight with --execution-actor agent and --export-consent, then use its exact confirmation values."]
    if status == "missing":
        return ["Create a scoped consent with vision-export-consent-create before agent execution."]
    return [f"Inspect or recreate the scoped consent: {path}"]
