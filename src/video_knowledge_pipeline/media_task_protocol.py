from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from typing import Any

from .artifact_validation import artifact_evidence, normalise_allowed_roots, validated_local_file
from .media_capability_registry import PROTOCOL, media_capability


RESULT_SCHEMA = "video_knowledge_pipeline.media_task_result.v1"
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "timeout"})
FORBIDDEN_PARAMETER_FRAGMENTS = (
    "url",
    "uri",
    "endpoint",
    "api_key",
    "authorization",
    "secret",
    "token",
    "callback",
)


def build_media_task_plan(
    task: str,
    *,
    execution_location: str,
    route_id: str = "",
    route_revision: str = "",
    deployment: str = "volcengine_mediakit",
    artifact_paths: list[str | Path] | tuple[str | Path, ...],
    artifact_hashes: list[str] | tuple[str, ...] | None = None,
    consent_id: str = "",
    parameters: dict[str, Any] | None = None,
    allowed_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict[str, Any]:
    capability = media_capability(task)
    location = _location(execution_location)
    roots = normalise_allowed_roots(allowed_roots)
    paths = [
        validated_local_file(value, label="artifact_path", allowed_roots=roots)
        for value in artifact_paths
    ]
    minimum = int(capability["min_artifacts"])
    maximum = int(capability["max_artifacts"])
    if len(paths) < minimum or len(paths) > maximum:
        raise ValueError(
            f"{capability['task']} requires between {minimum} and {maximum} artifacts"
        )
    artifacts = [_media_artifact(path) for path in paths]
    declared = list(artifact_hashes or [])
    if declared and len(declared) != len(artifacts):
        raise ValueError("artifact_hashes must match artifact_paths one-for-one")
    for index, row in enumerate(artifacts):
        if declared and _normalise_sha256(declared[index]) != row["sha256"]:
            raise ValueError(f"artifact hash mismatch: {row['path']}")
    normalised_parameters = _normalise_parameters(capability, parameters or {})
    route = str(route_id or "").strip()
    revision = str(route_revision or "").strip()
    consent = str(consent_id or "").strip()
    if location == "remote" and not route:
        raise ValueError("remote media task requires route_id")
    if location == "remote" and not revision:
        raise ValueError("remote media task requires route_revision")
    if str(deployment or "").strip() != "volcengine_mediakit":
        raise ValueError("media deployment must be the fixed volcengine_mediakit adapter")
    return {
        "schema_version": PROTOCOL,
        "task": capability["task"],
        "provider_task": capability["provider_task"],
        "execution_location": location,
        "route_id": route,
        "route_revision": revision,
        "deployment": "volcengine_mediakit",
        "artifacts": artifacts,
        "artifact_paths": [row["path"] for row in artifacts],
        "artifact_hashes": [row["sha256"] for row in artifacts],
        "consent_id": consent,
        "parameters": normalised_parameters,
        "submit": dict(capability["submit"]),
        "poll": dict(capability["poll"]),
        "candidate_only": True,
        "authorization_status": (
            "consent_bound_remote_execution"
            if location == "remote"
            else "unsupported_local_location"
        ),
        "upload_required": location == "remote",
        "provider_artifact_refs_present": False,
        "created_at_unix": time.time(),
        "operator_boundary": {
            "contains_local_paths": True,
            "contains_provider_urls": False,
            "contains_api_key": False,
            "network_calls_made": 0,
            "automatic_upload_allowed": "only_after_explicit_consent",
            "silent_fallback_allowed": False,
        },
    }


def normalise_provider_status(value: Any, *, has_task_id: bool = False) -> str:
    status = str(value or "").strip().lower()
    if status in {"queued", "pending", "submitted"}:
        return "submitted"
    if status in {"running", "processing", "in_progress"}:
        return "running"
    if status in {"completed", "succeeded", "success"}:
        return "succeeded"
    if status in {"failed", "error"}:
        return "failed"
    if status in {"canceled", "cancelled"}:
        return "cancelled"
    if status in {"timeout", "timed_out"}:
        return "timeout"
    return "submitted" if has_task_id else "failed"


def media_status_is_terminal(status: str) -> bool:
    return str(status or "").strip().lower() in TERMINAL_STATUSES


def normalise_media_task_result(
    plan: dict[str, Any],
    provider_payload: dict[str, Any] | None,
    *,
    forced_status: str = "",
    error: dict[str, Any] | None = None,
    latency_ms: int = 0,
    request_count: int = 0,
    transport: str = "fake_loopback",
) -> dict[str, Any]:
    payload = dict(provider_payload or {})
    task_id = str(payload.get("task_id") or "").strip()
    provider_status = str(payload.get("status") or "").strip()
    status = str(forced_status or "").strip() or normalise_provider_status(
        provider_status,
        has_task_id=bool(task_id),
    )
    if status not in TERMINAL_STATUSES | {"submitted", "running"}:
        raise ValueError(f"unsupported normalised media status: {status}")
    content_source = payload.get("result")
    if not isinstance(content_source, (dict, list)):
        content_source = {}
    content = sanitise_provider_content(content_source)
    evidence: list[dict[str, Any]] = []
    for row in plan.get("artifacts") or []:
        if isinstance(row, dict):
            evidence.append(
                {
                    "kind": "source_artifact",
                    "path": str(row.get("path") or ""),
                    "sha256": str(row.get("sha256") or ""),
                    "bytes": int(row.get("bytes") or 0),
                    "candidate_only": True,
                }
            )
    if content:
        evidence.append(
            {
                "kind": "provider_candidate",
                "task": str(plan.get("task") or ""),
                "candidate_only": True,
                "partial": status != "succeeded",
                "content": content,
            }
        )
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": PROTOCOL,
        "ok": status in {"submitted", "running", "succeeded"},
        "terminal": media_status_is_terminal(status),
        "task": str(plan.get("task") or ""),
        "status": status,
        "provider_status": provider_status,
        "task_id": task_id,
        "request_id": str(payload.get("request_id") or "").strip(),
        "execution_location": str(plan.get("execution_location") or ""),
        "route_id": str(plan.get("route_id") or ""),
        "route_revision": str(plan.get("route_revision") or ""),
        "deployment": str(plan.get("deployment") or ""),
        "provider": "volcengine_mediakit",
        "latency_ms": max(0, int(latency_ms)),
        "usage": sanitise_provider_content(payload.get("usage")),
        "estimated_cost": None,
        "content": content,
        "evidence": evidence,
        "consent_id": str(plan.get("consent_id") or ""),
        "error": sanitise_provider_content(error) if error else None,
        "candidate_only": True,
        "network_audit": {
            "transport": transport,
            "requests_made": max(0, int(request_count)),
            "remote_requests_made": transport != "fake_loopback",
            "fallback_attempted": False,
        },
        "writeback": {
            "timeline_updated": False,
            "bundle_updated": False,
            "smart_summary_updated": False,
        },
    }


def sanitise_provider_content(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if any(fragment in lowered for fragment in FORBIDDEN_PARAMETER_FRAGMENTS):
                continue
            clean[str(key)] = sanitise_provider_content(item)
        return clean
    if isinstance(value, list):
        return [sanitise_provider_content(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _location(value: str) -> str:
    location = str(value or "").strip().lower()
    if location not in {"local", "remote"}:
        raise ValueError("execution_location must be local or remote")
    return location


def _media_artifact(path: Path) -> dict[str, Any]:
    row = artifact_evidence(path)
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    row["mime_type"] = mime
    row["data_type"] = "audio" if mime.startswith("audio/") else "video"
    if row["data_type"] == "video" and not mime.startswith("video/"):
        raise ValueError(f"media artifact must be an audio or video file: {path}")
    return row


def _normalise_sha256(value: str) -> str:
    digest = str(value or "").strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[7:]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("artifact hash must be a SHA-256 hex digest")
    return digest


def _normalise_parameters(capability: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError("parameters must be an object")
    specs = {str(row["name"]): row for row in capability.get("parameters") or []}
    unknown = sorted(set(values) - set(specs))
    if unknown:
        raise ValueError(f"unsupported media parameters: {', '.join(unknown)}")
    for key in values:
        lowered = key.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_PARAMETER_FRAGMENTS):
            raise ValueError(f"provider URL, key, callback, and token parameters are forbidden: {key}")
    result: dict[str, Any] = {}
    for name, spec in specs.items():
        if name not in values:
            if spec.get("required"):
                raise ValueError(f"required media parameter missing: {name}")
            if "default" in spec:
                result[name] = spec["default"]
            continue
        value = values[name]
        _validate_parameter_value(name, value, spec)
        result[name] = value
    if capability["task"] == "scene_segmentation":
        minimum = result.get("min_duration")
        maximum = result.get("max_duration")
        if minimum is not None and maximum is not None and float(minimum) > float(maximum):
            raise ValueError("min_duration must be less than or equal to max_duration")
    if capability["task"] == "highlight_detection":
        expected_mode = "StorylineCuts" if result.get("model") == "Miniseries" else "HighlightExtract"
        if result.get("mode") != expected_mode:
            raise ValueError(f"highlight mode must be {expected_mode} for model {result.get('model')}")
    return result


def _validate_parameter_value(name: str, value: Any, spec: dict[str, Any]) -> None:
    value_type = str(spec.get("type") or "")
    if value_type == "boolean" and not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    if value_type == "string" and not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if value_type == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(f"{name} must be a number")
    if value_type == "object" and not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if spec.get("enum") and value not in spec["enum"]:
        raise ValueError(f"{name} must be one of: {', '.join(spec['enum'])}")
    if "minimum" in spec and float(value) < float(spec["minimum"]):
        raise ValueError(f"{name} is below its minimum")
    if "exclusive_minimum" in spec and float(value) <= float(spec["exclusive_minimum"]):
        raise ValueError(f"{name} must be greater than {spec['exclusive_minimum']}")
    if "exclusive_maximum" in spec and float(value) >= float(spec["exclusive_maximum"]):
        raise ValueError(f"{name} must be less than {spec['exclusive_maximum']}")
    if "max_bytes" in spec:
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        if len(encoded) > int(spec["max_bytes"]):
            raise ValueError(f"{name} exceeds the allowed size")
