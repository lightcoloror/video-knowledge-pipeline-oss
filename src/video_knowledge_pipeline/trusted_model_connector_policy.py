from __future__ import annotations

import json
import os
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_connector_consent import (
    SCHEMA_V2,
    UPLOAD_MANIFEST_SCHEMA,
    _upload_manifest_sha256,
)
from .model_task_gateway import model_task_api_call
from .provider_config_safety import secretless_provider_config


ALLOWED_ROOTS_ENV = "VKP_MODEL_CONNECTOR_ALLOWED_ROOTS"
ALLOWED_DESTINATIONS_ENV = "VKP_MODEL_CONNECTOR_ALLOWED_DESTINATIONS"
LOCAL_DESTINATIONS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class TrustedModelConnectorPolicy:
    """Runtime policy enforced before a remote MCP tool can read or export data."""

    allowed_roots: tuple[Path, ...]
    allowed_destinations: frozenset[str]

    @classmethod
    def from_environment(
        cls, *, default_root: str | Path
    ) -> "TrustedModelConnectorPolicy":
        roots_value = str(os.environ.get(ALLOWED_ROOTS_ENV) or "").strip()
        roots = _split_roots(roots_value) if roots_value else _default_allowed_roots(default_root)
        destinations_value = str(os.environ.get(ALLOWED_DESTINATIONS_ENV) or "").strip()
        destinations = {
            destination
            for value in re.split(r"[,\n]+", destinations_value)
            if (destination := _normalise_destination(value))
        }
        return cls(
            allowed_roots=tuple(Path(value).expanduser().resolve() for value in roots),
            allowed_destinations=frozenset(destinations),
        )

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "allowed_roots": [str(path) for path in self.allowed_roots],
            "allowed_remote_destinations": sorted(self.allowed_destinations),
            "local_destinations_always_allowed": sorted(LOCAL_DESTINATIONS),
            "arbitrary_urls_allowed": False,
            "remote_https_required": True,
        }

    def require_path(
        self,
        value: str | Path,
        *,
        label: str,
        must_exist: bool = True,
    ) -> Path:
        path = Path(value).expanduser().resolve()
        if not any(
            path == root or path.is_relative_to(root) for root in self.allowed_roots
        ):
            raise ValueError(
                f"{label} is outside VKP_MODEL_CONNECTOR_ALLOWED_ROOTS: {path}"
            )
        if must_exist and not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        return path

    def require_provider_destination(
        self,
        task: str,
        provider_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        config = secretless_provider_config(provider_config)
        if not config:
            raise ValueError("provider_config is required for a trusted connector call")
        preview = model_task_api_call(
            task,
            provider_config=config,
            execute=False,
            write=False,
        )
        plan = (
            preview.get("request_plan")
            if isinstance(preview.get("request_plan"), dict)
            else {}
        )
        provider = (
            plan.get("provider") if isinstance(plan.get("provider"), dict) else {}
        )
        self.require_destination_identity(provider)
        return provider

    def require_destination_identity(
        self, deployment: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(deployment, dict):
            raise ValueError("deployment identity must be an object")
        base_url = str(deployment.get("base_url") or "").strip()
        parsed = urllib.parse.urlsplit(base_url)
        hostname = str(parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not hostname:
            raise ValueError(
                "provider base_url must resolve to an explicit http(s) destination"
            )
        if hostname in LOCAL_DESTINATIONS:
            return deployment
        if parsed.scheme != "https":
            raise ValueError("remote model destinations must use https")
        destination = _destination_from_url(parsed)
        if (
            destination not in self.allowed_destinations
            and hostname not in self.allowed_destinations
        ):
            allowed = ", ".join(sorted(self.allowed_destinations)) or "none"
            raise ValueError(
                f"provider destination is not allowlisted: {destination}; allowed={allowed}"
            )
        return deployment

    def require_consent_scope(
        self,
        consent_path: str | Path,
        *,
        provider_config: dict[str, Any] | None = None,
        expected_task: str = "",
        require_execution_contract: bool = False,
    ) -> dict[str, Any]:
        path = self.require_path(consent_path, label="consent_path")
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"consent_path must contain valid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("consent payload must be an object")
        task = str(payload.get("task") or "").strip()
        if not task:
            raise ValueError("consent task is missing")
        if expected_task and str(expected_task).strip() != task:
            raise ValueError("requested task differs from consent")
        if require_execution_contract:
            _require_v2_execution_contract(payload, policy=self)
        if provider_config is not None:
            self.require_provider_destination(task, provider_config)
        else:
            deployments = payload.get("authorized_deployments")
            if not isinstance(deployments, list):
                provider = payload.get("provider")
                deployments = [provider] if isinstance(provider, dict) else []
            if not deployments:
                raise ValueError("consent has no authorized deployment destinations")
            for deployment in deployments:
                self.require_destination_identity(deployment)
        authorised_destinations = payload.get("authorized_destinations")
        if authorised_destinations is not None:
            if not isinstance(authorised_destinations, list) or not authorised_destinations:
                raise ValueError("consent authorized_destinations must be a non-empty list")
            for destination in authorised_destinations:
                self.require_destination_identity(
                    {"base_url": str(destination or "")}
                )
        for row in payload.get("artifacts") or []:
            if not isinstance(row, dict):
                raise ValueError("consent artifact record must be an object")
            self.require_path(str(row.get("path") or ""), label="consent artifact")
        return payload


def _require_v2_execution_contract(
    payload: dict[str, Any], *, policy: TrustedModelConnectorPolicy | None = None
) -> None:
    if payload.get("schema") != SCHEMA_V2:
        raise ValueError("remote execution requires consent v2")
    if payload.get("status") != "active" or not payload.get(
        "user_confirmed_data_export"
    ):
        raise ValueError("remote execution requires active operator-confirmed consent")
    artifacts = (
        payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
    )
    manifest = (
        payload.get("upload_manifest")
        if isinstance(payload.get("upload_manifest"), dict)
        else {}
    )
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    manifest_hash = _upload_manifest_sha256(files) if files else ""
    if manifest.get("schema") != UPLOAD_MANIFEST_SCHEMA or files != artifacts:
        raise ValueError("remote execution requires an exact per-file upload manifest")
    if str(manifest.get("manifest_sha256") or "") != manifest_hash:
        raise ValueError("upload manifest SHA-256 is invalid")
    confirmation = (
        payload.get("operator_confirmation")
        if isinstance(payload.get("operator_confirmation"), dict)
        else {}
    )
    if (
        not confirmation.get("confirmed")
        or not str(confirmation.get("confirmed_at") or "").strip()
        or str(confirmation.get("confirmation_method") or "")
        not in {"visible_operator_shell", "parent_business_authorization"}
        or str(confirmation.get("exact_manifest_sha256") or "") != manifest_hash
    ):
        raise ValueError("operator confirmation is not bound to the upload manifest")
    if (
        str(confirmation.get("confirmation_method") or "")
        == "parent_business_authorization"
    ):
        from .model_business_authorization import (
            validate_parent_authorized_child_consent,
        )

        parent_status = validate_parent_authorized_child_consent(
            payload, policy=policy
        )
        if not parent_status.get("valid"):
            messages = "; ".join(
                str(row.get("message") or row.get("key") or "blocked")
                for row in parent_status.get("blockers") or []
            )
            raise ValueError(
                f"parent business authorization is invalid: {messages or 'blocked'}"
            )
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    try:
        total_limit = float(scope.get("max_estimated_cost_usd") or 0)
        per_call_limit = float(scope.get("max_cost_per_call_usd") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("consent cost limits are invalid") from exc
    if total_limit <= 0 or per_call_limit <= 0 or per_call_limit > total_limit:
        raise ValueError(
            "remote execution requires positive call and total cost limits"
        )


def _default_allowed_roots(default_root: str | Path) -> list[Path]:
    root = Path(default_root).expanduser().resolve()
    managed_output = root.parent / "video-knowledge-output"
    if root.name == "video-knowledge-pipeline" and managed_output.is_dir():
        return [root, managed_output.resolve()]
    return [root]


def _split_roots(value: str) -> list[Path]:
    normalised = value.replace("\r", "\n")
    parts: list[str] = []
    for line in normalised.split("\n"):
        parts.extend(part for part in line.split(os.pathsep) if part.strip())
    if not parts:
        raise ValueError(f"{ALLOWED_ROOTS_ENV} must contain at least one path")
    return [Path(part.strip()) for part in parts]


def _normalise_destination(value: str) -> str:
    raw = str(value or "").strip().lower().rstrip("/")
    if not raw:
        return ""
    parsed = urllib.parse.urlsplit(raw if "://" in raw else f"https://{raw}")
    if not parsed.hostname:
        raise ValueError(f"invalid allowed destination: {value}")
    return _destination_from_url(parsed)


def _destination_from_url(parsed: urllib.parse.SplitResult) -> str:
    host = str(parsed.hostname or "").lower()
    port = parsed.port
    if (
        port is None
        or (parsed.scheme == "https" and port == 443)
        or (parsed.scheme == "http" and port == 80)
    ):
        return host
    return f"{host}:{port}"
