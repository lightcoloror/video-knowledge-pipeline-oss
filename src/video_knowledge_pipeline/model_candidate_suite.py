from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .file_hash import sha256_file
from .model_api_settings import SETTINGS_SCHEMA, load_model_api_settings
from .model_output_contracts import normalise_output_contract
from .model_route_settings import TASK_CAPABILITIES, resolve_model_route
from .storage import read_json, write_json
from .utils import now_iso


PLAN_SCHEMA = "video_knowledge_pipeline.model_candidate_fixed_suite.v1"
PREPARED_SCHEMA = "video_knowledge_pipeline.model_candidate_fixed_suite_prepared.v1"
GATEWAY_CONFIG_SCHEMA = "video_knowledge_pipeline.model_gateway_config.v1"


def prepare_model_candidate_suite(
    plan_path: str | Path,
    *,
    settings_path: str | Path,
    output_dir: str | Path,
    port: int | None = None,
) -> dict[str, Any]:
    plan_file = Path(plan_path).expanduser().resolve()
    source_settings_file = Path(settings_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    plan = read_json(plan_file)
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid fixed candidate suite plan")
    settings = load_model_api_settings(source_settings_file)
    profiles = {str(row["id"]): dict(row) for row in settings["profiles"]}
    requested_port = int(
        port or (plan.get("gateway") or {}).get("requested_port") or 8777
    )
    if requested_port < 1 or requested_port > 65535:
        raise ValueError("benchmark gateway port must be between 1 and 65535")

    destination.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, Any]] = []
    for case in plan.get("cases") or []:
        if not isinstance(case, dict):
            raise ValueError("candidate suite cases must be objects")
        case_id = _safe_id(str(case.get("id") or ""), field="case id")
        artifacts = [_artifact_row(value) for value in case.get("artifacts") or []]
        if not artifacts:
            raise ValueError(f"candidate suite case has no artifacts: {case_id}")
        instructions = str(case.get("instructions") or "")
        if not instructions:
            raise ValueError(f"candidate suite case has no instructions: {case_id}")
        asr_prompt = str(case.get("asr_prompt") or "")
        raw_output_contract = (
            dict(case["output_contract"])
            if isinstance(case.get("output_contract"), dict)
            else {}
        )
        raw_output_contract.setdefault(
            "format", str(case.get("expected_format") or "any")
        )
        raw_output_contract.setdefault(
            "required_term_groups", list(case.get("required_term_groups") or [])
        )
        raw_output_contract.setdefault(
            "forbidden_markers", list(case.get("forbidden_markers") or [])
        )
        output_contract = normalise_output_contract(raw_output_contract)
        for index, raw_candidate in enumerate(case.get("candidates") or [], start=1):
            candidate = dict(raw_candidate)
            profile_id = _safe_id(
                str(candidate.get("profile_id") or ""), field="profile id"
            )
            route_task = str(
                candidate.get("route_task") or case.get("model_type") or ""
            )
            capability = TASK_CAPABILITIES.get(route_task)
            if not capability:
                raise ValueError(f"unsupported candidate route task: {route_task}")
            profile = profiles.get(profile_id)
            if not profile:
                raise ValueError(f"candidate profile not found: {profile_id}")
            if not bool(profile.get("enabled", True)):
                raise ValueError(f"candidate profile is disabled: {profile_id}")
            if str(profile.get("location") or "") != "remote":
                raise ValueError(f"candidate profile must be remote: {profile_id}")
            if capability not in profile.get("capabilities", []):
                raise ValueError(
                    f"candidate profile {profile_id} does not support {capability}"
                )
            if asr_prompt and capability != "asr":
                raise ValueError(
                    f"asr_prompt is only allowed for ASR candidates: {profile_id}"
                )

            profile = dict(profile)
            route_locked_options = dict(profile.get("provider_options") or {})
            case_options = (
                dict(case["request_options"])
                if isinstance(case.get("request_options"), dict)
                else {}
            )
            candidate_options = (
                dict(candidate["request_options"])
                if isinstance(candidate.get("request_options"), dict)
                else {}
            )
            route_locked_options.update(case_options)
            route_locked_options.update(candidate_options)
            if (
                str(profile.get("provider") or "") == "volcengine_coding_plan"
                and str(case.get("expected_format") or "").lower() == "json"
            ):
                route_locked_options.setdefault("response_format", "json_object")
            if (
                str(profile.get("provider") or "") == "volcengine_coding_plan"
                and str(profile.get("model") or "").lower() == "minimax-m3"
            ):
                route_locked_options.setdefault("thinking_mode", "disabled")
            profile["provider_options"] = route_locked_options

            candidate_id = f"{case_id}--{profile_id}"
            candidate_dir = destination / f"{index:02d}-{_filename_id(candidate_id)}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            pool_id = _pool_id(case_id, profile_id, capability)
            candidate_settings = {
                "schema": SETTINGS_SCHEMA,
                "profiles": [profile],
                "task_routes": {},
                "route_pools": [
                    {
                        "id": pool_id,
                        "name": f"Benchmark {case_id} / {profile_id}",
                        "location": "remote_approved",
                        "capability": capability,
                        "deployments": [profile_id],
                        "retry_policy": {
                            "max_retries": 0,
                            "timeout_seconds": int(
                                profile.get("timeout_seconds") or 120
                            ),
                            "cooldown_seconds": 0,
                        },
                    }
                ],
                "route_bindings": {
                    route_task: {
                        "default_location": "remote",
                        "local_pool_id": "",
                        "remote_pool_id": pool_id,
                    }
                },
                "updated_at": now_iso(),
            }
            candidate_settings_path = candidate_dir / "model-api-settings.json"
            output_contract_path = candidate_dir / "output-contract.json"
            write_json(candidate_settings_path, candidate_settings)
            write_json(output_contract_path, output_contract)
            loaded = load_model_api_settings(candidate_settings_path)
            route = resolve_model_route(
                loaded,
                task=route_task,
                execution_location="remote",
            )
            gateway_config_path = candidate_dir / "model-gateway.json"
            write_json(
                gateway_config_path,
                {
                    "schema": GATEWAY_CONFIG_SCHEMA,
                    "host": "127.0.0.1",
                    "port": requested_port,
                    "telemetry": False,
                    "config_path": str(candidate_dir / "litellm-config.yaml"),
                    "pid_path": str(candidate_dir / "model-gateway.pid"),
                    "log_path": str(candidate_dir / "model-gateway.log"),
                },
            )
            prepared.append(
                {
                    "candidate_id": candidate_id,
                    "case_id": case_id,
                    "task": str(case.get("task") or ""),
                    "connector_task": str(
                        candidate.get("connector_task") or case.get("task") or ""
                    ),
                    "model_type": str(case.get("model_type") or route_task),
                    "contract_status": str(candidate.get("contract_status") or "exact"),
                    "route_task": route_task,
                    "profile_id": profile_id,
                    "provider": str(profile.get("provider") or ""),
                    "model": str(profile.get("model") or ""),
                    "destination": str(
                        urlsplit(str(profile.get("base_url") or "")).hostname or ""
                    ),
                    "settings_path": str(candidate_settings_path),
                    "gateway_config_path": str(gateway_config_path),
                    "route_id": str(route["route_id"]),
                    "route_revision": str(route["route_revision"]),
                    "virtual_model": str(route["virtual_model"]),
                    "artifacts": artifacts,
                    "instructions": instructions,
                    "instructions_sha256": hashlib.sha256(
                        instructions.encode("utf-8")
                    ).hexdigest(),
                    "asr_prompt": asr_prompt,
                    "asr_prompt_sha256": hashlib.sha256(
                        asr_prompt.encode("utf-8")
                    ).hexdigest()
                    if asr_prompt
                    else "",
                    "expires_hours": float(
                        candidate.get("expires_hours") or case.get("expires_hours") or 4
                    ),
                    "max_calls": 1,
                    "max_estimated_cost_usd": float(
                        candidate.get("max_estimated_cost_usd")
                        or case.get("max_estimated_cost_usd")
                        or 0.01
                    ),
                    "max_cost_per_call_usd": float(
                        candidate.get("max_cost_per_call_usd")
                        or case.get("max_cost_per_call_usd")
                        or candidate.get("max_estimated_cost_usd")
                        or case.get("max_estimated_cost_usd")
                        or 0.01
                    ),
                    "max_retries_per_call": int(
                        candidate.get("max_retries_per_call")
                        if candidate.get("max_retries_per_call") is not None
                        else case.get("max_retries_per_call") or 0
                    ),
                    "expected_format": str(case.get("expected_format") or "any"),
                    "route_locked_request_options": route_locked_options,
                    "output_contract": output_contract,
                    "output_contract_path": str(output_contract_path),
                    "consent_path": str(candidate_dir / "consent.v2.json"),
                    "execution_report_path": str(
                        candidate_dir / "connector-execution.json"
                    ),
                }
            )

    result = {
        "schema": PREPARED_SCHEMA,
        "status": "ready_for_operator_consent" if prepared else "incomplete",
        "plan_path": str(plan_file),
        "source_settings_path": str(source_settings_file),
        "output_dir": str(destination),
        "gateway": {
            "host": "127.0.0.1",
            "port": requested_port,
            "port_registration_required": True,
            "sequential_candidate_gateways": True,
        },
        "candidate_count": len(prepared),
        "candidates": prepared,
        "operator_boundary": {
            "remote_requests_made": 0,
            "consents_created": 0,
            "credentials_read": False,
            "artifact_bytes_uploaded": 0,
            "port_registered": False,
            "saving_this_plan_authorizes_egress": False,
            "default_routes_changed": False,
        },
        "updated_at": now_iso(),
    }
    write_json(destination / "prepared-suite.json", result)
    return result


def _artifact_row(value: Any) -> dict[str, Any]:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"candidate suite artifact not found: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _pool_id(case_id: str, profile_id: str, capability: str) -> str:
    digest = hashlib.sha256(
        f"{case_id}|{profile_id}|{capability}".encode()
    ).hexdigest()[:12]
    prefix = _filename_id(f"bench-{capability}-{profile_id}")[:76].strip("-")
    return f"{prefix}-{digest}"


def _safe_id(value: str, *, field: str) -> str:
    clean = str(value or "").strip().lower()
    if not clean or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", clean):
        raise ValueError(f"invalid {field}: {value!r}")
    return clean


def _filename_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare isolated VKP fixed-candidate settings"
    )
    parser.add_argument("plan")
    parser.add_argument("--settings-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    result = prepare_model_candidate_suite(
        args.plan,
        settings_path=args.settings_path,
        output_dir=args.output_dir,
        port=args.port or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready_for_operator_consent" else 2


if __name__ == "__main__":
    raise SystemExit(main())
