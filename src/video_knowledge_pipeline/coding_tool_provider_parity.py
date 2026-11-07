from __future__ import annotations

import argparse
import difflib
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file as _file_sha256
from .model_json import extract_last_json_document
from .model_api_settings import (
    _load_secret_document,
    _profile_secret_id,
    _read_secret,
    default_secrets_path,
    default_settings_path,
    load_model_api_settings,
)
from .model_connector_consent import (
    _normalise_route_snapshot,
    create_model_connector_consent,
    record_model_connector_attempt,
    reserve_model_connector_attempt,
    validate_model_connector_consent,
)
from .storage import read_json, write_json
from .text_llm_gateway import (
    build_openai_compatible_text_body,
    extract_json_document,
    openai_compatible_chat_completions_url,
)
from .trusted_model_connector_policy import TrustedModelConnectorPolicy
from .utils import now_iso


PLAN_SCHEMA = "video_knowledge_pipeline.coding_tool_provider_parity_plan.v1"
CONSENT_INDEX_SCHEMA = (
    "video_knowledge_pipeline.coding_tool_provider_parity_consent_index.v1"
)
EXECUTION_SCHEMA = "video_knowledge_pipeline.coding_tool_provider_parity_execution.v1"
BATCH_EXECUTION_SCHEMA = (
    "video_knowledge_pipeline.coding_tool_provider_parity_batch_execution.v1"
)
COMPARISON_SCHEMA = "video_knowledge_pipeline.coding_tool_provider_parity_comparison.v1"
DEFAULT_OUTPUT_DIR = ".local/coding-plan-siliconflow-parity-20260718"
DEFAULT_FIXTURE = "fixtures/coding-tool-provider-parity/atomic-quota-fixture.txt"
NATIVE_EXECUTION_CLIENT = "vkp_native_openai_compatible"
OPENCLAW_EXECUTION_CLIENT = "openclaw"
COMMON_FIELDS_PROFILE = "common_fields_v1"
CONTENT_QUALITY_PROFILE = "content_quality_v1"
CAPABILITY_CEILING_PROFILE = "capability_ceiling_v1"
REQUEST_PROFILES: dict[str, dict[str, Any]] = {
    COMMON_FIELDS_PROFILE: {
        "id": COMMON_FIELDS_PROFILE,
        "max_tokens": 1024,
        "stream": False,
        "timeout_seconds": 120,
        "max_cost_per_candidate_usd": 0.02,
        "vendor_reasoning_controls": "omitted",
        "purpose": "measure provider-default serving behavior with common fields",
    },
    CONTENT_QUALITY_PROFILE: {
        "id": CONTENT_QUALITY_PROFILE,
        "max_tokens": 16384,
        "stream": True,
        "timeout_seconds": 300,
        "max_cost_per_candidate_usd": 0.08,
        "vendor_reasoning_controls": "omitted_unless_exact_model_support_is_documented",
        "purpose": "obtain complete final answers for content-quality pairing",
    },
    CAPABILITY_CEILING_PROFILE: {
        "id": CAPABILITY_CEILING_PROFILE,
        "max_tokens": None,
        "stream": True,
        "timeout_seconds": 900,
        "max_cost_per_candidate_usd": 0.20,
        "vendor_reasoning_controls": "omitted_to_preserve_provider_managed_reasoning",
        "purpose": "measure natural task completion without a VKP output-token or thinking budget",
    },
}
EXPECTED_DESTINATIONS = frozenset(
    {"ark.cn-beijing.volces.com", "api.siliconflow.cn"}
)
OFFICIAL_REFERENCES = {
    "volcengine_coding_plan": {
        "url": "https://developer.volcengine.com/articles/7615528054736945158",
        "catalog_url": "https://www.volcengine.com/activity/codingplan",
        "other_tools_url": "https://www.volcengine.com/docs/82379/2188959?lang=zh",
        "reference_example": "Volcengine Coding Plan AI coding client configuration",
        "wire_protocol": "OpenAI-compatible chat completions",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "usage_scope": "coding_plan_ai_tool_integration",
        "forbidden_substitute": "https://ark.cn-beijing.volces.com/api/v3",
        "native_client_acceptance_not_preclaimed": True,
        "per_request_reasoning_control_documented_for_coding_endpoint": False,
        "standard_inference_reasoning_reference": "https://www.volcengine.com/docs/82379/1795150",
    },
    "siliconflow": {
        "url": "https://docs.siliconflow.cn/cn/usercases/use-siliconcloud-in-ccswitch",
        "api_reference": "https://docs.siliconflow.cn/en/api-reference/chat-completions/chat-completions",
        "model_catalog_url": "https://www.siliconflow.cn/models",
        "pricing_url": "https://www.siliconflow.cn/pricing",
        "reference_example": "SiliconCloud in CC Switch",
        "wire_protocol": "OpenAI-compatible chat completions",
        "base_url": "https://api.siliconflow.cn/v1",
        "endpoint": "/v1/chat/completions",
        "generic_reasoning_fields": ["enable_thinking", "thinking_budget"],
        "exact_candidate_support_list_complete": False,
    },
}

MODEL_PAIRS: tuple[dict[str, str], ...] = (
    {
        "id": "deepseek-v4-pro",
        "family": "DeepSeek V4 Pro",
        "ark_profile_id": "ark-deepseek-v4-pro",
        "ark_model": "deepseek-v4-pro",
        "siliconflow_profile_id": "siliconflow-deepseek-v4-pro",
        "siliconflow_model": "deepseek-ai/DeepSeek-V4-Pro",
    },
    {
        "id": "deepseek-v4-flash",
        "family": "DeepSeek V4 Flash",
        "ark_profile_id": "ark-deepseek-v4-flash",
        "ark_model": "deepseek-v4-flash",
        "siliconflow_profile_id": "siliconflow-deepseek-v4-flash",
        "siliconflow_model": "deepseek-ai/DeepSeek-V4-Flash",
    },
    {
        "id": "glm-5-2",
        "family": "GLM-5.2",
        "ark_profile_id": "ark-glm-latest",
        "ark_model": "glm-5.2",
        "siliconflow_profile_id": "siliconflow-glm-5-2",
        "siliconflow_model": "zai-org/GLM-5.2",
    },
    {
        "id": "kimi-k2-7-code",
        "family": "Kimi K2.7 Code",
        "ark_profile_id": "ark-kimi-k2-7-code",
        "ark_model": "kimi-k2.7-code",
        "siliconflow_profile_id": "siliconflow-kimi-k2-7-code",
        "siliconflow_model": "moonshotai/Kimi-K2.7-Code",
    },
    {
        "id": "kimi-k2-6",
        "family": "Kimi K2.6",
        "ark_profile_id": "ark-kimi-k2-6",
        "ark_model": "kimi-k2.6",
        "siliconflow_profile_id": "siliconflow-kimi-k2-6",
        "siliconflow_model": "Pro/moonshotai/Kimi-K2.6",
    },
)

BENCHMARK_INSTRUCTIONS = """你是代码审查与并发数据库专家。请审查随请求提供的 atomic_quota.py 固定样本。

目标：修复多个独立进程、独立 SQLite 连接并发调用 reserve_quota 时的 check-then-act 竞态，保证同一 user_id 最多只有 limit 次返回 true。

硬性要求：
1. 保持 public function signature 和 database schema 不变。
2. 不得使用进程内 Python lock。
3. 拒绝和失败路径不能遗留打开的事务。
4. 给出可运行的跨进程并发测试思路。
5. 只返回一个 JSON 对象，不要 Markdown 代码围栏，不要 <think> 或推理过程。

JSON 必须包含：bug_class、explanation、patch、tests、tradeoffs。patch 必须是 unified diff 字符串；tests 必须是字符串数组。"""

OUTPUT_CONTRACT = {
    "format": "json",
    "required_keys": {
        "bug_class": "string",
        "explanation": "string",
        "patch": "string",
        "tests": "array",
        "tradeoffs": "array",
    },
    "forbidden_markers": ["<think>", "```"],
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prepare_coding_tool_provider_parity(
    *,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    execution_client: str = NATIVE_EXECUTION_CLIENT,
    request_profile_id: str = COMMON_FIELDS_PROFILE,
) -> dict[str, Any]:
    root = project_root()
    settings_file = Path(settings_path or default_settings_path()).expanduser().resolve()
    secrets_file = Path(
        secrets_path or default_secrets_path(settings_path=settings_file)
    ).expanduser().resolve()
    artifact = Path(artifact_path or root / DEFAULT_FIXTURE).expanduser().resolve()
    destination = Path(output_dir or root / DEFAULT_OUTPUT_DIR).expanduser().resolve()
    _require_under_root(artifact, root, label="artifact")
    _require_under_root(destination, root, label="output_dir", must_exist=False)
    if not artifact.is_file():
        raise FileNotFoundError(f"benchmark artifact does not exist: {artifact}")
    client = str(execution_client or "").strip()
    if client not in {NATIVE_EXECUTION_CLIENT, OPENCLAW_EXECUTION_CLIENT}:
        raise ValueError(f"unsupported parity execution client: {execution_client!r}")
    request_profile = _request_profile_definition(request_profile_id)
    if client == OPENCLAW_EXECUTION_CLIENT and request_profile_id != COMMON_FIELDS_PROFILE:
        raise ValueError("content-quality profile requires the VKP native client")

    settings = load_model_api_settings(settings_file)
    profiles = {str(row["id"]): dict(row) for row in settings["profiles"]}
    secret_ids = set(_load_secret_document(secrets_file).get("items", {}))
    candidates: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    for pair in MODEL_PAIRS:
        pair_candidates: list[str] = []
        for side in ("ark", "siliconflow"):
            profile_id = pair[f"{side}_profile_id"]
            expected_provider = (
                "volcengine_coding_plan" if side == "ark" else "siliconflow"
            )
            expected_base_url = OFFICIAL_REFERENCES[expected_provider]["base_url"]
            expected_model = pair[f"{side}_model"]
            profile = profiles.get(profile_id)
            candidate_id = f"{pair['id']}--{side}"
            pair_candidates.append(candidate_id)
            if not profile:
                blockers.append(
                    {"key": "profile_missing", "message": f"missing profile {profile_id}"}
                )
                continue
            mismatch = _profile_mismatches(
                profile,
                provider=expected_provider,
                base_url=expected_base_url,
                model=expected_model,
            )
            if mismatch:
                blockers.append(
                    {
                        "key": "profile_contract_mismatch",
                        "message": f"{profile_id}: {', '.join(mismatch)}",
                    }
                )
            credential_ready = _profile_secret_id(profile) in secret_ids
            if not credential_ready:
                blockers.append(
                    {
                        "key": "credential_missing",
                        "message": f"credential is not configured for {profile_id}",
                    }
                )
            route = _route_snapshot(
                candidate_id,
                profile,
                request_profile_id=request_profile_id,
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "pair_id": pair["id"],
                    "family": pair["family"],
                    "side": side,
                    "profile_id": profile_id,
                    "provider": expected_provider,
                    "base_url": expected_base_url,
                    "destination": urllib.parse.urlsplit(expected_base_url).hostname,
                    "model": expected_model,
                    "api": "openai_chat_completions",
                    "interface": "openai_chat_completions",
                    "credential_ref": str(profile.get("secret_ref") or ""),
                    "credential_ready": credential_ready,
                    "route": route,
                    "result_path": str(destination / candidate_id / "execution.json"),
                    "consent_path": str(destination / candidate_id / "consent.v2.json"),
                }
            )
        pairs.append(
            {
                "id": pair["id"],
                "family": pair["family"],
                "candidate_ids": pair_candidates,
            }
        )

    artifact_record = _artifact_record(artifact)
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "ready_for_operator_consent" if not blockers else "incomplete",
        "purpose": (
            "same-model AI coding natural capability-ceiling comparison through VKP native OpenAI-compatible client"
            if client == NATIVE_EXECUTION_CLIENT
            and request_profile_id == CAPABILITY_CEILING_PROFILE
            else "same-model AI coding content-quality comparison through VKP native OpenAI-compatible client"
            if client == NATIVE_EXECUTION_CLIENT
            and request_profile_id == CONTENT_QUALITY_PROFILE
            else "same-model AI coding task comparison through VKP native OpenAI-compatible client"
            if client == NATIVE_EXECUTION_CLIENT
            else "same-model AI coding task comparison through OpenClaw"
        ),
        "settings_path": str(settings_file),
        "secrets_path": str(secrets_file),
        "output_dir": str(destination),
        "official_contracts": OFFICIAL_REFERENCES,
        "comparison_contract": {
            "execution_client": client,
            "same_tool": (
                "VKP native OpenAI-compatible client"
                if client == NATIVE_EXECUTION_CLIENT
                else "OpenClaw"
            ),
            "same_api_adapter": "openai_chat_completions",
            "same_artifact": True,
            "same_instructions": True,
            "temperature": 0,
            "request_profile_id": request_profile_id,
            "max_tokens": _request_max_tokens(request_profile),
            "thinking": (
                (
                    "provider_specific_field_omitted"
                    if request_profile_id == COMMON_FIELDS_PROFILE
                    else str(request_profile["vendor_reasoning_controls"])
                )
                if client == NATIVE_EXECUTION_CLIENT
                else "off"
            ),
            "reasoning_mode_scope": (
                (
                    "provider_default_without_vendor_specific_override"
                    if request_profile_id == COMMON_FIELDS_PROFILE
                    else "provider_default_without_unverified_vendor_specific_override"
                )
                if client == NATIVE_EXECUTION_CLIENT
                else "legacy_runtime_controlled"
            ),
            "measured_scope": (
                "natural_completion_content_quality_for_locked_model_aliases"
                if request_profile_id == CAPABILITY_CEILING_PROFILE
                else "complete_final_answer_content_quality_for_locked_model_aliases"
                if request_profile_id == CONTENT_QUALITY_PROFILE
                else "provider_default_serving_behavior_for_locked_model_aliases"
            ),
            "same_weights_assumed": False,
            "streaming": (
                bool(request_profile["stream"])
                if client == NATIVE_EXECUTION_CLIENT
                else True
            ),
            "timeout_seconds": int(request_profile["timeout_seconds"]),
            "complete_answer_strategy": (
                "omit_client_output_and_thinking_budgets_and_stream_until_provider_stop"
                if request_profile_id == CAPABILITY_CEILING_PROFILE
                else
                "stream_response_and_raise_combined_reasoning_plus_answer_budget"
                if request_profile_id == CONTENT_QUALITY_PROFILE
                else "fixed_small_common_budget"
            ),
            "reasoning_content_is_not_accepted_as_final_answer": True,
            "unverified_provider_fields_sent": [],
            "external_retry_count": 0,
            "fallbacks": [],
            "tool_surface": client == OPENCLAW_EXECUTION_CLIENT,
            "tools_denied": ["*"] if client == OPENCLAW_EXECUTION_CLIENT else [],
            "coding_plan_not_used_for_general_vkp_tasks": True,
            "official_scope_note": (
                "The request URL and fields follow the official Coding Plan OpenAI-compatible coding-client example; "
                "the VKP native client is not claimed as a separately listed Coding Plan product integration."
            ),
        },
        "artifact": artifact_record,
        "upload_manifest": {
            "files": [artifact_record],
            "destinations": sorted(EXPECTED_DESTINATIONS),
            "candidate_count": len(candidates),
            "max_calls": len(candidates),
            "max_cost_per_candidate_usd": float(
                request_profile["max_cost_per_candidate_usd"]
            ),
            "max_total_cost_usd": round(
                len(candidates)
                * float(request_profile["max_cost_per_candidate_usd"]),
                2,
            ),
        },
        "instructions": BENCHMARK_INSTRUCTIONS,
        "instructions_sha256": _text_sha256(BENCHMARK_INSTRUCTIONS),
        "output_contract": OUTPUT_CONTRACT,
        "pairs": pairs,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "blockers": blockers,
        "operator_boundary": {
            "remote_requests_made": 0,
            "credentials_read": False,
            "consents_created": 0,
            "artifact_bytes_uploaded": 0,
            "saving_plan_authorizes_egress": False,
            "platform_policy_may_still_block": True,
        },
        "updated_at": now_iso(),
    }
    plan["plan_sha256"] = _payload_sha256(plan)
    destination.mkdir(parents=True, exist_ok=True)
    plan_path = destination / "parity-plan.json"
    write_json(plan_path, plan)
    (destination / "operator-authorization-request.md").write_text(
        _render_authorization_request(plan), encoding="utf-8"
    )
    plan["artifacts"] = {
        "plan": str(plan_path),
        "authorization_request": str(destination / "operator-authorization-request.md"),
    }
    return plan


def create_coding_tool_provider_parity_consents(
    plan_path: str | Path,
    *,
    confirm_data_export: bool,
) -> dict[str, Any]:
    if not confirm_data_export:
        raise ValueError("--confirm-data-export is required in a visible operator shell")
    plan_file = Path(plan_path).expanduser().resolve()
    plan = _load_and_validate_plan(plan_file, require_ready=True)
    _revalidate_plan_profiles(plan)
    execution_client = _plan_execution_client(plan)
    request_profile = _plan_request_profile(plan)
    cost_per_candidate = float(request_profile["max_cost_per_candidate_usd"])
    artifact = _validate_artifact_record(plan["artifact"])
    rows: list[dict[str, Any]] = []
    for candidate in plan["candidates"]:
        candidate_dir = Path(candidate["consent_path"]).parent
        candidate_dir.mkdir(parents=True, exist_ok=True)
        consent = create_model_connector_consent(
            candidate_dir,
            task="provider_task_benchmark",
            artifact_paths=[artifact],
            route_snapshot=candidate["route"],
            instructions=str(plan["instructions"]),
            output_contract=dict(plan["output_contract"]),
            purpose=(
                f"same-model AI coding benchmark through {execution_client}; "
                f"candidate={candidate['candidate_id']}"
            ),
            expires_hours=24,
            max_calls=1,
            max_estimated_cost_usd=cost_per_candidate,
            max_cost_per_call_usd=cost_per_candidate,
            confirm_data_export=True,
            output_path=candidate["consent_path"],
            write=True,
        )
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "consent_path": consent["consent_path"],
                "consent_id": consent["consent_id"],
                "route_revision": consent["route"]["route_revision"],
                "upload_manifest_sha256": consent["upload_manifest"][
                    "manifest_sha256"
                ],
            }
        )
    result = {
        "schema": CONSENT_INDEX_SCHEMA,
        "status": "ready_for_operator_execution",
        "plan_path": str(plan_file),
        "plan_sha256": plan["plan_sha256"],
        "consent_count": len(rows),
        "consents": rows,
        "operator_boundary": {
            "remote_requests_made": 0,
            "credentials_read": False,
            "artifact_bytes_uploaded": 0,
            "consents_are_candidate_specific": True,
            "execution_still_requires_operator_confirm_network": True,
        },
        "updated_at": now_iso(),
    }
    index_path = Path(plan["output_dir"]) / "consent-index.json"
    write_json(index_path, result)
    result["consent_index_path"] = str(index_path)
    return result


def execute_coding_tool_provider_parity_candidate(
    plan_path: str | Path,
    consent_index_path: str | Path,
    *,
    candidate_id: str,
    operator_confirm_network: bool,
    openclaw_command: str = "openclaw",
) -> dict[str, Any]:
    if not operator_confirm_network:
        raise ValueError("--operator-confirm-network is required for a real provider call")
    plan_file = Path(plan_path).expanduser().resolve()
    index_file = Path(consent_index_path).expanduser().resolve()
    plan = _load_and_validate_plan(plan_file, require_ready=True)
    index = _load_and_validate_consent_index(index_file, plan=plan)
    candidate = _candidate(plan, candidate_id)
    _revalidate_plan_profiles(plan, candidate_ids={candidate_id})
    execution_client = _plan_execution_client(plan)
    resolved_openclaw_command = (
        _resolve_openclaw_command(openclaw_command)
        if execution_client == OPENCLAW_EXECUTION_CLIENT
        else ""
    )
    consent_row = next(
        (
            dict(row)
            for row in index.get("consents") or []
            if str(row.get("candidate_id") or "") == candidate_id
        ),
        None,
    )
    if not consent_row:
        raise ValueError(f"consent not found for candidate: {candidate_id}")
    consent_path = Path(consent_row["consent_path"]).expanduser().resolve()
    policy = TrustedModelConnectorPolicy(
        allowed_roots=(project_root(),),
        allowed_destinations=EXPECTED_DESTINATIONS,
    )
    policy.require_path(consent_path, label="consent_path")
    policy.require_destination_identity(candidate["route"]["deployments"][0])
    policy.require_consent_scope(
        consent_path,
        expected_task="provider_task_benchmark",
        require_execution_contract=True,
    )
    validation = validate_model_connector_consent(
        consent_path,
        route_snapshot=candidate["route"],
        expected_route_revision=candidate["route"]["route_revision"],
        expected_task="provider_task_benchmark",
        expected_calls=1,
    )
    if not validation.get("valid"):
        return _blocked_execution(candidate, consent_path, "consent_required", validation)
    secret_file = Path(plan["secrets_path"]).expanduser().resolve()
    settings = load_model_api_settings(plan["settings_path"])
    profile = next(
        row for row in settings["profiles"] if row["id"] == candidate["profile_id"]
    )
    api_key = _read_secret(_profile_secret_id(profile), secret_file)
    if not api_key:
        raise ValueError(f"credential is unavailable for {candidate['profile_id']}")
    reservation = reserve_model_connector_attempt(
        consent_path,
        route_snapshot=candidate["route"],
        expected_route_revision=candidate["route"]["route_revision"],
        expected_task="provider_task_benchmark",
        expected_calls=1,
    )
    if not reservation.get("reserved"):
        return _blocked_execution(candidate, consent_path, "consent_required", reservation)

    reserved_cost = float(reservation.get("reserved_cost_usd") or 0)
    result_path = Path(candidate["result_path"]).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    external_attempted = False
    completed = False
    reported_cost: float | None = 0.0
    started = time.perf_counter()
    audit: dict[str, Any] = {}
    command_result: dict[str, Any] = {}
    native_result: dict[str, Any] = {}
    proxy: SingleRequestAuditProxy | None = None
    try:
        prompt = _benchmark_prompt(plan)
        if execution_client == NATIVE_EXECUTION_CLIENT:
            request_profile = _plan_request_profile(plan)
            native_result = _call_native_openai_compatible_once(
                candidate=candidate,
                api_key=api_key,
                prompt=prompt,
                request_profile=request_profile,
                timeout_seconds=int(request_profile["timeout_seconds"]),
            )
            audit = dict(native_result.get("audit") or {})
            content = str(native_result.get("content") or "")
            completed = bool(native_result.get("ok"))
            usage = dict(native_result.get("usage") or {})
            reported_cost = native_result.get("estimated_cost")
        else:
            with SingleRequestAuditProxy(
                upstream_base_url=candidate["base_url"],
                expected_model=candidate["model"],
                api_key=api_key,
            ) as active_proxy:
                proxy = active_proxy
                config_path, state_dir, workspace = _write_openclaw_config(
                    result_path.parent,
                    candidate=candidate,
                    proxy_base_url=active_proxy.base_url,
                )
                command_result = _run_openclaw(
                    resolved_openclaw_command,
                    config_path=config_path,
                    state_dir=state_dir,
                    workspace=workspace,
                    prompt=prompt,
                    timeout_seconds=int(
                        candidate["route"]["deployments"][0].get("timeout_seconds")
                        or 120
                    ),
                    redactions=(api_key,),
                )
                audit = active_proxy.audit_snapshot()
            content = _openclaw_content(command_result.get("json"))
            completed = bool(
                command_result.get("returncode") == 0
                and audit.get("upstream_status")
                and int(audit["upstream_status"]) < 400
                and content
            )
            usage = _openclaw_usage(command_result.get("json"))
            reported_cost = _usage_cost(command_result.get("json"))
        external_attempted = bool(audit.get("external_request_count"))
        assessment = assess_coding_parity_output(content)
        model_identity = _model_identity(candidate, audit)
        tool_name = (
            "VKP native OpenAI-compatible client"
            if execution_client == NATIVE_EXECUTION_CLIENT
            else "OpenClaw"
        )
        result = {
            "schema": EXECUTION_SCHEMA,
            "ok": completed,
            "status": "completed" if completed else "provider_failed",
            "candidate_id": candidate_id,
            "pair_id": candidate["pair_id"],
            "side": candidate["side"],
            "provider": candidate["provider"],
            "requested_model": candidate["model"],
            "execution_client": execution_client,
            "model_identity": model_identity,
            "route": candidate["route"],
            "consent_path": str(consent_path),
            "consent_id": str(reservation.get("consent_id") or ""),
            "upload_manifest": validation.get("upload_manifest") or {},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "content": content,
            "assessment": assessment,
            "audit": audit,
            "openclaw": command_result if execution_client == OPENCLAW_EXECUTION_CLIENT else {},
            "native_client": (
                dict(native_result.get("response") or {})
                if execution_client == NATIVE_EXECUTION_CLIENT
                else {}
            ),
            "usage": usage,
            "estimated_cost": reported_cost,
            "operator_boundary": {
                "tool": tool_name,
                "api": "openai_chat_completions",
                "external_request_count": int(audit.get("external_request_count") or 0),
                "external_retry_count": max(
                    0, int(audit.get("external_request_count") or 0) - 1
                ),
                "fallbacks": [],
                "tool_surface": execution_client == OPENCLAW_EXECUTION_CLIENT,
                "tools_denied": (
                    ["*"] if execution_client == OPENCLAW_EXECUTION_CLIENT else []
                ),
                "delivered_to_channel": False,
                "secrets_persisted": False,
                "request_body_logged": False,
            },
            "updated_at": now_iso(),
        }
        write_json(result_path, result)
        return {**result, "result_path": str(result_path)}
    except Exception as exc:
        if proxy is not None:
            audit = proxy.audit_snapshot()
        external_attempted = bool(audit.get("external_request_count"))
        reported_cost = None if external_attempted else 0.0
        failure = {
            "schema": EXECUTION_SCHEMA,
            "ok": False,
            "status": "runner_failed",
            "candidate_id": candidate_id,
            "pair_id": candidate["pair_id"],
            "side": candidate["side"],
            "provider": candidate["provider"],
            "requested_model": candidate["model"],
            "execution_client": execution_client,
            "model_identity": _model_identity(candidate, audit),
            "route": candidate["route"],
            "consent_path": str(consent_path),
            "consent_id": str(reservation.get("consent_id") or ""),
            "upload_manifest": validation.get("upload_manifest") or {},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "content": "",
            "assessment": assess_coding_parity_output(""),
            "audit": audit,
            "openclaw": command_result if execution_client == OPENCLAW_EXECUTION_CLIENT else {},
            "native_client": (
                dict(native_result.get("response") or {})
                if execution_client == NATIVE_EXECUTION_CLIENT
                else {}
            ),
            "usage": {},
            "estimated_cost": None,
            "error": {
                "type": type(exc).__name__,
                "message": "Provider execution did not complete",
                "details_logged": False,
            },
            "operator_boundary": {
                "tool": (
                    "VKP native OpenAI-compatible client"
                    if execution_client == NATIVE_EXECUTION_CLIENT
                    else "OpenClaw"
                ),
                "api": "openai_chat_completions",
                "external_request_count": int(audit.get("external_request_count") or 0),
                "external_retry_count": 0,
                "fallbacks": [],
                "tool_surface": execution_client == OPENCLAW_EXECUTION_CLIENT,
                "tools_denied": (
                    ["*"] if execution_client == OPENCLAW_EXECUTION_CLIENT else []
                ),
                "delivered_to_channel": False,
                "secrets_persisted": False,
                "request_body_logged": False,
            },
            "updated_at": now_iso(),
        }
        write_json(result_path, failure)
        return {**failure, "result_path": str(result_path)}
    finally:
        record_model_connector_attempt(
            consent_path,
            completed_calls=1 if completed else 0,
            reserved_cost_usd=reserved_cost,
            reported_cost_usd=0.0 if not external_attempted else reported_cost,
            cost_unreported_calls=1 if external_attempted and reported_cost is None else 0,
        )


def execute_coding_tool_provider_parity_consent(
    consent_path: str | Path,
    *,
    expected_route_revision: str = "",
    write: bool = True,
) -> dict[str, Any]:
    """Execute the exact parity candidate locked by a candidate consent.

    This is the narrow Trusted Broker entrypoint for the fixed provider-parity
    suite. The caller cannot supply a provider, model, URL, credential, plan, or
    fallback. Those values are recovered from, and revalidated against, the
    content-addressed plan and consent index beside the candidate consent.
    """

    if not write:
        raise ValueError("provider parity execution requires write=true for audit")
    consent_file = Path(consent_path).expanduser().resolve()
    _require_under_root(consent_file, project_root(), label="consent_path")
    if not consent_file.is_file():
        raise FileNotFoundError(f"parity consent is missing: {consent_file}")

    output_dir = consent_file.parent.parent.resolve()
    plan_file = output_dir / "parity-plan.json"
    index_file = output_dir / "consent-index.json"
    plan = _load_and_validate_plan(plan_file, require_ready=True)
    index = _load_and_validate_consent_index(index_file, plan=plan)
    if Path(str(plan.get("output_dir") or "")).expanduser().resolve() != output_dir:
        raise ValueError("parity consent is not beside its locked plan output")

    candidate_id = consent_file.parent.name
    candidate = _candidate(plan, candidate_id)
    candidate_consent = Path(str(candidate.get("consent_path") or "")).expanduser().resolve()
    if candidate_consent != consent_file:
        raise ValueError("parity candidate consent path differs from the exact plan")
    consent_row = next(
        (
            dict(row)
            for row in index.get("consents") or []
            if str(row.get("candidate_id") or "") == candidate_id
        ),
        None,
    )
    if not consent_row:
        raise ValueError(f"consent not found for candidate: {candidate_id}")
    indexed_consent = Path(str(consent_row.get("consent_path") or "")).expanduser().resolve()
    if indexed_consent != consent_file:
        raise ValueError("parity consent index path differs from the exact consent")

    consent = read_json(consent_file)
    if str(consent.get("task") or "") != "provider_task_benchmark":
        raise ValueError("consent is not for provider_task_benchmark")
    candidate_revision = str(candidate["route"]["route_revision"])
    consent_revision = str((consent.get("route") or {}).get("route_revision") or "")
    indexed_revision = str(consent_row.get("route_revision") or "")
    if consent_revision != candidate_revision or indexed_revision != candidate_revision:
        raise ValueError("parity consent route revision differs from the exact plan")
    if expected_route_revision and expected_route_revision != candidate_revision:
        raise ValueError("parity route revision does not match the caller expectation")

    return execute_coding_tool_provider_parity_candidate(
        plan_file,
        index_file,
        candidate_id=candidate_id,
        operator_confirm_network=True,
    )


def recover_interrupted_coding_tool_provider_parity_candidate(
    plan_path: str | Path,
    consent_index_path: str | Path,
    *,
    candidate_id: str,
    reason: str,
) -> dict[str, Any]:
    """Finalize one already-attempted candidate after the Broker was terminated.

    Recovery never creates a new provider request. It is only valid when the
    candidate has exactly one reserved attempt, no completed call, and no
    terminal execution artifact.
    """

    allowed_reasons = {"wall_clock_timeout_after_broker_restart"}
    if reason not in allowed_reasons:
        raise ValueError(f"unsupported recovery reason: {reason}")
    plan_file = Path(plan_path).expanduser().resolve()
    index_file = Path(consent_index_path).expanduser().resolve()
    plan = _load_and_validate_plan(plan_file, require_ready=True)
    index = _load_and_validate_consent_index(index_file, plan=plan)
    request_profile = _plan_request_profile(plan)
    if str(request_profile.get("id") or "") != CAPABILITY_CEILING_PROFILE:
        raise ValueError("interrupted recovery is limited to capability_ceiling_v1")
    candidate = _candidate(plan, candidate_id)
    result_path = Path(candidate["result_path"]).expanduser().resolve()
    if result_path.is_file():
        existing = read_json(result_path)
        _validate_existing_execution_result(existing, candidate=candidate)
        return {**existing, "result_path": str(result_path), "recovered": False}

    consent_row = next(
        (
            dict(row)
            for row in index.get("consents") or []
            if str(row.get("candidate_id") or "") == candidate_id
        ),
        None,
    )
    if not consent_row:
        raise ValueError(f"consent not found for candidate: {candidate_id}")
    consent_path = Path(consent_row["consent_path"]).expanduser().resolve()
    consent = read_json(consent_path)
    usage = consent.get("usage") if isinstance(consent.get("usage"), dict) else {}
    if (
        int(usage.get("calls_attempted") or 0) != 1
        or int(usage.get("calls_completed") or 0) != 0
        or int(usage.get("cost_unreported_calls") or 0) != 0
    ):
        raise ValueError("consent usage is not an unreconciled single attempt")

    attempted_at = datetime.fromisoformat(str(usage["last_attempt_at"]))
    if attempted_at.tzinfo is None:
        attempted_at = attempted_at.replace(tzinfo=timezone.utc)
    elapsed_ms = max(
        0,
        round(
            (
                datetime.now(timezone.utc) - attempted_at.astimezone(timezone.utc)
            ).total_seconds()
            * 1000
        ),
    )
    timeout_seconds = int(request_profile["timeout_seconds"])
    audit = {
        "external_request_count": 1,
        "blocked_repeat_requests": 0,
        "request_model": candidate["model"],
        "request_profile_id": CAPABILITY_CEILING_PROFILE,
        "request_max_tokens": None,
        "request_max_tokens_omitted": True,
        "request_stream": True,
        "upstream_error": "WallClockTimeout",
        "wall_clock_timeout_seconds": timeout_seconds,
        "wall_clock_timeout_exceeded": True,
        "provider_response_model": "",
    }
    reserved_cost = float(
        (consent.get("scope") or {}).get("max_cost_per_call_usd")
        or request_profile["max_cost_per_candidate_usd"]
    )
    record_model_connector_attempt(
        consent_path,
        completed_calls=0,
        reserved_cost_usd=reserved_cost,
        reported_cost_usd=None,
        cost_unreported_calls=1,
    )
    failure = {
        "schema": EXECUTION_SCHEMA,
        "ok": False,
        "status": "runner_failed",
        "candidate_id": candidate_id,
        "pair_id": candidate["pair_id"],
        "side": candidate["side"],
        "provider": candidate["provider"],
        "requested_model": candidate["model"],
        "execution_client": _plan_execution_client(plan),
        "model_identity": _model_identity(candidate, audit),
        "route": candidate["route"],
        "consent_path": str(consent_path),
        "consent_id": str(consent.get("consent_id") or ""),
        "upload_manifest": consent.get("upload_manifest") or {},
        "latency_ms": elapsed_ms,
        "content": "",
        "assessment": assess_coding_parity_output(""),
        "audit": audit,
        "openclaw": {},
        "native_client": {
            "http_status": 0,
            "error_type": "WallClockTimeout",
        },
        "usage": {},
        "estimated_cost": None,
        "error": {
            "type": "WallClockTimeout",
            "message": "Provider execution exceeded the profile wall-clock timeout",
            "details_logged": False,
            "recovered_after_broker_restart": True,
        },
        "operator_boundary": {
            "tool": "VKP native OpenAI-compatible client",
            "api": "openai_chat_completions",
            "external_request_count": 1,
            "external_retry_count": 0,
            "fallbacks": [],
            "tool_surface": False,
            "tools_denied": [],
            "delivered_to_channel": False,
            "secrets_persisted": False,
            "request_body_logged": False,
        },
        "updated_at": now_iso(),
    }
    write_json(result_path, failure)
    return {**failure, "result_path": str(result_path), "recovered": True}


def execute_all_coding_tool_provider_parity_candidates(
    plan_path: str | Path,
    consent_index_path: str | Path,
    *,
    operator_confirm_network: bool,
    openclaw_command: str = "openclaw",
) -> dict[str, Any]:
    """Run the exact plan sequentially and persist progress after every candidate.

    Existing exact execution artifacts are reused and never called again. A provider
    failure is recorded without changing provider, model, URL, or fallback policy,
    then the next independent candidate is attempted.
    """

    if not operator_confirm_network:
        raise ValueError("--operator-confirm-network is required for real provider calls")
    plan_file = Path(plan_path).expanduser().resolve()
    index_file = Path(consent_index_path).expanduser().resolve()
    plan = _load_and_validate_plan(plan_file, require_ready=True)
    _load_and_validate_consent_index(index_file, plan=plan)
    _revalidate_plan_profiles(plan)
    progress_path = Path(plan["output_dir"]).expanduser().resolve() / "batch-execution.json"
    _require_under_root(progress_path, project_root(), label="batch_progress", must_exist=False)

    rows: list[dict[str, Any]] = []
    attempted_this_run = 0
    reused_existing = 0
    external_requests_this_run = 0

    def save_progress(status: str, *, next_candidate_id: str = "") -> dict[str, Any]:
        payload = {
            "schema": BATCH_EXECUTION_SCHEMA,
            "status": status,
            "plan_path": str(plan_file),
            "plan_sha256": plan["plan_sha256"],
            "consent_index_path": str(index_file),
            "candidate_count": len(plan["candidates"]),
            "completed_candidate_count": len(rows),
            "attempted_this_run": attempted_this_run,
            "reused_existing": reused_existing,
            "external_requests_this_run": external_requests_this_run,
            "successful_count": sum(bool(row.get("ok")) for row in rows),
            "failed_count": sum(not bool(row.get("ok")) for row in rows),
            "next_candidate_id": next_candidate_id,
            "candidates": rows,
            "operator_boundary": {
                "execution_order": "exact_plan_order",
                "max_external_requests_per_candidate": 1,
                "external_retry_count": 0,
                "fallbacks": [],
                "provider_model_or_url_overrides_accepted": False,
                "existing_results_reexecuted": False,
            },
            "updated_at": now_iso(),
        }
        write_json(progress_path, payload)
        return payload

    first_id = str(plan["candidates"][0]["candidate_id"])
    save_progress("running", next_candidate_id=first_id)
    for position, candidate in enumerate(plan["candidates"]):
        candidate_id = str(candidate["candidate_id"])
        result_path = Path(candidate["result_path"]).expanduser().resolve()
        source = "executed_this_run"
        if result_path.is_file():
            result = read_json(result_path)
            _validate_existing_execution_result(result, candidate=candidate)
            result = {**result, "result_path": str(result_path)}
            source = "reused_existing"
            reused_existing += 1
        else:
            attempted_this_run += 1
            try:
                result = execute_coding_tool_provider_parity_candidate(
                    plan_file,
                    index_file,
                    candidate_id=candidate_id,
                    operator_confirm_network=True,
                    openclaw_command=openclaw_command,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                result = {
                    "ok": False,
                    "status": "invalid_or_blocked",
                    "candidate_id": candidate_id,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "remote_requests_made": False,
                }
            external_requests_this_run += int(
                (result.get("audit") or {}).get("external_request_count") or 0
            )
        external_request_count = int(
            (result.get("audit") or {}).get("external_request_count") or 0
        )
        row = {
            "candidate_id": candidate_id,
            "pair_id": candidate["pair_id"],
            "side": candidate["side"],
            "provider": candidate["provider"],
            "requested_model": candidate["model"],
            "source": source,
            "ok": bool(result.get("ok")),
            "status": str(result.get("status") or "unknown"),
            "external_request_count": external_request_count,
            "result_path": str(result.get("result_path") or ""),
            "result_sha256": (
                _file_sha256(result_path) if result_path.is_file() else ""
            ),
        }
        if result.get("error"):
            row["error"] = result["error"]
        rows.append(row)
        next_id = (
            str(plan["candidates"][position + 1]["candidate_id"])
            if position + 1 < len(plan["candidates"])
            else ""
        )
        save_progress("running", next_candidate_id=next_id)

    comparison = compare_coding_tool_provider_parity(plan_file)
    final_status = (
        "completed" if all(bool(row.get("ok")) for row in rows) else "completed_with_failures"
    )
    result = save_progress(final_status)
    result["comparison"] = {
        "status": comparison["status"],
        "ready_pair_count": comparison["ready_pair_count"],
        "incomplete_pair_count": comparison["incomplete_pair_count"],
        "artifacts": comparison["artifacts"],
    }
    result["artifacts"] = {
        "batch_execution": str(progress_path),
        **comparison["artifacts"],
    }
    write_json(progress_path, result)
    return result


def compare_coding_tool_provider_parity(
    plan_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    plan = _load_and_validate_plan(Path(plan_path).expanduser().resolve())
    destination = Path(output_dir or plan["output_dir"]).expanduser().resolve()
    result_rows: dict[str, dict[str, Any]] = {}
    parsed_outputs: dict[str, dict[str, Any] | None] = {}
    for candidate in plan["candidates"]:
        path = Path(candidate["result_path"])
        if not path.is_file():
            result_rows[candidate["candidate_id"]] = {
                "candidate_id": candidate["candidate_id"],
                "loaded": False,
                "result_path": str(path),
            }
            continue
        raw = read_json(path)
        content = str(raw.get("content") or "")
        assessment = assess_coding_parity_output(content)
        parsed = assessment.get("parsed_json")
        parsed_outputs[candidate["candidate_id"]] = parsed
        result_rows[candidate["candidate_id"]] = {
            "candidate_id": candidate["candidate_id"],
            "loaded": True,
            "ok": bool(raw.get("ok")),
            "provider": str(raw.get("provider") or ""),
            "requested_model": str(raw.get("requested_model") or ""),
            "provider_response_model": str(
                (raw.get("model_identity") or {}).get("provider_response_model") or ""
            ),
            "identity_status": str(
                (raw.get("model_identity") or {}).get("status") or ""
            ),
            "latency_ms": raw.get("latency_ms"),
            "finish_reason": str(
                (raw.get("native_client") or {}).get("finish_reason") or ""
            ),
            "reasoning_chars": int(
                (raw.get("native_client") or {}).get("reasoning_chars") or 0
            ),
            "content_chars": len(content),
            "score": assessment["score"],
            "quality_gate_passed": bool(assessment["quality_gate_passed"]),
            "external_request_count": int(
                (raw.get("audit") or {}).get("external_request_count") or 0
            ),
            "content_sha256": _text_sha256(content) if content else "",
            "normalized_json_sha256": (
                _canonical_json_sha256(parsed) if isinstance(parsed, dict) else ""
            ),
            "patch_sha256": (
                _text_sha256(str(parsed.get("patch") or ""))
                if isinstance(parsed, dict) and parsed.get("patch")
                else ""
            ),
            "result_path": str(path),
            "result_sha256": _file_sha256(path),
        }
    pairs: list[dict[str, Any]] = []
    for pair in plan["pairs"]:
        rows = [result_rows[candidate_id] for candidate_id in pair["candidate_ids"]]
        comparable = bool(
            len(rows) == 2
            and all(row.get("loaded") and row.get("ok") for row in rows)
            and all(row.get("external_request_count") == 1 for row in rows)
        )
        output_difference = _pair_output_difference(
            rows,
            parsed_outputs=parsed_outputs,
            comparable=comparable,
        )
        pairs.append(
            {
                "id": pair["id"],
                "family": pair["family"],
                "status": "ready_for_human_review" if comparable else "incomplete",
                "comparable": comparable,
                "candidates": rows,
                "automatic_score_winner": _score_winner(rows) if comparable else "",
                "output_difference": output_difference,
                "human_review_required": True,
            }
        )
    ready = sum(row["comparable"] for row in pairs)
    result = {
        "schema": COMPARISON_SCHEMA,
        "status": "ready_for_human_review" if ready == len(pairs) else "incomplete",
        "plan_path": str(Path(plan_path).expanduser().resolve()),
        "pair_count": len(pairs),
        "ready_pair_count": ready,
        "incomplete_pair_count": len(pairs) - ready,
        "pairs": pairs,
        "limitations": [
            "This is one fixed coding task, not a general model capability ranking.",
            "Provider response model identity is only proven when the upstream response reports it.",
            "Matching family/version labels do not prove identical weights, quantization, system prompts, reasoning defaults, or serving backends.",
            (
                "Vendor-specific reasoning controls are omitted because the exact locked models and the Coding Plan raw endpoint lack one complete official per-model field matrix; provider default reasoning behavior remains part of the measured service difference."
            ),
            "Automatic score is a gate and not a substitute for patch review and test execution.",
        ],
        "operator_boundary": {
            "offline_comparison_only": True,
            "model_calls_made": 0,
            "model_content_copied_into_summary": False,
            "default_routes_changed": False,
        },
        "updated_at": now_iso(),
    }
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "parity-comparison.json"
    markdown_path = destination / "parity-comparison.md"
    write_json(json_path, result)
    markdown_path.write_text(_render_comparison(result), encoding="utf-8")
    return {
        **result,
        "artifacts": {"json": str(json_path), "markdown": str(markdown_path)},
    }


class SingleRequestAuditProxy(AbstractContextManager["SingleRequestAuditProxy"]):
    """Forward at most one HTTPS request and retain only hashes/metadata."""

    def __init__(
        self,
        *,
        upstream_base_url: str,
        expected_model: str,
        api_key: str,
        allow_http_upstream_for_tests: bool = False,
    ) -> None:
        parsed = urllib.parse.urlsplit(upstream_base_url)
        if parsed.scheme != "https" and not (
            allow_http_upstream_for_tests
            and parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError("production parity upstream must use HTTPS")
        if not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("invalid parity upstream base URL")
        self._upstream = parsed
        self._expected_model = expected_model
        self._api_key = api_key
        self._lock = threading.Lock()
        self._forwarded = False
        self._audit: dict[str, Any] = {
            "external_request_count": 0,
            "blocked_repeat_requests": 0,
            "expected_model": expected_model,
            "request_model": "",
            "provider_response_model": "",
            "request_body_sha256": "",
            "response_body_sha256": "",
            "upstream_status": None,
            "upstream_request_ids": {},
            "request_body_logged": False,
            "authorization_logged": False,
        }
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "VKPParityAuditProxy/1.0"

            def do_POST(self) -> None:  # noqa: N802
                owner._handle(self)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        port = int(self._server.server_address[1])
        return f"http://127.0.0.1:{port}/v1"

    def __enter__(self) -> "SingleRequestAuditProxy":
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        self._api_key = ""
        return None

    def audit_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._audit))

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        if urllib.parse.urlsplit(handler.path).path != "/v1/chat/completions":
            return _send_json(handler, HTTPStatus.NOT_FOUND, {"error": "unsupported path"})
        length = int(handler.headers.get("Content-Length") or 0)
        if length < 2 or length > 2 * 1024 * 1024:
            return _send_json(handler, HTTPStatus.BAD_REQUEST, {"error": "invalid body"})
        body = handler.rfile.read(length)
        try:
            request_json = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _send_json(handler, HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
        request_model = str(request_json.get("model") or "")
        if request_model != self._expected_model:
            return _send_json(
                handler,
                HTTPStatus.BAD_REQUEST,
                {"error": "request model differs from the locked candidate"},
            )
        with self._lock:
            if self._forwarded:
                self._audit["blocked_repeat_requests"] += 1
                return _send_json(
                    handler,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "external retry blocked by single-request audit proxy"},
                )
            self._forwarded = True
            self._audit["external_request_count"] = 1
            self._audit["request_model"] = request_model
            self._audit["request_body_sha256"] = hashlib.sha256(body).hexdigest()
        try:
            status, headers, response = self._forward(body, handler.headers)
        except Exception as exc:  # network failure is returned without exposing secrets
            with self._lock:
                self._audit["upstream_error"] = f"{type(exc).__name__}: {exc}"
            return _send_json(
                handler,
                HTTPStatus.BAD_GATEWAY,
                {"error": "upstream request failed"},
            )
        response_model, usage = _response_metadata(response, headers)
        request_ids = {
            key.lower(): value
            for key, value in headers.items()
            if key.lower()
            in {"x-request-id", "x-tt-logid", "request-id", "trace-id"}
        }
        with self._lock:
            self._audit.update(
                {
                    "upstream_status": status,
                    "provider_response_model": response_model,
                    "response_body_sha256": hashlib.sha256(response).hexdigest(),
                    "upstream_request_ids": request_ids,
                    "usage": usage,
                }
            )
        handler.send_response(status)
        handler.send_header("Content-Type", headers.get("content-type", "application/json"))
        handler.send_header("Content-Length", str(len(response)))
        for key, value in request_ids.items():
            handler.send_header(key, value)
        handler.end_headers()
        handler.wfile.write(response)

    def _forward(
        self, body: bytes, request_headers: Any
    ) -> tuple[int, dict[str, str], bytes]:
        base_path = self._upstream.path.rstrip("/")
        path = f"{base_path}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": str(request_headers.get("Accept") or "application/json"),
            "User-Agent": "VKP-OpenClaw-Parity/1.0",
        }
        port = self._upstream.port or (443 if self._upstream.scheme == "https" else 80)
        connection_type = (
            http.client.HTTPSConnection
            if self._upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(self._upstream.hostname, port, timeout=180)
        try:
            connection.request("POST", path, body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
            response_headers = {key.lower(): value for key, value in response.getheaders()}
            return int(response.status), response_headers, response_body
        finally:
            connection.close()


def assess_coding_parity_output(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    think_leak = bool(re.search(r"<think>|</think>", text, re.IGNORECASE))
    forbidden_marker = think_leak or "```" in text
    parsed = _parse_json_content(text)
    checks = {
        "non_empty": bool(text),
        "no_think_leak": not think_leak,
        "no_forbidden_markers": not forbidden_marker,
        "json_object": isinstance(parsed, dict),
        "output_contract_shape": False,
        "race_diagnosed": False,
        "atomic_patch": False,
        "transaction_cleanup": False,
        "success_path_persisted": False,
        "multiprocess_test": False,
        "preserves_signature_and_schema": False,
    }
    if isinstance(parsed, dict):
        joined = json.dumps(parsed, ensure_ascii=False).casefold()
        patch = str(parsed.get("patch") or "")
        tests = json.dumps(parsed.get("tests") or [], ensure_ascii=False).casefold()
        checks["output_contract_shape"] = bool(
            all(
                isinstance(parsed.get(key), str)
                and str(parsed.get(key) or "").strip()
                for key in ("bug_class", "explanation", "patch")
            )
            and all(
                isinstance(parsed.get(key), list)
                and bool(parsed.get(key))
                and all(
                    isinstance(item, str) and item.strip()
                    for item in parsed.get(key) or []
                )
                for key in ("tests", "tradeoffs")
            )
        )
        checks["race_diagnosed"] = any(
            marker in joined
            for marker in ("race", "check-then-act", "竞态", "并发")
        )
        checks["atomic_patch"] = any(
            marker in patch.casefold()
            for marker in ("begin immediate", "on conflict", "changes()", "where used <")
        )
        effective_patch = "\n".join(
            line
            for line in patch.splitlines()
            if not line.startswith("-") and not line.startswith("+++")
        )
        effective_lower = effective_patch.casefold()
        compact_patch = re.sub(r"\s+", "", effective_lower)
        writes_data = bool(
            re.search(r"\b(?:insert\s+into|update\s+quotas)\b", effective_lower)
        )
        autocommit = "isolation_level=none" in compact_patch
        commits = any(
            marker in compact_patch
            for marker in (".commit(", "execute(\"commit\")", "execute('commit')")
        )
        checks["success_path_persisted"] = not writes_data or autocommit or commits
        checks["transaction_cleanup"] = any(
            marker in joined for marker in ("rollback", "transaction", "回滚", "事务")
        )
        checks["multiprocess_test"] = any(
            marker in tests
            for marker in ("multiprocess", "process", "进程", "concurrent")
        ) and any(marker in tests for marker in ("limit", "最多", "sum"))
        signature = (
            "def reserve_quota(db_path: Path, user_id: str, limit: int) -> bool:"
        )
        schema_changed = any(
            line.startswith(("+", "-")) and "create table" in line.casefold()
            for line in patch.splitlines()
        )
        explicit_preservation = (
            any(marker in joined for marker in ("signature", "签名"))
            and any(marker in joined for marker in ("schema", "模式"))
        )
        checks["preserves_signature_and_schema"] = (
            (signature in patch or explicit_preservation) and not schema_changed
        )
    weights = {
        "non_empty": 5,
        "no_forbidden_markers": 10,
        "json_object": 5,
        "output_contract_shape": 10,
        "race_diagnosed": 10,
        "atomic_patch": 15,
        "transaction_cleanup": 10,
        "success_path_persisted": 20,
        "multiprocess_test": 10,
        "preserves_signature_and_schema": 5,
    }
    score = sum(weight for key, weight in weights.items() if checks[key])
    return {
        "score": score,
        "max_score": sum(weights.values()),
        "quality_gate_passed": bool(
            checks["json_object"]
            and checks["race_diagnosed"]
            and checks["atomic_patch"]
            and checks["success_path_persisted"]
            and checks["multiprocess_test"]
            and checks["no_forbidden_markers"]
            and checks["output_contract_shape"]
        ),
        "checks": checks,
        "parsed_json": parsed if isinstance(parsed, dict) else None,
        "human_patch_review_required": True,
    }


def _route_snapshot(
    candidate_id: str,
    profile: dict[str, Any],
    *,
    request_profile_id: str = COMMON_FIELDS_PROFILE,
) -> dict[str, Any]:
    request_profile = _request_profile_definition(request_profile_id)
    provider_options = dict(profile.get("provider_options") or {})
    required_provider_options = list(profile.get("required_provider_options") or [])
    timeout_seconds = int(profile.get("timeout_seconds") or 120)
    if request_profile_id in {
        CONTENT_QUALITY_PROFILE,
        CAPABILITY_CEILING_PROFILE,
    }:
        provider_options.update(
            {
                "stream": bool(request_profile["stream"]),
                "temperature": 0,
            }
        )
        max_tokens = _request_max_tokens(request_profile)
        if max_tokens is not None:
            provider_options["max_tokens"] = max_tokens
        required_provider_options = sorted(provider_options)
        timeout_seconds = int(request_profile["timeout_seconds"])
    deployment = {
        "id": str(profile["id"]),
        "provider": str(profile["provider"]),
        "litellm_provider": str(profile.get("litellm_provider") or ""),
        "model": str(profile["model"]),
        "base_url": str(profile["base_url"]),
        "interface": "openai_chat_completions",
        "auth_mode": str(profile.get("auth_mode") or "api_key_dpapi"),
        "api_key_optional": bool(profile.get("api_key_optional")),
        "provider_options": provider_options,
        "required_provider_options": required_provider_options,
        "environment_bindings": list(profile.get("environment_bindings") or []),
        "adapter_backend": str(profile.get("adapter_backend") or "proxy"),
        "timeout_seconds": timeout_seconds,
    }
    host = urllib.parse.urlsplit(str(profile["base_url"])).hostname
    return _normalise_route_snapshot(
        {
            "route_id": (
                f"coding-tool-parity-{request_profile_id}-{candidate_id}"
                if request_profile_id == CAPABILITY_CEILING_PROFILE
                else f"coding-tool-parity-{candidate_id}"
            ),
            "execution_location": "remote",
            "deployments": [deployment],
            "destinations": [f"https://{host}"],
        }
    )


def _profile_mismatches(
    profile: dict[str, Any], *, provider: str, base_url: str, model: str
) -> list[str]:
    mismatches: list[str] = []
    expected = {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "location": "remote",
    }
    for key, value in expected.items():
        if str(profile.get(key) or "") != value:
            mismatches.append(f"{key}={profile.get(key)!r}, expected={value!r}")
    if "text" not in profile.get("capabilities", []):
        mismatches.append("capabilities must include text")
    if not bool(profile.get("enabled")):
        mismatches.append("profile must be enabled")
    if profile.get("provider_options"):
        mismatches.append(
            "parity profile provider_options must be empty so both sides receive the same common fields"
        )
    return mismatches


def _load_and_validate_plan(path: Path, *, require_ready: bool = False) -> dict[str, Any]:
    plan = read_json(path)
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid coding provider parity plan")
    expected_hash = str(plan.get("plan_sha256") or "")
    if expected_hash != _payload_sha256(plan):
        raise ValueError("parity plan hash mismatch")
    if require_ready and plan.get("status") != "ready_for_operator_consent":
        raise ValueError("parity plan is not ready for operator consent")
    if int(plan.get("candidate_count") or 0) != 10:
        raise ValueError("parity plan must contain exactly ten candidates")
    if set(plan.get("upload_manifest", {}).get("destinations") or []) != set(
        EXPECTED_DESTINATIONS
    ):
        raise ValueError("parity plan destinations differ from the approved pair")
    _plan_request_profile(plan)
    _validate_artifact_record(plan["artifact"])
    return plan


def _load_and_validate_consent_index(
    path: Path, *, plan: dict[str, Any]
) -> dict[str, Any]:
    index = read_json(path)
    if not isinstance(index, dict) or index.get("schema") != CONSENT_INDEX_SCHEMA:
        raise ValueError("invalid parity consent index")
    if str(index.get("plan_sha256") or "") != str(plan["plan_sha256"]):
        raise ValueError("consent index is bound to another parity plan")
    expected_ids = {str(row["candidate_id"]) for row in plan["candidates"]}
    actual_ids = {
        str(row.get("candidate_id") or "") for row in index.get("consents") or []
    }
    if actual_ids != expected_ids or int(index.get("consent_count") or 0) != len(
        expected_ids
    ):
        raise ValueError("consent index candidate set differs from the exact plan")
    return index


def _validate_existing_execution_result(
    result: dict[str, Any], *, candidate: dict[str, Any]
) -> None:
    if not isinstance(result, dict) or result.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("existing parity execution has an invalid schema")
    expected = {
        "candidate_id": candidate["candidate_id"],
        "provider": candidate["provider"],
        "requested_model": candidate["model"],
    }
    for key, value in expected.items():
        if str(result.get(key) or "") != str(value):
            raise ValueError(f"existing parity execution changed {key}")
    result_route = result.get("route") or {}
    if str(result_route.get("route_revision") or "") != str(
        candidate["route"]["route_revision"]
    ):
        raise ValueError("existing parity execution route revision mismatch")
    audit = result.get("audit") or {}
    if int(audit.get("external_request_count") or 0) > 1:
        raise ValueError("existing parity execution exceeded the one-request contract")
    request_model = str(audit.get("request_model") or "")
    if request_model and request_model != str(candidate["model"]):
        raise ValueError("existing parity execution request model mismatch")


def _revalidate_plan_profiles(
    plan: dict[str, Any], *, candidate_ids: set[str] | None = None
) -> None:
    settings = load_model_api_settings(plan["settings_path"])
    profiles = {str(row["id"]): dict(row) for row in settings["profiles"]}
    secret_ids = set(
        _load_secret_document(Path(plan["secrets_path"])).get("items", {})
    )
    for candidate in plan["candidates"]:
        if candidate_ids and candidate["candidate_id"] not in candidate_ids:
            continue
        profile = profiles.get(candidate["profile_id"])
        if not profile:
            raise ValueError(f"profile disappeared: {candidate['profile_id']}")
        mismatches = _profile_mismatches(
            profile,
            provider=candidate["provider"],
            base_url=candidate["base_url"],
            model=candidate["model"],
        )
        if mismatches:
            raise ValueError(
                f"profile changed after plan creation: {candidate['profile_id']}: "
                + ", ".join(mismatches)
            )
        if _profile_secret_id(profile) not in secret_ids:
            raise ValueError(f"credential missing: {candidate['profile_id']}")
        current_route = _route_snapshot(
            candidate["candidate_id"],
            profile,
            request_profile_id=str(
                (plan.get("comparison_contract") or {}).get("request_profile_id")
                or COMMON_FIELDS_PROFILE
            ),
        )
        if current_route != candidate["route"]:
            raise ValueError(f"route changed after plan creation: {candidate['candidate_id']}")


def _plan_execution_client(plan: dict[str, Any]) -> str:
    contract = (
        plan.get("comparison_contract")
        if isinstance(plan.get("comparison_contract"), dict)
        else {}
    )
    value = str(contract.get("execution_client") or "").strip()
    if not value:
        value = (
            OPENCLAW_EXECUTION_CLIENT
            if str(contract.get("same_tool") or "") == "OpenClaw"
            else NATIVE_EXECUTION_CLIENT
        )
    if value not in {NATIVE_EXECUTION_CLIENT, OPENCLAW_EXECUTION_CLIENT}:
        raise ValueError(f"unsupported parity execution client in plan: {value!r}")
    return value


def _request_profile_definition(value: str) -> dict[str, Any]:
    profile_id = str(value or "").strip()
    profile = REQUEST_PROFILES.get(profile_id)
    if not profile:
        raise ValueError(f"unsupported parity request profile: {value!r}")
    return dict(profile)


def _request_max_tokens(profile: dict[str, Any]) -> int | None:
    value = profile.get("max_tokens")
    return None if value is None else max(1, int(value))


def _plan_request_profile(plan: dict[str, Any]) -> dict[str, Any]:
    contract = (
        plan.get("comparison_contract")
        if isinstance(plan.get("comparison_contract"), dict)
        else {}
    )
    explicit_profile_id = str(contract.get("request_profile_id") or "").strip()
    profile = _request_profile_definition(explicit_profile_id or COMMON_FIELDS_PROFILE)
    if not explicit_profile_id:
        if int(contract.get("max_tokens") or 0) != 1024:
            raise ValueError("legacy parity request profile changed max_tokens")
        return profile
    expected = {
        "max_tokens": _request_max_tokens(profile),
        "streaming": bool(profile["stream"]),
        "timeout_seconds": int(profile["timeout_seconds"]),
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"parity request profile changed {key}")
    return profile


def _call_native_openai_compatible_once(
    *,
    candidate: dict[str, Any],
    api_key: str,
    prompt: str,
    timeout_seconds: int,
    request_profile: dict[str, Any] | None = None,
    allow_http_upstream_for_tests: bool = False,
) -> dict[str, Any]:
    """Make one exact-model request without redirects, retries, or tool runtime."""

    locked_request = dict(
        request_profile or _request_profile_definition(COMMON_FIELDS_PROFILE)
    )
    profile_id = str(locked_request.get("id") or "")
    locked_request = _request_profile_definition(profile_id)
    config = {"base_url": candidate["base_url"], "model": candidate["model"]}
    body = build_openai_compatible_text_body(
        cfg=config,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format=None,
        max_tokens=_request_max_tokens(locked_request),
    )
    body["stream"] = bool(locked_request["stream"])
    if str(body.get("model") or "") != str(candidate["model"]):
        raise ValueError("native request model differs from the locked candidate")
    url = urllib.parse.urlsplit(openai_compatible_chat_completions_url(config))
    if url.scheme != "https" and not (
        allow_http_upstream_for_tests and url.scheme == "http"
    ):
        raise ValueError("native parity upstream must use HTTPS")
    if not url.hostname or str(url.hostname) != str(candidate["destination"]):
        raise ValueError("native parity destination differs from the locked candidate")
    if url.query or url.fragment or url.username or url.password:
        raise ValueError("native parity endpoint cannot include query, fragment, or userinfo")
    request_body = json.dumps(
        body, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    audit: dict[str, Any] = {
        "external_request_count": 0,
        "blocked_repeat_requests": 0,
        "request_model": str(body["model"]),
        "request_profile_id": profile_id,
        "request_max_tokens": body.get("max_tokens"),
        "request_max_tokens_omitted": "max_tokens" not in body,
        "request_stream": bool(body["stream"]),
        "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
        "request_body_logged": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "VKP-Native-Provider-Parity/1.0",
    }
    port = url.port or (443 if url.scheme == "https" else 80)
    connection_type = (
        http.client.HTTPSConnection
        if url.scheme == "https"
        else http.client.HTTPConnection
    )
    wall_clock_timeout_seconds = max(1, int(timeout_seconds))
    deadline = time.monotonic() + wall_clock_timeout_seconds
    connection = connection_type(
        url.hostname,
        port,
        timeout=wall_clock_timeout_seconds,
    )
    response_body = b""
    response_headers: dict[str, str] = {}
    status = 0
    try:
        audit["external_request_count"] = 1
        connection.request("POST", url.path, body=request_body, headers=headers)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "native provider request exceeded wall-clock timeout"
            )
        if connection.sock is not None:
            connection.sock.settimeout(max(0.1, remaining))
        response = connection.getresponse()
        status = int(response.status)
        response_headers = {
            key.lower(): value for key, value in response.getheaders()
        }
        chunks: list[bytes] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "native provider request exceeded wall-clock timeout"
                )
            if connection.sock is not None:
                connection.sock.settimeout(max(0.1, remaining))
            chunk = response.read1(64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        response_body = b"".join(chunks)
    except Exception as exc:
        audit["upstream_error"] = type(exc).__name__
        audit["wall_clock_timeout_seconds"] = wall_clock_timeout_seconds
        audit["wall_clock_timeout_exceeded"] = isinstance(exc, TimeoutError)
        return {
            "ok": False,
            "status": "provider_transport_failed",
            "content": "",
            "usage": {},
            "estimated_cost": None,
            "audit": audit,
            "response": {"http_status": 0, "error_type": type(exc).__name__},
        }
    finally:
        connection.close()

    response_model, usage = _response_metadata(response_body, response_headers)
    request_ids = {
        key: value
        for key, value in response_headers.items()
        if key in {"x-request-id", "x-tt-logid", "request-id", "trace-id"}
    }
    content, response_detail = _native_response_content(
        response_body, response_headers
    )
    audit.update(
        {
            "upstream_status": status,
            "provider_response_model": response_model,
            "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
            "upstream_request_ids": request_ids,
            "usage": usage,
        }
    )
    ok = bool(200 <= status < 300 and content)
    return {
        "ok": ok,
        "status": "completed" if ok else "provider_failed",
        "content": content,
        "usage": usage,
        "estimated_cost": None,
        "audit": audit,
        "response": {
            "http_status": status,
            "content_type": str(response_headers.get("content-type") or ""),
            "request_ids": request_ids,
            **response_detail,
        },
    }


def _native_response_content(
    body: bytes, headers: dict[str, str]
) -> tuple[str, dict[str, Any]]:
    text = body.decode("utf-8", errors="replace")
    content_type = str(headers.get("content-type") or "").lower()
    content_parts: list[str] = []
    finish_reason = ""
    reasoning_chars = 0
    objects: list[dict[str, Any]] = []
    if "text/event-stream" in content_type or text.lstrip().startswith("data:"):
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            value_text = line.removeprefix("data:").strip()
            if not value_text or value_text == "[DONE]":
                continue
            try:
                value = json.loads(value_text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                objects.append(value)
    else:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            objects.append(value)
    for value in objects:
        choices = value.get("choices") if isinstance(value.get("choices"), list) else []
        for choice in choices[:1]:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            payload = message or delta
            part = payload.get("content")
            if isinstance(part, str) and part:
                content_parts.append(part)
            reasoning_chars += len(str(payload.get("reasoning_content") or ""))
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
    content = "".join(content_parts).strip()
    return content, {
        "finish_reason": finish_reason,
        "reasoning_chars": reasoning_chars,
        "empty_content": not bool(content),
    }


def _write_openclaw_config(
    root: Path, *, candidate: dict[str, Any], proxy_base_url: str
) -> tuple[Path, Path, Path]:
    state_dir = root / "openclaw-state"
    workspace = root / "openclaw-workspace"
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    provider_id = "vkp-parity-provider"
    model_ref = f"{provider_id}/{candidate['model']}"
    config = {
        "models": {
            "mode": "replace",
            "providers": {
                provider_id: {
                    "baseUrl": proxy_base_url,
                    "apiKey": {
                        "source": "env",
                        "provider": "default",
                        "id": "VKP_PARITY_PROXY_API_KEY",
                    },
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": candidate["model"],
                            "name": candidate["model"],
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                            "contextWindow": 32768,
                            "maxTokens": 1024,
                            "compat": {
                                "supportsStore": False,
                                "supportsDeveloperRole": False,
                                "supportsReasoningEffort": False,
                                "supportsUsageInStreaming": False,
                                "maxTokensField": "max_tokens",
                            },
                        }
                    ],
                }
            },
        },
        "agents": {
            "defaults": {
                "workspace": str(workspace),
                "skipBootstrap": True,
                "model": {"primary": model_ref, "fallbacks": []},
                "models": {
                    model_ref: {
                        "params": {"temperature": 0, "maxTokens": 1024}
                    }
                },
                "thinkingDefault": "off",
                "verboseDefault": "off",
                "timeoutSeconds": 180,
                "maxConcurrent": 1,
            }
        },
        "tools": {"deny": ["*"]},
    }
    config_path = root / "openclaw.json"
    write_json(config_path, config)
    return config_path, state_dir, workspace


def _run_openclaw(
    command: str,
    *,
    config_path: Path,
    state_dir: Path,
    workspace: Path,
    prompt: str,
    timeout_seconds: int,
    redactions: tuple[str, ...],
) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "OPENCLAW_CONFIG_PATH": str(config_path),
            "OPENCLAW_STATE_DIR": str(state_dir),
            "OPENCLAW_AGENT_DIR": str(state_dir / "agent"),
            "VKP_PARITY_PROXY_API_KEY": "local-audit-proxy",
        }
    )
    validate = subprocess.run(
        [command, "config", "validate", "--json"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if validate.returncode != 0:
        return {
            "returncode": validate.returncode,
            "phase": "config_validation",
            "stdout": _redact(validate.stdout, redactions),
            "stderr": _redact(validate.stderr, redactions),
            "json": _json_from_stdout(validate.stdout),
        }
    session_id = f"vkp-parity-{uuid4().hex}"
    completed = subprocess.run(
        [
            command,
            "agent",
            "--local",
            "--json",
            "--thinking",
            "off",
            "--timeout",
            str(max(1, timeout_seconds)),
            "--session-id",
            session_id,
            "--message",
            prompt,
        ],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(60, timeout_seconds + 30),
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "phase": "agent",
        "stdout": _redact(completed.stdout, redactions),
        "stderr": _redact(completed.stderr, redactions),
        "json": _json_from_stdout(completed.stdout),
        "session_id": session_id,
        "delivery_requested": False,
    }


def _resolve_openclaw_command(command: str) -> str:
    """Resolve the Windows npm shim before consent reservation or network access."""

    clean = str(command or "").strip()
    if not clean:
        raise ValueError("OpenClaw command is required")
    explicit = Path(clean).expanduser()
    if explicit.is_absolute():
        resolved = explicit.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"OpenClaw command does not exist: {resolved}")
        return str(resolved)
    discovered = shutil.which(clean)
    if not discovered:
        raise FileNotFoundError(f"OpenClaw command was not found on PATH: {clean}")
    return str(Path(discovered).resolve())


def _benchmark_prompt(plan: dict[str, Any]) -> str:
    artifact = _validate_artifact_record(plan["artifact"])
    body = artifact.read_text(encoding="utf-8")
    return (
        f"{plan['instructions']}\n\n"
        f"ARTIFACT_SHA256: {plan['artifact']['sha256']}\n"
        f"ARTIFACT_BEGIN\n{body}\nARTIFACT_END"
    )


def _openclaw_content(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    result = value.get("result") if isinstance(value.get("result"), dict) else value
    payloads = result.get("payloads") if isinstance(result.get("payloads"), list) else []
    texts = [str(row.get("text") or "") for row in payloads if isinstance(row, dict)]
    if any(texts):
        return "\n".join(text for text in texts if text).strip()
    for key in ("text", "content", "message", "reply", "output"):
        if isinstance(result.get(key), str):
            return str(result[key]).strip()
    return ""


def _openclaw_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = value.get("result") if isinstance(value.get("result"), dict) else value
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
    usage = agent_meta.get("usage") if isinstance(agent_meta.get("usage"), dict) else {}
    return dict(usage)


def _usage_cost(value: Any) -> float | None:
    usage = _openclaw_usage(value)
    cost = usage.get("cost")
    if isinstance(cost, dict):
        cost = cost.get("total")
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None


def _model_identity(candidate: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    requested = str(candidate["model"])
    request_model = str(audit.get("request_model") or "")
    response_model = str(audit.get("provider_response_model") or "")
    request_locked = request_model == requested
    if response_model:
        status = "verified_exact" if response_model == requested else "response_model_differs"
    else:
        status = "request_locked_response_unreported"
    return {
        "status": status,
        "configured_model": requested,
        "proxied_request_model": request_model,
        "provider_response_model": response_model,
        "request_model_locked": request_locked,
        "provider_response_model_reported": bool(response_model),
        "actual_model_not_inferred_when_unreported": True,
    }


def _response_metadata(body: bytes, headers: dict[str, str]) -> tuple[str, dict[str, Any]]:
    content_type = str(headers.get("content-type") or "").lower()
    objects: list[dict[str, Any]] = []
    text = body.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type or text.lstrip().startswith("data:"):
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                objects.append(value)
    else:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            objects.append(value)
    model = next((str(row["model"]) for row in objects if row.get("model")), "")
    usage = next((dict(row["usage"]) for row in reversed(objects) if isinstance(row.get("usage"), dict)), {})
    return model, usage


def _json_from_stdout(stdout: str) -> dict[str, Any] | None:
    try:
        value = extract_last_json_document(stdout, require_object=True)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _parse_json_content(content: str) -> dict[str, Any] | None:
    try:
        value = extract_json_document(content, require_object=True)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "media_type": "text/plain",
    }


def _validate_artifact_record(row: dict[str, Any]) -> Path:
    if not isinstance(row, dict):
        raise ValueError("invalid parity artifact record")
    path = Path(str(row.get("path") or "")).expanduser().resolve()
    _require_under_root(path, project_root(), label="artifact")
    if not path.is_file():
        raise FileNotFoundError(f"parity artifact is missing: {path}")
    if path.stat().st_size != int(row.get("size_bytes") or -1):
        raise ValueError("parity artifact size changed")
    if _file_sha256(path) != str(row.get("sha256") or ""):
        raise ValueError("parity artifact SHA-256 changed")
    return path


def _candidate(plan: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    row = next(
        (
            dict(value)
            for value in plan["candidates"]
            if str(value.get("candidate_id") or "") == candidate_id
        ),
        None,
    )
    if not row:
        raise ValueError(f"unknown parity candidate: {candidate_id}")
    return row


def _blocked_execution(
    candidate: dict[str, Any], consent_path: Path, status: str, evidence: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": EXECUTION_SCHEMA,
        "ok": False,
        "status": status,
        "candidate_id": candidate["candidate_id"],
        "consent_path": str(consent_path),
        "evidence": evidence,
        "remote_requests_made": False,
        "updated_at": now_iso(),
    }


def _score_winner(rows: list[dict[str, Any]]) -> str:
    ranked = sorted(
        rows,
        key=lambda row: (
            -int(row.get("score") or 0),
            float(row.get("latency_ms") or float("inf")),
        ),
    )
    return str(ranked[0]["candidate_id"]) if ranked else ""


def _pair_output_difference(
    rows: list[dict[str, Any]],
    *,
    parsed_outputs: dict[str, dict[str, Any] | None],
    comparable: bool,
) -> dict[str, Any]:
    if not comparable or len(rows) != 2:
        return {
            "classification": "not_comparable",
            "exact_content_match": False,
            "normalized_json_match": False,
            "patch_exact_match": False,
            "field_similarity": {},
            "mean_field_similarity": None,
        }
    left, right = rows
    left_json = parsed_outputs.get(str(left["candidate_id"]))
    right_json = parsed_outputs.get(str(right["candidate_id"]))
    fields = ("bug_class", "explanation", "patch", "tests", "tradeoffs")
    similarities: dict[str, float] = {}
    if isinstance(left_json, dict) and isinstance(right_json, dict):
        for field in fields:
            left_value = _canonical_field_text(left_json.get(field))
            right_value = _canonical_field_text(right_json.get(field))
            similarities[field] = round(
                difflib.SequenceMatcher(None, left_value, right_value).ratio(), 4
            )
    exact_content_match = bool(
        left.get("content_sha256")
        and left.get("content_sha256") == right.get("content_sha256")
    )
    normalized_json_match = bool(
        left.get("normalized_json_sha256")
        and left.get("normalized_json_sha256")
        == right.get("normalized_json_sha256")
    )
    patch_exact_match = bool(
        left.get("patch_sha256")
        and left.get("patch_sha256") == right.get("patch_sha256")
    )
    if exact_content_match:
        classification = "exact_output_match"
    elif normalized_json_match:
        classification = "normalized_json_match"
    elif patch_exact_match:
        classification = "same_patch_different_explanation"
    else:
        classification = "different_patch_or_unstructured_output"
    return {
        "classification": classification,
        "exact_content_match": exact_content_match,
        "normalized_json_match": normalized_json_match,
        "patch_exact_match": patch_exact_match,
        "field_similarity": similarities,
        "mean_field_similarity": (
            round(sum(similarities.values()) / len(similarities), 4)
            if similarities
            else None
        ),
        "same_output_does_not_prove_same_weights": True,
    }


def _canonical_field_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return str(value or "").strip()


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    return canonical_json_sha256(value)


def _payload_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "plan_sha256"}
    return canonical_json_sha256(clean)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_under_root(
    path: Path, root: Path, *, label: str, must_exist: bool = True
) -> None:
    resolved = path.expanduser().resolve()
    root_resolved = root.expanduser().resolve()
    if resolved != root_resolved and not resolved.is_relative_to(root_resolved):
        raise ValueError(f"{label} must stay under the VKP project root: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")


def _send_json(
    handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _redact(value: str, secrets: tuple[str, ...]) -> str:
    text = str(value or "")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def _render_authorization_request(plan: dict[str, Any]) -> str:
    artifact = plan["artifact"]
    manifest = plan["upload_manifest"]
    client = _plan_execution_client(plan)
    client_label = (
        "VKP 原生 OpenAI-compatible 单请求客户端"
        if client == NATIVE_EXECUTION_CLIENT
        else "OpenClaw openai-completions"
    )
    contract = plan["comparison_contract"]
    profile_id = str(contract.get("request_profile_id") or COMMON_FIELDS_PROFILE)
    title = (
        "Coding Plan 与 SiliconFlow 同模型能力上限配对：操作者授权请求"
        if profile_id == CAPABILITY_CEILING_PROFILE
        else "Coding Plan 与 SiliconFlow 同模型内容质量配对：操作者授权请求"
        if profile_id == CONTENT_QUALITY_PROFILE
        else "Coding Plan 与 SiliconFlow 同模型对比：操作者授权请求"
    )
    max_tokens_label = (
        "provider_managed（请求不发送 max_tokens）"
        if contract.get("max_tokens") is None
        else str(contract["max_tokens"])
    )
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- 文件：`{artifact['path']}`",
            f"- 大小：`{artifact['size_bytes']}` 字节",
            f"- SHA-256：`{artifact['sha256']}`",
            f"- 目的地：`{', '.join(manifest['destinations'])}`",
            f"- 候选：`{manifest['candidate_count']}` 个，每个最多一次",
            f"- 总调用上限：`{manifest['max_calls']}`",
            f"- 总预留费用上限：`${manifest['max_total_cost_usd']:.2f}`",
            f"- 请求档：`{profile_id}`；`max_tokens={max_tokens_label}`；"
            f"`stream={str(bool(contract['streaming'])).lower()}`；"
            f"`timeout={contract['timeout_seconds']}s`",
            "- 思考控制：不发送未被精确模型官方支持矩阵确认的供应商专属字段；"
            "只把最终 content 作为答案，reasoning_content 仅记录长度",
            "- 费用上限只用于 consent 防失控 reservation，不作为请求参数，"
            "不会截断模型输出",
            "- 重试：`0`；fallback：`[]`；原生模式没有工具调用面",
            "",
            "如同意，请在聊天中回复：",
            "",
            (
                "我明确同意将 atomic-quota-fixture.txt（"
                f"{artifact['size_bytes']} 字节，SHA-256 {artifact['sha256']}）发送至 "
                "ark.cn-beijing.volces.com 和 api.siliconflow.cn，通过 "
                f"{client_label}对五组同系列模型执行共 10 次固定代码任务对比；"
                "每个候选最多 1 次、零外部重试、无 fallback，总预留费用上限 "
                f"{manifest['max_total_cost_usd']:.2f} 美元。我接受两家供应商的数据处理边界。"
            ),
            "",
        ]
    )


def _render_comparison(result: dict[str, Any]) -> str:
    lines = [
        "# Coding Plan 与 SiliconFlow 同模型对比",
        "",
        f"- 状态：`{result['status']}`",
        f"- 可比较：`{result['ready_pair_count']}/{result['pair_count']}`",
        "",
        "| 同系列模型 | Ark 状态/分数/延迟 | SiliconFlow 状态/分数/延迟 | 输出差异 | 自动门赢家 |",
        "|---|---|---|---|---|",
    ]
    for pair in result["pairs"]:
        values = {str(row.get("candidate_id") or "").split("--")[-1]: row for row in pair["candidates"]}
        ark = values.get("ark", {})
        sf = values.get("siliconflow", {})
        def cell(row: dict[str, Any]) -> str:
            if not row.get("loaded"):
                return "未执行"
            return f"{row.get('identity_status') or 'unknown'} / {row.get('score')} / {row.get('latency_ms')}ms"
        lines.append(
            f"| {pair['family']} | {cell(ark)} | {cell(sf)} | "
            f"{pair['output_difference']['classification']} / "
            f"{pair['output_difference']['mean_field_similarity'] if pair['output_difference']['mean_field_similarity'] is not None else '—'} | "
            f"{pair['automatic_score_winner'] or '—'} |"
        )
    lines.extend(
        [
            "",
            "自动分数和相似度只做质量门；即使输出完全相同，也不能据此证明两家运行的是同一权重或同一后端。最终仍需人工审查补丁正确性和真实测试。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare identical model families on Ark Coding Plan and SiliconFlow through a locked OpenAI-compatible client."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--settings-path", default="")
    prepare.add_argument("--secrets-path", default="")
    prepare.add_argument("--artifact-path", default="")
    prepare.add_argument("--output-dir", default="")
    prepare.add_argument(
        "--execution-client",
        choices=(NATIVE_EXECUTION_CLIENT, OPENCLAW_EXECUTION_CLIENT),
        default=NATIVE_EXECUTION_CLIENT,
    )
    prepare.add_argument(
        "--request-profile",
        choices=tuple(REQUEST_PROFILES),
        default=COMMON_FIELDS_PROFILE,
    )
    consents = sub.add_parser("create-consents")
    consents.add_argument("plan_path")
    consents.add_argument("--confirm-data-export", action="store_true")
    execute = sub.add_parser("execute")
    execute.add_argument("plan_path")
    execute.add_argument("consent_index_path")
    execute.add_argument("--candidate-id", required=True)
    execute.add_argument("--operator-confirm-network", action="store_true")
    execute.add_argument("--openclaw-command", default="openclaw")
    execute_all = sub.add_parser("execute-all")
    execute_all.add_argument("plan_path")
    execute_all.add_argument("consent_index_path")
    execute_all.add_argument("--operator-confirm-network", action="store_true")
    execute_all.add_argument("--openclaw-command", default="openclaw")
    recover = sub.add_parser("recover-interrupted")
    recover.add_argument("plan_path")
    recover.add_argument("consent_index_path")
    recover.add_argument("--candidate-id", required=True)
    recover.add_argument("--reason", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("plan_path")
    compare.add_argument("--output-dir", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_coding_tool_provider_parity(
                settings_path=args.settings_path or None,
                secrets_path=args.secrets_path or None,
                artifact_path=args.artifact_path or None,
                output_dir=args.output_dir or None,
                execution_client=args.execution_client,
                request_profile_id=args.request_profile,
            )
        elif args.command == "create-consents":
            result = create_coding_tool_provider_parity_consents(
                args.plan_path,
                confirm_data_export=bool(args.confirm_data_export),
            )
        elif args.command == "execute":
            result = execute_coding_tool_provider_parity_candidate(
                args.plan_path,
                args.consent_index_path,
                candidate_id=args.candidate_id,
                operator_confirm_network=bool(args.operator_confirm_network),
                openclaw_command=args.openclaw_command,
            )
        elif args.command == "recover-interrupted":
            result = recover_interrupted_coding_tool_provider_parity_candidate(
                args.plan_path,
                args.consent_index_path,
                candidate_id=args.candidate_id,
                reason=args.reason,
            )
        elif args.command == "execute-all":
            result = execute_all_coding_tool_provider_parity_candidates(
                args.plan_path,
                args.consent_index_path,
                operator_confirm_network=bool(args.operator_confirm_network),
                openclaw_command=args.openclaw_command,
            )
        else:
            result = compare_coding_tool_provider_parity(
                args.plan_path, output_dir=args.output_dir or None
            )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "status": "invalid_or_blocked",
                    "error": str(exc),
                    "remote_requests_made": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") in {
        "ready_for_operator_consent",
        "ready_for_operator_execution",
        "completed",
        "ready_for_human_review",
    }:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
