from __future__ import annotations

import copy
from typing import Any


LAB_SCHEMA = "video_knowledge_pipeline.model_screening_lab.v1"
SIMULATION_SCHEMA = "video_knowledge_pipeline.model_screening_simulation.v1"

TASK_ENDPOINTS = {
    "text": "/v1/chat/completions",
    "vision": "/v1/chat/completions",
    "asr": "/v1/audio/transcriptions",
    "ocr": "/v1/ocr",
}

SCREENING_CRITERIA = {
    "quality": {
        "weight": 45,
        "requires_real_result": True,
        "evidence": "human-reviewed fixed samples",
    },
    "latency": {
        "weight": 20,
        "requires_real_result": True,
        "evidence": "measured end-to-end latency",
    },
    "reliability": {
        "weight": 15,
        "requires_real_result": True,
        "evidence": "success, retry, and failure receipts",
    },
    "estimated_cost": {
        "weight": 15,
        "requires_real_result": True,
        "evidence": "provider usage and cost metadata",
    },
    "privacy_boundary": {
        "weight": 5,
        "requires_real_result": False,
        "evidence": "route, consent, and upload manifest review",
    },
}

SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "completed",
        "label": "Successful response",
        "ok": True,
        "status": "completed",
        "http_status": 200,
        "latency_ms": 120,
        "estimated_cost": 0.001,
        "retryable": False,
        "retry_after_seconds": None,
        "failure_recovery": "not_needed",
    },
    {
        "id": "rate_limited",
        "label": "HTTP 429 rate limit",
        "ok": False,
        "status": "rate_limited",
        "http_status": 429,
        "latency_ms": 35,
        "estimated_cost": 0,
        "retryable": True,
        "retry_after_seconds": 30,
        "failure_recovery": "retry_within_route_policy",
    },
    {
        "id": "provider_unavailable",
        "label": "HTTP 503 provider unavailable",
        "ok": False,
        "status": "provider_unavailable",
        "http_status": 503,
        "latency_ms": 80,
        "estimated_cost": 0,
        "retryable": True,
        "retry_after_seconds": 5,
        "failure_recovery": "retry_or_next_deployment_in_same_pool",
    },
    {
        "id": "gateway_timeout",
        "label": "Gateway timeout",
        "ok": False,
        "status": "gateway_timeout",
        "http_status": None,
        "latency_ms": 120000,
        "estimated_cost": "unknown",
        "retryable": True,
        "retry_after_seconds": None,
        "failure_recovery": "return_timeout_without_cross_location_fallback",
    },
    {
        "id": "gateway_response_invalid",
        "label": "Invalid JSON response",
        "ok": False,
        "status": "gateway_response_invalid",
        "http_status": 200,
        "latency_ms": 90,
        "estimated_cost": "unknown",
        "retryable": False,
        "retry_after_seconds": None,
        "failure_recovery": "block_invalid_response",
    },
    {
        "id": "local_gateway_unavailable",
        "label": "Local gateway unavailable",
        "ok": False,
        "status": "local_gateway_unavailable",
        "http_status": None,
        "latency_ms": 15,
        "estimated_cost": 0,
        "retryable": False,
        "retry_after_seconds": None,
        "failure_recovery": "return_local_error_without_remote_fallback",
    },
)


def model_screening_lab_status(
    profiles: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    onboarding: dict[str, Any],
) -> dict[str, Any]:
    profile_rows = [dict(row) for row in profiles if isinstance(row, dict)]
    onboarding_rows = [
        dict(row)
        for row in onboarding.get("entries") or []
        if isinstance(row, dict)
    ]
    candidates = []
    for row in onboarding_rows:
        profile_ids = [
            str(value)
            for value in row.get("matching_profile_ids") or []
            if str(value)
        ]
        candidates.append(
            {
                "id": str(row.get("id") or ""),
                "label": str(row.get("label") or ""),
                "priority": str(row.get("priority") or ""),
                "capabilities": list(row.get("capabilities") or []),
                "runtime_integration": str(row.get("runtime_integration") or ""),
                "readiness_status": str(row.get("status") or "unknown"),
                "matching_profile_ids": profile_ids,
                "configured_profile_count": len(profile_ids),
                "credential_configured": bool(row.get("credential_configured")),
                "route_configured": bool(row.get("route_configured")),
                "consent_status": str(row.get("consent_status") or "not_checked"),
                "eligible_for_real_smoke": str(row.get("status") or "")
                == "ready_for_consent",
            }
        )
    configured_ids = {
        str(row.get("id") or "")
        for row in profile_rows
        if str(row.get("id") or "")
    }
    return {
        "schema": LAB_SCHEMA,
        "mode": "offline_contract_simulation",
        "tasks": list(TASK_ENDPOINTS),
        "scenarios": [copy.deepcopy(row) for row in SCENARIOS],
        "criteria": copy.deepcopy(SCREENING_CRITERIA),
        "criteria_weight_total": sum(
            int(row["weight"]) for row in SCREENING_CRITERIA.values()
        ),
        "candidates": candidates,
        "candidate_count": len(candidates),
        "configured_profile_ids": sorted(configured_ids),
        "operator_boundary": {
            "provider_requests_made": 0,
            "source_artifacts_read": False,
            "payloads_generated": False,
            "credentials_read": False,
            "simulation_is_quality_evidence": False,
            "real_smoke_requires_route_and_consent": True,
            "cross_location_fallback_allowed": False,
        },
    }


def simulate_offline_gateway_contract(task: str, scenario: str) -> dict[str, Any]:
    task_key = str(task or "").strip().lower().replace("-", "_")
    task_aliases = {
        "text_llm": "text",
        "summary_rewrite": "text",
        "transcript_correction": "text",
        "semantic_frame": "vision",
        "temporal_sequence": "vision",
        "document_visual": "vision",
    }
    task_key = task_aliases.get(task_key, task_key)
    if task_key not in TASK_ENDPOINTS:
        raise ValueError("screening task must be text, vision, asr, or ocr")
    scenario_key = str(scenario or "").strip().lower().replace("-", "_")
    selected = next(
        (row for row in SCENARIOS if row["id"] == scenario_key),
        None,
    )
    if selected is None:
        raise ValueError("unsupported screening scenario")
    result = copy.deepcopy(selected)
    return {
        "schema": SIMULATION_SCHEMA,
        "simulation": True,
        "task": task_key,
        "endpoint": TASK_ENDPOINTS[task_key],
        "scenario": scenario_key,
        "ok": result["ok"],
        "status": result["status"],
        "http_status": result["http_status"],
        "latency_ms": result["latency_ms"],
        "call_count": 1,
        "estimated_cost": result["estimated_cost"],
        "retryable": result["retryable"],
        "retry_after_seconds": result["retry_after_seconds"],
        "failure_recovery": result["failure_recovery"],
        "quality_score": None,
        "content": None,
        "operator_boundary": {
            "provider_requests_made": 0,
            "loopback_control_requests_made": 1,
            "source_artifacts_read": False,
            "payloads_generated": False,
            "credentials_read": False,
            "eligible_for_quality_ranking": False,
            "cross_location_fallback_used": False,
        },
    }
