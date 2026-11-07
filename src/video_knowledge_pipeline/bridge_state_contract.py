from __future__ import annotations

from typing import Any


def evaluate_bridge_state(
    *,
    configured: bool,
    socket_probe: dict[str, Any],
    health_probe: dict[str, Any],
    scheduled_task: dict[str, Any],
    pipeline_capability: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate bridge and local-pipeline availability without side effects."""

    socket_state = str(socket_probe.get("state") or "not_checked")
    listening = bool(socket_probe.get("listening"))
    health_checked = bool(health_probe.get("checked"))
    healthy = listening and health_checked and bool(health_probe.get("ok"))
    task_state = str(scheduled_task.get("scheduled_task_state") or "").strip().lower()
    task_claims_running = task_state in {"running", "正在运行"}
    stale_runtime_record = bool(configured and task_claims_running and not listening)

    if not configured:
        state = "config_invalid"
    elif stale_runtime_record:
        state = "stale_runtime_record"
    elif socket_state == "timeout":
        state = "probe_timeout"
    elif not listening:
        state = "stopped_by_design"
    elif health_checked and not healthy:
        state = "listening_unhealthy"
    elif healthy:
        state = "healthy"
    else:
        state = "listening_health_not_checked"

    runtime_disposition = (
        "running" if listening else "stale_runtime_record" if stale_runtime_record else "stopped_by_design"
    )
    offline_ready = bool(pipeline_capability.get("ready"))
    return {
        "schema": "video_knowledge_pipeline.bridge_state_contract.v1",
        "configured": bool(configured),
        "listening": listening,
        "healthy": healthy,
        "bridge_state": state,
        "runtime_disposition": runtime_disposition,
        "stale_runtime_record": stale_runtime_record,
        "bridge_call_execution_allowed": healthy,
        "pipeline_offline_ready": offline_ready,
        "configured_is_online": healthy,
        "availability": {
            "http_bridge": {
                "configured": bool(configured),
                "listening": listening,
                "healthy": healthy,
                "execution_allowed": healthy,
            },
            "local_pipeline": {
                "offline_ready": offline_ready,
                "bridge_required": False,
                "offline_quality_route_discoverable": bool(
                    pipeline_capability.get("offline_quality_route_discoverable")
                ),
            },
        },
        "review_evidence_policy": {
            "review_page_exists_is_not_content_reviewed": True,
            "content_reviewed": False,
        },
    }