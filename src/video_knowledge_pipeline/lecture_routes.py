from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_runner import detect_asr_runners
from .asr_execution import run_asr_plan
from .extractor_execution import run_extractor_plan
from .lecture_pipeline import run_ready_lecture_pipeline, status_lecture_pipeline_plan
from .models import now_iso
from .storage import append_jsonl, ensure_project_dirs, read_json, read_jsonl
from .tool_research import recommended_trial_order


def recommended_route_status(plan_json: str | Path, *, rank: int = 1) -> dict[str, Any]:
    """Return the ready/remaining state for lecture recommended routes."""
    plan_path = Path(plan_json).expanduser().resolve()
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("lecture pipeline plan must be a JSON object")
    readiness = status_lecture_pipeline_plan(plan_path)
    routes = _current_clean_routes(plan)
    if not routes:
        raise ValueError("plan has no recommended_routes")
    try:
        wanted_rank = int(rank)
    except (TypeError, ValueError) as exc:
        raise ValueError("rank must be an integer") from exc

    enriched: list[dict[str, Any]] = []
    for item in sorted(routes, key=lambda route: int(route.get("rank") or 9999)):
        ready = _route_ready(item, readiness)
        available = bool(item.get("available"))
        runnable = available and not ready and int(item.get("rank") or 0) >= wanted_rank
        row = dict(item)
        row["ready"] = ready
        row["runnable"] = runnable
        row["blocked_reason"] = "" if runnable else _route_blocked_reason(item, ready=ready, available=available)
        enriched.append(row)

    next_route = next((row for row in enriched if row["runnable"]), None)
    for row in enriched:
        row["selected_next"] = bool(next_route and row.get("name") == next_route.get("name"))

    ready_count = sum(1 for row in enriched if row["ready"])
    available_count = sum(1 for row in enriched if row.get("available"))
    remaining = [row for row in enriched if row.get("available") and not row["ready"]]
    missing = [row for row in enriched if not row.get("available") and not row["ready"]]
    if next_route:
        status = "has_next"
    elif missing:
        status = "waiting_for_tools_or_outputs"
    else:
        status = "complete"

    return {
        "plan_path": str(plan_path),
        "rank": wanted_rank,
        "status": status,
        "next_route": next_route,
        "routes": enriched,
        "ready_count": ready_count,
        "available_count": available_count,
        "remaining_count": len(remaining),
        "missing_unavailable_count": len(missing),
        "readiness": readiness,
    }


def run_recommended_route(
    plan_json: str | Path,
    *,
    route: str | None = None,
    rank: int = 1,
    execute: bool = False,
    timeout_seconds: int = 0,
    normalize: bool = True,
) -> dict[str, Any]:
    """Preview or execute one route from lecture-pipeline-plan.json recommended_routes."""
    plan_path = Path(plan_json).expanduser().resolve()
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("lecture pipeline plan must be a JSON object")
    route_state = recommended_route_status(plan_path, rank=rank)
    readiness = route_state["readiness"]
    if route:
        selected = _select_route(plan, route=route, rank=rank, readiness=readiness)
    else:
        selected = route_state.get("next_route")
        if not isinstance(selected, dict):
            return {
                "plan_path": str(plan_path),
                "execute": execute,
                "timeout_seconds": int(timeout_seconds or 0),
                "normalize": normalize,
                "status": "complete",
                "returncode": 0,
                "command_name": "recommended:none",
                "command": "",
                "route_name": "",
                "route_rank": None,
                "operation_status": "skipped",
                "selected_route": None,
                "route_ready_before": False,
                "route_status": route_state,
                "readiness": readiness,
                "operation": {"status": "skipped", "reason": "no available unfinished recommended route"},
                "available_warning": "",
            }
    tool = str(selected.get("mcp_tool") or "").strip()
    args = selected.get("mcp_args") if isinstance(selected.get("mcp_args"), dict) else {}

    if tool == "run_asr_plan" or str(selected.get("name") or "") == "asr":
        operation = run_asr_plan(
            str(args.get("plan_json") or plan_path),
            execute=execute,
            normalize=normalize,
            timeout_seconds=timeout_seconds,
        )
    elif tool == "run_extractor_plan" or selected.get("name"):
        extractor = str(args.get("extractor") or selected.get("name") or "").strip()
        operation = run_extractor_plan(
            str(args.get("plan_json") or plan_path),
            extractor,
            execute=execute,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f"unsupported recommended route tool: {tool or '(missing)'}")

    route_name = str(selected.get("name") or selected.get("command_name") or "").strip()
    operation_status = str(operation.get("status") or "")
    return {
        "plan_path": str(plan_path),
        "execute": execute,
        "timeout_seconds": int(timeout_seconds or 0),
        "normalize": normalize,
        "status": operation_status,
        "returncode": operation.get("returncode"),
        "command_name": f"recommended:{route_name or selected.get('rank', rank)}",
        "command": operation.get("command") or selected.get("command") or "",
        "route_name": route_name,
        "route_rank": selected.get("rank"),
        "operation_status": operation_status,
        "selected_route": selected,
        "route_ready_before": _route_ready(selected, readiness),
        "route_status": route_state,
        "readiness": readiness,
        "operation": operation,
        "available_warning": "" if selected.get("available") else f"route {selected.get('name', '')} is not marked available in the plan",
    }


def run_recommended_route_queue(
    plan_json: str | Path,
    *,
    rank: int = 1,
    execute: bool = False,
    timeout_seconds: int = 0,
    normalize: bool = True,
    max_steps: int = 4,
) -> dict[str, Any]:
    """Preview or execute available unfinished recommended routes in order."""
    plan_path = Path(plan_json).expanduser().resolve()
    initial = recommended_route_status(plan_path, rank=rank)
    route_plan = [route for route in initial["routes"] if route.get("runnable")]
    if not execute:
        return {
            "plan_path": str(plan_path),
            "execute": False,
            "timeout_seconds": int(timeout_seconds or 0),
            "normalize": normalize,
            "max_steps": int(max_steps or 0),
            "status": "preview",
            "route_count": len(route_plan),
            "route_plan": route_plan,
            "initial_status": initial,
            "final_status": initial,
            "runs": [],
        }

    limit = max(1, int(max_steps or 1))
    runs: list[dict[str, Any]] = []
    current = initial
    status_text = "complete" if not current.get("next_route") else "running"
    stop_reason = ""
    for _ in range(limit):
        next_route = current.get("next_route")
        if not isinstance(next_route, dict):
            status_text = "complete"
            break
        route_name = str(next_route.get("name") or next_route.get("command_name") or "").strip()
        before_next = route_name
        result = run_recommended_route(
            plan_path,
            route=route_name,
            rank=rank,
            execute=True,
            timeout_seconds=timeout_seconds,
            normalize=normalize,
        )
        runs.append(result)
        operation_status = str(result.get("operation_status") or result.get("status") or "")
        if result.get("returncode") not in (None, 0) or operation_status in _STOP_ROUTE_STATUSES:
            status_text = "failed"
            stop_reason = operation_status or "nonzero_returncode"
            current = recommended_route_status(plan_path, rank=rank)
            break
        current = recommended_route_status(plan_path, rank=rank)
        after_next = current.get("next_route") if isinstance(current.get("next_route"), dict) else None
        after_name = str((after_next or {}).get("name") or (after_next or {}).get("command_name") or "").strip()
        if after_name and after_name == before_next:
            status_text = "stalled"
            stop_reason = "route_did_not_become_ready"
            break
    else:
        current = recommended_route_status(plan_path, rank=rank)
        status_text = "max_steps_reached" if current.get("next_route") else "complete"

    return {
        "plan_path": str(plan_path),
        "execute": True,
        "timeout_seconds": int(timeout_seconds or 0),
        "normalize": normalize,
        "max_steps": limit,
        "status": status_text,
        "stop_reason": stop_reason,
        "route_count": len(route_plan),
        "initial_status": initial,
        "final_status": current,
        "runs": runs,
    }


def recommended_workspace_advance(
    plan_json: str | Path,
    *,
    execute: bool = False,
    run_queue: bool = True,
    import_ready: bool = True,
    rank: int = 1,
    timeout_seconds: int = 0,
    normalize: bool = True,
    max_steps: int = 4,
    webui_output_dir: str | Path | None = None,
    vault: str | Path | None = None,
    folder: str = "00_Inbox/AI/课程视频知识包",
    target: str = "bilinote",
    merge_window: float = 1.0,
    force_reimport: bool = False,
    allow_draft_obsidian_export: bool = False,
) -> dict[str, Any]:
    """Preview or execute the normal workspace advance: route queue, then ready import."""
    plan_path = Path(plan_json).expanduser().resolve()
    initial_status = recommended_route_status(plan_path, rank=rank)
    initial_pipeline_status = initial_status["readiness"]
    queue_result: dict[str, Any] | None = None
    import_result: dict[str, Any] | None = None
    actions: list[dict[str, Any]] = []

    if run_queue:
        queue_result = run_recommended_route_queue(
            plan_path,
            rank=rank,
            execute=execute,
            timeout_seconds=timeout_seconds,
            normalize=normalize,
            max_steps=max_steps,
        )
        actions.append(
            {
                "name": "recommended_route_queue",
                "status": queue_result.get("status"),
                "execute": execute,
                "route_count": queue_result.get("route_count", 0),
            }
        )

    post_queue_pipeline_status = status_lecture_pipeline_plan(plan_path)
    ready_for_import = bool(post_queue_pipeline_status.get("recommended_pipeline_command"))
    if import_ready:
        if execute and ready_for_import:
            import_result = run_ready_lecture_pipeline(
                plan_path,
                webui_output_dir=webui_output_dir,
                vault=vault,
                folder=folder,
                merge_window=merge_window,
                target=target,
                force_reimport=force_reimport,
                allow_draft_obsidian_export=allow_draft_obsidian_export,
            )
            import_status = "ok"
        elif ready_for_import:
            import_result = {
                "status": "preview",
                "command": post_queue_pipeline_status.get("recommended_pipeline_command", ""),
                "reason": "ready outputs exist; pass execute=true to import them",
            }
            import_status = "preview"
        else:
            import_result = {
                "status": "skipped",
                "reason": "no ready planned outputs to import",
            }
            import_status = "skipped"
        actions.append({"name": "run_ready_lecture_pipeline", "status": import_status, "execute": execute and ready_for_import})

    final_route_status = recommended_route_status(plan_path, rank=rank)
    if execute and import_result and import_result.get("status") == "ok":
        status_text = "advanced"
    elif not execute:
        status_text = "preview"
    elif queue_result and queue_result.get("status") in {"failed", "stalled", "max_steps_reached"} and not ready_for_import:
        status_text = str(queue_result.get("status"))
    elif import_result and import_result.get("status") == "skipped":
        status_text = "waiting_for_outputs"
    else:
        status_text = "advanced"

    result = {
        "plan_path": str(plan_path),
        "execute": execute,
        "run_queue": run_queue,
        "import_ready": import_ready,
        "status": status_text,
        "actions": actions,
        "initial_route_status": initial_status,
        "initial_pipeline_status": initial_pipeline_status,
        "queue": queue_result,
        "post_queue_pipeline_status": post_queue_pipeline_status,
        "ready_for_import": ready_for_import,
        "import": import_result,
        "final_route_status": final_route_status,
    }
    return _write_workspace_advance_log(plan_path, result)


def recommended_workspace_advance_log(root: str | Path) -> dict[str, Any]:
    """Return and render the persisted recommended workspace advance log."""
    paths = ensure_project_dirs(root)
    log_path = paths["lecture_packages"] / "workspace-advance-runs.jsonl"
    markdown_path = paths["notes"] / "lecture-workspace-advance-runs.md"
    rows = read_jsonl(log_path)
    markdown_path.write_text(_render_workspace_advance_log_markdown(root, rows), encoding="utf-8")
    return {
        "project": str(root),
        "log_path": str(log_path),
        "markdown_path": str(markdown_path),
        "count": len(rows),
        "advances": rows,
        "last": rows[-1] if rows else {},
    }


def _select_route(
    plan: dict[str, Any],
    *,
    route: str | None,
    rank: int,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    clean_routes = _current_clean_routes(plan)
    if not clean_routes:
        raise ValueError("plan has no recommended_routes")
    if route:
        wanted = route.strip().lower()
        for item in clean_routes:
            if str(item.get("name") or "").lower() == wanted:
                return item
            if str(item.get("command_name") or "").lower() == wanted:
                return item
        raise ValueError(f"recommended route not found: {route}")
    try:
        wanted_rank = int(rank)
    except (TypeError, ValueError) as exc:
        raise ValueError("rank must be an integer") from exc
    ranked_routes = sorted(clean_routes, key=lambda item: int(item.get("rank") or 9999))
    for item in ranked_routes:
        if int(item.get("rank") or 0) < wanted_rank:
            continue
        if item.get("available") and not _route_ready(item, readiness):
            return item
    for item in clean_routes:
        if int(item.get("rank") or 0) == wanted_rank:
            return item
    raise ValueError(f"recommended route rank not found: {wanted_rank}")


_STOP_ROUTE_STATUSES = {"blocked", "command_not_found", "failed", "normalize_failed", "output_missing", "timeout"}


def _route_ready(route: dict[str, Any], readiness: dict[str, Any]) -> bool:
    name = str(route.get("name") or route.get("command_name") or "").strip()
    ready = readiness.get("ready") if isinstance(readiness.get("ready"), dict) else {}
    if name == "asr":
        return bool(ready.get("asr_transcript"))
    return bool(ready.get(name))


def _clean_routes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    routes = plan.get("recommended_routes") if isinstance(plan.get("recommended_routes"), list) else []
    return [item for item in routes if isinstance(item, dict)]


def _current_clean_routes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return _refresh_route_availability(_clean_routes(plan))


def _refresh_route_availability(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Refresh route availability from the current machine without rebuilding the plan."""
    video_tools = {
        str(row.get("name") or "").lower(): row
        for row in recommended_trial_order()
        if str(row.get("name") or "").lower() in {"vidclaude", "peepshow", "vidwise"}
    }
    asr = detect_asr_runners()
    asr_tools = [tool for tool in asr.get("tools") or [] if isinstance(tool, dict) and tool.get("available")]
    refreshed: list[dict[str, Any]] = []
    for route in routes:
        row = dict(route)
        name = str(row.get("name") or row.get("command_name") or "").strip().lower()
        planned_available = bool(row.get("available"))
        planned_paths = list(row.get("paths") or [])
        row["availability_refreshed"] = True
        row["planned_available"] = planned_available
        row["planned_paths"] = planned_paths
        if name in video_tools:
            tool = video_tools[name]
            paths = list(tool.get("installed_paths") or [])
            row["available"] = bool(tool.get("installed"))
            row["paths"] = paths
            row["current_paths"] = paths
            row["availability_source"] = "current_tool_matrix"
            if not row["available"]:
                row["reason"] = f"{name} 未发现本地可用路径"
        elif name == "asr":
            paths = [str(tool.get("command_path") or tool.get("command") or tool.get("name") or "") for tool in asr_tools]
            row["available"] = bool(asr_tools) or planned_available
            row["paths"] = paths if asr_tools else planned_paths
            row["current_paths"] = paths
            row["current_asr_tools"] = [tool.get("name") for tool in asr_tools]
            row["availability_source"] = "current_asr_runners" if asr_tools else "plan"
            if not row["available"]:
                row["reason"] = "未发现可运行 ASR 命令；先运行 asr-env-status 并应用 ASR 环境变量。"
        else:
            row["current_paths"] = planned_paths
            row["availability_source"] = "plan"
        row["availability_changed"] = planned_available != bool(row.get("available"))
        refreshed.append(row)
    return refreshed


def _route_blocked_reason(route: dict[str, Any], *, ready: bool, available: bool) -> str:
    if ready:
        return "ready"
    if not available:
        return "tool_missing"
    return "below_rank"


def _write_workspace_advance_log(plan_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        return result
    project = Path(str(plan.get("project") or ""))
    if not str(project):
        return result
    paths = ensure_project_dirs(project)
    log_path = paths["lecture_packages"] / "workspace-advance-runs.jsonl"
    record = _workspace_advance_record(result)
    append_jsonl(log_path, [record])
    log = recommended_workspace_advance_log(project)
    result["workspace_advance_log"] = {
        "record": record,
        "log_path": log["log_path"],
        "markdown_path": log["markdown_path"],
        "count": log["count"],
    }
    return result


def _workspace_advance_record(result: dict[str, Any]) -> dict[str, Any]:
    queue = result.get("queue") if isinstance(result.get("queue"), dict) else {}
    imported = result.get("import") if isinstance(result.get("import"), dict) else {}
    final = result.get("final_route_status") if isinstance(result.get("final_route_status"), dict) else {}
    return {
        "created_at": now_iso(),
        "plan_path": result.get("plan_path", ""),
        "execute": bool(result.get("execute")),
        "status": result.get("status", ""),
        "run_queue": bool(result.get("run_queue")),
        "import_ready": bool(result.get("import_ready")),
        "ready_for_import": bool(result.get("ready_for_import")),
        "queue_status": queue.get("status", ""),
        "queue_route_count": queue.get("route_count", 0),
        "queue_run_count": len(queue.get("runs") or []),
        "import_status": imported.get("status") or ("ok" if "pipeline" in imported else ""),
        "final_route_status": final.get("status", ""),
        "remaining_count": final.get("remaining_count", 0),
        "actions": [
            {
                "name": action.get("name", ""),
                "status": action.get("status", ""),
                "execute": bool(action.get("execute")),
            }
            for action in result.get("actions", [])
            if isinstance(action, dict)
        ],
    }


def _render_workspace_advance_log_markdown(root: str | Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Lecture Workspace Advance Runs",
        "",
        f"- Project: `{root}`",
        f"- Count: {len(rows)}",
        "",
        "| Time | Execute | Status | Queue | Import | Remaining |",
        "|---|---|---|---|---|---:|",
    ]
    for row in rows:
        execute = "yes" if row.get("execute") else "no"
        lines.append(
            f"| {row.get('created_at', '')} | {execute} | {row.get('status', '')} | {row.get('queue_status', '')} | {row.get('import_status', '')} | {row.get('remaining_count', 0)} |"
        )
    return "\n".join(lines).rstrip() + "\n"
