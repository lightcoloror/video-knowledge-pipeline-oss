from __future__ import annotations

from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal
from .run_artifact_registry import register_bundle_run

ps_quote = quote_powershell_literal


def register_external_reuse_run(
    bundle_dir: str | Path,
    *,
    run_type: str,
    title: str,
    result: dict[str, Any],
    status: str = "",
    summary: str = "",
    artifacts: dict[str, Any] | list[Any] | None = None,
    failed_items: list[dict[str, Any]] | None = None,
    retry_command: str = "",
    next_actions: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
    operator_boundary: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Register a local external-project-inspired VKP capability run.

    This is intentionally small glue around the shared vsummary-style run
    registry. It records outputs, retry commands, and operator boundaries for
    task-console/subqueue-action-plan without starting services or model calls.
    """

    if not write:
        return {}
    root = Path(bundle_dir).expanduser().resolve()
    failed = failed_items or []
    resolved_status = status or ("needs_retry" if failed else "completed")
    return register_bundle_run(
        root,
        run_type=run_type,
        status=resolved_status,
        title=title,
        summary=summary or _summary_from_result(result),
        inputs={"bundle_dir": str(root)},
        parameters=parameters or _dict(result.get("parameters")),
        artifacts=_artifact_rows(artifacts if artifacts is not None else result.get("artifacts")),
        failed_items=failed,
        retry_command=retry_command,
        next_actions=next_actions or [],
        operator_boundary=_operator_boundary(result, operator_boundary),
        write=True,
    )


def _summary_from_result(result: dict[str, Any]) -> str:
    summary = _dict(result.get("summary"))
    if not summary:
        return str(result.get("status") or result.get("schema") or "")
    parts = []
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
    return "; ".join(parts[:8])


def _artifact_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows = []
        for key, path in value.items():
            if isinstance(path, (str, Path)) and str(path).strip():
                rows.append({"key": str(key), "path": str(path)})
        return rows
    if isinstance(value, list):
        return [row if isinstance(row, dict) else {"path": str(row)} for row in value]
    return []


def _operator_boundary(result: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    boundary = {
        "local_only": True,
        "no_download": True,
        "no_cloud_call": True,
        "no_process_started": True,
        "source": "external_project_reuse_run_registry",
    }
    embedded = result.get("operator_boundary")
    if isinstance(embedded, dict):
        boundary.update(embedded)
    if extra:
        boundary.update(extra)
    return boundary


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
