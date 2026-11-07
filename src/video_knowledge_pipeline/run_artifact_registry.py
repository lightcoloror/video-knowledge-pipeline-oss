from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .artifact_freshness import canonical_json_sha256, validate_dependency_snapshot
from .artifact_validation import artifact_evidence
from .models import now_iso
from .storage import read_json, write_json
from .summary_model_input import discover_response_recovery

REGISTRY_SCHEMA = "video_knowledge_pipeline.run_artifact_registry.v1"
RUN_SCHEMA = "video_knowledge_pipeline.run_artifact.v1"
RUN_REGISTRY_MANIFEST_FIELDS = {
    "run_artifact_registry_json": "run-artifact-registry.json",
    "run_artifact_registry_report": "run-artifact-registry.md",
    "mcp_run_artifact_registry_args": "mcp-run-artifact-registry.args.json",
}


def register_bundle_run(
    bundle_dir: str | Path,
    *,
    run_type: str,
    run_id: str | None = None,
    status: str = "planned",
    title: str = "",
    summary: str = "",
    inputs: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    artifacts: list[Any] | None = None,
    failed_items: list[dict[str, Any]] | None = None,
    retry_command: str = "",
    next_actions: list[str] | None = None,
    operator_boundary: dict[str, Any] | None = None,
    resource_requirements: dict[str, Any] | None = None,
    dependency_snapshot: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe_type = _safe_id(run_type or "run")
    safe_run_id = _safe_id(run_id or safe_type)
    run_dir = root / "runs" / safe_run_id
    created_at = now_iso()
    run = {
        "schema": RUN_SCHEMA,
        "run_id": safe_run_id,
        "run_type": safe_type,
        "status": str(status or "planned"),
        "title": title or safe_type,
        "summary": summary or "",
        "bundle_dir": str(root),
        "created_at": created_at,
        "updated_at": created_at,
        "inputs": inputs or {},
        "parameters": parameters or {},
        "artifacts": [_artifact_row(root, item) for item in artifacts or []],
        "failed_items": failed_items or [],
        "retry_command": retry_command or "",
        "next_actions": next_actions or [],
        "operator_boundary": operator_boundary or {},
        "resource_requirements": _resource_requirements(resource_requirements),
        "dependency_snapshot": dependency_snapshot or {},
        "paths": {
            "run_dir": str(run_dir),
            "run_json": str(run_dir / "run.json"),
            "run_markdown": str(run_dir / "run.md"),
        },
    }
    if write:
        _write_run(root, run)
        build_run_artifact_registry(root, write=True)
    return run


def build_run_artifact_registry(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    runs_dir = root / "runs"
    runs: list[dict[str, Any]] = []
    if runs_dir.exists():
        for path in sorted(runs_dir.glob("*/run.json")):
            try:
                value = read_json(path)
            except Exception:
                continue
            if isinstance(value, dict) and value.get("schema") == RUN_SCHEMA:
                runs.append(value)
    runs.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
    recovery = discover_response_recovery(root)
    # These rows are derived from immutable receipts, never persisted as runs.
    # Both existing UI surfaces already consume this registry row contract.
    rows = _response_recovery_rows(root, recovery) + [_registry_row(root, run) for run in runs]
    counts: dict[str, int] = {}
    for run in rows:
        key = str(run.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    registry = {
        "schema": REGISTRY_SCHEMA,
        "bundle_dir": str(root),
        "generated_at": now_iso(),
        "run_count": len(rows),
        "registered_run_count": len(runs),
        "status_counts": counts,
        "runs": rows,
        "summary_response_recovery": recovery,
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "no_process_started": True,
            "purpose": "Index VKP task runs and artifacts for UI/MCP retry and audit.",
        },
    }
    if write:
        json_path = root / "run-artifact-registry.json"
        md_path = root / "run-artifact-registry.md"
        args_path = root / "mcp-run-artifact-registry.args.json"
        write_json(json_path, registry)
        md_path.write_text(render_run_artifact_registry_markdown(registry), encoding="utf-8")
        write_json(args_path, {"bundle_dir": str(root), "write": True})
        manifest_path = root / "manifest.json"
        manifest = _read_object(manifest_path)
        updated_manifest = dict(manifest)
        updated_manifest.update(RUN_REGISTRY_MANIFEST_FIELDS)
        # Do not reserialize an unchanged manifest after a Hash-bound commit.
        # On Windows an otherwise identical rewrite can change LF to CRLF and
        # invalidate the existing transaction's exact-byte restore contract.
        if updated_manifest != manifest:
            write_json(manifest_path, updated_manifest)
        registry["paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "mcp_args": str(args_path),
        }
    return registry


def _response_recovery_rows(root: Path, recovery: dict[str, Any]) -> list[dict[str, Any]]:
    items = list(recovery['items'])
    if recovery['errors']:
        items.append({'status': 'blocked', 'errors': recovery['errors'], 'path': 'discovery'})
    rows = []
    for item in items:
        available = item['status'] == 'recovery_available'
        unverified = item['status'] == 'unverified_current_input'
        artifacts = [{'key': key, **item[key], 'absolute_path': str(root / item[key]['path']), 'exists': True}
                     for key in ('request', 'response') if key in item]
        errors = item.get('errors') or []
        rows.append({
            'run_id': 'summary-response-recovery-' + _safe_id(Path(item['path']).stem),
            'run_type': 'smart_summary_global_reduce_recovery',
            'status': 'needs_review' if available or unverified else 'blocked',
            'title': ('Global Reduce 响应恢复候选' if available else
                      'Global Reduce 响应待验证' if unverified else 'Global Reduce 响应恢复受阻'),
            'summary': ('原请求、响应及当前输入身份已验证；显式恢复仍需通过 reader plan、质量及安装门。'
                        if available else '; '.join(errors)),
            'created_at': '', 'updated_at': '', 'artifact_count': len(artifacts),
            'artifacts': artifacts, 'failed_count': len(errors),
            'failed_items': [{'reason': 'summary_response_recovery_blocked', 'detail': error} for error in errors],
            'retry_command': item.get('recovery_command', ''),
            'next_actions': ['Review the immutable response and explicitly run the existing recovery command.'] if available else errors,
            'run_json': '', 'run_markdown': 'run-artifact-registry.md',
            'resource_requirements': {}, 'parameters': {}, 'recovery': item,
            'freshness': {'status': 'verified_current_input' if available else 'not_assessed' if unverified else 'blocked',
                          'passed': available, 'issues': errors},
            'operator_boundary': {'local_only': True, 'no_cloud_call': True, 'no_process_started': True,
                                  'projection_only': True, 'automatic_recovery': False,
                                  'model_observation': 'not_assessed'},
        })
    return rows


def render_run_markdown(run: dict[str, Any]) -> str:
    lines = [
        f"# Run Artifact: {run.get('title') or run.get('run_id')}",
        "",
        f"- Run ID: `{run.get('run_id', '')}`",
        f"- Type: `{run.get('run_type', '')}`",
        f"- Status: `{run.get('status', '')}`",
        f"- Created: `{run.get('created_at', '')}`",
        f"- Updated: `{run.get('updated_at', '')}`",
        f"- Bundle: `{run.get('bundle_dir', '')}`",
    ]
    if run.get("summary"):
        lines.extend(["", "## Summary", "", str(run.get("summary") or "")])
    resources = run.get("resource_requirements") or {}
    if resources:
        lines.extend(["", "## Resource Requirements", "", f"- `{resources}`"])

    lines.extend(["", "## Artifacts", "", "| Key | Exists | Path |", "| --- | --- | --- |"])
    artifacts = run.get("artifacts") or []
    if artifacts:
        for artifact in artifacts:
            lines.append(f"| `{artifact.get('key', '')}` | `{artifact.get('exists', False)}` | `{artifact.get('path', '')}` |")
    else:
        lines.append("| - | - | - |")
    lines.extend(["", "## Failed Items", "", "| Item | Reason | Detail |", "| --- | --- | --- |"])
    failed = run.get("failed_items") or []
    if failed:
        for item in failed:
            item_id = item.get("index") or item.get("id") or item.get("item") or ""
            lines.append(f"| `{item_id}` | `{item.get('reason', '')}` | {item.get('detail', '')} |")
    else:
        lines.append("| - | - | - |")
    if run.get("retry_command"):
        lines.extend(["", "## Retry Command", "", "```powershell", str(run.get("retry_command") or ""), "```"])
    next_actions = run.get("next_actions") or []
    if next_actions:
        lines.extend(["", "## Next Actions", ""])
        lines.extend(f"- {action}" for action in next_actions)
    return "\n".join(lines).rstrip() + "\n"


def render_run_artifact_registry_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# Run Artifact Registry",
        "",
        f"- Bundle: `{registry.get('bundle_dir', '')}`",
        f"- Generated: `{registry.get('generated_at', '')}`",
        f"- Runs: `{registry.get('run_count', 0)}`",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    counts = registry.get("status_counts") or {}
    if counts:
        for key, value in sorted(counts.items()):
            lines.append(f"| `{key}` | {value} |")
    else:
        lines.append("| - | 0 |")
    lines.extend(["", "## Runs", "", "| Run | Type | Status | Artifacts | Failed | Retry |", "| --- | --- | --- | ---: | ---: | --- |"])
    runs = registry.get("runs") or []
    if runs:
        for run in runs:
            retry = "yes" if run.get("retry_command") else ""
            lines.append(
                f"| `{run.get('run_id', '')}` | `{run.get('run_type', '')}` | `{run.get('status', '')}` | {run.get('artifact_count', 0)} | {run.get('failed_count', 0)} | {retry} |"
            )
    else:
        lines.append("| - | - | - | 0 | 0 | - |")
    recovery = registry.get('summary_response_recovery') or {}
    if recovery.get('items') or recovery.get('errors'):
        lines.extend(['', '## Global Reduce response recovery', '',
            f"- Status: `{recovery.get('status')}`; complete scan: `{recovery.get('scan_complete')}`",
            f"- Bounded direct directory: `{recovery.get('directory')}`; entries: {recovery.get('entries_examined')}/{recovery.get('entry_limit')}",
            '- Validated candidates verify request/response and current input identity only. Explicit recovery still passes existing Reduce gates.',
            '- Unverified current-input rows provide snapshot links only, with no recovery command or freshness claim.',
            '- No model call or automatic recovery; model observation and acceptance remain not assessed.'])
        for error in recovery.get('errors') or []:
            lines.append(f'- Blocked: {error}')
        for item in recovery.get('items') or []:
            lines.extend(['', f"### {item.get('status')}: {Path(item.get('path', '')).name}", ''])
            for key in ('request', 'response'):
                if key in item:
                    lines.append(f"- [{key}]({item[key]['path']}) · SHA256 `{item[key]['sha256']}`")
            lines.extend(f'- Blocked: {error}' for error in item.get('errors') or [])
            if item.get('recovery_command'):
                lines.extend(['', '```powershell', item['recovery_command'], '```'])
    return "\n".join(lines).rstrip() + "\n"


def _write_run(root: Path, run: dict[str, Any]) -> None:
    run_dir = Path(str(run["paths"]["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run.json", run)
    (run_dir / "run.md").write_text(render_run_markdown(run), encoding="utf-8")


def _registry_row(root: Path, run: dict[str, Any]) -> dict[str, Any]:
    snapshot = run.get("dependency_snapshot") if isinstance(run.get("dependency_snapshot"), dict) else {}
    freshness = validate_dependency_snapshot(root, snapshot) if snapshot else {
        "status": "not_recorded",
        "passed": False,
        "issues": [],
    }
    recovery = None
    if (run.get("parameters") or {}).get("recovery_profile") == "versioned_result_recovery.v1":
        from .run_artifact_recovery import inspect_result_run
        try:
            recovery = inspect_result_run(root, str(run.get("run_id") or ""))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            recovery = {"status": "blocked", "error": str(exc), "semantic_acceptance": False}
    return {
        "run_id": run.get("run_id", ""),
        "run_type": run.get("run_type", ""),
        "status": run.get("status", ""),
        "title": run.get("title", ""),
        "summary": run.get("summary", ""),
        "created_at": run.get("created_at", ""),
        "updated_at": run.get("updated_at", ""),
        "artifact_count": len(run.get("artifacts") or []),
        "failed_count": len(run.get("failed_items") or []),
        "retry_command": run.get("retry_command", ""),
        "resource_requirements": dict(run.get("resource_requirements") or {}),
        "parameters": dict(run.get("parameters") or {}),
        "operator_boundary": dict(run.get("operator_boundary") or {}),
        "artifacts": [dict(row) for row in run.get("artifacts") or [] if isinstance(row, dict)],
        "run_json": _relative_or_abs(root, str((run.get("paths") or {}).get("run_json") or "")),
        "freshness": freshness,
        **({"versioned_recovery": recovery} if recovery is not None else {}),
        "run_markdown": _relative_or_abs(root, str((run.get("paths") or {}).get("run_markdown") or "")),
    }


def _artifact_row(root: Path, item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        key = str(item.get("key") or item.get("name") or item.get("path") or "artifact")
        raw_path = str(item.get("path") or item.get("artifact_path") or "")
        desc = str(item.get("description") or "")
    else:
        raw_path = str(item)
        key = Path(raw_path).name or "artifact"
        desc = ""
    path = _resolve_artifact_path(root, raw_path)
    row = {
        "key": key,
        "path": _relative_or_abs(root, str(path)) if path else raw_path,
        "absolute_path": str(path) if path else "",
        "exists": bool(path and path.exists()),
        "description": desc,
    }
    if path and path.is_file():
        evidence = artifact_evidence(path)
        row["bytes"] = int(evidence["bytes"])
        row["sha256"] = str(evidence["sha256"])
        if path.suffix.lower() == ".json":
            try:
                value = read_json(path)
            except Exception:
                value = None
            if value is not None:
                row["canonical_json_sha256"] = canonical_json_sha256(value)
    return row


def _resolve_artifact_path(root: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve()
    except Exception:
        return path


def _relative_or_abs(root: Path, value: str) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    try:
        resolved = path.resolve()
        return str(resolved.relative_to(root))
    except Exception:
        return str(path)


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _safe_id(value: str) -> str:
    text = str(value or "run").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-._") or "run"


def _resource_requirements(value: dict[str, Any] | None) -> dict[str, int]:
    if not value:
        return {}
    allowed = {"cpu", "gpu", "network"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unsupported resource requirement categories: {unknown}")
    normalized: dict[str, int] = {}
    for key in ("cpu", "gpu", "network"):
        if key not in value:
            continue
        amount = int(value[key])
        if amount < 0:
            raise ValueError(f"resource requirement must be non-negative: {key}")
        normalized[key] = amount
    return normalized
