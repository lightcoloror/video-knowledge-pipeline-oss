from __future__ import annotations

from pathlib import Path
from typing import Any

from .bundle_status import controlled_execution_check
from .models import now_iso
from .multimodal_frame_analyzer import (
    run_multimodal_frame_analysis,
    vision_analysis_apply_restore,
    vision_analysis_restore_plan,
)
from .storage import read_json, write_json
from .temporal_visual_analyzer import run_temporal_visual_analysis
from .vision_preflight import vision_execution_preflight


SMOKE_SCHEMA = "lecture_controlled_execution_smoke.v1"


def controlled_execution_smoke(
    bundle_dir: str | Path,
    *,
    execute: bool = False,
    restore_after: bool = False,
    provider_config: dict[str, Any] | None = None,
    kind: str = "auto",
    index: int | None = None,
    frame_count: int = 8,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    cfg = dict(provider_config or {"provider": "fixture"})
    selected = _select_smoke_preflight(root, cfg=cfg, kind=kind, index=index, frame_count=frame_count)
    preflight = selected["preflight"]
    action = selected["action"]
    run_result: dict[str, Any] = {}
    restore_plan: dict[str, Any] = {}
    restore_apply: dict[str, Any] = {}
    if execute and preflight.get("ready_to_execute"):
        run_result = _run_confirmed_action(root, action=action, preflight=preflight, cfg=cfg, frame_count=frame_count)
        run_id = _run_id(run_result)
        if restore_after and run_id:
            restore_plan = vision_analysis_restore_plan(root, run_id=run_id, write=True)
            restore_apply = vision_analysis_apply_restore(
                root,
                plan_json=restore_plan.get("json_path"),
                execute=True,
                confirm_run_id=run_id,
            )
    check = controlled_execution_check(root, refresh=False, write=write)
    report = {
        "schema": SMOKE_SCHEMA,
        "created_at": now_iso(),
        "bundle_dir": str(root),
        "execute": bool(execute),
        "restore_after": bool(restore_after),
        "provider": _public_provider(cfg),
        "selected_action": action,
        "preflight": _preflight_summary(preflight),
        "run_summary": _run_summary(run_result),
        "restore_summary": _restore_summary(restore_plan, restore_apply),
        "controlled_execution_check": {
            "status": check.get("status"),
            "ready_for_real_vision_execution": bool(check.get("ready_for_real_vision_execution")),
            "report_path": check.get("report_path", ""),
            "report_markdown_path": check.get("report_markdown_path", ""),
        },
        "paths": {
            "preflight_path": preflight.get("preflight_path", ""),
            "preflight_json_path": preflight.get("preflight_json_path", ""),
            "run_report_path": run_result.get("report_path", ""),
            "restore_plan_path": restore_plan.get("json_path", ""),
            "restore_apply_audit_path": (restore_apply.get("audit") or {}).get("jsonl_path", "") if restore_apply else "",
        },
        "next_steps": _next_steps(execute=execute, preflight=preflight, run_result=run_result, restore_after=restore_after),
    }
    if write:
        report_path = root / "controlled-execution-smoke.json"
        markdown_path = root / "controlled-execution-smoke.md"
        args_path = root / "mcp-controlled-execution-smoke.args.json"
        write_json(report_path, report)
        markdown_path.write_text(render_controlled_execution_smoke_markdown(report), encoding="utf-8")
        write_json(
            args_path,
            {
                "bundle_dir": str(root),
                "execute": False,
                "restore_after": False,
                "provider_config": {"provider": "fixture"},
                "kind": "auto",
                "frame_count": frame_count,
                "write": True,
            },
        )
        manifest = read_json(root / "manifest.json")
        if isinstance(manifest, dict):
            manifest["controlled_execution_smoke"] = "controlled-execution-smoke.md"
            manifest["controlled_execution_smoke_json"] = "controlled-execution-smoke.json"
            manifest["mcp_controlled_execution_smoke_args"] = "mcp-controlled-execution-smoke.args.json"
            write_json(root / "manifest.json", manifest)
        report["report_path"] = str(report_path)
        report["report_markdown_path"] = str(markdown_path)
        report["mcp_args_path"] = str(args_path)
    return report


def render_controlled_execution_smoke_markdown(report: dict[str, Any]) -> str:
    preflight = report.get("preflight") if isinstance(report.get("preflight"), dict) else {}
    run = report.get("run_summary") if isinstance(report.get("run_summary"), dict) else {}
    restore = report.get("restore_summary") if isinstance(report.get("restore_summary"), dict) else {}
    check = report.get("controlled_execution_check") if isinstance(report.get("controlled_execution_check"), dict) else {}
    lines = [
        "# Controlled Execution Smoke",
        "",
        f"- Bundle: `{report.get('bundle_dir', '')}`",
        f"- Execute: `{report.get('execute', False)}`",
        f"- Restore after: `{report.get('restore_after', False)}`",
        f"- Provider: `{(report.get('provider') or {}).get('provider', '')}` / `{(report.get('provider') or {}).get('model', '')}`",
        f"- Selected: `{(report.get('selected_action') or {}).get('kind', '')}` indexes `{(report.get('selected_action') or {}).get('indexes', [])}`",
        f"- Preflight ready: `{preflight.get('ready_to_execute', False)}` calls `{preflight.get('confirm_vision_calls', 0)}` indexes `{preflight.get('confirm_vision_indexes', '')}`",
        f"- Run status: `{run.get('status', 'not_run')}` updated `{run.get('updated_count', 0)}` run `{run.get('run_id', '')}`",
        f"- Restore status: `{restore.get('status', 'not_run')}` applied `{restore.get('applied_count', 0)}`",
        f"- Controlled check: `{check.get('status', '')}` ready `{check.get('ready_for_real_vision_execution', False)}`",
        "",
        "## Next Steps",
        "",
    ]
    for step in report.get("next_steps") or []:
        lines.append(f"- {step}")
    return "\n".join(lines) + "\n"


def _select_smoke_preflight(
    root: Path,
    *,
    cfg: dict[str, Any],
    kind: str,
    index: int | None,
    frame_count: int,
) -> dict[str, Any]:
    normalized = str(kind or "auto").strip().lower()
    candidates = ["semantic", "temporal"] if normalized == "auto" else [normalized]
    last: dict[str, Any] = {}
    for candidate_kind in candidates:
        semantic = candidate_kind == "semantic"
        temporal = candidate_kind == "temporal"
        preflight = vision_execution_preflight(
            root,
            provider_config=cfg,
            semantic_limit=1 if semantic else 0,
            temporal_limit=1 if temporal else 0,
            frame_count=frame_count,
            include_semantic=semantic,
            include_temporal=temporal,
            semantic_indexes=[index] if semantic and index else None,
            temporal_indexes=[index] if temporal and index else None,
            write=True,
        )
        action = _action_from_preflight(preflight, candidate_kind)
        last = {"preflight": preflight, "action": action}
        if preflight.get("ready_to_execute"):
            return last
    return last


def _action_from_preflight(preflight: dict[str, Any], kind: str) -> dict[str, Any]:
    selected = preflight.get("selected_indexes") if isinstance(preflight.get("selected_indexes"), dict) else {}
    confirmation = preflight.get("confirmation") if isinstance(preflight.get("confirmation"), dict) else {}
    if kind == "temporal":
        indexes = [int(value) for value in selected.get("temporal") or []]
        return {
            "kind": "temporal",
            "indexes": indexes,
            "confirm_vision_calls": int(confirmation.get("temporal_confirm_vision_calls") or 0),
            "confirm_vision_indexes": str(confirmation.get("temporal_confirm_vision_indexes") or ""),
        }
    indexes = [int(value) for value in selected.get("semantic") or []]
    return {
        "kind": "semantic",
        "indexes": indexes,
        "confirm_vision_calls": int(confirmation.get("semantic_confirm_vision_calls") or 0),
        "confirm_vision_indexes": str(confirmation.get("semantic_confirm_vision_indexes") or ""),
    }


def _run_confirmed_action(
    root: Path,
    *,
    action: dict[str, Any],
    preflight: dict[str, Any],
    cfg: dict[str, Any],
    frame_count: int,
) -> dict[str, Any]:
    if not preflight.get("ready_to_execute"):
        return {"summary": {"status": "preflight_blocked", "error": "preflight_not_ready"}}
    if action.get("kind") == "temporal":
        return run_temporal_visual_analysis(
            root,
            execute=True,
            provider_config=cfg,
            frame_count=frame_count,
            limit=1,
            indexes=action.get("indexes") or None,
            confirm_vision_calls=int(action.get("confirm_vision_calls") or 0),
            confirm_vision_indexes=str(action.get("confirm_vision_indexes") or ""),
        )
    return run_multimodal_frame_analysis(
        root,
        execute=True,
        provider_config=cfg,
        limit=1,
        indexes=action.get("indexes") or None,
        confirm_vision_calls=int(action.get("confirm_vision_calls") or 0),
        confirm_vision_indexes=str(action.get("confirm_vision_indexes") or ""),
    )


def _preflight_summary(preflight: dict[str, Any]) -> dict[str, Any]:
    confirmation = preflight.get("confirmation") if isinstance(preflight.get("confirmation"), dict) else {}
    return {
        "ready_to_execute": bool(preflight.get("ready_to_execute")),
        "blockers": preflight.get("blockers") if isinstance(preflight.get("blockers"), list) else [],
        "expected_api_calls": int(preflight.get("expected_api_calls") or 0),
        "confirm_vision_calls": int(confirmation.get("confirm_vision_calls") or 0),
        "confirm_vision_indexes": str(confirmation.get("confirm_vision_indexes") or ""),
        "preflight_path": str(preflight.get("preflight_path") or ""),
        "preflight_json_path": str(preflight.get("preflight_json_path") or ""),
    }


def _run_summary(run_result: dict[str, Any]) -> dict[str, Any]:
    if not run_result:
        return {"status": "not_run", "updated_count": 0, "run_id": ""}
    summary = run_result.get("summary") if isinstance(run_result.get("summary"), dict) else {}
    record = (run_result.get("run_audit") or {}).get("record") if isinstance(run_result.get("run_audit"), dict) else {}
    return {
        "status": str(summary.get("status") or record.get("status") or ""),
        "error": str(summary.get("error") or record.get("error") or ""),
        "updated_count": int(summary.get("updated") or record.get("updated_count") or 0),
        "timeline_diff_count": int(record.get("timeline_diff_count") or 0),
        "run_id": str(record.get("run_id") or ""),
        "report_path": str(run_result.get("report_path") or ""),
    }


def _restore_summary(restore_plan: dict[str, Any], restore_apply: dict[str, Any]) -> dict[str, Any]:
    if not restore_plan and not restore_apply:
        return {"status": "not_run", "applied_count": 0}
    summary = restore_apply.get("summary") if isinstance(restore_apply.get("summary"), dict) else {}
    return {
        "status": str(summary.get("status") or restore_plan.get("status") or ""),
        "applied_count": int(summary.get("applied_count") or 0),
        "plan_status": str(restore_plan.get("status") or ""),
        "plan_json": str(restore_plan.get("json_path") or ""),
    }


def _run_id(run_result: dict[str, Any]) -> str:
    record = (run_result.get("run_audit") or {}).get("record") if isinstance(run_result.get("run_audit"), dict) else {}
    return str(record.get("run_id") or "")


def _public_provider(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cfg.items() if "key" not in str(key).lower() and "token" not in str(key).lower()}


def _next_steps(*, execute: bool, preflight: dict[str, Any], run_result: dict[str, Any], restore_after: bool) -> list[str]:
    if not preflight.get("ready_to_execute"):
        return ["Inspect controlled-execution-smoke.md and fix preflight blockers before execute=true."]
    if not execute:
        return ["Rerun controlled-execution-smoke with execute=true to perform one confirmed fixture/provider write."]
    run = _run_summary(run_result)
    if run.get("status") != "ok":
        return ["Inspect the vision run report and provider error before retrying controlled execution."]
    if not restore_after:
        return ["Review the write, then run vision-analysis-restore-plan/apply-restore or rerun smoke with restore_after=true."]
    return ["Controlled execution smoke completed and restore_after was applied."]
