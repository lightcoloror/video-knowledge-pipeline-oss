"""Narrow front door for the opt-in full courseware material channel."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .courseware_material import build_courseware_groups, build_courseware_material, import_courseware_page, import_courseware_review
from .courseware_scan import (add_courseware_candidate, collect_courseware_extractions, extract_courseware_candidates,
                              prepare_courseware_scan, run_courseware_scan)

COMMANDS = {"courseware-incremental-ocr", "courseware-prepare", "courseware-scan", "courseware-extract", "courseware-collect",
            "courseware-import", "courseware-review", "courseware-material", "courseware-status", "courseware-add-candidate", "courseware-groups", "courseware-dedupe-plan", "courseware-dedupe-plan-v2"}


def _compact(value: dict) -> dict:
    omitted = {"pages", "dependencies", "review_gaps", "candidates", "windows", "completed_windows",
               "observed_text_groups", "reading_groups", "unobserved_candidates", "items", "rows"}
    result = {k: v for k, v in value.items() if k not in omitted}
    for key in ("material", "ocr_result"):
        if isinstance(result.get(key), dict):
            result[key] = _compact(result[key])
    if "pages" in value:
        pending = [n for n, p in enumerate(value["pages"], 1) if p.get("status") != "extracted" or p.get("gaps")]
        result.update(pending_extraction_count=len(pending), next_pending_indexes=pending[:16],
                      next_pending_index_space="material_page_order",
                      review_gap_count=len(value.get("review_gaps", [])), audit_artifact="courseware/material.json")
    for key in ("candidates", "windows", "completed_windows", "items"):
        if key in value:
            result[key + "_count"] = len(value[key])
    return result


def _input(path: str) -> dict:
    file = Path(path)
    if file.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("courseware JSON input exceeds 16 MiB")
    value = json.loads(file.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("courseware input must be an object")
    return value


def _worker_plan(indexes: list[int], mode: str, *, resume: bool = False) -> dict:
    """Resolve explicit routing only; runtime owns resource and resume checks."""
    maximum = 8 if mode == "checkpoint" else 32
    if (mode not in {"checkpoint", "batch", "auto"}
            or not isinstance(indexes, list) or not 1 <= len(indexes) <= maximum
            or any(type(i) is not int or i < 1 for i in indexes)
            or len(set(indexes)) != len(indexes)):
        raise ValueError(f"{mode} OCR requires 1..{maximum} unique explicit scan indexes")
    selected = ("batch" if len(indexes) > 8 else "checkpoint") if mode == "auto" else mode
    return {"requested_mode": mode, "selected_mode": selected, "indexes": list(indexes),
            "index_space": "scan_candidate", "reuse_worker": selected == "batch",
            "checkpoint_groups": [indexes[n:n + 8] for n in range(0, len(indexes), 8)],
            "concurrency": 1, "execution_performed": False,
            "resume_requested": resume, "resume_identity_verified": False,
            "next_action": "Execution requires --execute and the existing input, model and resource gates.",
            "recovery": "For an interrupted run, inspect its existing receipt and repeat the original indexes in the same order, resolved worker mode and model policy with --resume. A pending-only subset is not that run's resume identity."}


def _next_ocr_plan(bundle_dir: str, material: dict) -> dict:
    """Suggest captured, unimported scan indexes without probing any runtime."""
    from .courseware_material import read_scan
    state = read_scan(Path(bundle_dir))
    if state["identity"] != material["scan_identity"]:
        raise ValueError("courseware status scan identity drift")
    # Material is time-sorted; independent review may append candidates out of
    # order. Only the scan's original positions are valid for incremental OCR.
    pages = {page["page_id"]: page for page in material["pages"]}
    pending = [n for n, candidate in enumerate(state["candidates"], 1)
               if pages[candidate["page_id"]]["status"] == "not_extracted"
               and candidate.get("frame")]
    result = {"status": "preview_only" if pending else "no_captured_unimported_pages",
              "pending_count": len(pending), "remaining_after_suggestion": max(0, len(pending) - 32),
              "index_space": "scan_candidate", "indexes": pending[:32],
              "execution_performed": False, "resource_gates_checked": False,
              "model_policy_required": True, "for_new_work_only": True,
              "recovery": "An interrupted run must use its existing receipt's full original selection and policy; this pending-only suggestion is not a resume command."}
    if pending:
        plan = _worker_plan(pending[:32], "auto")
        result.update(worker_plan=plan, preview_argv=["courseware-incremental-ocr", str(bundle_dir),
            "--indexes", ",".join(str(n) for n in pending[:32]),
            "--model-policy", "<models.json>", "--worker-mode", "auto"])
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Full courseware material: candidate scan != independent review")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in sorted(COMMANDS):
        command = commands.add_parser(name)
        command.add_argument("bundle_dir")
        command.add_argument("--full-json", action="store_true", help="Opt in to full audit payload; default is a compact receipt")
        if name == 'courseware-incremental-ocr':
            command.add_argument('--indexes',required=True)
            command.add_argument('--model-policy',required=True)
            command.add_argument('--execute',action='store_true')
            command.add_argument('--resume',action='store_true')
            worker = command.add_mutually_exclusive_group()
            worker.add_argument('--worker-mode', choices=('checkpoint', 'batch', 'auto'),
                                help='Explicit route: checkpoint <=8 (default); batch <=32 reuses one worker; auto chooses only from the supplied index count')
            worker.add_argument('--reuse-worker',action='store_true',
                                help='Alias for --worker-mode batch: <=32 frames, serial 8-frame checkpoints')
            command.add_argument('--refresh-material',action='store_true',
                                 help='With resolved batch mode: perform one full verified material refresh at batch end')
        if name == "courseware-prepare":
            command.add_argument("--source", required=True)
            command.add_argument("--scope-json", required=True, help="start/end and explicit excluded_intervals in seconds")
            command.add_argument("--window-seconds", type=float, default=30)
        if name in {"courseware-scan", "courseware-extract"}:
            command.add_argument("--execute", action="store_true")
            command.add_argument("--limit", type=int, default=2 if name == "courseware-scan" else 16)
            command.add_argument("--timeout-seconds", type=int, default=60)
            command.add_argument("--disk-budget-bytes", type=int, default=512 * 1024 * 1024)
        if name == "courseware-scan":
            command.add_argument("--resume", action="store_true")
        if name == "courseware-extract":
            command.add_argument("--execute-ocr", action="store_true")
            command.add_argument("--indexes", help="Exact 1-based scan candidate indexes; count must not exceed --limit")
        if name == "courseware-collect":
            command.add_argument("--executor", required=True)
            command.add_argument("--model-actual", required=True)
            command.add_argument('--indexes', help='Exact OCR-child indexes, 1..32; not scan indexes')
            command.add_argument('--defer-material', action='store_true',
                                 help='Collect bounded page receipts only; final courseware-material verification remains required')
        if name in {"courseware-dedupe-plan", "courseware-dedupe-plan-v2"}:
            command.add_argument("--indexes", required=True, help="Explicit captured candidates; plan is read-only")
            command.add_argument("--limit", type=int, default=256)
            command.add_argument("--roi", help="Preview-only native pixel x1,y1,x2,y2; never authorizes skipping")
        if name in {"courseware-import", "courseware-review", "courseware-add-candidate"}:
            command.add_argument("--input-json", required=True)
    args = parser.parse_args(argv)
    worker_plan = None
    try:
        name, root = args.command, args.bundle_dir
        if name == 'courseware-incremental-ocr':
            from .courseware_incremental_runtime import execute, preflight
            indexes = [int(n) for n in args.indexes.split(',')]
            mode = 'batch' if args.reuse_worker else (args.worker_mode or 'checkpoint')
            worker_plan = _worker_plan(indexes, mode, resume=args.resume)
            reuse_worker = worker_plan['reuse_worker']
            options = {'reuse_worker':True,'refresh_material':args.refresh_material} if reuse_worker else {}
            if args.refresh_material and not reuse_worker:
                raise ValueError('--refresh-material requires --reuse-worker or resolved --worker-mode batch')
            result = (execute(root,indexes,args.model_policy,resume=args.resume,**options) if args.execute
                      else preflight(args.model_policy))
            result = {**result, 'worker_plan': worker_plan}
        elif name == "courseware-prepare":
            scope = _input(args.scope_json)
            result = prepare_courseware_scan(root, args.source, start=scope["start"], end=scope["end"],
                        excluded_intervals=scope.get("excluded_intervals", []), window_seconds=args.window_seconds)
        elif name in {"courseware-scan", "courseware-extract"} and not args.execute:
            result = {"ok": True, "status": "preview", "execution_performed": False,
                      "next_action": "Use --execute for this bounded local derived-media batch; OCR additionally requires --execute-ocr."}
        elif name == "courseware-scan":
            result = run_courseware_scan(root, max_windows=args.limit, timeout_seconds=args.timeout_seconds,
                        disk_budget_bytes=args.disk_budget_bytes, resume=args.resume)
        elif name == "courseware-extract":
            result = extract_courseware_candidates(root, limit=args.limit, indexes=args.indexes, timeout_seconds=args.timeout_seconds,
                        disk_budget_bytes=args.disk_budget_bytes, execute_ocr=args.execute_ocr)
        elif name == "courseware-collect":
            if args.defer_material and not args.indexes:
                raise ValueError('--defer-material requires bounded --indexes')
            result = collect_courseware_extractions(root, executor=args.executor, model_actual=args.model_actual,
                        indexes=[int(n) for n in args.indexes.split(',')] if args.indexes else None,
                        refresh_material=not args.defer_material)
            if args.defer_material:
                result.update(material_refresh='deferred',material_freshness='not_verified',
                              next_action='courseware-material <bundle>; full verification required before summary/acceptance')
        elif name == "courseware-import":
            result = import_courseware_page(root, _input(args.input_json))
        elif name == "courseware-review":
            result = import_courseware_review(root, _input(args.input_json))
        elif name == "courseware-add-candidate":
            result = add_courseware_candidate(root, _input(args.input_json))
        elif name == "courseware-groups":
            result = build_courseware_groups(root)
        elif name in {"courseware-dedupe-plan", "courseware-dedupe-plan-v2"}:
            from .courseware_dedupe import plan_courseware_dedupe, plan_courseware_dedupe_v2
            roi = [int(n) for n in args.roi.split(',')] if args.roi else None
            planner = plan_courseware_dedupe_v2 if name.endswith('-v2') else plan_courseware_dedupe
            result = planner(root, indexes=args.indexes, limit=args.limit, roi=roi)
        else:
            result = build_courseware_material(root, write=name == "courseware-material")
            if name == "courseware-status" and result.get("active"):
                result = {**result, "next_ocr": _next_ocr_plan(root, result)}
        print(json.dumps(result if args.full_json else _compact(result), ensure_ascii=False, indent=2))
        return 0 if result.get("ok", True) else 2
    except (ValueError, OSError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        result = {"ok": False, "status": "blocked", "error": str(exc),
                  "automatic_retry": False, "original_asr_modified": False}
        if worker_plan is not None:
            result["worker_plan"] = worker_plan
        print(json.dumps(result, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
