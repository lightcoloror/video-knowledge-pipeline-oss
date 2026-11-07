from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .models import now_iso
from .storage import write_json
from .visual_structure import (
    DEFAULT_EBOOK_ROUTES,
    _read_timeline,
    _visual_structure_candidates,
)

SCHEMA = "video_knowledge_pipeline.visual_structure_ebook_batch.v1"
PROGRESS_SCHEMA = "video_knowledge_pipeline.progress_event.v1"
Runner = Callable[..., subprocess.CompletedProcess[str]]


def run_visual_structure_ebook_batches(
    bundle_dir: str | Path,
    *,
    execute: bool = False,
    include_routes: list[str] | None = None,
    indexes: list[int] | None = None,
    batch_size: int = 3,
    timeout_seconds: int = 120,
    resume: bool = True,
    write: bool = True,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run ebook OCR in short-lived child processes so native resources are released.

    Every batch uses the existing ``run-visual-structure`` CLI contract. A child
    crash or item failure is recorded and later batches continue. Completed OCR
    rows are skipped when ``resume`` is enabled.
    """

    root = Path(bundle_dir).expanduser().resolve()
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    size = max(1, int(batch_size))
    routes = sorted({str(value) for value in (include_routes or sorted(DEFAULT_EBOOK_ROUTES)) if str(value)})
    available = _candidate_indexes(root, include_routes=routes)
    requested = sorted({int(value) for value in (indexes or available) if int(value) > 0})
    selected = [value for value in requested if value in set(available)]
    completed_before = _successful_indexes(root)
    pending = [value for value in selected if not (resume and value in completed_before)]
    batches = [pending[offset : offset + size] for offset in range(0, len(pending), size)]
    exports = root / "exports"
    report_json = exports / "visual-structure-ebook-batch.json"
    report_markdown = exports / "visual-structure-ebook-batch.md"
    progress_json = exports / "visual-structure-ebook-batch-progress.json"
    progress_jsonl = exports / "visual-structure-ebook-batch-progress.jsonl"
    events: list[dict[str, Any]] = []
    batch_results: list[dict[str, Any]] = []

    def emit(*, status: str, completed: int, message: str) -> None:
        total = len(pending)
        percent = 100.0 if total == 0 else round(min(1.0, completed / total) * 100.0, 2)
        event = {
            "schema": PROGRESS_SCHEMA,
            "stage": "visual_structure_ebook_batch",
            "percent": percent,
            "current_item": min(completed, total),
            "total_items": total,
            "message": message,
            "status": status,
            "output_path": str(report_json),
            "report_path": str(report_markdown),
            "updated_at": now_iso(),
        }
        events.append(event)
        if write:
            exports.mkdir(parents=True, exist_ok=True)
            write_json(progress_json, event)
            with progress_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
                import json

                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    if not execute:
        result = _result(
            root=root,
            status="planned",
            execute=False,
            routes=routes,
            selected=selected,
            pending=pending,
            completed=sorted(completed_before.intersection(selected)),
            failed=[],
            batch_results=[],
            batch_size=size,
            timeout_seconds=timeout_seconds,
            resume=resume,
            report_json=report_json,
            report_markdown=report_markdown,
            progress_json=progress_json,
            progress_jsonl=progress_jsonl,
        )
        emit(status="completed", completed=0, message=f"planned {len(pending)} OCR item(s) in {len(batches)} child batch(es)")
        return _write_result(result, events=events, write=write)

    emit(status="running", completed=0, message=f"starting {len(pending)} OCR item(s) in {len(batches)} child batch(es)")
    attempted = 0
    for batch_number, batch in enumerate(batches, start=1):
        before = _successful_indexes(root)
        command = _child_command(
            root,
            indexes=batch,
            include_routes=routes,
            timeout_seconds=timeout_seconds,
        )
        error = ""
        returncode = -1
        try:
            completed_process = runner(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=_child_environment(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(60, int(timeout_seconds) * max(1, len(batch)) + 60),
                check=False,
            )
            returncode = int(completed_process.returncode)
            if returncode != 0:
                error = _bounded_error(completed_process.stderr or f"child exit code {returncode}")
        except subprocess.TimeoutExpired as exc:
            error = f"child process timed out after {exc.timeout} seconds"
        except Exception as exc:  # noqa: BLE001 - persisted as degraded batch evidence.
            error = _bounded_error(str(exc))
        after = _successful_indexes(root)
        succeeded = sorted(set(batch).intersection(after))
        failed = sorted(set(batch) - set(succeeded))
        if failed and not error:
            error = "ebook_pipeline_result_not_ok"
        attempted += len(batch)
        batch_results.append(
            {
                "batch": batch_number,
                "indexes": batch,
                "returncode": returncode,
                "succeeded_indexes": succeeded,
                "failed_indexes": failed,
                "newly_succeeded_indexes": sorted(set(succeeded) - before),
                "error": error,
            }
        )
        emit(
            status="running",
            completed=attempted,
            message=f"batch {batch_number}/{len(batches)} completed; success={len(succeeded)} failed={len(failed)}",
        )

    completed_after = _successful_indexes(root).intersection(selected)
    failed_indexes = sorted(set(selected) - completed_after)
    if failed_indexes and completed_after:
        status = "degraded"
    elif failed_indexes:
        status = "failed"
    else:
        status = "completed"
    result = _result(
        root=root,
        status=status,
        execute=True,
        routes=routes,
        selected=selected,
        pending=pending,
        completed=sorted(completed_after),
        failed=failed_indexes,
        batch_results=batch_results,
        batch_size=size,
        timeout_seconds=timeout_seconds,
        resume=resume,
        report_json=report_json,
        report_markdown=report_markdown,
        progress_json=progress_json,
        progress_jsonl=progress_jsonl,
    )
    emit(
        status=status,
        completed=len(pending),
        message=f"OCR batch terminal status={status}; completed={len(completed_after)} failed={len(failed_indexes)}",
    )
    return _write_result(result, events=events, write=write)


def _candidate_indexes(root: Path, *, include_routes: list[str]) -> list[int]:
    timeline = _read_timeline(root)
    rows = _visual_structure_candidates(root, timeline, include_routes=set(include_routes))
    return sorted({int(row.get("index") or 0) for row in rows if int(row.get("index") or 0) > 0})


def _successful_indexes(root: Path) -> set[int]:
    timeline = _read_timeline(root)
    values: set[int] = set()
    for row in timeline:
        if not isinstance(row, dict):
            continue
        status = row.get("ebook_pipeline_status")
        if isinstance(status, dict) and status.get("ok") is True:
            index = int(row.get("index") or 0)
            if index > 0:
                values.add(index)
    return values


def _child_command(root: Path, *, indexes: list[int], include_routes: list[str], timeout_seconds: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "video_knowledge_pipeline.cli",
        "run-visual-structure",
        str(root),
        "--execute-ebook-pipeline",
        "--indexes",
        ",".join(str(value) for value in indexes),
        "--include-routes",
        ",".join(include_routes),
        "--timeout-seconds",
        str(max(1, int(timeout_seconds))),
    ]


def _child_environment() -> dict[str, str]:
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1])
    current = str(env.get("PYTHONPATH") or "").strip()
    # The converter can delegate OCR to another Python minor version. Never
    # inject this parent's binary wheels (e.g. cp313 NumPy into cp311).
    source_paths = [part for part in current.split(os.pathsep) if part.strip()
                    and not {"site-packages", "dist-packages"}.intersection(
                        part.replace("\\", "/").casefold().split("/"))]
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([source_root, *source_paths]))
    return env


def _bounded_error(value: str) -> str:
    return " ".join(str(value or "").split())[-2000:]


def _result(
    *,
    root: Path,
    status: str,
    execute: bool,
    routes: list[str],
    selected: list[int],
    pending: list[int],
    completed: list[int],
    failed: list[int],
    batch_results: list[dict[str, Any]],
    batch_size: int,
    timeout_seconds: int,
    resume: bool,
    report_json: Path,
    report_markdown: Path,
    progress_json: Path,
    progress_jsonl: Path,
) -> dict[str, Any]:
    failed_csv = ",".join(str(value) for value in failed)
    retry = ""
    if failed_csv:
        retry = (
            f"python -m video_knowledge_pipeline.cli run-visual-structure-ebook-batches \"{root}\" "
            f"--execute --indexes {failed_csv} --batch-size {batch_size} --timeout-seconds {timeout_seconds}"
        )
    return {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": status,
        "ok": status in {"planned", "completed"},
        "execute": execute,
        "resume": resume,
        "include_routes": routes,
        "batch_size": batch_size,
        "timeout_seconds": timeout_seconds,
        "selected_indexes": selected,
        "pending_indexes": pending,
        "completed_indexes": completed,
        "failed_indexes": failed,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "batch_results": batch_results,
        "retry_command": retry,
        "artifacts": {
            "json": str(report_json),
            "markdown": str(report_markdown),
            "progress_json": str(progress_json),
            "progress_jsonl": str(progress_jsonl),
        },
        "operator_boundary": {
            "local_only": True,
            "child_process_per_batch": True,
            "continues_after_batch_failure": True,
            "no_remote_fallback": True,
        },
        "updated_at": now_iso(),
    }


def _write_result(result: dict[str, Any], *, events: list[dict[str, Any]], write: bool) -> dict[str, Any]:
    if not write:
        result["progress_events"] = events
        return result
    root = Path(str(result["bundle_dir"]))
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    write_json(exports / "visual-structure-ebook-batch.json", result)
    lines = [
        "# Visual Structure ebook OCR Batch",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Completed: `{result.get('completed_count')}`",
        f"- Failed: `{result.get('failed_count')}`",
        f"- Batch size: `{result.get('batch_size')}`",
        f"- Resume: `{result.get('resume')}`",
        "",
        "## Failed indexes",
        "",
        ", ".join(str(value) for value in result.get("failed_indexes") or []) or "none",
        "",
        "## Retry command",
        "",
        str(result.get("retry_command") or "none"),
    ]
    (exports / "visual-structure-ebook-batch.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return result
