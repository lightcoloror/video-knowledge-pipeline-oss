from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from .models import now_iso
from .storage import write_json, write_text_atomic


SCHEMA = "video_knowledge_pipeline.local_media_progress.v1"
EVENT_SCHEMA = "video_knowledge_pipeline.local_media_progress_event.v1"
STATUSES = frozenset({"running", "completed", "failed", "degraded"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "degraded"})
ProgressCallback = Callable[[dict[str, Any]], None]


class LocalMediaProgress:
    """Persist one monotonic progress stream and expose the same events to CLI callers."""

    def __init__(
        self,
        *,
        pipeline: str,
        snapshot_path: str | Path,
        events_path: str | Path,
        callback: ProgressCallback | None = None,
        run_id: str = "",
        reset: bool = True,
    ) -> None:
        self.pipeline = _required_text(pipeline, "pipeline")
        self.snapshot_path = Path(snapshot_path).expanduser().resolve()
        self.events_path = Path(events_path).expanduser().resolve()
        self.callback = callback
        self.run_id = str(run_id or f"local-media-{uuid4().hex[:12]}")
        self.sequence = 0
        self.percent = 0.0
        self.terminal = False
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            write_text_atomic(self.events_path, "")

    def emit(
        self,
        *,
        stage: str,
        percent: float,
        current_item: int = 0,
        total_items: int = 0,
        message: str,
        status: str = "running",
        output_paths: Iterable[str | Path] = (),
        report_paths: Iterable[str | Path] = (),
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.terminal:
            raise ValueError("cannot emit progress after a terminal event")
        clean_status = str(status or "").strip().lower()
        if clean_status not in STATUSES:
            raise ValueError(f"unsupported local media progress status: {status}")
        value = round(min(100.0, max(0.0, float(percent))), 3)
        if value < self.percent:
            raise ValueError("local media progress percent must be monotonic")
        if clean_status in TERMINAL_STATUSES:
            value = 100.0
        current = max(0, int(current_item or 0))
        total = max(0, int(total_items or 0))
        if total and current > total:
            raise ValueError("current_item cannot exceed total_items")
        self.sequence += 1
        self.percent = value
        event = {
            "schema": EVENT_SCHEMA,
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "sequence": self.sequence,
            "stage": _required_text(stage, "stage"),
            "percent": value,
            "current_item": current,
            "total_items": total,
            "message": _required_text(message, "message"),
            "status": clean_status,
            "output_paths": _paths(output_paths),
            "report_paths": _paths(report_paths),
            "details": dict(details or {}),
            "updated_at": now_iso(),
        }
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        snapshot = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "status": clean_status,
            "percent": value,
            "stage": event["stage"],
            "current_item": current,
            "total_items": total,
            "message": event["message"],
            "output_paths": event["output_paths"],
            "report_paths": event["report_paths"],
            "event_count": self.sequence,
            "events_path": str(self.events_path),
            "last_event": event,
            "updated_at": event["updated_at"],
        }
        write_json(self.snapshot_path, snapshot)
        self.terminal = clean_status in TERMINAL_STATUSES
        if self.callback:
            self.callback(dict(event))
        return event

    def artifacts(self) -> dict[str, str]:
        return {
            "progress_json": str(self.snapshot_path),
            "progress_jsonl": str(self.events_path),
        }


def render_progress_line(event: dict[str, Any]) -> str:
    """Render a human line exclusively from a persisted machine event."""

    percent = float(event.get("percent") or 0.0)
    current = int(event.get("current_item") or 0)
    total = int(event.get("total_items") or 0)
    item = f" {current}/{total}" if total else ""
    return (
        f"[{str(event.get('status') or '').upper()}] "
        f"{percent:6.1f}% {event.get('stage', '')}{item} - {event.get('message', '')}"
    )


def stderr_progress_callback(event: dict[str, Any]) -> None:
    import sys

    print(render_progress_line(event), file=sys.stderr, flush=True)


def _paths(values: Iterable[str | Path]) -> list[str]:
    rows: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in rows:
            rows.append(text)
    return rows


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text
