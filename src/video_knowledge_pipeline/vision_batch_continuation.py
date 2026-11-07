from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from .powershell import quote_powershell_literal
from .storage import write_json


def write_semantic_continuation(
    root: Path, *, pending_indexes: list[int], batch_size: int,
    summary: dict[str, Any], execution_actor: str = "operator",
) -> dict[str, Any]:
    """Persist the next bounded task, without issuing or authorizing model calls."""
    pending = sorted(set(pending_indexes))
    next_indexes = pending[:batch_size] if batch_size > 0 else pending
    failed = sorted(set(summary.get("run_failed_indexes") or []) & set(pending))
    blocked = summary.get("status") not in {"ok", "not_needed", "preview_only"}
    status = "complete" if not pending else ("blocked_current_batch" if blocked else "needs_execution")
    task_path = root / "semantic-frame-next-batch.json"
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "video-knowledge.ps1"
    command_base = (
        f"& {quote_powershell_literal(script_path)}" if script_path.is_file()
        else f"& {quote_powershell_literal(sys.executable)} -m video_knowledge_pipeline.cli"
    )
    root_arg = quote_powershell_literal(root)
    csv = ",".join(map(str, next_indexes))
    # The caller must supply its original private config. Never persist keys or
    # manufacture a new provider/consent while preparing a continuation.
    preflight = (
        f"{command_base} vision-execution-preflight {root_arg} --provider-config $visionConfig "
        f"--semantic-limit {len(next_indexes)} --semantic-indexes {csv} --no-temporal"
    ) if next_indexes else ""
    execute = (
        f"{command_base} run-multimodal-frame-analysis {root_arg} --execute "
        f"--provider-config $visionConfig --limit {len(next_indexes)} --indexes {csv} "
        f"--confirm-vision-calls {len(next_indexes)} --confirm-vision-indexes {csv} "
        f"--execution-actor {execution_actor}"
    ) if next_indexes else ""
    if execute:
        for key, flag in (
            ("image_probe_max_edge", "--image-probe-max-edge"),
            ("image_probe_jpeg_quality", "--image-probe-jpeg-quality"),
            ("request_max_tokens", "--max-tokens"),
        ):
            if summary.get(key) is not None:
                execute += f" {flag} {int(summary[key])}"
    if execute and execution_actor == "agent":
        execute += " --export-consent $visionConsent"
    result = {
        "schema": "video_knowledge_pipeline.semantic_continuation.v1",
        "status": status,
        "pending_count": len(pending), "pending_indexes": pending,
        "next_indexes": next_indexes, "failed_indexes": failed,
        "batch_size": batch_size, "task_path": str(task_path),
        "preflight_command": preflight, "execute_command_template": execute,
        "required_variables": (["visionConfig"] + (["visionConsent"] if execution_actor == "agent" else [])) if pending else [],
        "requires_fresh_preflight": bool(pending),
        "automatic_execution": False,
        "instruction": (
            "Set $visionConfig to the same private provider config path; inspect fresh preflight "
            "and confirm its calls/indexes match before executing. Resolve current batch failures first. "
            "The task is persisted, but no background worker is running."
        ) if pending else "No pending semantic frames. This does not prove OCR, coverage, or human acceptance.",
    }
    write_json(task_path, result)
    return result
