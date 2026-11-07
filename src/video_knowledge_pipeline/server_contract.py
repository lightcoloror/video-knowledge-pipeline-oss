"""Machine-readable SDK contract; no HTTP listener or plugin auto-loader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def server_api_contract() -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.server_contract.v1",
        "sdk": "video_knowledge_pipeline.server_execution:ServerExecution",
        "operations": {
            "start": {"arguments": ["owner", "operation_id", "input_root", "plan"], "effect": "persist plan; no model call"},
            "status": {"arguments": ["owner", "job"], "effect": "read-only; verify release/input/receipt hashes"},
            "cancel": {"arguments": ["owner", "job"], "effect": "stop future dispatch; provider cancellation not assumed"},
            "resume": {"arguments": ["owner", "job", "lease_guard", "max_steps", "reconcile"],
                       "effect": "bounded advance through explicit guarded adapters"},
            "result": {"arguments": ["owner", "job"], "effect": "read-only verified private receipts; not public download manifest"},
        },
        "states": ["queued", "running", "paused", "blocked", "reconciliation_required", "failed", "cancelled", "complete_draft"],
        "plan_json_schema": json.loads((Path(__file__).parent / "schemas/server-plan.v1.schema.json").read_text(encoding="utf-8")),
        "job_json_schema": json.loads((Path(__file__).parent / "schemas/server-job.v1.schema.json").read_text(encoding="utf-8")),
        "errors": {
            "ValueError": "invalid input/configuration/release/receipt binding; no implicit retry",
            "FileNotFoundError": "job/input/receipt unavailable; no completion claim",
            "RuntimeError": "native write-lock busy; host retries status before dispatch",
            "reconciliation_required": "unknown paid outcome; reconcile durable request_id, do not call provider again",
        },
        "host_requirements": ["authenticated owner identity", "media admission and duration proof",
                              "per-call budget and owner consent", "renewable lease", "canonical ledger and usage outbox",
                              "content QA before download", "separate charging policy"],
        "backend_policy": {"mode": "api", "fallback": False, "automatic_retries": 0,
                           "implemented_shared_capabilities": ["text", "vision"],
                           "blocked_until_owner_onboarding": ["asr", "alignment_api", "ocr_api"]},
        "execution_safety_v1": False, "real_provider_verified": False,
        "local_codex_mode": "retained in existing entrypoints; never implicit fallback for server jobs",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect VKP server contract or create a local source seal; no provider execution")
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("contract")
    seal = sub.add_parser("seal")
    seal.add_argument("--source-root", required=True)
    seal.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.operation == "seal":
        from .server_release import create_server_release
        result = create_server_release(args.source_root, args.output)
    else:
        result = server_api_contract()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
