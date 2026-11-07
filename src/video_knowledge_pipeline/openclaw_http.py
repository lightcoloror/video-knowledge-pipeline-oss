from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .config import config_status, service_config, service_url
from .content_asset_batch import batch_content_asset_status, content_handoff_pack
from .content_asset_status import content_asset_status
from .openclaw_bridge_status import openclaw_bridge_status
from .openclaw_bridge_doctor import openclaw_bridge_doctor
from .openclaw_docker_contract import openclaw_docker_contract_check
from .openclaw_integration import openclaw_video_ingest, openclaw_video_link, openclaw_video_plan
from .openclaw_live_smoke import openclaw_live_smoke
from .task_console import export_subqueue_action_plan, export_task_console
from .transcript_semantic_batch import transcript_semantic_batch_codex_review_draft, transcript_semantic_batch_import_review_notes, transcript_semantic_batch_review_pack, transcript_semantic_repair_run
from .vision_review_queue import vision_review_queue
from .vdo_handoff import ingest_vdo_handoff, vdo_handoff_plan

SERVER_NAME = "video-knowledge-openclaw-http"
SERVER_VERSION = "0.1.0"
TOKEN_ENV = "VIDEO_KNOWLEDGE_OPENCLAW_HTTP_TOKEN"


def main(argv: list[str] | None = None) -> int:
    default_service = service_config("openclaw_http")
    parser = argparse.ArgumentParser(description="HTTP bridge for Docker OpenClaw video knowledge calls.")
    parser.add_argument("--host", default=str(default_service.get("host") or "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(default_service.get("port") or 8931))
    parser.add_argument("--token", default=os.environ.get(TOKEN_ENV, ""))
    args = parser.parse_args(argv)

    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.token:
        print(f"Refusing non-local bind without --token or {TOKEN_ENV}.", file=sys.stderr)
        return 2

    handler = build_handler(args.token, bind_host=args.host, bind_port=args.port)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"{SERVER_NAME} listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def build_handler(token: str = "", *, bind_host: str = "", bind_port: int = 0):
    started_at = time.time()
    tools = _tool_map()

    class Handler(BaseHTTPRequestHandler):
        server_version = f"{SERVER_NAME}/{SERVER_VERSION}"

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.write_cors_headers()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if not self.authorized():
                self.write_error("unauthorized", "Unauthorized", status=401, retryable=False)
                return
            if self.path == "/health":
                status = config_status()
                self.write_json(
                    {
                        "ok": True,
                        "server": SERVER_NAME,
                        "version": SERVER_VERSION,
                        "transport": "http",
                        "tool_count": len(tools),
                        "tools": sorted(tools),
                        "service_urls": status.get("service_urls", {}),
                        "bind_host": bind_host,
                        "bind_port": bind_port,
                        "uptime_seconds": round(time.time() - started_at, 3),
                    }
                )
                return
            if self.path == "/tools":
                self.write_json({"ok": True, "tools": _tool_schemas()})
                return
            if self.path == "/contract":
                self.write_json(_contract_payload(bind_host=bind_host, bind_port=bind_port))
                return
            self.write_error("not_found", f"Not found: {self.path}", status=404, retryable=False)

        def do_POST(self) -> None:  # noqa: N802
            if not self.authorized():
                self.write_error("unauthorized", "Unauthorized", status=401, retryable=False)
                return
            if self.path != "/call":
                self.write_error("not_found", f"Not found: {self.path}", status=404, retryable=False)
                return
            request_id = self.headers.get("X-Request-Id") or f"req-{uuid.uuid4().hex[:12]}"
            try:
                body = self.read_json()
                name = str(body.get("name") or body.get("tool") or "")
                arguments = body.get("arguments") if "arguments" in body else body.get("args")
                arguments = arguments or {}
                if not name:
                    raise ValueError("name is required")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                if name not in tools:
                    raise KeyError(f"unsupported tool: {name}")
                result = tools[name](**arguments)
                envelope = {"request_id": request_id, "ok": True, "result": result}
                if isinstance(result, dict):
                    envelope.update(result)
                self.write_json(envelope)
            except json.JSONDecodeError as exc:
                self.write_error("invalid_json", f"Invalid JSON body: {exc}", status=400, retryable=False, request_id=request_id)
            except (KeyError, TypeError, ValueError) as exc:
                self.write_error("invalid_request", str(exc), status=400, retryable=False, request_id=request_id)
            except Exception as exc:  # noqa: BLE001
                self.write_error("tool_error", str(exc), status=500, retryable=True, request_id=request_id)

        def authorized(self) -> bool:
            if not token:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {token}" or self.headers.get("X-Api-Token", "") == token

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def write_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Api-Token, X-Request-Id")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
            encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.write_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def write_error(self, code: str, message: str, *, status: int, retryable: bool, request_id: str = "") -> None:
            self.write_json(
                {
                    "ok": False,
                    "request_id": request_id,
                    "code": code,
                    "message": message,
                    "retryable": retryable,
                },
                status=status,
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


def _tool_map() -> dict[str, Callable[..., dict[str, Any]]]:
    return {
        "openclaw_bridge_status": openclaw_bridge_status,
        "openclaw_bridge_doctor": openclaw_bridge_doctor,
        "openclaw_live_smoke": openclaw_live_smoke,
        "openclaw_docker_contract_check": openclaw_docker_contract_check,
        "openclaw_video_plan": openclaw_video_plan,
        "openclaw_video_ingest": openclaw_video_ingest,
        "openclaw_video_link": openclaw_video_link,
        "openclaw_video_from_vdo_handoff": vdo_handoff_plan,
        "openclaw_video_ingest_vdo_handoff": ingest_vdo_handoff,
        "content_asset_status": content_asset_status,
        "export_task_console": export_task_console,
        "task_console": export_task_console,
        "subqueue_action_plan": export_subqueue_action_plan,
        "transcript_semantic_repair_run": transcript_semantic_repair_run,
        "transcript_semantic_batch_review_pack": transcript_semantic_batch_review_pack,
        "transcript_semantic_batch_import_review_notes": transcript_semantic_batch_import_review_notes,
        "transcript_semantic_batch_codex_review_draft": transcript_semantic_batch_codex_review_draft,
        "vision_review_queue": vision_review_queue,
        "batch_content_asset_status": batch_content_asset_status,
        "content_handoff_pack": content_handoff_pack,
    }


def _tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "openclaw_bridge_status",
            "description": "Check whether the configured host OpenClaw HTTP bridge is running. No video processing or cloud calls.",
            "required": [],
        },
        {
            "name": "openclaw_bridge_doctor",
            "description": "Read-only lifecycle doctor for the VKP OpenClaw HTTP bridge.",
            "required": [],
        },
        {
            "name": "openclaw_live_smoke",
            "description": "Read-only smoke for bridge health, Docker contract, optional content asset status, and transcript semantic batch acceptance.",
            "required": [],
        },
        {
            "name": "openclaw_docker_contract_check",
            "description": "Read-only check for OpenClaw Docker VKP/VDO mount and environment contract.",
            "required": [],
        },
        {
            "name": "openclaw_video_plan",
            "description": "Plan a video link through video-download-orchestrator. Planning only; will_download=false.",
            "required": ["url_or_text"],
        },
        {
            "name": "openclaw_video_ingest",
            "description": "Prepare an existing local or already downloaded video for video knowledge extraction.",
            "required": ["media_path"],
        },
        {
            "name": "openclaw_video_link",
            "description": "Plan a link and optionally delegate explicit confirmed download to video-download-orchestrator.",
            "required": ["url_or_text"],
        },
        {
            "name": "openclaw_video_from_vdo_handoff",
            "description": "Normalize VDO manifest/report/review artifacts into a VKP ingest preview. Does not process video.",
            "required": [],
        },
        {
            "name": "openclaw_video_ingest_vdo_handoff",
            "description": "Preview or execute VKP ingest from a ready VDO handoff. Defaults to preview only.",
            "required": [],
        },
        {
            "name": "content_asset_status",
            "description": "Read-only status for exported content material cards and downstream inspiration readiness.",
            "required": ["bundle_dir"],
        },
        {
            "name": "export_task_console",
            "description": "Write or preview the static VKP task console for a bundle. No video processing or cloud calls.",
            "required": ["bundle_dir"],
        },
        {
            "name": "subqueue_action_plan",
            "description": "Write or preview the copyable VKP subqueue action plan for a bundle. No task execution or cloud calls.",
            "required": ["bundle_dir"],
        },
        {
            "name": "transcript_semantic_repair_run",
            "description": "Preview or execute local safe transcript semantic correction repair actions. Defaults to preview; cloud LLM and closure remain disabled unless explicitly allowed.",
            "required": ["batch_input"],
        },
        {
            "name": "transcript_semantic_batch_review_pack",
            "description": "Build a cross-bundle semantic correction review pack and todo JSON. No cloud calls or transcript writes.",
            "required": ["batch_input"],
        },
        {
            "name": "transcript_semantic_batch_import_review_notes",
            "description": "Import a filled cross-bundle semantic correction review notes JSON into each bundle. Closure remains a separate gated step.",
            "required": ["review_json"],
        },
        {
            "name": "transcript_semantic_batch_codex_review_draft",
            "description": "Generate a conservative local Codex-substitute review notes draft from a batch review pack. No cloud calls.",
            "required": ["review_pack_json"],
        },
        {
            "name": "vision_review_queue",
            "description": "Build a retryable batched multimodal queue from triage candidates. Does not execute cloud calls.",
            "required": ["bundle_dir"],
        },
        {
            "name": "batch_content_asset_status",
            "description": "Summarize content material card readiness for one or more bundles. No video processing.",
            "required": ["batch_input"],
        },
        {
            "name": "content_handoff_pack",
            "description": "Generate a review-only handoff pack from ready content material cards. No publication.",
            "required": ["batch_input"],
        },
    ]


def _contract_payload(*, bind_host: str = "", bind_port: int = 0) -> dict[str, Any]:
    try:
        local_call_url = service_url("openclaw_http")
    except Exception:
        local_call_url = ""
    docker_call_url = ""
    status = config_status()
    service_urls = status.get("service_urls") if isinstance(status.get("service_urls"), dict) else {}
    if service_urls:
        docker_call_url = str(service_urls.get("openclaw_http_docker") or "")
    return {
        "ok": True,
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "transport": "http",
        "endpoints": {
            "health": "/health",
            "tools": "/tools",
            "contract": "/contract",
            "call": "/call",
        },
        "call_payload": {"name": "openclaw_video_plan", "arguments": {"url_or_text": "https://example.com/video"}},
        "local_call_url": local_call_url,
        "docker_call_url": docker_call_url,
        "bind_host": bind_host,
        "bind_port": bind_port,
        "operator_boundary": {
            "download_default": "planning_only",
            "download_execution": "delegated_to_video-download-orchestrator_with_explicit_confirmation",
            "secrets": "Do not send API keys, cookies, or tokens in request bodies.",
            "content_assets": "VKP material cards are review-only evidence. publication_allowed=false.",
        },
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
