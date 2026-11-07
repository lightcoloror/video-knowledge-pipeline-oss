from __future__ import annotations

import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import read_json, read_jsonl, write_json
from .external_reuse_run_artifacts import ps_quote, register_external_reuse_run
from .video_rag_search import search_video_rag

SCHEMA = "video_knowledge_pipeline.video_rag_http.v1"
PLAN_SCHEMA = "video_knowledge_pipeline.video_rag_service_plan.v1"


def video_rag_service_plan(
    bundle_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8781,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    url = f"http://{host}:{int(port)}/search?q=<query>&top_k=8"
    result = {
        "schema": PLAN_SCHEMA,
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "created_at": now_iso(),
        "host": host,
        "port": int(port),
        "base_url": f"http://{host}:{int(port)}",
        "endpoints": {
            "health": f"http://{host}:{int(port)}/health",
            "search": url,
            "chunks": f"http://{host}:{int(port)}/chunks",
        },
        "commands": {
            "serve": f".\\scripts\\video-knowledge.ps1 video-rag-serve '{root}' --host {host} --port {int(port)}",
            "search_cli": f".\\scripts\\video-knowledge.ps1 video-rag-search '{root}' --query '<query>' --top-k 8",
        },
        "operator_boundary": {
            "local_http_only": True,
            "no_cloud_model_call": True,
            "no_vector_backend_started": True,
            "explicit_start_required": True,
        },
        "write": bool(write),
    }
    if write:
        exports = root / "exports"
        json_path = exports / "video-rag-service-plan.json"
        md_path = exports / "video-rag-service-plan.md"
        write_json(json_path, result)
        md_path.write_text(_render_plan_markdown(result), encoding="utf-8")
        manifest["video_rag_service_plan"] = "exports/video-rag-service-plan.json"
        manifest["video_rag_service_plan_markdown"] = "exports/video-rag-service-plan.md"
        manifest["mcp_video_rag_service_plan_args"] = "mcp-video-rag-service-plan.args.json"
        write_json(root / "mcp-video-rag-service-plan.args.json", {"bundle_dir": str(root), "host": host, "port": int(port), "write": True})
        write_json(manifest_path, manifest)
        register_external_reuse_run(
            root,
            run_type="video_rag_service",
            title="VideoRAG local service plan",
            result=result,
            status="needs_execution",
            failed_items=[],
            retry_command=f".\\scripts\\video-knowledge.ps1 video-rag-serve {ps_quote(root)} --host {host} --port {int(port)}",
            next_actions=["Start the local VideoRAG service explicitly only when an operator needs HTTP search."],
            operator_boundary={"explicit_start_required": True, "no_process_started": True, "local_http_only": True},
            write=True,
        )
    return result


def video_rag_http_response(bundle_dir: str | Path, raw_path: str) -> tuple[int, dict[str, str], bytes]:
    root = Path(bundle_dir).expanduser().resolve()
    parsed = urllib.parse.urlparse(raw_path)
    query = urllib.parse.parse_qs(parsed.query)
    try:
        if parsed.path == "/health":
            body = {
                "schema": SCHEMA,
                "ok": True,
                "bundle_dir": str(root),
                "chunks_jsonl_exists": (root / "exports" / "video-rag-chunks.jsonl").exists(),
                "operator_boundary": {"local_http_only": True, "no_cloud_model_call": True},
            }
            return _json_response(200, body)
        if parsed.path == "/search":
            q = _first(query, "q") or _first(query, "query")
            top_k = _int(_first(query, "top_k"), 8)
            body = search_video_rag(root, query=q, top_k=top_k, ensure_pack=True, write=False)
            return _json_response(200, body)
        if parsed.path == "/chunks":
            chunks_path = root / "exports" / "video-rag-chunks.jsonl"
            body = {
                "schema": "video_knowledge_pipeline.video_rag_chunks_response.v1",
                "bundle_dir": str(root),
                "chunks": read_jsonl(chunks_path) if chunks_path.exists() else [],
                "operator_boundary": {"local_http_only": True, "no_cloud_model_call": True},
            }
            return _json_response(200, body)
        return _json_response(404, {"ok": False, "error": "not_found", "path": parsed.path})
    except Exception as exc:  # noqa: BLE001 - service must return JSON failures.
        return _json_response(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def serve_video_rag(bundle_dir: str | Path, *, host: str = "127.0.0.1", port: int = 8781) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API.
            status, headers, body = video_rag_http_response(root, self.path)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature.
            return

    server = ThreadingHTTPServer((host, int(port)), Handler)
    try:
        print(json.dumps({"schema": SCHEMA, "status": "serving", "url": f"http://{host}:{int(port)}", "bundle_dir": str(root)}, ensure_ascii=False), flush=True)
        server.serve_forever()
    finally:
        server.server_close()
    return {"schema": SCHEMA, "status": "stopped", "bundle_dir": str(root), "host": host, "port": int(port)}


def _json_response(status: int, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return status, {"Content-Type": "application/json; charset=utf-8", "Content-Length": str(len(body)), "Access-Control-Allow-Origin": "*"}, body


def _first(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key) or []
    return str(values[0]) if values else ""


def _int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _render_plan_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# VideoRAG Local Service Plan",
        "",
        f"- Bundle: `{result.get('bundle_dir')}`",
        f"- Base URL: `{result.get('base_url')}`",
        "- Boundary: local HTTP only; no cloud model call; explicit start required.",
        "",
        "## Endpoints",
        "",
    ]
    for key, value in (result.get("endpoints") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Commands", ""])
    for key, value in (result.get("commands") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines).rstrip() + "\n"
