#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any


DEFAULT_API_BASE = "http://host.docker.internal:8931"
DEFAULT_CONTAINER_ROOT = "/mnt/used-by-codex"
DEFAULT_HOST_ROOT = os.environ.get("VKP_HOST_ROOT", r"C:\workspace")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    value = sys.stdin.read() if args.value == "-" else (args.value or "")
    payload = build_call_payload(args.command, value, vars(args))
    if args.print_payload:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    result = post_call(args.api_base, payload, timeout_seconds=args.timeout_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", False) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Docker/OpenClaw client for video-knowledge-pipeline /call.")
    parser.add_argument("command", choices=["status", "contract", "live-smoke", "plan", "ingest", "link", "handoff", "ingest-handoff", "content-status", "batch-content-status", "content-handoff-pack"])
    parser.add_argument("value", nargs="?", default="", help="URL/text/media path, or '-' to read stdin for plan/link text")
    parser.add_argument("--api-base", default=os.environ.get("VKP_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--mount-container-root", default=os.environ.get("VKP_CONTAINER_ROOT", DEFAULT_CONTAINER_ROOT))
    parser.add_argument("--mount-host-root", default=os.environ.get("VKP_HOST_ROOT", DEFAULT_HOST_ROOT))
    parser.add_argument("--workspace", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--actor-id", default="")
    parser.add_argument("--confirm-download", action="store_true")
    parser.add_argument("--confirm-sensitive", action="store_true")
    parser.add_argument("--ingest-after-download", action="store_true")
    parser.add_argument("--downloaded-media-path", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--review-checklist-path", default="")
    parser.add_argument("--media-path", default="")
    parser.add_argument("--handoff-path", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--semantic-batch-input", default="")
    parser.add_argument("--semantic-target-bundle-count", type=int, default=3)
    parser.add_argument("--semantic-limit", type=int, default=0, help="Maximum semantic-correction bundles to inspect; 0 means all discovered.")
    parser.add_argument("--print-payload", action="store_true", help="Print the /call JSON payload without making an HTTP request.")
    return parser


def build_call_payload(command: str, value: str, options: dict[str, Any]) -> dict[str, Any]:
    if command == "status":
        return {"name": "openclaw_bridge_status", "arguments": {"check_health": True}}
    if command == "contract":
        return {
            "name": "openclaw_docker_contract_check",
            "arguments": {
                "host_root": str(options.get("mount_host_root") or DEFAULT_HOST_ROOT),
                "container_root": str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT),
            },
        }
    if command == "live-smoke":
        container_root = str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT)
        host_root = str(options.get("mount_host_root") or DEFAULT_HOST_ROOT)
        bundle_dir = translate_container_path(value, container_root=container_root, host_root=host_root)
        semantic_batch_input = translate_container_path(
            str(options.get("semantic_batch_input") or ""),
            container_root=container_root,
            host_root=host_root,
        )
        return {
            "name": "openclaw_live_smoke",
            "arguments": {
                "bundle_dir": bundle_dir,
                "host_root": host_root,
                "container_root": container_root,
                "semantic_batch_input": semantic_batch_input,
                "semantic_target_bundle_count": int(options.get("semantic_target_bundle_count") or 3),
                "semantic_limit": int(options.get("semantic_limit") or 0),
            },
        }
    if command == "plan":
        return {"name": "openclaw_video_plan", "arguments": {"url_or_text": value}}
    if command == "ingest":
        media_path = translate_container_path(
            value,
            container_root=str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT),
            host_root=str(options.get("mount_host_root") or DEFAULT_HOST_ROOT),
        )
        workspace = translate_container_path(
            str(options.get("workspace") or ""),
            container_root=str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT),
            host_root=str(options.get("mount_host_root") or DEFAULT_HOST_ROOT),
        )
        arguments = {
            "media_path": media_path,
            "workspace": workspace,
            "title": str(options.get("title") or ""),
            "max_frames": int(options.get("max_frames") or 80),
            "sample_interval": float(options.get("sample_interval") or 5.0),
        }
        return {"name": "openclaw_video_ingest", "arguments": arguments}
    if command == "link":
        downloaded_media_path = translate_container_path(
            str(options.get("downloaded_media_path") or ""),
            container_root=str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT),
            host_root=str(options.get("mount_host_root") or DEFAULT_HOST_ROOT),
        )
        workspace = translate_container_path(
            str(options.get("workspace") or ""),
            container_root=str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT),
            host_root=str(options.get("mount_host_root") or DEFAULT_HOST_ROOT),
        )
        return {
            "name": "openclaw_video_link",
            "arguments": {
                "url_or_text": value,
                "allow_download": bool(options.get("allow_download")),
                "actor_id": str(options.get("actor_id") or ""),
                "confirm_download": bool(options.get("confirm_download")),
                "confirm_sensitive": bool(options.get("confirm_sensitive")),
                "ingest_after_download": bool(options.get("ingest_after_download")),
                "downloaded_media_path": downloaded_media_path,
                "workspace": workspace,
                "title": str(options.get("title") or ""),
                "max_frames": int(options.get("max_frames") or 80),
                "sample_interval": float(options.get("sample_interval") or 5.0),
            },
        }
    if command == "handoff":
        container_root = str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT)
        host_root = str(options.get("mount_host_root") or DEFAULT_HOST_ROOT)
        arguments = {
            "manifest_path": translate_container_path(str(options.get("manifest_path") or ""), container_root=container_root, host_root=host_root),
            "summary_path": translate_container_path(str(options.get("summary_path") or ""), container_root=container_root, host_root=host_root),
            "review_checklist_path": translate_container_path(str(options.get("review_checklist_path") or ""), container_root=container_root, host_root=host_root),
            "media_path": translate_container_path(str(options.get("media_path") or value or ""), container_root=container_root, host_root=host_root),
            "host_root": host_root,
            "container_root": container_root,
            "workspace": translate_container_path(str(options.get("workspace") or ""), container_root=container_root, host_root=host_root),
            "title": str(options.get("title") or ""),
        }
        return {"name": "openclaw_video_from_vdo_handoff", "arguments": arguments}
    if command == "ingest-handoff":
        container_root = str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT)
        host_root = str(options.get("mount_host_root") or DEFAULT_HOST_ROOT)
        arguments = {
            "handoff_path": translate_container_path(str(options.get("handoff_path") or ""), container_root=container_root, host_root=host_root),
            "manifest_path": translate_container_path(str(options.get("manifest_path") or ""), container_root=container_root, host_root=host_root),
            "summary_path": translate_container_path(str(options.get("summary_path") or ""), container_root=container_root, host_root=host_root),
            "review_checklist_path": translate_container_path(str(options.get("review_checklist_path") or ""), container_root=container_root, host_root=host_root),
            "media_path": translate_container_path(str(options.get("media_path") or value or ""), container_root=container_root, host_root=host_root),
            "host_root": host_root,
            "container_root": container_root,
            "workspace": translate_container_path(str(options.get("workspace") or ""), container_root=container_root, host_root=host_root),
            "title": str(options.get("title") or ""),
            "execute": bool(options.get("execute")),
            "max_frames": int(options.get("max_frames") or 80),
            "sample_interval": float(options.get("sample_interval") or 5.0),
        }
        return {"name": "openclaw_video_ingest_vdo_handoff", "arguments": arguments}
    if command == "content-status":
        bundle_dir = translate_container_path(
            value,
            container_root=str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT),
            host_root=str(options.get("mount_host_root") or DEFAULT_HOST_ROOT),
        )
        return {"name": "content_asset_status", "arguments": {"bundle_dir": bundle_dir}}
    if command == "batch-content-status":
        batch_input = translate_container_path(
            value,
            container_root=str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT),
            host_root=str(options.get("mount_host_root") or DEFAULT_HOST_ROOT),
        )
        return {"name": "batch_content_asset_status", "arguments": {"batch_input": batch_input}}
    if command == "content-handoff-pack":
        batch_input = translate_container_path(
            value,
            container_root=str(options.get("mount_container_root") or DEFAULT_CONTAINER_ROOT),
            host_root=str(options.get("mount_host_root") or DEFAULT_HOST_ROOT),
        )
        return {"name": "content_handoff_pack", "arguments": {"batch_input": batch_input}}
    raise ValueError(f"unsupported command: {command}")


def translate_container_path(path: str, *, container_root: str, host_root: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    normalized_root = str(PurePosixPath(container_root or DEFAULT_CONTAINER_ROOT))
    normalized = raw.replace("\\", "/")
    if normalized == normalized_root or normalized.startswith(normalized_root.rstrip("/") + "/"):
        rel = normalized[len(normalized_root) :].lstrip("/")
        host = PureWindowsPath(host_root or DEFAULT_HOST_ROOT)
        for part in PurePosixPath(rel).parts:
            host = host / part
        return str(host)
    return raw


def post_call(api_base: str, payload: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/call"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read().decode("utf-8"))
        except Exception:
            return {"ok": False, "code": f"http_{exc.code}", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "code": "request_failed", "message": str(exc), "url": url}


if __name__ == "__main__":
    raise SystemExit(main())
