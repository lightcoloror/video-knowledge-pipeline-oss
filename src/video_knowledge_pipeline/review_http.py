from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import mimetypes
import secrets
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .review_writeback import apply_review_payload_to_bundle
from .lecture_package import resolve_review_media_path
from .shot_review import apply_shot_review_notes
from .subtitle_editor import (
    apply_subtitle_review,
    build_subtitle_editor_projection,
    subtitle_translation_slice,
    validate_subtitle_review,
)
from .subtitle_editor_ui import render_subtitle_editor_page
from .video_workbench import export_video_workbench
from .storage import read_json
from .webui_bridge import refresh_bundle_review_html


MAX_REQUEST_BYTES = 2 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


def build_server(
    bundle_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    csrf_token: str | None = None,
    refresh: bool = True,
) -> ThreadingHTTPServer:
    _require_loopback_host(host)
    root = Path(bundle_dir).expanduser().resolve()
    _require_bundle(root)
    if refresh:
        refresh_bundle_review_html(root, write=True)
    token = str(csrf_token or secrets.token_urlsafe(32))

    class Handler(ReviewHandler):
        pass

    Handler.bundle_dir = root
    Handler.csrf_token = token
    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    return server


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "VKPReview/1.0"
    bundle_dir: Path = Path()
    csrf_token: str = ""

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            return self._json_error(HTTPStatus.MISDIRECTED_REQUEST, "invalid Host header")
        parsed_url = urlsplit(self.path)
        path = parsed_url.path
        if path in {"/", "/review.html"}:
            return self._serve_review_html()
        if path in {"/workbench", "/video-workbench.html"}:
            return self._serve_workbench_html()
        if path in {"/subtitle-editor", "/subtitle-editor.html"}:
            return self._serve_subtitle_editor_html()
        if path == "/api/subtitle-editor/project":
            return self._json_response(build_subtitle_editor_projection(self.bundle_dir, write=False))
        if path == "/api/subtitle-editor/translations":
            try:
                query = parse_qs(parsed_url.query, keep_blank_values=True)
                generation = int((query.get("generation") or ["0"])[0] or 0)
                return self._json_response(
                    subtitle_translation_slice(
                        self.bundle_dir,
                        projection_sha256=str((query.get("projection_sha256") or [""])[0]),
                        segment_ids=[str(value) for value in query.get("segment_id") or []],
                        generation=generation,
                    )
                )
            except (TypeError, ValueError) as exc:
                return self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
        if path == "/api/review/status":
            return self._json_response(
                {
                    "ok": True,
                    "service": "vkp-review",
                    "loopback_only": True,
                    "bundle_dir": str(self.bundle_dir),
                    "bundle_revision": review_bundle_revision(self.bundle_dir),
                    "writeback_available": True,
                }
            )
        if path == "/media":
            media_path = _bundle_media_path(self.bundle_dir)
            if media_path is None:
                return self._json_error(HTTPStatus.NOT_FOUND, "bundle media not found")
            return self._serve_file(media_path, allow_range=True)
        candidate = _safe_bundle_file(self.bundle_dir, path)
        if candidate is None or not candidate.is_file():
            return self._json_error(HTTPStatus.NOT_FOUND, "not found")
        return self._serve_file(candidate, allow_range=False)

    def do_POST(self) -> None:  # noqa: N802
        if not self._prepare_mutation():
            return
        endpoint = urlsplit(self.path).path
        if endpoint not in {
            "/api/review/apply",
            "/api/shot-review/apply",
            "/api/subtitle-editor/validate",
            "/api/subtitle-editor/apply",
        }:
            return self._json_error(HTTPStatus.NOT_FOUND, "not found")
        try:
            request = self._read_json_body()
            if endpoint.startswith("/api/subtitle-editor/"):
                projection = build_subtitle_editor_projection(self.bundle_dir, write=False)
                expected = str(request.get("bundle_revision") or "")
                if not expected or expected != str(projection["bundle_revision"]):
                    return self._json_error(
                        HTTPStatus.CONFLICT,
                        "subtitle projection changed after this page was loaded; reload before applying",
                        extra={"bundle_revision": projection["bundle_revision"]},
                    )
                notes = request.get("subtitle_review_notes")
                if not isinstance(notes, dict):
                    raise ValueError("subtitle_review_notes must be a JSON object")
                if endpoint == "/api/subtitle-editor/validate":
                    validated = validate_subtitle_review(self.bundle_dir, notes)
                    return self._json_response(
                        {
                            "ok": True,
                            "status": "validated",
                            "summary": validated["summary"],
                            "bundle_revision": projection["bundle_revision"],
                        }
                    )
                result = apply_subtitle_review(self.bundle_dir, review_json=notes, write=True)
                return self._json_response(
                    {
                        "ok": bool(result.get("ok")),
                        "status": result.get("status"),
                        "summary": result.get("summary"),
                        "bundle_revision": projection["bundle_revision"],
                    },
                    HTTPStatus.OK if result.get("ok") else HTTPStatus.UNPROCESSABLE_ENTITY,
                )
            expected = str(request.get("bundle_revision") or "")
            current = review_bundle_revision(self.bundle_dir)
            if not expected or expected != current:
                return self._json_error(
                    HTTPStatus.CONFLICT,
                    "bundle changed after this review page was loaded; reload before applying",
                    extra={"bundle_revision": current},
                )
            if endpoint == "/api/shot-review/apply":
                notes = request.get("shot_review_notes")
                if not isinstance(notes, dict):
                    raise ValueError("shot_review_notes must be a JSON object")
                result = apply_shot_review_notes(self.bundle_dir, notes, write=True)
                workbench = export_video_workbench(self.bundle_dir, write=True)
                return self._json_response(
                    {
                        "ok": bool(result.get("ok")),
                        "status": result.get("status"),
                        "writeback": result,
                        "workbench": workbench.get("paths"),
                        "bundle_revision": review_bundle_revision(self.bundle_dir),
                    },
                    HTTPStatus.OK if result.get("ok") else HTTPStatus.UNPROCESSABLE_ENTITY,
                )
            notes = request.get("review_notes")
            if not isinstance(notes, dict):
                raise ValueError("review_notes must be a JSON object")
            result = apply_review_payload_to_bundle(self.bundle_dir, notes, write=True, refresh_exports=True)
            refresh = refresh_bundle_review_html(self.bundle_dir, write=True)
            status = HTTPStatus.OK if result.get("ok") else HTTPStatus.UNPROCESSABLE_ENTITY
            return self._json_response(
                {
                    "ok": bool(result.get("ok")),
                    "status": result.get("status"),
                    "writeback": result,
                    "review_refresh": refresh,
                    "bundle_revision": review_bundle_revision(self.bundle_dir),
                },
                status,
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except (RuntimeError, OSError) as exc:
            return self._json_error(HTTPStatus.CONFLICT, f"{type(exc).__name__}: {exc}")

    def log_message(self, format: str, *args: Any) -> None:
        safe_args = tuple(str(value).replace("\r", " ").replace("\n", " ") for value in args)
        super().log_message(format, *safe_args)

    def _serve_review_html(self) -> None:
        path = self.bundle_dir / "review.html"
        if not path.is_file():
            refresh_bundle_review_html(self.bundle_dir, write=True)
        html_text = path.read_text(encoding="utf-8")
        config = json.dumps(
            {
                "apply_url": "/api/review/apply",
                "status_url": "/api/review/status",
                "token": self.csrf_token,
                "bundle_revision": review_bundle_revision(self.bundle_dir),
            },
            ensure_ascii=False,
        ).replace("</", "<\\/")
        bootstrap = f"<script>window.VKP_REVIEW_API = {config};</script>"
        html_text = html_text.replace("</head>", bootstrap + "</head>", 1)
        media = _bundle_media_path(self.bundle_dir)
        if media is not None:
            html_text = html_text.replace(media.as_uri(), "/media")
        payload = html_text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._security_headers("text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_workbench_html(self) -> None:
        path = self.bundle_dir / "video-workbench.html"
        if not path.is_file():
            export_video_workbench(self.bundle_dir, write=True)
        html_text = path.read_text(encoding="utf-8")
        config = json.dumps(
            {
                "apply_url": "/api/shot-review/apply",
                "status_url": "/api/review/status",
                "media_url": "/media",
                "token": self.csrf_token,
                "bundle_revision": review_bundle_revision(self.bundle_dir),
            },
            ensure_ascii=False,
        ).replace("</", "<\\/")
        bootstrap = f"<script>window.VKP_SHOT_REVIEW_API = {config};</script>"
        html_text = html_text.replace("</head>", bootstrap + "</head>", 1)
        payload = html_text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._security_headers("text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_subtitle_editor_html(self) -> None:
        projection = build_subtitle_editor_projection(self.bundle_dir, write=False)
        html_text = render_subtitle_editor_page(
            self.bundle_dir,
            projection=projection,
            csrf_token=self.csrf_token,
            lazy_translation=True,
        )
        payload = html_text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._security_headers("text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, path: Path, *, allow_range: bool) -> None:
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        if allow_range and self.headers.get("Range"):
            parsed = _parse_range(self.headers.get("Range", ""), size)
            if parsed is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = parsed
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(status)
        self._security_headers(content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _prepare_mutation(self) -> bool:
        if not self._host_allowed():
            self._json_error(HTTPStatus.MISDIRECTED_REQUEST, "invalid Host header")
            return False
        if self.headers.get("X-VKP-Review-Token", "") != self.csrf_token:
            self._json_error(HTTPStatus.FORBIDDEN, "invalid review token")
            return False
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._json_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
            return False
        origin = str(self.headers.get("Origin") or "")
        expected_origin = f"http://{self.headers.get('Host', '')}"
        if origin and origin != expected_origin:
            self._json_error(HTTPStatus.FORBIDDEN, "cross-origin review mutation is not allowed")
            return False
        return True

    def _host_allowed(self) -> bool:
        raw = str(self.headers.get("Host") or "").strip()
        if not raw:
            return False
        host = raw
        if raw.startswith("[") and "]" in raw:
            host = raw[1 : raw.index("]")]
        elif raw.count(":") == 1:
            host = raw.rsplit(":", 1)[0]
        return _is_loopback_host(host)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(str(self.headers.get("Content-Length") or "0"))
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _json_response(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self._security_headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json_error(self, status: HTTPStatus, message: str, *, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"ok": False, "error": str(message)}
        if extra:
            payload.update(extra)
        self._json_response(payload, status)

    def _security_headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )


def review_bundle_revision(bundle_dir: str | Path) -> str:
    root = Path(bundle_dir).expanduser().resolve()
    digest = hashlib.sha256()
    for name in (
        "manifest.json",
        "timeline.json",
        "transcript-semantic-correction-pack.json",
        "exports/technical-shot-boundaries.json",
        "exports/technical-shot-boundaries.reviewed.json",
        "exports/technical-shot-boundary-fusion.json",
        "exports/shot-facts.json",
    ):
        path = root / name
        digest.update(name.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _bundle_media_path(root: Path) -> Path | None:
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        return None
    return resolve_review_media_path(root, manifest)


def _safe_bundle_file(root: Path, request_path: str) -> Path | None:
    relative = unquote(request_path).lstrip("/")
    if not relative or "\\" in relative:
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _parse_range(value: str, size: int) -> tuple[int, int] | None:
    if not value.startswith("bytes=") or "," in value or size <= 0:
        return None
    raw_start, separator, raw_end = value.removeprefix("bytes=").partition("-")
    if not separator:
        return None
    try:
        if raw_start:
            start = int(raw_start)
            end = int(raw_end) if raw_end else size - 1
        else:
            suffix = int(raw_end)
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            end = size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or end < start:
        return None
    return start, min(end, size - 1)


def _require_bundle(root: Path) -> None:
    if not (root / "manifest.json").is_file() or not (root / "timeline.json").is_file():
        raise ValueError(f"not a VKP webui bundle: {root}")


def _is_loopback_host(value: str) -> bool:
    host = str(value or "").strip().lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_loopback_host(value: str) -> None:
    if not _is_loopback_host(value):
        raise ValueError("review UI must bind to a loopback host")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the loopback-only VKP review UI with explicit local writeback")
    parser.add_argument("bundle_dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args(argv)
    server = build_server(
        args.bundle_dir,
        host=args.host,
        port=args.port,
        refresh=not args.no_refresh,
    )
    address, port = server.server_address[:2]
    url = f"http://{address}:{port}/"
    print(f"VKP review UI: {url}", flush=True)
    print(f"VKP Workbench: {url.rstrip('/')}/workbench", flush=True)
    print(f"VKP subtitle editor: {url.rstrip('/')}/subtitle-editor", flush=True)
    print("Loopback only. Drafts stay in browser storage until Save to VKP is clicked.", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
