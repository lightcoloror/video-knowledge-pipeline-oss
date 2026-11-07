"""Render the pinned moys-asr-workflow application as a VKP projection.

Intent: reuse the complete upstream editing shell. Decision: inline the pinned
web assets and layer a VKP adapter after them. Reason: the page must work
offline without a CDN or second server. Evidence: Moyf/moys-asr-workflow
v1.3.1 commit 949bc84058cdae1d9c021c50203e6d2742f9392c. Effective scope:
editor rendering only; provider, ASR, FFmpeg, and media mutation stay outside.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .models import now_iso
from .lecture_package import resolve_review_media_path
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .subtitle_editor import build_subtitle_editor_projection
from .transcript import format_timestamp

STATIC_DIR = Path(__file__).with_name("static") / "moys-subtitle-editor"


def prepare_subtitle_editor(
    bundle_dir: str | Path,
    *,
    csrf_token: str = "",
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    projection = build_subtitle_editor_projection(root, write=write)
    page = render_subtitle_editor_page(root, projection=projection, csrf_token=csrf_token)
    output = root / "subtitle-editor.html"
    if write:
        output.write_text(page, encoding="utf-8")
        manifest = _read_object(root / "manifest.json")
        manifest["subtitle_editor_html"] = output.name
        manifest["subtitle_editor_project_json"] = "subtitle-editor-project.json"
        write_json(root / "manifest.json", manifest)
        register_bundle_run(
            root,
            run_type="subtitle_editor",
            run_id="prepare-subtitle-editor",
            status="completed",
            title="双轨字幕编辑器",
            summary=f"Prepared {len(projection['segments'])} source-linked subtitle segments.",
            artifacts=[
                {"key": "html", "path": output},
                {"key": "project", "path": root / "subtitle-editor-project.json"},
            ],
            retry_command=f".\\scripts\\video-knowledge.ps1 prepare-subtitle-editor '{root}'",
            operator_boundary=projection["operator_boundary"],
            write=True,
        )
    return {
        "ok": True,
        "status": "prepared",
        "schema": "video_knowledge_pipeline.prepare_subtitle_editor.v1",
        "bundle_dir": str(root),
        "html_path": str(output),
        "project_path": str(root / "subtitle-editor-project.json"),
        "projection_sha256": projection["projection_sha256"],
        "segment_count": len(projection["segments"]),
        "source_segment_count": len(projection['source_segments']),
        "read_only_gap_count": len(projection['read_only_gaps']),
        "translation_status": projection["tracks"]["mandarin"]["status"],
        "timing_review_status": projection["timing_review"]["status"],
        "timing_overlap_count": projection["timing_review"]["overlap_count"],
        "write": bool(write),
    }


def render_subtitle_editor_page(
    bundle_dir: str | Path,
    *,
    projection: dict[str, Any] | None = None,
    csrf_token: str = "",
    lazy_translation: bool = False,
) -> str:
    root = Path(bundle_dir).expanduser().resolve()
    projection = projection or build_subtitle_editor_projection(root, write=False)
    manifest = _read_object(root / "manifest.json")
    media_path = _media_path(root, manifest)
    audio_extensions = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
    media_tag = "audio" if media_path.suffix.lower() in audio_extensions else "video"
    media_html = (
        f'<{media_tag} id="player" controls preload="metadata" '
        f'src="/media" style="width:100%;display:block;"></{media_tag}>'
    )
    gaps = projection.get('read_only_gaps') or []
    if gaps:
        gap_rows = ''.join(
            '<li data-segment-id="' + html.escape(str(row['segment_id']), quote=True) + '">'
            + html.escape(format_timestamp(row['start_ms'] / 1000)) + ' – '
            + html.escape(format_timestamp(row['end_ms'] / 1000))
            + '：原转写为空，保留待核。</li>' for row in gaps)
        media_html += ('<details id="vkp-read-only-gaps"><summary>'
            + str(len(gaps)) + ' 处原转写缺口（只读，保存不会删除）</summary><ul>'
            + gap_rows + '</ul></details>')
    stickers = _approved_stickers(root)
    client_projection = _client_projection(projection, lazy_translation=lazy_translation)
    server_config = {
        "saveUrl": None,
        "canSave": False,
        "recentProjectsUrl": None,
        "attachUrl": None,
        "settingsUrl": None,
        "autoLoadedMediaName": media_path.name,
        "vkp": {
            "bundleId": projection["source_sha256"],
            "csrfToken": csrf_token,
            "validateUrl": "/api/subtitle-editor/validate",
            "applyUrl": "/api/subtitle-editor/apply",
            "translationUrl": "/api/subtitle-editor/translations",
            "lazyTranslation": bool(lazy_translation),
            "approvedStickerOnly": True,
            "projection": client_projection,
        },
    }
    replacements = {
        "__EDITOR_CSS__": _asset("editor.css").rstrip(),
        "__WAVEFORM_CSS__": _asset("waveform.css").rstrip(),
        "__VKP_ADAPTER_CSS__": _asset("vkp-adapter.css").rstrip(),
        "__EDITOR_UTILS_JS__": _asset("editor-utils.js").rstrip(),
        "__EDITOR_I18N_JS__": _asset("editor-i18n.js").rstrip(),
        "__WAVEFORM_JS__": _asset("waveform.js").rstrip(),
        "__EDITOR_JS__": _asset("editor.js").rstrip(),
        "__VKP_ADAPTER_JS__": _asset("vkp-adapter.js").rstrip(),
        "__TITLE__": html.escape(f"VKP 字幕编辑器 · {projection['title']}"),
        "__MEDIA_HTML__": media_html,
        "__DATA_JSON__": _script_json(
            _upstream_project(projection, lazy_translation=lazy_translation)
        ),
        "__FILENAME_BASE_JSON__": _script_json(_safe_filename(projection["title"])),
        "__STICKERS_JSON__": _script_json(stickers),
        "__STICKER_ROOT_JSON__": '""',
        "__STICKER_URL_PREFIX_JSON__": '"/stickers"' if stickers else '""',
        "__SERVER_CONFIG_JSON__": _script_json(server_config),
        "__UI_LANGUAGE_JSON__": '"zh-CN"',
        "__GENERATED_AT__": html.escape(now_iso()),
        "__JSON_DISPLAY__": "VKP 只读投影",
        "__JSON_NAME_CLASS__": "",
        "__MEDIA_NAME_DISPLAY__": html.escape(media_path.name),
        "__MEDIA_NAME_TITLE__": html.escape(f"媒体：{media_path.name}"),
        "__MEDIA_NAME_CLASS__": "",
    }
    page = _asset("editor-template.html")
    for token, value in replacements.items():
        page = page.replace(token, value)
    unresolved = sorted(token for token in replacements if token in page)
    if unresolved:
        raise ValueError(f"unresolved subtitle editor template tokens: {unresolved}")
    return page


def _upstream_project(
    projection: dict[str, Any],
    *,
    lazy_translation: bool = False,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    for row in projection["segments"]:
        color = _speaker_color(str(row.get("speaker_global_id") or ""), row["start_ms"], row["end_ms"])
        items = [
            {
                "text": word["text"],
                "start": word["start_ms"],
                "end": word["end_ms"],
                **(
                    {"speaker": word["speaker_global_id"]}
                    if word.get("speaker_global_id")
                    else {}
                ),
            }
            for word in row.get("words", [])
        ]
        segments.append(
            {
                "start": row["start_ms"],
                "end": row["end_ms"],
                "text": row["source_text"],
                "mandarin_text": "" if lazy_translation else row["mandarin_text"],
                "mandarin_loaded": not lazy_translation or not bool(row["mandarin_text"]),
                "translation_available": bool(row["mandarin_text"]),
                "segment_id": row["segment_id"],
                "source_segment_ids": row["source_segment_ids"],
                "source_lineage_ids": row["source_lineage_ids"],
                "speaker": row["speaker_global_id"] or None,
                "speaker_global_id": row["speaker_global_id"],
                "speaker_role": row["speaker_role"],
                "evidence_ids": row["evidence_ids"],
                "items": items or None,
                "color": color,
                "color_ref": None,
                "disabled": bool(row.get("disabled")),
                "timing_status": row.get("timing_status") or "ready",
            }
        )
    return {
        "schema": "moy.asr.project.v1",
        "language": projection["tracks"]["source"]["language"],
        "model": "vkp-projection",
        "segments": segments,
        "vkp_projection": {
            "schema": projection["schema"],
            "projection_sha256": projection["projection_sha256"],
            "source_sha256": projection["source_sha256"],
            "translation_status": projection["tracks"]["mandarin"]["status"],
        },
    }


def _client_projection(
    projection: dict[str, Any],
    *,
    lazy_translation: bool,
) -> dict[str, Any]:
    if not lazy_translation:
        return projection
    # Intent: lazy-load derived translations without changing the projection
    # identity. Decision: strip only mandarin_text from the inline client copy;
    # the exact sidecar remains available through the bounded loopback slice.
    # Reason: long bilingual transcripts should not inflate initial page load.
    # Evidence: YouTube Digest 1.1.5 IntersectionObserver/generation pattern.
    # Effective scope: Review Server HTML serialization only.
    value = json.loads(json.dumps(projection, ensure_ascii=False))
    for row in value.get("segments") or []:
        if isinstance(row, dict):
            row["translation_available"] = bool(row.get("mandarin_text"))
            row["mandarin_text"] = ""
    return value


def _speaker_color(speaker_id: str, start_ms: int, end_ms: int) -> dict[str, Any] | None:
    if not speaker_id:
        return None
    palette = [
        ("blue", "#168cff"),
        ("green", "#2ecc71"),
        ("purple", "#9b59b6"),
        ("yellow", "#f1c40f"),
        ("red", "#e74c3c"),
    ]
    index = sum(speaker_id.encode("utf-8")) % len(palette)
    name, value = palette[index]
    return {"name": name, "value": value, "start": start_ms, "end": end_ms}


def _approved_stickers(root: Path) -> list[dict[str, str]]:
    sticker_root = root / "stickers"
    if not sticker_root.is_dir():
        return []
    extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    result: list[dict[str, str]] = []
    for path in sorted(sticker_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        relative = path.relative_to(sticker_root).as_posix()
        result.append(
            {
                "name": Path(relative).with_suffix("").as_posix(),
                "filename": path.name,
                "rel": relative,
            }
        )
        if len(result) >= 500:
            break
    return result


def _media_path(root: Path, manifest: dict[str, Any]) -> Path:
    path = resolve_review_media_path(root, manifest)
    if path is None:
        raise ValueError("subtitle editor requires registered Bundle or source-package media")
    return path


def _asset(name: str) -> str:
    path = STATIC_DIR / name
    if not path.is_file():
        raise ValueError(f"vendored subtitle editor asset is missing: {path}")
    return path.read_text(encoding="utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def _script_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _safe_filename(value: object) -> str:
    text = str(value or "subtitle").strip() or "subtitle"
    return "".join("_" if char in '<>:"/\\|?*' else char for char in text)
