from __future__ import annotations

from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .models import now_iso
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.companion_courseware_text.v1"


def import_companion_courseware_text(
    bundle_dir: str | Path,
    source_path: str | Path,
    *,
    title: str = "",
    write: bool = True,
) -> dict[str, Any]:
    """Register local courseware text without claiming it was visible in video.

    A companion deck/transcript can cover courseware text for summaries, while
    speaker-only video remains exempt from video-frame OCR requirements.
    """

    root = Path(bundle_dir).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    if not (root / "manifest.json").is_file() or not (root / "timeline.json").is_file():
        raise ValueError("bundle_dir must contain manifest.json and timeline.json")
    if not source.is_file():
        raise FileNotFoundError(f"companion courseware text not found: {source}")
    text = source.read_text(encoding="utf-8-sig", errors="replace").strip()
    if not text:
        raise ValueError("companion courseware text is empty")
    exports = root / "exports"
    copied = exports / "companion-courseware-text.md"
    record = exports / "companion-courseware-text.json"
    payload = {
        "schema": SCHEMA,
        "status": "active",
        "kind": "companion_courseware_text",
        "title": str(title or source.stem),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "bundle_copy_path": str(copied),
        "bundle_copy_sha256": "",
        "text": text,
        "evidence_scope": "external_courseware_not_video_frame",
        "screen_text_coverage": "covered_by_external_courseware",
        "structured_visual_coverage": "covered_by_external_courseware",
        "video_frame_ocr_required": False,
        "created_at": now_iso(),
    }
    if write:
        exports.mkdir(parents=True, exist_ok=True)
        copied.write_text(text + "\n", encoding="utf-8")
        payload["bundle_copy_sha256"] = sha256_file(copied)
        write_json(record, payload)
        manifest = read_json(root / "manifest.json")
        if not isinstance(manifest, dict):
            raise ValueError("manifest.json must be a JSON object")
        manifest["companion_courseware_text"] = "exports/companion-courseware-text.json"
        manifest["companion_courseware_text_markdown"] = "exports/companion-courseware-text.md"
        manifest["courseware_evidence_scope"] = payload["evidence_scope"]
        manifest["screen_text_coverage"] = payload["screen_text_coverage"]
        manifest["structured_visual_coverage"] = payload["structured_visual_coverage"]
        write_json(root / "manifest.json", manifest)
    return payload


def load_companion_courseware_text(root: str | Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    bundle = Path(root).expanduser().resolve()
    raw = str(manifest.get("companion_courseware_text") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    path = path if path.is_absolute() else (bundle / path)
    path = path.resolve()
    if not path.is_file() or not path.is_relative_to(bundle):
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("status") != "active":
        return None
    copied = Path(str(payload.get("bundle_copy_path") or "")).resolve()
    if not copied.is_file() or not copied.is_relative_to(bundle):
        return None
    if str(payload.get("bundle_copy_sha256") or "") != sha256_file(copied):
        return None
    return payload
