from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .shot_review import build_shot_review_template, shot_review_status


WAVESURFER_VERSION = "7.12.11"
WAVESURFER_COMMIT = "ae8d3cd32ebb27273051935c01fc6e4001cde3af"
ASSET_RELATIVE_DIR = Path("assets") / f"wavesurfer-{WAVESURFER_VERSION}"
STATIC_ROOT = Path(__file__).resolve().parent / "static"
WAVESURFER_SOURCE = STATIC_ROOT / f"wavesurfer-{WAVESURFER_VERSION}"
GLUE_SOURCE = STATIC_ROOT / "shot-review-workbench.js"


def prepare_shot_review_workbench(
    bundle_dir: str | Path,
    *,
    write: bool,
) -> dict[str, Any]:
    """Prepare local WaveSurfer assets and a freshness-bound review draft."""

    root = Path(bundle_dir).expanduser().resolve()
    required = [
        WAVESURFER_SOURCE / "wavesurfer.min.js",
        WAVESURFER_SOURCE / "regions.min.js",
        WAVESURFER_SOURCE / "LICENSE",
        GLUE_SOURCE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    template = build_shot_review_template(root, write=write)
    if write and not missing:
        destination = root / ASSET_RELATIVE_DIR
        destination.mkdir(parents=True, exist_ok=True)
        for source in required[:3]:
            shutil.copy2(source, destination / source.name)
        shutil.copy2(GLUE_SOURCE, root / "assets" / GLUE_SOURCE.name)
    return {
        "status": (
            "ready"
            if template.get("ok") and not missing
            else template.get("status") or "blocked_missing_assets"
        ),
        "ok": bool(template.get("ok")) and not missing,
        "template": template,
        "applied": shot_review_status(root),
        "asset_paths": {
            "wavesurfer": (ASSET_RELATIVE_DIR / "wavesurfer.min.js").as_posix(),
            "regions": (ASSET_RELATIVE_DIR / "regions.min.js").as_posix(),
            "license": (ASSET_RELATIVE_DIR / "LICENSE").as_posix(),
            "glue": "assets/shot-review-workbench.js",
        },
        "asset_source": {
            "project": "wavesurfer.js",
            "version": WAVESURFER_VERSION,
            "commit": WAVESURFER_COMMIT,
            "license": "BSD-3-Clause",
            "source_files": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path) if path.is_file() else "",
                }
                for path in required
            ],
            "missing": missing,
        },
        "operator_boundary": {
            "browser_draft_storage": "localStorage",
            "formal_apply_requires_loopback_api": True,
            "static_file_cannot_write_bundle": True,
            "no_cloud_call": True,
            "no_new_review_service": True,
        },
    }
