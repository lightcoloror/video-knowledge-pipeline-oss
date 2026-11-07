from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from .stage_cache import atomic_copy_file


SCHEMA = "video_knowledge_pipeline.ui_shell_assets.v1"
STATIC_DIR = Path(__file__).with_name("static") / "vkp-ui-shell"
TARGET_DIR = Path("assets") / "vkp-ui-shell"

ASSET_SPECS: tuple[dict[str, str], ...] = (
    {
        "key": "tabler_css",
        "filename": "tabler-1.4.0.min.css",
        "sha256": "530253779f4ea7ecdf8c867195c9f54a66edd806596cce5ce2cc45cb33feacd7",
        "source": "Tabler 1.4.0",
    },
    {
        "key": "tabler_js",
        "filename": "tabler-1.4.0.min.js",
        "sha256": "b88ed24c6731f562d09686cae1560e78501c5d39f7189a10d124d72405856314",
        "source": "Tabler 1.4.0",
    },
    {
        "key": "media_chrome_js",
        "filename": "media-chrome-4.19.2.iife.min.js",
        "sha256": "16b5cb45b456ebae89aebcea6fdad297fc83f5497766b45511e99264e2d577b3",
        "source": "Media Chrome 4.19.2",
    },
)

SUPPORT_FILES: tuple[dict[str, str], ...] = (
    {
        "filename": "LICENSE.tabler.txt",
        "sha256": "58f3e9690546f1550d40084dc84a5180490e3abed591ba3480abc11647cfd715",
    },
    {
        "filename": "LICENSE.media-chrome.txt",
        "sha256": "24d3f27c193524e8ddfb3c3218e8982673a34d439b81fb10a7da880aa3dd5d97",
    },
    {
        "filename": "PROVENANCE.json",
        "sha256": "b37e94eb1bd1a573f0d0a51318fb12afda5f93ee4c205d8b16eee9154b446b46",
    },
)


def install_ui_shell_assets(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Validate and install pinned, offline UI shell assets into a bundle.

    The installer copies only package-owned files. It never downloads assets,
    starts a service, processes media, or mutates third-party source checkouts.
    """

    root = Path(bundle_dir).expanduser().resolve()
    target_dir = root / TARGET_DIR
    assets: dict[str, dict[str, Any]] = {}

    for spec in ASSET_SPECS:
        source = STATIC_DIR / spec["filename"]
        actual_hash = _verified_source_hash(source, expected=spec["sha256"])
        target = target_dir / spec["filename"]
        if write:
            atomic_copy_file(source, target)
        assets[spec["key"]] = {
            "filename": spec["filename"],
            "href": (TARGET_DIR / spec["filename"]).as_posix(),
            "path": str(target),
            "exists": target.is_file() if write else False,
            "sha256": actual_hash,
            "source": spec["source"],
        }

    support_files: list[dict[str, Any]] = []
    for spec in SUPPORT_FILES:
        source = STATIC_DIR / spec["filename"]
        actual_hash = _verified_source_hash(source, expected=spec["sha256"] or None)
        target = target_dir / spec["filename"]
        if write:
            atomic_copy_file(source, target)
        support_files.append(
            {
                "filename": spec["filename"],
                "href": (TARGET_DIR / spec["filename"]).as_posix(),
                "path": str(target),
                "exists": target.is_file() if write else False,
                "sha256": actual_hash,
            }
        )

    return {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "target_dir": str(target_dir),
        "write": write,
        "assets": assets,
        "support_files": support_files,
        "operator_boundary": {
            "local_only": True,
            "no_runtime_cdn": True,
            "no_network_call": True,
            "no_service_started": True,
            "no_media_processing": True,
        },
    }


def _verified_source_hash(path: Path, *, expected: str | None) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"UI shell package asset not found: {path}")
    actual = sha256(path.read_bytes()).hexdigest()
    if expected and actual != expected.lower():
        raise ValueError(f"UI shell package asset hash mismatch: {path.name}; expected={expected}; actual={actual}")
    return actual
