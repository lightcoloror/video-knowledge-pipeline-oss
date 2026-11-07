from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from video_knowledge_pipeline.ui_shell_assets import ASSET_SPECS, STATIC_DIR, install_ui_shell_assets


def test_packaged_ui_shell_assets_match_pinned_hashes() -> None:
    for spec in ASSET_SPECS:
        source = STATIC_DIR / spec["filename"]
        assert source.is_file()
        assert sha256(source.read_bytes()).hexdigest() == spec["sha256"]


def test_asset_installer_copies_offline_assets_and_support_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"

    result = install_ui_shell_assets(bundle, write=True)

    assert result["schema"] == "video_knowledge_pipeline.ui_shell_assets.v1"
    assert result["operator_boundary"]["no_runtime_cdn"] is True
    assert result["assets"]["tabler_css"]["href"] == "assets/vkp-ui-shell/tabler-1.4.0.min.css"
    assert result["assets"]["media_chrome_js"]["href"] == "assets/vkp-ui-shell/media-chrome-4.19.2.iife.min.js"
    for row in result["assets"].values():
        target = Path(row["path"])
        assert target.is_file()
        assert sha256(target.read_bytes()).hexdigest() == row["sha256"]
    assert {row["filename"] for row in result["support_files"]} == {
        "LICENSE.tabler.txt",
        "LICENSE.media-chrome.txt",
        "PROVENANCE.json",
    }


def test_asset_installer_plan_mode_does_not_create_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "plan-only"

    result = install_ui_shell_assets(bundle, write=False)

    assert result["write"] is False
    assert bundle.exists() is False
    assert all(row["exists"] is False for row in result["assets"].values())
