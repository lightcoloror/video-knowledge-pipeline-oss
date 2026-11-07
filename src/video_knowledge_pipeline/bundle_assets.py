from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .bundle_readiness import build_bundle_readiness
from .models import now_iso
from .storage import read_json, write_json


def repair_bundle_assets(bundle_dir: str | Path) -> dict[str, Any]:
    """Recopy missing WebUI bundle frame assets from their recorded source paths."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline_data, list):
        raise ValueError("timeline.json must be an array")

    copied: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for position, item in enumerate(timeline_data, start=1):
        if not isinstance(item, dict):
            continue
        item_index = _int_or_none(item.get("index")) or position
        for asset in item.get("assets") or []:
            if not isinstance(asset, dict):
                continue
            result = _repair_asset(root, asset, item_index=item_index)
            if result["status"] == "copied":
                copied.append(result)
            elif result["status"] == "missing_source":
                missing.append(result)
            else:
                skipped.append(result)

    _sync_manifest_assets(manifest, timeline_data)
    source_sync = _sync_source_package_assets(root, manifest, timeline_data)
    manifest["asset_manifest"] = (
        manifest.get("asset_manifest") or "assets/asset-manifest.json"
    )
    manifest["asset_repair"] = {
        "schema": "lecture_bundle_asset_repair.v1",
        "updated_at": now_iso(),
        "copied_count": len(copied),
        "missing_count": len(missing),
        "skipped_count": len(skipped),
        "source_package_updated": source_sync.get("updated", 0),
    }
    timeline = [item for item in timeline_data if isinstance(item, dict)]
    manifest["review_readiness"] = build_bundle_readiness(
        manifest, timeline, bundle_dir=root
    )
    write_json(timeline_path, timeline_data)
    write_json(manifest_path, manifest)
    asset_manifest_path = root / str(
        manifest.get("asset_manifest") or "assets/asset-manifest.json"
    )
    write_json(
        asset_manifest_path,
        _asset_manifest(
            manifest.get("assets") if isinstance(manifest.get("assets"), list) else []
        ),
    )
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "asset_manifest_path": str(asset_manifest_path),
        "summary": {
            "copied": len(copied),
            "missing": len(missing),
            "skipped": len(skipped),
            "source_package_updated": source_sync.get("updated", 0),
        },
        "copied": copied,
        "missing": missing,
        "skipped": skipped,
        "source_package_sync": source_sync,
        "review_readiness": manifest["review_readiness"],
    }


def _repair_asset(
    root: Path,
    asset: dict[str, Any],
    *,
    item_index: int,
) -> dict[str, str]:
    source_text = str(asset.get("source") or "").strip()
    path_text = str(asset.get("path") or "").strip()
    if not source_text or not path_text:
        return {
            "status": "skipped",
            "source": source_text,
            "path": path_text,
            "reason": "missing_source_or_path",
        }
    source = Path(source_text).expanduser()
    target = Path(path_text).expanduser()
    if target.is_absolute():
        try:
            relative_target = target.resolve().relative_to(root)
        except ValueError:
            relative_target = Path("assets") / f"{item_index:04d}-{source.name}"
        asset["path"] = relative_target.as_posix()
        target = root / relative_target
    else:
        target = root / target
    if str(asset.get("copied") or "").lower() == "true" and target.exists():
        return {
            "status": "skipped",
            "source": str(source),
            "path": str(target),
            "reason": "already_copied",
        }
    if not source.exists() or not source.is_file():
        asset["copied"] = "false"
        asset["exists"] = False
        return {
            "status": "missing_source",
            "source": str(source),
            "path": str(target),
            "reason": "source_not_found",
        }
    if source.resolve() == target.resolve():
        asset["copied"] = "true"
        asset["exists"] = True
        return {
            "status": "skipped",
            "source": str(source),
            "path": str(target),
            "reason": "source_is_target",
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    asset["copied"] = "true"
    asset["exists"] = True
    return {"status": "copied", "source": str(source), "path": str(target)}


def _sync_manifest_assets(manifest: dict[str, Any], timeline: list[Any]) -> None:
    assets_by_source: dict[str, dict[str, Any]] = {}
    for item in timeline:
        if not isinstance(item, dict):
            continue
        for asset in item.get("assets") or []:
            if isinstance(asset, dict):
                source = str(asset.get("source") or asset.get("path") or "")
                if source:
                    assets_by_source[source] = dict(asset)
    manifest["assets"] = list(assets_by_source.values())


def _sync_source_package_assets(
    root: Path, manifest: dict[str, Any], timeline: list[Any]
) -> dict[str, Any]:
    package_path = _source_package_path(root, manifest)
    if not package_path or not package_path.exists():
        return {
            "updated": 0,
            "package_path": str(package_path) if package_path else "",
            "reason": "source_package_missing",
        }
    package = read_json(package_path)
    if not isinstance(package, dict):
        return {
            "updated": 0,
            "package_path": str(package_path),
            "reason": "source_package_not_object",
        }
    package_timeline = (
        package.get("timeline") if isinstance(package.get("timeline"), list) else []
    )
    updated = 0
    for bundle_item in timeline:
        if not isinstance(bundle_item, dict):
            continue
        replacement_paths = _bundle_asset_paths(root, bundle_item)
        if not replacement_paths:
            continue
        package_item = _matching_package_item(package_timeline, bundle_item)
        if not package_item:
            continue
        current = [str(path) for path in package_item.get("frame_paths") or []]
        if current != replacement_paths:
            package_item.setdefault("original_frame_paths", current)
            package_item["frame_paths"] = replacement_paths
            updated += 1
    if updated:
        package["asset_repair_synced_at"] = now_iso()
        write_json(package_path, package)
    return {"updated": updated, "package_path": str(package_path)}


def _source_package_path(root: Path, manifest: dict[str, Any]) -> Path | None:
    text = str(manifest.get("source_package") or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else root / path


def _bundle_asset_paths(root: Path, item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for asset in item.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        if str(asset.get("copied") or "").lower() != "true":
            continue
        asset_path = str(asset.get("path") or "").strip()
        if not asset_path:
            continue
        path = Path(asset_path).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists() and path.is_file():
            paths.append(str(path))
    return paths


def _matching_package_item(
    package_timeline: list[Any], bundle_item: dict[str, Any]
) -> dict[str, Any] | None:
    wanted_segments = {
        str(value) for value in bundle_item.get("source_segment_ids") or []
    }
    if wanted_segments:
        for item in package_timeline:
            if not isinstance(item, dict):
                continue
            item_segments = {
                str(value) for value in item.get("source_segment_ids") or []
            }
            if wanted_segments and wanted_segments == item_segments:
                return item
    wanted_index = _int_or_none(bundle_item.get("index"))
    if wanted_index is not None and 0 < wanted_index <= len(package_timeline):
        item = package_timeline[wanted_index - 1]
        return item if isinstance(item, dict) else None
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _asset_manifest(assets: list[Any]) -> dict[str, Any]:
    rows = [asset for asset in assets if isinstance(asset, dict)]
    copied = [
        asset for asset in rows if str(asset.get("copied") or "").lower() == "true"
    ]
    missing = [
        asset for asset in rows if str(asset.get("copied") or "").lower() != "true"
    ]
    return {
        "schema": "lecture_webui_asset_manifest.v1",
        "updated_at": now_iso(),
        "copied_count": len(copied),
        "missing_count": len(missing),
        "assets": rows,
    }
