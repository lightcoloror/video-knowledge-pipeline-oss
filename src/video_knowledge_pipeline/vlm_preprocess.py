from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any


VLM_PREPROCESS_SCHEMA = "video_knowledge_pipeline.vlm_preprocess.v1"


def prepare_vlm_image_inputs(
    source_paths: list[str],
    *,
    output_dir: str | Path,
    max_edge: int = 0,
    jpeg_quality: int = 70,
    max_images: int = 0,
    role: str = "vision_input",
) -> dict[str, Any]:
    """Prepare local image inputs for cloud or local VLM providers.

    This is a Qwen/InternVL-style preprocessing boundary: image selection,
    optional resizing/compression, ordering, and audit metadata. It never calls a
    model and never embeds secrets or base64 payloads in reports.
    """
    selected = _select_source_paths(source_paths, max_images=max_images)
    source_total = _total_bytes(selected)
    max_edge = int(max_edge or 0)
    jpeg_quality = max(1, min(int(jpeg_quality or 70), 95))
    if max_edge <= 0:
        return _report(
            status="disabled",
            selected=selected,
            prepared=selected,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
            total_source_bytes=source_total,
            total_prepared_bytes=source_total,
            items=[
                _item_row(index=index, source_path=Path(path), prepared_path=Path(path), role=role, ok=Path(path).expanduser().exists())
                for index, path in enumerate(selected, start=1)
            ],
        )
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on optional local Pillow install.
        return _report(
            status="unavailable",
            selected=selected,
            prepared=selected,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
            total_source_bytes=source_total,
            total_prepared_bytes=source_total,
            items=[],
            error=f"Pillow unavailable: {exc}",
        )

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prepared: list[str] = []
    items: list[dict[str, Any]] = []
    for index, source in enumerate(selected, start=1):
        source_path = Path(source).expanduser().resolve()
        target = output_root / f"{index:02d}-{source_path.stem}-vlm.jpg"
        row = _item_row(index=index, source_path=source_path, prepared_path=target, role=role, ok=False)
        row["source_bytes"] = _path_bytes(source_path)
        try:
            with Image.open(source_path) as image:
                converted = image.convert("RGB")
                original_width, original_height = converted.size
                converted.thumbnail((max_edge, max_edge))
                prepared_width, prepared_height = converted.size
                converted.save(target, format="JPEG", quality=jpeg_quality, optimize=True)
            row.update(
                {
                    "ok": target.exists(),
                    "prepared_bytes": _path_bytes(target),
                    "original_width": original_width,
                    "original_height": original_height,
                    "prepared_width": prepared_width,
                    "prepared_height": prepared_height,
                    "error": "",
                }
            )
            if target.exists():
                prepared.append(str(target))
        except Exception as exc:
            row["error"] = str(exc)
        items.append(row)
    if len(prepared) == len(selected):
        status = "ok"
        image_paths = prepared
    elif prepared:
        status = "partial"
        image_paths = prepared
    else:
        status = "unavailable"
        image_paths = selected
    return _report(
        status=status,
        selected=selected,
        prepared=image_paths,
        max_edge=max_edge,
        jpeg_quality=jpeg_quality,
        total_source_bytes=source_total,
        total_prepared_bytes=_total_bytes(image_paths),
        items=items,
    )


def prepare_image_probe(
    source_paths: list[str],
    *,
    output_dir: str | Path,
    max_edge: int = 0,
    jpeg_quality: int = 70,
    max_images: int = 0,
    role: str = "image_probe",
) -> dict[str, Any]:
    """Backward-compatible image probe wrapper used by existing vision reports."""
    report = prepare_vlm_image_inputs(
        source_paths,
        output_dir=output_dir,
        max_edge=max_edge,
        jpeg_quality=jpeg_quality,
        max_images=max_images,
        role=role,
    )
    return {
        **report,
        "image_paths": report.get("prepared_image_paths", []),
        "source_image_paths": report.get("source_image_paths", []),
        "total_probe_bytes": report.get("total_prepared_bytes", 0),
    }


def _report(
    *,
    status: str,
    selected: list[str],
    prepared: list[str],
    max_edge: int,
    jpeg_quality: int,
    total_source_bytes: int,
    total_prepared_bytes: int,
    items: list[dict[str, Any]],
    error: str = "",
) -> dict[str, Any]:
    return {
        "schema": VLM_PREPROCESS_SCHEMA,
        "status": status,
        "error": error,
        "max_edge": int(max_edge or 0),
        "jpeg_quality": int(jpeg_quality or 70),
        "source_image_paths": selected,
        "prepared_image_paths": prepared,
        "total_source_bytes": total_source_bytes,
        "total_prepared_bytes": total_prepared_bytes,
        "items": items,
    }


def _select_source_paths(source_paths: list[str], *, max_images: int) -> list[str]:
    selected = [str(path) for path in source_paths if str(path)]
    if int(max_images or 0) > 0:
        selected = selected[: int(max_images)]
    return selected


def _item_row(*, index: int, source_path: Path, prepared_path: Path, role: str, ok: bool) -> dict[str, Any]:
    return {
        "index": index,
        "role": role,
        "source_path": str(source_path),
        "prepared_path": str(prepared_path),
        "source_bytes": _path_bytes(source_path),
        "prepared_bytes": _path_bytes(prepared_path),
        "mime_type": mimetypes.guess_type(str(prepared_path))[0] or "image/jpeg",
        "ok": ok,
        "error": "",
    }


def _total_bytes(paths: list[str]) -> int:
    return sum(_path_bytes(Path(path).expanduser()) for path in paths)


def _path_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
