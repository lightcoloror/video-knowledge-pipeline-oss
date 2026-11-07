from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .vlm_preprocess import prepare_vlm_image_inputs


TEMPORAL_VLM_PREPROCESS_SCHEMA = "video_knowledge_pipeline.temporal_vlm_preprocess.v1"
_FRAME_TIMESTAMP_RE = re.compile(r"_(\d+)ms$")


def build_temporal_frame_manifest(source_paths: list[str]) -> list[dict[str, Any]]:
    """Build system-owned frame IDs and timestamps without asking a model to count."""
    rows: list[dict[str, Any]] = []
    for index, value in enumerate((str(path) for path in source_paths if str(path)), start=1):
        path = Path(value).expanduser()
        timestamp_ms = _timestamp_ms_from_path(path)
        rows.append(
            {
                "frame_id": f"F{index:02d}",
                "source_index": index,
                "filename": path.name,
                "source_path": str(path),
                "timestamp_ms": timestamp_ms,
                "timestamp": _timestamp_label(timestamp_ms),
                "bytes": _path_bytes(path),
            }
        )
    return rows


def prepare_temporal_image_probe(
    source_paths: list[str],
    *,
    output_dir: str | Path,
    max_edge: int = 0,
    jpeg_quality: int = 70,
    max_images: int = 0,
    role: str = "temporal_image_probe",
    use_contact_sheet: bool = False,
    representative_limit: int = 2,
    near_duplicate_rms_threshold: float = 0.02,
    near_duplicate_changed_ratio_threshold: float = 0.008,
    write: bool = True,
) -> dict[str, Any]:
    """Prepare ordered temporal evidence using mature Pillow primitives.

    VKP supplies only the evidence contract, conservative clustering,
    representative mapping, and audit metadata. Pillow owns image decoding,
    differences, statistics, resizing, drawing, and encoding.
    """
    selected = [str(path) for path in source_paths if str(path)]
    if int(max_images or 0) > 0:
        selected = selected[: int(max_images)]
    manifest = build_temporal_frame_manifest(selected)
    metrics, pillow_info = _temporal_image_metrics(selected)
    for row, metric in zip(manifest, metrics):
        row.update(metric)
    clusters = _temporal_clusters(
        manifest,
        rms_threshold=float(near_duplicate_rms_threshold),
        changed_ratio_threshold=float(near_duplicate_changed_ratio_threshold),
    )
    representatives = _temporal_representatives(manifest, clusters, max(1, int(representative_limit or 1)))
    representative_paths = [str(row["source_path"]) for row in representatives]
    frame_mapping = _temporal_frame_mapping(manifest, representatives)
    output_root = Path(output_dir).expanduser().resolve()

    if not write:
        image_paths = representative_paths
        status = "planned_no_write" if selected else "unavailable"
        sent_strategy = "representatives_no_write"
        contact_sheet_path = None
        contact_sheet_error = ""
        prepared_items = []
    elif use_contact_sheet and selected and pillow_info.get("available"):
        output_root.mkdir(parents=True, exist_ok=True)
        contact_sheet_path = output_root / "temporal-contact-sheet.jpg"
        contact_sheet_error = _write_temporal_contact_sheet(
            manifest,
            contact_sheet_path,
            jpeg_quality=max(1, min(int(jpeg_quality or 70), 95)),
        )
        representative_report = prepare_vlm_image_inputs(
            representative_paths,
            output_dir=output_root / "representatives",
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
            role=f"{role}_representative",
        )
        prepared_representatives = list(representative_report.get("prepared_image_paths") or representative_paths)
        contact_sheet_ready = contact_sheet_path.is_file() and not contact_sheet_error
        image_paths = ([str(contact_sheet_path)] if contact_sheet_ready else []) + prepared_representatives
        status = "ok" if contact_sheet_ready and prepared_representatives else ("partial" if image_paths else "unavailable")
        sent_strategy = "contact_sheet_plus_representatives" if contact_sheet_ready else "representatives_only"
        prepared_items = list(representative_report.get("items") or [])
    else:
        prepared_report = prepare_vlm_image_inputs(
            selected,
            output_dir=output_root,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
            role=role,
        )
        image_paths = list(prepared_report.get("prepared_image_paths") or selected)
        status = str(prepared_report.get("status") or "unavailable")
        sent_strategy = "ordered_frames"
        contact_sheet_path = None
        contact_sheet_error = "Pillow unavailable" if use_contact_sheet and not pillow_info.get("available") else ""
        prepared_items = list(prepared_report.get("items") or [])

    source_bytes = _total_bytes(selected)
    prepared_bytes = _total_bytes(image_paths)
    return {
        "schema": TEMPORAL_VLM_PREPROCESS_SCHEMA,
        "status": status,
        "error": contact_sheet_error,
        "role": role,
        "source_image_paths": selected,
        "prepared_image_paths": image_paths,
        "image_paths": image_paths,
        "original_frame_count": len(selected),
        "representative_frame_count": len(representatives),
        "sent_image_count": len(image_paths),
        "sent_strategy": sent_strategy,
        "contact_sheet_path": str(contact_sheet_path) if contact_sheet_path else "",
        "frame_manifest": manifest,
        "frame_mapping": frame_mapping,
        "near_duplicate_cluster_count": len(clusters),
        "near_duplicate_rms_threshold": float(near_duplicate_rms_threshold),
        "near_duplicate_changed_ratio_threshold": float(near_duplicate_changed_ratio_threshold),
        "total_source_bytes": source_bytes,
        "total_prepared_bytes": prepared_bytes,
        "total_probe_bytes": prepared_bytes,
        "byte_reduction_ratio": round(1.0 - prepared_bytes / source_bytes, 6) if source_bytes else 0.0,
        "items": prepared_items,
        "implementation": {
            "library": "Pillow",
            "version": str(pillow_info.get("version") or ""),
            "source_path": str(pillow_info.get("source_path") or ""),
            "primitives": [
                "PIL.ImageChops.difference",
                "PIL.ImageStat.Stat",
                "PIL.ImageFilter.FIND_EDGES",
                "PIL.ImageOps.contain",
                "PIL.ImageDraw.text",
            ],
        },
    }


def _temporal_image_metrics(source_paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import PIL
        from PIL import Image, ImageChops, ImageFilter, ImageStat
    except Exception as exc:  # pragma: no cover - optional local Pillow dependency.
        return ([{} for _ in source_paths], {"available": False, "error": str(exc)})
    metrics: list[dict[str, Any]] = []
    previous = None
    for value in source_paths:
        try:
            with Image.open(Path(value).expanduser()) as image:
                gray = image.convert("L").resize((256, 144))
            edges = gray.filter(ImageFilter.FIND_EDGES).crop((1, 1, 255, 143))
            row: dict[str, Any] = {
                "sharpness_score": round(float(ImageStat.Stat(edges).var[0]), 6),
                "difference_rms_to_previous": None,
                "changed_ratio_to_previous": None,
            }
            if previous is not None:
                difference = ImageChops.difference(previous, gray)
                row["difference_rms_to_previous"] = round(float(ImageStat.Stat(difference).rms[0]) / 255.0, 6)
                row["changed_ratio_to_previous"] = round(sum(difference.histogram()[16:]) / float(256 * 144), 6)
            previous = gray.copy()
            metrics.append(row)
        except Exception as exc:
            metrics.append({"metric_error": str(exc)})
            previous = None
    return metrics, {"available": True, "version": str(PIL.__version__), "source_path": str(Path(PIL.__file__).parent)}


def _temporal_clusters(
    manifest: list[dict[str, Any]],
    *,
    rms_threshold: float,
    changed_ratio_threshold: float,
) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for row in manifest:
        rms = row.get("difference_rms_to_previous")
        changed_ratio = row.get("changed_ratio_to_previous")
        near_duplicate = (
            bool(clusters)
            and isinstance(rms, (int, float))
            and isinstance(changed_ratio, (int, float))
            and float(rms) <= rms_threshold
            and float(changed_ratio) <= changed_ratio_threshold
        )
        if near_duplicate:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    return clusters


def _temporal_representatives(
    manifest: list[dict[str, Any]],
    clusters: list[list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    representatives = [max(cluster, key=lambda row: float(row.get("sharpness_score") or 0.0)) for cluster in clusters if cluster]
    if len(representatives) <= limit:
        return representatives
    indexes = [round(i * (len(representatives) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [0]
    selected: list[dict[str, Any]] = []
    for index in indexes:
        row = representatives[int(index)]
        if row not in selected:
            selected.append(row)
    return selected or manifest[:1]


def _temporal_frame_mapping(
    manifest: list[dict[str, Any]],
    representatives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not representatives:
        return []
    return [
        {
            "frame_id": str(row["frame_id"]),
            "source_index": int(row["source_index"]),
            "representative_frame_id": str(
                min(
                    representatives,
                    key=lambda candidate: abs(int(candidate["source_index"]) - int(row["source_index"])),
                )["frame_id"]
            ),
        }
        for row in manifest
    ]


def _write_temporal_contact_sheet(
    manifest: list[dict[str, Any]],
    target: Path,
    *,
    jpeg_quality: int,
) -> str:
    try:
        from PIL import Image, ImageDraw, ImageOps

        columns = min(4, max(1, len(manifest)))
        rows = max(1, math.ceil(len(manifest) / columns))
        tile_width, tile_height, label_height = 480, 270, 30
        sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "#111111")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(manifest):
            column = index % columns
            row_index = index // columns
            x = column * tile_width
            y = row_index * (tile_height + label_height)
            with Image.open(Path(str(row["source_path"])).expanduser()) as image:
                fitted = ImageOps.contain(image.convert("RGB"), (tile_width, tile_height))
            sheet.paste(fitted, (x + (tile_width - fitted.width) // 2, y + (tile_height - fitted.height) // 2))
            label = f'{row["frame_id"]} {row.get("timestamp") or ""}'.strip()
            draw.rectangle((x, y + tile_height, x + tile_width, y + tile_height + label_height), fill="#111111")
            draw.text((x + 8, y + tile_height + 7), label, fill="white")
        sheet.save(target, format="JPEG", quality=jpeg_quality, optimize=True)
        return ""
    except Exception as exc:
        return str(exc)


def _timestamp_ms_from_path(path: Path) -> int | None:
    match = _FRAME_TIMESTAMP_RE.search(path.stem)
    return int(match.group(1)) if match else None


def _timestamp_label(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return ""
    total_seconds, milliseconds = divmod(int(timestamp_ms), 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _total_bytes(paths: list[str]) -> int:
    return sum(_path_bytes(Path(path).expanduser()) for path in paths)


def _path_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
