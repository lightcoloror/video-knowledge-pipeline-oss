from __future__ import annotations

from typing import Any

SUPPORTED_SOURCE_FORMATS = {
    "transnetv2_scenes": {
        "project": "soCzech/TransNetV2",
        "commit": "85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed",
        "reason": "transnetv2_saved_scene_boundary",
    },
    "autoshot_scenes": {
        "project": "wentaozhu/AutoShot",
        "commit": "77c82ff826a9301bb173d9be786297a49d73d081",
        "reason": "autoshot_saved_scene_boundary",
    },
    "omnishotcut_scenes": {
        "project": "UVA-Computer-Vision-Lab/OmniShotCut",
        "commit": "23ad6fb41b296fb9258b0e7825125a914573b906",
        "reason": "omnishotcut_saved_scene_boundary",
    },
}


def normalize_saved_shot_boundaries(
    payload: Any,
    *,
    source_format: str,
    frame_rate: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize saved predictions_to_scenes output without running a model."""

    if not isinstance(payload, dict):
        raise ValueError("saved shot-boundary payload must be a JSON object")
    format_key = str(source_format or "").strip().lower()
    definition = SUPPORTED_SOURCE_FORMATS.get(format_key)
    if definition is None:
        raise ValueError(
            "source_format must be transnetv2_scenes, autoshot_scenes, or omnishotcut_scenes"
        )
    fps = _positive_float(
        frame_rate
        if frame_rate is not None
        else payload.get("frame_rate", payload.get("fps")),
        "frame_rate",
    )
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("saved shot-boundary payload contains no scenes")

    scenes: list[dict[str, int]] = []
    previous_end = -1
    for position, raw in enumerate(raw_scenes, start=1):
        start_frame, end_frame = _scene_frames(raw, position=position)
        if start_frame <= previous_end:
            raise ValueError(
                f"saved scenes must be ordered and non-overlapping: row {position}"
            )
        scenes.append(
            {
                "source_scene_index": position,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
        )
        previous_end = end_frame

    rows = [
        {
            **scene,
            "seconds": round(scene["start_frame"] / fps, 6),
            "reason": definition["reason"],
        }
        for scene in scenes[1:]
    ]
    metadata = {
        "source_format": format_key,
        "boundary_kind": "shot",
        "frame_rate": fps,
        "prediction_threshold": _optional_float(payload.get("threshold")),
        "upstream_project": definition["project"],
        "upstream_commit": definition["commit"],
        "upstream_api": "predictions_to_scenes",
        "source_scene_count": len(scenes),
        "boundary_count": len(rows),
        "model_execution_performed": False,
    }
    return rows, metadata


def _scene_frames(value: Any, *, position: int) -> tuple[int, int]:
    if isinstance(value, dict):
        start = value.get("start_frame", value.get("start"))
        end = value.get("end_frame", value.get("end"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
    else:
        raise ValueError(f"saved scene row {position} must be a frame pair")
    start_frame = _nonnegative_integer(start, f"scene {position} start_frame")
    end_frame = _nonnegative_integer(end, f"scene {position} end_frame")
    if end_frame < start_frame:
        raise ValueError(f"saved scene row {position} ends before it starts")
    return start_frame, end_frame


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if number < 0 or not number.is_integer():
        raise ValueError(f"{label} must be a non-negative integer")
    return int(number)


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive") from exc
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _optional_float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None