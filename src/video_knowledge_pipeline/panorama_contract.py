from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PANORAMA_PROFILE_SCHEMA = "video_knowledge_pipeline.panorama_profile.v1"
SUPPORTED_PROJECTIONS = frozenset(
    {
        "equirectangular_mono",
        "equirectangular_stereo_sbs",
        "equirectangular_stereo_tb",
        "cubemap_3x2",
        "cubemap_6x1",
        "cubemap_1x6",
        "dual_fisheye_explicit",
    }
)
LENS_ORDERS = frozenset({"left_right", "right_left", "top_bottom", "bottom_top"})
PROPRIETARY_SUFFIXES = frozenset({".insv"})


@dataclass(frozen=True)
class PanoramaProfile:
    projection: str
    stereo_mode: str
    input_fov: float | None
    lens_order: str
    projection_source: str
    metadata_confidence: str
    metadata_evidence: tuple[tuple[str, str, str], ...]
    aspect_ratio_hint: str
    issues: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": PANORAMA_PROFILE_SCHEMA,
            "projection": self.projection,
            "stereo_mode": self.stereo_mode,
            "input_fov": self.input_fov,
            "lens_order": self.lens_order,
            "projection_source": self.projection_source,
            "metadata_confidence": self.metadata_confidence,
            "metadata_evidence": [
                {"source": source, "key": key, "value": value}
                for source, key, value in self.metadata_evidence
            ],
            "aspect_ratio_hint": self.aspect_ratio_hint,
            "issues": [{"key": key, "message": message} for key, message in self.issues],
        }
        payload["ready"] = self.projection in SUPPORTED_PROJECTIONS and not self.issues
        identity = {
            key: payload[key]
            for key in (
                "schema",
                "projection",
                "stereo_mode",
                "input_fov",
                "lens_order",
                "projection_source",
                "metadata_confidence",
                "metadata_evidence",
                "aspect_ratio_hint",
            )
        }
        payload["profile_sha256"] = _sha256_json(identity)
        return payload


def build_panorama_profile(
    *,
    projection: str | None = None,
    source_path: str | Path | None = None,
    ffprobe_payload: Mapping[str, Any] | None = None,
    input_fov: float | int | str | None = None,
    lens_order: str | None = None,
) -> dict[str, Any]:
    """Build a path-redacted, fail-closed spherical projection profile."""

    stream = _video_stream(ffprobe_payload)
    detected, stereo, evidence = _projection_from_metadata(stream)
    explicit = str(projection or "").strip().lower()
    issues: list[tuple[str, str]] = []

    if explicit and explicit not in SUPPORTED_PROJECTIONS:
        issues.append(("projection_unsupported", f"unsupported panorama projection: {explicit}"))
        selected = "unknown"
        projection_source = "explicit_override"
    elif explicit:
        selected = explicit
        projection_source = "explicit_override"
    elif detected:
        selected = detected
        projection_source = "ffprobe_metadata"
    else:
        selected = "unknown"
        projection_source = "unresolved"

    if explicit and explicit in SUPPORTED_PROJECTIONS and detected and _conflicts(explicit, detected):
        issues.append(
            (
                "projection_metadata_conflict",
                f"explicit projection {explicit} conflicts with metadata projection {detected}",
            )
        )

    suffix = Path(str(source_path or "")).suffix.lower()
    if suffix in PROPRIETARY_SUFFIXES and not (explicit in SUPPORTED_PROJECTIONS and stream):
        issues.append(
            (
                "proprietary_source_requires_vendor_export",
                "proprietary camera media requires a decodable video stream and an explicit projection, or a vendor export",
            )
        )

    aspect_hint = _aspect_ratio_hint(stream)
    if selected == "unknown" and not any(key == "projection_unsupported" for key, _ in issues):
        issues.append(
            (
                "projection_required",
                "spherical projection is not confirmed; supply --projection instead of relying on aspect ratio",
            )
        )

    normalized_fov = _optional_float(input_fov)
    normalized_lens_order = str(lens_order or "").strip().lower()
    if selected == "dual_fisheye_explicit":
        if input_fov is None:
            issues.append(("input_fov_required", "dual fisheye requires an explicit input FOV"))
        elif normalized_fov is None or not 0.0 < normalized_fov <= 360.0:
            issues.append(("input_fov_invalid", "input FOV must be greater than 0 and at most 360 degrees"))
        if not normalized_lens_order:
            issues.append(("lens_order_required", "dual fisheye requires an explicit lens order"))
        elif normalized_lens_order not in LENS_ORDERS:
            issues.append(("lens_order_invalid", f"unsupported dual-fisheye lens order: {normalized_lens_order}"))
    else:
        normalized_fov = None
        if normalized_lens_order:
            issues.append(("lens_order_not_applicable", "lens order is only valid for dual_fisheye_explicit"))
        normalized_lens_order = ""

    stereo_mode = _stereo_mode(selected, stereo)
    profile = PanoramaProfile(
        projection=selected,
        stereo_mode=stereo_mode,
        input_fov=normalized_fov,
        lens_order=normalized_lens_order,
        projection_source=projection_source,
        metadata_confidence="confirmed" if detected else "unavailable",
        metadata_evidence=tuple(evidence),
        aspect_ratio_hint=aspect_hint,
        issues=tuple(_dedupe_issues(issues)),
    )
    return profile.to_dict()


def _video_stream(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return {}
    return next((dict(row) for row in streams if isinstance(row, Mapping) and row.get("codec_type") == "video"), {})


def _projection_from_metadata(stream: Mapping[str, Any]) -> tuple[str, str, list[tuple[str, str, str]]]:
    if not stream:
        return "", "", []
    evidence: list[tuple[str, str, str]] = []
    candidates: list[str] = []
    stereo_candidates: list[str] = []
    tags = stream.get("tags")
    if isinstance(tags, Mapping):
        for raw_key, raw_value in tags.items():
            key = str(raw_key).strip().lower()
            value = str(raw_value).strip().lower()
            if not value:
                continue
            if any(token in key for token in ("projection", "spherical", "stereo")):
                evidence.append(("stream_tag", key, value[:200]))
            projection = _projection_token(value)
            if projection and ("projection" in key or "spherical" in key):
                candidates.append(projection)
            stereo = _stereo_token(value) if "stereo" in key else ""
            if stereo:
                stereo_candidates.append(stereo)

    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for row in side_data:
            if not isinstance(row, Mapping):
                continue
            side_type = str(row.get("side_data_type") or "").strip().lower()
            projection_value = str(row.get("projection") or "").strip().lower()
            if "spherical" in side_type:
                evidence.append(("side_data", "side_data_type", side_type[:200]))
                projection = _projection_token(projection_value) or "equirectangular_mono"
                candidates.append(projection)
                if projection_value:
                    evidence.append(("side_data", "projection", projection_value[:200]))

    projection = candidates[0] if candidates else ""
    stereo = stereo_candidates[0] if stereo_candidates else ""
    if projection == "equirectangular_mono" and stereo == "sbs":
        projection = "equirectangular_stereo_sbs"
    elif projection == "equirectangular_mono" and stereo == "tb":
        projection = "equirectangular_stereo_tb"
    return projection, stereo, _dedupe_evidence(evidence)


def _projection_token(value: str) -> str:
    normalized = value.replace("-", "_").replace(" ", "_")
    if "equirect" in normalized or normalized in {"erp", "spherical", "360"}:
        return "equirectangular_mono"
    if normalized in {"cubemap_3x2", "c3x2", "3x2"}:
        return "cubemap_3x2"
    if normalized in {"cubemap_6x1", "c6x1", "6x1"}:
        return "cubemap_6x1"
    if normalized in {"cubemap_1x6", "c1x6", "1x6"}:
        return "cubemap_1x6"
    if normalized in {"dual_fisheye", "dfisheye"}:
        return "dual_fisheye_explicit"
    return ""


def _stereo_token(value: str) -> str:
    normalized = value.replace("-", "_").replace(" ", "_")
    if normalized in {"left_right", "right_left", "side_by_side", "sbs"}:
        return "sbs"
    if normalized in {"top_bottom", "bottom_top", "over_under", "tb"}:
        return "tb"
    return ""


def _stereo_mode(projection: str, metadata_stereo: str) -> str:
    if projection == "equirectangular_stereo_sbs":
        return "sbs"
    if projection == "equirectangular_stereo_tb":
        return "tb"
    return metadata_stereo if metadata_stereo in {"sbs", "tb"} and projection == "unknown" else "2d"


def _aspect_ratio_hint(stream: Mapping[str, Any]) -> str:
    try:
        width = float(stream.get("width") or 0)
        height = float(stream.get("height") or 0)
    except (TypeError, ValueError):
        return ""
    if height > 0 and abs((width / height) - 2.0) <= 0.02:
        return "possible_equirectangular"
    return ""


def _conflicts(explicit: str, detected: str) -> bool:
    explicit_family = "equirectangular" if explicit.startswith("equirectangular_") else explicit.split("_", 1)[0]
    detected_family = "equirectangular" if detected.startswith("equirectangular_") else detected.split("_", 1)[0]
    if explicit_family != detected_family:
        return True
    if detected in {"equirectangular_stereo_sbs", "equirectangular_stereo_tb"}:
        return explicit != detected
    return False


def _optional_float(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_issues(values: list[tuple[str, str]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, message in values:
        if key not in seen:
            result.append((key, message))
            seen.add(key)
    return result


def _dedupe_evidence(values: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
