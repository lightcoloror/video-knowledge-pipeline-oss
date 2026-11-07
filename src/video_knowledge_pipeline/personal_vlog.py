from __future__ import annotations

import hashlib
import json
import math
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .file_hash import sha256_file
from .storage import bundle_write_lock, read_json, write_json


PROJECT_SCHEMA = "video_knowledge_pipeline.personal_vlog_project.v1"
EVIDENCE_SCHEMA = "video_knowledge_pipeline.personal_vlog_asset_evidence.v1"
EVENT_PLAN_SCHEMA = "video_knowledge_pipeline.personal_vlog_event_plan.v1"
RECIPE_SCHEMA = "video_knowledge_pipeline.personal_vlog_edit_recipe.v1"
RUN_SCHEMA = "video_knowledge_pipeline.personal_vlog_run.v1"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".m4v", ".avi"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"})
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

STORY_ROLES = (
    "cold_open",
    "setup",
    "activity",
    "reaction",
    "detail",
    "payoff",
    "closing",
)

SCORE_WEIGHTS: dict[str, float] = {
    "activity_relevance": 0.25,
    "technical_quality": 0.20,
    "appearance_or_interaction_continuity": 0.15,
    "audio_value": 0.15,
    "novelty_and_diversity": 0.15,
    "chronology_and_story_role": 0.10,
}

PRESETS: dict[str, dict[str, Any]] = {
    "highlight": {
        "target_seconds": 75.0,
        "minimum_seconds": 60.0,
        "maximum_seconds": 90.0,
        "aspect_ratio": "9:16",
    },
    "standard": {
        "target_seconds": 240.0,
        "minimum_seconds": 180.0,
        "maximum_seconds": 300.0,
        "aspect_ratio": "16:9",
    },
    "long": {
        "target_seconds": 600.0,
        "minimum_seconds": 480.0,
        "maximum_seconds": 720.0,
        "aspect_ratio": "16:9",
    },
}


class PersonalVlogError(RuntimeError):
    """Base error for the fail-closed personal Vlog adapter."""


class PersonalVlogBoundaryError(PersonalVlogError):
    """Raised when the bounded source traversal cannot be proven safe."""


class PersonalVlogDriftError(PersonalVlogError):
    """Raised when source identity changed after the project manifest was made."""


@dataclass(frozen=True)
class _ObservedFile:
    path: Path
    relative_path: str
    media_type: str | None
    size_bytes: int
    mtime_ns: int
    sha256: str


def prepare_personal_vlog_project(
    event_root: str | Path,
    project_dir: str | Path,
    *,
    max_depth: int = 2,
    max_items: int = 2_000,
    appearance_clustering: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Create a bounded, content-addressed manifest without copying source media."""

    source_root = Path(event_root).expanduser().resolve(strict=True)
    target_root = Path(project_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise PersonalVlogBoundaryError(f"event_root_not_directory:{source_root}")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if max_items < 1:
        raise ValueError("max_items must be positive")
    _reject_linklike(source_root, label="event_root")
    _reject_overlapping_roots(source_root, target_root)

    observed = _bounded_inventory(
        source_root,
        max_depth=max_depth,
        max_items=max_items,
    )
    assets: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    digest_groups: dict[str, list[str]] = {}
    for item in observed:
        digest_groups.setdefault(item.sha256, []).append(item.relative_path)
        if item.media_type is None:
            unsupported.append(
                {
                    "relative_path": item.relative_path,
                    "extension": item.path.suffix.lower(),
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                    "status": "unsupported_format",
                }
            )
            continue
        asset_id = _stable_id("asset", item.sha256, item.relative_path)
        assets.append(
            {
                "asset_id": asset_id,
                "media_type": item.media_type,
                "source_path": str(item.path),
                "source_relative_path": item.relative_path,
                "source_sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "capture_time": _mtime_iso(item.mtime_ns),
                "capture_time_source": "filesystem_mtime_fallback",
                "orientation": "unknown",
                "duration_seconds": None,
                "duplicate_group": (
                    f"digest-{item.sha256[:16]}"
                    if len(digest_groups.get(item.sha256, [])) > 1
                    else None
                ),
                "child_bundle": (
                    f"bundles/{asset_id}" if item.media_type == "video" else None
                ),
                "source_read_only": True,
            }
        )

    # The first pass above sees digest groups incrementally; normalize duplicate
    # markers after all files are known so identical inputs produce identical output.
    for asset in assets:
        paths = digest_groups.get(str(asset["source_sha256"]), [])
        asset["duplicate_group"] = (
            f"digest-{str(asset['source_sha256'])[:16]}" if len(paths) > 1 else None
        )

    assets.sort(key=lambda row: (str(row["capture_time"]), str(row["source_relative_path"])))
    unsupported.sort(key=lambda row: str(row["relative_path"]))
    identity_payload = {
        "schema": PROJECT_SCHEMA,
        "policy": {
            "profile": "personal-vlog-v1",
            "max_depth": int(max_depth),
            "max_items": int(max_items),
            "appearance_clustering": bool(appearance_clustering),
            "remote_fallback": False,
            "source_mutation": False,
            "output_mode": "candidate_only",
        },
        "assets": [
            {
                key: asset[key]
                for key in (
                    "asset_id",
                    "media_type",
                    "source_relative_path",
                    "source_sha256",
                    "size_bytes",
                    "capture_time",
                )
            }
            for asset in assets
        ],
        "unsupported": unsupported,
    }
    project_identity = _sha256_json(identity_payload)
    manifest = {
        "schema": PROJECT_SCHEMA,
        "profile": "personal-vlog-v1",
        "project_id": f"vlog-{project_identity[:16]}",
        "project_identity_sha256": project_identity,
        "contract_sha256": _contract_hash(PROJECT_SCHEMA),
        "event_root": str(source_root),
        "project_dir": str(target_root),
        "policy": identity_payload["policy"],
        "assets": assets,
        "unsupported_files": unsupported,
        "counts": {
            "enumerated": len(observed),
            "supported": len(assets),
            "unsupported": len(unsupported),
            "images": sum(row["media_type"] == "image" for row in assets),
            "videos": sum(row["media_type"] == "video" for row in assets),
            "audio": sum(row["media_type"] == "audio" for row in assets),
        },
        "source_policy": {
            "read_only": True,
            "copied_into_project": False,
            "hash_revalidation_required": True,
        },
        "status": "prepared",
        "review_required": True,
    }
    if write:
        target_root.mkdir(parents=True, exist_ok=True)
        with bundle_write_lock(target_root, operation="personal_vlog_prepare"):
            write_json(target_root / "personal-vlog-project.json", manifest)
    return manifest


def validate_project_sources(project: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate every recorded source before any downstream stage is written."""

    root = Path(str(project.get("event_root") or "")).resolve(strict=True)
    _reject_linklike(root, label="event_root")
    validated = 0
    for asset in _mapping_rows(project.get("assets")):
        relative = str(asset.get("source_relative_path") or "")
        if not relative:
            raise PersonalVlogDriftError("asset_missing_relative_path")
        path = (root / Path(relative)).resolve(strict=True)
        _require_within(root, path, label="source_asset")
        _reject_link_components(root, path)
        if not path.is_file():
            raise PersonalVlogDriftError(f"source_not_file:{relative}")
        actual = sha256_file(path)
        expected = str(asset.get("source_sha256") or "")
        if not expected or actual != expected:
            raise PersonalVlogDriftError(
                f"source_hash_drift:{relative}:expected={expected}:actual={actual}"
            )
        validated += 1
    return {
        "status": "valid",
        "validated_assets": validated,
        "project_identity_sha256": project.get("project_identity_sha256"),
    }


def build_asset_evidence(
    project: Mapping[str, Any],
    *,
    imported: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build preset-independent source evidence plus explicit backend gaps."""

    validate_project_sources(project)
    imported = imported or {}
    rows: list[dict[str, Any]] = []
    review_gaps: list[dict[str, Any]] = []
    for asset in _mapping_rows(project.get("assets")):
        asset_id = str(asset["asset_id"])
        kind = str(asset["media_type"])
        extra = dict(imported.get(asset_id) or {})
        backend_status = {
            "deterministic_probe": "available",
            "openclip": str(extra.get("openclip_status") or "unavailable"),
            "clap": str(extra.get("clap_status") or "unavailable"),
            "appearance_clustering": str(
                extra.get("appearance_clustering_status") or "unavailable"
            ),
            "local_vlm": str(extra.get("local_vlm_status") or "not_requested"),
            "remote_fallback": "disabled",
        }
        duration = _optional_nonnegative_float(extra.get("duration_seconds"))
        width = _optional_positive_int(extra.get("width"))
        height = _optional_positive_int(extra.get("height"))
        technical_quality = _clamp01(extra.get("technical_quality", 0.5))
        relevance = _clamp01(extra.get("activity_relevance", 0.5))
        audio_value = _clamp01(extra.get("audio_value", 0.5 if kind != "image" else 0.0))
        observation = {
            "asset_id": asset_id,
            "source_sha256": str(asset["source_sha256"]),
            "media_type": kind,
            "capture_time": str(asset["capture_time"]),
            "technical": {
                "size_bytes": int(asset["size_bytes"]),
                "duration_seconds": duration,
                "width": width,
                "height": height,
                "orientation": _orientation(width, height),
                "quality_score": technical_quality,
            },
            "semantic": {
                "tags": _clean_strings(extra.get("tags")),
                "asr_excerpt": str(extra.get("asr_excerpt") or "")[:2_000],
                "ocr_excerpt": str(extra.get("ocr_excerpt") or "")[:2_000],
                "activity_relevance": relevance,
                "evidence_level": str(extra.get("evidence_level") or "candidate_weak"),
            },
            "audio": {
                "value_score": audio_value,
                "events": _clean_strings(extra.get("audio_events")),
                "preserve_original_sound": bool(extra.get("preserve_original_sound", False)),
            },
            "appearance": {
                "cluster_ids": _anonymous_clusters(extra.get("appearance_cluster_ids")),
                "identity_claims_allowed": False,
                "cross_project_linking_allowed": False,
            },
            "location": {
                "cluster_id": _safe_cluster_id(extra.get("location_cluster_id"), "location"),
                "precise_coordinates_exported": False,
                "private_location_ref": (
                    f"private/locations/{asset_id}.json"
                    if extra.get("has_precise_location")
                    else None
                ),
            },
            "backend_status": backend_status,
            "limitations": [],
        }
        if kind == "video":
            observation["child_bundle"] = str(asset.get("child_bundle") or "")
            observation["child_bundle_profile"] = "filmed-v1"
            observation["child_bundle_status"] = str(
                extra.get("child_bundle_status") or "planned"
            )
        if str(extra.get("probe_status") or "ok") != "ok":
            observation["limitations"].append("media_probe_failed")
            observation["probe_receipt"] = {
                "status": str(extra.get("probe_status") or "failed"),
                "error_class": str(extra.get("probe_error_class") or "media_probe_failed"),
                "source_mutated": False,
            }
        if kind == "image" and backend_status["openclip"] == "unavailable":
            observation["limitations"].append("openclip_unavailable")
        if kind == "audio" and backend_status["clap"] == "unavailable":
            observation["limitations"].append("clap_unavailable")
        if kind == "video" and observation.get("child_bundle_status") not in {
            "completed",
            "imported_valid",
        }:
            observation["limitations"].append("filmed_v1_child_bundle_not_completed")
        if backend_status["appearance_clustering"] == "unavailable":
            observation["limitations"].append("appearance_clustering_unavailable")
        if observation["limitations"]:
            review_gaps.append(
                {
                    "asset_id": asset_id,
                    "reason": ",".join(observation["limitations"]),
                    "blocking": False,
                }
            )
        rows.append(observation)

    payload = {
        "schema": EVIDENCE_SCHEMA,
        "project_id": project.get("project_id"),
        "project_identity_sha256": project.get("project_identity_sha256"),
        "contract_sha256": _contract_hash(EVIDENCE_SCHEMA),
        "assets": rows,
        "review_gaps": review_gaps,
        "policy": {
            "local_first": True,
            "remote_fallback": False,
            "model_downloads": False,
            "source_mutation": False,
            "exact_location_export": False,
        },
    }
    payload["evidence_sha256"] = _sha256_json(payload)
    return payload


def build_event_plan(
    project: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    manual_boundaries: Sequence[int] | None = None,
    time_gap_seconds: float = 1_800.0,
) -> dict[str, Any]:
    """Create deterministic ordered event groups; manual boundaries win."""

    _require_project_match(project, evidence, label="asset_evidence")
    evidence_by_id = {
        str(row["asset_id"]): row for row in _mapping_rows(evidence.get("assets"))
    }
    assets = list(_mapping_rows(project.get("assets")))
    if not assets:
        boundaries: list[int] = []
        engine = "empty"
        engine_gap = None
    elif manual_boundaries is not None:
        boundaries = _validate_boundaries(manual_boundaries, len(assets))
        engine = "manual_boundaries"
        engine_gap = None
    else:
        boundaries, engine, engine_gap = _automatic_boundaries(
            assets,
            evidence_by_id,
            time_gap_seconds=float(time_gap_seconds),
        )

    events: list[dict[str, Any]] = []
    start = 0
    for number, stop in enumerate(boundaries, start=1):
        members = assets[start:stop]
        if not members:
            continue
        events.append(
            {
                "event_id": f"event-{number:03d}",
                "order": number,
                "member_asset_ids": [str(row["asset_id"]) for row in members],
                "start_capture_time": str(members[0]["capture_time"]),
                "end_capture_time": str(members[-1]["capture_time"]),
                "segmentation_reason": engine,
                "location_cluster_ids": sorted(
                    {
                        str(
                            (evidence_by_id.get(str(row["asset_id"])) or {})
                            .get("location", {})
                            .get("cluster_id")
                        )
                        for row in members
                        if (
                            isinstance(
                                (evidence_by_id.get(str(row["asset_id"])) or {}).get(
                                    "location"
                                ),
                                Mapping,
                            )
                            and (
                                (evidence_by_id.get(str(row["asset_id"])) or {})
                                .get("location", {})
                                .get("cluster_id")
                            )
                        )
                    }
                ),
                "review_required": True,
            }
        )
        start = stop

    payload = {
        "schema": EVENT_PLAN_SCHEMA,
        "project_id": project.get("project_id"),
        "project_identity_sha256": project.get("project_identity_sha256"),
        "evidence_sha256": evidence.get("evidence_sha256"),
        "contract_sha256": _contract_hash(EVENT_PLAN_SCHEMA),
        "segmentation": {
            "engine": engine,
            "manual_boundary_priority": True,
            "fallback_or_gap": engine_gap,
            "time_order_preserved": True,
        },
        "events": events,
        "review_gaps": ([{"reason": engine_gap, "blocking": False}] if engine_gap else []),
    }
    payload["event_plan_sha256"] = _sha256_json(payload)
    return payload


def build_edit_recipes(
    project: Mapping[str, Any],
    evidence: Mapping[str, Any],
    event_plan: Mapping[str, Any],
    *,
    presets: str | Iterable[str] = "all",
) -> dict[str, dict[str, Any]]:
    """Build deterministic candidate-only recipes for the requested presets."""

    _require_project_match(project, evidence, label="asset_evidence")
    _require_project_match(project, event_plan, label="event_plan")
    if str(event_plan.get("evidence_sha256") or "") != str(
        evidence.get("evidence_sha256") or ""
    ):
        raise PersonalVlogDriftError("event_plan_evidence_hash_mismatch")
    if presets == "all":
        names = list(PRESETS)
    elif isinstance(presets, str):
        names = [presets]
    else:
        names = [str(name) for name in presets]
    unknown = sorted(set(names) - set(PRESETS))
    if unknown:
        raise ValueError(f"unknown presets: {','.join(unknown)}")

    evidence_by_id = {
        str(row["asset_id"]): row for row in _mapping_rows(evidence.get("assets"))
    }
    event_by_asset: dict[str, str] = {}
    for event in _mapping_rows(event_plan.get("events")):
        for asset_id in _clean_strings(event.get("member_asset_ids")):
            event_by_asset[asset_id] = str(event.get("event_id") or "")

    candidates = []
    for index, asset in enumerate(_mapping_rows(project.get("assets"))):
        asset_id = str(asset["asset_id"])
        fact = evidence_by_id.get(asset_id) or {}
        score = _score_asset(asset, fact, index=index)
        candidates.append((score["total"], index, asset, fact, score))
    candidates.sort(key=lambda value: (-value[0], value[1], str(value[2]["asset_id"])))

    recipes: dict[str, dict[str, Any]] = {}
    for preset_name in names:
        preset = PRESETS[preset_name]
        clips: list[dict[str, Any]] = []
        elapsed = 0.0
        selected_ids: set[str] = set()
        for _total, original_index, asset, fact, score in candidates:
            asset_id = str(asset["asset_id"])
            if asset_id in selected_ids or elapsed >= float(preset["maximum_seconds"]):
                continue
            media_type = str(asset["media_type"])
            clip_duration = _candidate_duration(media_type, fact)
            remaining = float(preset["maximum_seconds"]) - elapsed
            clip_duration = min(clip_duration, remaining)
            if clip_duration <= 0:
                continue
            role = STORY_ROLES[min(original_index, len(STORY_ROLES) - 1)]
            clip: dict[str, Any] = {
                "clip_id": f"{preset_name}-clip-{len(clips) + 1:03d}",
                "source_asset_id": asset_id,
                "media_type": media_type,
                "event_id": event_by_asset.get(asset_id),
                "story_role": role,
                "timeline_start_seconds": round(elapsed, 3),
                "timeline_duration_seconds": round(clip_duration, 3),
                "score_breakdown": score,
                "evidence_refs": [
                    {
                        "asset_id": asset_id,
                        "source_sha256": str(asset["source_sha256"]),
                        "evidence_sha256": evidence.get("evidence_sha256"),
                    }
                ],
                "audio_policy": _audio_policy(media_type, fact),
                "crop_plan": _crop_plan(media_type, preset["aspect_ratio"], fact),
                "uncertainty": _uncertainty(fact),
                "candidate_only": True,
                "deleted": False,
            }
            if media_type == "image":
                clip["hold_seconds"] = round(clip_duration, 3)
            else:
                clip["source_range"] = {
                    "start_seconds": 0.0,
                    "end_seconds": round(clip_duration, 3),
                }
            clips.append(clip)
            selected_ids.add(asset_id)
            elapsed += clip_duration

        excluded = [
            {
                "source_asset_id": str(asset["asset_id"]),
                "reason": "not_selected_for_current_candidate_recipe",
                "deleted": False,
                "review_required": True,
            }
            for asset in _mapping_rows(project.get("assets"))
            if str(asset["asset_id"]) not in selected_ids
        ]
        gaps = list(evidence.get("review_gaps") or [])
        if elapsed < float(preset["minimum_seconds"]):
            gaps.append(
                {
                    "reason": "insufficient_candidate_duration",
                    "actual_seconds": round(elapsed, 3),
                    "minimum_seconds": preset["minimum_seconds"],
                    "blocking": False,
                }
            )
        recipe = {
            "schema": RECIPE_SCHEMA,
            "project_id": project.get("project_id"),
            "project_identity_sha256": project.get("project_identity_sha256"),
            "evidence_sha256": evidence.get("evidence_sha256"),
            "event_plan_sha256": event_plan.get("event_plan_sha256"),
            "contract_sha256": _contract_hash(RECIPE_SCHEMA),
            "preset": preset_name,
            "target": dict(preset),
            "output_mode": "candidate_only",
            "review_required": True,
            "clips": clips,
            "unselected_candidates": excluded,
            "review_gaps": gaps,
            "privacy": {
                "source_paths_exported": False,
                "precise_gps_exported": False,
                "appearance_vectors_exported": False,
                "appearance_identity_claims": False,
            },
            "safety": {
                "source_mutation": False,
                "render_requested": False,
                "upload_requested": False,
                "publish_requested": False,
            },
        }
        recipe["recipe_sha256"] = _sha256_json(recipe)
        recipes[preset_name] = recipe
    return recipes


def write_personal_vlog_stage(
    project_dir: str | Path,
    filename: str,
    payload: Mapping[str, Any],
    *,
    operation: str,
) -> Path:
    """Write one validated project artifact under the project lock."""

    root = Path(project_dir).expanduser().resolve()
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise PersonalVlogBoundaryError(f"invalid_stage_filename:{filename}")
    with bundle_write_lock(root, operation=operation):
        target = root / filename
        write_json(target, dict(payload))
    return target


def load_personal_vlog_project(project_dir: str | Path) -> dict[str, Any]:
    value = read_json(Path(project_dir).expanduser().resolve() / "personal-vlog-project.json")
    if not isinstance(value, dict) or value.get("schema") != PROJECT_SCHEMA:
        raise PersonalVlogError("invalid_personal_vlog_project")
    return value


def public_recipe_projection(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Return the recipe after enforcing the no-path/no-precise-location contract."""

    payload = json.loads(json.dumps(recipe, ensure_ascii=False))
    forbidden_keys = {
        "source_path",
        "event_root",
        "project_dir",
        "latitude",
        "longitude",
        "gps",
        "appearance_vector",
        "face_embedding",
    }
    _reject_forbidden_keys(payload, forbidden_keys)
    return payload


def _bounded_inventory(root: Path, *, max_depth: int, max_items: int) -> list[_ObservedFile]:
    observed: list[_ObservedFile] = []
    seen = 0
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        _reject_linklike(directory, label="directory")
        try:
            entries = sorted(directory.iterdir(), key=lambda value: value.name.casefold())
        except OSError as exc:
            raise PersonalVlogBoundaryError(f"cannot_enumerate:{directory}:{exc}") from exc
        child_directories: list[Path] = []
        for entry in entries:
            if _is_hidden(entry):
                raise PersonalVlogBoundaryError(
                    f"hidden_entry_requires_explicit_review:{entry.relative_to(root).as_posix()}"
                )
            _reject_linklike(entry, label="source_entry")
            _require_within(root, entry, label="source_entry")
            if entry.is_dir():
                if depth >= max_depth:
                    raise PersonalVlogBoundaryError(
                        f"maximum_depth_exceeded:{entry.relative_to(root).as_posix()}"
                    )
                child_directories.append(entry)
                continue
            if not entry.is_file():
                raise PersonalVlogBoundaryError(
                    f"unsupported_filesystem_entry:{entry.relative_to(root).as_posix()}"
                )
            seen += 1
            if seen > max_items:
                raise PersonalVlogBoundaryError(
                    f"maximum_item_count_exceeded:{seen}>{max_items}"
                )
            before = entry.stat()
            digest = sha256_file(entry)
            after = entry.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or getattr(before, "st_ino", None) != getattr(after, "st_ino", None)
            ):
                raise PersonalVlogDriftError(
                    f"source_changed_during_inventory:{entry.relative_to(root).as_posix()}"
                )
            media_type = _media_type(entry.suffix)
            observed.append(
                _ObservedFile(
                    path=entry.resolve(strict=True),
                    relative_path=entry.relative_to(root).as_posix(),
                    media_type=media_type,
                    size_bytes=int(after.st_size),
                    mtime_ns=int(after.st_mtime_ns),
                    sha256=digest,
                )
            )
        for child in reversed(child_directories):
            stack.append((child, depth + 1))
    return observed


def _automatic_boundaries(
    assets: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    *,
    time_gap_seconds: float,
) -> tuple[list[int], str, str | None]:
    if len(assets) < 4:
        return _time_gap_boundaries(assets, time_gap_seconds), "time_gap_v1", (
            "ruptures_pelt_requires_at_least_four_assets"
        )
    timestamps = [_parse_time(str(row.get("capture_time") or "")) for row in assets]
    features: list[list[float]] = []
    previous = timestamps[0]
    for index, (asset, timestamp) in enumerate(zip(assets, timestamps, strict=True)):
        gap = 0.0 if index == 0 else max(0.0, timestamp - previous)
        previous = timestamp
        kind = str(asset.get("media_type") or "")
        fact = evidence_by_id.get(str(asset.get("asset_id") or "")) or {}
        semantic = fact.get("semantic") if isinstance(fact.get("semantic"), Mapping) else {}
        audio = fact.get("audio") if isinstance(fact.get("audio"), Mapping) else {}
        features.append(
            [
                min(gap / max(time_gap_seconds, 1.0), 4.0),
                1.0 if kind == "image" else 0.0,
                1.0 if kind == "video" else 0.0,
                1.0 if kind == "audio" else 0.0,
                _clamp01(semantic.get("activity_relevance", 0.5)),
                _clamp01(audio.get("value_score", 0.0)),
            ]
        )
    try:
        from .filmed_structure import _run_pelt

        predicted = _run_pelt(features, penalty=2.0, min_size=2, jump=1)
        boundaries = _validate_boundaries(predicted, len(assets))
        return boundaries, "ruptures.Pelt", None
    except RuntimeError as exc:
        return (
            _time_gap_boundaries(assets, time_gap_seconds),
            "time_gap_v1",
            f"ruptures_unavailable:{str(exc)[:240]}",
        )


def _time_gap_boundaries(
    assets: Sequence[Mapping[str, Any]], time_gap_seconds: float
) -> list[int]:
    if not assets:
        return []
    boundaries: list[int] = []
    previous = _parse_time(str(assets[0].get("capture_time") or ""))
    for index, asset in enumerate(assets[1:], start=1):
        current = _parse_time(str(asset.get("capture_time") or ""))
        if current - previous >= max(1.0, float(time_gap_seconds)):
            boundaries.append(index)
        previous = current
    boundaries.append(len(assets))
    return boundaries


def _validate_boundaries(values: Iterable[int], item_count: int) -> list[int]:
    boundaries = sorted({int(value) for value in values})
    if any(value <= 0 or value > item_count for value in boundaries):
        raise PersonalVlogBoundaryError("event_boundary_out_of_range")
    if item_count and (not boundaries or boundaries[-1] != item_count):
        boundaries.append(item_count)
    return boundaries


def _score_asset(
    asset: Mapping[str, Any], fact: Mapping[str, Any], *, index: int
) -> dict[str, Any]:
    semantic = fact.get("semantic") if isinstance(fact.get("semantic"), Mapping) else {}
    technical = fact.get("technical") if isinstance(fact.get("technical"), Mapping) else {}
    audio = fact.get("audio") if isinstance(fact.get("audio"), Mapping) else {}
    appearance = fact.get("appearance") if isinstance(fact.get("appearance"), Mapping) else {}
    components = {
        "activity_relevance": _clamp01(semantic.get("activity_relevance", 0.5)),
        "technical_quality": _clamp01(technical.get("quality_score", 0.5)),
        "appearance_or_interaction_continuity": (
            0.7 if _clean_strings(appearance.get("cluster_ids")) else 0.4
        ),
        "audio_value": _clamp01(audio.get("value_score", 0.0)),
        "novelty_and_diversity": 0.2 if asset.get("duplicate_group") else 0.7,
        "chronology_and_story_role": max(0.4, 1.0 - min(index, 20) * 0.02),
    }
    penalties = {
        "duplicate": 0.20 if asset.get("duplicate_group") else 0.0,
        "privacy_risk": 0.20 if semantic.get("privacy_risk") else 0.0,
        "severe_quality_failure": 0.30 if technical.get("severe_failure") else 0.0,
        "evidence_conflict": 0.20 if semantic.get("evidence_conflict") else 0.0,
    }
    weighted = sum(components[key] * SCORE_WEIGHTS[key] for key in SCORE_WEIGHTS)
    total = max(0.0, weighted - sum(penalties.values()))
    return {
        "weights": dict(SCORE_WEIGHTS),
        "components": {key: round(value, 4) for key, value in components.items()},
        "penalties": {key: round(value, 4) for key, value in penalties.items()},
        "total": round(total, 4),
    }


def _candidate_duration(media_type: str, fact: Mapping[str, Any]) -> float:
    technical = fact.get("technical") if isinstance(fact.get("technical"), Mapping) else {}
    duration = _optional_nonnegative_float(technical.get("duration_seconds"))
    if media_type == "image":
        quality = _clamp01(technical.get("quality_score", 0.5))
        return round(min(4.5, max(2.0, 2.0 + quality * 2.5)), 3)
    if duration is None:
        return 8.0 if media_type == "video" else 7.0
    return max(0.0, min(duration, 12.0 if media_type == "video" else 15.0))


def _audio_policy(media_type: str, fact: Mapping[str, Any]) -> dict[str, Any]:
    audio = fact.get("audio") if isinstance(fact.get("audio"), Mapping) else {}
    preserve = bool(audio.get("preserve_original_sound")) or _clamp01(
        audio.get("value_score", 0.0)
    ) >= 0.65
    return {
        "role": "independent_audio_candidate" if media_type == "audio" else "source_audio",
        "preserve_original_sound": preserve,
        "background_music_must_duck": preserve,
        "automatic_music_selection": False,
    }


def _crop_plan(media_type: str, aspect_ratio: str, fact: Mapping[str, Any]) -> dict[str, Any]:
    appearance = fact.get("appearance") if isinstance(fact.get("appearance"), Mapping) else {}
    safe_subject = bool(appearance.get("cluster_ids"))
    return {
        "target_aspect_ratio": aspect_ratio,
        "mode": "review_required" if media_type != "audio" else "not_applicable",
        "ken_burns_candidate": bool(media_type == "image" and safe_subject),
        "requires_safe_subject_region": media_type == "image",
        "automatic_execution": False,
    }


def _uncertainty(fact: Mapping[str, Any]) -> dict[str, Any]:
    semantic = fact.get("semantic") if isinstance(fact.get("semantic"), Mapping) else {}
    limitations = _clean_strings(fact.get("limitations"))
    return {
        "evidence_level": str(semantic.get("evidence_level") or "candidate_weak"),
        "limitations": limitations,
        "human_review_required": True,
    }


def _require_project_match(
    project: Mapping[str, Any], payload: Mapping[str, Any], *, label: str
) -> None:
    expected = str(project.get("project_identity_sha256") or "")
    actual = str(payload.get("project_identity_sha256") or "")
    if not expected or expected != actual:
        raise PersonalVlogDriftError(f"{label}_project_identity_mismatch")


def _reject_overlapping_roots(source_root: Path, target_root: Path) -> None:
    if _is_relative_to(target_root, source_root) or _is_relative_to(source_root, target_root):
        raise PersonalVlogBoundaryError(
            "project_dir_and_event_root_must_not_overlap"
        )


def _reject_link_components(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        _reject_linklike(current, label="source_component")


def _reject_linklike(path: Path, *, label: str) -> None:
    try:
        if path.is_symlink():
            raise PersonalVlogBoundaryError(f"{label}_is_symlink:{path}")
        is_junction = getattr(path, "is_junction", None)
        if callable(is_junction) and is_junction():
            raise PersonalVlogBoundaryError(f"{label}_is_junction:{path}")
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0) or 0)
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0) or 0)
        if reparse_flag and attributes & reparse_flag:
            raise PersonalVlogBoundaryError(f"{label}_is_reparse_point:{path}")
    except FileNotFoundError as exc:
        raise PersonalVlogBoundaryError(f"{label}_missing:{path}") from exc


def _is_hidden(path: Path) -> bool:
    if path.name.startswith("."):
        return True
    try:
        hidden_flag = int(getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0) or 0)
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0) or 0)
        return bool(hidden_flag and attributes & hidden_flag)
    except FileNotFoundError:
        return False


def _require_within(root: Path, path: Path, *, label: str) -> None:
    if not _is_relative_to(path.resolve(strict=True), root.resolve(strict=True)):
        raise PersonalVlogBoundaryError(f"{label}_outside_root:{path}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _media_type(suffix: str) -> str | None:
    extension = suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    return None


def _mapping_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _clean_strings(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _anonymous_clusters(value: object) -> list[str]:
    return sorted(
        {
            item
            for item in _clean_strings(value)
            if item.startswith("appearance_cluster_")
            and item.replace("appearance_cluster_", "").replace("-", "").isalnum()
        }
    )


def _safe_cluster_id(value: object, prefix: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if not text.startswith(f"{prefix}_cluster_"):
        raise PersonalVlogBoundaryError(f"invalid_{prefix}_cluster_id")
    if not text.replace("_", "").replace("-", "").isalnum():
        raise PersonalVlogBoundaryError(f"invalid_{prefix}_cluster_id")
    return text


def _optional_nonnegative_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("expected a finite non-negative number")
    return round(number, 6)


def _optional_positive_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    number = int(value)
    if number <= 0:
        raise ValueError("expected a positive integer")
    return number


def _clamp01(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("score must be finite")
    return min(1.0, max(0.0, number))


def _orientation(width: int | None, height: int | None) -> str:
    if not width or not height:
        return "unknown"
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def _parse_time(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PersonalVlogBoundaryError(f"invalid_capture_time:{value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _mtime_iso(mtime_ns: int) -> str:
    return datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    )


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:20]}"


def _contract_hash(schema: str) -> str:
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_forbidden_keys(value: object, forbidden: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in forbidden:
                raise PersonalVlogBoundaryError(f"forbidden_public_field:{key}")
            _reject_forbidden_keys(child, forbidden)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child, forbidden)
