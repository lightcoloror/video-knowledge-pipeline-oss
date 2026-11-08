"""Export an auditable local frame manifest from a VKP bundle.

Public adaptation of VKP's original task-local frame-manifest exporter. Source
evidence: upstream commit ``d8541dc19247e37e4d6bc6a3a1e1689a2a98b6f0``, blob
``0dca348febbd3b5aac8725020ab32137751705e8``, path
``scripts/export-offline-frame-manifest.py`` (AGPL-3.0-only). The
``video_knowledge_pipeline.frame_manifest.v1`` schema, the parsing/row/hash
functions and the original row fields are retained.

Boundaries added here: timeline/manifest shapes are validated and only genuinely
malformed input is converted to an input error; permission and other runtime
errors propagate instead of becoming blocked rows. The two fixed bundle JSON
members are resolved, contained by the resolved bundle root and required to be
regular files before any byte is read, so an outside-root symlink/junction,
directory or FIFO is refused before open while an absent optional
``manifest.json`` remains allowed. Frame references are
nonempty strings resolved against the bundle root and must remain inside the
resolved root before any read; traversal, absolute-outside, foreign-drive/UNC and
resolved symlink escapes are reported as blocked. Missing/blocked/directory
references stay visible with counts/locations and a nonzero result. Repeated
timeline context is kept when files are deduplicated. Source-media discovery
retains the original ``video_key`` / ``media_path`` / ``source_media_path`` /
``video_path`` order for regular resolved paths contained by the bundle root;
outside values remain unread unless the operator passes ``--source-media``
explicitly; contained same-root bundle media may be discovered and hashed
without that flag, and the manifest records which authorization was used.
Both output destinations are validated before any write, so a
symlink/escape/nonregular/unrelated artifact is refused while a legitimate owned
frame-manifest refresh remains allowed; a non-regular destination is refused even
when its JSON bytes were parsed earlier. A pure ``build_frame_manifest`` result
reports destination validation as not checked/deferred; only the writing path,
after successful checks, reports the actual validated status. ``manifest.json``
is only updated with ``--register``. No FFmpeg probe is performed, so extractor
provenance is reported as unknown rather than fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "video_knowledge_pipeline.frame_manifest.v1"
TIMELINE_NAME = "timeline.json"
BUNDLE_MANIFEST_NAME = "manifest.json"
FRAME_MANIFEST_NAME = "frame-manifest.json"

DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")

BLOCKED_EMPTY = "empty_reference"
BLOCKED_UNC = "unc_reference"
BLOCKED_FOREIGN_DRIVE = "foreign_drive_reference"
BLOCKED_DRIVE_RELATIVE = "drive_relative_reference"
BLOCKED_UNRESOLVABLE = "unresolvable_reference"
BLOCKED_OUTSIDE = "resolved_outside_bundle_root"
BLOCKED_TRAVERSAL = "path_traversal_outside_bundle_root"
BLOCKED_NOT_REGULAR = "not_a_regular_file"


class FramePlanError(ValueError):
    """The bundle input is structurally invalid; no frame manifest is written."""


# ---------------------------------------------------------------------------
# Small local seams. Tests replace these to stay synthetic and file-free.
# ---------------------------------------------------------------------------


def _resolve_bundle_dir(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def _resolve_operator_path(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def _path_exists(path: Path) -> bool:
    # Deliberately stat-based instead of Path.exists(): FileNotFoundError and
    # NotADirectoryError mean "absent", while PermissionError/EACCES and other
    # runtime failures propagate instead of being silently treated as missing.
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def _is_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.stat().st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False


def _is_symlink(path: Path) -> bool:
    return path.is_symlink()


def _resolve_destination(root: Path, destination: Path) -> Path:
    # ``root`` is accepted for symmetry with the other sealed helpers; the
    # destination is fixed by this exporter, so only its own resolution matters.
    return destination.resolve()


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Resolved-root containment
# ---------------------------------------------------------------------------


def _reference_drive(text: str) -> str | None:
    match = DRIVE_PREFIX.match(text)
    return match.group(0) if match else None


def _is_unc_reference(text: str) -> bool:
    return text.startswith("\\\\") or text.startswith("//")


def _is_absolute_reference(text: str) -> bool:
    return bool(text.startswith(("/", "\\")) or DRIVE_PREFIX.match(text) and text[2:3] in ("/", "\\"))


def _join_reference(root: Path, reference: Path) -> Path:
    """Join a relative reference against the bundle root; absolute stays absolute."""
    return reference if reference.is_absolute() else root / reference


def _is_within_root(root: Path, path: Path) -> bool:
    root_text = os.path.normcase(os.path.abspath(str(root)))
    path_text = os.path.normcase(os.path.abspath(str(path)))
    if path_text == root_text:
        return True
    prefix = root_text if root_text.endswith(os.sep) else root_text + os.sep
    return path_text.startswith(prefix)


def _resolve(root: Path, reference: Path) -> Path:
    """Adapted original ``_resolve``: join against the bundle root, then resolve.

    Containment is checked by the caller before any read; this helper no longer
    accepts manifest values as authority and no longer silently falls back.
    """
    return _join_reference(root, reference).resolve()


def _blocked(reference: str, reason: str, detail: str = "") -> dict[str, Any]:
    return {"status": "blocked", "reference": reference, "resolved": None, "blocked_reason": reason, "blocked_detail": detail}


def _classify_reference(root: Path, raw: Any) -> dict[str, Any]:
    """Classify one declared frame reference without reading its bytes.

    Only malformed reference text is converted to a blocked row. PermissionError
    and other OSError from resolution propagate so the caller stops before any
    later probe or write.
    """
    if not isinstance(raw, str) or not raw.strip():
        return _blocked("" if raw is None else str(raw), BLOCKED_EMPTY)
    if "\x00" in raw:
        return _blocked(raw, BLOCKED_UNRESOLVABLE, "reference contains a NUL byte")
    if _is_unc_reference(raw):
        return _blocked(raw, BLOCKED_UNC)
    drive = _reference_drive(raw)
    root_drive = _reference_drive(str(root))
    if drive is not None and os.path.normcase(drive) != os.path.normcase(root_drive or ""):
        return _blocked(raw, BLOCKED_FOREIGN_DRIVE)
    try:
        reference = Path(raw)
    except ValueError as exc:  # malformed path text only
        return _blocked(raw, BLOCKED_UNRESOLVABLE, str(exc))
    if drive is not None and not _is_absolute_reference(raw):
        return _blocked(raw, BLOCKED_DRIVE_RELATIVE)
    resolved = _resolve(root, reference)  # OSError/PermissionError propagates
    if not _is_within_root(root, resolved):
        reason = BLOCKED_TRAVERSAL if ".." in reference.parts else BLOCKED_OUTSIDE
        return _blocked(raw, reason, str(resolved))
    return {"status": "contained", "reference": raw, "resolved": resolved, "blocked_reason": None, "blocked_detail": ""}


# ---------------------------------------------------------------------------
# Timeline parsing/validation
# ---------------------------------------------------------------------------


def _list_field(container: Any, field: str, context: str) -> list[Any]:
    if container is None:
        return []
    if not isinstance(container, dict):
        raise FramePlanError(f"{context} must be a JSON object")
    value = container.get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        raise FramePlanError(f"{context}.{field} must be a JSON array")
    return value


def _iter_timeline_items(timeline: Any) -> list[dict[str, Any]]:
    if not isinstance(timeline, list):
        raise FramePlanError("timeline.json must contain a JSON array")
    items: list[dict[str, Any]] = []
    for position, item in enumerate(timeline):
        if not isinstance(item, dict):
            raise FramePlanError(f"timeline[{position}] must be a JSON object")
        items.append(item)
    return items


def _iter_references(items: Iterable[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], str, str]], list[dict[str, Any]]]:
    """Yield ``(timeline_item, role, raw_reference)`` plus visible skipped assets."""
    occurrences: list[tuple[dict[str, Any], str, str]] = []
    skipped: list[dict[str, Any]] = []
    for item in items:
        index = item.get("index")
        visual = item.get("integrated_visual")
        evidence = _list_field(visual, "evidence_frame_paths", f"timeline[{index}].integrated_visual")
        temporal = _list_field(item, "temporal_frame_paths", f"timeline[{index}]")
        assets = _list_field(item, "assets", f"timeline[{index}]")
        keyframes: list[Any] = []
        for asset_index, asset in enumerate(assets):
            if not isinstance(asset, dict):
                raise FramePlanError(f"timeline[{index}].assets[{asset_index}] must be a JSON object")
            source = asset.get("source")
            path_value = asset.get("path")
            if source is not None and not isinstance(source, str):
                raise FramePlanError(f"timeline[{index}].assets[{asset_index}].source must be a string")
            if path_value is not None and not isinstance(path_value, str):
                raise FramePlanError(f"timeline[{index}].assets[{asset_index}].path must be a string")
            value = source if source else path_value
            if value is None:
                skipped.append({"timeline_index": index, "reason": "asset_without_source_or_path"})
                continue
            keyframes.append(value)
        keyframes.extend(evidence)
        for role, values in (("keyframe", keyframes), ("temporal_frame", temporal)):
            for value in values:
                if not isinstance(value, str):
                    raise FramePlanError(f"timeline[{index}].{role} reference must be a string")
                occurrences.append((item, role, value))
    return occurrences, skipped


# ---------------------------------------------------------------------------
# Rows and manifest
# ---------------------------------------------------------------------------


def _occurrence_context(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "timeline_index": item.get("index"),
        "start_seconds": item.get("start"),
        "end_seconds": item.get("end"),
    }


def _row_for(root: Path, item: dict[str, Any], role: str, raw: str) -> dict[str, Any]:
    classified = _classify_reference(root, raw)
    context = _occurrence_context(item)
    if classified["status"] == "blocked":
        return {
            "timeline_index": item.get("index"),
            "start_seconds": item.get("start"),
            "end_seconds": item.get("end"),
            "role": role,
            "path": classified["reference"],
            "reference": classified["reference"],
            "exists": False,
            "sha256": "",
            "evidence_status": "blocked",
            "blocked_reason": classified["blocked_reason"],
            "blocked_detail": classified["blocked_detail"],
            "resolved_path": None,
            "occurrences": [context],
            "occurrence_count": 1,
        }

    resolved: Path = classified["resolved"]
    if _is_regular_file(resolved):  # PermissionError propagates
        status = "available"
        exists = True
        digest = _sha256(resolved)
        reason = None
        detail = ""
    elif _path_exists(resolved):
        status = "blocked"
        exists = False
        digest = ""
        reason = BLOCKED_NOT_REGULAR
        detail = "reference exists but is not a regular file; no bytes were hashed"
    else:
        status = "missing"
        exists = False
        digest = ""
        reason = None
        detail = ""
    return {
        "timeline_index": item.get("index"),
        "start_seconds": item.get("start"),
        "end_seconds": item.get("end"),
        "role": role,
        "path": str(resolved),
        "reference": raw,
        "exists": exists,
        "sha256": digest,
        "evidence_status": status,
        "blocked_reason": reason,
        "blocked_detail": detail,
        "resolved_path": str(resolved),
        "occurrences": [context],
        "occurrence_count": 1,
    }


def _dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    # Case-sensitive on purpose: two distinct declared paths must stay distinct.
    if row["evidence_status"] == "blocked":
        return ("blocked", row["role"], row["reference"], row["blocked_reason"])
    return (row["evidence_status"], row["role"], row["resolved_path"])


def _build_rows(root: Path, occurrences: list[tuple[dict[str, Any], str, str]]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item, role, raw in occurrences:
        row = _row_for(root, item, role, raw)
        key = _dedupe_key(row)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = row
            rows.append(row)
            continue
        existing["occurrences"].append(_occurrence_context(item))
        existing["occurrence_count"] = len(existing["occurrences"])
    occurrence_count = sum(row["occurrence_count"] for row in rows)
    return rows, occurrence_count


# ---------------------------------------------------------------------------
# Source media: original selection order, resolved-root containment
# ---------------------------------------------------------------------------


def _media_candidates(timeline: list[dict[str, Any]], bundle_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in timeline:
        value = item.get("video_key")
        if value not in (None, ""):
            candidates.append(
                {
                    "origin": "bundle_reported",
                    "field": "timeline.json:video_key",
                    "timeline_index": item.get("index"),
                    "value": value,
                }
            )
    for field in ("media_path", "source_media_path", "video_path"):
        value = bundle_manifest.get(field)
        if value not in (None, ""):
            candidates.append(
                {
                    "origin": "bundle_reported",
                    "field": f"manifest.json:{field}",
                    "timeline_index": None,
                    "value": value,
                }
            )
    return candidates


def _discover_source_media(
    root: Path,
    timeline: list[dict[str, Any]],
    bundle_manifest: dict[str, Any],
    explicit_path: str | None,
) -> dict[str, Any]:
    """Select source media without inferring authority from untrusted values.

    Original selection order is retained (timeline ``video_key`` first, then
    ``media_path`` / ``source_media_path`` / ``video_path``). Relative values
    resolve against the bundle root, not the ambient working directory. Only
    regular resolved paths contained by the bundle root are read/hashed; outside
    values are recorded as blocked and never read. An explicit operator
    ``--source-media`` path is the only way an outside-root media file is read.
    """
    if explicit_path:
        path = _resolve_operator_path(explicit_path)  # OSError propagates
        if not _is_regular_file(path):
            raise FramePlanError(f"declared source media is not a regular local file: {path}")
        entry = {
            "origin": "explicit_operator",
            "field": "--source-media",
            "timeline_index": None,
            "value": explicit_path,
            "status": "available",
            "blocked_reason": None,
            "path": str(path),
            "sha256": _sha256(path),
        }
        return {
            "declared": True,
            "status": "explicit_operator_declared",
            "path": entry["path"],
            "sha256": entry["sha256"],
            "selection": "--source-media",
            "contained_by_bundle_root": _is_within_root(root, path),
            "candidates": [entry],
        }

    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate in _media_candidates(timeline, bundle_manifest):
        value = candidate["value"]
        entry = dict(candidate)
        if not isinstance(value, str) or not value.strip():
            entry.update(status="ignored_invalid_value", blocked_reason=None, path="", sha256="")
            candidates.append(entry)
            continue
        classified = _classify_reference(root, value)
        if classified["status"] == "blocked":
            entry.update(
                status="blocked",
                blocked_reason=classified["blocked_reason"],
                path=classified["reference"],
                sha256="",
            )
            candidates.append(entry)
            continue
        resolved = classified["resolved"]
        if _is_regular_file(resolved):
            entry.update(status="available", blocked_reason=None, path=str(resolved), sha256=_sha256(resolved))
            candidates.append(entry)
            if selected is None:
                selected = entry
                break
        elif _path_exists(resolved):
            entry.update(
                status="blocked",
                blocked_reason=BLOCKED_NOT_REGULAR,
                path=str(resolved),
                sha256="",
            )
            candidates.append(entry)
        else:
            entry.update(status="missing", blocked_reason=None, path=str(resolved), sha256="")
            candidates.append(entry)

    if selected is not None:
        return {
            "declared": False,
            "status": "discovered_contained",
            "path": selected["path"],
            "sha256": selected["sha256"],
            "selection": selected["field"],
            "contained_by_bundle_root": True,
            "candidates": candidates,
        }
    return {
        "declared": False,
        "status": "not_declared" if not candidates else "no_contained_regular_candidate",
        "path": "",
        "sha256": "",
        "selection": None,
        "contained_by_bundle_root": None,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Output destination validation
# ---------------------------------------------------------------------------


def _validate_output_destination(
    root: Path,
    destination: Path,
    *,
    kind: str,
    known_valid_existing: bool = False,
) -> None:
    """Refuse unsafe output destinations before any write.

    Symlink destinations, resolved escapes/nonregular paths and unrelated
    existing ``frame-manifest.json`` content are refused. An existing destination
    must be a regular file even when ``known_valid_existing`` is true: parsing
    earlier JSON is not regular-file proof. A pre-existing owned frame manifest
    (same schema) is a legitimate refresh and is allowed. The regular
    ``manifest.json`` destination may be updated only under explicit
    ``--register`` and only when its existing bytes were already read as a JSON
    object. This is not a TOCTOU or OS-sandbox guarantee.
    """
    if _is_symlink(destination):
        raise FramePlanError(f"refusing to write through symlink output destination: {destination}")
    resolved = _resolve_destination(root, destination)  # OSError/PermissionError propagates
    if not _is_within_root(root, resolved):
        raise FramePlanError(f"output destination resolves outside bundle root: {destination} -> {resolved}")
    if not _path_exists(destination):
        return
    # ``known_valid_existing`` only records that the bundle manifest JSON was
    # already parsed; it never substitutes for the regular-file check below.
    if not _is_regular_file(destination):
        raise FramePlanError(f"refusing to replace non-regular output destination: {destination}")
    if kind == "frame_manifest":
        try:
            existing = _read(destination)
        except (json.JSONDecodeError, UnicodeDecodeError, IsADirectoryError) as exc:
            raise FramePlanError(f"refusing to replace unreadable frame-manifest.json: {exc}") from exc
        if not isinstance(existing, dict) or existing.get("schema") != SCHEMA:
            raise FramePlanError("refusing to replace an unrelated existing frame-manifest.json")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _incomplete_locations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for row in rows:
        if row["evidence_status"] == "available":
            continue
        locations.append(
            {
                "timeline_index": row["timeline_index"],
                "role": row["role"],
                "reference": row["reference"],
                "path": row["path"],
                "evidence_status": row["evidence_status"],
                "blocked_reason": row["blocked_reason"],
            }
        )
    return locations


def _source_media_authorization(source: dict[str, Any]) -> str:
    """Describe how this export's selected source media was authorized.

    Contained bundle-reported media is legitimately read without
    ``--source-media`` (same-root autodiscovery); only an outside-root selection
    needs the explicit operator path. ``none`` means no source media was selected.
    """
    status = source.get("status")
    if status == "discovered_contained":
        return "bundle_root_contained_autodiscovery"
    if status == "explicit_operator_declared":
        if source.get("contained_by_bundle_root"):
            return "explicit_operator_path_inside_bundle_root"
        return "explicit_operator_path_outside_bundle_root"
    return "none"


def build_frame_manifest(
    root: Path,
    timeline: Any,
    bundle_manifest: Any,
    *,
    source_media_path: str | None = None,
    bundle_manifest_present: bool = True,
) -> dict[str, Any]:
    """Compute (but do not write) the frame manifest; raises FramePlanError on bad input.

    Source-media metadata distinguishes contained bundle-root autodiscovery from
    an explicit operator path; it never implies that same-root discovery requires
    the flag or that outside-root media can be read without it.
    """
    items = _iter_timeline_items(timeline)
    if not isinstance(bundle_manifest, dict):
        raise FramePlanError("manifest.json must contain a JSON object")
    occurrences, skipped = _iter_references(items)
    if not occurrences:
        raise FramePlanError("timeline contains no frame references; refusing to export an empty manifest")
    source = _discover_source_media(root, items, bundle_manifest, source_media_path)
    source_media_authorization = _source_media_authorization(source)
    source["authorization"] = source_media_authorization
    rows, occurrence_count = _build_rows(root, occurrences)
    available = sum(1 for row in rows if row["evidence_status"] == "available")
    missing = sum(1 for row in rows if row["evidence_status"] == "missing")
    blocked = sum(1 for row in rows if row["evidence_status"] == "blocked")
    incomplete = bool(missing or blocked)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bundle_dir": str(root),
        "source_media_path": source["path"],
        "source_media_sha256": source["sha256"],
        "source_media": source,
        "extractor": {
            "name": "unknown",
            "version": None,
            "provenance": "not_probed",
            "note": (
                "this exporter does not run ffmpeg and cannot verify which process created frames; "
                "bundle-reported extractor metadata would be unverified"
            ),
        },
        "input_validation": {
            "timeline_item_count": len(items),
            "timeline_reference_count": occurrence_count,
            "bundle_manifest_present": bool(bundle_manifest_present),
            "skipped_asset_entries": skipped,
        },
        "frame_count": len(rows),
        "occurrence_count": occurrence_count,
        "available_frame_count": available,
        "missing_frame_count": missing,
        "blocked_frame_count": blocked,
        "incomplete": incomplete,
        "status": "incomplete" if incomplete else "complete",
        "incomplete_locations": _incomplete_locations(rows),
        "frames": rows,
        "operator_boundary": {
            "local_only": True,
            "no_external_vision": True,
            "no_upload": True,
            "bundle_root_source_media_autodiscovery": True,
            "outside_root_source_media_requires_explicit_operator_path": True,
            "source_media_authorization": source_media_authorization,
            "bundle_manifest_updated_only_with_register": True,
            "output_destinations_validated": False,
            "output_destinations_validation": "not_checked_pure_computation",
            "no_transaction_or_toctou_guarantee": True,
        },
    }


def _contained_bundle_input(root: Path, name: str) -> Path:
    """Resolve one fixed bundle member and require resolved-root containment.

    The existing resolution/containment helpers run before any byte read, so a
    member whose resolved target leaves the bundle root is refused before open
    even when the link itself sits inside the bundle directory. Resolution
    errors (for example ``PermissionError``) propagate and stop the export; they
    are not converted into a missing optional member.
    """
    resolved = _resolve(root, root / name)  # OSError/PermissionError propagates
    if not _is_within_root(root, resolved):
        raise FramePlanError(f"bundle input resolves outside bundle root: {name} -> {resolved}")
    return resolved


def _load_bundle(root: Path) -> tuple[Any, dict[str, Any], bool]:
    if not _is_directory(root):
        raise FramePlanError(f"bundle directory does not exist or is not a directory: {root}")
    timeline_path = _contained_bundle_input(root, TIMELINE_NAME)
    if not _path_exists(timeline_path):  # PermissionError propagates
        raise FramePlanError(f"missing required bundle input: {TIMELINE_NAME}")
    if not _is_regular_file(timeline_path):  # directory/FIFO/nonregular refused before open
        raise FramePlanError(f"required bundle input is not a regular file: {TIMELINE_NAME}")
    try:
        timeline = _read(timeline_path)
    except (json.JSONDecodeError, UnicodeDecodeError, IsADirectoryError) as exc:  # genuinely malformed input only
        raise FramePlanError(f"cannot parse {TIMELINE_NAME}: {exc}") from exc
    manifest_path = _contained_bundle_input(root, BUNDLE_MANIFEST_NAME)
    manifest_present = _path_exists(manifest_path)  # absent optional member remains allowed
    bundle_manifest: dict[str, Any] = {}
    if manifest_present:
        if not _is_regular_file(manifest_path):  # directory/FIFO/nonregular refused before open
            raise FramePlanError(f"bundle input is not a regular file: {BUNDLE_MANIFEST_NAME}")
        try:
            bundle_manifest = _read(manifest_path)
        except (json.JSONDecodeError, UnicodeDecodeError, IsADirectoryError) as exc:  # genuinely malformed only
            raise FramePlanError(f"cannot parse {BUNDLE_MANIFEST_NAME}: {exc}") from exc
        if not isinstance(bundle_manifest, dict):
            raise FramePlanError(f"{BUNDLE_MANIFEST_NAME} must contain a JSON object")
    return timeline, bundle_manifest, manifest_present


def run_export(
    bundle_dir: str | Path,
    *,
    source_media_path: str | None = None,
    register_bundle_manifest: bool = False,
) -> dict[str, Any]:
    """Validate, compute, then write the manifest; returns a summary.

    Raises FramePlanError before any write when the input is structurally invalid
    or an output destination is unsafe. Permission/runtime errors propagate.
    An incomplete bundle still writes its visible manifest and returns exit 1.
    """
    root = _resolve_bundle_dir(str(bundle_dir))
    timeline, bundle_manifest, manifest_present = _load_bundle(root)
    manifest = build_frame_manifest(
        root,
        timeline,
        bundle_manifest,
        source_media_path=source_media_path,
        bundle_manifest_present=manifest_present,
    )
    frame_destination = root / FRAME_MANIFEST_NAME
    _validate_output_destination(root, frame_destination, kind="frame_manifest")
    registration_destination: Path | None = None
    registration_skipped: str | None = None
    if register_bundle_manifest:
        if manifest["incomplete"]:
            registration_skipped = "incomplete_manifest_not_registered"
        else:
            registration_destination = root / BUNDLE_MANIFEST_NAME
            _validate_output_destination(
                root,
                registration_destination,
                kind="bundle_manifest",
                known_valid_existing=manifest_present,
            )
    # The pure computation above deliberately reported destination validation as
    # not checked. Only now, after the writing path has validated every
    # destination it will touch, may the persisted manifest report the actual
    # validated status. This is not a race-free guarantee.
    manifest["operator_boundary"]["output_destinations_validated"] = True
    manifest["operator_boundary"]["output_destinations_validation"] = "validated_before_write"
    _write_json(frame_destination, manifest)
    registered = False
    if registration_destination is not None:
        updated = dict(bundle_manifest)
        updated["frame_manifest"] = FRAME_MANIFEST_NAME
        _write_json(registration_destination, updated)
        registered = True
    return {
        "ok": not manifest["incomplete"],
        "status": manifest["status"],
        "exit_code": 1 if manifest["incomplete"] else 0,
        "manifest_path": str(frame_destination),
        "frame_count": manifest["frame_count"],
        "occurrence_count": manifest["occurrence_count"],
        "available_frame_count": manifest["available_frame_count"],
        "missing_frame_count": manifest["missing_frame_count"],
        "blocked_frame_count": manifest["blocked_frame_count"],
        "incomplete_locations": manifest["incomplete_locations"],
        "source_media_status": manifest["source_media"]["status"],
        "source_media_selection": manifest["source_media"]["selection"],
        "source_media_authorization": manifest["operator_boundary"]["source_media_authorization"],
        "registered_bundle_manifest": registered,
        "registration_skipped": registration_skipped,
        "extractor_provenance": manifest["extractor"]["provenance"],
        "output_destinations_validated": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export an auditable frame manifest from a VKP bundle")
    parser.add_argument("bundle_dir", help="bundle directory containing timeline.json")
    parser.add_argument(
        "--source-media",
        default=None,
        help="explicit operator-declared source media path; the only way outside-bundle media is read",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="opt in to updating manifest.json with the frame-manifest pointer (complete exports only)",
    )
    args = parser.parse_args(argv)
    try:
        summary = run_export(
            args.bundle_dir,
            source_media_path=args.source_media,
            register_bundle_manifest=args.register,
        )
    except FramePlanError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
