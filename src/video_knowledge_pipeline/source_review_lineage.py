from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from .canonical_json import canonical_json_sha256
from .content_asset_batch import discover_bundles
from .file_hash import sha256_file
from .models import now_iso
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.source_review_lineage.v1"
REVIEW_ARTIFACTS = {
    "human_corrected_transcript": (
        "human-corrected-transcript.json",
        "corrected-transcript.json",
        "source-arbitrated-transcript.json",
    ),
    "speaker_review": ("speaker-review.json", "speaker-global-alignment.json"),
    "subtitle_review": ("human-reviewed-subtitle-track.json",),
    "review_notes": ("review-notes.json",),
    "privacy_review": ("privacy-review.json",),
    "semantic_fact_review": ("medical-insurance-fact-review.json", "fact-review.json"),
}


def discover_source_review_lineage(
    bundle_dir: str | Path,
    *,
    search_roots: Iterable[str | Path] | None = None,
    apply: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Discover exact same-source human work without silently applying it."""

    root = Path(bundle_dir).expanduser().resolve()
    identity = _bundle_source_identity(root)
    roots = _search_roots(root, search_roots)
    candidates: list[dict[str, Any]] = []
    for search_root in roots:
        for candidate in discover_bundles(search_root):
            if candidate == root:
                continue
            candidate_identity = _bundle_source_identity(candidate)
            match = _identity_match(identity, candidate_identity)
            if not match["matched"]:
                continue
            artifacts = _review_artifacts(candidate)
            human_artifacts = [row for row in artifacts if row.get("human_confirmed")]
            if not human_artifacts:
                continue
            candidates.append(
                {
                    "bundle_dir": str(candidate),
                    "identity": candidate_identity,
                    "match": match,
                    "review_artifacts": artifacts,
                    "review_artifact_count": len(artifacts),
                    "human_review_artifact_count": len(human_artifacts),
                }
            )
    candidates.sort(
        key=lambda row: (
            int(row["match"].get("strength") or 0),
            int(row.get("review_artifact_count") or 0),
            str(row.get("bundle_dir") or ""),
        ),
        reverse=True,
    )
    selected = candidates[0] if candidates else None
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "source_identity": identity,
        "search_roots": [str(path) for path in roots],
        "status": "prior_review_available" if selected else "no_prior_review_found",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_candidate": selected or {},
        "applied": False,
        "lineage_revision": "",
        "operator_boundary": {
            "read_only_discovery": True,
            "explicit_apply_required": True,
            "does_not_copy_or_overwrite_reviewed_text": True,
            "same_title_is_not_source_identity": True,
            "unresolved_speaker_mapping_remains_unresolved": True,
        },
        "updated_at": now_iso(),
    }
    if apply:
        if not selected:
            result["status"] = "blocked_no_prior_review"
        elif int(selected["match"].get("strength") or 0) < 2:
            result["status"] = "blocked_weak_source_identity"
        else:
            result["status"] = "review_lineage_bound"
            result["applied"] = True
    result["lineage_revision"] = canonical_json_sha256(
        {key: value for key, value in result.items() if key not in {"updated_at", "lineage_revision"}}
    )
    if write:
        write_json(root / "source-review-lineage.json", result)
        if result["applied"]:
            _bind_manifest(root, result)
    return result


def load_bound_source_review_lineage(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    path = root / "source-review-lineage.json"
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return {}
    return value


def validate_bound_source_review_lineage(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    lineage = load_bound_source_review_lineage(root)
    if not lineage or not lineage.get("applied"):
        return {"status": "not_bound", "passed": False, "lineage": lineage}
    expected_revision = str(lineage.get("lineage_revision") or "")
    actual_revision = canonical_json_sha256(
        {key: value for key, value in lineage.items() if key not in {"updated_at", "lineage_revision"}}
    )
    failures: list[str] = []
    if expected_revision != actual_revision:
        failures.append("lineage_revision_drift")
    selected = lineage.get("selected_candidate") if isinstance(lineage.get("selected_candidate"), dict) else {}
    for row in selected.get("review_artifacts") or []:
        if not isinstance(row, dict):
            failures.append("invalid_review_artifact_row")
            continue
        path = Path(str(row.get("path") or "")).expanduser()
        if not path.is_file():
            failures.append(f"missing_review_artifact:{path}")
            continue
        if sha256_file(path) != str(row.get("sha256") or ""):
            failures.append(f"review_artifact_hash_drift:{path}")
    current_identity = _bundle_source_identity(root)
    match = _identity_match(current_identity, lineage.get("source_identity") or {})
    if not match.get("matched"):
        failures.append("current_source_identity_drift")
    return {
        "status": "valid" if not failures else "invalid",
        "passed": not failures,
        "failures": failures,
        "lineage": lineage,
    }


def _bundle_source_identity(root: Path) -> dict[str, Any]:
    manifest = _mapping(root / "manifest.json")
    index_path = root / str(manifest.get("source_artifacts_json") or "source-artifacts.json")
    index = _mapping(index_path)
    rows = index.get("artifacts") if isinstance(index.get("artifacts"), list) else []
    media = next(
        (
            row
            for row in rows
            if isinstance(row, dict)
            and (row.get("kind") == "media" or row.get("key") == "video")
        ),
        {},
    )
    video_id = str(media.get("video_id") or manifest.get("video_id") or "").strip()
    media_path_value = str(media.get("path") or media.get("video_path") or manifest.get("media_path") or "").strip()
    media_path = Path(media_path_value).expanduser() if media_path_value else None
    declared_sha = str(
        media.get("sha256")
        or manifest.get("source_media_sha256")
        or manifest.get("media_sha256")
        or ""
    ).strip().lower()
    source_sha = declared_sha if len(declared_sha) == 64 else ""
    if not source_sha and media_path and media_path.is_file() and media_path.stat().st_size <= 64 * 1024 * 1024:
        source_sha = sha256_file(media_path)
    return {
        "video_id": video_id,
        "source_media_sha256": source_sha,
        "media_path": str(media_path.resolve()) if media_path and media_path.exists() else media_path_value,
        "media_exists": bool(media_path and media_path.is_file()),
    }


def _identity_match(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_sha = str(left.get("source_media_sha256") or "")
    right_sha = str(right.get("source_media_sha256") or "")
    if left_sha and right_sha:
        return {"matched": left_sha == right_sha, "kind": "sha256", "strength": 3}
    left_id = str(left.get("video_id") or "")
    right_id = str(right.get("video_id") or "")
    if left_id and right_id and left_id == right_id and left_id.startswith("video_"):
        return {
            "matched": True,
            "kind": "video_id_sha256_prefix",
            "strength": 2,
            "detail": "VKP video_id is derived from the first 12 SHA-256 characters",
        }
    return {"matched": False, "kind": "none", "strength": 0}


def _review_artifacts(bundle: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, names in REVIEW_ARTIFACTS.items():
        path = next((bundle / name for name in names if (bundle / name).is_file()), None)
        if path is None:
            path = next((bundle / "exports" / name for name in names if (bundle / "exports" / name).is_file()), None)
        if path is None:
            continue
        payload = _mapping(path)
        status = str(payload.get("status") or payload.get("final_status") or "present").strip()
        if kind == "review_notes" and not _review_notes_have_decisions(payload):
            continue
        human_confirmed = _human_confirmed(kind, payload, path=path)
        rows.append(
            {
                "kind": kind,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "status": status,
                "human_confirmed": human_confirmed,
                "binding_capability": _binding_capability(kind, payload),
            }
        )
    return rows


def _human_confirmed(kind: str, payload: dict[str, Any], *, path: Path) -> bool:
    status = str(payload.get("status") or payload.get("final_status") or "").lower()
    if payload.get("human_confirmed") is True:
        return True
    if path.name.lower().startswith("human-"):
        return True
    if kind == "review_notes":
        return _review_notes_have_decisions(payload)
    if kind == "speaker_review":
        return status.startswith("human_confirmed")
    return status in {"human_confirmed", "approved", "complete", "completed", "applied"}


def _binding_capability(kind: str, payload: dict[str, Any]) -> str:
    if kind != "speaker_review":
        return "review_evidence"
    status = str(payload.get("status") or "").lower()
    mappings = payload.get("speaker_mappings") or payload.get("mappings") or []
    if status == "human_confirmed_count" and not mappings:
        return "participant_count_only"
    return "speaker_roles" if mappings else "speaker_clusters_only"


def _review_notes_have_decisions(payload: dict[str, Any]) -> bool:
    for key in ("reviews", "decisions", "items", "notes"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _search_roots(root: Path, values: Iterable[str | Path] | None) -> list[Path]:
    raw = list(values or [])
    if not raw:
        env_value = os.environ.get("VKP_REVIEW_SEARCH_ROOTS", "")
        if env_value:
            raw.extend(value for value in env_value.split(os.pathsep) if value)
    if not raw:
        for parent in root.parents:
            if parent.name.lower() == "used-by-codex":
                raw.extend([parent / "video-knowledge-runs", parent / "video-knowledge-output"])
                break
    roots: list[Path] = []
    seen: set[str] = set()
    for value in raw:
        path = Path(value).expanduser().resolve()
        key = str(path).lower()
        if path.is_dir() and key not in seen:
            seen.add(key)
            roots.append(path)
    return roots


def _bind_manifest(root: Path, result: dict[str, Any]) -> None:
    manifest_path = root / "manifest.json"
    manifest = _mapping(manifest_path)
    selected = result.get("selected_candidate") if isinstance(result.get("selected_candidate"), dict) else {}
    manifest["source_review_lineage_json"] = "source-review-lineage.json"
    manifest["source_review_lineage_revision"] = result.get("lineage_revision")
    manifest["inherited_review_artifacts"] = [
        row
        for row in selected.get("review_artifacts") or []
        if isinstance(row, dict) and row.get("human_confirmed")
    ]
    manifest["inherited_review_source_bundle"] = selected.get("bundle_dir") or ""
    for row in manifest["inherited_review_artifacts"]:
        kind = str(row.get("kind") or "")
        path = str(row.get("path") or "")
        if kind == "human_corrected_transcript":
            manifest["human_corrected_transcript_json"] = path
        elif kind == "speaker_review":
            manifest["speaker_review_json"] = path
        elif kind == "privacy_review":
            manifest["privacy_review_json"] = path
        elif kind == "semantic_fact_review":
            manifest["medical_insurance_fact_review_json"] = path
    write_json(manifest_path, manifest)


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}
