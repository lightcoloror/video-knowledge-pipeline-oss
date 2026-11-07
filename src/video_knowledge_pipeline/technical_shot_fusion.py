from __future__ import annotations

from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .technical_shot_detection import SCHEMA as TECHNICAL_SHOT_SCHEMA
from .transcript import format_timestamp


SCHEMA = "video_knowledge_pipeline.technical_shot_boundary_fusion.v1"
OUTPUT_PATH = "exports/technical-shot-boundary-fusion.json"
MARKDOWN_PATH = "exports/technical-shot-boundary-fusion.md"


def fuse_technical_shot_boundaries(
    bundle_dir: str | Path,
    candidate_paths: list[str | Path],
    *,
    frame_rate: float,
    tolerance_frames: int = 2,
    write: bool = True,
) -> dict[str, Any]:
    """Cluster exact detector candidates without choosing disputed boundaries.

    The output is review evidence. It deliberately is not a
    ``technical_shot_boundaries.v1`` artifact, so downstream shot analysis
    cannot silently treat an unresolved detector vote as a confirmed cut.
    """

    root = Path(bundle_dir).expanduser().resolve()
    fps = float(frame_rate)
    if fps <= 0:
        raise ValueError("frame_rate must be positive")
    tolerance = max(0, int(tolerance_frames)) / fps
    sources = [_load_source(root, value) for value in candidate_paths]
    if len(sources) < 2:
        raise ValueError("at least two technical-shot candidate artifacts are required")
    media_hashes = {
        str(source["payload"].get("media", {}).get("sha256") or "")
        for source in sources
        if str(source["payload"].get("media", {}).get("sha256") or "")
    }
    if len(media_hashes) > 1:
        raise ValueError("candidate artifacts bind different media SHA-256 values")

    observations = sorted(
        (
            {
                "seconds": float(boundary["seconds"]),
                "backend": source["backend"],
                "source_path": source["path"],
                "source_sha256": source["sha256"],
                "source_boundary_id": str(boundary.get("boundary_id") or ""),
            }
            for source in sources
            for boundary in source["payload"].get("boundaries") or []
            if isinstance(boundary, dict) and boundary.get("seconds") is not None
        ),
        key=lambda row: (row["seconds"], row["backend"]),
    )
    groups = _cluster(observations, tolerance)
    all_backends = sorted({source["backend"] for source in sources})
    candidates: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        backend_votes = sorted({str(row["backend"]) for row in group})
        seconds = round(sum(float(row["seconds"]) for row in group) / len(group), 6)
        agreement = len(backend_votes) >= 2
        candidates.append(
            {
                "candidate_id": f"fused-boundary-{index:04d}",
                "seconds": seconds,
                "time": format_timestamp(seconds),
                "status": "candidate_agreement" if agreement else "needs_human_review",
                "backend_votes": backend_votes,
                "missing_backend_votes": sorted(set(all_backends) - set(backend_votes)),
                "observations": group,
                "spread_seconds": round(
                    max(float(row["seconds"]) for row in group)
                    - min(float(row["seconds"]) for row in group),
                    6,
                ),
                "candidate_only": True,
                "automatically_selected": False,
            }
        )
    result = {
        "schema": SCHEMA,
        "status": "needs_human_review" if any(
            row["status"] == "needs_human_review" for row in candidates
        ) else "candidate_agreement_complete",
        "ok": bool(candidates),
        "bundle_dir": str(root),
        "boundary_kind": "technical_shot_candidate_fusion",
        "media_sha256": next(iter(media_hashes), ""),
        "frame_rate": fps,
        "tolerance_frames": max(0, int(tolerance_frames)),
        "tolerance_seconds": round(tolerance, 9),
        "source_artifacts": [
            {
                "path": source["path"],
                "sha256": source["sha256"],
                "backend": source["backend"],
                "boundary_count": len(source["payload"].get("boundaries") or []),
            }
            for source in sources
        ],
        "candidate_count": len(candidates),
        "agreement_count": sum(
            row["status"] == "candidate_agreement" for row in candidates
        ),
        "review_count": sum(
            row["status"] == "needs_human_review" for row in candidates
        ),
        "candidates": candidates,
        "candidate_only": True,
        "human_confirmation_required": True,
        "timeline_mutated": False,
        "operator_boundary": {
            "no_automatic_boundary_selection": True,
            "no_detector_fallback": True,
            "no_timeline_or_media_mutation": True,
            "no_network_call": True,
        },
        "artifacts": {"json": OUTPUT_PATH, "markdown": MARKDOWN_PATH},
        "updated_at": now_iso(),
    }
    if write:
        _write_result(root, result)
    return result


def _load_source(root: Path, value: str | Path) -> dict[str, Any]:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"technical-shot candidate must be a JSON object: {path}")
    if (
        payload.get("schema") != TECHNICAL_SHOT_SCHEMA
        or payload.get("boundary_kind") != "technical_shot"
        or payload.get("ok") is not True
    ):
        raise ValueError(f"not a verified technical-shot artifact: {path}")
    backend = str(payload.get("backend") or "").strip()
    if not backend:
        raise ValueError(f"technical-shot artifact has no backend identity: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "backend": backend,
        "payload": payload,
    }


def _cluster(
    observations: list[dict[str, Any]],
    tolerance: float,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in observations:
        if not groups:
            groups.append([row])
            continue
        current = groups[-1]
        centre = sum(float(item["seconds"]) for item in current) / len(current)
        if abs(float(row["seconds"]) - centre) <= tolerance + 1e-9:
            current.append(row)
        else:
            groups.append([row])
    return groups


def _write_result(root: Path, result: dict[str, Any]) -> None:
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    with bundle_write_lock(root, operation="technical_shot_fusion", timeout_seconds=1.0):
        write_json(root / OUTPUT_PATH, result)
        (root / MARKDOWN_PATH).write_text(_render_markdown(result), encoding="utf-8")
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        if isinstance(manifest, dict):
            manifest["technical_shot_boundary_fusion_json"] = OUTPUT_PATH
            manifest["technical_shot_boundary_fusion_markdown"] = MARKDOWN_PATH
            write_json(manifest_path, manifest)


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 技术镜头边界多证据候选",
        "",
        f"- 状态：`{result.get('status')}`",
        f"- 两帧容差：`{result.get('tolerance_seconds')}s`",
        f"- 候选：`{result.get('candidate_count')}`",
        f"- 需人工复核：`{result.get('review_count')}`",
        "- 本报告不会自动选择边界，也不会修改 Timeline 或媒体。",
        "",
    ]
    for row in result.get("candidates") or []:
        lines.extend(
            [
                f"## {row['candidate_id']} · {row['time']}",
                "",
                f"- 状态：`{row['status']}`",
                f"- Backend：`{', '.join(row['backend_votes'])}`",
                f"- 缺失票：`{', '.join(row['missing_backend_votes']) or '无'}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
