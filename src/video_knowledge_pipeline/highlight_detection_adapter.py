from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .models import now_iso
from .path_defaults import source_reviews_root
from .storage import bundle_write_lock, read_json, write_json
from .transcript import format_timestamp
from .video import probe_video, sha256_file


SCHEMA = "video_knowledge_pipeline.highlight_detection.v1"
LIGHTHOUSE_COMMIT = "629bc6790c66ff2a682f0dbb3e8ab2c0c8ff814f"
DEFAULT_SOURCE_ROOT = source_reviews_root() / "vkp-pullfilm-v2-20260801" / "lighthouse"
MODEL = "cg_detr"
IMPLEMENTATION = "lighthouse_cg_detr"
MAX_DIRECT_VIDEO_SECONDS = 150.0


def highlight_detection_status(
    *,
    source_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    source = _source_root(source_root)
    checkpoint = _optional_file(checkpoint_path or os.environ.get("VKP_LIGHTHOUSE_CGDETR_CHECKPOINT"))
    blockers: list[str] = []
    source_commit = _git_commit(source) if source is not None else ""
    if source is None:
        blockers.append("lighthouse_source_missing")
    elif source_commit != LIGHTHOUSE_COMMIT:
        blockers.append("lighthouse_source_commit_mismatch")
    if checkpoint is None:
        blockers.append("cg_detr_checkpoint_missing")
    return {
        "schema": "video_knowledge_pipeline.highlight_detection_status.v1",
        "status": "ready" if not blockers else "needs_setup",
        "implementation": IMPLEMENTATION,
        "model": MODEL,
        "feature_name": "clip",
        "source_root": str(source or ""),
        "source_commit": source_commit,
        "expected_source_commit": LIGHTHOUSE_COMMIT,
        "source_verified": source_commit == LIGHTHOUSE_COMMIT,
        "checkpoint_path": str(checkpoint or ""),
        "blockers": blockers,
        "limits": {
            "max_direct_video_seconds": MAX_DIRECT_VIDEO_SECONDS,
            "long_video_policy": "split_into_local_scene_clips_first",
        },
        "operator_boundary": {
            "local_only": True,
            "automatic_model_download": False,
            "automatic_remote_fallback": False,
            "native_whole_video_understanding": False,
            "gpu_required": True,
            "cpu_fallback_allowed": False,
        },
        "updated_at": now_iso(),
    }


def run_highlight_detection(
    bundle_dir: str | Path,
    *,
    query: str,
    media_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    source_root: str | Path | None = None,
    predictions_json: str | Path | None = None,
    feature_name: str = "clip",
    device: str = "cuda",
    execute: bool = False,
    write: bool = True,
    _inference_backend: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Plan, execute, or import local query-dependent highlight detection.

    The adapter intentionally requires a caller-supplied checkpoint and never
    downloads weights or falls back to a remote provider.
    """

    root = Path(bundle_dir).expanduser().resolve()
    prompt = str(query or "").strip()
    if not prompt:
        raise ValueError("highlight query is required")
    if feature_name not in {"clip", "clip_slowfast", "clip_slowfast_pann"}:
        raise ValueError("feature_name must be clip, clip_slowfast, or clip_slowfast_pann")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")

    setup = highlight_detection_status(source_root=source_root, checkpoint_path=checkpoint_path)
    prediction_source = _optional_file(predictions_json)
    media = _resolve_media(root, media_path)
    checkpoint = _optional_file(checkpoint_path or setup.get("checkpoint_path"))
    source = _source_root(source_root)
    source_commit = _git_commit(source) if source is not None else ""
    raw_prediction: dict[str, Any] = {}
    status = "planned"
    execution_mode = "preview"
    error: dict[str, Any] | None = None

    if prediction_source is not None:
        payload = read_json(prediction_source)
        if not isinstance(payload, dict):
            raise ValueError("highlight predictions JSON must be an object")
        raw_prediction = payload
        status = "completed_import"
        execution_mode = "saved_prediction_import"
    elif execute:
        execution_mode = "local_model"
        blockers = list(setup.get("blockers") or [])
        if media is None:
            blockers.append("media_path_missing")
        duration = _duration_seconds(root, media)
        if duration > MAX_DIRECT_VIDEO_SECONDS:
            blockers.append("scene_clip_required_for_long_video")
        if blockers and _inference_backend is None:
            status = "blocked_setup"
            error = {"code": "highlight_runtime_not_ready", "blockers": sorted(set(blockers))}
        else:
            assert media is not None
            try:
                raw_prediction = (
                    _inference_backend(media)
                    if _inference_backend is not None
                    else _execute_lighthouse(
                        media,
                        query=prompt,
                        checkpoint=checkpoint,
                        source_root=source,
                        feature_name=feature_name,
                        device=device,
                    )
                )
                status = "completed"
            except Exception as exc:  # local optional runtime errors are evidence, not fallback triggers
                status = "failed"
                error = {"code": "local_highlight_execution_failed", "message": f"{type(exc).__name__}: {exc}"}

    highlights = _normalise_prediction(raw_prediction, query=prompt)
    result = {
        "schema": SCHEMA,
        "status": status,
        "ok": status in {"completed", "completed_import"},
        "bundle_dir": str(root),
        "query": prompt,
        "implementation": IMPLEMENTATION,
        "model": MODEL,
        "feature_name": feature_name,
        "execution_mode": execution_mode,
        "media_path": str(media or ""),
        "source_root": str(source or ""),
        "source_commit": source_commit,
        "expected_source_commit": LIGHTHOUSE_COMMIT,
        "source_verified": source_commit == LIGHTHOUSE_COMMIT,
        "checkpoint_path": str(checkpoint or ""),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint is not None else "",
        "predictions_json": str(prediction_source or ""),
        "highlight_count": len(highlights),
        "highlights": highlights,
        "saliency_scores": _numbers(raw_prediction.get("pred_saliency_scores")),
        "error": error,
        "next_actions": _next_actions(status),
        "operator_boundary": {
            "candidate_only": True,
            "no_remote_call": True,
            "remote_calls_made": 0,
            "automatic_model_download": False,
            "automatic_remote_fallback": False,
            "native_whole_video_understanding": False,
            "long_video_requires_local_scene_clips": True,
            "gpu_required": True,
            "cpu_fallback_allowed": False,
        },
        "artifacts": {
            "json": "exports/highlight-detection.json",
            "markdown": "exports/highlight-detection.md",
        },
        "updated_at": now_iso(),
    }
    if write:
        _write_result(root, result)
    return result


def _execute_lighthouse(
    media: Path,
    *,
    query: str,
    checkpoint: Path | None,
    source_root: Path | None,
    feature_name: str,
    device: str,
) -> dict[str, Any]:
    if checkpoint is None:
        raise FileNotFoundError("CG-DETR checkpoint is required")
    if source_root is None:
        raise FileNotFoundError("pinned Lighthouse source is required")
    actual_commit = _git_commit(source_root)
    if actual_commit != LIGHTHOUSE_COMMIT:
        raise RuntimeError(
            "Lighthouse source commit mismatch: "
            f"expected={LIGHTHOUSE_COMMIT} actual={actual_commit}"
        )
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import torch
    from lighthouse.models import CGDETRPredictor

    if device == "cpu":
        raise RuntimeError("gpu_required_for_local_model")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable_gpu_required")
    resolved_device = "cuda"
    predictor = CGDETRPredictor(str(checkpoint), device=resolved_device, feature_name=feature_name)
    encoded = predictor.encode_video(str(media))
    prediction = predictor.predict(query, encoded)
    if not isinstance(prediction, dict):
        raise ValueError("Lighthouse prediction must be a JSON object")
    return prediction


def _normalise_prediction(payload: dict[str, Any], *, query: str) -> list[dict[str, Any]]:
    windows = payload.get("pred_relevant_windows") or payload.get("highlights") or []
    result: list[dict[str, Any]] = []
    for index, row in enumerate(windows, start=1):
        if isinstance(row, dict):
            start = _float(row.get("start"))
            end = _float(row.get("end"))
            score = _float(row.get("score"))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            start = _float(row[0])
            end = _float(row[1])
            score = _float(row[2]) if len(row) >= 3 else 0.0
        else:
            continue
        if end <= start:
            continue
        result.append(
            {
                "index": index,
                "start": round(max(0.0, start), 3),
                "end": round(max(0.0, end), 3),
                "start_time": format_timestamp(max(0.0, start)),
                "end_time": format_timestamp(max(0.0, end)),
                "score": round(score, 6),
                "query": query,
                "source": IMPLEMENTATION,
                "candidate_only": True,
                "human_review_required": True,
            }
        )
    return result


def _write_result(root: Path, result: dict[str, Any]) -> None:
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    with bundle_write_lock(root, operation="highlight_detection", timeout_seconds=1.0):
        write_json(exports / "highlight-detection.json", result)
        (exports / "highlight-detection.md").write_text(_render_markdown(result), encoding="utf-8")
        write_json(
            root / "mcp-highlight-detection.args.json",
            {
                "bundle_dir": str(root),
                "query": result["query"],
                "media_path": result["media_path"],
                "checkpoint_path": result["checkpoint_path"],
                "source_root": result["source_root"],
                "feature_name": result["feature_name"],
                "execute": False,
                "write": True,
            },
        )
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if isinstance(manifest, dict):
            manifest["highlight_detection_json"] = "exports/highlight-detection.json"
            manifest["highlight_detection_markdown"] = "exports/highlight-detection.md"
            manifest["mcp_highlight_detection_args"] = "mcp-highlight-detection.args.json"
            manifest["highlight_detection_summary"] = {
                "status": result["status"],
                "model": result["model"],
                "highlight_count": result["highlight_count"],
                "updated_at": result["updated_at"],
            }
            write_json(manifest_path, manifest)


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Highlight Detection",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Model: `{result.get('model')}`",
        f"- Query: {result.get('query')}",
        f"- Highlights: `{result.get('highlight_count')}`",
        "- Boundary: local specialized model or saved-result import; no remote fallback; candidate evidence only.",
        "",
        "| # | Time | Score | Query |",
        "- Device policy: CUDA GPU required; no automatic CPU fallback.",
        "| ---: | --- | ---: | --- |",
    ]
    for row in result.get("highlights") or []:
        lines.append(
            f"| {row.get('index')} | `{row.get('start_time')} - {row.get('end_time')}` | {row.get('score')} | {_md(row.get('query'))} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _resolve_media(root: Path, value: str | Path | None) -> Path | None:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    candidate = value
    if not candidate and isinstance(manifest, dict):
        candidate = manifest.get("media_path") or manifest.get("source_path")
    return _optional_file(candidate, relative_to=root)


def _git_commit(root: Path | None) -> str:
    if root is None:
        return ""
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _source_root(value: str | Path | None) -> Path | None:
    candidate = value or os.environ.get("VKP_LIGHTHOUSE_SOURCE")
    if candidate:
        path = Path(candidate).expanduser()
        if path.exists() and path.is_dir():
            return path.resolve()
    return DEFAULT_SOURCE_ROOT.resolve() if DEFAULT_SOURCE_ROOT.exists() else None


def _optional_file(value: str | Path | None, *, relative_to: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return path.resolve() if path.exists() and path.is_file() else None


def _duration_seconds(root: Path, media: Path | None) -> float:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if isinstance(manifest, dict):
        for key in ("duration_seconds", "media_duration_seconds", "video_duration_seconds"):
            value = _float(manifest.get(key))
            if value > 0:
                return value
    if media is None:
        return 0.0
    try:
        return float(probe_video(media).duration_seconds)
    except Exception:
        return 0.0


def _numbers(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [round(_float(item), 6) for item in value if isinstance(item, (int, float))]


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _next_actions(status: str) -> list[str]:
    if status == "planned":
        return ["Provide an exact local CG-DETR checkpoint and run again with --execute, or import a saved prediction JSON."]
    if status == "blocked_setup":
        return ["Install Lighthouse in an operator-managed environment and configure an exact checkpoint; no model download is automatic."]
    if status == "failed":
        return ["Inspect the local execution error and retry the same model explicitly; no fallback was attempted."]
    return ["Review candidate highlight windows before they are used by editing or summary workflows."]


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
