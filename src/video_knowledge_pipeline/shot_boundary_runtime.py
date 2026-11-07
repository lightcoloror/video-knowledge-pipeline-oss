from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .media_tools import local_tool_subprocess_env
from .path_defaults import dataset_root, source_reviews_root
from .video import probe_video


AUTOSHOT_COMMIT = "77c82ff826a9301bb173d9be786297a49d73d081"
OMNISHOTCUT_COMMIT = "23ad6fb41b296fb9258b0e7825125a914573b906"
DEFAULT_AUTOSHOT_ROOT = source_reviews_root() / "shot-breakdown-wave-20260721" / "autoshot"
DEFAULT_OMNISHOTCUT_ROOT = source_reviews_root() / "vkp-pullfilm-v2-20260801" / "omnishotcut"
DEFAULT_AUTOSHOT_CHECKPOINT = dataset_root() / "archives" / "AutoShot" / "AutoShot" / "ckpt_0_200_0.pth"


def run_shot_boundary_runtime(
    *,
    backend: str,
    media_path: str | Path,
    source_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    frame_rate: float | None = None,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Execute one pinned upstream detector in an isolated GPU subprocess."""

    backend_key = str(backend or "").strip().lower()
    if backend_key not in {"autoshot", "omnishotcut"}:
        raise ValueError("runtime backend must be autoshot or omnishotcut")
    media = Path(media_path).expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(media)

    if backend_key == "autoshot":
        source = _existing_source(source_root, DEFAULT_AUTOSHOT_ROOT)
        checkpoint = _existing_checkpoint(
            checkpoint_path,
            DEFAULT_AUTOSHOT_CHECKPOINT,
            backend=backend_key,
        )
        expected_commit = AUTOSHOT_COMMIT
        threshold = 0.296
        max_duration_seconds = 3600.0
        max_decoded_frames = 108000
    else:
        source = _existing_source(source_root, DEFAULT_OMNISHOTCUT_ROOT)
        checkpoint = _existing_checkpoint(
            checkpoint_path,
            None,
            backend=backend_key,
        )
        expected_commit = OMNISHOTCUT_COMMIT
        threshold = 0.0
        max_duration_seconds = 150.0
        max_decoded_frames = 9000

    actual_commit = _git_commit(source)
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"{backend_key} source commit mismatch: "
            f"expected={expected_commit} actual={actual_commit}"
        )

    metadata = probe_video(media)
    fps = float(frame_rate or metadata.fps or 0.0)
    if fps <= 0:
        raise ValueError("video frame rate must be available")
    if metadata.duration_seconds > max_duration_seconds:
        raise ValueError(
            f"{backend_key} runtime accepts at most {max_duration_seconds:g}s; "
            "use a bounded disagreement clip or saved predictions"
        )

    command = [
        sys.executable,
        "-m",
        "video_knowledge_pipeline.shot_boundary_worker",
        "--backend",
        backend_key,
        "--media-path",
        str(media),
        "--source-root",
        str(source),
        "--checkpoint-path",
        str(checkpoint),
        "--frame-rate",
        str(fps),
        "--threshold",
        str(threshold),
        "--max-decoded-frames",
        str(max_decoded_frames),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=local_tool_subprocess_env(),
        timeout=max(1.0, float(timeout_seconds)),
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{backend_key} worker failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    payload = _last_json_object(completed.stdout)
    payload.update(
        {
            "backend": backend_key,
            "source_root": str(source),
            "source_commit": actual_commit,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "media_sha256": metadata.sha256,
            "duration_seconds": metadata.duration_seconds,
            "gpu_process_isolated": True,
            "automatic_fallback": False,
            "network_calls": 0,
        }
    )
    return payload


def _existing_source(value: str | Path | None, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"pinned source root not found: {path}")
    return path


def _existing_checkpoint(
    value: str | Path | None,
    default: Path | None,
    *,
    backend: str,
) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
    elif default is not None:
        path = default.resolve()
    else:
        raise FileNotFoundError(
            f"{backend} checkpoint is required; automatic download is disabled"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"{backend} checkpoint not found: {path}; automatic download is disabled; "
            "pass --checkpoint-path or configure VKP_DATASET_ROOT"
        )
    return path


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"cannot verify source commit: {root}"
        )
    return completed.stdout.strip()


def _last_json_object(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("shot-boundary worker returned no JSON object")
