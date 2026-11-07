from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .asr_chunk_batch_merge import SCHEMA as CHUNK_MERGE_SCHEMA
from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .models import now_iso
from .storage import read_json, write_json
from .file_hash import sha256_file as _sha256


SCHEMA = "video_knowledge_pipeline.asr_retry_snippets.v1"


def prepare_asr_retry_snippets(
    media_path: str | Path,
    quality_report: dict[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    ffmpeg_path: str | Path = "ffmpeg",
    execute: bool = False,
) -> dict[str, Any]:
    """Plan or extract only the ASR ranges selected by the quality gate."""

    media = Path(media_path).expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(f"media not found: {media}")
    report = _quality_report(quality_report)
    media_sha256 = _sha256(media)
    source_media_verified = _validate_source_media(report, media, media_sha256)
    quality = _quality_payload(report)
    retry_plan = (
        quality.get("retry_plan")
        if isinstance(quality.get("retry_plan"), dict)
        else {}
    )
    windows = [dict(row) for row in retry_plan.get("windows") or [] if isinstance(row, dict)]
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved_ffmpeg = _resolve_ffmpeg(ffmpeg_path)

    artifacts: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for position, window in enumerate(windows, start=1):
        start = max(0.0, _number(window.get("start"), 0.0))
        end = max(start, _number(window.get("end"), start))
        retry_id = str(window.get("retry_id") or f"retry-{position:04d}")
        filename = f"{retry_id}-{round(start * 1000):010d}-{round(end * 1000):010d}.wav"
        destination = root / filename
        command = [
            resolved_ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{max(0.0, end - start):.6f}",
            "-i",
            str(media),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        item = {
            "retry_id": retry_id,
            "source_segment_ids": list(window.get("source_segment_ids") or []),
            "start": start,
            "end": end,
            "duration_seconds": round(end - start, 6),
            "alignment_source": str(window.get("alignment_source") or "provider_segment_boundary"),
            "reasons": list(window.get("reasons") or []),
            "path": str(destination),
            "command": command,
            "status": "planned",
            "sha256": "",
            "bytes": 0,
        }
        if execute:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=local_tool_subprocess_env(),
                )
                if completed.returncode != 0 or not destination.is_file():
                    raise RuntimeError((completed.stderr or completed.stdout or "ffmpeg produced no artifact").strip())
                item.update(
                    {
                        "status": "completed",
                        "sha256": _sha256(destination),
                        "bytes": destination.stat().st_size,
                    }
                )
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = f"{type(exc).__name__}: {exc}"
                failed.append(
                    {
                        "retry_id": retry_id,
                        "path": str(destination),
                        "reason": item["error"],
                        "retry_command": command,
                    }
                )
        artifacts.append(item)

    completed_count = sum(row["status"] == "completed" for row in artifacts)
    if not windows:
        status = "completed"
    elif not execute:
        status = "planned"
    elif failed and completed_count:
        status = "degraded"
    elif failed:
        status = "failed"
    else:
        status = "completed"
    result = {
        "schema": SCHEMA,
        "status": status,
        "execute": bool(execute),
        "media_path": str(media),
        "media_sha256": media_sha256,
        "source_media_verified": source_media_verified,
        "quality_report_schema": str(report.get("schema") or ""),
        "quality_schema": str(quality.get("schema") or ""),
        "output_dir": str(root),
        "window_count": len(windows),
        "completed_count": completed_count,
        "failed_count": len(failed),
        "artifacts": artifacts,
        "failed_chunks": failed,
        "remote_retry_policy": {
            "requires_new_exact_consent": bool(windows),
            "consent_artifacts_must_match_completed_sha256": True,
            "silent_provider_fallback_allowed": False,
            "silent_location_fallback_allowed": False,
            "remote_execution_performed": False,
        },
        "created_at": now_iso(),
    }
    write_json(root / "asr-retry-snippets.json", result)
    return result


def _quality_report(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    payload = read_json(Path(value).expanduser().resolve())
    if not isinstance(payload, dict):
        raise ValueError("ASR quality report must be a JSON object")
    return payload


def _quality_payload(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema") != CHUNK_MERGE_SCHEMA:
        return report
    quality = report.get("asr_quality")
    if not isinstance(quality, dict):
        raise ValueError("ASR chunk merge report is missing asr_quality")
    return quality


def _validate_source_media(
    report: dict[str, Any], media: Path, media_sha256: str
) -> bool:
    if report.get("schema") != CHUNK_MERGE_SCHEMA:
        return False
    source = report.get("source_media")
    if not isinstance(source, dict) or not source:
        raise ValueError("ASR chunk merge report is missing source_media identity")
    if int(source.get("bytes") or -1) != media.stat().st_size:
        raise ValueError("retry media byte count does not match ASR chunk source")
    if str(source.get("sha256") or "") != media_sha256:
        raise ValueError("retry media SHA-256 does not match ASR chunk source")
    return True


def _resolve_ffmpeg(value: str | Path) -> str:
    configured = str(value or "ffmpeg").strip() or "ffmpeg"
    if configured.casefold() != "ffmpeg":
        return configured
    return str(resolve_media_tool("ffmpeg") or "ffmpeg")


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or extract exact ASR retry snippets")
    parser.add_argument("media_path")
    parser.add_argument("quality_report_json")
    parser.add_argument("output_dir")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_asr_retry_snippets(
        args.media_path,
        args.quality_report_json,
        args.output_dir,
        ffmpeg_path=args.ffmpeg,
        execute=args.execute,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"completed", "planned"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
