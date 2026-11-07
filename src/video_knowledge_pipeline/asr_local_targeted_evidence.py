from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .file_hash import sha256_file as _sha256
from .models import now_iso
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.local_targeted_asr_evidence.v1"


def build_local_targeted_asr_evidence(
    snippet_manifests: Sequence[str | Path],
    raw_outputs: Sequence[str | Path],
    *,
    output_json: str | Path | None = None,
    require_gpu: bool = True,
    write: bool = False,
) -> dict[str, Any]:
    """Combine verified local clip ASR outputs into candidate-only global-time evidence."""

    manifest_paths = [Path(value).expanduser().resolve() for value in snippet_manifests]
    output_paths = [Path(value).expanduser().resolve() for value in raw_outputs]
    if not manifest_paths:
        raise ValueError("at least one snippet manifest is required")
    if not output_paths:
        raise ValueError("at least one raw ASR output is required")

    artifacts = _artifacts(manifest_paths)
    available_by_sha: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        available_by_sha.setdefault(str(artifact["sha256"]), []).append(artifact)

    segments: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    used_artifact_keys: set[str] = set()
    models: set[str] = set()
    devices: set[str] = set()
    providers: set[str] = set()
    for raw_path in output_paths:
        try:
            run, shifted = _verified_run(
                raw_path,
                available_by_sha=available_by_sha,
                used_artifact_keys=used_artifact_keys,
                require_gpu=require_gpu,
            )
            runs.append(run)
            segments.extend(shifted)
            models.add(run["model"])
            devices.add(run["device"])
            providers.add(run["provider"])
        except Exception as exc:
            failures.append(
                {
                    "raw_output_path": str(raw_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    missing = [
        {
            "retry_id": artifact["retry_id"],
            "path": artifact["path"],
            "sha256": artifact["sha256"],
        }
        for artifact in artifacts
        if artifact["artifact_key"] not in used_artifact_keys
    ]
    segments.sort(key=lambda row: (float(row["start"]), float(row["end"]), row["segment_id"]))
    if failures or missing:
        status = "degraded" if segments else "failed"
    else:
        status = "completed"

    destination = Path(output_json).expanduser().resolve() if output_json else None
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "ok": status == "completed",
        "candidate_only": True,
        "provider": sorted(providers),
        "model": sorted(models),
        "device": sorted(devices),
        "snippet_manifests": [
            {"path": str(path), "sha256": _sha256(path)} for path in manifest_paths
        ],
        "raw_outputs": [str(path) for path in output_paths],
        "expected_window_count": len(artifacts),
        "completed_window_count": len(runs),
        "missing_window_count": len(missing),
        "failed_output_count": len(failures),
        "segment_count": len(segments),
        "segments": segments,
        "runs": runs,
        "missing_windows": missing,
        "failed_outputs": failures,
        "text": "\n".join(str(row["text"]) for row in segments),
        "operator_boundary": {
            "local_only": True,
            "gpu_required": bool(require_gpu),
            "no_network": True,
            "no_upload": True,
            "no_provider_fallback": True,
            "no_location_fallback": True,
            "canonical_transcript_modified": False,
            "automatic_promotion_allowed": False,
            "timing_source": "verified_clip_start_plus_local_asr_timestamp",
        },
        "write": bool(write),
        "updated_at": now_iso(),
    }
    if destination is not None:
        result["output_json"] = str(destination)
    if write:
        if destination is None:
            raise ValueError("output_json is required when write=True")
        write_json(destination, result)
        result["output_sha256"] = _sha256(destination)
    return result


def _artifacts(manifest_paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        manifest = _object(read_json(manifest_path), f"snippet manifest {manifest_path}")
        if str(manifest.get("schema") or "") != "video_knowledge_pipeline.asr_retry_snippets.v1":
            raise ValueError(f"unsupported snippet manifest schema: {manifest_path}")
        for item in manifest.get("artifacts") or []:
            if not isinstance(item, dict) or str(item.get("status") or "") != "completed":
                continue
            path = Path(str(item.get("path") or "")).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"snippet artifact not found: {path}")
            expected_sha = str(item.get("sha256") or "")
            actual_sha = _sha256(path)
            if not expected_sha or actual_sha != expected_sha:
                raise ValueError(f"snippet artifact hash mismatch: {path}")
            expected_bytes = int(item.get("bytes") or -1)
            if expected_bytes != path.stat().st_size:
                raise ValueError(f"snippet artifact byte count mismatch: {path}")
            artifact_key = f"{manifest_path}:{item.get('retry_id')}:{actual_sha}"
            if artifact_key in seen:
                raise ValueError(f"duplicate snippet artifact: {artifact_key}")
            seen.add(artifact_key)
            rows.append(
                {
                    "artifact_key": artifact_key,
                    "manifest_path": str(manifest_path),
                    "retry_id": str(item.get("retry_id") or ""),
                    "source_segment_ids": list(item.get("source_segment_ids") or []),
                    "start": float(item.get("start") or 0.0),
                    "end": float(item.get("end") or 0.0),
                    "duration_seconds": float(item.get("duration_seconds") or 0.0),
                    "path": str(path),
                    "sha256": actual_sha,
                    "bytes": path.stat().st_size,
                }
            )
    if not rows:
        raise ValueError("snippet manifests contain no completed artifacts")
    return rows


def _normalise_local_raw_output(raw: dict[str, Any], *, source_path: Path | None = None) -> dict[str, Any]:
    """Normalise supported local runners without inventing timestamps."""
    schema = str(raw.get("schema") or "")
    if schema not in {"video_knowledge_funasr_raw_output.v1", "video_knowledge_funasr_chunked_raw_output.v1"}:
        return raw
    if raw.get("ok") is False or raw.get("status") != "completed" or raw.get("failed_chunk_count"):
        return {**raw,"ok":False,"status":"failed","segments":[]}
    if schema == "video_knowledge_funasr_chunked_raw_output.v1":
        if raw.get("canonical_complete") is not True or raw.get("unresolved_chunk_indexes") or source_path is None:
            return {**raw,"ok":False,"status":"failed","segments":[]}
        from .asr_adapter import read_asr_segment_dicts
        segments = read_asr_segment_dicts(source_path, provider="funasr")
        return {**raw,"input_path":str(raw.get("input") or ""),
                "segments":segments}
    rows = raw.get("result") if isinstance(raw.get("result"), list) else []
    text = "\n".join(
        str(row.get("text") or "") for row in rows if isinstance(row, dict)
    )
    text = re.sub(r"<\s*\|[^>]*>", "", text).strip()
    duration = float(raw.get("duration_seconds") or 0.0)
    if not text or duration <= 0:
        return {**raw, "ok": False, "status": "failed", "segments": []}
    return {
        **raw,
        "ok": True,
        "status": "completed",
        "failed_chunk_count": 0,
        "input_path": str(raw.get("input") or ""),
        "model": str(raw.get("resolved_model") or raw.get("model") or ""),
        "segments": [{
            "segment_id": str((rows[0] or {}).get("key") or "funasr-clip") if rows else "funasr-clip",
            "start": 0.0,
            "end": duration,
            "text": text,
            "timestamp_precision": "audio_window_only",
        }],
    }
def _verified_run(
    raw_path: Path,
    *,
    available_by_sha: dict[str, list[dict[str, Any]]],
    used_artifact_keys: set[str],
    require_gpu: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _normalise_local_raw_output(_object(read_json(raw_path), f"raw ASR output {raw_path}"), source_path=raw_path)
    if not bool(raw.get("ok")) or str(raw.get("status") or "") != "completed":
        raise ValueError("local ASR output is not completed")
    if int(raw.get("failed_chunk_count") or 0) != 0:
        raise ValueError("local ASR output contains failed chunks")
    input_path = Path(str(raw.get("input_path") or "")).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"local ASR input artifact not found: {input_path}")
    input_sha = _sha256(input_path)
    matches = [
        row
        for row in available_by_sha.get(input_sha, [])
        if row["artifact_key"] not in used_artifact_keys
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one unused snippet artifact for input SHA-256 {input_sha}; found {len(matches)}"
        )
    artifact = matches[0]
    device = str(raw.get("device") or "")
    if require_gpu and not device.lower().startswith("cuda"):
        raise ValueError(f"local ASR evidence did not run on CUDA: {device or 'missing'}")
    provider = str(raw.get("provider") or "")
    model = str(raw.get("model") or "")
    if not provider or not model:
        raise ValueError("local ASR output is missing provider or model identity")

    shifted: list[dict[str, Any]] = []
    duration = float(artifact["duration_seconds"])
    for position, item in enumerate(raw.get("segments") or [], start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        local_start = float(item.get("start") or 0.0)
        local_end = float(item.get("end") or local_start)
        if local_start < -0.001 or local_end < local_start or local_end > duration + 1.0:
            raise ValueError(
                f"local ASR segment is outside the authorized clip window: {local_start}-{local_end}/{duration}"
            )
        segment_id = str(item.get("segment_id") or f"segment-{position:04d}")
        shifted.append(
            {
                "segment_id": f"{artifact['retry_id']}::{segment_id}",
                "start": round(float(artifact["start"]) + local_start, 6),
                "end": round(float(artifact["start"]) + local_end, 6),
                "text": text,
                "source_segment_ids": list(artifact["source_segment_ids"]),
                "candidate_only": True,
                "timing_source": "verified_clip_start_plus_local_asr_timestamp",
                "timestamp_precision": str(item.get("timestamp_precision") or "provider_timed_or_unknown"),
                "local_timing": {"start": local_start, "end": local_end},
                "retry_id": artifact["retry_id"],
                "artifact_path": artifact["path"],
                "artifact_sha256": artifact["sha256"],
                "raw_output_path": str(raw_path),
                "raw_output_sha256": _sha256(raw_path),
                "provider": provider,
                "model": model,
                "device": device,
            }
        )
    if not shifted:
        raise ValueError("local ASR output contains no usable segments")
    used_artifact_keys.add(artifact["artifact_key"])
    return (
        {
            "retry_id": artifact["retry_id"],
            "source_segment_ids": list(artifact["source_segment_ids"]),
            "window_start": artifact["start"],
            "window_end": artifact["end"],
            "artifact_path": artifact["path"],
            "artifact_sha256": artifact["sha256"],
            "raw_input_path": str(input_path),
            "raw_input_sha256": input_sha,
            "raw_output_path": str(raw_path),
            "raw_output_sha256": _sha256(raw_path),
            "provider": provider,
            "model": model,
            "device": device,
            "segment_count": len(shifted),
            "status": "completed",
        },
        shifted,
    )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build candidate-only global-time ASR evidence from verified local clip runs"
    )
    parser.add_argument("output_json")
    parser.add_argument("--snippet-manifest", action="append", required=True)
    parser.add_argument("--raw-output", action="append", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_local_targeted_asr_evidence(
        args.snippet_manifest,
        args.raw_output,
        output_json=args.output_json,
        require_gpu=not args.allow_cpu,
        write=args.write,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
