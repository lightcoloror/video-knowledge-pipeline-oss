"""Source-checkout round-trip example for the VKP offline fixture tools.

This is intentionally small orchestration of the SAME public helpers, not a
second renderer:

1. ``scripts/build-offline-acceptance-fixtures.py`` builds all three tiny
   first-party-tone MP4 fixtures into ``OUTPUT/fixtures`` using local FFmpeg.
2. This example decodes at least one real frame per generated MP4 with local
   FFmpeg into ``OUTPUT/bundle/frames``.
3. A minimal synthetic ``timeline.json`` references those decoded frames
   relative to the bundle root, and a small ``manifest.json`` records that the
   bundle is synthetic without inventing a schema.
4. ``scripts/export-offline-frame-manifest.py`` exports
   ``OUTPUT/bundle/frame-manifest.json`` through the same exporter.
5. Every reported relative path is resolved from a documented base and its file
   and SHA-256 are re-verified before the report is written.

Every generated file stays under the explicit owned output root. A nonempty
output root is refused rather than overwritten or cleaned. Failures keep partial
evidence and exit nonzero. No sample media, captions, fonts, binaries or model
weights are added to the public tree; no ASR/OCR/model/knowledge quality is
claimed and a successful round-trip is not the full 11-repository workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


FRAME_TIMESTAMP_SECONDS = 0.5
EXTRACT_TIMEOUT_SECONDS = 120
EXTRACT_THREADS = 2


def _load_module(module_name: str, file_name: str) -> ModuleType:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(module_name, scripts_dir / file_name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _ensure_parent_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _is_regular_file(path: Path) -> bool:
    return path.is_file()


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fresh_root(root: Path) -> None:
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"output root exists and is not a directory: {root}")
        if any(root.iterdir()):
            raise ValueError(
                "refusing to write into a nonempty output root; pass a fresh explicit directory owned by this run"
            )
    root.mkdir(parents=True, exist_ok=True)


def _run_process(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
    )


def _frame_command(ffmpeg: str, video: Path, destination: Path) -> list[str]:
    """Decode one local frame with bounded decoder/encoder/filter thread limits.

    The thread values are argv requests, not a measured OS thread/RAM peak. The
    input protocol request is local-file only, not a proof of network blocking.
    """
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-filter_threads",
        str(EXTRACT_THREADS),
        "-ss",
        f"{FRAME_TIMESTAMP_SECONDS:.3f}",
        "-threads",
        str(EXTRACT_THREADS),
        "-protocol_whitelist",
        "file",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-threads",
        str(EXTRACT_THREADS),
        "-f",
        "image2",
        str(destination),
    ]


def _extract_frame(ffmpeg: str, video: Path, destination: Path) -> Path:
    _ensure_parent_dir(destination.parent)
    _run_process(_frame_command(ffmpeg, video, destination), EXTRACT_TIMEOUT_SECONDS)
    if not _is_regular_file(destination) or _file_size(destination) <= 0:
        raise RuntimeError(f"FFmpeg did not decode a nonempty frame: {destination}")
    return destination


def _report_paths(fixture_id: str, video_name: str, frame_name: str) -> dict[str, str]:
    """Relative paths with explicit documented bases.

    ``output_root`` base is the explicit output directory; ``bundle_root`` base
    is ``bundle`` under it. No path is reported without a stated base.
    """
    return {
        "media_path_relative_to_output_root": f"fixtures/{fixture_id}/{video_name}",
        "frame_path_relative_to_output_root": f"bundle/frames/{frame_name}",
        "frame_reference_in_timeline_relative_to_bundle_root": f"frames/{frame_name}",
    }


def _verify_reported_artifacts(root: Path, report: dict[str, Any], frame_manifest: dict[str, Any]) -> None:
    """Resolve every reported reference from its documented base and re-hash it."""
    bundle_root = root / "bundle"
    for relative in (
        report["fixture_contract_relative_to_output_root"],
        report["timeline_relative_to_output_root"],
        report["frame_manifest_relative_to_output_root"],
        report["bundle_manifest_relative_to_output_root"],
    ):
        candidate = root / relative
        if not _is_regular_file(candidate):
            raise RuntimeError(f"reported artifact is missing: {relative}")
    for fixture in report["fixtures"]:
        media = root / fixture["media_path_relative_to_output_root"]
        frame = root / fixture["frame_path_relative_to_output_root"]
        timeline_reference = bundle_root / fixture["frame_reference_in_timeline_relative_to_bundle_root"]
        for candidate, label in (
            (media, "media_path_relative_to_output_root"),
            (frame, "frame_path_relative_to_output_root"),
            (timeline_reference, "frame_reference_in_timeline_relative_to_bundle_root"),
        ):
            if not _is_regular_file(candidate):
                raise RuntimeError(f"reported {label} does not resolve to a regular file: {candidate}")
        media_hash = _sha256(media)
        if media_hash != fixture["media_sha256"]:
            raise RuntimeError(f"reported media hash mismatch: {media}")
        frame_hash = _sha256(frame)
        if frame_hash != fixture["frame_sha256"]:
            raise RuntimeError(f"reported frame hash mismatch: {frame}")
    if frame_manifest.get("frame_count") != len(report["fixtures"]):
        raise RuntimeError("exported frame manifest count does not match the requested fixtures")
    if frame_manifest.get("available_frame_count") != len(report["fixtures"]):
        raise RuntimeError("exported frame manifest has unavailable frames")


def run(output_root: str | Path, *, ffmpeg: str | None = None, ffprobe: str | None = None, tone_seconds: float = 2.0) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    _require_fresh_root(root)

    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required on PATH or must be passed explicitly")

    builder = _load_module("vkp_offline_fixture_builder", "build-offline-acceptance-fixtures.py")
    exporter = _load_module("vkp_offline_frame_manifest", "export-offline-frame-manifest.py")

    fixture_root = root / "fixtures"
    built = builder.run(fixture_root, ffmpeg=ffmpeg, ffprobe=ffprobe, tone_seconds=tone_seconds)
    fixtures = built["manifest"]["fixtures"]

    bundle_dir = root / "bundle"
    frames_dir = bundle_dir / "frames"
    _ensure_parent_dir(frames_dir)

    timeline: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    for index, fixture in enumerate(fixtures):
        video = Path(fixture["media_path"])
        if not _is_regular_file(video) or _file_size(video) <= 0:
            raise RuntimeError(f"fixture MP4 is missing or empty: {video}")
        frame = _extract_frame(ffmpeg, video, frames_dir / f"frame-{index:02d}.png")
        paths = _report_paths(fixture["fixture_id"], video.name, frame.name)
        timeline.append(
            {
                "index": index,
                "start": 0.0,
                "end": min(tone_seconds, float(fixture["duration_seconds"])),
                "source_fixture_id": fixture["fixture_id"],
                "temporal_frame_paths": [paths["frame_reference_in_timeline_relative_to_bundle_root"]],
                "note": "real frame decoded from a generated first-party-tone MP4",
            }
        )
        generated.append(
            {
                "fixture_id": fixture["fixture_id"],
                "fixture_type": fixture["fixture_type"],
                **paths,
                "media_sha256": fixture["sha256"],
                "media_duration_seconds": fixture["duration_seconds"],
                "frame_sha256": _sha256(frame),
                "frame_bytes": _file_size(frame),
            }
        )

    _write_json(bundle_dir / "timeline.json", timeline)
    _write_json(
        bundle_dir / "manifest.json",
        {
            "note": "synthetic bundle written by the round-trip example; no real media, ASR, OCR, review or knowledge output",
            "fixtures": [{"fixture_id": fixture["fixture_id"]} for fixture in fixtures],
            "source_fixture_contract": {
                "relative_to_output_root": "fixtures/fixture-contract.json",
                "inside_bundle": False,
                "note": "the fixture contract is a sibling of the bundle, not a bundle member",
            },
        },
    )

    export_summary = exporter.run_export(bundle_dir)
    if not export_summary["ok"] or export_summary["available_frame_count"] != len(fixtures):
        raise RuntimeError(f"frame manifest export was incomplete: {json.dumps(export_summary, ensure_ascii=False)}")
    frame_manifest = json.loads((bundle_dir / "frame-manifest.json").read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "ok": True,
        "bases": {"output_root": ".", "fixture_root": "fixtures", "bundle_root": "bundle"},
        "output_root": str(root),
        "fixture_count": len(fixtures),
        "decoded_frame_count": len(generated),
        "fixture_contract_relative_to_output_root": "fixtures/fixture-contract.json",
        "timeline_relative_to_output_root": "bundle/timeline.json",
        "bundle_manifest_relative_to_output_root": "bundle/manifest.json",
        "frame_manifest_relative_to_output_root": "bundle/frame-manifest.json",
        "fixtures": generated,
        "verification": {"reported_references_verified": True, "reported_hashes_verified": True},
        "claims": {
            "audio": "first-party generated PCM tone; not speech",
            "speech_or_asr_quality": False,
            "ocr_or_visual_quality": False,
            "human_review_created": False,
            "knowledge_notes_created": False,
            "full_11_repository_workflow": False,
        },
        "boundaries": {
            "local_ffmpeg_only": True,
            "file_protocol_requested_for_media_inputs": True,
            "network_blocking_proven": False,
            "media_thread_limits_are_argv_requests_not_measured_peaks": True,
            "media_threads_per_process": EXTRACT_THREADS,
            "media_timeout_seconds": EXTRACT_TIMEOUT_SECONDS,
            "no_sample_media_committed": True,
        },
    }
    _verify_reported_artifacts(root, report, frame_manifest)
    _write_json(root / "roundtrip-report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline fixture build/decode/export round-trip example")
    parser.add_argument("output_root", help="fresh explicit output root owned by this run")
    parser.add_argument("--ffmpeg", default=None, help="explicit local FFmpeg executable")
    parser.add_argument("--ffprobe", default=None, help="explicit local ffprobe executable")
    parser.add_argument("--tone-seconds", type=float, default=2.0, help="generated tone duration per fixture")
    args = parser.parse_args(argv)
    try:
        report = run(
            args.output_root,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            tone_seconds=args.tone_seconds,
        )
    except Exception as exc:  # preserve the first error message; last partial outputs stay on disk
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "frame_manifest": report["frame_manifest_relative_to_output_root"], "frame_count": report["decoded_frame_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
