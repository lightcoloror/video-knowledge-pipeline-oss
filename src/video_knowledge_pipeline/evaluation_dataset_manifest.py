from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .storage import write_json
from .time_utils import utc_now_iso_seconds

SCHEMA = "video_knowledge_pipeline.evaluation_dataset_manifest.v1"
_IGNORED_PARTS = {".git", "__pycache__", "._____temp"}


def _visible_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not any(part in _IGNORED_PARTS for part in path.parts)
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def _file_rows(
    files: Iterable[Path],
    *,
    dataset_root: Path,
    include_sha256: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        stat = path.stat()
        row: dict[str, Any] = {
            "path": path.relative_to(dataset_root).as_posix(),
            "bytes": stat.st_size,
        }
        if include_sha256:
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return rows


def _aggregate(files: Iterable[Path]) -> dict[str, int]:
    paths = list(files)
    return {
        "file_count": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
    }


def _clipshots_status(dataset_root: Path, *, include_sha256: bool) -> dict[str, Any]:
    archive_root = dataset_root / "archives" / "ClipShots" / "ClipShots"
    raw_root = dataset_root / "raw" / "ClipShots"
    pieces = [archive_root / name for name in ("ClipShots-a", "ClipShots-b", "ClipShots-c")]
    present_pieces = [path for path in pieces if path.is_file()]
    videos = _visible_files(raw_root / "videos")
    media_videos = [path for path in videos if path.suffix.lower() not in {".txt", ".md"}]
    annotations = _visible_files(raw_root / "annotations")
    complete_pieces = len(present_pieces) == len(pieces)
    extracted = bool(media_videos)
    if complete_pieces and extracted:
        status = "benchmark_ready"
    elif complete_pieces:
        status = "archive_complete_not_extracted"
    else:
        status = "incomplete_archive"
    issues = []
    if not complete_pieces:
        issues.append("missing_split_archive_piece")
    if not extracted:
        issues.append("video_media_not_extracted")
    return {
        "dataset_id": "clipshots",
        "status": status,
        "archive_format": "split_gzip_tar",
        "archive_parts_expected": [path.name for path in pieces],
        "archive_parts": _file_rows(
            present_pieces,
            dataset_root=dataset_root,
            include_sha256=include_sha256,
        ),
        "annotations": _aggregate(annotations),
        "extracted_video_media": _aggregate(media_videos),
        "issues": issues,
    }


def _autoshot_status(dataset_root: Path, *, include_sha256: bool) -> dict[str, Any]:
    archive_root = dataset_root / "archives" / "AutoShot" / "AutoShot"
    raw_root = dataset_root / "raw" / "AutoShot"
    subset_root = dataset_root / "subsets" / "AutoShot-GT-present-167"
    archives = _visible_files(archive_root)
    raw_files = _visible_files(raw_root)
    subset_files = _visible_files(subset_root)
    pilot_manifest = subset_root / "pilot-32-manifest.json"
    inference_report = subset_root / "autoshot-gpu-inference-pilot-32.json"
    status = (
        "benchmark_completed"
        if pilot_manifest.is_file() and inference_report.is_file()
        else "downloaded_not_benchmarked"
        if archives
        else "not_downloaded"
    )
    issues = []
    if not archives:
        issues.append("archives_missing")
    if not pilot_manifest.is_file():
        issues.append("pilot_manifest_missing")
    if not inference_report.is_file():
        issues.append("gpu_inference_report_missing")
    return {
        "dataset_id": "autoshot",
        "status": status,
        "archive_files": _file_rows(
            archives,
            dataset_root=dataset_root,
            include_sha256=include_sha256,
        ),
        "raw_repository": _aggregate(raw_files),
        "benchmark_subset": _aggregate(subset_files),
        "pilot_manifest": (
            _file_rows(
                [pilot_manifest],
                dataset_root=dataset_root,
                include_sha256=include_sha256,
            )[0]
            if pilot_manifest.is_file()
            else None
        ),
        "gpu_inference_report": (
            _file_rows(
                [inference_report],
                dataset_root=dataset_root,
                include_sha256=include_sha256,
            )[0]
            if inference_report.is_file()
            else None
        ),
        "issues": issues,
    }


def _aishell_status(dataset_root: Path, *, include_sha256: bool) -> dict[str, Any]:
    subset_root = dataset_root / "subsets" / "AISHELL1-ModelScope-subset"
    files = _visible_files(subset_root)
    audio_extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    audio = [path for path in files if path.suffix.lower() in audio_extensions]
    metadata = [path for path in files if path not in audio]
    status = "benchmark_ready" if audio else "metadata_only"
    issues = [] if audio else ["audio_payload_missing"]
    return {
        "dataset_id": "aishell1_modelscope_subset",
        "status": status,
        "metadata_files": _file_rows(
            metadata,
            dataset_root=dataset_root,
            include_sha256=include_sha256,
        ),
        "audio": _aggregate(audio),
        "issues": issues,
    }


def build_evaluation_dataset_manifest(
    dataset_root: str | Path,
    *,
    include_sha256: bool = True,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    datasets = [
        _clipshots_status(root, include_sha256=include_sha256),
        _autoshot_status(root, include_sha256=include_sha256),
        _aishell_status(root, include_sha256=include_sha256),
    ]
    identity = {
        "schema": SCHEMA,
        "dataset_root": str(root),
        "hash_mode": "sha256" if include_sha256 else "size_only",
        "datasets": datasets,
    }
    return {
        **identity,
        "generated_at": utc_now_iso_seconds(),
        "manifest_sha256": canonical_json_sha256(identity),
        "summary": {
            "dataset_count": len(datasets),
            "benchmark_ready_or_completed": sum(
                item["status"] in {"benchmark_ready", "benchmark_completed"}
                for item in datasets
            ),
            "incomplete": sum(bool(item.get("issues")) for item in datasets),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory VKP evaluation datasets without downloading or extracting them."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("--output")
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Record sizes only. The default hashes every archive and metadata artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_evaluation_dataset_manifest(
        args.dataset_root,
        include_sha256=not args.no_hash,
    )
    if args.output:
        write_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
