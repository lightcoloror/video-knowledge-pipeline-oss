from __future__ import annotations

import json
import os
import shutil
import time
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import now_iso
from .storage import replace_file_with_retry, write_json, write_text_atomic

STAGE_CACHE_SCHEMA = "video_knowledge_pipeline.stage_cache.v1"
STAGE_CACHE_VERSION = 1


class StageCache:
    """vsummary-style local stage cache for expensive VKP generation steps.

    Cache validity is tied to source path, source size/mtime, stage name, and an
    implementation identity. It is intentionally small and dependency-free so it
    can be reused by ASR, OCR, ebook, smart-summary, and content export jobs.
    """

    def __init__(self, cache_dir: str | Path, source_path: str | Path) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.source_path = Path(source_path).expanduser().resolve()

    def source_fingerprint(self) -> str:
        stat = self.source_path.stat()
        digest = sha256()
        digest.update(str(self.source_path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return digest.hexdigest()

    def is_valid(self, stage: str, *, identity: str) -> bool:
        manifest = self._read_manifest(stage)
        return manifest == self._manifest_payload(stage, identity=identity)

    def load_json(self, stage: str, name: str, *, identity: str) -> dict[str, Any] | list[Any] | None:
        if not self.is_valid(stage, identity=identity):
            return None
        path = self.stage_path(stage, name)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def store_json(self, stage: str, name: str, payload: object, *, identity: str) -> Path:
        path = self.stage_path(stage, name)
        write_json(path, payload)
        self.write_manifest(stage, identity=identity)
        return path

    def load_text(self, stage: str, name: str, *, identity: str) -> str | None:
        if not self.is_valid(stage, identity=identity):
            return None
        path = self.stage_path(stage, name)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def store_text(self, stage: str, name: str, text: str, *, identity: str) -> Path:
        path = self.stage_path(stage, name)
        atomic_write_text(path, text)
        self.write_manifest(stage, identity=identity)
        return path

    def restore_file(self, stage: str, name: str, target_path: str | Path, *, identity: str) -> bool:
        if not self.is_valid(stage, identity=identity):
            return False
        source = self.stage_path(stage, name)
        if not source.exists():
            return False
        atomic_copy_file(source, Path(target_path).expanduser().resolve())
        return True

    def store_file(self, stage: str, name: str, source_path: str | Path, *, identity: str) -> Path:
        target = self.stage_path(stage, name)
        atomic_copy_file(Path(source_path).expanduser().resolve(), target)
        self.write_manifest(stage, identity=identity)
        return target

    def stage_path(self, stage: str, name: str) -> Path:
        return self.cache_dir / _safe_stage(stage) / name

    def manifest_path(self, stage: str) -> Path:
        return self.cache_dir / _safe_stage(stage) / "manifest.json"

    def write_manifest(self, stage: str, *, identity: str) -> Path:
        path = self.manifest_path(stage)
        write_json(path, self._manifest_payload(stage, identity=identity))
        return path

    def status(self, stage: str, *, identity: str) -> dict[str, Any]:
        manifest = self._read_manifest(stage)
        path = self.manifest_path(stage)
        return {
            "schema": STAGE_CACHE_SCHEMA,
            "stage": stage,
            "identity": identity,
            "cache_dir": str(self.cache_dir),
            "source_path": str(self.source_path),
            "manifest_path": str(path),
            "manifest_exists": path.exists(),
            "valid": self.is_valid(stage, identity=identity),
            "manifest": manifest,
            "expected": self._manifest_payload(stage, identity=identity),
        }

    def _manifest_payload(self, stage: str, *, identity: str) -> dict[str, Any]:
        return {
            "schema": STAGE_CACHE_SCHEMA,
            "version": STAGE_CACHE_VERSION,
            "stage": stage,
            "identity": identity,
            "source_fingerprint": self.source_fingerprint(),
            "source_path": str(self.source_path),
        }

    def _read_manifest(self, stage: str) -> dict[str, Any]:
        path = self.manifest_path(stage)
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}


def atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path).expanduser().resolve()
    write_text_atomic(target, text)
    return target


def atomic_copy_file(source_path: str | Path, target_path: str | Path) -> Path:
    source = Path(source_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.{uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temp)
        replace_file_with_retry(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return target


def build_stage_cache_report(cache_dir: str | Path, source_path: str | Path, *, stages: list[str] | None = None, identity: str = "") -> dict[str, Any]:
    cache = StageCache(cache_dir, source_path)
    stage_names = stages or []
    return {
        "schema": "video_knowledge_pipeline.stage_cache_report.v1",
        "created_at": now_iso(),
        "cache_dir": str(cache.cache_dir),
        "source_path": str(cache.source_path),
        "source_fingerprint": cache.source_fingerprint() if cache.source_path.exists() else "",
        "stages": [cache.status(stage, identity=identity) for stage in stage_names],
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "no_process_started": True,
        },
    }


def _safe_stage(stage: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(stage).strip())
    return value or "stage"
