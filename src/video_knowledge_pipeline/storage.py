from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import ResearchProject, dataclass_to_dict, project_paths

_LOCK_DEPTHS: dict[tuple[Path, int], int] = {}


def ensure_project_dirs(root: str | Path) -> dict[str, Path]:
    paths = project_paths(root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["videos"].mkdir(parents=True, exist_ok=True)
    paths["transcripts"].mkdir(parents=True, exist_ok=True)
    paths["notes"].mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    write_text_atomic(path, payload)


def write_text_atomic(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
    """Replace a text artifact atomically so inherited target ACLs cannot block truncation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp_path.write_text(payload, encoding=encoding)
        replace_file_with_retry(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def replace_file_with_retry(source: Path, target: Path, *, attempts: int = 8) -> None:
    total_attempts = max(1, int(attempts))
    for attempt in range(total_attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= total_attempts:
                raise
            time.sleep(0.025 * (attempt + 1))


_replace_file_with_retry = replace_file_with_retry

def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_object_or_empty(path: str | Path) -> dict[str, Any]:
    """Reuse VKP's established optional-object artifact read contract."""

    try:
        value = read_json(Path(path))
    except Exception:  # noqa: BLE001 - optional artifacts fail closed to an empty object.
        return {}
    return value if isinstance(value, dict) else {}


@contextmanager
def bundle_write_lock(
    bundle_dir: str | Path,
    *,
    operation: str = "bundle_write",
    timeout_seconds: float = 0.0,
    lock_name: str = ".bundle-write.lock",
    stale_after_seconds: float = 0.0,
) -> Iterator[None]:
    root = Path(bundle_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    clean_lock_name = str(lock_name or "").strip()
    if not clean_lock_name or Path(clean_lock_name).name != clean_lock_name:
        raise ValueError("lock_name must be a filename without path separators")
    lock_path = root / clean_lock_name
    lock_key = (lock_path.resolve(), threading.get_ident())
    if _LOCK_DEPTHS.get(lock_key, 0) > 0:
        _LOCK_DEPTHS[lock_key] += 1
        try:
            yield
        finally:
            _LOCK_DEPTHS[lock_key] -= 1
        return

    deadline = time.monotonic() + max(0.0, float(timeout_seconds or 0.0))
    lock_id = f"{os.getpid()}:{threading.get_ident()}:{time.time_ns()}"
    payload = {
        "schema": "video_knowledge_pipeline.bundle_write_lock.v1",
        "lock_id": lock_id,
        "operation": str(operation or "bundle_write"),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "created_at_unix": time.time(),
    }
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
            break
        except (FileExistsError, PermissionError) as exc:
            if isinstance(exc, PermissionError) and not lock_path.exists():
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
                continue
            if _remove_stale_lock(lock_path, stale_after_seconds=float(stale_after_seconds or 0.0)):
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError(f"bundle_write_lock_busy: {lock_path}") from exc
            time.sleep(0.05)

    _LOCK_DEPTHS[lock_key] = 1
    try:
        yield
    finally:
        _LOCK_DEPTHS.pop(lock_key, None)
        _release_owned_lock(lock_path, lock_id=lock_id)


def _remove_stale_lock(lock_path: Path, *, stale_after_seconds: float) -> bool:
    if stale_after_seconds <= 0:
        return False
    try:
        stat = lock_path.stat()
    except FileNotFoundError:
        return True
    if max(0.0, time.time() - stat.st_mtime) < stale_after_seconds:
        return False
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        payload = {}
    try:
        pid = int(payload.get("pid") or 0) if isinstance(payload, dict) else 0
    except (TypeError, ValueError):
        pid = 0
    if pid > 0 and _process_is_alive(pid):
        return False
    try:
        current = lock_path.stat()
        if current.st_mtime_ns != stat.st_mtime_ns or current.st_size != stat.st_size:
            return False
        lock_path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _process_is_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _release_owned_lock(lock_path: Path, *, lock_id: str) -> None:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict) or str(payload.get("lock_id") or "") != lock_id:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def append_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_project(root: str | Path) -> ResearchProject:
    paths = project_paths(root)
    data = read_json(paths["project"])
    return ResearchProject(**data)


def save_project(root: str | Path, project: ResearchProject) -> None:
    paths = ensure_project_dirs(root)
    write_json(paths["project"], dataclass_to_dict(project))
