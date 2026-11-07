from __future__ import annotations

import os
from pathlib import Path

from video_knowledge_pipeline.cuda_runtime import cuda_dll_discovery_status, ensure_windows_cuda_dll_dirs
from video_knowledge_pipeline.stage_cache import StageCache, atomic_copy_file, atomic_write_text


def test_stage_cache_validates_identity_and_source_fingerprint(tmp_path: Path) -> None:
    source = tmp_path / "lesson.mp4"
    source.write_bytes(b"video-v1")
    cache = StageCache(tmp_path / "cache", source)

    cache.store_json("asr", "normalized.json", {"segments": [1]}, identity="sensevoice-v1")

    assert cache.is_valid("asr", identity="sensevoice-v1") is True
    assert cache.load_json("asr", "normalized.json", identity="sensevoice-v1") == {"segments": [1]}
    assert cache.load_json("asr", "normalized.json", identity="sensevoice-v2") is None

    source.write_bytes(b"video-v2-longer")
    assert cache.is_valid("asr", identity="sensevoice-v1") is False
    assert cache.load_json("asr", "normalized.json", identity="sensevoice-v1") is None


def test_stage_cache_atomic_text_and_file_restore(tmp_path: Path) -> None:
    source = tmp_path / "lesson.mp4"
    source.write_bytes(b"video")
    cache = StageCache(tmp_path / "cache", source)

    text_path = cache.store_text("summary", "draft.md", "# Draft\n", identity="summary-v1")
    assert text_path.read_text(encoding="utf-8") == "# Draft\n"
    assert cache.load_text("summary", "draft.md", identity="summary-v1") == "# Draft\n"

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    cache.store_file("media", "audio.wav", audio, identity="ffmpeg-v1")
    restored = tmp_path / "restored" / "audio.wav"
    assert cache.restore_file("media", "audio.wav", restored, identity="ffmpeg-v1") is True
    assert restored.read_bytes() == b"wav"
    assert cache.restore_file("media", "audio.wav", restored, identity="ffmpeg-v2") is False

    atomic_write_text(tmp_path / "atomic" / "note.md", "ok")
    assert (tmp_path / "atomic" / "note.md").read_text(encoding="utf-8") == "ok"
    copied = atomic_copy_file(audio, tmp_path / "atomic" / "audio.wav")
    assert copied.read_bytes() == b"wav"


def test_stage_cache_atomic_text_reuses_storage_owner(monkeypatch, tmp_path: Path) -> None:
    import video_knowledge_pipeline.stage_cache as stage_cache

    calls: list[tuple[Path, str]] = []

    def fake_write_text_atomic(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
        calls.append((path, payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding=encoding)

    monkeypatch.setattr(stage_cache, "write_text_atomic", fake_write_text_atomic)
    destination = tmp_path / "delegated" / "note.md"

    assert stage_cache.atomic_write_text(destination, "shared") == destination.resolve()
    assert calls == [(destination.resolve(), "shared")]

def test_cuda_runtime_dry_run_and_registration(monkeypatch, tmp_path: Path) -> None:
    import video_knowledge_pipeline.cuda_runtime as cuda_runtime

    bin_dir = tmp_path / "nvidia" / "cublas" / "bin"
    bin_dir.mkdir(parents=True)
    monkeypatch.setattr(cuda_runtime.sys, "platform", "win32")
    monkeypatch.setattr(cuda_runtime, "_CUDA_DLL_DIRS_READY", False)
    monkeypatch.setattr(cuda_runtime, "_CUDA_DLL_HANDLES", [])
    monkeypatch.setattr(cuda_runtime, "discover_nvidia_bin_dirs", lambda package_names=cuda_runtime.CUDA_PACKAGE_NAMES: [bin_dir])
    handles = []

    def fake_add_dll_directory(path: str):
        handles.append(path)
        return {"path": path}

    monkeypatch.setattr(cuda_runtime.os, "add_dll_directory", fake_add_dll_directory, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))

    dry = ensure_windows_cuda_dll_dirs(register=False)
    assert dry["status"] == "dry_run"
    assert dry["registered_dirs"] == []
    assert handles == []

    result = ensure_windows_cuda_dll_dirs(register=True)
    assert result["status"] == "registered"
    assert result["registered_dirs"] == [str(bin_dir)]
    assert handles == [str(bin_dir)]
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)

    status = cuda_dll_discovery_status()
    assert status["ready"] is True
