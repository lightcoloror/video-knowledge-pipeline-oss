from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Iterable

CUDA_PACKAGE_NAMES = (
    "nvidia.cublas",
    "nvidia.cudnn",
    "nvidia.cuda_nvrtc",
    "nvidia.cuda_runtime",
)

_CUDA_DLL_HANDLES: list[object] = []
_CUDA_DLL_DIRS_READY = False


def discover_nvidia_bin_dirs(package_names: Iterable[str] = CUDA_PACKAGE_NAMES) -> list[Path]:
    """Discover pip-installed NVIDIA CUDA package bin directories.

    Adapted from vsummary's faster-whisper Windows helper, kept dependency-free
    so VKP can use it for ASR environment diagnostics and fallback runners.
    """

    candidates: list[Path] = []
    seen: set[str] = set()
    for package_name in package_names:
        try:
            spec = importlib.util.find_spec(package_name)
        except ModuleNotFoundError:
            continue
        if spec is None:
            continue
        locations = getattr(spec, "submodule_search_locations", None)
        if not locations:
            continue
        for location in locations:
            bin_dir = Path(location) / "bin"
            key = str(bin_dir.resolve()) if bin_dir.exists() else str(bin_dir)
            if bin_dir.exists() and key not in seen:
                seen.add(key)
                candidates.append(bin_dir)
    return candidates


def cuda_dll_discovery_status(package_names: Iterable[str] = CUDA_PACKAGE_NAMES) -> dict[str, Any]:
    dirs = discover_nvidia_bin_dirs(package_names)
    path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    return {
        "schema": "video_knowledge_pipeline.cuda_dll_discovery.v1",
        "platform": sys.platform,
        "supported": sys.platform == "win32" and hasattr(os, "add_dll_directory"),
        "ready": _CUDA_DLL_DIRS_READY,
        "package_names": list(package_names),
        "discovered_bin_dirs": [str(path) for path in dirs],
        "discovered_count": len(dirs),
        "missing_from_path": [str(path) for path in dirs if str(path) not in path_entries],
        "path_already_contains": [str(path) for path in dirs if str(path) in path_entries],
        "operator_boundary": {
            "local_only": True,
            "no_install": True,
            "no_process_started": True,
        },
    }


def ensure_windows_cuda_dll_dirs(package_names: Iterable[str] = CUDA_PACKAGE_NAMES, *, register: bool = True) -> dict[str, Any]:
    """Register discovered NVIDIA package bin dirs with Windows DLL search path.

    The function is safe to call repeatedly. On non-Windows platforms it returns a
    skipped report and does not mutate PATH. `register=False` can be used for dry
    run diagnostics.
    """

    global _CUDA_DLL_DIRS_READY
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        report = cuda_dll_discovery_status(package_names)
        report.update({"status": "skipped_non_windows_or_unsupported", "registered_dirs": []})
        return report
    if _CUDA_DLL_DIRS_READY:
        report = cuda_dll_discovery_status(package_names)
        report.update({"status": "already_ready", "registered_dirs": []})
        return report

    dirs = discover_nvidia_bin_dirs(package_names)
    existing_path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    prepended: list[str] = []
    registered: list[str] = []
    for path in dirs:
        resolved = str(path)
        if register:
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(resolved))
            registered.append(resolved)
        if resolved not in existing_path_entries and resolved not in prepended:
            prepended.append(resolved)
    if register and prepended:
        os.environ["PATH"] = os.pathsep.join([*prepended, *existing_path_entries])
    if register:
        _CUDA_DLL_DIRS_READY = True

    report = cuda_dll_discovery_status(package_names)
    report.update(
        {
            "status": "registered" if register else "dry_run",
            "registered_dirs": registered,
            "prepended_path_dirs": prepended,
            "handle_count": len(_CUDA_DLL_HANDLES),
        }
    )
    return report
