"""Bounded, side-effect-free health probe for the VKP command surface.

This module intentionally uses only the Python standard library.  It verifies
that the registered project, configuration, schema and CLI entrypoints are
readable and mutually compatible without importing ASR, Torch, CUDA, provider
or service modules.  Detailed ASR readiness remains the responsibility of
``asr-env-status``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


SCHEMA = "video_knowledge_pipeline.quick_health.v1"
SCHEMA_VERSION = "1.0"
PROJECT_NAME = "video-knowledge-pipeline"
CONFIG_SCHEMA = "video_knowledge_pipeline.config.v1"
FORBIDDEN_HEAVY_MODULES = (
    "funasr",
    "jieba",
    "modelscope",
    "qwen_asr",
    "torch",
    "transformers",
    "whisperx",
)


def build_quick_health(project_root: str | Path | None = None) -> dict[str, Any]:
    """Build a live, deterministic quick-health snapshot without side effects."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    checks: list[dict[str, Any]] = []

    def check(check_id: str, ok: bool, detail: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "passed" if ok else "failed",
                "detail": detail,
            }
        )

    required_paths = {
        "cli": root / "src" / "video_knowledge_pipeline" / "cli.py",
        "package": root / "src" / "video_knowledge_pipeline" / "__init__.py",
        "quick_health": root / "src" / "video_knowledge_pipeline" / "quick_health.py",
        "portable": root / "src" / "video_knowledge_pipeline" / "portable.py",
        "agent_manifest": root / "agent-tool-manifest.v1.json",
        "portable_lock": root / "portable-contract.lock.json",
        "config": root / "config" / "video-knowledge-pipeline.json",
        "pyproject": root / "pyproject.toml",
        "schema": root
        / "src"
        / "video_knowledge_pipeline"
        / "schemas"
        / "quick-health.v1.schema.json",
    }

    check("project_root", root.is_dir(), "project root is a readable directory")
    artifacts: list[dict[str, Any]] = []
    for key, path in required_paths.items():
        readable = _is_readable_file(path)
        check(f"required_path:{key}", readable, f"{key} is readable")
        if readable:
            artifacts.append(_artifact_snapshot(root, key, path))

    project_version = ""
    required_python = ""
    pyproject = _read_toml(required_paths["pyproject"])
    project = pyproject.get("project") if isinstance(pyproject, dict) else None
    if isinstance(project, dict):
        project_name = str(project.get("name") or "")
        project_version = str(project.get("version") or "")
        required_python = str(project.get("requires-python") or "")
        check(
            "project_identity",
            project_name == PROJECT_NAME and bool(project_version),
            f"pyproject name must be {PROJECT_NAME!r} and version must be non-empty",
        )
        check(
            "python_contract",
            _python_contract_is_supported(required_python),
            "requires-python must retain the supported Python 3.11+ contract",
        )
    else:
        check("project_identity", False, "pyproject.toml is invalid or lacks [project]")
        check("python_contract", False, "Python compatibility cannot be established")

    package_version = _read_package_version(required_paths["package"])
    check(
        "package_version",
        bool(package_version) and package_version == project_version,
        "package __version__ must match pyproject version",
    )

    config = _read_json(required_paths["config"])
    check(
        "config_schema",
        isinstance(config, dict) and config.get("schema") == CONFIG_SCHEMA,
        f"config schema must be {CONFIG_SCHEMA!r}",
    )

    schema = _read_json(required_paths["schema"])
    schema_valid = _quick_schema_is_compatible(schema)
    check(
        "quick_health_schema",
        schema_valid,
        f"quick-health schema must enforce {SCHEMA!r} at version {SCHEMA_VERSION!r}",
    )

    cli_text = _read_text(required_paths["cli"])
    pyproject_text = _read_text(required_paths["pyproject"])
    wrapper_path = root / "scripts" / "video-knowledge.ps1"
    wrapper_text = _read_text(wrapper_path)
    check(
        "python_entrypoint",
        "video-knowledge-quick-health" in pyproject_text
        and "video-knowledge-portable" in pyproject_text,
        "pyproject must expose cross-platform quick-health and portable commands",
    )
    check(
        "detailed_diagnostic_preserved",
        "asr-env-status" in cli_text,
        "the existing detailed ASR diagnostic entrypoint must remain available",
    )
    check(
        "cli_entrypoint",
        "def main(" in cli_text and "def build_parser(" in cli_text,
        "the primary CLI entrypoint and parser must remain present",
    )
    wrapper_cli_runtime_ok = not wrapper_text or (
        '$venvPython = Join-Path $root ".venv\\Scripts\\python.exe"'
        in wrapper_text
        and "python -m video_knowledge_pipeline.cli @args" not in wrapper_text
        and wrapper_text.count(
            "& $pythonCommand -m video_knowledge_pipeline.cli @args"
        )
        >= 2
        and "& $pythonCommand -m video_knowledge_pipeline.model_destination_allowlist"
        in wrapper_text
    )
    check(
        "windows_wrapper_cli_runtime",
        wrapper_cli_runtime_ok,
        (
            "Windows full CLI commands must prefer the project venv and reuse the "
            "same interpreter for provider allowlist discovery"
            if wrapper_text
            else "Windows wrapper is absent; cross-platform Python entrypoints remain authoritative"
        ),
    )

    heavy_modules_loaded = sorted(
        name for name in FORBIDDEN_HEAVY_MODULES if name in sys.modules
    )
    check(
        "heavy_module_import_boundary",
        _source_avoids_forbidden_imports(Path(__file__)),
        "quick-health source must not import ASR, model, Torch or CUDA stacks",
    )
    python_supported = sys.version_info >= (3, 11)
    check(
        "runtime_python",
        python_supported,
        "quick health requires Python 3.11 or newer",
    )
    entity_capability = _entity_correction_capability()
    check(
        "entity_correction_capability",
        True,
        (
            f"status={entity_capability['status']}; "
            f"pypinyin_available={str(entity_capability['pypinyin_available']).lower()}; "
            "entity_glossary=v1; candidate_only=true; automatic_apply=false; "
            f"gap={entity_capability['gap']}; recovery={entity_capability['recovery']}"
        ),
    )

    failed = [item["check_id"] for item in checks if item["status"] == "failed"]
    ok = not failed
    snapshot_sha256 = _snapshot_sha256(artifacts) if artifacts else ""
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "status": "ready" if ok else "not_ready",
        "registration": {
            "status": "registered" if ok else "invalid",
            "project_name": PROJECT_NAME,
            "project_version": project_version,
            "project_root": str(root),
        },
        "runtime": {
            "status": "ready" if python_supported else "unsupported",
            "python_version": ".".join(str(value) for value in sys.version_info[:3]),
            "required_python": required_python,
            "heavy_modules_loaded": heavy_modules_loaded,
            "model_stack_evaluated": False,
        },
        "callable": {
            "status": "callable" if ok else "not_callable",
            "quick_command": "python -m video_knowledge_pipeline.quick_health",
            "detailed_asr_command": "python -m video_knowledge_pipeline.cli asr-env-status",
            "windows_wrapper": "scripts\\video-knowledge.ps1 quick-health" if wrapper_text else "",
            "asr_capability_evaluated": False,
        },
        "freshness": {
            "status": "current" if ok and snapshot_sha256 else "unavailable",
            "basis": "live_required_artifact_snapshot",
            "cached": False,
            "artifact_count": len(artifacts),
            "snapshot_sha256": snapshot_sha256,
            "artifacts": artifacts,
        },
        "checks": checks,
        "failed_checks": failed,
        "boundaries": {
            "starts_service": False,
            "writes_business_data": False,
            "loads_model_stack": False,
            "checks_provider": False,
            "proves_asr_ready": False,
        },
    }


def _is_readable_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _read_package_version(path: Path) -> str:
    source = _read_text(path)
    if not source:
        return ""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return ""


def _source_avoids_forbidden_imports(path: Path) -> bool:
    source = _read_text(path)
    if not source:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    return not imported_roots.intersection(FORBIDDEN_HEAVY_MODULES)


def _entity_correction_capability() -> dict[str, Any]:
    """Probe the optional phonetic backend without importing it."""

    try:
        available = importlib.util.find_spec("pypinyin") is not None
    except (ImportError, AttributeError, ValueError):
        available = False
    return {
        "status": "available" if available else "degraded",
        "pypinyin_available": available,
        "gap": "none" if available else "phonetic_similarity_unavailable",
        "recovery": (
            "none"
            if available
            else "install pypinyin in the VKP project environment and rerun entity-lexicon"
        ),
    }


def _python_contract_is_supported(value: str) -> bool:
    normalized = value.replace(" ", "")
    return normalized.startswith(">=3.11")


def _quick_schema_is_compatible(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    properties = value.get("properties")
    required = value.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return False
    return (
        value.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
        and (properties.get("schema") or {}).get("const") == SCHEMA
        and (properties.get("schema_version") or {}).get("const") == SCHEMA_VERSION
        and {"schema", "schema_version", "ok", "status"}.issubset(set(required))
    )


def _artifact_snapshot(root: Path, key: str, path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "key": key,
        "path": path.relative_to(root).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _snapshot_sha256(artifacts: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        artifacts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VKP's bounded static quick-health probe without loading model stacks."
    )
    parser.add_argument("--project-root", default="")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_quick_health(args.project_root or None)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
            separators=None if args.pretty else (",", ":"),
        )
    )
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
