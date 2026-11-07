from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import jsonschema

from video_knowledge_pipeline.quick_health import SCHEMA, build_quick_health
from portable_test_runtime import portable_test_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    PROJECT_ROOT
    / "src"
    / "video_knowledge_pipeline"
    / "schemas"
    / "quick-health.v1.schema.json"
)


def test_quick_health_is_schema_valid_and_deterministic() -> None:
    first = build_quick_health(PROJECT_ROOT)
    second = build_quick_health(PROJECT_ROOT)

    jsonschema.validate(first, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    assert first == second
    assert first["schema"] == SCHEMA
    assert first["ok"] is True
    assert first["registration"]["status"] == "registered"
    assert isinstance(first["runtime"]["heavy_modules_loaded"], list)
    assert first["runtime"]["model_stack_evaluated"] is False
    assert first["callable"]["asr_capability_evaluated"] is False
    assert first["callable"]["quick_command"] == "python -m video_knowledge_pipeline.quick_health"
    assert "windows_wrapper_cli_runtime" not in first["failed_checks"]
    assert any(
        item["check_id"] == "windows_wrapper_cli_runtime"
        and item["status"] == "passed"
        for item in first["checks"]
    )
    assert first["freshness"]["cached"] is False
    assert first["boundaries"] == {
        "starts_service": False,
        "writes_business_data": False,
        "loads_model_stack": False,
        "checks_provider": False,
        "proves_asr_ready": False,
    }


def test_quick_health_fails_closed_for_missing_project() -> None:
    with portable_test_directory("missing-project") as tmp_path:
        result = build_quick_health(tmp_path / "missing")

    assert result["ok"] is False
    assert result["status"] == "not_ready"
    assert "project_root" in result["failed_checks"]
    assert "required_path:config" in result["failed_checks"]


def test_quick_health_fails_closed_for_corrupt_config() -> None:
    with portable_test_directory("corrupt-config") as tmp_path:
        root = _copy_minimal_project(tmp_path)
        (root / "config" / "video-knowledge-pipeline.json").write_text(
            "{", encoding="utf-8"
        )

        result = build_quick_health(root)

    assert result["ok"] is False
    assert "config_schema" in result["failed_checks"]


def test_quick_health_fails_closed_for_schema_version_drift() -> None:
    with portable_test_directory("schema-drift") as tmp_path:
        root = _copy_minimal_project(tmp_path)
        schema_path = (
            root
            / "src"
            / "video_knowledge_pipeline"
            / "schemas"
            / "quick-health.v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["properties"]["schema_version"]["const"] = "2.0"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")

        result = build_quick_health(root)

    assert result["ok"] is False
    assert "quick_health_schema" in result["failed_checks"]


def test_quick_health_fails_closed_for_project_identity_drift() -> None:
    with portable_test_directory("identity-drift") as tmp_path:
        root = _copy_minimal_project(tmp_path)
        pyproject_path = root / "pyproject.toml"
        pyproject_path.write_text(
            pyproject_path.read_text(encoding="utf-8").replace(
                'name = "video-knowledge-pipeline"', 'name = "different-project"'
            ),
            encoding="utf-8",
        )

        result = build_quick_health(root)

    assert result["ok"] is False
    assert "project_identity" in result["failed_checks"]


def test_quick_health_cli_is_bounded_idempotent_and_read_only() -> None:
    before = _required_hashes(PROJECT_ROOT)
    outputs: list[bytes] = []
    durations: list[float] = []
    command = [sys.executable, "-m", "video_knowledge_pipeline.quick_health"]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    for _ in range(3):
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            timeout=10,
        )
        durations.append(time.perf_counter() - started)
        assert completed.returncode == 0, completed.stderr.decode(
            "utf-8", errors="replace"
        )
        payload = json.loads(completed.stdout.decode("utf-8"))
        assert payload["runtime"]["heavy_modules_loaded"] == []
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1] == outputs[2]
    assert max(durations) < 10
    assert _required_hashes(PROJECT_ROOT) == before


def _copy_minimal_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    paths = (
        "src/video_knowledge_pipeline/cli.py",
        "src/video_knowledge_pipeline/__init__.py",
        "src/video_knowledge_pipeline/quick_health.py",
        "src/video_knowledge_pipeline/portable.py",
        "src/video_knowledge_pipeline/schemas/quick-health.v1.schema.json",
        "scripts/video-knowledge.ps1",
        "agent-tool-manifest.v1.json",
        "portable-contract.lock.json",
        "config/video-knowledge-pipeline.json",
        "pyproject.toml",
    )
    for relative in paths:
        source = PROJECT_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


def _required_hashes(root: Path) -> dict[str, str]:
    paths = (
        root / "src" / "video_knowledge_pipeline" / "cli.py",
        root / "src" / "video_knowledge_pipeline" / "__init__.py",
        root / "src" / "video_knowledge_pipeline" / "quick_health.py",
        root / "src" / "video_knowledge_pipeline" / "portable.py",
        root
        / "src"
        / "video_knowledge_pipeline"
        / "schemas"
        / "quick-health.v1.schema.json",
        root / "scripts" / "video-knowledge.ps1",
        root / "agent-tool-manifest.v1.json",
        root / "portable-contract.lock.json",
        root / "config" / "video-knowledge-pipeline.json",
        root / "pyproject.toml",
    )
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
