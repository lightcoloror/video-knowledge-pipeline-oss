from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.local_tool_inventory as inventory
import video_knowledge_pipeline.general_tagger_adapter as tagger_adapter
from video_knowledge_pipeline.general_tagger_adapter import general_tagger_status


def test_local_runtime_preflight_reports_windows_venv_media_and_dependencies(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "video-knowledge-pipeline"
requires-python = ">=3.11"
dependencies = ["jieba>=0.42", "jsonschema>=4", "markdown-it-py>=3", "rapidfuzz>=3"]
""".strip(),
        encoding="utf-8",
    )
    venv = project / ".venv"
    python = venv / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    (venv / "pyvenv.cfg").write_text("home = C:\\Python311\n", encoding="utf-8")

    monkeypatch.setattr(inventory.sys, "executable", str(python))
    monkeypatch.setattr(inventory.sys, "prefix", str(venv))
    monkeypatch.setattr(inventory.sys, "base_prefix", str(project / "Python311"))
    monkeypatch.setattr(inventory, "resolve_media_tool", lambda name: "" if name == "ffmpeg" else str(project / "ffprobe.exe"))
    monkeypatch.setattr(inventory, "resolve_tesseract", lambda: "")
    monkeypatch.setattr(inventory.shutil, "which", lambda name: str(project / "uv.exe") if name == "uv" else None)
    monkeypatch.setattr(
        inventory.importlib.util,
        "find_spec",
        lambda name: object() if name in {"jsonschema", "markdown_it", "rapidfuzz"} else None,
    )

    result = inventory.local_runtime_preflight(project)

    assert result["schema"] == "video_knowledge_pipeline.local_runtime_preflight.v1"
    assert result["ok"] is False
    assert result["runtime"]["python"]["executable"] == str(python.resolve())
    assert result["runtime"]["python"]["absolute"] is True
    assert result["runtime"]["venv"]["active"] is True
    assert result["runtime"]["venv"]["pyvenv_cfg_exists"] is True
    assert result["runtime"]["uv"]["available"] is True
    assert result["capabilities"]["media"]["ffmpeg"]["available"] is False
    assert result["capabilities"]["media"]["ffprobe"]["available"] is True
    assert "jieba" in result["capabilities"]["dependencies"]["missing"]
    rendered = json.dumps(result["recovery_commands"], ensure_ascii=False)
    assert "lightcolor" not in rendered.lower()
    assert "used-by-codex" not in rendered.lower()
    assert result["boundaries"]["installs_dependencies"] is False
    assert result["boundaries"]["starts_service"] is False


def test_cli_registers_local_runtime_preflight() -> None:
    cli_path = Path(inventory.__file__).with_name("cli.py")
    source = cli_path.read_text(encoding="utf-8")

    assert 'args.command == "local-runtime-preflight"' in source
    assert 'sub.add_parser("local-runtime-preflight"' in source


def test_general_tagger_discovers_ram_assets_from_source_inventory(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "recognize-anything"
    source.mkdir()
    deployment = tmp_path / "models" / "recognize-anything"
    deployment.mkdir(parents=True)
    checkpoint = deployment / "ram_plus_swin_large_14m.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    tokenizer = deployment.parent / "bert-base-uncased-tokenizer"
    tokenizer.mkdir()
    (tokenizer / "vocab.txt").write_text("synthetic", encoding="utf-8")
    source_inventory = tmp_path / "SOURCE_INVENTORY.json"
    source_inventory.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "recognize-anything",
                        "local_path": str(source),
                        "deployment_path": str(deployment),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    for name in ("VKP_RECOGNIZE_ANYTHING_SOURCE", "VKP_RAM_PLUS_CHECKPOINT", "VKP_RAM_PLUS_TOKENIZER", "VKP_LOCAL_MODEL_ROOT"):
        monkeypatch.delenv(name, raising=False)

    status = general_tagger_status(source_inventory_path=source_inventory)

    assert status["status"] == "ready"
    assert status["source_root"] == str(source.resolve())
    assert status["checkpoint_path"] == str(checkpoint.resolve())
    assert status["tokenizer_root"] == str(tokenizer.resolve())
    assert status["model_discovery"]["source"] == "source_inventory"
    assert status["model_discovery"]["inventory_path"] == str(source_inventory.resolve())


def test_general_tagger_inventory_empty_paths_do_not_resolve_to_working_directory(tmp_path: Path) -> None:
    source_inventory = tmp_path / "SOURCE_INVENTORY.json"
    source_inventory.write_text(
        json.dumps({"entries": [{"name": "recognize-anything", "local_path": "", "deployment_path": None}]}),
        encoding="utf-8",
    )

    discovery = tagger_adapter._inventory_model_discovery(source_inventory)

    assert discovery["entry_found"] is True
    assert discovery["source_root"] == ""
    assert discovery["deployment_path"] == ""
    assert discovery["checkpoint_path"] == ""
    assert discovery["tokenizer_root"] == ""


def test_local_runtime_preflight_reports_heavy_python_and_transformers_compatibility(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "video-knowledge-pipeline"\nrequires-python = ">=3.11"\ndependencies = []\n',
        encoding="utf-8",
    )
    heavy_python = tmp_path / "tools" / "heavy-venv" / "Scripts" / "python.exe"
    heavy_python.parent.mkdir(parents=True)
    heavy_python.write_bytes(b"")
    monkeypatch.setenv("VKP_HEAVY_MODEL_PYTHON", str(heavy_python))
    monkeypatch.setattr(inventory, "resolve_media_tool", lambda name: str(tmp_path / f"{name}.exe"))
    monkeypatch.setattr(inventory, "resolve_tesseract", lambda: str(tmp_path / "tesseract.exe"))
    monkeypatch.setattr(inventory, "general_tagger_status", lambda **kwargs: {"status": "ready", "blockers": []})
    monkeypatch.setattr(
        inventory,
        "_probe_heavy_model_runtime",
        lambda executable: {
            "ok": False,
            "status": "incompatible",
            "python_version": "3.11.9",
            "modules": {"torch": True, "transformers": True, "timm": True, "fairscale": True, "PIL": True},
            "transformers": {
                "version": "4.57.6",
                "compatible": False,
                "required_symbols": {
                    "modeling_utils.PreTrainedModel": True,
                    "pytorch_utils.apply_chunking_to_forward": False,
                    "pytorch_utils.find_pruneable_heads_and_indices": True,
                    "pytorch_utils.prune_linear_layer": True,
                },
            },
            "error": "",
        },
    )

    result = inventory.local_runtime_preflight(project)

    assert result["runtime"]["heavy_model_python"]["executable"] == str(heavy_python.resolve())
    assert result["runtime"]["heavy_model_python"]["source"] == "VKP_HEAVY_MODEL_PYTHON"
    assert result["capabilities"]["general_tagger"]["status"] == "ready"
    compatibility = result["capabilities"]["heavy_model_runtime"]
    assert compatibility["transformers"]["version"] == "4.57.6"
    assert compatibility["transformers"]["compatible"] is False
    assert "heavy_model:transformers_compatibility" in result["failed_checks"]
    assert any(row["key"] == "repair_heavy_model_runtime" for row in result["recovery_commands"])
    rendered = json.dumps(result["recovery_commands"], ensure_ascii=False)
    assert str(heavy_python) not in rendered
    assert result["boundaries"]["loads_model"] is False
    assert result["boundaries"]["initializes_gpu"] is False
