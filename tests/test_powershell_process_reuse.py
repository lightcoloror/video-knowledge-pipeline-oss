from __future__ import annotations

from pathlib import Path

import video_knowledge_pipeline.extractor_execution as extractor_execution
import video_knowledge_pipeline.lecture_command as lecture_command
import video_knowledge_pipeline.powershell as powershell
from video_knowledge_pipeline.powershell import run_powershell_command


def test_lecture_process_front_doors_share_the_existing_owner() -> None:
    assert extractor_execution._run_command is run_powershell_command
    assert lecture_command._run_command is run_powershell_command


def test_shared_process_runner_preserves_existing_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_run(command: list[str], **kwargs: object) -> object:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(powershell.subprocess, "run", fake_run)
    monkeypatch.setattr(
        powershell,
        "local_tool_subprocess_env",
        lambda: {"PATH": "registered-tools"},
    )

    result = run_powershell_command(
        "Write-Output ready",
        cwd=tmp_path,
        timeout_seconds=0,
    )

    assert result is sentinel
    assert captured["command"] == [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "Write-Output ready",
    ]
    assert captured["kwargs"] == {
        "cwd": str(tmp_path),
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "capture_output": True,
        "timeout": None,
        "check": False,
        "env": {"PATH": "registered-tools"},
    }


def test_only_shared_owner_contains_the_lecture_process_contract() -> None:
    source_root = (
        Path(__file__).resolve().parents[1] / "src" / "video_knowledge_pipeline"
    )
    signature = '["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass"'
    owners = {
        path.name
        for path in source_root.glob("*.py")
        if signature in path.read_text(encoding="utf-8-sig")
    }
    assert owners == {"powershell.py"}
