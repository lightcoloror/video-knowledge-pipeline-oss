from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path

from video_knowledge_pipeline.personal_vlog_cli import doctor_main
from video_knowledge_pipeline.personal_vlog_mcp import build_server


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_registers_short_personal_vlog_entries():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    scripts = payload["project"]["scripts"]

    assert scripts["personal-vlog-prepare"].endswith(":prepare_main")
    assert scripts["personal-vlog-run"].endswith(":run_main")
    assert scripts["personal-vlog-resume"].endswith(":resume_main")
    assert scripts["personal-vlog-status"].endswith(":status_main")
    assert scripts["personal-vlog-doctor"].endswith(":doctor_main")
    assert scripts["video-knowledge-personal-vlog-mcp"].endswith(":main")


def test_personal_vlog_mcp_profile_exposes_exactly_four_tools():
    server = build_server()
    tools = asyncio.run(server.list_tools())

    assert [tool.name for tool in tools] == [
        "personal_vlog_prepare",
        "personal_vlog_run",
        "personal_vlog_resume",
        "personal_vlog_status",
    ]


def test_doctor_cli_is_machine_readable_and_side_effect_free(capsys):
    doctor_main([])
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema"] == "video_knowledge_pipeline.personal_vlog_doctor.v1"
    assert payload["ready_to_prepare"] is True
    assert payload["policy"]["automatic_render"] is False
    assert payload["policy"]["automatic_publish"] is False
