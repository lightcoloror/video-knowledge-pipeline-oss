from __future__ import annotations

import inspect

from video_knowledge_pipeline import mcp_server
from video_knowledge_pipeline.cli import _mcp_callables


def test_model_task_tools_are_registered_for_local_mcp_dispatch() -> None:
    tools = _mcp_callables()

    assert tools["model_task_coverage_audit"]
    assert tools["run_term_arbitration_model"]
    assert tools["run_bilinote_mind_map_model"]


def test_fastmcp_server_declares_model_task_tools() -> None:
    source = inspect.getsource(mcp_server.main)

    assert "def model_task_coverage_audit_tool(" in source
    assert "def run_term_arbitration_model_tool(" in source
    assert "def run_bilinote_mind_map_model_tool(" in source
