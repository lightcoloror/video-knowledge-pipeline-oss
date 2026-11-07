from __future__ import annotations

import subprocess

from video_knowledge_pipeline import asr_runner


def test_moss_runtime_probe_allows_heavy_upstream_import_budget(monkeypatch) -> None:
    """The fixed MOSS CLI imports Torch/Transformers before argparse exits.

    Intent: avoid a false runtime blocker on a healthy local GPU environment.
    Decision: keep the upstream mtd-subtitle --help readiness contract and
    allow a 60-second bounded import budget instead of building a second probe.
    Reason: the real isolated Windows runtime completed in 33.6 seconds, beyond
    the previous 20-second budget, with exit code zero.
    Evidence: the exact local MOSS CLI at the fixed upstream adapter entrypoint.
    Effective scope: local readiness probing only; no inference or download.
    """

    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(command, 0, "usage: mtd-subtitle", "")

    monkeypatch.setattr(asr_runner.subprocess, "run", fake_run)

    result = asr_runner._command_runtime_probe(
        preset="moss-transcribe-diarize",
        command_path="mtd-subtitle",
    )

    assert observed == {
        "command": ["mtd-subtitle", "--help"],
        "timeout": 60,
    }
    assert result["ready"] is True
    assert result["status"] == "ready"
