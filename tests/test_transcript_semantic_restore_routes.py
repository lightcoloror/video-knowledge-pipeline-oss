from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from video_knowledge_pipeline import cli, mcp_server
from video_knowledge_pipeline.storage import write_text_atomic
from video_knowledge_pipeline.transcript_semantic_transaction import (
    commit_transcript_semantic_artifacts,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_restore_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    manifest = bundle / "manifest.json"
    target = bundle / "source-arbitrated-transcript.json"
    write_text_atomic(manifest, '{"revision": 1}')
    write_text_atomic(target, '{"segments": ["before"]}')
    result = commit_transcript_semantic_artifacts(
        bundle,
        artifacts={"source-arbitrated-transcript.json": '{"segments": ["after"]}'},
        expected_inputs={"manifest.json": _sha(manifest)},
        operation_identity={"pack_sha256": "a" * 64},
    )
    return bundle, Path(result["restore_plan"]), target, str(result["transaction_id"])


def _run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> dict[str, Any]:
    assert cli.main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def test_cli_semantic_restore_defaults_to_preview_and_requires_exact_confirmation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle, plan, target, transaction_id = _semantic_restore_fixture(tmp_path)
    after = target.read_bytes()

    preview = _run_cli(
        capsys,
        "transcript-semantic-restore",
        str(bundle),
        "--plan-json",
        str(plan),
    )
    assert preview["status"] == "ready"
    assert preview["execute"] is False
    assert target.read_bytes() == after

    mismatch = _run_cli(
        capsys,
        "transcript-semantic-restore",
        str(bundle),
        "--plan-json",
        str(plan),
        "--execute",
        "--confirm-transaction-id",
        f"{transaction_id}-wrong",
    )
    assert mismatch["status"] == "blocked"
    assert "confirm_transaction_id_mismatch" in mismatch["blockers"]
    assert target.read_bytes() == after


def test_cli_semantic_restore_fails_closed_on_hash_drift_and_repeat(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle, plan, target, transaction_id = _semantic_restore_fixture(tmp_path / "drift")
    write_text_atomic(target, "operator edit")

    drift = _run_cli(
        capsys,
        "transcript-semantic-restore",
        str(bundle),
        "--plan-json",
        str(plan),
        "--execute",
        "--confirm-transaction-id",
        transaction_id,
    )
    assert drift["status"] == "blocked"
    assert "output_drift:source-arbitrated-transcript.json" in drift["blockers"]
    assert target.read_text(encoding="utf-8") == "operator edit"

    bundle, plan, target, transaction_id = _semantic_restore_fixture(tmp_path / "repeat")
    restored = _run_cli(
        capsys,
        "transcript-semantic-restore",
        str(bundle),
        "--plan-json",
        str(plan),
        "--execute",
        "--confirm-transaction-id",
        transaction_id,
    )
    assert restored["status"] == "restored"
    assert target.read_text(encoding="utf-8") == '{"segments": ["before"]}'

    repeated = _run_cli(
        capsys,
        "transcript-semantic-restore",
        str(bundle),
        "--plan-json",
        str(plan),
        "--execute",
        "--confirm-transaction-id",
        transaction_id,
    )
    assert repeated["status"] == "already_restored"
    assert repeated["ok"] is True
    assert repeated["restored_targets"] == []
    assert repeated["already_restored_targets"] == [
        "source-arbitrated-transcript.json"
    ]


def test_cli_semantic_restore_rejects_plan_outside_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle, plan, _target, _transaction_id = _semantic_restore_fixture(tmp_path / "inside")
    outside_plan = tmp_path / "outside-plan.json"
    outside_plan.write_bytes(plan.read_bytes())

    with pytest.raises(ValueError, match="restore_plan_outside_bundle"):
        _run_cli(
            capsys,
            "transcript-semantic-restore",
            str(bundle),
            "--plan-json",
            str(outside_plan),
        )


class _FakeFastMCP:
    last: "_FakeFastMCP | None" = None

    def __init__(self, _name: str) -> None:
        self.tools: dict[str, Any] = {}
        _FakeFastMCP.last = self

    def tool(self, *args: Any, **kwargs: Any):
        explicit_name = kwargs.get("name")

        def decorator(func):
            self.tools[str(explicit_name or func.__name__)] = func
            return func

        if args and callable(args[0]):
            return decorator(args[0])
        return decorator

    def run(self) -> None:
        return None


def _install_fake_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP
    mcp_module.server = server_module
    server_module.fastmcp = fastmcp_module
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)


def test_mcp_semantic_restore_uses_same_preview_and_confirmation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, plan, target, transaction_id = _semantic_restore_fixture(tmp_path)
    after = target.read_bytes()
    _install_fake_mcp(monkeypatch)

    mcp_server.main()
    server = _FakeFastMCP.last
    assert server is not None
    restore = server.tools["apply_transcript_semantic_restore"]

    preview = restore(str(bundle), str(plan))
    assert preview["status"] == "ready"
    assert preview["execute"] is False
    assert target.read_bytes() == after

    mismatch = restore(
        str(bundle),
        str(plan),
        execute=True,
        confirm_transaction_id=f"{transaction_id}-wrong",
    )
    assert mismatch["status"] == "blocked"
    assert "confirm_transaction_id_mismatch" in mismatch["blockers"]
    assert target.read_bytes() == after


def test_powershell_routes_semantic_restore_as_project_local_offline_command() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "video-knowledge.ps1").read_text(encoding="utf-8-sig")

    assert '"transcript-semantic-restore"' in script
