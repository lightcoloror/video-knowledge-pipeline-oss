from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from video_knowledge_pipeline import cli, mcp_server
from video_knowledge_pipeline.transcript_semantic_benchmark import (
    run_transcript_semantic_benchmark,
)


FIXTURE = Path(__file__).parent / "fixtures" / "transcript-semantic-gold-v2.json"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _benchmark_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    gold = json.loads(FIXTURE.read_text(encoding="utf-8"))
    expected = [
        row
        for row in gold["cases"]
        if row["expect_candidate"]
        and row["projection_disposition"] == "retained_semantic"
    ]
    candidates = []
    decisions = []
    for row in expected:
        candidate_id = f"candidate-{row['case_id']}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "segment_index": row["segment_index"],
                "original_text": row["original_text"],
                "correction_type": row["correction_type"],
            }
        )
        decisions.append(
            {
                "candidate_id": candidate_id,
                "correction_type": row["correction_type"],
                "corrected_text": row["corrected_text"],
                "human_confirmed": row["correction_type"] != "negation",
            }
        )
    pack = _write_json(tmp_path / "candidate-pack.json", {"candidates": candidates})
    validation = _write_json(
        tmp_path / "validation.json", {"accepted_decisions": decisions}
    )
    full_body = tmp_path / "full-body.md"
    full_body.write_text(
        "\n".join(str(row["corrected_text"]) for row in expected),
        encoding="utf-8",
    )
    smart_summary = tmp_path / "smart-summary.md"
    smart_summary.write_text("本摘要仅概述脱敏合成案例的复核结果。", encoding="utf-8")
    return pack, validation, full_body, smart_summary


def test_v2_fixture_is_synthetic_and_covers_high_risk_types() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["schema"] == "video_knowledge_pipeline.transcript_semantic_gold.v2"
    assert payload["deidentified_fixture_only"] is True
    assert payload["synthetic_only"] is True
    assert len(payload["cases"]) >= 12
    assert all(row["synthetic"] is True for row in payload["cases"])
    assert {
        "negation",
        "condition",
        "exception",
        "person_name",
        "organization",
        "date_time",
        "money",
        "percentage",
    }.issubset({row["correction_type"] for row in payload["cases"]})
    assert {"ordinary_homophone", "collocation_anomaly"}.issubset(
        {row["correction_type"] for row in payload["cases"]}
    )


def test_file_wrapper_reads_projection_files_and_writes_receipt(
    tmp_path: Path,
) -> None:
    pack, validation, full_body, smart_summary = _benchmark_inputs(tmp_path)
    output = tmp_path / "benchmark.json"

    result = run_transcript_semantic_benchmark(
        FIXTURE,
        pack,
        validation,
        full_body_path=full_body,
        smart_summary_path=smart_summary,
        output_json=output,
    )

    assert result["ok"] is True
    assert result["counts"]["gold_cases"] == 13
    assert result["counts"]["expected_candidates"] == 11
    assert result["metrics"]["auto_apply_accuracy"] == 1.0
    assert result["dangerous_misedit_by_type"]["negation"] == {
        "accepted_decisions": 1,
        "auto_applied": 1,
        "incorrect_auto": 0,
        "rate": 0.0,
    }
    assert result["dangerous_misedit_by_type"]["person_name"][
        "accepted_decisions"
    ] == 1
    assert result["dangerous_misedit_by_type"]["money"][
        "accepted_decisions"
    ] == 1
    assert result["projection_inputs"]["full_body"]["path"] == str(
        full_body.resolve()
    )
    assert result["projection_inputs"]["full_body"]["sha256"]
    assert result["projection_inputs"]["smart_summary"]["path"] == str(
        smart_summary.resolve()
    )
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_cli_transcript_semantic_benchmark_uses_file_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pack, validation, full_body, smart_summary = _benchmark_inputs(tmp_path)
    output = tmp_path / "cli-benchmark.json"

    exit_code = cli.main(
        [
            "transcript-semantic-benchmark",
            str(FIXTURE),
            str(pack),
            str(validation),
            "--full-body-path",
            str(full_body),
            "--smart-summary-path",
            str(smart_summary),
            "--output-json",
            str(output),
            "--verbose",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["counts"]["gold_cases"] == 13
    assert output.exists()


def test_cli_completed_benchmark_failure_uses_result_status_not_process_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pack, validation, full_body, smart_summary = _benchmark_inputs(tmp_path)
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    negation = next(
        row
        for row in validation_payload["accepted_decisions"]
        if row["correction_type"] == "negation"
    )
    negation["corrected_text"] = "该申请一定可以直接通过"
    _write_json(validation, validation_payload)

    exit_code = cli.main(
        [
            "transcript-semantic-benchmark",
            str(FIXTURE),
            str(pack),
            str(validation),
            "--full-body-path",
            str(full_body),
            "--smart-summary-path",
            str(smart_summary),
            "--verbose",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "failed"
    assert payload["ok"] is False
    assert payload["dangerous_misedit_by_type"]["negation"]["rate"] == 1.0


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


def test_mcp_transcript_semantic_benchmark_uses_same_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack, validation, full_body, smart_summary = _benchmark_inputs(tmp_path)
    _install_fake_mcp(monkeypatch)

    mcp_server.main()
    server = _FakeFastMCP.last
    assert server is not None
    assert "transcript_semantic_benchmark" in server.tools
    assert "transcript_semantic_benchmark_tool" in server.tools
    assert "run_transcript_semantic_benchmark" not in server.tools
    assert "evaluate_transcript_semantic_gold" not in server.tools
    assert "transcript_semantic_benchmark" in cli._mcp_callables()

    result = server.tools["transcript_semantic_benchmark"](
        str(FIXTURE),
        str(pack),
        str(validation),
        full_body_path=str(full_body),
        smart_summary_path=str(smart_summary),
    )

    assert result["ok"] is True
    assert result["counts"]["gold_cases"] == 13

    args_json = _write_json(
        tmp_path / "mcp-args.json",
        {
            "gold_json": str(FIXTURE),
            "candidate_pack_json": str(pack),
            "validation_json": str(validation),
            "full_body_path": str(full_body),
            "smart_summary_path": str(smart_summary),
        },
    )
    local_dispatch = cli.run_mcp_call("transcript_semantic_benchmark", args_json)
    assert local_dispatch["ok"] is True
