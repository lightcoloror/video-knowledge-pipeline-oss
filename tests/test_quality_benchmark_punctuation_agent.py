from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.quality_benchmark_punctuation_agent import (
    build_quality_benchmark_punctuation_agent_pack,
    evaluate_quality_benchmark_punctuation_agent,
)


def _write_transcript(path: Path, text: str) -> str:
    path.write_text(
        json.dumps(
            {"segments": [{"index": 1, "start": 0.0, "end": 10.0, "text": text}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return str(path)


def _manifest(tmp_path: Path, *, sample_count: int = 1) -> Path:
    samples = []
    for index in range(sample_count):
        source = _write_transcript(
            tmp_path / f"source-{index}.json",
            "大家好，今天讲明亚保险获客。首先建立信任。",
        )
        samples.append(
            {
                "sample_id": f"sample-{index + 1:02d}",
                "category": "company_names",
                "start_seconds": index * 10,
                "end_seconds": (index + 1) * 10,
                "reference_text": "大家好，今天讲明亚保险获客。首先，建立信任。",
                "protected_entities": ["明亚保险"],
                "variants": {"sensevoice_full_punc": source},
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"samples": samples}, ensure_ascii=False), encoding="utf-8")
    return path


def _write_decisions(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.quality_benchmark_punctuation_agent_decisions.v1",
                "rows": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_public_pack_and_template_never_leak_reference(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    result = build_quality_benchmark_punctuation_agent_pack(manifest, write=True)

    pack_json = (tmp_path / "quality-benchmark-punctuation-agent-pack.json").read_text(encoding="utf-8")
    pack_md = (tmp_path / "quality-benchmark-punctuation-agent-pack.md").read_text(encoding="utf-8")
    todo_json = (tmp_path / "quality-benchmark-punctuation-agent.todo.json").read_text(encoding="utf-8")
    for public_artifact in (pack_json, pack_md, todo_json):
        assert "reference_text" not in public_artifact
        assert "大家好，今天讲明亚保险获客。首先，建立信任。" not in public_artifact
        assert str(manifest) not in public_artifact
    assert result["samples"][0]["input_text"] == "大家好，今天讲明亚保险获客。首先建立信任。"
    assert result["input_mode"] == "baseline_preserving_minimal_repair"
    assert result["operator_boundary"]["human_reference_excluded"] is True
    todo = json.loads(todo_json)
    assert todo["rows"][0]["candidate_text"] == ""


def test_evaluator_accepts_character_locked_punctuation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    pack = build_quality_benchmark_punctuation_agent_pack(manifest, write=False)
    sample = pack["samples"][0]
    decisions = _write_decisions(
        tmp_path / "decisions.json",
        [
            {
                "sample_id": sample["sample_id"],
                "input_sha256": sample["input_sha256"],
                "candidate_text": "大家好，今天讲明亚保险获客。\n\n首先，建立信任。",
            }
        ],
    )

    result = evaluate_quality_benchmark_punctuation_agent(
        manifest,
        decisions,
        write=True,
    )

    assert result["status"] == "completed"
    assert result["accepted_count"] == 1
    assert result["rows"][0]["status"] == "accepted"
    assert result["rows"][0]["char_lock_passed"] is True
    assert result["metrics"]["content_character_change_rate"] == 0.0
    assert result["metrics"]["candidate"]["punctuation_f1"] == 1.0
    assert result["metrics"]["candidate"]["sentence_boundary_f1"] == 1.0
    assert result["metrics"]["candidate"]["cer"] == 0.0


def test_evaluator_rejects_any_content_character_change(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    decisions = _write_decisions(
        tmp_path / "decisions.json",
        [
            {
                "sample_id": "sample-01",
                "candidate_text": "大家好，今天讲米亚保险获客。首先，建立信任。",
            }
        ],
    )

    result = evaluate_quality_benchmark_punctuation_agent(manifest, decisions, write=False)

    row = result["rows"][0]
    assert result["status"] == "completed_with_blockers"
    assert row["status"] == "rejected_character_change"
    assert row["char_lock_passed"] is False
    assert row["content_character_change_rate"] > 0
    assert row["candidate"] is None
    assert result["metrics"]["candidate"]["sample_count"] == 0


def test_evaluator_reports_missing_decision_rows(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, sample_count=2)
    decisions = _write_decisions(
        tmp_path / "decisions.json",
        [
            {
                "sample_id": "sample-01",
                "candidate_text": "大家好，今天讲明亚保险获客。首先，建立信任。",
            }
        ],
    )

    result = evaluate_quality_benchmark_punctuation_agent(manifest, decisions, write=False)

    assert result["status"] == "completed_with_blockers"
    assert result["accepted_count"] == 1
    assert result["blocker_count"] == 1
    assert result["rows"][1]["sample_id"] == "sample-02"
    assert result["rows"][1]["status"] == "missing_decision"
    assert result["blockers"] == [
        {
            "sample_id": "sample-02",
            "status": "missing_decision",
            "reason": "decision_row_missing",
        }
    ]


def test_cli_parses_punctuation_agent_actions() -> None:
    from video_knowledge_pipeline.cli import build_parser

    parser = build_parser()
    build = parser.parse_args(["quality-benchmark", "build-punctuation-agent", "manifest.json"])
    evaluate = parser.parse_args(["quality-benchmark", "evaluate-punctuation-agent", "manifest.json", "--decisions-json", "decisions.json"])

    assert build.action == "build-punctuation-agent"
    assert evaluate.action == "evaluate-punctuation-agent"
    assert evaluate.decisions_json == "decisions.json"


def test_unicode_quotes_are_allowed_by_character_lock(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    pack = build_quality_benchmark_punctuation_agent_pack(manifest, write=False)
    sample = pack["samples"][0]
    candidate = sample["input_text"].replace("首先", "“首先”")
    decisions = _write_decisions(tmp_path / "decisions.json", [{"sample_id": sample["sample_id"], "input_sha256": sample["input_sha256"], "candidate_text": candidate}])
    result = evaluate_quality_benchmark_punctuation_agent(manifest, decisions, write=False)
    assert result["rows"][0]["status"] == "accepted"
    assert result["metrics"]["content_character_change_rate"] == 0.0
