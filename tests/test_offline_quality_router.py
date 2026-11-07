from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.cli import _mcp_callables
from video_knowledge_pipeline.offline_quality_router import offline_quality_route


FIXTURES = Path(__file__).parent / "fixtures" / "offline_quality_router" / "scenarios.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_transcripts(bundle: Path) -> None:
    raw = {
        "segments": [
            {"start": 0, "end": 5, "text": "大家好今天讲获客"},
            {"start": 5, "end": 10, "text": "首先建立信任"},
            {"start": 10, "end": 15, "text": "然后确认需求"},
            {"start": 15, "end": 20, "text": "最后约定下一步"},
        ]
    }
    postprocessed = {
        "segments": [
            {"start": 0, "end": 10, "text": "大家好，今天讲获客。首先，建立信任。"},
            {"start": 10, "end": 20, "text": "然后，确认需求。最后，约定下一步。"},
        ]
    }
    _write_json(bundle / "normalized-transcript.json", raw)
    _write_json(bundle / "postprocessed-transcript.json", postprocessed)
    _write_json(bundle / "corrected-transcript.json", postprocessed)


def test_offline_quality_route_is_available_to_agent_dispatch() -> None:
    assert "offline_quality_route" in _mcp_callables()


def test_offline_quality_route_quantifies_text_preservation_and_review_state(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_json(bundle / "manifest.json", {})
    _write_transcripts(bundle)
    _write_json(
        bundle / "timeline.json",
        [
            {"index": 1, "start": 0, "end": 10, "visual_route": "document_visual"},
            {"index": 2, "start": 10, "end": 20, "visual_route": "semantic_frame"},
        ],
    )
    benchmark = tmp_path / "benchmark" / "quality-benchmark-manifest.json"
    _write_json(
        benchmark,
        {
            "samples": [
                {
                    "sample_id": "s1",
                    "asr_draft_text": "ASR prefill",
                    "reference_text": "",
                    "human_review_status": "asr_prefilled_todo",
                }
            ]
        },
    )
    (benchmark.parent / "quality-benchmark-review.html").write_text("<html></html>", encoding="utf-8")

    first = offline_quality_route(bundle, benchmark_manifest=benchmark, write=False)
    second = offline_quality_route(bundle, benchmark_manifest=benchmark, write=False)

    assert first == second
    assert first["comparisons"]["normalized_to_postprocessed"]["content_exact"] is True
    assert first["comparisons"]["normalized_to_postprocessed"]["only_punctuation_or_segmentation_changed"] is True
    assert first["stages"]["punctuation"]["stage_quality"] == "good"
    assert first["stages"]["ocr"]["stage_quality"] == "missing"
    assert first["stages"]["vision"]["stage_quality"] == "stopped_by_design"
    review = first["review_page_machine_summary"]
    assert review["review_page_exists"] is True
    assert review["asr_prefilled_count"] == 1
    assert review["human_reference_count"] == 0
    assert review["content_reviewed"] is False
    assert all(action["cloud_allowed"] is False for action in first["routing_proposal"]["actions"])
    assert all(action["auto_execute"] is False for action in first["routing_proposal"]["actions"])


def test_offline_quality_route_writes_machine_readable_artifacts(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_json(bundle / "manifest.json", {})
    _write_transcripts(bundle)
    _write_json(bundle / "timeline.json", [])
    output = tmp_path / "report"

    result = offline_quality_route(bundle, output_dir=output, write=True)

    assert result["operator_boundary"]["local_only"] is True
    assert (output / "offline-quality-route.json").exists()
    assert (output / "routing-proposal.json").exists()
    assert (output / "review-page-machine-summary.json").exists()
    assert "Review page existence means reviewed: false" in (output / "offline-quality-route.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("scenario", json.loads(FIXTURES.read_text(encoding="utf-8"))["scenarios"])
def test_offline_quality_route_failure_fixtures(tmp_path: Path, scenario: dict[str, object]) -> None:
    bundle = tmp_path / str(scenario["name"])
    bundle.mkdir()
    _write_json(bundle / "manifest.json", scenario["manifest"])
    _write_json(bundle / "timeline.json", scenario["timeline"])
    if scenario["name"] != "asr_missing":
        _write_transcripts(bundle)

    result = offline_quality_route(bundle, write=False)

    expected = scenario["expected"]
    stage = result["stages"][expected["stage"]]
    assert stage["stage_quality"] == expected["stage_quality"]
    assert expected["failure_class"] in stage["failure_class"]
    assert stage["fallback"] == expected["fallback"]
    assert stage["cloud_allowed"] is False
    assert stage["auto_execute"] is False