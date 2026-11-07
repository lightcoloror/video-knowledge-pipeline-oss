from __future__ import annotations

from typing import Any

import pytest

from video_knowledge_pipeline import transcript_stability_evaluation as stability
from video_knowledge_pipeline.speaker_diarization_evaluation import (
    evaluate_speaker_diarization,
)


class _FakeSegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self) -> None:
        self.rows: list[tuple[float, float, int, str]] = []

    def __setitem__(self, key: tuple[_FakeSegment, int], label: str) -> None:
        segment, track = key
        self.rows.append((segment.start, segment.end, track, label))


class _FakeDiarizationErrorRate:
    def __init__(self, *, collar: float, skip_overlap: bool) -> None:
        assert collar == 0.0
        assert skip_overlap is False

    def optimal_mapping(
        self, reference: _FakeAnnotation, hypothesis: _FakeAnnotation
    ) -> dict[str, str]:
        assert reference.rows
        assert hypothesis.rows
        return {"Bob": "Alice", "Alice": "Bob"}

    def __call__(
        self,
        reference: _FakeAnnotation,
        hypothesis: _FakeAnnotation,
        *,
        detailed: bool,
    ) -> dict[str, float]:
        assert detailed is True
        assert reference.rows
        assert hypothesis.rows
        return {
            "diarization error rate": 0.0,
            "total": 10.0,
            "correct": 10.0,
            "confusion": 0.0,
            "false alarm": 0.0,
            "missed detection": 0.0,
        }


def _fake_runtime() -> tuple[Any, Any, Any]:
    return _FakeAnnotation, _FakeSegment, _FakeDiarizationErrorRate


def test_anonymous_label_permutation_passes_without_leaking_names() -> None:
    reference = [
        {"start": 0, "end": 5, "text": "甲", "speaker": "Alice"},
        {"start": 5, "end": 10, "text": "乙", "speaker": "Bob"},
    ]
    hypothesis = [
        {"start": 0, "end": 5, "text": "甲", "speaker": "Bob"},
        {"start": 5, "end": 10, "text": "乙", "speaker": "Alice"},
    ]

    result = evaluate_speaker_diarization(
        reference,
        hypothesis,
        required=True,
        runtime_loader=_fake_runtime,
    )

    assert result["status"] == "evaluated"
    assert result["passed"] is True
    assert result["value"] == 0.0
    assert len(result["optimal_mapping"]) == 2
    assert "Alice" not in str(result)
    assert "Bob" not in str(result)
    assert result["transcript_text_is_not_included"] is True


def test_missing_speaker_rows_fail_only_when_required() -> None:
    without_speaker = [{"start": 0, "end": 5, "text": "内容"}]

    optional = evaluate_speaker_diarization(
        without_speaker,
        without_speaker,
        required=False,
        runtime_loader=_fake_runtime,
    )
    required = evaluate_speaker_diarization(
        without_speaker,
        without_speaker,
        required=True,
        runtime_loader=_fake_runtime,
    )

    assert optional["status"] == "not_evaluated_missing_timed_speaker_rows"
    assert optional["passed"] is True
    assert required["passed"] is False


def test_missing_optional_runtime_fails_closed_when_required() -> None:
    def missing_runtime() -> tuple[Any, Any, Any]:
        exc = ModuleNotFoundError("No module named 'pyannote'")
        exc.name = "pyannote"
        raise exc

    rows = [{"start": 0, "end": 5, "text": "内容", "speaker": "S01"}]

    result = evaluate_speaker_diarization(
        rows,
        rows,
        required=True,
        runtime_loader=missing_runtime,
    )

    assert result["status"] == "runtime_not_ready"
    assert result["passed"] is False
    assert result["blocker"] == "optional_dependency_missing:pyannote"
    assert result["install_extra"] == "evaluation"


def test_real_pyannote_der_maps_swapped_anonymous_labels() -> None:
    pytest.importorskip("pyannote.metrics")
    rows_a = [
        {"start": 0, "end": 5, "text": "甲", "speaker": "S01"},
        {"start": 5, "end": 10, "text": "乙", "speaker": "S02"},
    ]
    rows_b = [
        {"start": 0, "end": 5, "text": "甲", "speaker": "X02"},
        {"start": 5, "end": 10, "text": "乙", "speaker": "X01"},
    ]

    result = evaluate_speaker_diarization(rows_a, rows_b, required=True)

    assert result["status"] == "evaluated"
    assert result["passed"] is True
    assert result["value"] == 0.0


def test_stability_loader_reuses_speaker_timestamp_markdown_parser(
    tmp_path,
) -> None:
    source = tmp_path / "combined.md"
    source.write_text(
        "\n".join(
            [
                "# 智能总结",
                "摘要不能进入评测逐字稿。",
                "# 逐字稿",
                "🟢 说话人1 [00:00:00]",
                "第一段。",
                "🟣 说话人2 [00:00:05]",
                "第二段。",
            ]
        ),
        encoding="utf-8",
    )

    payload, metadata = stability._load_evaluation_input(source)

    assert metadata["format"] == "speaker_timestamp_text"
    assert metadata["segment_count"] == 2
    assert [row["speaker"] for row in payload["segments"]] == [
        "说话人1",
        "说话人2",
    ]
    assert [row["text"] for row in payload["segments"]] == ["第一段。", "第二段。"]
    assert "摘要不能进入评测逐字稿" not in str(payload)


def test_stability_loader_rejects_summary_only_markdown(tmp_path) -> None:
    """Reject summary prose before it can masquerade as transcript evidence.

    Intent: keep transcript stability metrics scoped to timestamped transcript
    evidence. Decision: fail closed for Markdown without either supported
    transcript boundary. Reason: generic Markdown parsing would count headings
    and summaries as spoken content. Evidence: the accepted combined fixture
    above contains explicit speaker timestamps; this fixture intentionally does
    not. Effective scope: evaluation input loading only.
    """

    source = tmp_path / "summary-only.md"
    source.write_text(
        "# 智能总结\n\n这是一份总结，不是带时间戳的逐字稿。\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="recognized speaker-timestamp transcript headers",
    ):
        stability._load_evaluation_input(source)


def test_required_speaker_attribution_participates_in_stability_gate(
    monkeypatch,
) -> None:
    def failed_speaker_metric(*_args, **_kwargs):
        return {
            "schema": "speaker-test",
            "status": "evaluated",
            "passed": False,
        }

    monkeypatch.setattr(
        stability, "evaluate_speaker_diarization", failed_speaker_metric
    )
    payload = {
        "segments": [
            {
                "start": 0,
                "end": 5,
                "text": "完全相同的正文",
                "speaker": "S01",
            }
        ]
    }

    result = stability.evaluate_transcript_stability(
        payload,
        payload,
        require_speaker_attribution=True,
    )

    assert result["metric"]["value"] == 0.0
    assert result["status"] == "failed"
    assert result["gates"]["speaker_attribution"] is False
    assert "speaker_attribution_evaluated" in result["diagnostic_statuses"]
