from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from video_knowledge_pipeline import transcript_stability_evaluation as stability
from video_knowledge_pipeline.speaker_transcription_evaluation import (
    evaluate_speaker_transcription_tokens,
)


def _result(
    *,
    error_rate: float,
    errors: int,
    length: int,
    assignment: tuple[tuple[str | None, str | None], ...],
) -> SimpleNamespace:
    return SimpleNamespace(
        error_rate=error_rate,
        errors=errors,
        length=length,
        insertions=errors,
        deletions=0,
        substitutions=0,
        missed_speaker=0,
        falarm_speaker=0,
        scored_speaker=2,
        assignment=assignment,
    )


def test_reuses_upstream_metrics_without_leaking_tokens_or_labels() -> None:
    calls: dict[str, Any] = {}

    def cp_metric(reference, hypothesis):
        calls["cp"] = (reference, hypothesis)
        return _result(
            error_rate=0.0,
            errors=0,
            length=4,
            assignment=(("Alice", "cluster-b"), ("Bob", "cluster-a")),
        )

    def tcp_metric(reference, hypothesis, *, collar):
        calls["tcp"] = (reference, hypothesis, collar)
        return _result(
            error_rate=0.0,
            errors=0,
            length=4,
            assignment=(("Alice", "cluster-b"), ("Bob", "cluster-a")),
        )

    reference = [
        {
            "speaker": "Alice",
            "start": 0.0,
            "end": 1.0,
            "tokens": list("你好"),
        },
        {
            "speaker": "Bob",
            "start": 1.0,
            "end": 2.0,
            "tokens": list("收到"),
        },
    ]
    hypothesis = [
        {
            "speaker": "cluster-b",
            "start": 0.0,
            "end": 1.0,
            "tokens": list("你好"),
        },
        {
            "speaker": "cluster-a",
            "start": 1.0,
            "end": 2.0,
            "tokens": list("收到"),
        },
    ]

    report = evaluate_speaker_transcription_tokens(
        reference,
        hypothesis,
        required=True,
        runtime_loader=lambda: (cp_metric, tcp_metric),
    )

    assert report["status"] == "evaluated"
    assert report["passed"] is True
    assert report["cp"]["value"] == 0.0
    assert report["tcp"]["value"] == 0.0
    assert calls["cp"][0] == {"Alice": list("你好"), "Bob": list("收到")}
    assert calls["tcp"][2] == 1.0
    serialized = str(report)
    for sensitive in ("Alice", "Bob", "cluster-a", "cluster-b", "你好", "收到"):
        assert sensitive not in serialized


def test_missing_runtime_fails_closed_only_when_required() -> None:
    rows = [
        {
            "speaker": "speaker-1",
            "start": 0.0,
            "end": 1.0,
            "tokens": ["甲"],
        }
    ]

    def missing_runtime():
        raise ModuleNotFoundError("No module named 'meeteval'", name="meeteval")

    optional = evaluate_speaker_transcription_tokens(
        rows,
        rows,
        required=False,
        runtime_loader=missing_runtime,
    )
    required = evaluate_speaker_transcription_tokens(
        rows,
        rows,
        required=True,
        runtime_loader=missing_runtime,
    )

    assert optional["status"] == "runtime_not_ready"
    assert optional["passed"] is True
    assert required["status"] == "runtime_not_ready"
    assert required["passed"] is False
    assert required["blocker"] == "optional_dependency_missing:meeteval"


def test_missing_speaker_tokens_fail_closed_only_when_required() -> None:
    rows = [{"speaker": "", "start": 0.0, "end": 1.0, "tokens": ["甲"]}]

    optional = evaluate_speaker_transcription_tokens(rows, rows, required=False)
    required = evaluate_speaker_transcription_tokens(rows, rows, required=True)

    assert optional["status"] == "not_evaluated_missing_speaker_tokens"
    assert optional["passed"] is True
    assert required["status"] == "not_evaluated_missing_speaker_tokens"
    assert required["passed"] is False


def test_real_meeteval_cp_and_tcp_cer_map_swapped_anonymous_labels() -> None:
    pytest.importorskip("meeteval")
    reference = [
        {
            "speaker": "reference-a",
            "start": 0.0,
            "end": 1.0,
            "tokens": list("根据情况"),
        },
        {
            "speaker": "reference-b",
            "start": 1.0,
            "end": 2.0,
            "tokens": list("会议纪要"),
        },
    ]
    hypothesis = [
        {
            "speaker": "hypothesis-y",
            "start": 0.0,
            "end": 1.0,
            "tokens": list("根据情况"),
        },
        {
            "speaker": "hypothesis-x",
            "start": 1.0,
            "end": 2.0,
            "tokens": list("会议纪要"),
        },
    ]

    report = evaluate_speaker_transcription_tokens(
        reference,
        hypothesis,
        required=True,
    )

    assert report["passed"] is True
    assert report["cp"]["value"] == 0.0
    assert report["tcp"]["value"] == 0.0
    assert len(report["optimal_mapping"]) == 2


def test_thresholds_and_positive_duration_are_validated() -> None:
    row = {
        "speaker": "speaker-1",
        "start": 0.0,
        "end": 0.0,
        "tokens": ["甲"],
    }

    with pytest.raises(ValueError, match="max_cp_token_error_rate"):
        evaluate_speaker_transcription_tokens(
            [row],
            [row],
            max_cp_token_error_rate=1.1,
        )
    with pytest.raises(ValueError, match="collar_seconds"):
        evaluate_speaker_transcription_tokens(
            [row],
            [row],
            collar_seconds=-1.0,
        )
    report = evaluate_speaker_transcription_tokens(
        [row],
        [row],
        required=True,
    )
    assert (
        report["status"]
        == "not_evaluated_missing_positive_duration_speaker_tokens"
    )
    assert report["passed"] is False


def test_required_speaker_transcription_participates_in_stability_gate(
    monkeypatch,
) -> None:
    """Keep speaker-text failures independent from overall text and DER.

    Intent: prove the new metric can block a transcript whose combined text is
    unchanged. Decision: inject the mature-metric adapter boundary rather than
    duplicating its algorithm in this integration fixture. Reason: this test
    targets gate wiring and normalized token projection. Evidence: the captured
    rows contain normalized character tokens and the failed adapter status is
    preserved. Effective scope: evaluation orchestration only.
    """

    captured: dict[str, Any] = {}

    def failed_speaker_text(reference_rows, hypothesis_rows, **kwargs):
        captured["reference"] = reference_rows
        captured["hypothesis"] = hypothesis_rows
        captured["kwargs"] = kwargs
        return {
            "status": "evaluated",
            "passed": False,
            "cp": {"value": 0.5},
            "tcp": {"value": 0.5},
        }

    monkeypatch.setattr(
        stability,
        "evaluate_speaker_transcription_tokens",
        failed_speaker_text,
    )
    payload = {
        "duration_seconds": 2.0,
        "segments": [
            {
                "speaker": "说话人1",
                "start": 0.0,
                "end": 1.0,
                "text": "会议纪要。",
            },
            {
                "speaker": "说话人2",
                "start": 1.0,
                "end": 2.0,
                "text": "发了一份材料！",
            },
        ],
    }

    report = stability.evaluate_transcript_stability(
        payload,
        payload,
        require_speaker_transcription=True,
    )

    assert report["status"] == "failed"
    assert report["gates"]["speaker_transcription"] is False
    assert report["diagnostic_statuses"] == ["speaker_transcription_evaluated"]
    assert captured["reference"][0]["tokens"] == list("会议纪要")
    assert captured["reference"][1]["tokens"] == list("发了一份材料")
    assert captured["kwargs"]["required"] is True


def test_real_meeteval_stability_gate_passes_identical_speaker_transcript() -> None:
    pytest.importorskip("meeteval")
    payload = {
        "duration_seconds": 2.0,
        "segments": [
            {
                "speaker": "说话人1",
                "start": 0.0,
                "end": 1.0,
                "text": "根据排期来的嘛。",
            },
            {
                "speaker": "说话人2",
                "start": 1.0,
                "end": 2.0,
                "text": "星河系统。",
            },
        ],
    }

    report = stability.evaluate_transcript_stability(
        payload,
        payload,
        require_speaker_transcription=True,
    )

    assert report["status"] == "passed"
    assert report["speaker_transcription"]["cp"]["value"] == 0.0
    assert report["speaker_transcription"]["tcp"]["value"] == 0.0
    assert report["gates"]["speaker_transcription"] is True


def test_cli_exposes_explicit_speaker_transcription_gate() -> None:
    args = stability.build_parser().parse_args(
        [
            "candidate.txt",
            "reference.txt",
            "report.json",
            "--require-speaker-transcription",
            "--max-cp-speaker-character-error-rate",
            "0.1",
            "--max-tcp-speaker-character-error-rate",
            "0.2",
            "--speaker-transcription-collar-seconds",
            "1.5",
        ]
    )

    assert args.require_speaker_transcription is True
    assert args.max_cp_speaker_character_error_rate == 0.1
    assert args.max_tcp_speaker_character_error_rate == 0.2
    assert args.speaker_transcription_collar_seconds == 1.5
