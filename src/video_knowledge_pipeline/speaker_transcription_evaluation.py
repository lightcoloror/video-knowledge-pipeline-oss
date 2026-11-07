from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .speaker_diarization_evaluation import _anonymous_mapping

SCHEMA = "video_knowledge_pipeline.speaker_transcription_evaluation.v1"
_MEETEVAL_COMMIT = "184ff17eb77fd6db4aba27a9e303a6a3edb09364"


def evaluate_speaker_transcription_tokens(
    reference_rows: Sequence[Mapping[str, Any]],
    hypothesis_rows: Sequence[Mapping[str, Any]],
    *,
    max_cp_token_error_rate: float = 0.05,
    max_tcp_token_error_rate: float = 0.05,
    collar_seconds: float = 1.0,
    token_unit: str = "character",
    required: bool = False,
    runtime_loader: Callable[[], tuple[Callable[..., Any], Callable[..., Any]]]
    | None = None,
) -> dict[str, Any]:
    """Evaluate speaker-attributed tokens with MeetEval cpWER and tcpWER.

    Intent: detect words or characters assigned to the wrong anonymous speaker,
    even when the overall transcript text and diarization timeline both look
    complete.
    Decision: call MeetEval's official minimum-permutation cpWER and
    time-constrained tcpWER implementations through a lazy optional runtime.
    Reason: the required assignment, Levenshtein, timing and speaker-count
    accounting are mature upstream algorithms and must not be reimplemented in
    VKP.
    Evidence: pinned MeetEval commit
    184ff17eb77fd6db4aba27a9e303a6a3edb09364 builds on Windows/Python 3.12
    with its C++20 fix; 30 official cpWER/tcpWER tests pass locally.
    Effective scope: evaluation reports only. Input tokens and raw speaker
    labels are omitted from the report, and no transcript or production route
    is modified.
    """

    cp_threshold = _threshold(
        max_cp_token_error_rate,
        name="max_cp_token_error_rate",
    )
    tcp_threshold = _threshold(
        max_tcp_token_error_rate,
        name="max_tcp_token_error_rate",
    )
    collar = float(collar_seconds)
    if collar < 0.0:
        raise ValueError("collar_seconds must be non-negative")

    reference = _token_rows(reference_rows)
    hypothesis = _token_rows(hypothesis_rows)
    reference_timed = [row for row in reference if row["end"] > row["start"]]
    hypothesis_timed = [row for row in hypothesis if row["end"] > row["start"]]
    base = {
        "schema": SCHEMA,
        "evaluation_only": True,
        "required": bool(required),
        "token_unit": str(token_unit),
        "source": {
            "project": "MeetEval",
            "commit": _MEETEVAL_COMMIT,
            "algorithms": [
                "cp_word_error_rate",
                "tcp_word_error_rate",
            ],
        },
        "thresholds_exclusive": {
            "cp_token_error_rate": cp_threshold,
            "tcp_token_error_rate": tcp_threshold,
        },
        "collar_seconds": collar,
        "reference": _row_shape(reference_rows, reference_timed),
        "hypothesis": _row_shape(hypothesis_rows, hypothesis_timed),
        "speaker_labels_are_anonymous": True,
        "speaker_roles_are_not_inferred": True,
        "transcript_text_and_tokens_are_not_included": True,
    }
    if not reference or not hypothesis:
        return {
            **base,
            "status": "not_evaluated_missing_speaker_tokens",
            "passed": not required,
            "blocker": "speaker and normalized tokens are required for both inputs",
        }
    if not reference_timed or not hypothesis_timed:
        return {
            **base,
            "status": "not_evaluated_missing_positive_duration_speaker_tokens",
            "passed": not required,
            "blocker": "positive-duration speaker token rows are required for tcpWER",
        }

    loader = runtime_loader or _load_meeteval_runtime
    try:
        cp_word_error_rate, tcp_word_error_rate = loader()
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            **base,
            "status": "runtime_not_ready",
            "passed": not required,
            "blocker": f"optional_dependency_missing:{_missing_module(exc)}",
            "install_extra": "evaluation",
        }

    cp_result = cp_word_error_rate(
        _speaker_token_sequences(reference),
        _speaker_token_sequences(hypothesis),
    )
    tcp_result = tcp_word_error_rate(
        _timed_segments(reference_timed),
        _timed_segments(hypothesis_timed),
        collar=collar,
    )
    cp_metric = _metric_record(cp_result, name="cp_token_error_rate")
    tcp_metric = _metric_record(tcp_result, name="tcp_token_error_rate")
    passed = (
        float(cp_metric["value"]) < cp_threshold
        and float(tcp_metric["value"]) < tcp_threshold
    )
    assignment = {
        str(hypothesis_label): str(reference_label)
        for reference_label, hypothesis_label in tuple(
            getattr(cp_result, "assignment", ())
        )
        if reference_label is not None and hypothesis_label is not None
    }
    return {
        **base,
        "status": "evaluated",
        "passed": passed,
        "cp": cp_metric,
        "tcp": tcp_metric,
        "optimal_mapping": _anonymous_mapping(
            assignment,
            reference_labels={row["speaker"] for row in reference},
            hypothesis_labels={row["speaker"] for row in hypothesis},
        ),
    }


def _threshold(value: float, *, name: str) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return threshold


def _token_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        speaker = str(row.get("speaker") or "").strip()
        raw_tokens = row.get("tokens")
        if not speaker or not isinstance(raw_tokens, Sequence) or isinstance(
            raw_tokens,
            (str, bytes),
        ):
            continue
        tokens = tuple(
            str(token)
            for token in raw_tokens
            if str(token)
        )
        if not tokens:
            continue
        try:
            start = float(row.get("start"))
            end = float(row.get("end", start))
        except (TypeError, ValueError):
            continue
        normalized.append(
            {
                "source_index": source_index,
                "speaker": speaker,
                "start": start,
                "end": max(start, end),
                "tokens": tokens,
            }
        )
    return sorted(
        normalized,
        key=lambda row: (row["start"], row["end"], row["source_index"]),
    )


def _speaker_token_sequences(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row["speaker"]), []).extend(row["tokens"])
    return grouped


def _timed_segments(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "speaker": str(row["speaker"]),
            "words": list(row["tokens"]),
            "start_time": float(row["start"]),
            "end_time": float(row["end"]),
        }
        for row in rows
    ]


def _metric_record(result: Any, *, name: str) -> dict[str, Any]:
    value = getattr(result, "error_rate", None)
    if value is None:
        raise ValueError(f"{name} has no comparable reference tokens")
    return {
        "name": name,
        "value": round(float(value), 8),
        "errors": int(getattr(result, "errors", 0)),
        "reference_tokens": int(getattr(result, "length", 0)),
        "insertions": int(getattr(result, "insertions", 0)),
        "deletions": int(getattr(result, "deletions", 0)),
        "substitutions": int(getattr(result, "substitutions", 0)),
        "missed_speakers": int(getattr(result, "missed_speaker", 0)),
        "false_alarm_speakers": int(getattr(result, "falarm_speaker", 0)),
        "scored_speakers": int(getattr(result, "scored_speaker", 0)),
    }


def _row_shape(
    source_rows: Sequence[Mapping[str, Any]],
    evaluated_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "source_row_count": len(source_rows),
        "evaluated_positive_duration_speaker_token_row_count": len(evaluated_rows),
        "speaker_count": len({str(row["speaker"]) for row in evaluated_rows}),
        "token_count": sum(len(row["tokens"]) for row in evaluated_rows),
    }


def _load_meeteval_runtime() -> tuple[Callable[..., Any], Callable[..., Any]]:
    from meeteval.wer.wer.cp import cp_word_error_rate
    from meeteval.wer.wer.time_constrained import tcp_word_error_rate

    return cp_word_error_rate, tcp_word_error_rate


def _missing_module(exc: BaseException) -> str:
    name = getattr(exc, "name", None)
    if name:
        return str(name)
    text = str(exc).strip()
    return text or exc.__class__.__name__
