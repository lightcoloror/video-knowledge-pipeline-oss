from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any


SCHEMA = "video_knowledge_pipeline.speaker_diarization_evaluation.v1"
_PYANNOTE_COMMIT = "e8000509ee06331ef3e0fec08fa3605af834efbb"


def evaluate_speaker_diarization(
    reference_rows: Sequence[Mapping[str, Any]],
    hypothesis_rows: Sequence[Mapping[str, Any]],
    *,
    max_diarization_error_rate: float = 0.05,
    required: bool = False,
    runtime_loader: Callable[[], tuple[Any, Any, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate anonymous speaker attribution with upstream pyannote.metrics.

    Intent: detect speaker swaps and timing-attribution errors that a text-only
    transcript comparison cannot see.
    Decision: call pyannote.metrics ``DiarizationErrorRate`` and its Hungarian
    optimal mapping through a lazy optional runtime.
    Reason: anonymous ``说话人1``/``说话人2`` identifiers may be permuted
    without being wrong, while a home-grown exact-label comparison would
    mis-score them.
    Evidence: pinned upstream pyannote.metrics commit
    e8000509ee06331ef3e0fec08fa3605af834efbb; its official diarization and
    identification tests pass 12/12 in the isolated review environment.
    Effective scope: evaluation reports only. No transcript text, speaker role,
    production routing, ASR output, or correction decision is modified.
    """

    threshold = float(max_diarization_error_rate)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("max_diarization_error_rate must be between 0 and 1")

    reference = _timed_speaker_rows(reference_rows)
    hypothesis = _timed_speaker_rows(hypothesis_rows)
    base = {
        "schema": SCHEMA,
        "evaluation_only": True,
        "required": bool(required),
        "metric": "diarization_error_rate",
        "threshold_exclusive": threshold,
        "source": {
            "project": "pyannote.metrics",
            "commit": _PYANNOTE_COMMIT,
            "algorithm": "DiarizationErrorRate + Hungarian optimal mapping",
        },
        "reference": _row_shape(reference_rows, reference),
        "hypothesis": _row_shape(hypothesis_rows, hypothesis),
        "speaker_labels_are_anonymous": True,
        "speaker_roles_are_not_inferred": True,
        "transcript_text_is_not_included": True,
    }
    if not reference or not hypothesis:
        return {
            **base,
            "status": "not_evaluated_missing_timed_speaker_rows",
            "passed": not required,
            "blocker": "timed speaker rows are required for both inputs",
        }

    loader = runtime_loader or _load_pyannote_runtime
    try:
        Annotation, Segment, DiarizationErrorRate = loader()
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            **base,
            "status": "runtime_not_ready",
            "passed": not required,
            "blocker": f"optional_dependency_missing:{_missing_module(exc)}",
            "install_extra": "evaluation",
        }

    reference_annotation = _annotation(reference, Annotation, Segment)
    hypothesis_annotation = _annotation(hypothesis, Annotation, Segment)
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    mapping = metric.optimal_mapping(reference_annotation, hypothesis_annotation)
    details = metric(reference_annotation, hypothesis_annotation, detailed=True)
    rate = float(details["diarization error rate"])
    passed = rate < threshold
    return {
        **base,
        "status": "evaluated",
        "passed": passed,
        "value": round(rate, 8),
        "components_seconds": {
            "total": round(float(details.get("total", 0.0)), 8),
            "correct": round(float(details.get("correct", 0.0)), 8),
            "confusion": round(float(details.get("confusion", 0.0)), 8),
            "false_alarm": round(float(details.get("false alarm", 0.0)), 8),
            "missed_detection": round(
                float(details.get("missed detection", 0.0)), 8
            ),
        },
        "optimal_mapping": _anonymous_mapping(
            mapping,
            reference_labels={row["speaker"] for row in reference},
            hypothesis_labels={row["speaker"] for row in hypothesis},
        ),
    }


def _timed_speaker_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            start = float(row.get("start"))
            end = float(row.get("end", start))
        except (TypeError, ValueError):
            continue
        speaker = _speaker(row)
        if not speaker or end <= start:
            continue
        normalized.append({"start": start, "end": end, "speaker": speaker})
    return normalized


def _speaker(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return str(
        row.get("speaker")
        or row.get("speaker_id")
        or metadata.get("speaker")
        or metadata.get("speaker_id")
        or ""
    ).strip()


def _annotation(rows: Sequence[Mapping[str, Any]], Annotation: Any, Segment: Any) -> Any:
    annotation = Annotation()
    for index, row in enumerate(rows):
        annotation[Segment(float(row["start"]), float(row["end"])), index] = str(
            row["speaker"]
        )
    return annotation


def _anonymous_mapping(
    mapping: Mapping[str, str],
    *,
    reference_labels: set[str],
    hypothesis_labels: set[str],
) -> list[dict[str, str]]:
    reference_aliases = _label_aliases(reference_labels, "reference_speaker")
    hypothesis_aliases = _label_aliases(hypothesis_labels, "hypothesis_speaker")
    return [
        {
            "hypothesis": hypothesis_aliases[str(hypothesis)],
            "reference": reference_aliases[str(reference)],
        }
        for hypothesis, reference in sorted(
            mapping.items(), key=lambda item: (str(item[0]), str(item[1]))
        )
    ]


def _label_aliases(labels: set[str], prefix: str) -> dict[str, str]:
    return {
        label: f"{prefix}_{index}_{hashlib.sha256(label.encode('utf-8')).hexdigest()[:8]}"
        for index, label in enumerate(sorted(labels), start=1)
    }


def _row_shape(
    source_rows: Sequence[Mapping[str, Any]],
    evaluated_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        "source_row_count": len(source_rows),
        "evaluated_positive_duration_speaker_row_count": len(evaluated_rows),
        "speaker_count": len({str(row["speaker"]) for row in evaluated_rows}),
    }


def _load_pyannote_runtime() -> tuple[Any, Any, Any]:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    return Annotation, Segment, DiarizationErrorRate


def _missing_module(exc: BaseException) -> str:
    name = str(getattr(exc, "name", "") or "").strip()
    return name or "pyannote.metrics"
