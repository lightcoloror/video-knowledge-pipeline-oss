from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .models import TranscriptCue, now_iso
from .punctuation_model_stage import _content_only, _run_model_blocks
from .quality_benchmark import transcript_quality_metrics
from .storage import read_json, write_json
from .transcript import parse_transcript


SCHEMA = "video_knowledge_pipeline.quality_benchmark_punctuation.v1"
_PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:\s]+")


def run_quality_benchmark_punctuation(
    manifest_json: str | Path,
    *,
    output_dir: str | Path | None = None,
    source_variant: str = "sensevoice_full_punc",
    model: str = "ct-punc",
    device: str = "auto",
    execute: bool = False,
    write: bool = True,
    model_runner: Callable[..., tuple[list[str], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """A/B standalone punctuation restoration on the fixed human-gold windows."""

    manifest_path = Path(manifest_json).expanduser().resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("quality benchmark manifest must be a JSON object")
    if execute and not write:
        raise ValueError("punctuation benchmark execution requires write=true")
    out = Path(output_dir).expanduser().resolve() if output_dir else manifest_path.parent
    prepared: list[dict[str, Any]] = []
    blocks: list[list[TranscriptCue]] = []
    missing: list[dict[str, str]] = []
    for sample in manifest.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "")
        variants = sample.get("variants") if isinstance(sample.get("variants"), dict) else {}
        source_path = Path(str(variants.get(source_variant) or "")).expanduser()
        reference = str(sample.get("reference_text") or "")
        if not source_path.exists() or not reference:
            missing.append({"sample_id": sample_id, "reason": "source_or_reference_missing"})
            continue
        source_text = "".join(str(cue.text or "") for cue in parse_transcript(source_path))
        model_input = _strip_punctuation(source_text)
        prepared.append(
            {
                "sample": sample,
                "sample_id": sample_id,
                "source_path": str(source_path.resolve()),
                "source_text": source_text,
                "model_input": model_input,
                "reference": reference,
            }
        )
        blocks.append(
            [
                TranscriptCue(
                    start=float(sample.get("start_seconds") or 0.0),
                    end=float(sample.get("end_seconds") or 0.0),
                    text=model_input,
                )
            ]
        )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "manifest_json": str(manifest_path),
        "source_variant": source_variant,
        "model": model,
        "device": device,
        "execute": bool(execute),
        "status": "planned" if not execute else "running",
        "sample_count": len(prepared),
        "missing": missing,
        "rows": [],
        "operator_boundary": {
            "local_only": True,
            "human_reference_evaluation_only": True,
            "reference_not_sent_to_model": True,
            "content_character_lock_required": True,
            "does_not_promote_transcript": True,
        },
        "updated_at": now_iso(),
    }
    if not execute:
        if write:
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "quality-benchmark-punctuation.json", result)
            (out / "quality-benchmark-punctuation.md").write_text(_render(result), encoding="utf-8")
        return result

    runner = model_runner or _run_model_blocks
    try:
        outputs, runner_info = runner(out, blocks, model=model, device=device)
    except Exception as exc:
        result.update({"status": "model_not_ready", "error": f"{type(exc).__name__}: {exc}"})
        if write:
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "quality-benchmark-punctuation.json", result)
            (out / "quality-benchmark-punctuation.md").write_text(_render(result), encoding="utf-8")
        return result

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(prepared):
        candidate = str(outputs[index] or "") if index < len(outputs) else ""
        sample = item["sample"]
        protected = [str(value) for value in sample.get("protected_entities") or [] if str(value)]
        char_lock = bool(candidate) and _content_only(item["model_input"]) == _content_only(candidate)
        baseline = transcript_quality_metrics(
            item["reference"], item["source_text"], protected_entities=protected
        )
        candidate_metrics = (
            transcript_quality_metrics(
                item["reference"],
                candidate,
                source_text=item["source_text"],
                protected_entities=protected,
            )
            if candidate
            else {}
        )
        row = {
            "sample_id": item["sample_id"],
            "source_path": item["source_path"],
            "char_lock_passed": char_lock,
            "content_character_mutation_rate": 0.0 if char_lock else _mutation_rate(item["model_input"], candidate),
            "baseline": baseline,
            "candidate": candidate_metrics,
            "candidate_text": candidate,
        }
        rows.append(row)
        if write and candidate and char_lock:
            sample_dir = out / "punctuation-runs" / _safe_name(item["sample_id"])
            write_json(
                sample_dir / "punctuated-transcript.json",
                {
                    "schema": "video_knowledge_pipeline.benchmark_punctuated_transcript.v1",
                    "source_path": item["source_path"],
                    "model": model,
                    "character_lock": {"passed": True},
                    "segments": [
                        {
                            "index": 1,
                            "start": float(sample.get("start_seconds") or 0.0),
                            "end": float(sample.get("end_seconds") or 0.0),
                            "text": candidate,
                        }
                    ],
                },
            )
            row["candidate_path"] = str(sample_dir / "punctuated-transcript.json")

    char_lock_passed = bool(rows) and all(bool(row["char_lock_passed"]) for row in rows)
    result.update(
        {
            "status": "completed" if char_lock_passed and not missing else "completed_with_blockers",
            "runner": runner_info,
            "rows": rows,
            "char_lock_passed": char_lock_passed,
            "metrics": {
                "baseline": _aggregate(rows, "baseline"),
                "candidate": _aggregate(rows, "candidate"),
                "content_character_mutation_rate": round(
                    sum(float(row["content_character_mutation_rate"]) for row in rows) / max(1, len(rows)), 6
                ),
            },
            "updated_at": now_iso(),
        }
    )
    baseline = result["metrics"]["baseline"]
    candidate = result["metrics"]["candidate"]
    result["improvement"] = {
        "punctuation_f1": _delta(candidate, baseline, "punctuation_f1"),
        "sentence_boundary_f1": _delta(candidate, baseline, "sentence_boundary_f1"),
        "cer": _delta(candidate, baseline, "cer"),
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "quality-benchmark-punctuation.json", result)
        (out / "quality-benchmark-punctuation.md").write_text(_render(result), encoding="utf-8")
    return result


def _strip_punctuation(text: str) -> str:
    return _PUNCT_RE.sub("", str(text or ""))


def _mutation_rate(source: str, candidate: str) -> float:
    left = _content_only(source)
    right = _content_only(candidate)
    if left == right:
        return 0.0
    import difflib

    return round(1.0 - difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio(), 6)


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows if isinstance(row.get(key), dict) and row[key]]
    keys = ("cer", "punctuation_f1", "sentence_boundary_f1", "entity_accuracy")
    result: dict[str, Any] = {"sample_count": len(values)}
    for metric in keys:
        usable = [float(value[metric]) for value in values if value.get(metric) is not None]
        result[metric] = round(sum(usable) / len(usable), 6) if usable else None
    result["number_error_count"] = sum(int(value.get("number_error_count") or 0) for value in values)
    return result


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    if candidate.get(key) is None or baseline.get(key) is None:
        return None
    return round(float(candidate[key]) - float(baseline[key]), 6)


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^\w.-]+", "-", str(value or "").strip(), flags=re.UNICODE).strip("-._")
    return clean or "sample"


def _render(result: dict[str, Any]) -> str:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    baseline = metrics.get("baseline") if isinstance(metrics.get("baseline"), dict) else {}
    candidate = metrics.get("candidate") if isinstance(metrics.get("candidate"), dict) else {}
    improvement = result.get("improvement") if isinstance(result.get("improvement"), dict) else {}
    return "\n".join(
        [
            "# 24 段标点与句界 A/B",
            "",
            f"- 状态：`{result.get('status')}`",
            f"- 样本：`{result.get('sample_count')}`",
            f"- 模型：`{result.get('model')}`",
            f"- 字符锁：`{result.get('char_lock_passed')}`",
            f"- 原标点 F1：`{baseline.get('punctuation_f1')}`",
            f"- 新标点 F1：`{candidate.get('punctuation_f1')}`，变化 `{improvement.get('punctuation_f1')}`",
            f"- 原句界 F1：`{baseline.get('sentence_boundary_f1')}`",
            f"- 新句界 F1：`{candidate.get('sentence_boundary_f1')}`，变化 `{improvement.get('sentence_boundary_f1')}`",
            f"- 内容字符变更率：`{metrics.get('content_character_mutation_rate')}`",
            "",
            "> 人工参考稿仅由评估器读取，不传给标点模型，也不进入生产纠错证据。",
            "",
        ]
    )
