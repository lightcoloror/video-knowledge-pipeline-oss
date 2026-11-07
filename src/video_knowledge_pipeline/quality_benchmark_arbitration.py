from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .asr_diff_adjudication import _cluster_differences, _positioned_differences
from .models import now_iso
from .quality_benchmark import transcript_quality_metrics
from .storage import read_json, write_json
from .transcript import parse_transcript


PACK_SCHEMA = "video_knowledge_pipeline.quality_benchmark_arbitration_pack.v1"
PRIVATE_SCHEMA = "video_knowledge_pipeline.quality_benchmark_arbitration_private.v1"
RESULT_SCHEMA = "video_knowledge_pipeline.quality_benchmark_arbitration_result.v1"


def build_quality_benchmark_arbitration(
    manifest_json: str | Path,
    *,
    output_dir: str | Path | None = None,
    primary_variant: str = "sensevoice_full_punc",
    secondary_variant: str = "qwen3_asr_1_7b",
    write: bool = True,
) -> dict[str, Any]:
    """Build an anonymous dual-ASR review pack without human reference text."""

    manifest_path = Path(manifest_json).expanduser().resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("quality benchmark manifest must be a JSON object")
    out = Path(output_dir).expanduser().resolve() if output_dir else manifest_path.parent
    public_samples: list[dict[str, Any]] = []
    private_samples: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for sample in manifest.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "")
        variants = sample.get("variants") if isinstance(sample.get("variants"), dict) else {}
        primary_path = Path(str(variants.get(primary_variant) or "")).expanduser()
        secondary_path = Path(str(variants.get(secondary_variant) or "")).expanduser()
        if not primary_path.exists() or not secondary_path.exists():
            missing.append({"sample_id": sample_id, "reason": "variant_transcript_missing"})
            continue
        primary_text = _transcript_text(primary_path)
        secondary_text = _transcript_text(secondary_path)
        window = {
            "window_id": sample_id,
            "primary_index": 0,
            "secondary_index": 0,
            "start": float(sample.get("start_seconds") or 0.0),
            "end": float(sample.get("end_seconds") or 0.0),
            "audio_review_window": {
                "start": float(sample.get("start_seconds") or 0.0),
                "end": float(sample.get("end_seconds") or 0.0),
                "audio_clip_path": str(sample.get("audio_clip_path") or ""),
            },
        }
        differences = _positioned_differences(window, primary_text, secondary_text)
        clusters = _cluster_differences(window, differences, max_gap=6)
        public_differences = [_public_difference(row) for row in differences]
        public_samples.append(
            {
                "sample_id": sample_id,
                "category": str(sample.get("category") or ""),
                "start_seconds": window["start"],
                "end_seconds": window["end"],
                "audio_clip_path": str(sample.get("audio_clip_path") or ""),
                "difference_count": len(public_differences),
                "cluster_count": len(clusters),
                "differences": public_differences,
                "clusters": clusters,
            }
        )
        private_samples.append(
            {
                "sample_id": sample_id,
                "primary_variant": primary_variant,
                "secondary_variant": secondary_variant,
                "primary_path": str(primary_path.resolve()),
                "secondary_path": str(secondary_path.resolve()),
                "primary_text": primary_text,
                "secondary_text": secondary_text,
                "differences": differences,
            }
        )

    public_pack = {
        "schema": PACK_SCHEMA,
        "manifest_json": str(manifest_path),
        "status": "ready_for_review" if public_samples and not missing else "incomplete",
        "sample_count": len(public_samples),
        "difference_count": sum(int(row["difference_count"]) for row in public_samples),
        "cluster_count": sum(int(row["cluster_count"]) for row in public_samples),
        "samples": public_samples,
        "missing": missing,
        "decision_schema": {
            "rows": [
                {
                    "diff_id": "<diff-id>",
                    "choice": "A | B | keep_primary",
                    "confidence": "0..1",
                    "evidence_refs": ["audio clip or independent evidence path"],
                    "reason": "short evidence-based reason",
                }
            ]
        },
        "operator_boundary": {
            "candidate_sources_anonymous": True,
            "human_reference_excluded": True,
            "evaluation_truth_never_correction_evidence": True,
            "does_not_apply_patches": True,
            "local_only": True,
        },
        "updated_at": now_iso(),
    }
    private_pack = {
        "schema": PRIVATE_SCHEMA,
        "manifest_json": str(manifest_path),
        "primary_variant": primary_variant,
        "secondary_variant": secondary_variant,
        "samples": private_samples,
        "human_reference_excluded": True,
        "updated_at": public_pack["updated_at"],
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "quality-benchmark-arbitration-pack.json", public_pack)
        write_json(out / "quality-benchmark-arbitration.private.json", private_pack)
        write_json(
            out / "quality-benchmark-arbitration.todo.json",
            {
                "schema": "video_knowledge_pipeline.quality_benchmark_arbitration_decisions.v1",
                "rows": [
                    {
                        "diff_id": diff["diff_id"],
                        "choice": "",
                        "confidence": None,
                        "evidence_refs": [],
                        "reason": "",
                    }
                    for sample in public_samples
                    for diff in sample["differences"]
                ],
            },
        )
        (out / "quality-benchmark-arbitration-pack.md").write_text(
            _render_pack(public_pack), encoding="utf-8"
        )
    return public_pack


def evaluate_quality_benchmark_arbitration(
    manifest_json: str | Path,
    *,
    private_json: str | Path | None = None,
    decisions_json: str | Path | None = None,
    output_dir: str | Path | None = None,
    min_confidence: float = 0.75,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate dual-ASR arbitration; human truth is read only inside this evaluator."""

    manifest_path = Path(manifest_json).expanduser().resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("quality benchmark manifest must be a JSON object")
    out = Path(output_dir).expanduser().resolve() if output_dir else manifest_path.parent
    private_path = (
        Path(private_json).expanduser().resolve()
        if private_json
        else out / "quality-benchmark-arbitration.private.json"
    )
    private_pack = read_json(private_path)
    if not isinstance(private_pack, dict):
        raise ValueError("private arbitration pack must be a JSON object")
    sample_by_id = {
        str(row.get("sample_id") or ""): row
        for row in manifest.get("samples") or []
        if isinstance(row, dict)
    }
    decision_rows = _decision_rows(decisions_json)
    rows: list[dict[str, Any]] = []
    rejected_decisions: list[dict[str, str]] = []
    for private_sample in private_pack.get("samples") or []:
        if not isinstance(private_sample, dict):
            continue
        sample_id = str(private_sample.get("sample_id") or "")
        sample = sample_by_id.get(sample_id) or {}
        reference = str(sample.get("reference_text") or "")
        primary = str(private_sample.get("primary_text") or "")
        secondary = str(private_sample.get("secondary_text") or "")
        if not reference or not primary or not secondary:
            continue
        protected_entities = [str(value) for value in sample.get("protected_entities") or [] if str(value)]
        primary_metrics = transcript_quality_metrics(reference, primary, protected_entities=protected_entities)
        secondary_metrics = transcript_quality_metrics(reference, secondary, source_text=primary, protected_entities=protected_entities)
        whole_oracle = secondary if float(secondary_metrics.get("cer") or 1.0) < float(primary_metrics.get("cer") or 1.0) else primary
        local_oracle, oracle_count = _local_oracle(reference, primary, private_sample.get("differences") or [])
        actual, applied, rejected = _apply_reviewed_decisions(
            primary,
            private_sample.get("differences") or [],
            decision_rows,
            min_confidence=min_confidence,
        )
        rejected_decisions.extend(rejected)
        rows.append(
            {
                "sample_id": sample_id,
                "primary": primary_metrics,
                "secondary": secondary_metrics,
                "whole_sample_oracle": transcript_quality_metrics(reference, whole_oracle, source_text=primary, protected_entities=protected_entities),
                "local_patch_oracle": transcript_quality_metrics(reference, local_oracle, source_text=primary, protected_entities=protected_entities),
                "reviewed_patch": transcript_quality_metrics(reference, actual, source_text=primary, protected_entities=protected_entities) if decision_rows else None,
                "oracle_patch_count": oracle_count,
                "reviewed_patch_count": applied,
            }
        )

    result = {
        "schema": RESULT_SCHEMA,
        "manifest_json": str(manifest_path),
        "sample_count": len(rows),
        "metrics": {
            "primary": _combined_metrics(rows, "primary"),
            "secondary": _combined_metrics(rows, "secondary"),
            "whole_sample_oracle": _combined_metrics(rows, "whole_sample_oracle"),
            "local_patch_oracle": _combined_metrics(rows, "local_patch_oracle"),
            "reviewed_patch": _combined_metrics(rows, "reviewed_patch") if decision_rows else None,
        },
        "rows": rows,
        "rejected_decisions": rejected_decisions,
        "operator_boundary": {
            "oracle_is_evaluation_only_never_production": True,
            "human_reference_used_only_by_evaluator": True,
            "decision_changes_require_evidence": True,
            "no_model_default_switch": True,
        },
        "updated_at": now_iso(),
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "quality-benchmark-arbitration-result.json", result)
        (out / "quality-benchmark-arbitration-result.md").write_text(
            _render_result(result), encoding="utf-8"
        )
    return result


def _public_difference(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "diff_id": str(row.get("diff_id") or ""),
        "cluster_id": str(row.get("cluster_id") or ""),
        "operation": str(row.get("operation") or ""),
        "candidate_a": str(row.get("candidate_a_text") or ""),
        "candidate_b": str(row.get("candidate_b_text") or ""),
        "context_before": str(row.get("left_context") or ""),
        "context_after": str(row.get("right_context") or ""),
        "estimated_time": row.get("estimated_time") or {},
    }


def _transcript_text(path: Path) -> str:
    return "".join(str(cue.text or "") for cue in parse_transcript(path))


def _decision_rows(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = read_json(Path(path).expanduser().resolve())
    if not isinstance(payload, dict):
        return {}
    return {
        str(row.get("diff_id") or ""): row
        for row in payload.get("rows") or []
        if isinstance(row, dict) and row.get("diff_id")
    }


def _local_oracle(reference: str, primary: str, differences: list[Any]) -> tuple[str, int]:
    text = primary
    selected = 0
    for diff in sorted(_dict_rows(differences), key=lambda row: int(row.get("primary_char_start") or 0), reverse=True):
        start = int(diff.get("primary_char_start") or 0)
        end = int(diff.get("primary_char_end") or start)
        replacement = str(diff.get("secondary_text") or "")
        candidate = text[:start] + replacement + text[end:]
        if _cer(reference, candidate) < _cer(reference, text):
            text = candidate
            selected += 1
    return text, selected


def _apply_reviewed_decisions(
    primary: str,
    differences: list[Any],
    decisions: dict[str, dict[str, Any]],
    *,
    min_confidence: float,
) -> tuple[str, int, list[dict[str, str]]]:
    text = primary
    applied = 0
    rejected: list[dict[str, str]] = []
    for diff in sorted(_dict_rows(differences), key=lambda row: int(row.get("primary_char_start") or 0), reverse=True):
        diff_id = str(diff.get("diff_id") or "")
        decision = decisions.get(diff_id)
        if not decision:
            continue
        choice = str(decision.get("choice") or "").strip().upper()
        if choice in {"KEEP_PRIMARY", "PRIMARY", "NO_CHANGE"}:
            continue
        confidence = float(decision.get("confidence") or 0.0)
        evidence = [str(value) for value in decision.get("evidence_refs") or [] if str(value).strip()]
        if confidence < min_confidence or not evidence:
            rejected.append({"diff_id": diff_id, "reason": "confidence_or_evidence_gate"})
            continue
        primary_piece = str(diff.get("primary_text") or "")
        secondary_piece = str(diff.get("secondary_text") or "")
        candidate_a = str(diff.get("candidate_a_text") or "")
        candidate_b = str(diff.get("candidate_b_text") or "")
        replacement = candidate_a if choice == "A" else candidate_b if choice == "B" else primary_piece
        if replacement == primary_piece:
            continue
        if replacement != secondary_piece:
            rejected.append({"diff_id": diff_id, "reason": "invalid_anonymous_choice"})
            continue
        start = int(diff.get("primary_char_start") or 0)
        end = int(diff.get("primary_char_end") or start)
        if text[start:end] != primary_piece:
            rejected.append({"diff_id": diff_id, "reason": "base_span_mismatch"})
            continue
        text = text[:start] + replacement + text[end:]
        applied += 1
    return text, applied, rejected


def _combined_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows if isinstance(row.get(key), dict)]
    if not values:
        return {}
    return {
        "sample_count": len(values),
        "mean_cer": round(sum(float(row.get("cer") or 0.0) for row in values) / len(values), 6),
        "mean_punctuation_f1": round(sum(float(row.get("punctuation_f1") or 0.0) for row in values) / len(values), 6),
        "mean_sentence_boundary_f1": round(sum(float(row.get("sentence_boundary_f1") or 0.0) for row in values) / len(values), 6),
        "mean_entity_accuracy": round(sum(float(row.get("entity_accuracy") or 0.0) for row in values) / len(values), 6),
        "number_error_count": sum(int(row.get("number_error_count") or 0) for row in values),
    }


def _cer(reference: str, candidate: str) -> float:
    return float(transcript_quality_metrics(reference, candidate).get("cer") or 0.0)


def _dict_rows(rows: list[Any]) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict)]


def _render_pack(pack: dict[str, Any]) -> str:
    lines = [
        "# 24 段双 ASR 匿名局部仲裁包",
        "",
        f"- 样本数：`{pack.get('sample_count')}`",
        f"- 实质差异：`{pack.get('difference_count')}`",
        f"- 差异簇：`{pack.get('cluster_count')}`",
        "- 人工金标不在此包中，仅允许独立评估器读取。",
        "",
    ]
    for sample in pack.get("samples") or []:
        lines.extend(
            [
                f"## {sample.get('sample_id')}",
                "",
                f"- 分类：`{sample.get('category')}`",
                f"- 差异：`{sample.get('difference_count')}`；差异簇：`{sample.get('cluster_count')}`",
                f"- 音频：`{sample.get('audio_clip_path')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_result(result: dict[str, Any]) -> str:
    lines = ["# 双 ASR 局部仲裁评估", ""]
    for name, metrics in (result.get("metrics") or {}).items():
        if not isinstance(metrics, dict):
            continue
        lines.append(
            f"- `{name}`：CER `{metrics.get('mean_cer')}`，实体准确率 `{metrics.get('mean_entity_accuracy')}`，数字错误 `{metrics.get('number_error_count')}`"
        )
    lines.extend(
        [
            "",
            "> Whole-sample/local-patch oracle 仅用于估计理论上限，严禁作为生产纠错决定。",
            "",
        ]
    )
    return "\n".join(lines)
