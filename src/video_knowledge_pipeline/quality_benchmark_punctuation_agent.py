from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .models import now_iso
from .punctuation_model_stage import _content_only
from .quality_benchmark import transcript_quality_metrics
from .quality_benchmark_punctuation import _aggregate, _delta, _mutation_rate, _strip_punctuation
from .storage import read_json, write_json
from .transcript import parse_transcript


PACK_SCHEMA = "video_knowledge_pipeline.quality_benchmark_punctuation_agent_pack.v1"
DECISIONS_SCHEMA = "video_knowledge_pipeline.quality_benchmark_punctuation_agent_decisions.v1"
RESULT_SCHEMA = "video_knowledge_pipeline.quality_benchmark_punctuation_agent_result.v1"

PACK_JSON_NAME = "quality-benchmark-punctuation-agent-pack.json"
PACK_MD_NAME = "quality-benchmark-punctuation-agent-pack.md"
TODO_JSON_NAME = "quality-benchmark-punctuation-agent.todo.json"
RESULT_JSON_NAME = "quality-benchmark-punctuation-agent-result.json"
RESULT_MD_NAME = "quality-benchmark-punctuation-agent-result.md"


def build_quality_benchmark_punctuation_agent_pack(
    manifest_json: str | Path,
    *,
    output_dir: str | Path | None = None,
    source_variant: str = "sensevoice_full_punc",
    write: bool = True,
) -> dict[str, Any]:
    """Build a blind punctuation task pack that never exposes human reference text."""

    manifest_path = Path(manifest_json).expanduser().resolve()
    manifest = _manifest(manifest_path)
    out = Path(output_dir).expanduser().resolve() if output_dir else manifest_path.parent
    samples: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for sample in manifest.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "").strip()
        if not sample_id:
            missing.append({"sample_id": "", "reason": "sample_id_missing"})
            continue
        source_path = _variant_path(sample, source_variant, manifest_path.parent)
        if source_path is None or not source_path.exists():
            missing.append({"sample_id": sample_id, "reason": "source_variant_missing"})
            continue
        source_text = _transcript_text(source_path)
        input_text = source_text
        if not input_text:
            missing.append({"sample_id": sample_id, "reason": "source_text_empty"})
            continue
        samples.append(
            {
                "sample_id": sample_id,
                "category": str(sample.get("category") or ""),
                "start_seconds": float(sample.get("start_seconds") or 0.0),
                "end_seconds": float(sample.get("end_seconds") or 0.0),
                "audio_clip_path": str(sample.get("audio_clip_path") or ""),
                "input_text": input_text,
                "input_sha256": _fingerprint(_content_only(input_text)),
            }
        )

    pack = {
        "schema": PACK_SCHEMA,
        "status": "ready_for_agent" if samples and not missing else "incomplete",
        "source_variant": source_variant,
        "input_mode": "baseline_preserving_minimal_repair",
        "sample_count": len(samples),
        "samples": samples,
        "missing": missing,
        "instructions": [
            "Return every sample exactly once.",
            "Preserve existing correct punctuation; only repair clearly missing or incorrect punctuation and paragraph breaks.",
            "Do not rewrite content merely to improve style.",
            "Do not add, remove, replace, reorder, normalize, or translate content characters.",
            "Write the result to candidate_text; leave input_sha256 unchanged.",
        ],
        "decision_schema": {
            "schema": DECISIONS_SCHEMA,
            "rows": [
                {
                    "sample_id": "<sample-id>",
                    "input_sha256": "<copy-from-pack>",
                    "candidate_text": "<same-content-characters-with-punctuation-and-paragraphs>",
                }
            ],
        },
        "operator_boundary": {
            "local_pack_only": True,
            "human_reference_excluded": True,
            "manifest_path_excluded": True,
            "content_character_lock_required": True,
            "does_not_call_model": True,
            "does_not_modify_transcript": True,
        },
        "updated_at": now_iso(),
    }
    decisions = {
        "schema": DECISIONS_SCHEMA,
        "source_variant": source_variant,
        "rows": [
            {
                "sample_id": sample["sample_id"],
                "input_sha256": sample["input_sha256"],
                "candidate_text": "",
                "status": "todo",
            }
            for sample in samples
        ],
        "operator_boundary": {
            "human_reference_excluded": True,
            "insert_punctuation_and_paragraphs_only": True,
        },
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / PACK_JSON_NAME, pack)
        write_json(out / TODO_JSON_NAME, decisions)
        (out / PACK_MD_NAME).write_text(_render_pack(pack), encoding="utf-8")
    return pack


def evaluate_quality_benchmark_punctuation_agent(
    manifest_json: str | Path,
    decisions_json: str | Path,
    *,
    output_dir: str | Path | None = None,
    source_variant: str = "sensevoice_full_punc",
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate blind punctuation decisions; only this function reads human references."""

    manifest_path = Path(manifest_json).expanduser().resolve()
    manifest = _manifest(manifest_path)
    decisions_path = Path(decisions_json).expanduser().resolve()
    decisions = read_json(decisions_path)
    if not isinstance(decisions, dict):
        raise ValueError("punctuation decisions must be a JSON object")
    out = Path(output_dir).expanduser().resolve() if output_dir else manifest_path.parent
    decision_rows = _decision_rows(decisions)
    rows: list[dict[str, Any]] = []

    for sample in manifest.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "").strip()
        reference = str(sample.get("reference_text") or "")
        source_path = _variant_path(sample, source_variant, manifest_path.parent)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "status": "pending",
            "char_lock_passed": False,
            "content_character_change_rate": None,
            "baseline": None,
            "candidate": None,
        }
        if not sample_id:
            row.update({"status": "sample_id_missing", "reason": "sample_id_missing"})
            rows.append(row)
            continue
        if source_path is None or not source_path.exists():
            row.update({"status": "source_missing", "reason": "source_variant_missing"})
            rows.append(row)
            continue
        if not reference:
            row.update({"status": "reference_missing", "reason": "human_reference_missing"})
            rows.append(row)
            continue

        source_text = _transcript_text(source_path)
        input_text = source_text
        protected = [str(value) for value in sample.get("protected_entities") or [] if str(value)]
        row["baseline"] = transcript_quality_metrics(
            reference,
            source_text,
            protected_entities=protected,
        )
        decision = decision_rows.get(sample_id)
        if decision is None:
            row.update({"status": "missing_decision", "reason": "decision_row_missing"})
            rows.append(row)
            continue
        candidate = str(decision.get("candidate_text") or "")
        if not candidate:
            row.update({"status": "missing_decision", "reason": "candidate_text_empty"})
            rows.append(row)
            continue
        expected_fingerprint = _fingerprint(_content_only(input_text))
        supplied_fingerprint = str(decision.get("input_sha256") or "")
        if supplied_fingerprint and supplied_fingerprint != expected_fingerprint:
            row.update({"status": "input_mismatch", "reason": "input_sha256_mismatch"})
            rows.append(row)
            continue

        char_lock_passed = _content_only(input_text) == _content_only(candidate)
        mutation_rate = _mutation_rate(input_text, candidate)
        row.update(
            {
                "char_lock_passed": char_lock_passed,
                "content_character_change_rate": mutation_rate,
            }
        )
        if not char_lock_passed:
            row.update(
                {
                    "status": "rejected_character_change",
                    "reason": "content_character_sequence_changed",
                }
            )
            rows.append(row)
            continue

        row.update(
            {
                "status": "accepted",
                "candidate": transcript_quality_metrics(
                    reference,
                    candidate,
                    source_text=source_text,
                    protected_entities=protected,
                ),
            }
        )
        rows.append(row)

    accepted_count = sum(row["status"] == "accepted" for row in rows)
    blockers = [
        {"sample_id": row["sample_id"], "status": row["status"], "reason": row.get("reason", "")}
        for row in rows
        if row["status"] != "accepted"
    ]
    accepted_rows = [row for row in rows if row.get("status") == "accepted"]
    baseline_all = _aggregate(rows, "baseline")
    baseline = _aggregate(accepted_rows, "baseline")
    candidate = _aggregate(accepted_rows, "candidate")
    provided_mutations = [
        float(row["content_character_change_rate"])
        for row in rows
        if row.get("content_character_change_rate") is not None
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "status": "completed" if rows and not blockers else "completed_with_blockers",
        "source_variant": source_variant,
        "sample_count": len(rows),
        "accepted_count": accepted_count,
        "blocker_count": len(blockers),
        "metrics": {
            "baseline": baseline,
            "baseline_all": baseline_all,
            "candidate": candidate,
            "content_character_change_rate": round(
                sum(provided_mutations) / len(provided_mutations), 6
            )
            if provided_mutations
            else None,
        },
        "improvement": {
            "punctuation_f1": _delta(candidate, baseline, "punctuation_f1"),
            "sentence_boundary_f1": _delta(candidate, baseline, "sentence_boundary_f1"),
            "cer": _delta(candidate, baseline, "cer"),
        },
        "rows": rows,
        "blockers": blockers,
        "operator_boundary": {
            "human_reference_read_by_evaluator_only": True,
            "rejected_candidates_excluded_from_candidate_metrics": True,
            "does_not_call_model": True,
            "does_not_modify_transcript": True,
        },
        "updated_at": now_iso(),
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / RESULT_JSON_NAME, result)
        (out / RESULT_MD_NAME).write_text(_render_result(result), encoding="utf-8")
    return result


def _manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("quality benchmark manifest must be a JSON object")
    return payload


def _variant_path(sample: dict[str, Any], variant: str, base_dir: Path) -> Path | None:
    variants = sample.get("variants") if isinstance(sample.get("variants"), dict) else {}
    value = str(variants.get(variant) or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _transcript_text(path: Path) -> str:
    return "".join(str(cue.text or "") for cue in parse_transcript(path))


def _fingerprint(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _decision_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id and sample_id not in rows:
            rows[sample_id] = row
    return rows


def _render_pack(pack: dict[str, Any]) -> str:
    lines = [
        "# 字符锁定 Agent/Codex 标点任务包",
        "",
        f"- 状态：`{pack.get('status')}`",
        f"- 输入变体：`{pack.get('source_variant')}`",
        f"- 样本数：`{pack.get('sample_count')}`",
        "- 任务边界：只允许插入标点、空白和段落，不得改动内容字符及其顺序。",
        "- 人工参考稿和 manifest 路径均未包含在公开任务包中。",
        "",
    ]
    for sample in pack.get("samples") or []:
        lines.extend(
            [
                f"## {sample.get('sample_id')}",
                "",
                f"- 分类：`{sample.get('category')}`",
                f"- 时间：`{sample.get('start_seconds')}` - `{sample.get('end_seconds')}`",
                f"- 输入指纹：`{sample.get('input_sha256')}`",
                "",
                str(sample.get("input_text") or ""),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_result(result: dict[str, Any]) -> str:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    baseline = metrics.get("baseline") if isinstance(metrics.get("baseline"), dict) else {}
    candidate = metrics.get("candidate") if isinstance(metrics.get("candidate"), dict) else {}
    lines = [
        "# 字符锁定 Agent/Codex 标点 A/B 评估",
        "",
        f"- 状态：`{result.get('status')}`",
        f"- 样本：`{result.get('sample_count')}`；通过：`{result.get('accepted_count')}`；阻塞：`{result.get('blocker_count')}`",
        f"- Baseline 标点 F1：`{baseline.get('punctuation_f1')}`",
        f"- Candidate 标点 F1：`{candidate.get('punctuation_f1')}`",
        f"- Baseline 句界 F1：`{baseline.get('sentence_boundary_f1')}`",
        f"- Candidate 句界 F1：`{candidate.get('sentence_boundary_f1')}`",
        f"- Baseline CER：`{baseline.get('cer')}`",
        f"- Candidate CER：`{candidate.get('cer')}`",
        f"- 内容字符变更率：`{metrics.get('content_character_change_rate')}`",
        "",
        "## 逐样本状态",
        "",
        "| 样本 | 状态 | 字符锁 | 字符变更率 |",
        "|---|---|---:|---:|",
    ]
    for row in result.get("rows") or []:
        lines.append(
            f"| {row.get('sample_id')} | {row.get('status')} | {row.get('char_lock_passed')} | {row.get('content_character_change_rate')} |"
        )
    lines.extend(
        [
            "",
            "> 人工参考稿只在独立评估阶段读取；非法候选不会进入 Candidate 聚合指标。",
            "",
        ]
    )
    return "\n".join(lines)
