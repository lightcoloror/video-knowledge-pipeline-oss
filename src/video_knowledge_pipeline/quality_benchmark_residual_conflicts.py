from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .models import now_iso
from .numeric_normalization import (
    number_evidence_map,
    numeric_mentions_equivalent,
    strip_number_mentions,
)
from .punctuation_model_stage import _content_only
from .quality_benchmark_arbitration import PACK_SCHEMA as ARBITRATION_PACK_SCHEMA
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.quality_benchmark_residual_conflicts.v1"
TODO_SCHEMA = "video_knowledge_pipeline.quality_benchmark_residual_conflict_decisions.v1"

PACK_JSON_NAME = "quality-benchmark-residual-conflicts.json"
PACK_MD_NAME = "quality-benchmark-residual-conflicts.md"
TODO_JSON_NAME = "quality-benchmark-residual-conflicts.todo.json"

PUNCTUATION_ONLY = "punctuation_only"
DISCOURSE_PARTICLE_EQUIVALENT = "discourse_particle_equivalent"
NUMERIC_FORMAT_EQUIVALENT = "numeric_format_equivalent"
ENTITY_LEXICON_RESOLVED = "entity_lexicon_resolved"
RESIDUAL_TRUE_CONFLICT = "residual_true_conflict"

_LATIN_OR_CODE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+/#-]*")
_NUMBER_LIKE_RE = re.compile(r"\d|[零〇一二两三四五六七八九十百千万亿点]")
_DISCOURSE_PARTICLE_RE = re.compile(
    r"(?:啊+|呃+|嗯+|额+|唔+|哦+|噢+|喔+|哎+|唉+|诶+|欸+|呀+|嘛+|呢+|吧+|哈+|哼+)+"
)


def build_quality_benchmark_residual_conflicts(
    arbitration_pack_json: str | Path,
    *,
    manifest_json: str | Path | None = None,
    entity_lexicon_json: str | Path | None = None,
    output_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Filter an anonymous dual-ASR pack down to unresolved content conflicts.

    The benchmark manifest is optional and is used only to map ``sample_id`` to
    ``bundle_dir``. Human reference fields are deliberately never accessed.
    """

    pack_path = Path(arbitration_pack_json).expanduser().resolve()
    pack = read_json(pack_path)
    if not isinstance(pack, dict):
        raise ValueError("quality benchmark arbitration pack must be a JSON object")
    if pack.get("schema") not in {None, "", ARBITRATION_PACK_SCHEMA}:
        raise ValueError("unsupported quality benchmark arbitration pack schema")

    out = Path(output_dir).expanduser().resolve() if output_dir else pack_path.parent
    bundle_by_sample = _manifest_bundle_map(manifest_json)
    explicit_lexicon_path, explicit_terms = _read_entity_terms(entity_lexicon_json)
    lexicon_cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    samples: list[dict[str, Any]] = []
    all_residual_diffs: list[dict[str, Any]] = []
    all_residual_clusters: list[dict[str, Any]] = []
    difference_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()

    for raw_sample in pack.get("samples") or []:
        if not isinstance(raw_sample, dict):
            continue
        sample_id = str(raw_sample.get("sample_id") or "").strip()
        bundle_dir = bundle_by_sample.get(sample_id, "")
        bundle_lexicon_path, bundle_terms = _entity_terms(bundle_dir, lexicon_cache)
        terms = [*explicit_terms, *bundle_terms]
        lexicon_paths = _unique([explicit_lexicon_path, bundle_lexicon_path])
        lexicon_path = lexicon_paths[0] if lexicon_paths else ""
        classified_diffs: list[dict[str, Any]] = []
        diff_by_id: dict[str, dict[str, Any]] = {}

        for raw_diff in raw_sample.get("differences") or []:
            if not isinstance(raw_diff, dict):
                continue
            classified = _classify_difference(
                raw_diff,
                sample_id=sample_id,
                category=str(raw_sample.get("category") or ""),
                audio_clip_path=str(raw_sample.get("audio_clip_path") or ""),
                terms=terms,
            )
            classified_diffs.append(classified)
            diff_by_id[classified["diff_id"]] = classified
            difference_counts[classified["classification"]] += 1
            if classified["classification"] == RESIDUAL_TRUE_CONFLICT:
                all_residual_diffs.append(classified)

        classified_clusters: list[dict[str, Any]] = []
        residual_clusters: list[dict[str, Any]] = []
        for raw_cluster in raw_sample.get("clusters") or []:
            if not isinstance(raw_cluster, dict):
                continue
            cluster = _classify_cluster(
                raw_cluster,
                sample_id=sample_id,
                sample_audio_path=str(raw_sample.get("audio_clip_path") or ""),
                diff_by_id=diff_by_id,
            )
            classified_clusters.append(cluster)
            cluster_counts[cluster["classification"]] += 1
            if cluster["classification"] == RESIDUAL_TRUE_CONFLICT:
                residual_clusters.append(cluster)
                all_residual_clusters.append(cluster)

        sample_counts = Counter(row["classification"] for row in classified_diffs)
        samples.append(
            {
                "sample_id": sample_id,
                "category": str(raw_sample.get("category") or ""),
                "start_seconds": raw_sample.get("start_seconds"),
                "end_seconds": raw_sample.get("end_seconds"),
                "audio_clip_path": str(raw_sample.get("audio_clip_path") or ""),
                "bundle_dir": bundle_dir,
                "entity_lexicon_path": lexicon_path,
                "entity_lexicon_paths": lexicon_paths,
                "classification_counts": _all_class_counts(sample_counts),
                "differences": classified_diffs,
                "clusters": classified_clusters,
                "residual_conflicts": [
                    row for row in classified_diffs if row["classification"] == RESIDUAL_TRUE_CONFLICT
                ],
                "residual_clusters": residual_clusters,
            }
        )

    input_difference_count = sum(difference_counts.values())
    result = {
        "schema": SCHEMA,
        "status": "ready_for_review" if all_residual_diffs else "no_residual_conflicts",
        "source_arbitration_pack": str(pack_path),
        "manifest_used_for_bundle_lookup": bool(manifest_json),
        "explicit_entity_lexicon_path": explicit_lexicon_path,
        "explicit_entity_lexicon_used": bool(explicit_lexicon_path),
        "sample_count": len(samples),
        "input_difference_count": input_difference_count,
        "excluded_difference_count": input_difference_count - len(all_residual_diffs),
        "residual_difference_count": len(all_residual_diffs),
        "input_cluster_count": sum(cluster_counts.values()),
        "residual_cluster_count": len(all_residual_clusters),
        "classification_counts": _all_class_counts(difference_counts),
        "cluster_classification_counts": _all_class_counts(cluster_counts),
        "samples": samples,
        "residual_conflicts": all_residual_diffs,
        "residual_clusters": all_residual_clusters,
        "operator_boundary": {
            "source_pack_is_anonymous": True,
            "human_reference_not_read": True,
            "manifest_used_only_for_bundle_dir": True,
            "explicit_entity_lexicon_is_independent_of_bundle_lookup": True,
            "does_not_call_model": True,
            "does_not_apply_patches": True,
            "review_required_entity_terms_never_auto_resolve": True,
        },
        "updated_at": now_iso(),
    }
    todo = {
        "schema": TODO_SCHEMA,
        "status": "todo" if all_residual_clusters else "nothing_to_review",
        "row_count": len(all_residual_clusters),
        "rows": [
            {
                "sample_id": row["sample_id"],
                "cluster_id": row["cluster_id"],
                "diff_ids": row["residual_diff_ids"],
                "choice": "",
                "confidence": None,
                "evidence_refs": [],
                "reason": "",
            }
            for row in all_residual_clusters
        ],
        "operator_boundary": {
            "contains_no_answers": True,
            "does_not_apply_patches": True,
            "human_reference_excluded": True,
        },
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / PACK_JSON_NAME, result)
        write_json(out / TODO_JSON_NAME, todo)
        (out / PACK_MD_NAME).write_text(_render_markdown(result), encoding="utf-8")
    return result


def _manifest_bundle_map(manifest_json: str | Path | None) -> dict[str, str]:
    if not manifest_json:
        return {}
    path = Path(manifest_json).expanduser().resolve()
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("quality benchmark manifest must be a JSON object")
    mapping: dict[str, str] = {}
    for sample in manifest.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "").strip()
        bundle_dir = str(sample.get("bundle_dir") or "").strip()
        if sample_id and bundle_dir:
            mapping[sample_id] = str(Path(bundle_dir).expanduser().resolve())
    return mapping


def _entity_terms(
    bundle_dir: str,
    cache: dict[str, tuple[str, list[dict[str, Any]]]],
) -> tuple[str, list[dict[str, Any]]]:
    if not bundle_dir:
        return "", []
    cache_key = str(Path(bundle_dir).expanduser())
    if cache_key in cache:
        return cache[cache_key]
    path = Path(bundle_dir).expanduser().resolve() / "entity-lexicon.json"
    if not path.exists():
        cache[cache_key] = (str(path), [])
        return cache[cache_key]
    payload = read_json(path)
    terms = [row for row in (payload.get("terms") if isinstance(payload, dict) else []) or [] if isinstance(row, dict)]
    cache[cache_key] = (str(path), terms)
    return cache[cache_key]


def _read_entity_terms(entity_lexicon_json: str | Path | None) -> tuple[str, list[dict[str, Any]]]:
    if not entity_lexicon_json:
        return "", []
    path = Path(entity_lexicon_json).expanduser().resolve()
    payload = read_json(path)
    if isinstance(payload, dict):
        raw_terms = payload.get("terms") or []
    elif isinstance(payload, list):
        raw_terms = payload
    else:
        raise ValueError("entity lexicon must be a JSON object with terms or a JSON array")
    if not isinstance(raw_terms, list):
        raise ValueError("entity lexicon terms must be a JSON array")
    return str(path), [row for row in raw_terms if isinstance(row, dict)]


def _classify_difference(
    raw_diff: dict[str, Any],
    *,
    sample_id: str,
    category: str,
    audio_clip_path: str,
    terms: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_a = str(raw_diff.get("candidate_a") or "")
    candidate_b = str(raw_diff.get("candidate_b") or "")
    before = str(raw_diff.get("context_before") or "")
    after = str(raw_diff.get("context_after") or "")
    scope_a = before + candidate_a + after
    scope_b = before + candidate_b + after
    classification = RESIDUAL_TRUE_CONFLICT
    exclusion_reason = ""
    recommended_canonical = ""
    entity_evidence: dict[str, Any] | None = None
    numeric_evidence: dict[str, Any] | None = None

    if _content_only(candidate_a) == _content_only(candidate_b):
        classification = PUNCTUATION_ONLY
        exclusion_reason = "content_equal_after_removing_punctuation_and_whitespace"
    elif _discourse_particle_equivalent(candidate_a, candidate_b):
        classification = DISCOURSE_PARTICLE_EQUIVALENT
        exclusion_reason = "difference_contains_only_conservative_discourse_particles_or_fillers"
    elif _numeric_format_equivalent(scope_a, scope_b):
        classification = NUMERIC_FORMAT_EQUIVALENT
        exclusion_reason = "unit_aware_numeric_evidence_and_non_numeric_content_are_equal"
        numeric_evidence = {
            "candidate_a_keys": sorted(_numeric_evidence_keys(scope_a)),
            "candidate_b_keys": sorted(_numeric_evidence_keys(scope_b)),
        }
    else:
        entity_evidence = _entity_resolution(scope_a, scope_b, candidate_a, candidate_b, terms)
        if entity_evidence and entity_evidence.get("resolved"):
            classification = ENTITY_LEXICON_RESOLVED
            exclusion_reason = "explicit_non_review_entity_alias_maps_to_canonical"
            recommended_canonical = str(entity_evidence.get("canonical") or "")

    risk_reasons = []
    if classification == RESIDUAL_TRUE_CONFLICT:
        risk_reasons = _risk_reasons(
            candidate_a,
            candidate_b,
            category=category,
            operation=str(raw_diff.get("operation") or ""),
            terms=terms,
            scope_a=scope_a,
            scope_b=scope_b,
        )
    return {
        "sample_id": sample_id,
        "diff_id": str(raw_diff.get("diff_id") or ""),
        "cluster_id": str(raw_diff.get("cluster_id") or ""),
        "classification": classification,
        "excluded_from_review": classification != RESIDUAL_TRUE_CONFLICT,
        "exclusion_reason": exclusion_reason,
        "recommended_canonical": recommended_canonical,
        "operation": str(raw_diff.get("operation") or ""),
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "context_before": before,
        "context_after": after,
        "estimated_time": raw_diff.get("estimated_time") or {},
        "audio_clip_path": audio_clip_path,
        "risk_reasons": risk_reasons,
        "numeric_evidence": numeric_evidence,
        "entity_evidence": entity_evidence,
    }


def _numeric_format_equivalent(scope_a: str, scope_b: str) -> bool:
    left_keys = _numeric_evidence_keys(scope_a)
    right_keys = _numeric_evidence_keys(scope_b)
    if not numeric_mentions_equivalent(scope_a, scope_b) and not (
        left_keys and left_keys == right_keys
    ):
        return False
    left_rest = _content_only(strip_number_mentions(scope_a))
    right_rest = _content_only(strip_number_mentions(scope_b))
    return left_rest == right_rest


def _discourse_particle_equivalent(candidate_a: str, candidate_b: str) -> bool:
    content_a = _content_only(candidate_a)
    content_b = _content_only(candidate_b)
    if not content_a and not content_b:
        return False
    return all(
        not value or bool(_DISCOURSE_PARTICLE_RE.fullmatch(value))
        for value in (content_a, content_b)
    )


def _numeric_evidence_keys(text: str) -> set[str]:
    """Drop duplicate keys caused when a unit suffix is parsed as a number."""

    evidence = number_evidence_map(text)
    unit_only_mentions = {"万", "亿", "万元", "亿元", "百分比"}
    return {
        key
        for key, mentions in evidence.items()
        if not mentions
        or not all(str(mention).strip() in unit_only_mentions for mention in mentions)
    }


def _entity_resolution(
    scope_a: str,
    scope_b: str,
    candidate_a: str,
    candidate_b: str,
    terms: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for term in sorted(terms, key=lambda row: len(str(row.get("canonical") or "")), reverse=True):
        canonical = str(term.get("canonical") or term.get("canonical_term") or "").strip()
        aliases = _unique([canonical, *[str(value) for value in term.get("aliases") or term.get("raw_mentions") or []]])
        if not canonical or len(aliases) < 2:
            continue
        touched_a = [value for value in aliases if value and value in scope_a]
        touched_b = [value for value in aliases if value and value in scope_b]
        if not touched_a or not touched_b or set(touched_a) == set(touched_b):
            continue
        normalized_a = _replace_aliases(scope_a, aliases, canonical)
        normalized_b = _replace_aliases(scope_b, aliases, canonical)
        if _content_only(normalized_a) != _content_only(normalized_b):
            continue
        evidence = {
            "resolved": not bool(term.get("review_required")),
            "canonical": canonical,
            "candidate_a_mentions": touched_a,
            "candidate_b_mentions": touched_b,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "review_required": bool(term.get("review_required")),
            "source_types": [str(value) for value in term.get("source_types") or []],
        }
        return evidence
    return None


def _replace_aliases(text: str, aliases: list[str], canonical: str) -> str:
    result = str(text or "")
    for alias in sorted(aliases, key=len, reverse=True):
        result = result.replace(alias, canonical)
    return result


def _risk_reasons(
    candidate_a: str,
    candidate_b: str,
    *,
    category: str,
    operation: str,
    terms: list[dict[str, Any]],
    scope_a: str,
    scope_b: str,
) -> list[str]:
    combined = candidate_a + candidate_b
    reasons: list[str] = []
    if operation in {"insert", "delete"} or not candidate_a or not candidate_b:
        reasons.append("content_insertion_or_deletion")
    if _NUMBER_LIKE_RE.search(combined):
        reasons.append("numeric_value_or_expression_conflict")
    if _LATIN_OR_CODE_RE.search(combined):
        reasons.append("proper_noun_tool_or_code_term_risk")
    if category:
        reasons.append(f"benchmark_category:{category}")
    for term in terms:
        if not term.get("review_required"):
            continue
        canonical = str(term.get("canonical") or term.get("canonical_term") or "").strip()
        aliases = _unique([canonical, *[str(value) for value in term.get("aliases") or term.get("raw_mentions") or []]])
        if any(value and (value in scope_a or value in scope_b) for value in aliases):
            reasons.append("dynamic_entity_requires_review")
            break
    reasons.append("lexical_content_conflict")
    return _unique(reasons)


def _classify_cluster(
    raw_cluster: dict[str, Any],
    *,
    sample_id: str,
    sample_audio_path: str,
    diff_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    diff_ids = [str(value) for value in raw_cluster.get("diff_ids") or [] if str(value)]
    rows = [diff_by_id[value] for value in diff_ids if value in diff_by_id]
    counts = Counter(row["classification"] for row in rows)
    if counts.get(RESIDUAL_TRUE_CONFLICT):
        classification = RESIDUAL_TRUE_CONFLICT
    elif counts.get(ENTITY_LEXICON_RESOLVED):
        classification = ENTITY_LEXICON_RESOLVED
    elif counts.get(NUMERIC_FORMAT_EQUIVALENT):
        classification = NUMERIC_FORMAT_EQUIVALENT
    elif counts.get(DISCOURSE_PARTICLE_EQUIVALENT):
        classification = DISCOURSE_PARTICLE_EQUIVALENT
    else:
        classification = PUNCTUATION_ONLY
    residual_rows = [row for row in rows if row["classification"] == RESIDUAL_TRUE_CONFLICT]
    review_window = raw_cluster.get("audio_review_window") if isinstance(raw_cluster.get("audio_review_window"), dict) else {}
    audio_path = str(review_window.get("audio_clip_path") or sample_audio_path)
    risk_reasons = _unique(
        [reason for row in residual_rows for reason in row.get("risk_reasons") or []]
        + (["multi_difference_cluster"] if len(residual_rows) > 1 else [])
    )
    return {
        "sample_id": sample_id,
        "cluster_id": str(raw_cluster.get("cluster_id") or ""),
        "classification": classification,
        "classification_counts": _all_class_counts(counts),
        "excluded_from_review": classification != RESIDUAL_TRUE_CONFLICT,
        "diff_ids": diff_ids,
        "residual_diff_ids": [row["diff_id"] for row in residual_rows],
        "candidate_a": str(raw_cluster.get("candidate_a") or ""),
        "candidate_b": str(raw_cluster.get("candidate_b") or ""),
        "context_before": str(raw_cluster.get("context_before") or ""),
        "context_after": str(raw_cluster.get("context_after") or ""),
        "time_range": str(raw_cluster.get("time_range") or ""),
        "estimated_time": _cluster_estimated_time(residual_rows),
        "audio_clip_path": audio_path,
        "audio_review_window": review_window,
        "risk_reasons": risk_reasons,
    }


def _cluster_estimated_time(rows: list[dict[str, Any]]) -> dict[str, float] | dict[str, Any]:
    windows = [row.get("estimated_time") for row in rows if isinstance(row.get("estimated_time"), dict)]
    starts = [float(row["start"]) for row in windows if row.get("start") is not None]
    ends = [float(row["end"]) for row in windows if row.get("end") is not None]
    return {"start": min(starts), "end": max(ends)} if starts and ends else {}


def _all_class_counts(counts: Counter[str]) -> dict[str, int]:
    return {
        PUNCTUATION_ONLY: int(counts.get(PUNCTUATION_ONLY, 0)),
        DISCOURSE_PARTICLE_EQUIVALENT: int(counts.get(DISCOURSE_PARTICLE_EQUIVALENT, 0)),
        NUMERIC_FORMAT_EQUIVALENT: int(counts.get(NUMERIC_FORMAT_EQUIVALENT, 0)),
        ENTITY_LEXICON_RESOLVED: int(counts.get(ENTITY_LEXICON_RESOLVED, 0)),
        RESIDUAL_TRUE_CONFLICT: int(counts.get(RESIDUAL_TRUE_CONFLICT, 0)),
    }


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in (str(item).strip() for item in values) if value))


def _render_markdown(result: dict[str, Any]) -> str:
    counts = result.get("classification_counts") if isinstance(result.get("classification_counts"), dict) else {}
    lines = [
        "# 双 ASR 剩余真实冲突复核包",
        "",
        f"- 输入差异：`{result.get('input_difference_count')}`",
        f"- 自动排除：`{result.get('excluded_difference_count')}`",
        f"- 剩余真实冲突：`{result.get('residual_difference_count')}`",
        f"- 剩余冲突簇：`{result.get('residual_cluster_count')}`",
        f"- 仅标点/空白：`{counts.get(PUNCTUATION_ONLY, 0)}`",
        f"- 语气词/口头填充词等价：`{counts.get(DISCOURSE_PARTICLE_EQUIVALENT, 0)}`",
        f"- 数字格式等价：`{counts.get(NUMERIC_FORMAT_EQUIVALENT, 0)}`",
        f"- 实体词库已解决：`{counts.get(ENTITY_LEXICON_RESOLVED, 0)}`",
        "- 此包不读取人工参考稿、不调用模型，也不应用任何补丁。",
        "",
        "| 样本 | 冲突簇 | 时间 | A | B | 风险 | 音频证据 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.get("residual_clusters") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("sample_id")),
                    _md(row.get("cluster_id")),
                    _md(row.get("time_range")),
                    _md(row.get("candidate_a")),
                    _md(row.get("candidate_b")),
                    _md(", ".join(row.get("risk_reasons") or [])),
                    _md(row.get("audio_clip_path")),
                ]
            )
            + " |"
        )
    lines.extend(["", "> todo 仅包含空白决策字段，必须依据音频或独立证据完成复核。", ""])
    return "\n".join(lines)


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
