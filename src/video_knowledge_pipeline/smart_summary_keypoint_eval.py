from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import jieba
import rapidfuzz
from rapidfuzz import fuzz, process

from .file_hash import sha256_file
from .numeric_normalization import number_evidence_map


SCHEMA = "video_knowledge_pipeline.smart_summary_key_point_recall.v2"
GOLDSET_SCHEMA = "video_knowledge_pipeline.human_key_points.v2"
REVIEWED_RAPIDFUZZ_COMMIT = "edf9f3c2d016c878dae1511301f8b4a501bba871"
REVIEWED_JIEBA_COMMIT = "1e20c89b66f56c9301b0feed211733ffaa1bd72a"
RAPIDFUZZ_THRESHOLD = 88.0
TOKEN_SET_THRESHOLD = 90.0
LEGACY_BIGRAM_RECALL_THRESHOLD = 0.60
_FRAGMENT_SPLIT_RE = re.compile(r"(?:\r?\n)+|(?<=[。！？!?；;])")
_COMPACT_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
CLAIM_SCOPE_FIELDS = frozenset({'subject', 'property', 'value', 'unit', 'negation', 'condition', 'time', 'version'})


def evaluate_human_key_point_recall(
    payload: Any,
    summary_body: str,
    *,
    source_path: str | Path | None = None,
    bundle_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a human gold set without promoting lexical similarity to truth.

    Intent: make key-point recall explainable and paraphrase-aware enough for
    stable Smart Summary quality checks.
    Decision: reuse Jieba deterministic tokenization plus RapidFuzz
    ``process.extractOne``/``fuzz.token_set_ratio``/``fuzz.WRatio`` for local
    lexical alignment, while retaining the character-bigram compatibility path.
    Reason: edit similarity can tolerate ASR punctuation/order drift but cannot
    prove that two differently worded claims mean the same thing; human-supplied
    aliases remain the only semantic-equivalence authority.
    Evidence: RapidFuzz v3.14.5 commit ``edf9f3c...`` tests WRatio and
    extractOne against both C++ and pure-Python implementations; Jieba v0.42.1
    commit ``1e20c89...`` exposes accurate-mode ``lcut(..., HMM=False)``.
    Effective scope: local Smart Summary gold-set evaluation only; no transcript
    text, summary prose, Timeline evidence, or provider route is modified.
    """

    freshness = reference_binding_status(payload, bundle_root=bundle_root)
    if freshness['status'] not in {'fresh', 'legacy_unbound'}:
        result = _not_evaluated('human reference bindings need review: ' + freshness['status'], source_path=source_path)
        result['reference_freshness'] = freshness
        return result
    raw = payload.get("key_points") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        return _not_evaluated("human key points file is empty", source_path=source_path)

    entries: list[dict[str, Any]] = []
    invalid_entries: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        entry, error = _normalise_key_point(item, index=index)
        if error:
            invalid_entries.append({"index": index, "error": error})
        elif entry:
            entries.append(entry)
    if invalid_entries:
        result = _not_evaluated(
            "human key points contain invalid entries; fix the gold set before scoring",
            source_path=source_path,
        )
        result["invalid_entries"] = invalid_entries
        return result
    if not entries:
        return _not_evaluated("human key points file is empty", source_path=source_path)

    summary = str(summary_body or "")
    compact_summary = _compact(summary)
    summary_bigrams = _bigrams(summary)
    summary_number_keys = set(number_evidence_map(summary))
    fragments = _summary_fragments(summary)
    compact_fragments = [_compact(fragment) for fragment in fragments]
    tokenized_fragments = [_jieba_token_string(fragment) for fragment in fragments]

    decisions = [
        _evaluate_key_point(
            entry,
            compact_summary=compact_summary,
            summary_bigrams=summary_bigrams,
            summary_number_keys=summary_number_keys,
            fragments=fragments,
            compact_fragments=compact_fragments,
            tokenized_fragments=tokenized_fragments,
        )
        for entry in entries
    ]
    hits = sum(1 for decision in decisions if decision["matched"])
    missed = [decision["text"] for decision in decisions if not decision["matched"]]
    recall = round(hits / len(decisions), 6)
    source = _source_metadata(source_path)
    return {
        "schema": SCHEMA,
        "goldset_schema": (
            str(payload.get("schema") or GOLDSET_SCHEMA)
            if isinstance(payload, dict)
            else "legacy_list"
        ),
        "evaluated": True,
        "assessment_scope": "reference_recall_not_factual_truth",
        "reference_freshness": freshness,
        "recall": recall,
        "hit_count": hits,
        "total": len(decisions),
        "missed": missed[:12],
        "decisions": decisions,
        "matcher": {
            "libraries": {
                "rapidfuzz": {
                    "runtime_version": rapidfuzz.__version__,
                    "reviewed_version": "3.14.5",
                    "reviewed_source_commit": REVIEWED_RAPIDFUZZ_COMMIT,
                },
                "jieba": {
                    "runtime_version": jieba.__version__,
                    "reviewed_version": "0.42.1",
                    "reviewed_source_commit": REVIEWED_JIEBA_COMMIT,
                },
            },
            "algorithm": "jieba.lcut(HMM=False)+process.extractOne+fuzz.token_set_ratio/fuzz.WRatio",
            "rapidfuzz_threshold": RAPIDFUZZ_THRESHOLD,
            "token_set_threshold": TOKEN_SET_THRESHOLD,
            "legacy_bigram_recall_threshold": LEGACY_BIGRAM_RECALL_THRESHOLD,
            "semantic_equivalence_policy": "explicit_alias_only",
        },
        "source": source,
        "detail": (
            f"recall={recall}; target>=0.85; missed={missed[:5]}; "
            f"matcher=RapidFuzz-{rapidfuzz.__version__}"
        ),
    }


def _evaluate_key_point(
    entry: dict[str, Any],
    *,
    compact_summary: str,
    summary_bigrams: set[str],
    summary_number_keys: set[str],
    fragments: list[str],
    compact_fragments: list[str],
    tokenized_fragments: list[str],
) -> dict[str, Any]:
    # Reuse the established RapidFuzz/Jieba matcher, but score each fragment
    # with its own numbers and anchors. Global bags of words/numbers cannot
    # establish a claim's subject-value relationship (RAGChecker claim scope).
    choices = []
    for index, fragment in enumerate(fragments):
        decision = _evaluate_key_point_fragment(
            entry, compact_summary=compact_fragments[index],
            summary_bigrams=_bigrams(fragment),
            summary_number_keys=set(number_evidence_map(fragment)),
            fragments=[fragment], compact_fragments=[compact_fragments[index]],
            tokenized_fragments=[tokenized_fragments[index]],
        )
        decision['fragment_index'] = index
        choices.append(decision)
    if not choices:
        return _evaluate_key_point_fragment(
            entry, compact_summary='', summary_bigrams=set(), summary_number_keys=set(),
            fragments=[], compact_fragments=[], tokenized_fragments=[])
    return max(choices, key=lambda row: (row['matched'], row['score'], -row['fragment_index']))


def _evaluate_key_point_fragment(
    entry: dict[str, Any],
    *,
    compact_summary: str,
    summary_bigrams: set[str],
    summary_number_keys: set[str],
    fragments: list[str],
    compact_fragments: list[str],
    tokenized_fragments: list[str],
) -> dict[str, Any]:
    variants = [entry["text"], *entry["aliases"]]
    compact_variants = [_compact(value) for value in variants if _compact(value)]
    exact_index = next(
        (
            index
            for index, compact_variant in enumerate(compact_variants)
            if compact_variant and compact_variant in compact_summary
        ),
        None,
    )
    bigram_scores = [
        len(_bigrams(variant) & summary_bigrams) / max(1, len(_bigrams(variant)))
        for variant in variants
    ]
    bigram_score = max(bigram_scores, default=0.0)
    best_fuzzy: tuple[str, float, int] | None = None
    best_variant = ""
    if compact_fragments:
        for variant, compact_variant in zip(variants, compact_variants, strict=False):
            match = process.extractOne(
                compact_variant,
                compact_fragments,
                scorer=fuzz.WRatio,
            )
            if match and (best_fuzzy is None or float(match[1]) > float(best_fuzzy[1])):
                best_fuzzy = (str(match[0]), float(match[1]), int(match[2]))
                best_variant = variant
    best_token: tuple[str, float, int] | None = None
    best_token_variant = ""
    if tokenized_fragments:
        for variant in variants:
            tokenized_variant = _jieba_token_string(variant)
            match = process.extractOne(
                tokenized_variant,
                tokenized_fragments,
                scorer=fuzz.token_set_ratio,
            )
            if match and (best_token is None or float(match[1]) > float(best_token[1])):
                best_token = (str(match[0]), float(match[1]), int(match[2]))
                best_token_variant = variant

    method = "not_matched"
    matched = False
    score = max(
        float(bigram_score * 100.0),
        float(best_fuzzy[1]) if best_fuzzy else 0.0,
        float(best_token[1]) if best_token else 0.0,
    )
    matched_variant = ""
    if exact_index is not None:
        matched = True
        score = 100.0
        method = "exact_text" if exact_index == 0 else "explicit_alias"
        matched_variant = variants[exact_index]
    elif best_token and float(best_token[1]) >= TOKEN_SET_THRESHOLD:
        matched = True
        method = "jieba_rapidfuzz_token_set"
        matched_variant = best_token_variant
    elif best_fuzzy and float(best_fuzzy[1]) >= RAPIDFUZZ_THRESHOLD:
        matched = True
        method = "rapidfuzz_wratio"
        matched_variant = best_variant
    elif bigram_score >= LEGACY_BIGRAM_RECALL_THRESHOLD:
        matched = True
        method = "legacy_char_bigram"
        matched_variant = variants[bigram_scores.index(bigram_score)]

    required_number_keys = set(number_evidence_map(entry["text"]))
    missing_number_keys = sorted(required_number_keys - summary_number_keys)
    if matched and missing_number_keys:
        matched = False
        method = "numeric_evidence_missing"

    # Conservative vetoes only, not an entailment model. Approved aliases are
    # compared in their own wording so explicit human paraphrases still work.
    constraint_reasons = []
    if matched:
        from .transcript_semantic_correction import _semantic_anchor_map

        reference_anchors = _semantic_anchor_map(matched_variant)
        fragment_anchors = _semantic_anchor_map(fragments[0])
        for anchor in ('negation', 'condition', 'exception'):
            if reference_anchors.get(anchor, []) != fragment_anchors.get(anchor, []):
                constraint_reasons.append(f'{anchor}_scope_mismatch')
        for key, value in entry.get('claim_scope', {}).items():
            if _compact(value) not in compact_summary:
                constraint_reasons.append(f'{key}_scope_missing')
        if constraint_reasons:
            matched = False
            method = 'claim_constraint_mismatch'

    matched_excerpt = ""
    if method == "jieba_rapidfuzz_token_set" and best_token:
        matched_excerpt = fragments[best_token[2]][:240]
    elif best_fuzzy:
        matched_excerpt = fragments[best_fuzzy[2]][:240]
    return {
        "id": entry["id"],
        "text": entry["text"],
        "aliases": entry["aliases"],
        "matched": matched,
        "method": method,
        "score": round(score, 4),
        "matched_variant": matched_variant,
        "matched_excerpt": matched_excerpt,
        "bigram_recall": round(bigram_score, 4),
        "rapidfuzz_wratio": round(float(best_fuzzy[1]) if best_fuzzy else 0.0, 4),
        "jieba_token_set_ratio": round(float(best_token[1]) if best_token else 0.0, 4),
        "required_number_evidence": sorted(required_number_keys),
        "missing_number_evidence": missing_number_keys,
        "time_range": entry["time_range"],
        "evidence_ids": entry["evidence_ids"],
        "source_kind": entry["source_kind"],
        "claim_scope": entry.get('claim_scope', {}),
        "constraint_reasons": constraint_reasons,
        "semantic_support": 'not_assessed',
    }


def normalize_claim_scope(value: Any) -> dict[str, str]:
    """Validate optional human-declared anchors; never invent entity identity."""
    if not isinstance(value, dict) or set(value) - CLAIM_SCOPE_FIELDS or any(
        not isinstance(text, str) or not _compact(text) or len(text) > 512
        for text in value.values()
    ):
        raise ValueError('invalid claim_scope; use bounded explicit string anchors')
    return dict(value)


def reference_binding_status(payload: Any, *, bundle_root: str | Path | None) -> dict[str, Any]:
    """Read-only freshness check, reusing VKP's confined-path and hash gates."""
    if not isinstance(payload, dict) or 'source_bindings' not in payload:
        return {'status': 'legacy_unbound', 'checked_count': 0}
    bindings = payload['source_bindings']
    if not bundle_root or not isinstance(bindings, list) or not 1 <= len(bindings) <= 64:
        return {'status': 'invalid', 'checked_count': 0}
    from .courseware_material import safe_path

    root = Path(bundle_root).absolute()
    checked = 0
    seen = set()
    try:
        for row in bindings:
            if not isinstance(row, dict) or not re.fullmatch(r'[0-9a-fA-F]{64}', str(row.get('sha256') or '')):
                raise ValueError('invalid binding identity')
            path = safe_path(root, str(row.get('path') or ''))
            if path.suffix.lower() not in {'.json', '.srt', '.vtt', '.txt', '.md'} or path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError('reference binding is not bounded text evidence')
            if path in seen:
                raise ValueError('duplicate reference binding')
            seen.add(path)
            if sha256_file(path) != str(row['sha256']).lower():
                return {'status': 'stale', 'checked_count': checked, 'reason': 'source_hash_drift'}
            checked += 1
    except (ValueError, OSError):
        return {'status': 'invalid', 'checked_count': checked, 'reason': 'unsafe_missing_or_invalid_binding'}
    return {'status': 'fresh', 'checked_count': checked}


def _normalise_key_point(item: Any, *, index: int) -> tuple[dict[str, Any] | None, str]:
    if isinstance(item, str):
        text = item.strip()
        if not _compact(text):
            return None, "key point text is empty"
        return {
            "id": f"kp-{index:04d}",
            "text": text,
            "aliases": [],
            "time_range": "",
            "evidence_ids": [],
            "source_kind": "human_confirmed",
        }, ""
    if not isinstance(item, dict):
        return None, "key point must be a string or object"
    text = str(item.get("text") or "").strip()
    if not _compact(text):
        return None, "key point object requires non-empty text"
    raw_aliases = item.get("aliases") or []
    if isinstance(raw_aliases, str):
        raw_aliases = [raw_aliases]
    if not isinstance(raw_aliases, list) or any(not isinstance(alias, str) for alias in raw_aliases):
        return None, "aliases must be a string or list of strings"
    aliases = [alias.strip() for alias in raw_aliases if _compact(alias)]
    try:
        claim_scope = normalize_claim_scope(item.get('claim_scope', {}))
    except ValueError as exc:
        return None, str(exc)
    raw_evidence_ids = item.get("evidence_ids") or []
    if isinstance(raw_evidence_ids, str):
        raw_evidence_ids = [raw_evidence_ids]
    if not isinstance(raw_evidence_ids, list) or any(
        not isinstance(evidence_id, str) for evidence_id in raw_evidence_ids
    ):
        return None, "evidence_ids must be a string or list of strings"
    return {
        "id": str(item.get("id") or "").strip() or f"kp-{index:04d}",
        "text": text,
        "aliases": aliases,
        "claim_scope": dict(claim_scope),
        "time_range": str(item.get("time_range") or "").strip(),
        "evidence_ids": [
            evidence_id.strip() for evidence_id in raw_evidence_ids if evidence_id.strip()
        ],
        "source_kind": str(item.get("source_kind") or "human_confirmed").strip(),
    }, ""


def _summary_fragments(value: str) -> list[str]:
    return [
        fragment.strip()
        for fragment in _FRAGMENT_SPLIT_RE.split(str(value or ""))
        if len(_compact(fragment)) >= 2
    ]


def _compact(value: str) -> str:
    return _COMPACT_RE.sub("", str(value or "")).lower()


def _jieba_token_string(value: str) -> str:
    jieba.setLogLevel(logging.ERROR)
    return " ".join(
        token.strip() for token in jieba.lcut(str(value or ""), HMM=False) if _compact(token)
    )


def _bigrams(value: str) -> set[str]:
    compact = _compact(value)
    return {
        compact[index : index + 2]
        for index in range(max(0, len(compact) - 1))
    }


def _source_metadata(source_path: str | Path | None) -> dict[str, Any]:
    if not source_path:
        return {"path": "", "sha256": ""}
    path = Path(source_path).expanduser().resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def _not_evaluated(
    detail: str,
    *,
    source_path: str | Path | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "evaluated": False,
        "assessment_scope": "reference_recall_not_factual_truth",
        "recall": None,
        "detail": detail,
        "source": _source_metadata(source_path),
    }
