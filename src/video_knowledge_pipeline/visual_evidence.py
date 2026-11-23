from __future__ import annotations

import math
import re
from typing import Any, Iterable


ACCEPTED_REVIEW_STATUSES = frozenset(
    {
        "accepted",
        "reviewed",
        "keep_image",
        "accepted_known_gap",
        "accepted_no_visual_content",
        "accepted_provider_blocked",
        "corrected_visual_text",
        "corrected_visual_understanding",
        "corrected_temporal_visual_understanding",
    }
)

VISUAL_CONTENT_FIELDS = (
    "objects",
    "actions",
    "interface_state",
    "spatial_relations",
    "non_text_information",
    "instructor_focus",
)
TEMPORAL_CONTENT_FIELDS = (
    "event_sequence",
    "state_changes",
    "operation_steps",
    "causal_links",
)


def multimodal_document_ocr_state(item: dict[str, Any]) -> dict[str, Any]:
    """State whether local OCR and a valid VLM result jointly cover a slide.

    This is deliberately narrower than ``has_visual_evidence``.  A plain
    machine caption is not enough to retire a document-visual review item: the
    frame also needs independently extracted screen text, a high-confidence
    visual result, and an explicit layout/relationship description.

    ASR/OCR conflicts are evaluated by the escalation gate, which has the
    transcript and frame-change context.  Callers must therefore keep a row in
    review when that gate reports a conflict.
    """
    visual_text = (
        str(item.get("human_corrected_visual_text") or "").strip()
        or str(item.get("visual_text") or "").strip()
        or str(item.get("ocr_text") or "").strip()
    )
    state = visual_evidence_state(item)
    understanding = state.get("understanding") if isinstance(state.get("understanding"), dict) else {}
    raw_confidence = understanding.get("confidence")
    confidence = 0.0
    confidence_valid = False
    if not isinstance(raw_confidence, bool):
        try:
            parsed = float(raw_confidence)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None and math.isfinite(parsed):
            if 0 <= parsed <= 1:
                confidence = parsed
                confidence_valid = True
            elif 1 < parsed <= 100:
                confidence = parsed / 100
                confidence_valid = 0 <= confidence <= 1
    has_layout = bool(str(understanding.get("interface_state") or "").strip()) and bool(
        understanding.get("objects")
        or understanding.get("spatial_relations")
        or understanding.get("non_text_information")
    )
    sufficient = bool(
        visual_text
        and state.get("model_complete")
        and state.get("source") == "model"
        and confidence_valid
        and confidence >= 0.85
        and has_layout
    )
    return {
        "sufficient": sufficient,
        "ocr_available": bool(visual_text),
        "visual_model_complete": bool(state.get("model_complete")),
        "visual_model_source": str(state.get("source") or "missing"),
        "confidence": confidence,
        "has_layout": has_layout,
    }


def visual_evidence_state(item: dict[str, Any]) -> dict[str, Any]:
    return _evidence_state(
        item,
        machine_key="visual_understanding",
        corrected_key="human_corrected_visual_understanding",
        review_corrected_key="corrected_visual_understanding",
        content_fields=VISUAL_CONTENT_FIELDS,
    )


def temporal_evidence_state(item: dict[str, Any]) -> dict[str, Any]:
    return _evidence_state(
        item,
        machine_key="temporal_visual_understanding",
        corrected_key="human_corrected_temporal_visual_understanding",
        review_corrected_key="corrected_temporal_visual_understanding",
        content_fields=TEMPORAL_CONTENT_FIELDS,
    )


def understanding_source_selector(item: dict[str, Any], *, temporal: bool = False) -> list[str]:
    """Locate the exact object selected by the existing evidence adapter."""
    key = 'temporal_visual_understanding' if temporal else 'visual_understanding'
    state = temporal_evidence_state(item) if temporal else visual_evidence_state(item)
    if not state.get('understanding'):
        return []
    return {
        'human_corrected': ['human_corrected_' + key],
        'human_review_corrected': ['human_review', 'corrected_' + key],
        'model': [key],
    }.get(state['source'], [])


def visual_text_source_selector(item: dict[str, Any]) -> list[str]:
    """Share the established reader precedence with correction evidence."""
    key = next((key for key in ('human_corrected_visual_text', 'visual_text', 'ocr_text')
                if str(item.get(key) or '').strip()), None)
    return [key] if key else ['structured_visual'] if item.get('structured_visual') else []


def has_complementary_structured_visual(item: dict[str, Any]) -> bool:
    """Keep unique structural content beside OCR, respecting human replacement."""
    return bool(item.get('structured_visual')) and visual_text_source_selector(item) in (
        ['visual_text'], ['ocr_text'])


def visual_recognition_provenance(item: dict[str, Any], selector: list[str | int], *, text: str) -> dict[str, Any]:
    from .evidence_lineage import project_recognition_provenance
    from .canonical_json import canonical_json_sha256
    value: Any = item
    for key in selector:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
            value = value[key]
        else:
            value = None
    if selector == ['visual_text'] and 'visual_text_recognition_provenance' in item:
        record = item['visual_text_recognition_provenance']
        if isinstance(record, dict):
            record = dict(record)
            if record.get('observation_text_sha256') != canonical_json_sha256(text):
                record['problems'] = ['selected_ocr_text_identity_drift']
        value = {'recognition_provenance': record}
    # Never borrow the old machine result's identity for a human-selected field.
    return project_recognition_provenance(value, selected_field=selector, text=text)


def continuous_tag_evidence_state(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("temporal_tag_delta")
    if not isinstance(value, dict) or not value:
        return {"seen": False, "evidence_level": "not_seen", "decision": "not_seen", "consumable_weak": False}
    decision = str(value.get("decision") or "").strip()
    if decision == "coarse_weak_evidence" and value.get("stable_tags"):
        level = "candidate_weak"
    elif decision in {"temporal_multimodal_required", "audit_only_formal_temporal_result"}:
        level = "escalated"
    else:
        level = "unavailable"
    return {
        "seen": True,
        "evidence_level": level,
        "decision": decision or "review_gap",
        "consumable_weak": level == "candidate_weak",
        "stable_tags": [str(tag) for tag in value.get("stable_tags") or [] if str(tag).strip()],
        "coarse_summary": str(value.get("coarse_summary") or ""),
        "run_id": str(value.get("run_id") or ""),
        "value": dict(value),
    }


def weak_visual_language_error(text: str, *, tag_only: bool = False) -> str:
    """Shared conservative wording rule; not a semantic entailment model."""
    if tag_only and any(term in text for term in ("点击", "打开", "关闭", "拖动", "走向", "拿起", "导致", "因为", "随后", "先后", "因此")):
        return 'tag_only_action_or_causality_claim'
    if not any(term in text for term in ("可能", "推断", "疑似", "似乎", "看起来", "画面标签显示")):
        return 'weak_evidence_unqualified'
    return ''


def continuous_tag_language_gate(timeline: Iterable[dict[str, Any]], summary_text: str) -> dict[str, Any]:
    """Reject unqualified or action/causality claims backed only by weak tags."""
    review_match = re.search(r"(?m)^##\s*(?:\d+(?:\.\d+)?\s*)?待复核点", str(summary_text or ""))
    main_text = str(summary_text or "")[: review_match.start()] if review_match else str(summary_text or "")
    violations: list[dict[str, Any]] = []
    weak_claims = 0
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        state = continuous_tag_evidence_state(item)
        if state.get("evidence_level") != "candidate_weak":
            continue
        for claim in state.get("stable_tags") or []:
            weak_claims += 1
            for line in main_text.splitlines():
                if claim not in line:
                    continue
                reason = weak_visual_language_error(line, tag_only=True)
                if reason == 'tag_only_action_or_causality_claim':
                    violations.append(
                        {
                            "timeline_index": int(item.get("index") or position),
                            "kind": "continuous_tag_weak_evidence",
                            "claim": claim,
                            "reason": "tag_only_action_or_causality_claim",
                            "line": line.strip()[:240],
                        }
                    )
                    break
                if reason == 'weak_evidence_unqualified':
                    violations.append(
                        {
                            "timeline_index": int(item.get("index") or position),
                            "kind": "continuous_tag_weak_evidence",
                            "claim": claim,
                            "reason": "weak_evidence_unqualified",
                            "line": line.strip()[:240],
                        }
                    )
                    break
    return {
        "passed": not violations,
        "detail": f"continuous_tag_weak_claims={weak_claims}; violations={len(violations)}",
        "weak_claims": weak_claims,
        "violations": violations[:20],
    }

def panorama_evidence_state(item: dict[str, Any]) -> dict[str, Any]:
    """Return orthogonal spherical evidence without changing legacy visual counts."""
    from .panorama_evidence import aggregate_panorama_evidence

    evidence = aggregate_panorama_evidence(item)
    level = str((evidence.get("semantic") or {}).get("status") or "not_seen")
    complete = (evidence.get("geometric") or {}).get("status") == "complete"
    return {
        "model_complete": bool(complete and level in {"confirmed", "inferred"}),
        "human_review_accepted": False,
        "consumable": bool(complete and level in {"confirmed", "inferred"}),
        "seen": bool(evidence.get("seen")),
        "evidence_level": level,
        "source": "panorama_visual_understanding" if evidence.get("seen") else "missing",
        "understanding": item.get("panorama_visual_understanding") if evidence.get("seen") else {},
        "coverage": evidence,
    }


def has_panorama_evidence(item: dict[str, Any]) -> bool:
    return bool(panorama_evidence_state(item)["consumable"])


def has_visual_evidence(item: dict[str, Any], *, include_human_review: bool = True) -> bool:
    state = visual_evidence_state(item)
    return bool(state["consumable"] if include_human_review else state["model_complete"])


def has_temporal_evidence(item: dict[str, Any], *, include_human_review: bool = True) -> bool:
    state = temporal_evidence_state(item)
    return bool(state["consumable"] if include_human_review else state["model_complete"])


def visual_understanding_value(item: dict[str, Any]) -> dict[str, Any]:
    value = visual_evidence_state(item).get("understanding")
    return dict(value) if isinstance(value, dict) else {}


def temporal_understanding_value(item: dict[str, Any]) -> dict[str, Any]:
    value = temporal_evidence_state(item).get("understanding")
    return dict(value) if isinstance(value, dict) else {}


def evidence_index_sets(timeline: Iterable[dict[str, Any]]) -> dict[str, list[int]]:
    visual_model: list[int] = []
    visual_consumable: list[int] = []
    temporal_model: list[int] = []
    temporal_consumable: list[int] = []
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        index = _positive_int(item.get("index")) or position
        visual = visual_evidence_state(item)
        temporal = temporal_evidence_state(item)
        if visual["model_complete"]:
            visual_model.append(index)
        if visual["consumable"]:
            visual_consumable.append(index)
        if temporal["model_complete"]:
            temporal_model.append(index)
        if temporal["consumable"]:
            temporal_consumable.append(index)
    return {
        "visual_model_complete": visual_model,
        "visual_export_consumable": visual_consumable,
        "temporal_model_complete": temporal_model,
        "temporal_export_consumable": temporal_consumable,
    }


def evidence_level_counts(timeline: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {
        "visual": {"confirmed": 0, "inferred": 0, "unavailable": 0, "not_seen": 0},
        "temporal": {"confirmed": 0, "inferred": 0, "unavailable": 0, "not_seen": 0},
        "continuous_tag": {"candidate_weak": 0, "escalated": 0, "unavailable": 0, "not_seen": 0},
    }
    for item in timeline:
        if not isinstance(item, dict):
            continue
        counts["visual"][str(visual_evidence_state(item)["evidence_level"])] += 1
        counts["temporal"][str(temporal_evidence_state(item)["evidence_level"])] += 1
        counts["continuous_tag"][str(continuous_tag_evidence_state(item)["evidence_level"])] += 1
    return counts


def human_review_accepted(item: dict[str, Any]) -> bool:
    review = _human_review(item)
    status = str(item.get("review_status") or review.get("status") or "").strip().lower()
    return status in ACCEPTED_REVIEW_STATUSES


def _evidence_state(
    item: dict[str, Any],
    *,
    machine_key: str,
    corrected_key: str,
    review_corrected_key: str,
    content_fields: tuple[str, ...],
) -> dict[str, Any]:
    review = _human_review(item)
    candidates = (
        ("human_corrected", item.get(corrected_key)),
        ("human_review_corrected", review.get(review_corrected_key)),
        ("model", item.get(machine_key)),
    )
    selected: dict[str, Any] = {}
    source = "missing"
    observed_level = "not_seen"
    for candidate_source, value in candidates:
        if not isinstance(value, dict) or not value:
            continue
        level = _normalised_evidence_level(value)
        if level == "unavailable":
            selected = dict(value)
            source = candidate_source
            observed_level = level
            break
        if _valid_understanding(value, item=item, content_fields=content_fields):
            selected = dict(value)
            source = candidate_source
            observed_level = level
            break
    model_complete = bool(selected) and observed_level in {"confirmed", "inferred"}
    review_accepted = human_review_accepted(item)
    return {
        "model_complete": model_complete,
        "human_review_accepted": review_accepted,
        "consumable": model_complete or (review_accepted and observed_level != "unavailable"),
        "seen": observed_level != "not_seen",
        "evidence_level": observed_level,
        "inference_basis": str(selected.get("inference_basis") or ""),
        "unavailable_reason": str(selected.get("unavailable_reason") or ""),
        "source": source if selected else ("human_review_status" if review_accepted else "missing"),
        "understanding": selected,
    }


def _normalised_evidence_level(value: dict[str, Any]) -> str:
    level = str(value.get("evidence_level") or "").strip().lower()
    if level in {"confirmed", "inferred", "unavailable"}:
        return level
    # Historical model/human results predate evidence_level and remain compatible.
    return "confirmed"


def _valid_understanding(
    value: Any,
    *,
    item: dict[str, Any],
    content_fields: tuple[str, ...],
) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    if value.get("parse_failed") is True:
        return False
    if str(value.get("validation_status") or "").strip().lower() in {
        "blocked",
        "failed",
        "incomplete",
        "invalid",
    }:
        return False
    has_content = any(_non_empty(value.get(field)) for field in content_fields)
    has_frame_evidence = bool(_as_list(value.get("evidence_frame_paths"))) or _item_has_frame_evidence(item)
    legacy_summary = not str(value.get("evidence_level") or "").strip() and _non_empty(value.get("summary"))
    return (has_content and has_frame_evidence) or legacy_summary


def _item_has_frame_evidence(item: dict[str, Any]) -> bool:
    for key in (
        "frame_path",
        "frame_paths",
        "assets",
        "evidence_frame_paths",
        "temporal_frame_paths",
    ):
        values = item.get(key)
        rows = values if isinstance(values, list) else [values]
        for row in rows:
            if isinstance(row, dict):
                if _non_empty(row.get("path")) or _non_empty(row.get("source_path")):
                    return True
            elif _non_empty(row):
                return True
    integrated = item.get("integrated_visual")
    if isinstance(integrated, dict):
        return bool(_as_list(integrated.get("evidence_frame_paths")))
    return False


def _human_review(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("human_review")
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for item in value if _non_empty(item)]
    if _non_empty(value):
        return [value]
    return []


def _non_empty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _positive_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0
