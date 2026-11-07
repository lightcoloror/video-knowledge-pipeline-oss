from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

import jieba
from jsonschema import Draft202012Validator
from rapidfuzz import fuzz

from .summary_citation_audit import build_citation_index
from .structured_claim_verification import CLAIMS_SCHEMA, audit_structured_claims


SCHEMA = "video_knowledge_pipeline.smart_summary_reader_plan.v1"

_TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s*(?:-|–|—|至)\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)"
)
_COMPACT_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_META_PATTERNS = (
    "这份总结围绕",
    "章节修订",
    "章节改写",
    "根据输入json",
    "根据章节map",
    "以下是总结",
    "好的，这是",
    "作为ai",
    "source_arbitrated_transcript",
    "arbitrated_or_reviewed",
    "evidence_id",
    "source_kind",
    "schema:",
    "workflow",
    "bundle",
)
_GENERIC_TITLE_FRAGMENTS = (
    "章节",
    "这一段",
    "这部分",
    "内容总结",
    "课程总结",
    "我们来看看",
    "接下来",
)
_ACTION_VERBS = (
    "确认",
    "整理",
    "记录",
    "写下",
    "检查",
    "执行",
    "复盘",
    "准备",
    "建立",
    "选择",
    "联系",
    "发送",
    "完成",
    "组织",
    "提供",
    "开放",
    "发布",
    "通过",
    "利用",
    "引导",
    "学习",
    "主动",
    "设置",
    "拍摄",
    "分享",
    "邀约",
    "强制",
    "补充",
    "比较",
    "保留",
    "避免",
    "制定",
    "拆分",
    "跟进",
    "验证",
    "更新",
    "提交",
    "先",
    "把",
    "将",
    "保持",
)


_CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "text", "time_ranges", "evidence_ids", "source_modalities"],
    "properties": {
        "title": {"type": "string", "minLength": 2, "maxLength": 36},
        "text": {"type": "string", "minLength": 8, "maxLength": 360},
        "time_ranges": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 8, "maxLength": 40},
        },
        "evidence_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 256,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 512},
        },
        "source_modalities": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
    },
}


READER_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "source_section_ids",
        "overview",
        "core_insights",
        "themes",
        "principles",
        "actions",
        "reusable_expressions",
        "review_items",
    ],
    "properties": {
        "schema": {"const": SCHEMA},
        "structured_claims": CLAIMS_SCHEMA,
        "source_section_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "overview": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text", "evidence_ids"],
            "properties": {
                "text": {"type": "string", "minLength": 24, "maxLength": 240},
                "evidence_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 256,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1, "maxLength": 512},
                },
            },
        },
        "core_insights": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": _CLAIM_SCHEMA,
        },
        "themes": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "title",
                    "time_range",
                    "summary",
                    "problem",
                    "reason",
                    "method",
                    "case",
                    "action",
                    "evidence_ids",
                    "source_modalities",
                ],
                "properties": {
                    "title": {"type": "string", "minLength": 2, "maxLength": 36},
                    "time_range": {"type": "string", "minLength": 8, "maxLength": 40},
                    "summary": {"type": "string", "minLength": 16, "maxLength": 520},
                    "problem": {"type": "string", "maxLength": 240},
                    "reason": {"type": "string", "maxLength": 240},
                    "method": {"type": "string", "maxLength": 320},
                    "case": {"type": "string", "maxLength": 320},
                    "action": {"type": "string", "maxLength": 240},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 256,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1, "maxLength": 512},
                    },
                    "source_modalities": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        "principles": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": _CLAIM_SCHEMA,
        },
        "actions": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "time_range", "evidence_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 4, "maxLength": 220},
                    "time_range": {"type": "string", "minLength": 8, "maxLength": 40},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 256,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1, "maxLength": 512},
                    },
                },
            },
        },
        "reusable_expressions": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "kind", "time_range", "evidence_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 4, "maxLength": 220},
                    "kind": {"enum": ["verbatim_quote", "reusable_expression"]},
                    "time_range": {"type": "string", "minLength": 8, "maxLength": 40},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 256,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1, "maxLength": 512},
                    },
                },
            },
        },
        "review_items": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "time_range", "evidence_ids", "missing_evidence"],
                "properties": {
                    "text": {"type": "string", "minLength": 4, "maxLength": 260},
                    "time_range": {"type": "string", "maxLength": 40},
                    "evidence_ids": {
                        "type": "array",
                        "maxItems": 256,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1, "maxLength": 512},
                    },
                    "missing_evidence": {"type": "string", "minLength": 2, "maxLength": 180},
                },
            },
        },
    },
}


def reader_plan_prompt_contract() -> str:
    """Return a compact provider contract; full validation remains local."""

    claim = {
        "title": "语义标题",
        "text": "证据化结论",
        "time_ranges": ["HH:MM:SS.mmm - HH:MM:SS.mmm"],
        "evidence_ids": ["existing-id"],
        "source_modalities": ["asr"],
    }
    contract = {
        "schema": SCHEMA,
        "source_section_ids": ["all-input-section-ids"],
        "overview": {"text": "24-240字读者概览", "evidence_ids": ["existing-id"]},
        "core_insights": [claim, "3-8 items"],
        "themes": [
            {
                "title": "语义标题",
                "time_range": "ordered non-overlapping range",
                "summary": "主题摘要",
                "problem": "可空",
                "reason": "可空",
                "method": "可空",
                "case": "可空",
                "action": "可空",
                "evidence_ids": ["existing-id"],
                "source_modalities": ["asr"],
            },
            "3-8 items",
        ],
        "principles": [claim, "3-8 items"],
        "actions": [{"text": "动作", "time_range": "range", "evidence_ids": ["existing-id"]}],
        "reusable_expressions": [
            {
                "text": "原句或转述",
                "kind": "verbatim_quote|reusable_expression",
                "time_range": "range",
                "evidence_ids": ["existing-id"],
            }
        ],
        "review_items": [
            {"text": "缺口", "time_range": "range或空", "evidence_ids": [], "missing_evidence": "缺少什么"}
        ],
    }
    return json.dumps(contract, ensure_ascii=False, separators=(",", ":"))


def parse_reader_plan(value: str) -> dict[str, Any]:
    """Parse raw model output, accepting only one JSON object and no prose wrapper."""

    raw = str(value or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, count=1, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw, count=1)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "plan": {}, "errors": [f"invalid_json:{exc.msg}@{exc.pos}"]}
    if not isinstance(parsed, dict):
        return {"ok": False, "plan": {}, "errors": ["reader_plan_must_be_an_object"]}
    return {"ok": True, "plan": parsed, "errors": []}


def normalize_reader_plan_candidate(
    plan: dict[str, Any],
    *,
    fact_pack: dict[str, Any],
) -> dict[str, Any]:
    """Apply lossless provider-output repairs before the strict quality gate.

    Intent: salvage a complete paid response without inventing missing content.
    Decision: remove internal meta review rows, remove external-fact-check requests
    for already evidence-bound speaker claims, remove non-action statements from the
    action list, and downgrade unverifiable verbatim quotes to reusable expressions.
    Reason: these are contract-label errors, not missing semantic content.
    Evidence: the reader-plan gate already detects all three conditions against the
    exact chapter fact pack and evidence snippets.
    Effective scope: one parsed Smart Summary reader plan; original evidence,
    transcript, Timeline and provider response remain immutable.
    """

    normalized = json.loads(json.dumps(plan, ensure_ascii=False))
    citation_index = build_citation_index(fact_pack)
    eligible = citation_index.eligible if not citation_index.errors else set()
    repairs: list[dict[str, Any]] = []
    content_profile = str(fact_pack.get("content_profile") or "course_or_general")

    overview = normalized.get("overview")
    if isinstance(overview, dict):
        original_overview = str(overview.get("text") or "").strip()
        if len(original_overview) > 240:
            # Intent: preserve a complete local/paid Reduce response when the
            # provider writes a paragraph where the contract asks for a short
            # reader-facing overview.
            # Decision: keep whole leading sentences up to the schema limit;
            # all omitted detail remains in evidence-bound insights/themes.
            # Reason: this is a presentation-length repair, not a new claim or
            # a substitute for missing evidence.
            # Evidence: the 2026-08-10 local Qwen interview Reduce passed every
            # provenance gate and failed only overview.text maxLength=240.
            # Effective scope: overview display text only; the immutable raw
            # response, chapter fact pack, transcript and evidence are intact.
            overview["text"] = _bounded_overview(original_overview, limit=240)
            repairs.append(
                {
                    "kind": "compress_overlong_overview",
                    "location": "overview.text",
                    "before_chars": len(original_overview),
                    "after_chars": len(str(overview["text"])),
                }
            )

    if _is_interview_profile(content_profile):
        repairs.extend(_anonymize_unbound_interview_identities(normalized, fact_pack=fact_pack))
        repairs.extend(_normalize_interview_theme_fields(normalized.get("themes") or []))

    repairs.extend(_partition_overlapping_theme_ranges(normalized.get("themes") or []))

    review_items: list[dict[str, Any]] = []
    for index, item in enumerate(normalized.get("review_items") or [], start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        compact = _compact(text)
        if any(_compact(pattern) in compact for pattern in _META_PATTERNS):
            repairs.append(
                {
                    "kind": "drop_internal_meta_review",
                    "location": f"review_items[{index}]",
                }
            )
            continue
        ids = {str(value) for value in item.get("evidence_ids") or []}
        missing = _compact(str(item.get("missing_evidence") or ""))
        if ids & eligible and "外部" in missing and ("验证" in missing or "核实" in missing):
            repairs.append(
                {
                    "kind": "drop_external_fact_check_for_speaker_claim",
                    "location": f"review_items[{index}]",
                }
            )
            continue
        review_items.append(item)
    normalized["review_items"] = review_items

    actions: list[dict[str, Any]] = []
    for index, item in enumerate(normalized.get("actions") or [], start=1):
        if not isinstance(item, dict):
            continue
        if _is_interview_profile(content_profile) and _is_prescriptive_health_or_insurance_action(
            str(item.get("text") or "")
        ):
            repairs.append(
                {
                    "kind": "drop_interview_prescriptive_action",
                    "location": f"actions[{index}]",
                }
            )
            continue
        if _looks_actionable(str(item.get("text") or "")):
            actions.append(item)
            continue
        repairs.append(
            {
                "kind": "drop_non_action_item",
                "location": f"actions[{index}]",
            }
        )
    normalized["actions"] = actions

    for index, item in enumerate(
        normalized.get("reusable_expressions") or [], start=1
    ):
        if not isinstance(item, dict) or item.get("kind") != "verbatim_quote":
            continue
        ids = [str(value) for value in item.get("evidence_ids") or []]
        if citation_index.quote_matches(str(item.get("text") or ""), ids):
            continue
        item["kind"] = "reusable_expression"
        repairs.append(
            {
                "kind": "downgrade_unverified_verbatim_quote",
                "location": f"reusable_expressions[{index}]",
            }
        )

    return {
        "plan": normalized,
        "repairs": repairs,
        "repair_count": len(repairs),
    }


def _bounded_overview(value: str, *, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    sentences = [item.strip() for item in re.split(r"(?<=[。！？!?])", text) if item.strip()]
    kept: list[str] = []
    for sentence in sentences:
        candidate = "".join(kept) + sentence
        if len(candidate) > limit:
            break
        kept.append(sentence)
    bounded = "".join(kept).strip()
    if len(bounded) >= 24:
        return bounded
    clipped = text[:limit].rstrip("，、；：,;: ")
    return clipped if clipped.endswith(("。", "！", "？", "!", "?")) else clipped + "。"


def _anonymize_unbound_interview_identities(
    plan: dict[str, Any],
    *,
    fact_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    source = _compact(
        "".join(
            str(ref.get("snippet") or "")
            for section in fact_pack.get("sections") or []
            if isinstance(section, dict)
            for ref in section.get("evidence_refs") or []
            if isinstance(ref, dict)
        )
    )
    pattern = re.compile(r"受访者[\u4e00-\u9fff]{1,3}(?:女士|先生)|[\u4e00-\u9fff]{1,3}(?:女士|先生)")
    replacements: set[str] = set()

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, str):
            return value

        def replace(match: re.Match[str]) -> str:
            identity = match.group(0)
            compact_identity = _compact(identity.replace("受访者", ""))
            if compact_identity and compact_identity in source:
                return identity
            replacements.add(identity)
            return "受访者"

        return pattern.sub(replace, value)

    updated = visit(plan)
    plan.clear()
    plan.update(updated)
    return [
        {
            "kind": "anonymize_unbound_interview_identity",
            "location": "reader_plan",
            "identity": identity,
        }
        for identity in sorted(replacements)
    ]


def _is_prescriptive_health_or_insurance_action(value: str) -> bool:
    compact = _compact(value)
    high_stakes = any(
        token in compact
        for token in ("手术", "放疗", "治疗", "用药", "投保", "保险", "理赔", "报销")
    )
    prescriptive = any(
        token in compact
        for token in ("优先", "应该", "应当", "建议", "必须", "无需", "避免延误", "即刻执行")
    )
    return high_stakes and prescriptive


def _normalize_interview_theme_fields(
    themes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep interview experiences distinct from reader-facing advice.

    Intent: prevent a valid evidence-bound personal treatment or insurance choice
    from becoming prescriptive guidance merely because the reader schema uses
    ``method`` and ``action`` fields.
    Decision: remove prescriptive high-stakes theme actions and explicitly attribute
    high-stakes methods to the interviewee when the provider omitted attribution.
    Reason: the underlying account remains useful evidence, but its speech act is a
    personal experience rather than a recommendation to the reader.
    Evidence: the 2026-08-10 interview Reduce rendered "优先选择非手术方案" and
    "若经济无虞，应优先..." despite the top-level action list being empty.
    Effective scope: deterministic normalization of interview reader plans only;
    raw model output, transcript and chapter facts remain immutable.
    """

    repairs: list[dict[str, Any]] = []
    for index, theme in enumerate(themes, start=1):
        if not isinstance(theme, dict):
            continue
        action = str(theme.get("action") or "").strip()
        if action and _is_prescriptive_health_or_insurance_action(action):
            theme["action"] = ""
            repairs.append(
                {
                    "kind": "drop_interview_prescriptive_theme_action",
                    "location": f"themes[{index}].action",
                }
            )

        method = str(theme.get("method") or "").strip()
        if not method or not _mentions_health_or_insurance_choice(method):
            continue
        if any(token in method for token in ("受访者", "患者", "客户", "采访者", "讲者")):
            continue
        theme["method"] = f"受访者当时的个人选择是：{method}"
        repairs.append(
            {
                "kind": "attribute_interview_high_stakes_method",
                "location": f"themes[{index}].method",
            }
        )
    return repairs


def _mentions_health_or_insurance_choice(value: str) -> bool:
    compact = _compact(value)
    domain = any(
        token in compact
        for token in ("手术", "放疗", "治疗", "用药", "投保", "保险", "理赔", "报销")
    )
    choice = any(
        token in compact
        for token in ("选择", "决定", "倾向", "采用", "办理", "申请", "补办", "支付")
    )
    return domain and choice


def _partition_overlapping_theme_ranges(
    themes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Partition ordered overlapping theme windows without inventing content.

    Intent: make reader themes mutually exclusive when upstream chapter windows
    overlap or carry a stale zero start.
    Decision: preserve row order and the union of adjacent windows; when the next
    start falls inside the previous window, assign the overlap to the next theme,
    while a containing/stale-zero next window starts at the previous end.
    Reason: semantic chapter Map inputs intentionally overlap, but the reader-plan
    contract requires ordered, non-overlapping navigation ranges.
    Evidence: each changed range remains bounded by the two provider-supplied
    ranges; text, evidence IDs and source artifacts are untouched.
    Effective scope: provider-returned theme display ranges only.
    """

    repairs: list[dict[str, Any]] = []
    for index in range(1, len(themes)):
        previous = themes[index - 1]
        current = themes[index]
        if not isinstance(previous, dict) or not isinstance(current, dict):
            continue
        previous_range = _parse_time_range(str(previous.get("time_range") or ""))
        current_range = _parse_time_range(str(current.get("time_range") or ""))
        if not previous_range or not current_range:
            continue
        previous_start, previous_end = previous_range
        current_start, current_end = current_range
        if current_start >= previous_end - 0.5:
            continue
        if current_start > previous_start and current_start < previous_end:
            old_value = str(previous.get("time_range") or "")
            previous["time_range"] = _format_time_range(previous_start, current_start)
            repairs.append(
                {
                    "kind": "partition_overlapping_theme_ranges",
                    "location": f"themes[{index}]",
                    "before": old_value,
                    "after": previous["time_range"],
                }
            )
            continue
        if current_end > previous_end:
            old_value = str(current.get("time_range") or "")
            current["time_range"] = _format_time_range(previous_end, current_end)
            repairs.append(
                {
                    "kind": "clamp_containing_theme_start",
                    "location": f"themes[{index + 1}]",
                    "before": old_value,
                    "after": current["time_range"],
                }
            )
    return repairs


def _format_time_range(start: float, end: float) -> str:
    return f"{_format_time(start)} - {_format_time(end)}"


def _format_time(value: float) -> str:
    milliseconds = max(0, int(round(float(value) * 1000)))
    hour, remainder = divmod(milliseconds, 3_600_000)
    minute, remainder = divmod(remainder, 60_000)
    second, millisecond = divmod(remainder, 1000)
    return f"{hour:02d}:{minute:02d}:{second:02d}.{millisecond:03d}"

def validate_reader_plan(
    plan: dict[str, Any],
    *,
    fact_pack: dict[str, Any],
    expected_section_ids: set[str],
) -> dict[str, Any]:
    """Validate shape, provenance, chronology and reader-level semantic maturity.

    Intent: prevent fluent-but-immature summaries from entering the canonical export.
    Decision: apply the same validate-before-consume branch used by Haystack's
    JsonSchemaValidator, then add VKP evidence and chronology gates.
    Reason: JSON syntax alone cannot prove that claims are bound to the supplied
    chapter facts or that the chapter plan is coherent.
    Evidence: reviewed Haystack commit acbf725a and VKP fact-pack contract v1.
    Effective scope: global Smart Summary Reduce output only; transcript, Timeline,
    chapter Map artifacts and provider routing remain unchanged.
    """

    errors: list[str] = []
    validator = Draft202012Validator(READER_PLAN_JSON_SCHEMA)
    for error in sorted(validator.iter_errors(plan), key=lambda row: list(row.absolute_path)):
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"schema:{path}:{error.message}")
    if errors:
        return _validation_result(errors, plan)

    actual_sections = {str(value) for value in plan.get("source_section_ids") or []}
    if actual_sections != expected_section_ids:
        missing = sorted(expected_section_ids - actual_sections)
        extra = sorted(actual_sections - expected_section_ids)
        errors.append(f"source_section_ids_mismatch:missing={missing};extra={extra}")

    citation_index = build_citation_index(fact_pack)
    errors.extend(citation_index.errors)
    eligible, review_only = citation_index.eligible, citation_index.review_only
    text_fields = dict(_plan_text_values(plan))
    for location, item in _evidence_bound_items(plan):
        ids = {str(value) for value in item.get("evidence_ids") or []}
        unknown = sorted(ids - eligible - review_only)
        if unknown:
            errors.append(f"unknown_evidence:{location}:{unknown}")
        if ids and not (ids & eligible):
            errors.append(f"review_only_evidence_promoted:{location}:{sorted(ids)}")
        leaves = citation_index.resolved_ids(ids, usable_only=True)
        if leaves and all(citation_index.refs[name].get('evidence_level') in {'inferred', 'candidate_weak'} for name in leaves):
            from .visual_evidence import weak_visual_language_error
            tag_only = all(citation_index.refs[name].get('evidence_level') == 'candidate_weak' for name in leaves)
            for field, text in text_fields.items():
                if field.rsplit('.', 1)[0] == location and not field.endswith('.title'):
                    reason = weak_visual_language_error(text, tag_only=tag_only)
                    if text.strip() and reason:
                        errors.append(f'{reason}:{field}')

    for index, item in enumerate(plan.get("review_items") or [], start=1):
        ids = {str(value) for value in item.get("evidence_ids") or []}
        unknown = sorted(ids - eligible - review_only)
        if unknown:
            errors.append(f"unknown_review_evidence:review_items[{index}]:{unknown}")

    errors.extend(_theme_range_errors(plan.get("themes") or []))
    errors.extend(_semantic_text_errors(plan))
    errors.extend(_duplicate_claim_errors(plan))

    source_has_actions = not _is_interview_profile(
        str(fact_pack.get("content_profile") or "")
    ) and any(
        str(fact.get("fact_type") or "") == "actions"
        for section in fact_pack.get("sections") or []
        if isinstance(section, dict)
        for fact in section.get("facts") or []
        if isinstance(fact, dict) and str(fact.get("fact_status") or "") != "review_gap_not_fact"
    )
    if source_has_actions and not plan.get("actions"):
        errors.append("source_actions_were_dropped")
    for index, item in enumerate(plan.get("actions") or [], start=1):
        if not _looks_actionable(str(item.get("text") or "")):
            errors.append(f"non_action_item:actions[{index}]")

    for index, item in enumerate(plan.get("reusable_expressions") or [], start=1):
        if item.get("kind") != "verbatim_quote":
            continue
        ids = [str(value) for value in item.get("evidence_ids") or []]
        if not citation_index.quote_matches(str(item.get("text") or ""), ids):
            errors.append(f"verbatim_quote_not_found:reusable_expressions[{index}]")

    structured_audit: dict[str, Any] | None = None
    if "claim_verification" in fact_pack or "structured_claims" in plan:
        output_texts = {location: text for location, text in _plan_text_values(plan)
                        if not location.startswith("review_items[")}
        parent_citations = {location: list(item.get("evidence_ids") or [])
                            for location, item in _evidence_bound_items(plan)}
        output_citations = {location: parent_citations.get(location.rsplit(".", 1)[0], [])
                            for location in output_texts}
        structured_audit = audit_structured_claims(
            plan.get("structured_claims"), scope=fact_pack.get("claim_verification"),
            citation_index=citation_index, output_texts=output_texts, output_citations=output_citations)
        errors.extend(structured_audit["errors"])
        structured_audit["unassessed_output_locations"] = sorted(
            output_texts.keys() - set(structured_audit.get("required_locations", [])))
        for claim in structured_audit["claims"]:
            if claim["status"] != "supported":
                errors.append(f"structured_claim_not_supported:{claim['claim_id']}:{claim['status']}")
    result = {**_validation_result(errors, plan),
              "citation_audit": citation_index.audit(_evidence_bound_items(plan))}
    if structured_audit is not None:
        result["structured_claim_audit"] = structured_audit
    return result


def render_reader_summary(
    plan: dict[str, Any],
    *,
    title: str,
    first_time: str,
    last_time: str,
    content_profile: str = "",
) -> str:
    """Render validated content into deterministic reader Markdown."""

    lines = [f"# {title} - 智能总结", "", "<!-- codex_llm_rewrite_final -->", "", "## 基本信息"]
    lines.extend([f"- 标题：{title}", f"- 内容范围：{first_time} - {last_time}", "", "## 一句话概览", str(plan["overview"]["text"]).strip()])

    explicit_interview = _is_interview_profile(content_profile)
    is_interview = explicit_interview or "采访" in str(title)
    lines.extend(["", "## 核心主题 / 事实主线" if explicit_interview else ("## 核心主题 / 内容主线" if is_interview else "## 核心主题 / 课程主线")])
    for item in plan.get("core_insights") or []:
        times = "、".join(item.get("time_ranges") or [])
        lines.append(f"- **{item['title']}**（{times}）：{item['text']}")

    lines.extend(["", "## 事实时间线" if explicit_interview else "## 分段总结"])
    for theme in plan.get("themes") or []:
        lines.extend(["", f"### {theme['title']}（{theme['time_range']}）", str(theme["summary"]).strip()])
        labels = (
            (("问题", "problem"), ("原因", "reason"), ("个人选择/经历", "method"), ("案例", "case"), ("明确后续动作", "action"))
            if is_interview
            else (("问题", "problem"), ("原因", "reason"), ("方法", "method"), ("案例", "case"), ("行动", "action"))
        )
        for label, key in labels:
            value = str(theme.get(key) or "").strip()
            if value:
                lines.append(f"- {label}：{value}")

    lines.extend(["", "## 受访者原话与感受" if explicit_interview else "## 关键观点 / 方法论"])
    for item in plan.get("principles") or []:
        times = "、".join(item.get("time_ranges") or [])
        lines.append(f"- **{item['title']}**（{times}）：{item['text']}")

    lines.extend(["", "## 明确后续事项" if explicit_interview else "## 可执行动作清单"])
    actions = plan.get("actions") or []
    lines.extend(
        [f"- {item['time_range']}：{item['text']}" for item in actions]
        or ["- 未从当前证据中识别到明确行动项。"]
    )

    lines.extend(["", "## 原话摘录" if explicit_interview else "## 高频话术 / 可复用表达"])
    expressions = plan.get("reusable_expressions") or []
    for item in expressions:
        text = str(item["text"]).strip()
        rendered = f"“{text.strip('“”')}”" if item.get("kind") == "verbatim_quote" else text
        lines.append(f"- {item['time_range']}：{rendered}")
    if not expressions:
        lines.append("- 无可确认原句；不生成伪金句。")

    lines.extend(["", "## 待核实事项 / 隐私与发布边界" if explicit_interview else "## 待复核点 / 低置信内容"])
    review_items = plan.get("review_items") or []
    for item in review_items:
        time_range = str(item.get("time_range") or "未定位")
        lines.append(f"- {time_range}：{item['text']}（缺少：{item['missing_evidence']}）")
    if not review_items:
        lines.append("- 无。")
    return "\n".join(lines).strip() + "\n"


def evaluate_reader_markdown_semantics(content: str) -> dict[str, Any]:
    """Detect known fluent-but-immature reader-summary patterns offline."""

    text = str(content or "")
    problems: list[str] = []
    compact = _compact(text)
    for value in _META_PATTERNS:
        if _compact(value) in compact:
            problems.append(f"internal_or_meta_language:{value}")

    overview = _markdown_section(text, "一句话概览")
    if len(_compact(overview)) < 20:
        problems.append("overview_too_thin")
    if any(_compact(value) in _compact(overview) for value in _META_PATTERNS):
        problems.append("overview_describes_generation_instead_of_content")

    segment = _markdown_section(text, "事实时间线") or _markdown_section(text, "分段总结")
    headings = re.findall(r"(?m)^###\s+(.+?)\s*$", segment)
    for heading in headings:
        title = _TIME_RANGE_RE.sub("", heading).strip(" （()）-—")
        if not _meaningful_title(title):
            problems.append(f"weak_theme_title:{heading[:80]}")
    ranges = [match.group(0) for match in _TIME_RANGE_RE.finditer(segment)]
    problems.extend(_ordered_range_errors(ranges, prefix="markdown_theme"))

    action_section = _markdown_section(text, "明确后续事项") or _markdown_section(text, "可执行动作清单")
    action_rows = [re.sub(r"^\s*[-*]\s*", "", line).strip() for line in action_section.splitlines() if re.match(r"^\s*[-*]\s+", line)]
    for index, row in enumerate(action_rows, start=1):
        if "未从当前证据" in row:
            continue
        action_text = _TIME_RANGE_RE.sub("", row, count=1).lstrip(":： -—")
        action_text = re.sub(r"^\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?\s*[:：]\s*", "", action_text)
        if not _looks_actionable(action_text):
            problems.append(f"non_action_item:markdown[{index}]")

    return {
        "passed": not problems,
        "problems": problems,
        "checks": {
            "meta_language_absent": not any(value.startswith("internal_or_meta_language") for value in problems),
            "overview_is_reader_facing": "overview_describes_generation_instead_of_content" not in problems and "overview_too_thin" not in problems,
            "theme_titles_are_meaningful": not any(value.startswith("weak_theme_title") for value in problems),
            "theme_ranges_are_ordered": not any(value.startswith("markdown_theme") for value in problems),
            "actions_are_actionable": not any(value.startswith("non_action_item") for value in problems),
        },
    }


def _validation_result(errors: list[str], plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": not errors,
        "validation_scope": "reader_contract_and_reference_integrity",
        "semantic_support": "not_assessed",
        "errors": errors,
        "error_count": len(errors),
        "source_section_count": len(plan.get("source_section_ids") or []),
        "core_insight_count": len(plan.get("core_insights") or []),
        "theme_count": len(plan.get("themes") or []),
        "principle_count": len(plan.get("principles") or []),
        "action_count": len(plan.get("actions") or []),
        "expression_count": len(plan.get("reusable_expressions") or []),
        "review_item_count": len(plan.get("review_items") or []),
    }


def _evidence_bound_items(plan: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    yield "overview", plan.get("overview") or {}
    for key in ("core_insights", "themes", "principles", "actions", "reusable_expressions"):
        for index, item in enumerate(plan.get(key) or [], start=1):
            if isinstance(item, dict):
                yield f"{key}[{index}]", item


def _theme_range_errors(themes: list[dict[str, Any]]) -> list[str]:
    return _ordered_range_errors([str(row.get("time_range") or "") for row in themes], prefix="theme")


def _ordered_range_errors(values: list[str], *, prefix: str) -> list[str]:
    errors: list[str] = []
    previous_end = -1.0
    for index, value in enumerate(values, start=1):
        parsed = _parse_time_range(value)
        if not parsed:
            errors.append(f"{prefix}_invalid_time_range[{index}]:{value}")
            continue
        start, end = parsed
        if end <= start:
            errors.append(f"{prefix}_non_positive_range[{index}]:{value}")
        if previous_end >= 0 and start < previous_end - 0.5:
            errors.append(f"{prefix}_overlap_or_out_of_order[{index}]:{value}")
        previous_end = max(previous_end, end)
    return errors


def _parse_time_range(value: str) -> tuple[float, float] | None:
    match = _TIME_RANGE_RE.search(str(value or ""))
    if not match:
        return None
    return _time_seconds(match.group("start")), _time_seconds(match.group("end"))


def _time_seconds(value: str) -> float:
    hour, minute, second = str(value).split(":", 2)
    return int(hour) * 3600 + int(minute) * 60 + float(second)


def _semantic_text_errors(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for location, text in _plan_text_values(plan):
        compact = _compact(text)
        for pattern in _META_PATTERNS:
            if _compact(pattern) in compact:
                errors.append(f"meta_language:{location}:{pattern}")
    for key in ("core_insights", "themes", "principles"):
        for index, item in enumerate(plan.get(key) or [], start=1):
            if not _meaningful_title(str(item.get("title") or "")):
                errors.append(f"weak_title:{key}[{index}]")
    return errors


def _plan_text_values(plan: dict[str, Any]) -> Iterable[tuple[str, str]]:
    yield "overview.text", str((plan.get("overview") or {}).get("text") or "")
    for key in ("core_insights", "themes", "principles", "actions", "reusable_expressions", "review_items"):
        for index, item in enumerate(plan.get(key) or [], start=1):
            if not isinstance(item, dict):
                continue
            for field in ("title", "text", "summary", "problem", "reason", "method", "case", "action"):
                if item.get(field):
                    yield f"{key}[{index}].{field}", str(item[field])


def _meaningful_title(value: str) -> bool:
    title = str(value or "").strip()
    compact = _compact(title)
    if len(compact) < 4 or len(title) > 36:
        return False
    if title.endswith(("。", "，", ",", "；", ";", "…", "?", "？")):
        return False
    return not any(_compact(fragment) in compact for fragment in _GENERIC_TITLE_FRAGMENTS)


def _duplicate_claim_errors(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("core_insights", "principles"):
        rows = [f"{item.get('title', '')} {item.get('text', '')}" for item in plan.get(key) or []]
        tokenized = [_tokenized(value) for value in rows]
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                if fuzz.token_set_ratio(tokenized[left], tokenized[right]) >= 94:
                    errors.append(f"duplicate_claim:{key}[{left + 1}]~{key}[{right + 1}]")
    return errors


def _tokenized(value: str) -> str:
    jieba.setLogLevel(logging.ERROR)
    return " ".join(token.strip() for token in jieba.lcut(str(value or ""), HMM=False) if _compact(token))


def _looks_actionable(value: str) -> bool:
    compact = _compact(value)
    if len(compact) < 4:
        return False
    head = compact[:16]
    return any(_compact(verb) in head for verb in _ACTION_VERBS)


def _is_interview_profile(value: str) -> bool:
    return str(value or "").strip().lower() in {
        "interview",
        "interview-v1",
        "medical-insurance-interview-v1",
    }


def _compact(value: str) -> str:
    return _COMPACT_RE.sub("", str(value or "")).lower()


def _markdown_section(content: str, heading: str) -> str:
    match = re.search(rf"(?m)^##\s+{re.escape(heading)}(?:\s*/[^\n]+)?\s*$", str(content or ""))
    if not match:
        return ""
    tail = str(content or "")[match.end() :]
    next_heading = re.search(r"(?m)^##\s+", tail)
    return tail[: next_heading.start()] if next_heading else tail
