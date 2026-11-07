from __future__ import annotations

from typing import Any


from .canonical_json import canonical_json_sha256

SCHEMA = "video_knowledge_pipeline.scene_taxonomy.v1"
QUALITY_SCHEMA = "video_knowledge_pipeline.shot_quality_explanation.v1"
UPSTREAM_PROJECT = "Encorebao/video-pilot"
UPSTREAM_COMMIT = "eaf7434a26c0ce235c4097439c11ad1fa5232b62"

_FIELDS: dict[str, dict[str, Any]] = {
    "shot_type": {
        "label": "shot type",
        "values": {
            "wide": ["wide", "wide shot", "long shot", "\u8fdc\u666f", "\u5168\u666f"],
            "medium": ["medium", "medium shot", "\u4e2d\u666f", "\u4e2d\u8fd1\u666f"],
            "close_up": ["close-up", "close up", "closeup", "\u8fd1\u666f", "\u7279\u5199"],
            "extreme_close_up": ["extreme close-up", "macro", "\u6781\u7279\u5199", "\u5fae\u8ddd"],
            "unknown": ["unknown", "uncertain", "\u4e0d\u786e\u5b9a", "\u672a\u77e5"],
        },
    },
    "camera_movement": {
        "label": "camera movement",
        "values": {
            "static": ["static", "locked", "\u56fa\u5b9a\u955c\u5934", "\u9759\u6b62"],
            "pan_or_tilt": ["pan", "tilt", "\u6447\u955c\u5934", "\u4fef\u4ef0"],
            "tracking": ["tracking", "follow", "\u8ddf\u62cd", "\u79fb\u955c\u5934"],
            "handheld": ["handheld", "\u624b\u6301"],
            "zoom": ["zoom", "\u53d8\u7126"],
            "unknown": ["unknown", "uncertain", "\u4e0d\u786e\u5b9a", "\u672a\u77e5"],
        },
    },
    "environment_type": {
        "label": "environment",
        "values": {
            "indoor": ["indoor", "interior", "\u5ba4\u5185", "\u529e\u516c"],
            "outdoor": ["outdoor", "exterior", "\u6237\u5916", "\u81ea\u7136"],
            "street": ["street", "road", "\u8857\u9053", "\u9a6c\u8def"],
            "event": ["event", "crowd", "\u6d3b\u52a8\u73b0\u573a", "\u4eba\u7fa4"],
            "unknown": ["unknown", "uncertain", "\u4e0d\u786e\u5b9a", "\u672a\u77e5"],
        },
    },
    "edit_role": {
        "label": "edit role",
        "values": {
            "establishing": ["establishing", "opening", "\u5f00\u573a", "\u5efa\u7acb\u955c\u5934"],
            "transition": ["transition", "\u8fc7\u6e21", "\u8f6c\u573a"],
            "b_roll": ["b-roll", "b roll", "\u8865\u5145\u955c\u5934", "\u7a7a\u955c"],
            "information": ["information", "explanation", "\u4fe1\u606f\u8bf4\u660e"],
            "candidate_primary": ["primary", "hero", "\u5019\u9009\u4e3b\u955c\u5934"],
            "unknown": ["unknown", "uncertain", "\u4e0d\u786e\u5b9a", "\u672a\u77e5"],
        },
    },
}

_CLOSE_UP_TYPES = {"close_up", "extreme_close_up"}
_FOCUS_ISSUE_TOKENS = (
    "blur",
    "focus",
    "bokeh",
    "\u6a21\u7cca",
    "\u865a\u7126",
    "\u666f\u6df1",
)


def taxonomy_definition() -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "version": "v1",
        "upstream_reference": {
            "project": UPSTREAM_PROJECT,
            "commit": UPSTREAM_COMMIT,
            "reuse_mode": "independent_adaptation",
        },
        "fields": _FIELDS,
        "candidate_only": True,
        "replaces_evidence_schema": False,
    }
    payload["taxonomy_sha256"] = _sha256_json(payload)
    return payload


def normalize_scene_tags(values: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(values or {})
    normalized = {
        field_id: normalize_taxonomy_value(field_id, raw.get(field_id))
        for field_id in _FIELDS
        if raw.get(field_id) not in (None, "")
    }
    definition = taxonomy_definition()
    return {
        "schema": SCHEMA,
        "version": definition["version"],
        "taxonomy_sha256": definition["taxonomy_sha256"],
        "raw": raw,
        "normalized": normalized,
        "candidate_only": True,
        "display_and_search_only": True,
    }


def normalize_taxonomy_value(field_id: str, value: Any) -> str:
    field = _FIELDS.get(str(field_id))
    if field is None:
        raise ValueError(f"unknown taxonomy field: {field_id}")
    needle = str(value or "").strip().casefold()
    for canonical, aliases in field["values"].items():
        if needle == canonical.casefold() or needle in {str(alias).casefold() for alias in aliases}:
            return canonical
    return "unknown"


def explain_quality_for_shot_type(quality: dict[str, Any] | None, shot_type: Any) -> dict[str, Any]:
    raw_quality = dict(quality or {})
    normalized_shot_type = normalize_taxonomy_value("shot_type", shot_type)
    issues = [str(item) for item in raw_quality.get("issues") or []]
    focus_only = bool(issues) and all(
        any(token in issue.casefold() for token in _FOCUS_ISSUE_TOKENS)
        for issue in issues
    )
    contextual_review = normalized_shot_type in _CLOSE_UP_TYPES and focus_only
    return {
        "schema": QUALITY_SCHEMA,
        "shot_type": normalized_shot_type,
        "raw_quality": raw_quality,
        "raw_grade_preserved": True,
        "contextual_disposition": "review_context" if contextual_review else "unchanged",
        "explanation": (
            "Close-up focus or blur evidence needs shot-aware human review; raw grade is preserved."
            if contextual_review
            else "No shot-type-specific reinterpretation was applied."
        ),
        "candidate_only": True,
        "display_and_search_only": True,
    }


def _sha256_json(value: Any) -> str:
    return canonical_json_sha256(value)
