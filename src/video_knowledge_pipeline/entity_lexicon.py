from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .models import now_iso
from .page_metadata import load_page_metadata, page_metadata_context
from .storage import read_json, write_json
from .visual_evidence import visual_evidence_state


SCHEMA = "video_knowledge_pipeline.entity_lexicon.v1"
GLOSSARY_SCHEMA = "video_knowledge_pipeline.entity_glossary.v1"
HOTWORD_AUDIT_SCHEMA = "video_knowledge_pipeline.entity_hotword_audit.v1"
PHASES = {"pre_asr", "post_asr"}
SOURCE_WEIGHTS = {
    "base_lexicon": 10,
    "metadata": 8,
    "structured_visual": 7,
    "ocr": 7,
    "subtitle": 5,
    "visual_understanding": 5,
    "tagger": 4,
    "asr": 1,
}
_CJK = "\u4e00-\u9fff"
_GENERIC = {"这个老师", "那个老师", "我们老师", "相关产品", "这个产品", "一个产品"}
_GENERIC_ENTITY_TERMS = {
    "拓客",
    "获客",
    "成交",
    "客户",
    "保险",
    "产品",
    "课程",
    "老师",
    "活动",
    "沟通",
    "营销",
    "陈述",
    "讲解",
    "介绍",
    "展示",
}
_KNOWN_PLATFORMS = {
    "小红书",
    "抖音",
    "哔哩哔哩",
    "Bilibili",
    "TikTok",
    "YouTube",
    "微信",
}
_PIPELINE_TERMS = {"asr", "sensevoice", "sensevoice asr", "funasr", "qwen3-asr", "vkp"}
_TECHNICAL_DYNAMIC_TERMS = {
    "api",
    "assets",
    "audio",
    "base_url",
    "boolean",
    "content",
    "document",
    "document_visual",
    "false",
    "file",
    "frame",
    "image",
    "img",
    "jpeg",
    "jpg",
    "json",
    "markdown",
    "metadata",
    "model",
    "multimodal",
    "ocr",
    "ok",
    "online",
    "online_multimodal",
    "online_ocr",
    "provider",
    "lecture",
    "ms",
    "new",
    "org",
    "part",
    "route",
    "schema",
    "source",
    "status",
    "task",
    "temporal",
    "true",
    "understanding",
    "v1",
    "visual",
}
_TECHNICAL_VISUAL_KEYS = {
    "artifact",
    "artifact_path",
    "artifact_paths",
    "base_url",
    "capability",
    "destination",
    "error",
    "evidence_frame_path",
    "evidence_frame_paths",
    "execution_location",
    "image_path",
    "image_paths",
    "latency_ms",
    "mime_type",
    "model",
    "ok",
    "protocol",
    "provider",
    "route_id",
    "route_revision",
    "schema",
    "source",
    "source_type",
    "status",
    "task",
    "usage",
}
_TECHNICAL_DYNAMIC_PATTERNS = (
    re.compile(
        r"^(?:gemini|glm|mistral|openai|qwen|siliconflow|zai)(?:[-_.].*)?$",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:img|frame)[-_.]?\d+", re.IGNORECASE),
    re.compile(r"^t\d{1,4}$", re.IGNORECASE),
    re.compile(
        r"^(?:(?:company|employees?|logo|new|part)(?:\s+|$)){1,4}$", re.IGNORECASE
    ),
)
_FILE_OR_SCHEMA_PATTERN = re.compile(
    r"(?:[/\\]|https?://|\.(?:bmp|gif|jpe?g|json|md|png|webp)\b|^[a-z0-9_.-]+\.v\d+$)",
    re.IGNORECASE,
)

HIGH_RISK_ENTITY_TYPES = {"company", "organization", "product", "tool", "person"}


def build_entity_lexicon(
    bundle_dir: str | Path,
    *,
    base_lexicon_json: str | Path | None = None,
    phase: str = "post_asr",
    write: bool = True,
) -> dict[str, Any]:
    """Build Chinese entity aliases and ASR hotwords without changing transcripts."""

    root = Path(bundle_dir).expanduser().resolve()
    normalized_phase = str(phase or "post_asr").strip().lower()
    if normalized_phase not in PHASES:
        raise ValueError(f"unsupported entity lexicon phase: {phase}")
    manifest = _read_json_if_exists(root / "manifest.json")
    timeline = _read_json_if_exists(root / "timeline.json")
    if not isinstance(manifest, dict):
        manifest = {}
    if not isinstance(timeline, list):
        timeline = []
    evidence = _collect_evidence(manifest, timeline, phase=normalized_phase, root=root)
    terms = _base_terms(
        _load_base_lexicon(base_lexicon_json),
        base_lexicon_json=base_lexicon_json,
    )
    rejected_hotwords: list[dict[str, Any]] = []
    _merge_dynamic_terms(terms, evidence, rejected_hotwords=rejected_hotwords)
    glossary = _build_entity_glossary(
        root,
        manifest=manifest,
        timeline=timeline,
        terms=terms,
        base_lexicon_json=base_lexicon_json,
        phase=normalized_phase,
    )
    candidates = _alias_candidates(terms, evidence)
    candidates.extend(
        _same_segment_variant_candidates(
            glossary.get("terms") or [], evidence, start_number=len(candidates) + 1
        )
    )
    unresolved_high_risk_terms = _unresolved_high_risk_terms(candidates)
    hotwords = [str(row["canonical"]) for row in terms if row.get("hotword_allowed")]
    hotword_variants = _hotword_variants(terms)
    updated_at = now_iso()
    audit = {
        "schema": HOTWORD_AUDIT_SCHEMA,
        "phase": normalized_phase,
        "accepted_count": len(hotwords),
        "accepted_variant_count": len(hotword_variants),
        "rejected_count": len(rejected_hotwords),
        "accepted_hotwords": hotwords,
        "accepted_hotword_variants": hotword_variants,
        "variant_policy": "canonical_plus_safe_explicit_ascii_aliases",
        "rejected_candidates": rejected_hotwords,
        "unresolved_high_risk_term_count": len(unresolved_high_risk_terms),
        "eligible_sources": _eligible_sources(normalized_phase),
        "base_lexicon_terms_are_explicit": True,
        "technical_metadata_is_never_an_asr_hint": True,
        "post_asr_terms_do_not_trigger_asr_rerun": True,
        "updated_at": updated_at,
    }
    result = {
        "schema": SCHEMA,
        "status": "review_required" if unresolved_high_risk_terms else "passed",
        "quality_gate_passed": not unresolved_high_risk_terms,
        "phase": normalized_phase,
        "bundle_dir": str(root),
        "base_lexicon_json": str(Path(base_lexicon_json).expanduser().resolve())
        if base_lexicon_json
        else "",
        "pinyin_backend": _pinyin_backend(),
        "term_count": len(terms),
        "hotword_count": len(hotwords),
        "hotword_variant_count": len(hotword_variants),
        "correction_candidate_count": len(candidates),
        "rejected_hotword_count": len(rejected_hotwords),
        "terms": terms,
        "unresolved_high_risk_term_count": len(unresolved_high_risk_terms),
        "unresolved_high_risk_terms": unresolved_high_risk_terms,
        "hotwords": hotwords,
        "hotword_variants": hotword_variants,
        "hotword_text": " ".join(hotword_variants),
        "correction_candidates": candidates,
        "entity_glossary": glossary,
        "hotword_audit": audit,
        "operator_boundary": {
            "does_not_apply_corrections": True,
            "base_aliases_are_explicit_evidence": True,
            "dynamic_terms_require_non_asr_support": True,
            "pinyin_similarity_never_directly_replaces_text": True,
            "human_reference_not_used": True,
            "asr_input_allowed": normalized_phase == "pre_asr",
            "asr_rerun_allowed": False,
            "post_asr_terms_are_correction_evidence_only": normalized_phase
            == "post_asr",
            "unresolved_high_risk_terms_cannot_pass_quality_gate": True,
            "asr_single_source_cannot_promote_glossary_term": True,
            "discovered_variants_are_candidate_only": True,
        },
        "updated_at": updated_at,
    }
    if write:
        root.mkdir(parents=True, exist_ok=True)
        stage_slug = normalized_phase.replace("_", "-")
        write_json(root / f"entity-lexicon.{stage_slug}.json", result)
        (root / f"entity-hotwords.{stage_slug}.txt").write_text(
            result["hotword_text"] + ("\n" if hotwords else ""), encoding="utf-8"
        )
        write_json(root / f"entity-hotword-audit.{stage_slug}.json", audit)
        (root / f"entity-hotword-audit.{stage_slug}.md").write_text(
            _render_audit(audit), encoding="utf-8"
        )
        manifest[f"entity_lexicon_{normalized_phase}_json"] = (
            f"entity-lexicon.{stage_slug}.json"
        )
        manifest[f"entity_hotwords_{normalized_phase}"] = (
            f"entity-hotwords.{stage_slug}.txt"
        )
        manifest[f"entity_hotword_audit_{normalized_phase}_json"] = (
            f"entity-hotword-audit.{stage_slug}.json"
        )
        manifest[f"entity_hotword_audit_{normalized_phase}_markdown"] = (
            f"entity-hotword-audit.{stage_slug}.md"
        )
        glossary_stage_name = f"entity-glossary.{stage_slug}.json"
        write_json(root / glossary_stage_name, glossary)
        manifest[f"entity_glossary_{normalized_phase}_json"] = glossary_stage_name
        if normalized_phase == "post_asr":
            write_json(root / "entity-glossary.json", glossary)
            manifest["entity_glossary_json"] = "entity-glossary.json"
            manifest["entity_glossary_schema"] = GLOSSARY_SCHEMA
            manifest["entity_glossary_revision"] = int(
                glossary.get("revision") or 1
            )
        summary = {
            "phase": normalized_phase,
            "term_count": len(terms),
            "hotword_count": len(hotwords),
            "hotword_variant_count": len(hotword_variants),
            "rejected_hotword_count": len(rejected_hotwords),
            "correction_candidate_count": len(candidates),
            "pinyin_backend": result["pinyin_backend"],
            "status": result["status"],
            "quality_gate_passed": result["quality_gate_passed"],
            "unresolved_high_risk_term_count": len(unresolved_high_risk_terms),
        }
        manifest[f"entity_lexicon_{normalized_phase}_summary"] = summary
        if normalized_phase == "post_asr":
            write_json(root / "entity-lexicon.json", result)
            (root / "entity-hotwords.txt").write_text(
                result["hotword_text"] + ("\n" if hotwords else ""), encoding="utf-8"
            )
            (root / "entity-lexicon.md").write_text(_render(result), encoding="utf-8")
            manifest["entity_lexicon_json"] = "entity-lexicon.json"
            manifest["entity_lexicon_markdown"] = "entity-lexicon.md"
            manifest["entity_hotwords"] = "entity-hotwords.txt"
            manifest["entity_lexicon_summary"] = summary
        write_json(root / "manifest.json", manifest)
    return result


def _load_base_lexicon(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload = read_json(Path(path).expanduser().resolve())
    if not isinstance(payload, dict):
        raise ValueError("entity lexicon must be a JSON object")
    rows = [row for row in payload.get("terms") or [] if isinstance(row, dict)]
    if (
        str(payload.get("schema") or "")
        == "video_knowledge_pipeline.transcript_semantic_correction_review_notes.v1"
    ):
        rows.extend(_confirmed_review_terms(payload.get("reviews")))
    return rows


def _confirmed_review_terms(value: Any) -> list[dict[str, Any]]:
    """Convert explicit human-confirmed short corrections into hotword terms."""

    terms: list[dict[str, Any]] = []
    for row in value or []:
        if not isinstance(row, dict):
            continue
        if row.get("human_confirmed") is not True:
            continue
        if str(row.get("status") or "") != "corrected_transcript":
            continue
        original = str(row.get("original_text") or "").strip()
        corrected = str(row.get("corrected_transcript") or "").strip()
        if not _confirmed_hotword_shape(original) or not _confirmed_hotword_shape(
            corrected
        ):
            continue
        if original.casefold() == corrected.casefold():
            continue
        terms.append(
            {
                "canonical": corrected,
                "aliases": [original],
                "entity_type": "human_confirmed_term",
                "source_kind": "human_confirmed",
                "confidence": 1.0,
                "hotword_allowed": True,
                "review_required": False,
            }
        )
        if len(terms) >= 80:
            break
    return terms


def _confirmed_hotword_shape(value: str) -> bool:
    text = str(value or "").strip()
    return (
        bool(text)
        and len(text) <= 64
        and "\n" not in text
        and "\r" not in text
        and not re.search(r"[。！？!?；;]", text)
    )


def _base_terms(
    rows: list[dict[str, Any]],
    *,
    base_lexicon_json: str | Path | None = None,
) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    base_path = (
        Path(base_lexicon_json).expanduser().resolve()
        if base_lexicon_json
        else None
    )
    base_sha256 = _file_sha256(base_path) if base_path and base_path.is_file() else ""
    for row in rows:
        canonical = str(
            row.get("canonical") or row.get("canonical_term") or row.get("name") or ""
        ).strip()
        if not canonical:
            continue
        source_kind = str(row.get("source_kind") or "base_lexicon")
        evidence_path = str(base_path) if base_path else "explicit_base_glossary"
        evidence_row = {
            "source": source_kind,
            "source_type": source_kind,
            "text": canonical,
            "timeline_index": 0,
            "weight": 10,
            "path": evidence_path,
            "sha256": base_sha256 or _value_sha256(row),
            "evidence_level": "confirmed",
            "trusted_for_glossary": True,
        }
        terms.append(
            {
                "canonical": canonical,
                "aliases": _unique(
                    [
                        canonical,
                        *[
                            str(value)
                            for value in (
                                row.get("aliases") or row.get("raw_mentions") or []
                            )
                        ],
                    ]
                ),
                "entity_type": str(
                    row.get("entity_type") or row.get("type") or "other"
                ),
                "pinyin": str(row.get("pinyin") or _pinyin(canonical)),
                "source_types": [source_kind],
                "evidence": [evidence_row],
                "confidence": float(row.get("confidence") or 0.98),
                "hotword_allowed": bool(row.get("hotword_allowed", True)),
                "review_required": bool(row.get("review_required", False)),
            }
        )
    return terms


def _collect_evidence(
    manifest: dict[str, Any], timeline: list[Any], *, phase: str, root: Path | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata_parts = [
        str(manifest.get(key) or "")
        for key in (
            "title",
            "source_title",
            "description",
            "uploader",
            "author",
            "webpage_title",
        )
        if manifest.get(key)
    ]
    page_metadata = load_page_metadata(root, manifest) if root is not None else (manifest.get("page_metadata") if isinstance(manifest.get("page_metadata"), dict) else {})
    page_context = page_metadata_context(page_metadata, max_chars=2400)
    if page_context:
        metadata_parts.append(page_context)
    metadata = "\n".join(metadata_parts)
    if metadata:
        rows.append(
            _evidence(
                "metadata",
                metadata,
                0,
                path="manifest.json",
                evidence_level="unverified",
                trusted_for_glossary=False,
            )
        )
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        index = int(item.get("index") or position)
        pointer = f"timeline.json#/{position - 1}"
        corrected_ocr = str(item.get("human_corrected_visual_text") or "").strip()
        ocr_text = corrected_ocr or str(
            item.get("visual_text") or item.get("ocr_text") or ""
        ).strip()
        ocr_level = "confirmed" if corrected_ocr else _ocr_evidence_level(item)
        structured_value = item.get("structured_visual")
        confirmed_structured = _confirmed_structured_visual_text(structured_value)
        structured_text = confirmed_structured or _flatten_visual_evidence(
            structured_value
        )
        structured_level = "confirmed" if confirmed_structured else "unverified"
        visual_state = visual_evidence_state(item)
        visual_value = visual_state.get("understanding")
        visual_text = _flatten_visual_evidence(visual_value)
        visual_level = str(visual_state.get("evidence_level") or "not_seen")
        fields = (
            (
                "asr",
                item.get("transcript") or item.get("asr_text") or item.get("text"),
                f"{pointer}/transcript",
                "raw",
                False,
            ),
            (
                "subtitle",
                item.get("subtitle")
                or item.get("caption")
                or item.get("original_subtitle"),
                f"{pointer}/subtitle",
                "unverified",
                False,
            ),
            (
                "ocr",
                ocr_text,
                f"{pointer}/{'human_corrected_visual_text' if corrected_ocr else 'visual_text'}",
                ocr_level,
                ocr_level == "confirmed",
            ),
            (
                "structured_visual",
                structured_text,
                f"{pointer}/structured_visual",
                structured_level,
                structured_level == "confirmed",
            ),
            (
                "visual_understanding",
                visual_text,
                f"{pointer}/visual_understanding",
                visual_level,
                bool(visual_state.get("consumable")),
            ),
            (
                "tagger",
                " ".join(
                    str(value)
                    for value in item.get("tagger_tags") or item.get("tags") or []
                ),
                f"{pointer}/tags",
                "candidate_weak",
                False,
            ),
        )
        for source, value, evidence_path, evidence_level, trusted in fields:
            if source not in _eligible_sources(phase):
                continue
            text = str(value or "").strip()
            if text:
                rows.append(
                    _evidence(
                        source,
                        text,
                        index,
                        path=evidence_path,
                        evidence_level=evidence_level,
                        trusted_for_glossary=trusted,
                    )
                )
    return rows


def _evidence(
    source: str,
    text: str,
    index: int,
    *,
    path: str = "",
    evidence_level: str = "unverified",
    trusted_for_glossary: bool = False,
) -> dict[str, Any]:
    return {
        "source": source,
        "source_type": source,
        "text": text,
        "timeline_index": index,
        "weight": SOURCE_WEIGHTS.get(source, 1),
        "path": path,
        "sha256": _value_sha256(text),
        "evidence_level": evidence_level,
        "trusted_for_glossary": bool(trusted_for_glossary),
    }


def _merge_dynamic_terms(
    terms: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    *,
    rejected_hotwords: list[dict[str, Any]],
) -> None:
    by_name = {str(row["canonical"]).casefold(): row for row in terms}
    mentions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in evidence:
        if row["source"] == "asr":
            continue
        for canonical, entity_type in _extract_entities(str(row["text"])):
            if canonical.casefold() in _PIPELINE_TERMS or any(
                term in canonical.casefold()
                for term in ("sensevoice", "funasr", "qwen3-asr")
            ):
                continue
            mentions.setdefault((canonical, entity_type), []).append(row)
    for (canonical, entity_type), rows in mentions.items():
        if canonical in _GENERIC_ENTITY_TERMS:
            rejected_hotwords.append(
                _rejected_hotword(canonical, entity_type, rows, "generic_business_term")
            )
            continue
        rejection_reason = _dynamic_hotword_rejection_reason(canonical, entity_type)
        if rejection_reason:
            rejected_hotwords.append(
                _rejected_hotword(canonical, entity_type, rows, rejection_reason)
            )
            continue
        entity_type = _infer_entity_type(canonical, entity_type, mentions)
        existing = by_name.get(canonical.casefold())
        if existing:
            existing["source_types"] = _unique(
                [*existing["source_types"], *[str(row["source"]) for row in rows]]
            )
            existing["evidence"].extend({**row, "text": canonical} for row in rows[:20])
            continue
        source_types = _unique([str(row["source"]) for row in rows])
        support = sum(int(row["weight"]) for row in rows)
        term = {
            "canonical": canonical,
            "aliases": [canonical],
            "entity_type": entity_type,
            "pinyin": _pinyin(canonical),
            "source_types": source_types,
            "evidence": [{**row, "text": canonical} for row in rows[:20]],
            "confidence": round(min(0.95, 0.55 + support / 40), 3),
            "hotword_allowed": True,
            "review_required": len(source_types) < 2,
        }
        terms.append(term)
        by_name[canonical.casefold()] = term
    terms.sort(
        key=lambda row: (-float(row.get("confidence") or 0), str(row["canonical"]))
    )


def _infer_entity_type(
    canonical: str,
    entity_type: str,
    mentions: dict[tuple[str, str], list[dict[str, Any]]],
) -> str:
    """Prefer an explicit longer organization/product mention over a POS guess."""

    if entity_type != "person":
        return entity_type
    for other, other_type in mentions:
        if (
            len(other) > len(canonical)
            and other.startswith(canonical)
            and other_type in {"company", "organization", "product"}
        ):
            return (
                "company" if other_type in {"company", "organization"} else other_type
            )
    return entity_type


def _alias_candidates(
    terms: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    asr_rows = [row for row in evidence if row["source"] == "asr"]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for term in terms:
        canonical = str(term["canonical"])
        for alias_value in term.get("aliases") or []:
            alias = str(alias_value).strip()
            if not alias or alias.casefold() == canonical.casefold():
                continue
            for row in asr_rows:
                if alias not in str(row["text"]):
                    continue
                key = (int(row["timeline_index"]), alias, canonical)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "candidate_id": f"entity-{len(candidates) + 1:04d}",
                        "timeline_index": int(row["timeline_index"]),
                        "original_text": alias,
                        "corrected_text": canonical,
                        "entity_type": term.get("entity_type"),
                        "reason": "explicit_lexicon_alias",
                        "phonetic_similarity": _phonetic_similarity(
                            alias, canonical, term
                        ),
                        "confidence": float(term.get("confidence") or 0),
                        "evidence_source_types": term.get("source_types") or [],
                        "auto_apply_allowed": bool(
                            float(term.get("confidence") or 0) >= 0.95
                            and not term.get("review_required")
                            and any(
                                source != "asr"
                                for source in term.get("source_types") or []
                            )
                        ),
                    }
                )
    return candidates


def _build_entity_glossary(
    root: Path,
    *,
    manifest: dict[str, Any],
    timeline: list[Any],
    terms: list[dict[str, Any]],
    base_lexicon_json: str | Path | None,
    phase: str,
) -> dict[str, Any]:
    """Project trusted entity evidence into a versioned Bundle-local glossary."""

    base_path = (
        Path(base_lexicon_json).expanduser().resolve()
        if base_lexicon_json
        else None
    )
    source_hashes = {
        "manifest": _manifest_source_sha256(manifest),
        "timeline": _value_sha256(timeline),
        "base_glossary": _file_sha256(base_path)
        if base_path and base_path.is_file()
        else "",
    }
    previous_path = (
        root / "entity-glossary.json"
        if phase == "post_asr"
        else root / f"entity-glossary.{phase.replace('_', '-')}.json"
    )
    previous = _read_json_if_exists(previous_path)
    previous_revision = (
        int(previous.get("revision") or 0) if isinstance(previous, dict) else 0
    )
    if isinstance(previous, dict) and previous.get("source_hashes") == source_hashes:
        revision = max(1, previous_revision)
    else:
        revision = max(1, previous_revision + 1)

    glossary_terms: list[dict[str, Any]] = []
    for term in terms:
        trusted = [
            _glossary_evidence_row(row)
            for row in term.get("evidence") or []
            if isinstance(row, dict) and row.get("trusted_for_glossary") is True
        ]
        trusted = [row for row in trusted if row]
        if not trusted:
            continue
        source_types = _unique(
            [str(row.get("source_type") or "") for row in trusted]
        )
        if not set(source_types).issubset(
            {
                "base_lexicon",
                "human_confirmed",
                "ocr",
                "structured_visual",
                "visual_understanding",
            }
        ):
            continue
        glossary_terms.append(
            {
                "canonical": str(term.get("canonical") or ""),
                "aliases": _unique(
                    [str(value) for value in term.get("aliases") or []]
                ),
                "entity_type": str(term.get("entity_type") or "other"),
                "evidence": trusted,
                "revision": revision,
                "source_hashes": sorted(
                    {
                        str(row.get("sha256") or "")
                        for row in trusted
                        if str(row.get("sha256") or "")
                    }
                ),
            }
        )
    glossary_terms.sort(key=lambda row: str(row.get("canonical") or ""))
    pinyin_backend = _pinyin_backend()
    pinyin_available = pinyin_backend == "pypinyin"
    return {
        "schema": GLOSSARY_SCHEMA,
        "revision": revision,
        "phase": phase,
        "bundle_dir": str(root),
        "source_hashes": source_hashes,
        "term_count": len(glossary_terms),
        "terms": glossary_terms,
        "capability": {
            "status": "available" if pinyin_available else "degraded",
            "pypinyin_available": pinyin_available,
            "matching_mode": (
                "pinyin_plus_character_and_confirmed_evidence"
                if pinyin_available
                else "safe_character_and_confirmed_evidence_only"
            ),
            "automatic_apply": False,
            "gap": "" if pinyin_available else "phonetic_similarity_unavailable",
            "recovery": (
                "none"
                if pinyin_available
                else "install pypinyin in the VKP project environment, then rebuild the glossary"
            ),
        },
        "operator_boundary": {
            "asr_single_source_cannot_promote": True,
            "candidate_discovery_never_applies": True,
            "trusted_sources": [
                "explicit_base_glossary",
                "human_confirmed",
                "confirmed_ocr",
                "confirmed_structured_visual",
                "consumable_visual",
            ],
        },
        "updated_at": now_iso(),
    }


def _glossary_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    source_type = str(row.get("source_type") or row.get("source") or "")
    sha256 = str(row.get("sha256") or _value_sha256(row.get("text")))
    path = str(row.get("path") or "")
    if not source_type or not path or len(sha256) != 64:
        return {}
    return {
        "source_type": source_type,
        "path": path,
        "timeline_index": int(row.get("timeline_index") or 0),
        "sha256": sha256,
        "evidence_level": str(row.get("evidence_level") or "confirmed"),
    }


def _same_segment_variant_candidates(
    glossary_terms: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    *,
    start_number: int,
) -> list[dict[str, Any]]:
    asr_by_index = {
        int(row.get("timeline_index") or 0): str(row.get("text") or "")
        for row in evidence
        if row.get("source") == "asr" and int(row.get("timeline_index") or 0) > 0
    }
    pinyin_available = _pinyin_backend() == "pypinyin"
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for term in glossary_terms:
        canonical = str(term.get("canonical") or "").strip()
        if not (2 <= len(canonical) <= 16 and re.fullmatch(rf"[{_CJK}]+", canonical)):
            continue
        indexes = sorted(
            {
                int(row.get("timeline_index") or 0)
                for row in term.get("evidence") or []
                if int(row.get("timeline_index") or 0) > 0
            }
        )
        for index in indexes:
            asr_text = asr_by_index.get(index, "")
            variant, character_similarity = _best_character_variant(
                asr_text, canonical
            )
            if not variant:
                continue
            phonetic_similarity = (
                _phonetic_pair_similarity(variant, canonical)
                if pinyin_available
                else None
            )
            if pinyin_available:
                eligible = bool(
                    phonetic_similarity is not None
                    and phonetic_similarity >= 0.72
                    and character_similarity >= 0.25
                )
                matching_mode = "pinyin_and_confirmed_evidence"
            else:
                eligible = _safe_degraded_character_match(
                    variant, canonical, character_similarity
                )
                matching_mode = "degraded_character_evidence"
            key = (index, variant, canonical)
            if not eligible or key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "candidate_id": f"entity-{start_number + len(candidates):04d}",
                    "timeline_index": index,
                    "original_text": variant,
                    "corrected_text": canonical,
                    "entity_type": str(term.get("entity_type") or "other"),
                    "reason": "same_segment_confirmed_canonical_variant",
                    "character_similarity": round(character_similarity, 6),
                    "phonetic_similarity": phonetic_similarity,
                    "matching_mode": matching_mode,
                    "evidence_source_types": _unique(
                        [
                            str(row.get("source_type") or "")
                            for row in term.get("evidence") or []
                        ]
                    ),
                    "candidate_only": True,
                    "auto_apply_allowed": False,
                }
            )
    return candidates


def _best_character_variant(asr_text: str, canonical: str) -> tuple[str, float]:
    best = ""
    best_score = 0.0
    for run in re.findall(rf"[{_CJK}]+", str(asr_text or "")):
        if len(run) < len(canonical):
            continue
        for offset in range(0, len(run) - len(canonical) + 1):
            value = run[offset : offset + len(canonical)]
            if value == canonical:
                continue
            score = difflib.SequenceMatcher(
                a=value, b=canonical, autojunk=False
            ).ratio()
            if score > best_score:
                best = value
                best_score = score
    return best, best_score


def _safe_degraded_character_match(
    variant: str, canonical: str, similarity: float
) -> bool:
    if len(variant) != len(canonical) or similarity < 0.5:
        return False
    prefix = _common_edge_length(variant, canonical)
    suffix = _common_edge_length(variant[::-1], canonical[::-1])
    return similarity >= 0.75 or max(prefix, suffix) >= 2


def _common_edge_length(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _phonetic_pair_similarity(left_value: str, right_value: str) -> float | None:
    left = _pinyin(left_value)
    right = _pinyin(right_value)
    if not left or not right:
        return None
    return round(
        difflib.SequenceMatcher(
            a=_phonetic_key(left), b=_phonetic_key(right), autojunk=False
        ).ratio(),
        6,
    )


def _unresolved_high_risk_terms(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unresolved: list[dict[str, Any]] = []
    for row in candidates:
        if bool(row.get("auto_apply_allowed")):
            continue
        if str(row.get("entity_type") or "") not in HIGH_RISK_ENTITY_TYPES:
            continue
        unresolved.append(
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "timeline_index": int(row.get("timeline_index") or 0),
                "original_text": str(row.get("original_text") or ""),
                "corrected_text": str(row.get("corrected_text") or ""),
                "entity_type": str(row.get("entity_type") or ""),
                "confidence": float(row.get("confidence") or 0),
                "reason": "high_risk_entity_alias_requires_review",
            }
        )
    return unresolved


def _extract_entities(text: str) -> list[tuple[str, str]]:
    found = _jieba_entities(text)
    fallback_patterns = (
        (
            rf"[{_CJK}A-Za-z0-9-]{{2,8}}(?:保险|人寿|科技|集团|公司|工作室|平台|团队)",
            "company",
        ),
        (
            rf"[{_CJK}A-Za-z0-9-]{{2,10}}(?:重疾险|医疗险|寿险|意外险|年金险|课程|系统|工具|模型|产品|计划)",
            "product",
        ),
        (rf"[{_CJK}]{{2,8}}(?:保费|保额|费率|赔付率|佣金)", "metric"),
        (rf"([{_CJK}]{{2,4}})(?:老师|博士|教授)", "person"),
    )
    for pattern, entity_type in fallback_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = (
                match.group(1) if entity_type == "person" else match.group(0)
            ).strip(" ，。！？；：、,.!?;:()[]")
            if (
                value
                and value not in _GENERIC
                and len(value) >= 2
                and not _looks_mojibake(value)
            ):
                found.append((value, entity_type))
    for match in re.finditer(
        r"[A-Za-z][A-Za-z0-9+_.-]{2,}(?:\s+[A-Za-z][A-Za-z0-9+_.-]{2,}){0,2}", text
    ):
        value = match.group(0).strip()
        if value and not _looks_mojibake(value):
            found.append((value, "tool"))
    return list(dict.fromkeys(found))


def _hotword_variants(terms: list[dict[str, Any]]) -> list[str]:
    """Keep canonical terms and only explicit, safe ASCII pronunciation variants."""

    variants: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term.get("hotword_allowed"):
            continue
        canonical = str(term.get("canonical") or "").strip()
        for value in (canonical, *[str(alias) for alias in term.get("aliases") or []]):
            text = str(value or "").strip()
            folded = text.casefold()
            if not text or folded in seen:
                continue
            if text != canonical and not _safe_explicit_hotword_alias(text):
                continue
            seen.add(folded)
            variants.append(text)
    return variants


def _safe_explicit_hotword_alias(value: str) -> bool:
    text = str(value or "").strip()
    if not 2 <= len(text) <= 40 or re.search(rf"[{_CJK}]", text):
        return False
    if not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9+_.-]*(?:\s+[A-Za-z][A-Za-z0-9+_.-]*){0,2}", text
    ):
        return False
    return not _dynamic_hotword_rejection_reason(text, "tool")


def _eligible_sources(phase: str) -> list[str]:
    if phase == "pre_asr":
        return ["metadata", "ocr", "structured_visual"]
    return [
        "metadata",
        "asr",
        "subtitle",
        "ocr",
        "structured_visual",
        "visual_understanding",
        "tagger",
    ]


def _dynamic_hotword_rejection_reason(canonical: str, entity_type: str) -> str:
    text = str(canonical or "").strip()
    folded = text.casefold()
    if not text:
        return "empty"
    if _looks_mojibake(text):
        return "invalid_text_encoding"
    if "\n" in text or "\r" in text:
        return "multiline_or_concatenated_metadata"
    if len(text) > 40:
        return "too_long_for_asr_hint"
    if folded in _TECHNICAL_DYNAMIC_TERMS or folded in _PIPELINE_TERMS:
        return "pipeline_or_transport_metadata"
    if _FILE_OR_SCHEMA_PATTERN.search(text):
        return "file_path_url_or_schema_metadata"
    if any(pattern.search(text) for pattern in _TECHNICAL_DYNAMIC_PATTERNS):
        return "provider_model_or_placeholder_metadata"
    if entity_type == "product" and re.match(
        r"^(?:介绍|陈述|讲解|展示).{0,4}产品$", text
    ):
        return "generic_layout_phrase"
    if folded in {"bullet", "heading", "layout"} or text in {
        "层次分明",
        "介绍产品",
        "陈述产品",
    }:
        return "generic_layout_phrase"
    return ""


def _rejected_hotword(
    canonical: str,
    entity_type: str,
    rows: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    return {
        "candidate": canonical,
        "entity_type": entity_type,
        "reason": reason,
        "source_types": _unique([str(row.get("source") or "") for row in rows]),
        "timeline_indexes": sorted(
            {
                int(row.get("timeline_index") or 0)
                for row in rows
                if int(row.get("timeline_index") or 0) > 0
            }
        )[:20],
    }


def _jieba_entities(text: str) -> list[tuple[str, str]]:
    try:
        import jieba.posseg as pseg  # type: ignore
    except Exception:
        return []
    tokens = [
        (str(item.word).strip(), str(item.flag))
        for item in pseg.cut(text)
        if str(item.word).strip()
    ]
    found: list[tuple[str, str]] = []
    suffix_types = {
        "保险": "company",
        "科技": "company",
        "集团": "company",
        "公司": "company",
        "工作室": "company",
        "平台": "company",
        "团队": "company",
        "产品": "product",
        "课程": "course",
        "系统": "product",
        "工具": "tool",
        "模型": "product",
    }
    for index, (word, flag) in enumerate(tokens):
        if _looks_mojibake(word) or word in _GENERIC:
            continue
        if word in _KNOWN_PLATFORMS:
            found.append((word, "tool"))
        elif flag == "eng" and len(word) >= 2:
            found.append((word, "tool"))
        elif flag in {"nr", "nt", "nz"} and 2 <= len(word) <= 12:
            found.append((word, "person" if flag == "nr" else "organization"))
        if index + 1 < len(tokens) and word not in _KNOWN_PLATFORMS:
            next_word = tokens[index + 1][0]
            matched_suffix = next(
                (suffix for suffix in suffix_types if next_word.startswith(suffix)), ""
            )
            if (
                matched_suffix
                and 2 <= len(word) <= 10
                and re.search(rf"[{_CJK}A-Za-z0-9]", word)
            ):
                found.append((word + matched_suffix, suffix_types[matched_suffix]))
    return found


def _looks_mojibake(value: str) -> bool:
    text = str(value or "")
    if "�" in text or "???" in text:
        return True
    suspicious = sum(
        text.count(marker) for marker in ("锟", "烫", "屯", "鈥", "銆", "娴", "缁")
    )
    return suspicious >= 2


def _pinyin(value: str) -> str:
    try:
        from pypinyin import lazy_pinyin  # type: ignore

        return " ".join(lazy_pinyin(value, errors="ignore"))
    except Exception:
        return ""


def _pinyin_backend() -> str:
    try:
        import pypinyin  # type: ignore  # noqa: F401

        return "pypinyin"
    except Exception:
        return "unavailable_explicit_alias_only"


def _phonetic_similarity(
    alias: str, canonical: str, term: dict[str, Any]
) -> float | None:
    left = _pinyin(alias)
    right = str(term.get("pinyin") or _pinyin(canonical)).strip()
    if not left or not right:
        return None
    return round(
        difflib.SequenceMatcher(
            a=_phonetic_key(left), b=_phonetic_key(right), autojunk=False
        ).ratio(),
        6,
    )


def _phonetic_key(value: str) -> str:
    key = re.sub(r"[^a-z]", "", value.casefold())
    for old, new in (
        ("zh", "z"),
        ("ch", "c"),
        ("sh", "s"),
        ("eng", "en"),
        ("ing", "in"),
        ("ang", "an"),
    ):
        key = key.replace(old, new)
    return key


def _flatten_visual_evidence(value: Any, *, parent_key: str = "") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            _flatten_visual_evidence(item, parent_key=parent_key) for item in value
        )
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            normalized_key = str(key or "").strip().casefold()
            if (
                normalized_key in _TECHNICAL_VISUAL_KEYS
                or normalized_key.endswith("_path")
                or normalized_key.endswith("_paths")
            ):
                continue
            parts.append(_flatten_visual_evidence(item, parent_key=normalized_key))
        return "\n".join(part for part in parts if part)
    return "" if value is None else str(value)


def _ocr_evidence_level(item: dict[str, Any]) -> str:
    for key in ("ocr_evidence_level", "visual_text_evidence_level"):
        level = str(item.get(key) or "").strip().lower()
        if level in {"confirmed", "inferred", "unavailable"}:
            return level
    validation = item.get("ocr_validation")
    if isinstance(validation, dict) and (
        validation.get("ok") is True
        or str(validation.get("status") or "").strip().lower()
        in {"ok", "passed", "confirmed"}
    ):
        return "confirmed"
    return "unverified"


def _confirmed_structured_visual_text(value: Any) -> str:
    rows = value if isinstance(value, list) else [value]
    accepted: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        level = str(row.get("evidence_level") or "").strip().lower()
        validation = row.get("validation")
        validated = isinstance(validation, dict) and (
            validation.get("ok") is True
            or str(validation.get("status") or "").strip().lower()
            in {"ok", "passed", "confirmed"}
        )
        if level != "confirmed" and not validated:
            continue
        text = _flatten_visual_evidence(row)
        if text:
            accepted.append(text)
    return "\n".join(accepted)


def _value_sha256(value: Any) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_source_sha256(manifest: dict[str, Any]) -> str:
    source_fields = {
        key: value
        for key, value in manifest.items()
        if not str(key).startswith("entity_")
    }
    return _value_sha256(source_fields)


def _read_json_if_exists(path: Path) -> Any:
    return read_json(path) if path.exists() else {}


def _unique(values: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            value for value in (str(item).strip() for item in values) if value
        )
    )


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# 实体词库与动态热词",
        "",
        f"- 词条：`{result.get('term_count')}`",
        f"- 热词：`{result.get('hotword_count')}`",
        f"- 纠错候选：`{result.get('correction_candidate_count')}`",
        f"- 拼音后端：`{result.get('pinyin_backend')}`",
        "",
        "| 标准名 | 类型 | 别名 | 证据源 | 置信度 | 复核 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in result.get("terms") or []:
        lines.append(
            f"| {row.get('canonical')} | {row.get('entity_type')} | {', '.join(row.get('aliases') or [])} | {', '.join(row.get('source_types') or [])} | {row.get('confidence')} | {row.get('review_required')} |"
        )
    lines.extend(["", "> 拼音相近只生成候选，不允许单独触发自动替换。", ""])
    return "\n".join(lines)


def _render_audit(audit: dict[str, Any]) -> str:
    lines = [
        "# ASR 热词审计",
        "",
        f"- 阶段: {audit.get('phase')}",
        f"- 接受: {audit.get('accepted_count')}",
        f"- 拒绝: {audit.get('rejected_count')}",
        f"- 合格来源: {', '.join(audit.get('eligible_sources') or [])}",
        "- ASR 后新词不会自动触发重跑: true",
        "",
        "## 已接受热词",
        "",
        ", ".join(str(value) for value in audit.get("accepted_hotwords") or [])
        or "（无）",
        "",
        "## 已拒绝候选",
        "",
        "| 候选 | 原因 | 来源 |",
        "| --- | --- | --- |",
    ]
    for row in audit.get("rejected_candidates") or []:
        lines.append(
            f"| {row.get('candidate')} | {row.get('reason')} | {', '.join(row.get('source_types') or [])} |"
        )
    return "\n".join(lines).rstrip() + "\n"
