from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .term_text import apply_high_confidence_term_replacements, high_confidence_term_replacements
from .visual_integration import integrated_visual

SCHEMA = "video_knowledge_pipeline.term_resolution.v1"

SOURCE_WEIGHTS = {
    "structured_visual": 6,
    "ocr": 5,
    "metadata": 4,
    "visual_understanding": 4,
    "temporal_visual_understanding": 4,
    "tagger": 3,
    "subtitle": 2,
    "asr": 1,
}

STOP_TERMS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "not", "you", "are", "was", "but", "api", "json", "http", "https",
}


def resolve_terms(
    bundle_dir: str | Path,
    *,
    metadata_json: str | Path | None = None,
    glossary_json: str | Path | None = None,
    min_mentions: int = 1,
    write: bool = True,
) -> dict[str, Any]:
    """Resolve likely canonical terminology across ASR, subtitles, OCR, visual evidence, tags, and metadata."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must be a JSON array")

    metadata = _load_optional_json(metadata_json)
    glossary = _load_glossary(glossary_json)
    evidence = _collect_evidence(timeline, metadata=metadata, glossary=glossary)
    clusters = _cluster_mentions(evidence, glossary=glossary)
    resolutions = [_resolve_cluster(cluster, glossary=glossary) for cluster in clusters]
    resolutions = [row for row in resolutions if row["mention_count"] >= max(1, int(min_mentions or 1))]
    resolutions.sort(key=lambda row: (-float(row.get("risk_score") or 0), row.get("canonical_term", "").lower()))

    summary = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "metadata_json": str(Path(metadata_json).expanduser().resolve()) if metadata_json else "",
        "glossary_json": str(Path(glossary_json).expanduser().resolve()) if glossary_json else "",
        "write": bool(write),
        "timeline_items": len([item for item in timeline if isinstance(item, dict)]),
        "evidence_count": len(evidence),
        "resolved_terms": len(resolutions),
        "needs_human_review": sum(1 for row in resolutions if row.get("needs_human_review")),
        "updated_at": now_iso(),
    }
    result = {
        **summary,
        "terms": resolutions,
        "source_weights": SOURCE_WEIGHTS,
        "artifacts": {
            "json": str(root / "term-resolution.json"),
            "glossary_markdown": str(root / "term-glossary.md"),
            "review_markdown": str(root / "term-review.md"),
            "corrected_transcript_json": str(root / "corrected-transcript.json"),
            "corrected_transcript_markdown": str(root / "corrected-transcript.md"),
            "corrected_transcript_srt": str(root / "corrected-transcript.srt"),
        },
    }

    if write:
        with bundle_write_lock(root, operation="resolve_terms", timeout_seconds=1.0):
            _write_timeline_terms(timeline, resolutions)
            corrected_payload = _corrected_transcript_payload(root, timeline)
            result["corrected_transcript_summary"] = corrected_payload["summary"]
            write_json(timeline_path, timeline)
            write_json(root / "corrected-transcript.json", corrected_payload)
            (root / "corrected-transcript.md").write_text(_render_corrected_transcript_markdown(corrected_payload), encoding="utf-8")
            (root / "corrected-transcript.srt").write_text(_render_corrected_transcript_srt(corrected_payload), encoding="utf-8")
            write_json(root / "term-resolution.json", result)
            (root / "term-glossary.md").write_text(_render_glossary(result), encoding="utf-8")
            (root / "term-review.md").write_text(_render_review(result), encoding="utf-8")
            write_json(
                root / "mcp-resolve-terms.args.json",
                {
                    "bundle_dir": str(root),
                    "metadata_json": str(Path(metadata_json).expanduser().resolve()) if metadata_json else "",
                    "glossary_json": str(Path(glossary_json).expanduser().resolve()) if glossary_json else "",
                    "min_mentions": min_mentions,
                    "write": True,
                },
            )
            manifest["term_resolution"] = "term-resolution.json"
            manifest["term_glossary"] = "term-glossary.md"
            manifest["term_review"] = "term-review.md"
            manifest["corrected_transcript_json"] = "corrected-transcript.json"
            manifest["corrected_transcript_markdown"] = "corrected-transcript.md"
            manifest["corrected_transcript_srt"] = "corrected-transcript.srt"
            manifest["mcp_resolve_terms_args"] = "mcp-resolve-terms.args.json"
            manifest["term_resolution_summary"] = summary
            write_json(manifest_path, manifest)
    return result


def _collect_evidence(timeline: list[Any], *, metadata: Any, glossary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        index = _item_index(item, position)
        time_range = {"start": item.get("start"), "end": item.get("end")}
        _add_text(rows, item.get("transcript") or item.get("text") or item.get("asr_text"), source="asr", index=index, time_range=time_range, glossary=glossary)
        _add_text(rows, item.get("subtitle") or item.get("caption") or item.get("original_subtitle"), source="subtitle", index=index, time_range=time_range, glossary=glossary)
        _add_text(rows, item.get("visual_text") or item.get("ocr_text"), source="ocr", index=index, time_range=time_range, glossary=glossary)
        _add_text(rows, _structured_text(item.get("structured_visual")), source="structured_visual", index=index, time_range=time_range, glossary=glossary)
        _add_text(rows, _mapping_text(item.get("visual_understanding")), source="visual_understanding", index=index, time_range=time_range, glossary=glossary)
        _add_text(rows, _mapping_text(item.get("temporal_visual_understanding")), source="temporal_visual_understanding", index=index, time_range=time_range, glossary=glossary)
        _add_text(rows, item.get("tagger_visual_summary"), source="tagger", index=index, time_range=time_range, glossary=glossary)
        tag_text = " ".join(str(tag) for tag in (item.get("tagger_tags") or item.get("tags") or []) if str(tag))
        _add_text(rows, tag_text, source="tagger", index=index, time_range=time_range, glossary=glossary)
    _add_text(rows, _metadata_text(metadata), source="metadata", index=0, time_range={}, glossary=glossary)
    return rows


def _add_text(rows: list[dict[str, Any]], value: Any, *, source: str, index: int, time_range: dict[str, Any], glossary: dict[str, Any]) -> None:
    text = _text(value)
    if not text:
        return
    for mention in _extract_mentions(text, glossary=glossary):
        rows.append(
            {
                "mention": mention,
                "normalised": _normalise_term(mention),
                "source": source,
                "weight": SOURCE_WEIGHTS.get(source, 1),
                "timeline_index": index,
                "start": time_range.get("start"),
                "end": time_range.get("end"),
                "context": _context(text, mention),
            }
        )


def _extract_mentions(text: str, *, glossary: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for term in glossary.get("all_terms", []):
        if term and re.search(re.escape(term), text, flags=re.IGNORECASE):
            found.append(term)
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9_-]{1,}(?:\s+[A-Za-z][A-Za-z0-9_-]{1,}){0,2}", text):
        value = match.group(0).strip(" -_.,:;()[]{}<>\"'")
        if _valid_mention(value):
            found.append(value)
    return _unique(found)


def _valid_mention(value: str) -> bool:
    clean = value.strip()
    if len(clean) < 2:
        return False
    lower = clean.lower()
    if lower in STOP_TERMS:
        return False
    if clean.isdigit():
        return False
    if not re.search(r"[A-Za-z]", clean):
        return False
    return True


def _cluster_mentions(evidence: list[dict[str, Any]], *, glossary: dict[str, Any]) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for row in evidence:
        canonical = _glossary_canonical(row["mention"], glossary)
        row["glossary_canonical"] = canonical
        placed = False
        for cluster in clusters:
            anchor = cluster[0]
            if canonical and canonical == anchor.get("glossary_canonical"):
                cluster.append(row)
                placed = True
                break
            if _similar_terms(row["normalised"], anchor["normalised"]):
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])
    return clusters


def _resolve_cluster(cluster: list[dict[str, Any]], *, glossary: dict[str, Any]) -> dict[str, Any]:
    raw_mentions = _unique([row["mention"] for row in cluster])
    canonical_from_glossary = next((row.get("glossary_canonical") for row in cluster if row.get("glossary_canonical")), "")
    canonical = canonical_from_glossary or _choose_canonical(cluster)
    support = sum(float(row.get("weight") or 0) for row in cluster if _normalise_term(row["mention"]) == _normalise_term(canonical) or row.get("glossary_canonical") == canonical)
    total = sum(float(row.get("weight") or 0) for row in cluster) or 1.0
    source_counts: dict[str, int] = {}
    for row in cluster:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
    conflict_count = len({_normalise_term(value) for value in raw_mentions})
    confidence = min(0.95, max(0.15, support / total if support else _best_source_weight(cluster) / total))
    needs_review = conflict_count > 1 or confidence < 0.72 or not any(row["source"] in {"ocr", "structured_visual", "metadata", "visual_understanding"} for row in cluster)
    evidence = sorted(cluster, key=lambda row: (-int(row.get("weight") or 0), int(row.get("timeline_index") or 0)))[:20]
    return {
        "canonical_term": canonical,
        "raw_mentions": raw_mentions,
        "mention_count": len(cluster),
        "source_counts": source_counts,
        "confidence": round(confidence, 3),
        "risk_score": round((1.0 - confidence) * 10 + max(0, conflict_count - 1) * 2 + (3 if needs_review else 0), 3),
        "decision": "suggest_correction" if conflict_count > 1 else "keep_or_confirm",
        "needs_human_review": needs_review,
        "evidence": evidence,
    }


def _choose_canonical(cluster: list[dict[str, Any]]) -> str:
    scores: dict[str, float] = {}
    display: dict[str, str] = {}
    for row in cluster:
        key = _normalise_term(row["mention"])
        scores[key] = scores.get(key, 0.0) + float(row.get("weight") or 0) + _display_bonus(row["mention"])
        current = display.get(key, "")
        if not current or _display_bonus(row["mention"]) > _display_bonus(current) or len(row["mention"]) > len(current):
            display[key] = row["mention"]
    best = sorted(scores.items(), key=lambda item: (-item[1], -len(display.get(item[0], "")), display.get(item[0], "").lower()))[0][0]
    return display[best]


def _display_bonus(value: str) -> float:
    bonus = 0.0
    if any(char.isupper() for char in value):
        bonus += 0.4
    if "-" in value:
        bonus += 0.2
    if " " not in value:
        bonus += 0.1
    return bonus


def _similar_terms(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) < 4 or len(right) < 4:
        return False
    if left in right or right in left:
        return True
    return difflib.SequenceMatcher(None, left, right).ratio() >= 0.84


def _write_timeline_terms(timeline: list[Any], resolutions: list[dict[str, Any]]) -> None:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for term in resolutions:
        for evidence in term.get("evidence") or []:
            index = int(evidence.get("timeline_index") or 0)
            if index <= 0:
                continue
            by_index.setdefault(index, []).append(
                {
                    "canonical_term": term.get("canonical_term"),
                    "raw_mentions": term.get("raw_mentions"),
                    "confidence": term.get("confidence"),
                    "decision": term.get("decision"),
                    "needs_human_review": term.get("needs_human_review"),
                    "evidence_source": evidence.get("source"),
                }
            )
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        index = _item_index(item, position)
        rows = by_index.get(index, [])
        if rows:
            item["term_candidates"] = rows
            issues = item.get("quality_issues") if isinstance(item.get("quality_issues"), list) else []
            if any(row.get("needs_human_review") for row in rows):
                item["quality_issues"] = _unique([*[str(issue) for issue in issues], "term_resolution_needs_review"])
            item["term_resolution_updated_at"] = now_iso()
            _write_corrected_transcript(item)
            item["integrated_visual"] = integrated_visual(item)


def _write_corrected_transcript(item: dict[str, Any]) -> None:
    raw = _text(item.get("transcript") or item.get("original_transcript") or item.get("asr_text"))
    if not raw:
        return
    corrected = apply_high_confidence_term_replacements(raw, item)
    corrections = high_confidence_term_replacements(item)
    if corrected and corrected != raw:
        item["corrected_transcript"] = corrected
        item["corrected_transcript_source"] = "term_resolution"
        item["corrected_transcript_updated_at"] = now_iso()
        item["transcript_corrections"] = corrections


def _corrected_transcript_payload(root: Path, timeline: list[Any]) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    corrected_count = 0
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        raw = _text(item.get("transcript") or item.get("original_transcript") or item.get("asr_text"))
        corrected = _text(item.get("corrected_transcript")) or raw
        if raw and corrected != raw:
            corrected_count += 1
        if raw or corrected:
            segments.append(
                {
                    "index": _item_index(item, position),
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "raw_transcript": raw,
                    "corrected_transcript": corrected,
                    "corrections": item.get("transcript_corrections") if isinstance(item.get("transcript_corrections"), list) else [],
                }
            )
    return {
        "schema": "video_knowledge_pipeline.corrected_transcript.v1",
        "bundle_dir": str(root),
        "updated_at": now_iso(),
        "summary": {"segments": len(segments), "corrected_segments": corrected_count},
        "segments": segments,
    }


def _render_corrected_transcript_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Corrected Transcript",
        "",
        f"- Segments: `{summary.get('segments', 0)}`",
        f"- Corrected segments: `{summary.get('corrected_segments', 0)}`",
        "",
    ]
    for segment in payload.get("segments") or []:
        lines.extend(
            [
                f"## {segment.get('index')}. {_format_time(segment.get('start'))} - {_format_time(segment.get('end'))}",
                "",
                _text(segment.get("corrected_transcript")) or "（空）",
                "",
            ]
        )
        corrections = segment.get("corrections") if isinstance(segment.get("corrections"), list) else []
        if corrections:
            lines.append("- Corrections: " + "; ".join(f"{row.get('raw')} -> {row.get('canonical')}" for row in corrections))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_corrected_transcript_srt(payload: dict[str, Any]) -> str:
    blocks: list[str] = []
    for position, segment in enumerate(payload.get("segments") or [], start=1):
        text = _text(segment.get("corrected_transcript"))
        if not text:
            continue
        blocks.append(
            "\n".join(
                [
                    str(position),
                    f"{_format_srt_time(segment.get('start'))} --> {_format_srt_time(segment.get('end'))}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks).rstrip() + ("\n" if blocks else "")


def _seconds(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    if ":" in text:
        try:
            total = 0.0
            for part in text.split(":"):
                total = total * 60 + float(part.replace(",", "."))
            return total
        except ValueError:
            return None
    return None


def _format_time(value: Any) -> str:
    seconds = _seconds(value) or 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{sec:02d}.{ms:03d}"


def _format_srt_time(value: Any) -> str:
    return _format_time(value).replace(".", ",")

def _render_glossary(result: dict[str, Any]) -> str:
    lines = ["# Term Glossary", "", f"- Resolved terms: `{result.get('resolved_terms', 0)}`", "", "| Canonical | Raw mentions | Confidence | Sources | Review |", "| --- | --- | ---: | --- | --- |"]
    for term in result.get("terms") or []:
        sources = ", ".join(f"{key}:{value}" for key, value in (term.get("source_counts") or {}).items())
        lines.append(f"| {_md(term.get('canonical_term'))} | {_md(', '.join(term.get('raw_mentions') or []))} | {term.get('confidence')} | {_md(sources)} | {term.get('needs_human_review')} |")
    return "\n".join(lines).rstrip() + "\n"


def _render_review(result: dict[str, Any]) -> str:
    lines = ["# Term Review", "", "Review rows where different evidence sources disagree or confidence is low.", ""]
    for term in result.get("terms") or []:
        if not term.get("needs_human_review"):
            continue
        lines.extend([f"## {term.get('canonical_term')}", "", f"- Raw mentions: `{', '.join(term.get('raw_mentions') or [])}`", f"- Confidence: `{term.get('confidence')}`", f"- Decision: `{term.get('decision')}`", "", "| Source | Index | Mention | Context |", "| --- | ---: | --- | --- |"])
        for row in term.get("evidence") or []:
            lines.append(f"| `{row.get('source')}` | {row.get('timeline_index')} | {_md(row.get('mention'))} | {_md(row.get('context'))} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_optional_json(path: str | Path | None) -> Any:
    if not path:
        return {}
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"metadata_json not found: {target}")
    return read_json(target)


def _load_glossary(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"aliases": {}, "all_terms": []}
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"glossary_json not found: {target}")
    data = read_json(target)
    aliases: dict[str, str] = {}
    display_terms: set[str] = set()

    def add_alias(alias: Any, canonical: str) -> None:
        alias_text = _text(alias)
        if not alias_text or not canonical:
            return
        aliases[_normalise_term(alias_text)] = canonical
        display_terms.add(alias_text)

    if isinstance(data, dict):
        terms = data.get("terms") if isinstance(data.get("terms"), list) else []
        for item in terms:
            if isinstance(item, dict):
                canonical = _text(item.get("canonical") or item.get("term") or item.get("name"))
                if canonical:
                    add_alias(canonical, canonical)
                    for alias in item.get("aliases") or []:
                        add_alias(alias, canonical)
        for key, value in data.items():
            if key == "terms":
                continue
            if isinstance(value, str):
                canonical = value
                add_alias(key, canonical)
                add_alias(value, canonical)
            elif isinstance(value, list):
                canonical = str(key)
                add_alias(canonical, canonical)
                for alias in value:
                    add_alias(alias, canonical)
    all_terms = sorted(display_terms, key=len, reverse=True)
    return {"aliases": aliases, "all_terms": all_terms}

def _glossary_canonical(value: str, glossary: dict[str, Any]) -> str:
    return str((glossary.get("aliases") or {}).get(_normalise_term(value)) or "")


def _normalise_term(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _structured_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for entry in value:
        if isinstance(entry, dict):
            parts.append(_text(entry.get("markdown") or entry.get("text") or entry.get("content")))
    return "\n".join(part for part in parts if part)


def _mapping_text(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    parts: list[str] = []
    for item in value.values():
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, list):
            parts.extend(str(part) for part in item if isinstance(part, (str, int, float)))
        elif isinstance(item, dict):
            parts.append(_mapping_text(item))
    return "\n".join(part for part in parts if part)


def _metadata_text(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key in ("title", "description", "desc", "summary", "tags", "keywords", "uploader", "author"):
            raw = value.get(key)
            if isinstance(raw, list):
                parts.extend(str(item) for item in raw)
            elif raw:
                parts.append(str(raw))
        return "\n".join(parts)
    return _text(value)


def _context(text: str, mention: str, radius: int = 60) -> str:
    lower = text.lower()
    pos = lower.find(mention.lower())
    if pos < 0:
        return text[: radius * 2]
    return text[max(0, pos - radius) : pos + len(mention) + radius]


def _best_source_weight(cluster: list[dict[str, Any]]) -> float:
    return max(float(row.get("weight") or 0) for row in cluster) if cluster else 0.0


def _item_index(item: dict[str, Any], position: int) -> int:
    try:
        value = int(item.get("index") or 0)
    except Exception:
        value = 0
    return value or position


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _unique(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _md(value: Any) -> str:
    return str(value or "-").replace("\n", " ").replace("|", "\\|")
