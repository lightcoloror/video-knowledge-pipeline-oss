from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.summary_consistency.v1"
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?|[一二三四五六七八九十百千万两]+")
_ASCII_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_.+-]{2,}(?:\s+[A-Z][A-Za-z0-9_.+-]{1,})*\b")


def run_summary_consistency_check(
    bundle_dir: str | Path,
    *,
    summary_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Check chapter-to-summary entity, number and event consistency with abstention."""

    root = Path(bundle_dir).expanduser().resolve()
    chapters = _chapter_evidence(root)
    summary_file = _summary_path(root, summary_path)
    summary_text = summary_file.read_text(encoding="utf-8") if summary_file and summary_file.exists() else ""
    evidence_text = "\n".join(row["text"] for row in chapters)
    known_entities = sorted(_entities(evidence_text))
    summary_entities = sorted(_entities(summary_text))
    known_numbers = sorted(_numbers(evidence_text))
    summary_numbers = sorted(_numbers(summary_text))

    entity_rows = [
        {
            "entity": entity,
            "status": "supported" if entity in known_entities else "unknown",
            "evidence_chapters": [row["chapter_id"] for row in chapters if entity in row["text"]],
        }
        for entity in summary_entities
    ]
    number_rows = [
        {
            "value": number,
            "status": "supported" if number in known_numbers else "conflict",
            "evidence_chapters": [row["chapter_id"] for row in chapters if number in _numbers(row["text"])],
        }
        for number in summary_numbers
    ]
    event_rows = _event_rows(chapters, summary_text)
    conflicts = [{"kind": "number", **row} for row in number_rows if row["status"] == "conflict"]
    unknown = (
        [{"kind": "entity", **row} for row in entity_rows if row["status"] == "unknown"]
        + [{"kind": "event", **row} for row in event_rows if row["status"] == "unknown"]
    )
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "summary_path": str(summary_file) if summary_file else "",
        "status": "conflict" if conflicts else ("passed_with_unknowns" if unknown else "passed"),
        "ok": bool(summary_text and chapters) and not conflicts,
        "chapter_count": len(chapters),
        "known_entity_count": len(known_entities),
        "known_number_count": len(known_numbers),
        "entity_checks": entity_rows,
        "number_checks": number_rows,
        "event_checks": event_rows,
        "conflicts": conflicts,
        "unknown_or_insufficient": unknown,
        "quality": {
            "high_risk_conflicts": len(conflicts),
            "unknown_items": len(unknown),
            "supported_entity_ratio": round(sum(row["status"] == "supported" for row in entity_rows) / max(1, len(entity_rows)), 4),
            "supported_number_ratio": round(sum(row["status"] == "supported" for row in number_rows) / max(1, len(number_rows)), 4),
        },
        "operator_boundary": {
            "evidence_insufficient_is_allowed": True,
            "unknown_is_not_promoted_to_fact": True,
            "only_explicit_number_conflicts_block": True,
            "no_llm_or_cloud_call": True,
            "does_not_rewrite_summary": True,
        },
        "artifacts": {
            "json": "exports/summary-consistency.json",
            "markdown": "exports/summary-consistency.md",
        },
        "updated_at": now_iso(),
    }
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "summary-consistency.json", result)
        (exports / "summary-consistency.md").write_text(_render_markdown(result), encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest = _mapping(manifest_path)
        manifest["summary_consistency_json"] = "exports/summary-consistency.json"
        manifest["summary_consistency_markdown"] = "exports/summary-consistency.md"
        manifest["summary_consistency_summary"] = {
            "status": result["status"],
            "high_risk_conflicts": len(conflicts),
            "unknown_items": len(unknown),
            "updated_at": result["updated_at"],
        }
        write_json(manifest_path, manifest)
    return result


def _chapter_evidence(root: Path) -> list[dict[str, str]]:
    revisions = _mapping(root / "exports" / "smart-summary-section-llm-revisions.json")
    rows = revisions.get("rows") if isinstance(revisions.get("rows"), list) else []
    chapters = [
        {
            "chapter_id": str(row.get("section_id") or f"section-{index + 1}"),
            "title": str(row.get("title") or ""),
            "text": _section_text(row),
        }
        for index, row in enumerate(rows)
        if isinstance(row, dict) and _section_text(row)
    ]
    if chapters:
        return chapters
    pack = _mapping(root / "exports" / "smart-summary-chapters.json")
    return [
        {
            "chapter_id": str(row.get("index") or f"chapter-{index + 1}"),
            "title": str(row.get("title") or ""),
            "text": " ".join(
                [
                    str(row.get("title") or ""),
                    " ".join(str(value) for value in row.get("summary_sentences") or []),
                    " ".join(str(value) for value in row.get("key_points") or []),
                    " ".join(str(value) for value in row.get("actions") or []),
                ]
            ),
        }
        for index, row in enumerate(pack.get("chapters") or [])
        if isinstance(row, dict)
    ]


def _event_rows(chapters: list[dict[str, str]], summary_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chapter in chapters:
        title = str(chapter.get("title") or "").strip()
        tokens = _topic_tokens(title)
        if not tokens:
            rows.append({"chapter_id": chapter["chapter_id"], "event": title, "status": "unknown", "reason": "chapter_title_has_no_stable_topic_token"})
            continue
        supported = any(token in summary_text for token in tokens)
        rows.append(
            {
                "chapter_id": chapter["chapter_id"],
                "event": title,
                "status": "supported" if supported else "unknown",
                "reason": "" if supported else "chapter_event_not_explicit_in_reduced_summary",
            }
        )
    return rows


def _entities(text: str) -> set[str]:
    entities = {match.group(0).strip() for match in _ASCII_ENTITY_RE.finditer(str(text or ""))}
    quoted = re.findall(r"[《「“](.{2,24}?)[》」”]", str(text or ""))
    entities.update(value.strip() for value in quoted if value.strip())
    return entities


def _numbers(text: str) -> set[str]:
    return {match.group(0) for match in _NUMBER_RE.finditer(str(text or ""))}


def _topic_tokens(title: str) -> list[str]:
    ascii_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,}", title)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,8}", title)
    return [*ascii_tokens, *chinese]


def _summary_path(root: Path, value: str | Path | None) -> Path | None:
    if value:
        path = Path(value).expanduser()
        path = path if path.is_absolute() else root / path
        return path.resolve() if path.exists() else None
    for candidate in (root / "exports" / "smart-summary.codex.md", root / "exports" / "smart-summary.md"):
        if candidate.exists():
            return candidate.resolve()
    return None


def _section_text(row: dict[str, Any]) -> str:
    for key in ("final_markdown", "revised_markdown", "draft_markdown", "markdown", "content"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Summary Entity/Event Consistency",
        "",
        f"- Status: {result.get('status')}",
        f"- High-risk conflicts: {(result.get('quality') or {}).get('high_risk_conflicts', 0)}",
        f"- Unknown / insufficient: {(result.get('quality') or {}).get('unknown_items', 0)}",
        "",
        "## Conflicts",
        "",
    ]
    if result.get("conflicts"):
        for row in result["conflicts"]:
            lines.append(f"- {row.get('kind')}: {row.get('value') or row.get('entity')}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Unknown / Evidence Insufficient", ""])
    for row in result.get("unknown_or_insufficient") or []:
        lines.append(f"- {row.get('kind')}: {row.get('entity') or row.get('event')} ({row.get('reason', '')})")
    return "\n".join(lines).rstrip() + "\n"