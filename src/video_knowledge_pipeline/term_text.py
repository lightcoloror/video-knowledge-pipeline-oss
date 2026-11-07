from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .storage import read_json_object_or_empty as _read_json_object
from .text_normalization import compact_ascii_cjk_after_lowering as _normalise_term

AUTO_REPLACE_MIN_CONFIDENCE = 0.88


def apply_high_confidence_term_replacements(
    value: Any,
    item: dict[str, Any],
    *,
    min_confidence: float = AUTO_REPLACE_MIN_CONFIDENCE,
) -> str:
    """Return display text with high-confidence term candidates normalised.

    This is intentionally export-only: it does not mutate ASR, subtitles, OCR, or
    timeline evidence. Raw variants remain available in term_candidates.
    """

    text = str(value or "").strip()
    if not text:
        return ""
    replacements = _replacement_pairs(item, min_confidence=min_confidence)
    for raw, canonical in replacements:
        text = _replace_term(text, raw, canonical)
    return text


def high_confidence_term_replacements(
    item: dict[str, Any],
    *,
    min_confidence: float = AUTO_REPLACE_MIN_CONFIDENCE,
) -> list[dict[str, str]]:
    return [{"raw": raw, "canonical": canonical} for raw, canonical in _replacement_pairs(item, min_confidence=min_confidence)]


def is_high_confidence_term_candidate(
    row: dict[str, Any],
    *,
    min_confidence: float = AUTO_REPLACE_MIN_CONFIDENCE,
) -> bool:
    canonical = str(row.get("canonical_term") or "").strip()
    if not canonical:
        return False
    try:
        confidence = float(row.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < min_confidence:
        return False
    raw_mentions = row.get("raw_mentions") if isinstance(row.get("raw_mentions"), list) else []
    return any(_display_differs(raw, canonical) for raw in raw_mentions if _valid_raw_term(raw))


def _replacement_pairs(item: dict[str, Any], *, min_confidence: float) -> list[tuple[str, str]]:
    rows = item.get("term_candidates") if isinstance(item.get("term_candidates"), list) else []
    pairs: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or not is_high_confidence_term_candidate(row, min_confidence=min_confidence):
            continue
        canonical = str(row.get("canonical_term") or "").strip()
        raw_values = [str(raw or "").strip() for raw in (row.get("raw_mentions") or [])]
        raw_values.extend(_implied_display_variants(canonical))
        for raw_text in raw_values:
            if not _valid_raw_term(raw_text):
                continue
            if not _display_differs(raw_text, canonical):
                continue
            if _unsafe_short_subterm(raw_text, canonical):
                continue
            pairs.append((raw_text, canonical))
    pairs.sort(key=lambda pair: (-len(pair[0]), pair[0].lower()))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw, canonical in pairs:
        key = (raw.lower(), canonical)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((raw, canonical))
    return deduped


def _replace_term(text: str, raw: str, canonical: str) -> str:
    parts = [re.escape(part) for part in raw.split() if part]
    escaped = r"\s+".join(parts) if len(parts) > 1 else re.escape(raw)
    if _ascii_edge(raw):
        pattern = rf"(?<![A-Za-z0-9_-]){escaped}(?![A-Za-z0-9_-])"
    else:
        pattern = escaped
    # Glossary values can include literal backslashes.  Use a callable so re
    # does not parse the canonical value as a replacement-template escape.
    return re.sub(pattern, lambda _match: canonical, text, flags=re.IGNORECASE)


def _implied_display_variants(canonical: str) -> list[str]:
    variants: list[str] = []
    if "-" in canonical:
        variants.append(canonical.replace("-", " "))
    groups = re.findall(r"[A-Za-z0-9]+", canonical)
    if len(groups) >= 2 and 1 < len(groups[0]) <= 4:
        variants.append(" ".join([*groups[0], *groups[1:]]))
    return variants


def _display_differs(raw: Any, canonical: str) -> bool:
    return str(raw or "").strip() != canonical.strip()


def _unsafe_short_subterm(raw: str, canonical: str) -> bool:
    raw_norm = _normalise_term(raw)
    canonical_norm = _normalise_term(canonical)
    if not raw_norm or not canonical_norm or raw_norm == canonical_norm:
        return False
    return raw_norm in canonical_norm and len(raw_norm) / max(1, len(canonical_norm)) < 0.8


def _valid_raw_term(value: Any) -> bool:
    text = str(value or "").strip()
    if re.search(r"[A-Za-z]", text):
        return len(text) >= 3
    # Chinese aliases are only consumed from reviewed, high-confidence
    # glossary rows. Require two characters to avoid unsafe single-character
    # substitutions while still supporting names such as 米娅.
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)
    return len(cjk) >= 2 and len(cjk) == len(text)


def _ascii_edge(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped and re.match(r"[A-Za-z0-9_-]", stripped[0]) and re.search(r"[A-Za-z0-9_-]$", stripped))

def load_bundle_term_replacements(
    bundle_dir: str | Path,
    *,
    min_confidence: float = AUTO_REPLACE_MIN_CONFIDENCE,
) -> list[tuple[str, str]]:
    """Load reviewed glossary replacements for export-time text normalisation.

    Only imported/reviewed glossary terms are used here. Draft Codex packs can
    exist in the bundle, but rows marked review_required remain non-applicable.
    """

    root = Path(bundle_dir).expanduser().resolve()
    pairs: list[tuple[str, str]] = []
    for path in _bundle_glossary_paths(root):
        data = _read_json_object(path)
        terms = data.get("terms") if isinstance(data.get("terms"), list) else []
        for item in terms:
            if not isinstance(item, dict):
                continue
            if bool(item.get("review_required")):
                continue
            try:
                confidence = float(item.get("confidence") or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < min_confidence:
                continue
            canonical = str(item.get("canonical") or item.get("canonical_term") or item.get("term") or "").strip()
            aliases = [str(value or "").strip() for value in item.get("aliases") or item.get("raw_mentions") or []]
            for alias in aliases:
                if not canonical or not _valid_raw_term(alias):
                    continue
                if not _display_differs(alias, canonical):
                    continue
                if _unsafe_short_subterm(alias, canonical):
                    continue
                pairs.append((alias, canonical))
    pairs.sort(key=lambda pair: (-len(pair[0]), pair[0].lower()))
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw, canonical in pairs:
        key = (raw.lower(), canonical)
        if key in seen:
            continue
        seen.add(key)
        out.append((raw, canonical))
    return out


def apply_term_replacement_pairs(value: Any, pairs: list[tuple[str, str]]) -> str:
    text = str(value or "")
    if not text or not pairs:
        return text
    for raw, canonical in pairs:
        text = _replace_term(text, raw, canonical)
    return text


def _bundle_glossary_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    manifest = _read_json_object(root / "manifest.json")
    value = str(manifest.get("term_arbitration_glossary_json") or "").strip()
    if value:
        candidate = Path(value)
        paths.append(candidate if candidate.is_absolute() else root / candidate)
    default_path = root / "term-arbitration-glossary.json"
    if default_path.exists():
        paths.append(default_path)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve()).lower()
        except Exception:
            key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped
