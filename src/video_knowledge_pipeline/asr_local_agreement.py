from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any


SCHEMA = "video_knowledge_pipeline.asr_local_agreement.v1"
TIMESTAMPED_SCHEMA = "video_knowledge_pipeline.asr_timestamped_local_agreement.v1"
BOUNDARY_LCS_SCHEMA = "video_knowledge_pipeline.asr_boundary_lcs_dedup.v1"
UPSTREAM_PROJECT = "ufal/SimulStreaming"
UPSTREAM_COMMIT = "077ea37d5ab4ff98bc567e4507f140dc4e5d5ad6"
UPSTREAM_API = "LocalAgreement.trim_longest_common_prefix"
UPSTREAM_LICENSE = "MIT"
TIMESTAMPED_UPSTREAM_PROJECT = "ufal/whisper_streaming"
TIMESTAMPED_UPSTREAM_COMMIT = "6da90b44b7e50d79695e68166d2a2c7609c75abb"
TIMESTAMPED_UPSTREAM_API = "HypothesisBuffer.insert/flush"
BOUNDARY_LCS_UPSTREAM_PROJECT = "CrispASR / NVIDIA NeMo"
BOUNDARY_LCS_UPSTREAM_COMMIT = "9deefe8f47273722415e4b4be5d87361b96177c9"
BOUNDARY_LCS_UPSTREAM_API = "crispasr_lcs.lcs_dedup_prefix_count"


def measure_local_agreement(
    left_text: Any,
    right_text: Any,
    *,
    language: str = "",
) -> dict[str, Any]:
    """Measure a candidate stable prefix without changing either transcript.

    Adapted from SimulStreaming's MIT-licensed LocalAgreement policy. VKP uses
    the result only as review evidence for overlapping ASR chunks; it never
    confirms, deletes, or merges transcript text automatically.
    """

    left = str(left_text or "").strip()
    right = str(right_text or "").strip()
    token_mode = _token_mode(language, left, right)
    left_units = _units(left, token_mode)
    right_units = _units(right, token_mode)

    common_count = 0
    for left_unit, right_unit in zip(left_units, right_units, strict=False):
        if left_unit != right_unit:
            break
        common_count += 1

    shorter_count = min(len(left_units), len(right_units))
    return {
        "schema": SCHEMA,
        "token_mode": token_mode,
        "left_unit_count": len(left_units),
        "right_unit_count": len(right_units),
        "common_prefix_unit_count": common_count,
        "common_prefix": _render(left_units[:common_count], token_mode),
        "agreement_over_shorter": round(
            common_count / shorter_count if shorter_count else 0.0,
            6,
        ),
        "exact_match": bool(left_units) and left_units == right_units,
        "candidate_only": True,
        "automatic_merge_allowed": False,
        "upstream": {
            "project": UPSTREAM_PROJECT,
            "commit": UPSTREAM_COMMIT,
            "api": UPSTREAM_API,
            "license": UPSTREAM_LICENSE,
        },
    }


def measure_boundary_lcs_dedup(
    left_text: Any,
    right_text: Any,
    *,
    language: str = "",
    maximum_units: int = 256,
    minimum_match_units: int = 8,
    maximum_edge_gap_units: int = 6,
) -> dict[str, Any]:
    """Adapt CrispASR/NeMo overlap-save LCS to adjacent text chunks.

    Intent: remove only a confidently repeated prefix from an untimed right
    chunk while preserving the raw chunk outputs.
    Decision: use Python difflib matching blocks over punctuation-free units,
    constrained to the previous tail and current head.
    Reason: SenseVoice may return chunk text without sentence/word timestamps;
    leaving overlap unchanged duplicates content, while unconstrained fuzzy
    matching could delete novel words.
    Evidence: pinned CrispASR ``crispasr_lcs.h`` and NeMo LCS stitching
    semantics, with VKP edge-proximity and minimum-length fail-closed gates.
    Effective scope: adjacent local ASR overlap prefixes only. A low-confidence
    result remains review evidence and never changes canonical text.
    """

    left = str(left_text or "")
    right = str(right_text or "")
    token_mode = _token_mode(language, left, right)
    left_units = _indexed_units(left, token_mode)[-max(1, int(maximum_units)) :]
    right_units = _indexed_units(right, token_mode)[: max(1, int(maximum_units))]
    matcher = SequenceMatcher(
        None,
        [row["normalized"] for row in left_units],
        [row["normalized"] for row in right_units],
        autojunk=False,
    )
    candidates: list[dict[str, int]] = []
    for match in matcher.get_matching_blocks():
        if match.size < int(minimum_match_units):
            continue
        left_gap = len(left_units) - (match.a + match.size)
        right_gap = match.b
        if left_gap > int(maximum_edge_gap_units):
            continue
        if right_gap > int(maximum_edge_gap_units):
            continue
        candidates.append(
            {
                "left_start": match.a,
                "right_start": match.b,
                "match_units": match.size,
                "left_edge_gap_units": left_gap,
                "right_edge_gap_units": right_gap,
            }
        )

    selected = max(
        candidates,
        key=lambda row: (
            row["match_units"],
            -row["left_edge_gap_units"],
            -row["right_edge_gap_units"],
        ),
        default=None,
    )
    if selected is None:
        return {
            "schema": BOUNDARY_LCS_SCHEMA,
            "status": "unmatched",
            "token_mode": token_mode,
            "matched_unit_count": 0,
            "right_prefix_unit_count": 0,
            "right_prefix_character_count": 0,
            "matched_text": "",
            "confidence": 0.0,
            "automatic_merge_allowed": False,
            "requires_human_review": bool(left.strip() and right.strip()),
            "upstream": _boundary_lcs_upstream(),
        }

    prefix_unit_count = selected["right_start"] + selected["match_units"]
    prefix_character_count = right_units[prefix_unit_count - 1]["end"]
    prefix_character_count = _include_trailing_separators(
        right, prefix_character_count
    )
    confidence = selected["match_units"] / max(
        1,
        selected["match_units"]
        + selected["left_edge_gap_units"]
        + selected["right_edge_gap_units"],
    )
    matched = right_units[
        selected["right_start"] : selected["right_start"] + selected["match_units"]
    ]
    return {
        "schema": BOUNDARY_LCS_SCHEMA,
        "status": "matched",
        "token_mode": token_mode,
        "matched_unit_count": selected["match_units"],
        "right_prefix_unit_count": prefix_unit_count,
        "right_prefix_character_count": prefix_character_count,
        "matched_text": _render(
            [row["normalized"] for row in matched], token_mode
        ),
        "left_edge_gap_units": selected["left_edge_gap_units"],
        "right_edge_gap_units": selected["right_edge_gap_units"],
        "confidence": round(confidence, 6),
        "automatic_merge_allowed": True,
        "requires_human_review": False,
        "upstream": _boundary_lcs_upstream(),
    }


def _indexed_units(text: str, token_mode: str) -> list[dict[str, Any]]:
    if token_mode == "character":
        return [
            {"normalized": character.casefold(), "start": index, "end": index + 1}
            for index, character in enumerate(text)
            if character.isalnum()
        ]
    return [
        {
            "normalized": match.group(0).casefold(),
            "start": match.start(),
            "end": match.end(),
        }
        for match in re.finditer(r"\w+", text, flags=re.UNICODE)
    ]


def _include_trailing_separators(text: str, start: int) -> int:
    index = max(0, int(start))
    while index < len(text) and not text[index].isalnum():
        index += 1
    return index


def _boundary_lcs_upstream() -> dict[str, str]:
    return {
        "project": BOUNDARY_LCS_UPSTREAM_PROJECT,
        "commit": BOUNDARY_LCS_UPSTREAM_COMMIT,
        "api": BOUNDARY_LCS_UPSTREAM_API,
        "license": UPSTREAM_LICENSE,
    }


def measure_timestamped_local_agreement(
    left_words: Any,
    right_words: Any,
    *,
    overlap_start: float,
    overlap_end: float,
) -> dict[str, Any]:
    """Adapt WhisperStreaming HypothesisBuffer evidence to a static overlap."""

    start = float(overlap_start)
    end = float(overlap_end)
    if end <= start:
        raise ValueError("timestamped local agreement requires a positive overlap")
    left = _timestamped_units(left_words, start=start, end=end)
    right = _timestamped_units(right_words, start=start, end=end)

    common_count = 0
    for left_unit, right_unit in zip(left, right, strict=False):
        if left_unit["normalized"] != right_unit["normalized"]:
            break
        common_count += 1

    shorter_count = min(len(left), len(right))
    first_start_delta = (
        abs(float(left[0]["start"]) - float(right[0]["start"]))
        if left and right
        else None
    )
    available = bool(left and right)
    aligned = bool(first_start_delta is not None and first_start_delta < 1.0)
    return {
        "schema": TIMESTAMPED_SCHEMA,
        "status": "available" if available else "unavailable",
        "overlap_start": round(start, 6),
        "overlap_end": round(end, 6),
        "left_word_count": len(left),
        "right_word_count": len(right),
        "common_prefix_word_count": common_count,
        "common_prefix_words": [row["text"] for row in left[:common_count]],
        "agreement_over_shorter": round(
            common_count / shorter_count if shorter_count else 0.0,
            6,
        ),
        "first_word_start_delta_seconds": (
            round(first_start_delta, 6) if first_start_delta is not None else None
        ),
        "within_upstream_one_second_tolerance": aligned,
        "usable_for_review_ranking": available and aligned,
        "candidate_only": True,
        "automatic_merge_allowed": False,
        "unavailable_reason": "" if available else "word_timestamps_missing_in_overlap",
        "upstream": {
            "project": TIMESTAMPED_UPSTREAM_PROJECT,
            "commit": TIMESTAMPED_UPSTREAM_COMMIT,
            "api": TIMESTAMPED_UPSTREAM_API,
            "license": UPSTREAM_LICENSE,
        },
    }


def _token_mode(language: str, left: str, right: str) -> str:
    normalized = str(language or "").strip().casefold().replace("_", "-")
    if normalized.split("-", 1)[0] in {"zh", "ja"}:
        return "character"
    if any(_is_cjk(character) for character in f"{left}{right}"):
        return "character"
    return "word"


def _units(text: str, token_mode: str) -> list[str]:
    if token_mode == "character":
        return [character.casefold() for character in text if not character.isspace()]
    return text.casefold().split()


def _render(units: list[str], token_mode: str) -> str:
    return "".join(units) if token_mode == "character" else " ".join(units)


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x3040 <= codepoint <= 0x30FF
    )


def _timestamped_units(
    value: Any,
    *,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in value or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("word") or raw.get("text") or "").strip()
        if not text:
            continue
        word_start = raw.get("start")
        word_end = raw.get("end")
        if not isinstance(word_start, (int, float)) or isinstance(word_start, bool):
            continue
        if not isinstance(word_end, (int, float)) or isinstance(word_end, bool):
            continue
        word_start = float(word_start)
        word_end = float(word_end)
        if word_end < word_start:
            continue
        if word_end == word_start:
            if not start <= word_start < end:
                continue
        elif word_end <= start or word_start >= end:
            continue
        rows.append(
            {
                "start": word_start,
                "end": word_end,
                "text": text,
                "normalized": text.casefold(),
            }
        )
    return sorted(rows, key=lambda row: (row["start"], row["end"], row["text"]))