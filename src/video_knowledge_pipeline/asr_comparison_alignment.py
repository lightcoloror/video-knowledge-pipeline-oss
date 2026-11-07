"""Time-connected ASR alignment and conservative character disagreement evidence."""
from __future__ import annotations

import difflib
import math
import re
from types import SimpleNamespace
from typing import Any

from .text_normalization import compact_ascii_cjk_after_lowering


def align_groups(primary: list[Any], secondary: list[Any]) -> list[tuple[list[int], list[int]]]:
    # Bipartite overlap components support 1:N and N:M without consuming a
    # secondary segment once for each short primary sentence.
    count = len(primary)
    parents = list(range(count + len(secondary)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i: int, j: int) -> None:
        parents[find(j)] = find(i)

    events = []
    for side, rows in enumerate((primary, secondary)):
        for i, row in enumerate(rows):
            start, end = float(row.start), float(row.end)
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError("ASR alignment requires finite, positive, nonnegative time intervals")
            events.append((start, end, side, i))
    active: list[list[tuple[float, int]]] = [[], []]
    for start, end, side, i in sorted(events):
        active[0] = [(e, j) for e, j in active[0] if e > start]
        active[1] = [(e, j) for e, j in active[1] if e > start]
        for _, j in active[1 - side]:
            union(i if side == 0 else count + i, count + j if side == 0 else j)
        active[side].append((end, i))
    groups: dict[int, tuple[list[int], list[int]]] = {}
    for side, rows in enumerate((primary, secondary)):
        for i in range(len(rows)):
            groups.setdefault(find(i + (count if side else 0)), ([], []))[side].append(i)
    return sorted(groups.values(), key=lambda g: min(
        [primary[i].start for i in g[0]] + [secondary[i].start for i in g[1]]))


def aggregate(rows: list[Any], indexes: list[int]) -> tuple[Any | None, list[dict[str, Any]]]:
    if not indexes:
        return None, []
    pieces, spans = [], []
    offset = 0
    for index in sorted(indexes, key=lambda i: (rows[i].start, i)):
        text = str(rows[index].text or "")
        # Keep original characters and exact offsets for patch projection.
        spans.append({"index": index, "start": offset, "end": offset + len(text)})
        pieces.append(text)
        offset += len(text)
    return SimpleNamespace(start=min(rows[i].start for i in indexes),
                           end=max(rows[i].end for i in indexes), text="".join(pieces)), spans


def character_differences(left: str, right: str) -> list[dict[str, Any]]:
    def tokens(text: str) -> list[tuple[int, str]]:
        return [(i, c.casefold()) for i, c in enumerate(text)
                if c.isalnum() or c in "+-%％￥¥$€£<>≤≥±≈−–" or (c in ".,:/" and i and i + 1 < len(text) and text[i-1].isdigit() and text[i+1].isdigit())]
    a, b = tokens(left), tokens(right)
    rows = []
    for tag, a0, a1, b0, b1 in difflib.SequenceMatcher(
            a=[v for _, v in a], b=[v for _, v in b], autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        start = a[a0][0] if a0 < len(a) else len(left)
        end = a[a1 - 1][0] + 1 if a1 > a0 else start
        bs = b[b0][0] if b0 < len(b) else len(right)
        be = b[b1 - 1][0] + 1 if b1 > b0 else bs
        rows.append({"operation": tag, "primary_char_start": start, "primary_char_end": end,
                     "secondary_char_start": bs, "secondary_char_end": be,
                     "primary_text": left[start:end], "secondary_text": right[bs:be]})
    return rows


def critical_differences(left: str, right: str, terms: list[str] | None = None) -> list[str]:
    patterns = {
        "currency": r"人民币|美元|美金|港元|港币|欧元|英镑|日元|[￥¥$€£]",
        "numeric_operator_or_sign": r"<=|>=|[<>≤≥±≈]|[+−–-](?=\d)",
        "number_or_amount": r"[-+]?\d+(?:[.,]\d+)*(?:[%％万亿千百元块岁年月日号点小时分钟秒]|人民币|美元)*|[零〇一二两三四五六七八九十百千万亿]+",
        "negation": r"没有|不能|不得|不必|无需|尚未|未曾|不是|不会|不|没|未|无|别",
        "date_or_time": r"\d{1,4}[-/:]\d{1,2}(?:[-/:]\d{1,2})?|[零〇一二两三四五六七八九十百\d]+[年月日号点]",
    }
    flags = [name for name, pattern in patterns.items()
             if re.findall(pattern, left) != re.findall(pattern, right)]
    if any(left.count(t) != right.count(t) for t in (terms or []) if t):
        flags.append("named_entity_or_glossary")
    # Names cannot be reliably identified from unlabelled Chinese text. Retain
    # every lexical change as review evidence instead of claiming NER coverage.
    if compact_ascii_cjk_after_lowering(left) != compact_ascii_cjk_after_lowering(right):
        flags.append("lexical_change")
    return flags


def locate_primary_span(spans: list[dict[str, Any]], start: int, end: int) -> tuple[int, int, int] | None:
    candidates = [s for s in spans if s["start"] <= start and end <= s["end"]]
    # Insertions exactly between two original segments are ambiguous.
    if len(candidates) != 1:
        return None
    span = candidates[0]
    return int(span["index"]), start - span["start"], end - span["start"]
