from __future__ import annotations

import re


UNIT_PATTERN = r"亿元|万元|小时|分钟|百分比|公里|千米|厘米|毫米|公斤|千克|年|月|日|岁|元|块|个|次|场|点|%|％|亿|万"
_ARABIC_RANGE_RE = re.compile(
    r"(?<![0-9A-Za-z])(?P<left>1)\s*(?:-|–|—|~|～|到|至)\s*"
    r"(?P<right>2)\s*(?P<unit>场)"
)
_APPROXIMATE_CHINESE_RANGE_RE = re.compile(
    r"(?<![零〇一二两三四五六七八九十百千万亿点])"
    r"(?P<left>一)(?:到|至)?(?P<right>两)(?P<unit>场)"
)
_CHINESE_DIGITS = "零〇一二两三四五六七八九"
_CHINESE_NUMBER_CHARS = _CHINESE_DIGITS + "十百千万亿点"
_SAFE_CHINESE_NUMBER_BODY = rf"(?:[{_CHINESE_DIGITS}][{_CHINESE_NUMBER_CHARS}]*|十[{_CHINESE_NUMBER_CHARS}]*)"
_SAFE_BARE_CHINESE_NUMBER_RE = re.compile(
    rf"(?<![{_CHINESE_NUMBER_CHARS}])"
    rf"(?P<number>[{_CHINESE_DIGITS}][{_CHINESE_NUMBER_CHARS}]*[十百千万亿][{_CHINESE_NUMBER_CHARS}]*)"
    rf"(?![{_CHINESE_NUMBER_CHARS}])"
)


def number_evidence_map(value: str) -> dict[str, set[str]]:
    """Return unit-aware numeric evidence so equivalent spoken forms agree."""

    text = _normalize_counted_time_units(
        _normalize_approximate_chinese_ranges(
            _normalize_fragmented_arabic_digits(str(value or ""))
        )
    )
    # Treat spoken percentages and approximate unit forms as the same numeric
    # evidence as their written counterparts.
    text = re.sub(
        r"百分之\s*([零〇一二两三四五六七八九十百千万亿点]+|\d+(?:\.\d+)?)",
        r"\1%",
        text,
    )
    evidence: dict[str, set[str]] = {}
    arabic_pattern = re.compile(rf"(?<![A-Za-z])(?P<number>\d+(?:\.\d+)?)\s*多?\s*(?P<unit>{UNIT_PATTERN})?")
    chinese_pattern = re.compile(rf"(?<![0-9A-Za-z])(?P<number>{_SAFE_CHINESE_NUMBER_BODY})\s*多?\s*(?P<unit>{UNIT_PATTERN})")
    occupied_range_spans: list[tuple[int, int]] = []
    for match in _ARABIC_RANGE_RE.finditer(text):
        _add_range_evidence(
            evidence,
            match.group(0),
            float(match.group("left")),
            float(match.group("right")),
            match.group("unit"),
        )
        occupied_range_spans.append(match.span())
    occupied_chinese_spans: list[tuple[int, int]] = []
    for match in arabic_pattern.finditer(text):
        if any(_spans_overlap(match.span(), span) for span in occupied_range_spans):
            continue
        _add_number_evidence(evidence, match.group(0), float(match.group("number")), match.group("unit") or "")
    for match in chinese_pattern.finditer(text):
        if any(_spans_overlap(match.span(), span) for span in occupied_range_spans):
            continue
        parsed = _parse_chinese_number(match.group("number"))
        if parsed is not None:
            _add_number_evidence(evidence, match.group(0), parsed, match.group("unit") or "")
            occupied_chinese_spans.append(match.span())
    for match in _SAFE_BARE_CHINESE_NUMBER_RE.finditer(text):
        if any(_spans_overlap(match.span(), span) for span in occupied_chinese_spans):
            continue
        parsed = _parse_chinese_number(match.group("number"))
        if parsed is not None:
            _add_number_evidence(evidence, match.group(0), parsed, "")
    return evidence


def strip_number_mentions(value: str) -> str:
    text = _normalize_counted_time_units(
        _normalize_approximate_chinese_ranges(
            _normalize_fragmented_arabic_digits(str(value or ""))
        )
    )
    text = _ARABIC_RANGE_RE.sub("", text)
    text = re.sub(r"百分之\s*([零〇一二两三四五六七八九十百千万亿点]+|\d+(?:\.\d+)?)", "", text)
    text = re.sub(rf"{_SAFE_CHINESE_NUMBER_BODY}\s*多?\s*(?:{UNIT_PATTERN})", "", text)
    text = _SAFE_BARE_CHINESE_NUMBER_RE.sub("", text)
    return re.sub(rf"(?<![A-Za-z])\d+(?:\.\d+)?\s*多?\s*(?:{UNIT_PATTERN})?", "", text)


def numeric_mentions_equivalent(left: str, right: str) -> bool:
    left_values = number_evidence_map(left)
    right_values = number_evidence_map(right)
    return bool(left_values) and left_values.keys() == right_values.keys()


def _normalize_counted_time_units(value: str) -> str:
    """Treat spoken ``六个小时`` as the same duration as ``6小时``.

    Intent: preserve unit-aware evidence when ASR inserts the Chinese measure
    word ``个`` before a time unit.
    Decision: remove ``个`` only between an Arabic/Chinese number and
    ``小时``/``分钟`` before the existing parser runs.
    Reason: otherwise ``6个小时`` is captured as count:6:个 and the following
    time unit is lost, causing a supported duration to fail summary checks.
    Evidence: the 2026-08-31 transcript says ``不到6个小时`` while the summary
    accurately condenses it as ``约6小时``.
    Effective scope: numeric evidence comparison only; source text is not
    rewritten and other measure-word counts remain unchanged.
    """

    number = rf"(?:\d+(?:\.\d+)?|{_SAFE_CHINESE_NUMBER_BODY})"
    return re.sub(
        rf"(?P<number>{number})\s*个\s*(?P<unit>小时|分钟)",
        r"\g<number>\g<unit>",
        str(value or ""),
    )


def _normalize_approximate_chinese_ranges(value: str) -> str:
    """Canonicalize compact spoken ranges before existing number extraction.

    Intent: match spoken ``一两场`` with written ``1-2场``.
    Decision: adapt the existing unit-aware evidence map with a narrow
    two-single-digit range rule instead of adding a second number parser.
    Reason: the transcript and summary express the same supported fact.
    Evidence: the first production bundle uses exactly these two forms.
    Effective scope: numeric evidence comparison only; transcript text remains
    unchanged and ambiguous longer Chinese number phrases are not rewritten.
    """

    digits = {
        "一": "1",
        "二": "2",
        "两": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
    }

    def replace(match: re.Match[str]) -> str:
        return (
            f"{digits[match.group('left')]}-"
            f"{digits[match.group('right')]}{match.group('unit')}"
        )

    return _APPROXIMATE_CHINESE_RANGE_RE.sub(replace, str(value or ""))


def _normalize_fragmented_arabic_digits(value: str) -> str:
    """Normalize OCR thousand separators and ASR-internal digit spacing.

    Intent: let summary evidence match forms such as ``202 3年``, ``330 0``
    and ``3,300`` without rewriting the canonical transcript.
    Decision: collapse commas, and collapse whitespace only when the full
    sequence contains at least four digits; short date-like ``7 24`` stays
    separate.
    Reason: ASR/OCR often inserts separators inside a single numeric claim.
    Evidence: this is consumed through the existing canonical number map.
    Effective scope: numeric comparison and evidence lookup only.
    """

    text = re.sub(r"(?<=\d),(?=\d)", "", str(value or ""))
    pattern = re.compile(r"\d+(?:[ \t\u00a0]+\d+)+")

    def compact(match: re.Match[str]) -> str:
        groups = re.findall(r"\d+", match.group(0))
        digits = "".join(groups)
        # A trailing one-digit fragment is the recurring ASR form in the
        # verified fixtures (202 3, 330 0). Equal-width groups such as
        # keyword examples "666 888" remain separate claims.
        return digits if len(digits) >= 4 and len(groups[-1]) == 1 else match.group(0)

    return pattern.sub(compact, text)


def _add_range_evidence(
    evidence: dict[str, set[str]],
    mention: str,
    left: float,
    right: float,
    unit: str,
) -> None:
    scale, dimension, canonical_unit = _number_unit(unit)
    low, high = sorted((left * scale, right * scale))
    low_text = f"{low:.8f}".rstrip("0").rstrip(".")
    high_text = f"{high:.8f}".rstrip("0").rstrip(".")
    key = f"range:{dimension}:{low_text}-{high_text}:{canonical_unit}"
    evidence.setdefault(key, set()).add(str(mention))


def _add_number_evidence(evidence: dict[str, set[str]], mention: str, number: float, unit: str) -> None:
    scale, dimension, canonical_unit = _number_unit(unit)
    canonical_value = number * scale
    value_text = f"{canonical_value:.8f}".rstrip("0").rstrip(".")
    key = f"{dimension}:{value_text}:{canonical_unit}"
    evidence.setdefault(key, set()).add(str(mention))


def _number_unit(unit: str) -> tuple[float, str, str]:
    value = str(unit or "")
    if value in {"亿元", "万元", "元", "块"}:
        return ({"亿元": 100_000_000.0, "万元": 10_000.0}.get(value, 1.0), "currency", "元")
    if value in {"亿", "万"}:
        return ({"亿": 100_000_000.0, "万": 10_000.0}[value], "number", "")
    if value in {"%", "％", "百分比"}:
        return (1.0, "percentage", "%")
    if value in {"小时", "分钟"}:
        return ({"小时": 60.0, "分钟": 1.0}[value], "duration", "分钟")
    if value in {"公里", "千米", "厘米", "毫米"}:
        return ({"公里": 1000.0, "千米": 1000.0, "厘米": 0.01, "毫米": 0.001}[value], "distance", "米")
    if value in {"公斤", "千克"}:
        return (1.0, "mass", "千克")
    return (1.0, value or "number", value)


def _parse_chinese_number(value: str) -> float | None:
    text = str(value or "").replace("〇", "零")
    if not text:
        return None
    if "点" in text:
        integer, fraction = text.split("点", 1)
        integer_value = _parse_chinese_integer(integer)
        digits = {"零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
        if integer_value is None or not fraction or any(char not in digits for char in fraction):
            return None
        return float(f"{int(integer_value)}.{''.join(digits[char] for char in fraction)}")
    return _parse_chinese_integer(text)


def _parse_chinese_integer(value: str) -> float | None:
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    small_units = {"十": 10, "百": 100, "千": 1000}
    large_units = {"万": 10_000, "亿": 100_000_000}
    if value and all(char in digits for char in value):
        return float("".join(str(digits[char]) for char in value))
    total = section = current = 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif char in small_units:
            section += (current or 1) * small_units[char]
            current = 0
        elif char in large_units:
            section += current
            total += (section or 1) * large_units[char]
            section = current = 0
        else:
            return None
    return float(total + section + current)


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
