from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from .storage import read_json, write_json

SCHEMA = "video_knowledge_pipeline.subtitle_regression_guard.v1"


def load_canonical_corrections(
    corrections_source: str | Path | dict[str, Any],
) -> dict[str, str]:
    """
    Loads text correction mapping from a dict, a manuscript corrections json,
    or a targeted review diffs file (e.g. draft_vs_input_diffs.json).
    """
    if isinstance(corrections_source, dict):
        # Could be raw mapping or structured schema
        if "corrections" in corrections_source and isinstance(corrections_source["corrections"], dict):
            return {str(k): str(v) for k, v in corrections_source["corrections"].items()}
        return {str(k): str(v) for k, v in corrections_source.items()}

    p = Path(corrections_source)
    if not p.is_file():
        raise FileNotFoundError(f"Corrections file not found: {p}")

    data = read_json(p) if hasattr(read_json, "__call__") else json.loads(p.read_text(encoding="utf-8"))

    # Case 1: Targeted diff list: [{"tag": "replace", "draft": "...", "input": "..."}, ...]
    if isinstance(data, list):
        out: dict[str, str] = {}
        for item in data:
            if isinstance(item, dict) and item.get("tag") == "replace":
                d_val = str(item.get("draft", "")).strip()
                i_val = str(item.get("input", "")).strip()
                if d_val and i_val and d_val != i_val:
                    out[d_val] = i_val
        return out

    # Case 2: Dict format
    if isinstance(data, dict):
        if "corrections" in data and isinstance(data["corrections"], dict):
            return {str(k): str(v) for k, v in data["corrections"].items()}
        return {str(k): str(v) for k, v in data.items()}

    return {}


def apply_canonical_patches(
    text_or_cues: str | list[dict[str, Any]],
    corrections: dict[str, str],
) -> tuple[str | list[dict[str, Any]], int]:
    """
    Applies canonical corrections (SSOT) to raw text or a list of cue dicts.
    Returns (patched_content, total_replacements_applied).
    """
    if not corrections:
        return text_or_cues, 0

    total_applied = 0

    if isinstance(text_or_cues, str):
        content = text_or_cues
        for typo, correct in corrections.items():
            if typo in content:
                count = content.count(typo)
                content = content.replace(typo, correct)
                total_applied += count
        return content, total_applied

    # List of cue dicts (e.g. [{"text": "...", "start": ...}])
    patched_cues = []
    for cue in text_or_cues:
        cue_copy = dict(cue)
        text_val = str(cue_copy.get("text", ""))
        for typo, correct in corrections.items():
            if typo in text_val:
                count = text_val.count(typo)
                text_val = text_val.replace(typo, correct)
                total_applied += count
        cue_copy["text"] = text_val
        patched_cues.append(cue_copy)

    return patched_cues, total_applied


def check_text_regression(
    srt_or_text: str | Path,
    corrections: dict[str, str],
) -> list[dict[str, str]]:
    """
    Scans subtitle content (handling inner line wraps) against a corrections blacklist.
    Returns list of detected regression violations.
    """
    if isinstance(srt_or_text, Path):
        content = srt_or_text.read_text(encoding="utf-8")
    else:
        content = str(srt_or_text)

    # Flatten newlines within subtitle cue bodies to detect words split across lines
    flattened_cues = re.sub(r"\n(?!\d+\n\d{2}:)", "", content)

    violations = []
    for typo, correct in corrections.items():
        if len(typo) >= 2 and (typo in flattened_cues or typo in content):
            violations.append({
                "typo": typo,
                "expected": correct,
                "message": f"Found reverted typo '{typo}' -> Expected '{correct}'"
            })
    return violations


def assert_no_text_regression(
    srt_path_or_content: str | Path,
    corrections_source: str | Path | dict[str, Any],
) -> None:
    """
    Hard CI Gate: Throws AssertionError if ANY resurrected typo is found.
    Used before video rendering or artifact publication.
    """
    corrections = load_canonical_corrections(corrections_source)
    name = srt_path_or_content.name if isinstance(srt_path_or_content, Path) else "inline_content"
    violations = check_text_regression(srt_path_or_content, corrections)

    if violations:
        details = "\n".join(f"  ❌ {v['message']}" for v in violations)
        msg = f"[CRITICAL GATE FAILED] Subtitle '{name}' has {len(violations)} regressions:\n{details}"
        raise AssertionError(msg)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify subtitles against canonical corrections dictionary to prevent regression."
    )
    parser.add_argument("srt", type=Path, help="Path to SRT subtitle file to verify")
    parser.add_argument(
        "--corrections",
        type=Path,
        required=True,
        help="Path to JSON corrections dictionary or review diffs file",
    )
    args = parser.parse_args(argv)

    try:
        assert_no_text_regression(args.srt, args.corrections)
        print(f"✅ [Gate Passed] Subtitle '{args.srt.name}' passed all regression checks.")
        return 0
    except AssertionError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during check: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
