from __future__ import annotations

import json
import re


# Adapted from alpha03123/vsummary src/backend/shared/llm/json_mode.py.
def extract_json_document(raw_text: str, *, require_object: bool = False) -> object:
    """Extract one JSON document from direct, fenced, or surrounding model text."""

    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("model returned empty content; cannot parse JSON")
    direct = _try_json_load(text)
    if direct is not None:
        if require_object and not isinstance(direct, dict):
            raise ValueError("top-level JSON must be an object")
        return direct
    fenced = _extract_fenced_json(text, require_object=require_object)
    if fenced is not None:
        return fenced
    balanced = _extract_balanced_json(text, require_object=require_object)
    if balanced is not None:
        return balanced
    raise ValueError(
        "model output did not contain parseable JSON object"
        if require_object
        else "model output did not contain parseable JSON"
    )


def extract_last_json_document(
    raw_text: str, *, require_object: bool = False
) -> object:
    """Extract the last complete JSON document from subprocess output."""

    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("subprocess returned empty stdout; cannot parse JSON")
    direct = _try_json_load(text)
    if direct is not None:
        if require_object and not isinstance(direct, dict):
            raise ValueError("top-level JSON must be an object")
        return direct
    opening_chars = "{" if require_object else "{["
    documents: list[object] = []
    start_index = 0
    while start_index < len(text):
        while start_index < len(text) and text[start_index] not in opening_chars:
            start_index += 1
        if start_index >= len(text):
            break
        end_index = _find_balanced_end(text, start_index)
        if end_index is None:
            start_index += 1
            continue
        loaded = _try_json_load(text[start_index : end_index + 1])
        if loaded is None:
            start_index += 1
            continue
        if require_object and not isinstance(loaded, dict):
            start_index = end_index + 1
            continue
        documents.append(loaded)
        start_index = end_index + 1
    if documents:
        return documents[-1]
    raise ValueError(
        "subprocess stdout did not contain parseable JSON object"
        if require_object
        else "subprocess stdout did not contain parseable JSON"
    )


def _try_json_load(candidate: str) -> object | None:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _extract_fenced_json(text: str, *, require_object: bool = False) -> object | None:
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        loaded = _try_json_load(match.group(1).strip())
        if loaded is None:
            continue
        if require_object and not isinstance(loaded, dict):
            continue
        return loaded
    return None


def _extract_balanced_json(text: str, *, require_object: bool = False) -> object | None:
    opening_chars = "{" if require_object else "{["
    for start_index, char in enumerate(text):
        if char not in opening_chars:
            continue
        end_index = _find_balanced_end(text, start_index)
        if end_index is None:
            continue
        loaded = _try_json_load(text[start_index : end_index + 1])
        if loaded is None:
            continue
        if require_object and not isinstance(loaded, dict):
            continue
        return loaded
    return None


def _find_balanced_end(text: str, start_index: int) -> int | None:
    opening = text[start_index]
    expected_closing = "}" if opening == "{" else "]"
    stack: list[str] = [opening]
    in_string = False
    escaped = False
    for index in range(start_index + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            stack.append(char)
            continue
        if char not in "}]":
            continue
        current_open = stack.pop()
        if (current_open == "{" and char != "}") or (
            current_open == "[" and char != "]"
        ):
            return None
        if not stack:
            return index if char == expected_closing else None
    return None
