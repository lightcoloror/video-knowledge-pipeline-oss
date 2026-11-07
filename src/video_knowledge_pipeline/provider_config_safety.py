from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Any


_FORBIDDEN_KEY_TERMS = (
    "apikey",
    "secret",
    "authorization",
    "cookie",
    "password",
    "credential",
)
_HEADER_CONTAINER_KEYS = {"headers", "extraheaders", "httpheaders", "requestheaders"}
_URL_KEY_TERMS = ("url", "endpoint", "baseurl")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)^\s*(?:bearer|basic)\s+\S+"),
    re.compile(r"(?i)^\s*sk-[A-Za-z0-9_-]{16,}"),
)


def secretless_provider_config(provider_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a shallow copy after rejecting inline credentials at any nesting level."""

    config = dict(provider_config or {})
    findings: list[str] = []
    _inspect_value(config, path="provider_config", findings=findings)
    if findings:
        unique = sorted(set(findings))
        raise ValueError(
            "provider_config must not include secrets; use runtime environment variables: "
            + ", ".join(unique)
        )
    return config


def _inspect_value(value: Any, *, path: str, findings: list[str]) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = _normalize_key(key)
            child_path = f"{path}.{key}"
            if normalized in _HEADER_CONTAINER_KEYS:
                if child:
                    findings.append(child_path)
                continue
            if _is_forbidden_key(key):
                if child not in (None, "", False):
                    findings.append(child_path)
                continue
            if isinstance(child, str):
                if any(pattern.search(child) for pattern in _SECRET_VALUE_PATTERNS):
                    findings.append(child_path)
                    continue
                if any(term in normalized for term in _URL_KEY_TERMS):
                    _inspect_url(child, path=child_path, findings=findings)
            _inspect_value(child, path=child_path, findings=findings)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _inspect_value(child, path=f"{path}[{index}]", findings=findings)


def _inspect_url(value: str, *, path: str, findings: list[str]) -> None:
    parsed = urllib.parse.urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        findings.append(path)
        return
    for key, query_value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        if query_value and _is_forbidden_key(key):
            findings.append(f"{path}?{key}")


def _is_forbidden_key(value: str) -> bool:
    normalized = _normalize_key(value)
    if any(term in normalized for term in _FORBIDDEN_KEY_TERMS):
        return True
    lower = str(value or "").lower()
    return bool(
        normalized.endswith("token")
        or re.search(r"(?:^|[^a-z0-9])token(?:$|[^a-z0-9])", lower)
    )


def _normalize_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())
