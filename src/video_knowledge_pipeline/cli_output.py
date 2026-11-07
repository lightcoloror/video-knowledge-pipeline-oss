from __future__ import annotations

import json
from typing import Any


DEFAULT_INLINE_RESULT_LIMIT = 16_000


def render_cli_result(
    result: Any,
    *,
    verbose: bool = False,
    inline_limit: int = DEFAULT_INLINE_RESULT_LIMIT,
) -> str:
    """Render small results in full and large persisted results as an envelope."""
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if verbose or len(payload) <= max(1, int(inline_limit)):
        return payload
    concise = _concise_result(result)
    concise["stdout_policy"] = {
        "mode": "concise",
        "full_result_omitted": True,
        "original_character_count": len(payload),
        "hint": "完整 JSON 已由命令写入列明产物；需要终端展开时追加 --verbose。",
    }
    return json.dumps(concise, ensure_ascii=False, indent=2)


def _concise_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"status": "completed", "result_type": type(result).__name__}
    concise: dict[str, Any] = {}
    for key in (
        "schema",
        "ok",
        "status",
        "bundle_dir",
        "output_dir",
        "command",
        "write",
        "run_id",
        "semantic_acceptance",
        "model_adjudication",
        "review_only",
    ):
        if key in result and _is_scalar(result[key]):
            concise[key] = result[key]
    summary = result.get("summary")
    if isinstance(result.get("version_binding"), dict):
        concise["version_binding"] = _compact_mapping(result["version_binding"])
    if isinstance(summary, dict):
        concise["summary"] = _compact_mapping(summary)
    paths: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for key, value in result.items():
        lower = str(key).lower()
        if _path_key(lower) and _compact_path_value(value) is not None:
            paths[key] = _compact_path_value(value)
        elif _count_key(lower) and _is_scalar(value):
            counts[key] = value
        elif (
            "sha256" in lower
            or lower.endswith("_hash")
            or lower.endswith("_revision")
        ) and _is_scalar(value):
            identities[key] = value
    if paths:
        concise["paths"] = paths
    if counts:
        concise["counts"] = counts
    if identities:
        concise["identities"] = identities
    return concise


def _compact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if _is_scalar(item):
            result[str(key)] = item
        elif isinstance(item, dict):
            nested = {
                str(nested_key): nested_value
                for nested_key, nested_value in item.items()
                if _is_scalar(nested_value)
            }
            if nested:
                result[str(key)] = nested
        elif isinstance(item, list):
            result[f"{key}_count"] = len(item)
    return result


def _compact_path_value(value: Any) -> Any | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list):
        paths = [str(item) for item in value if isinstance(item, str) and item]
        return paths[:30] if paths else None
    return None


def _path_key(key: str) -> bool:
    return key.endswith(("_path", "_paths", "_json", "_markdown")) or key in {
        "report_path",
        "manifest_path",
        "timeline_path",
        "receipt_path",
        "consent_path",
    }


def _count_key(key: str) -> bool:
    return key.endswith(("_count", "_total", "_updated", "_succeeded", "_failed"))


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))
