from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .file_hash import sha256_file as _sha256
from .storage import read_json


SCHEMA = "video_knowledge_pipeline.model_output_contract_result.v1"
CONTRACT_SCHEMA = "video_knowledge_pipeline.model_output_contract.v1"
_ALLOWED_FORMATS = {"any", "json", "text", "ocr_pages"}
_ALLOWED_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def normalise_output_contract(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        value = {}
    if not isinstance(value, dict):
        raise ValueError("output_contract must be an object")
    expected_format = str(value.get("format") or value.get("expected_format") or "any").strip().lower()
    if expected_format not in _ALLOWED_FORMATS:
        raise ValueError(f"unsupported output contract format: {expected_format}")

    required_keys: dict[str, str] = {}
    raw_required = value.get("required_keys") or {}
    if isinstance(raw_required, list):
        raw_required = {str(key): "any" for key in raw_required}
    if not isinstance(raw_required, dict):
        raise ValueError("output_contract.required_keys must be an object or list")
    for raw_key, raw_type in raw_required.items():
        key = _safe_key(raw_key, field="required key")
        value_type = str(raw_type or "any").strip().lower()
        if value_type != "any" and value_type not in _ALLOWED_TYPES:
            raise ValueError(f"unsupported required key type: {value_type}")
        required_keys[key] = value_type

    aliases: dict[str, str] = {}
    raw_aliases = value.get("aliases") or {}
    if not isinstance(raw_aliases, dict):
        raise ValueError("output_contract.aliases must be an object")
    for raw_alias, raw_canonical in raw_aliases.items():
        alias = _safe_key(raw_alias, field="alias")
        canonical = _safe_key(raw_canonical, field="canonical key")
        if alias == canonical:
            continue
        aliases[alias] = canonical

    return {
        "schema": CONTRACT_SCHEMA,
        "format": expected_format,
        "target": _normalise_target(value.get("target")),
        "required_keys": required_keys,
        "nonempty_keys": _string_list(value.get("nonempty_keys"), field="nonempty_keys"),
        "aliases": aliases,
        "additional_keys_allowed": bool(value.get("additional_keys_allowed", True)),
        "required_term_groups": _term_groups(value.get("required_term_groups")),
        "required_all_terms": _string_list(value.get("required_all_terms"), field="required_all_terms"),
        "forbidden_markers": _string_list(value.get("forbidden_markers"), field="forbidden_markers"),
        "correction_policy": _normalise_correction_policy(value.get("correction_policy")),
        "array_item_contracts": _normalise_array_item_contracts(value.get("array_item_contracts")),
    }


def validate_model_output(
    content: Any,
    output_contract: dict[str, Any] | None = None,
    *,
    transport_ok: bool = True,
) -> dict[str, Any]:
    contract = normalise_output_contract(output_contract)
    targets = _contract_targets(content, contract["target"])
    evaluations = [_evaluate_target(target, contract) for target in targets]
    contract_issues = [
        issue for evaluation in evaluations for issue in evaluation["contract_issues"]
    ]
    if not evaluations:
        contract_issues.append(
            _issue("contract_target_missing", f"no {contract['target']} target found")
        )
    quality_issues = [
        issue for evaluation in evaluations for issue in evaluation["quality_issues"]
    ]
    applied_aliases = [
        row for evaluation in evaluations for row in evaluation["applied_aliases"]
    ]
    contract_ok = bool(transport_ok) and bool(evaluations) and not contract_issues
    quality_gate_passed = contract_ok and not quality_issues
    if not transport_ok:
        status = "transport_failed"
    elif not contract_ok:
        status = "contract_failed"
    elif not quality_gate_passed:
        status = "quality_gate_failed"
    else:
        status = "qualified"
    return {
        "schema": SCHEMA,
        "status": status,
        "transport_ok": bool(transport_ok),
        "contract_ok": contract_ok,
        "quality_gate_passed": quality_gate_passed,
        "target_count": len(targets),
        "contract": contract,
        "applied_aliases": applied_aliases,
        "contract_issues": contract_issues,
        "quality_issues": quality_issues,
        "content_sha256": hashlib.sha256(_content_text(content).encode("utf-8")).hexdigest(),
        "content_persisted": False,
    }


def validate_execution_report(
    report_path: str | Path,
    output_contract: dict[str, Any],
) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    report = read_json(path)
    if not isinstance(report, dict):
        raise ValueError("execution report must be an object")
    model_result = report.get("model_result") if isinstance(report.get("model_result"), dict) else report
    runtime = _runtime_result(model_result)
    content = model_result.get("content", runtime.get("content"))
    result = validate_model_output(
        content,
        output_contract,
        transport_ok=bool(report.get("ok", runtime.get("ok"))),
    )
    return {
        **result,
        "execution_report_path": str(path),
        "execution_report_sha256": _sha256(path),
    }


def _evaluate_target(content: Any, contract: dict[str, Any]) -> dict[str, Any]:
    parsed, format_ok = _parse_content(content, contract["format"])
    contract_issues: list[dict[str, str]] = []
    quality_issues: list[dict[str, str]] = []
    applied_aliases: list[dict[str, str]] = []
    if not format_ok:
        contract_issues.append(_issue("format_mismatch", f"expected {contract['format']} output"))
        return {
            "contract_issues": contract_issues,
            "quality_issues": quality_issues,
            "applied_aliases": applied_aliases,
        }

    if isinstance(parsed, dict):
        parsed = dict(parsed)
        for alias, canonical in contract["aliases"].items():
            if alias not in parsed:
                continue
            if canonical in parsed and parsed[canonical] != parsed[alias]:
                contract_issues.append(
                    _issue("alias_collision", f"{alias} conflicts with {canonical}")
                )
                continue
            if canonical not in parsed:
                parsed[canonical] = parsed[alias]
            parsed.pop(alias, None)
            applied_aliases.append({"alias": alias, "canonical": canonical})
        for key, value_type in contract["required_keys"].items():
            if key not in parsed:
                contract_issues.append(_issue("missing_required_key", key))
            elif value_type != "any" and not _matches_type(parsed[key], value_type):
                contract_issues.append(
                    _issue("required_key_type_mismatch", f"{key}: expected {value_type}")
                )
        for key in contract["nonempty_keys"]:
            if key not in parsed or not _nonempty(parsed.get(key)):
                quality_issues.append(_issue("required_value_empty", key))
        if not contract["additional_keys_allowed"] and contract["required_keys"]:
            allowed = set(contract["required_keys"])
            extras = sorted(set(parsed) - allowed)
            for key in extras:
                contract_issues.append(_issue("unexpected_key", key))
    elif contract["required_keys"]:
        contract_issues.append(_issue("object_required", "required_keys needs a JSON object"))

    if isinstance(parsed, dict):
        contract_issues.extend(
            _array_item_contract_issues(parsed, contract["array_item_contracts"])
        )

    text = _content_text(parsed)
    folded = text.casefold()
    for index, alternatives in enumerate(contract["required_term_groups"]):
        if not any(term.casefold() in folded for term in alternatives):
            quality_issues.append(
                _issue("required_term_group_missing", f"group {index + 1}")
            )
    for term in contract["required_all_terms"]:
        if term.casefold() not in folded:
            quality_issues.append(_issue("required_term_missing", term))
    for marker in contract["forbidden_markers"]:
        if marker.casefold() in folded:
            quality_issues.append(_issue("forbidden_marker_present", marker))
    quality_issues.extend(_correction_issues(parsed, contract["correction_policy"]))
    return {
        "contract_issues": contract_issues,
        "quality_issues": quality_issues,
        "applied_aliases": applied_aliases,
    }


def _correction_issues(content: Any, policy: dict[str, Any]) -> list[dict[str, str]]:
    if not policy:
        return []
    if not isinstance(content, dict) or not isinstance(content.get("decisions"), list):
        return [_issue("correction_decisions_missing", "decisions must be an array")]
    actual: list[tuple[str, str]] = []
    for row in content["decisions"]:
        if not isinstance(row, dict):
            return [_issue("correction_decision_invalid", "each decision must be an object")]
        actual.append((str(row.get("source") or ""), str(row.get("replacement") or "")))
    required = {
        (str(row["source"]), str(row["replacement"]))
        for row in policy.get("required_replacements") or []
    }
    actual_set = set(actual)
    issues: list[dict[str, str]] = []
    for source, replacement in sorted(required - actual_set):
        issues.append(_issue("required_correction_missing", f"{source} -> {replacement}"))
    if not policy.get("allow_additional", False):
        for source, replacement in sorted(actual_set - required):
            issues.append(_issue("unlisted_correction", f"{source} -> {replacement}"))
    if len(actual) != len(actual_set):
        issues.append(_issue("duplicate_correction", "duplicate source/replacement pair"))
    return issues


def _parse_content(content: Any, expected_format: str) -> tuple[Any, bool]:
    if expected_format == "any":
        return content, True
    if expected_format == "text":
        return content, isinstance(content, str) and bool(content.strip())
    if expected_format == "ocr_pages":
        return (
            content,
            isinstance(content, dict)
            and isinstance(content.get("pages"), list)
            and bool(content["pages"]),
        )
    if isinstance(content, (dict, list)):
        return content, True
    text = str(content or "").strip()
    if text.startswith(">>>"):
        return content, False
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text), True
    except (TypeError, ValueError, json.JSONDecodeError):
        return content, False


def _contract_targets(content: Any, target: str) -> list[Any]:
    if target == "content":
        return [content]
    if target == "temporal_each_group":
        groups = content.get("groups") if isinstance(content, dict) else None
        if not isinstance(groups, list) or not groups:
            return []
        return [
            row.get("content")
            for row in groups
            if isinstance(row, dict)
        ]
    raise ValueError(f"unsupported output contract target: {target}")


def _runtime_result(model_result: dict[str, Any]) -> dict[str, Any]:
    if model_result.get("schema") == "video_knowledge_pipeline.model_runtime_result.v1":
        return model_result
    nested = model_result.get("runtime_result")
    if isinstance(nested, dict):
        return nested
    calls = model_result.get("calls") if isinstance(model_result.get("calls"), list) else []
    for call in calls:
        if isinstance(call, dict) and isinstance(call.get("runtime_result"), dict):
            return call["runtime_result"]
    return model_result


def _normalise_correction_policy(value: Any) -> dict[str, Any]:
    if value in (None, "") or value == {}:
        return {}
    if not isinstance(value, dict):
        raise ValueError("correction_policy must be an object")
    rows = []
    for row in value.get("required_replacements") or []:
        if not isinstance(row, dict):
            raise ValueError("required_replacements entries must be objects")
        source = str(row.get("source") or "")
        replacement = str(row.get("replacement") or "")
        if not source or not replacement:
            raise ValueError("required correction source and replacement are required")
        rows.append({"source": source, "replacement": replacement})
    return {
        "required_replacements": rows,
        "allow_additional": bool(value.get("allow_additional", False)),
    }


def _normalise_array_item_contracts(value: Any) -> dict[str, dict[str, Any]]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("output_contract.array_item_contracts must be an object")
    contracts: dict[str, dict[str, Any]] = {}
    for raw_array_key, raw_contract in value.items():
        array_key = _safe_key(raw_array_key, field="array item contract key")
        if not isinstance(raw_contract, dict):
            raise ValueError(
                f"output_contract.array_item_contracts.{array_key} must be an object"
            )
        raw_required = raw_contract.get("required_keys") or {}
        if isinstance(raw_required, list):
            raw_required = {str(key): "any" for key in raw_required}
        if not isinstance(raw_required, dict):
            raise ValueError(
                f"output_contract.array_item_contracts.{array_key}.required_keys "
                "must be an object or list"
            )
        required_keys: dict[str, str] = {}
        for raw_key, raw_type in raw_required.items():
            key = _safe_key(raw_key, field="array item required key")
            value_type = str(raw_type or "any").strip().lower()
            if value_type != "any" and value_type not in _ALLOWED_TYPES:
                raise ValueError(f"unsupported array item required key type: {value_type}")
            required_keys[key] = value_type
        contracts[array_key] = {
            "required_keys": required_keys,
            "nonempty_keys": _string_list(
                raw_contract.get("nonempty_keys"),
                field=f"array_item_contracts.{array_key}.nonempty_keys",
            ),
            "additional_keys_allowed": bool(
                raw_contract.get("additional_keys_allowed", True)
            ),
        }
    return contracts


def _array_item_contract_issues(
    content: dict[str, Any], contracts: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for array_key, item_contract in contracts.items():
        rows = content.get(array_key)
        if not isinstance(rows, list):
            issues.append(
                _issue("array_item_contract_target_invalid", f"{array_key} must be an array")
            )
            continue
        for index, row in enumerate(rows):
            prefix = f"{array_key}[{index}]"
            if not isinstance(row, dict):
                issues.append(
                    _issue("array_item_not_object", f"{prefix} must be an object")
                )
                continue
            required_keys = item_contract["required_keys"]
            for key, value_type in required_keys.items():
                detail = f"{prefix}.{key}"
                if key not in row:
                    issues.append(_issue("array_item_missing_required_key", detail))
                elif value_type != "any" and not _matches_type(row[key], value_type):
                    issues.append(
                        _issue(
                            "array_item_required_key_type_mismatch",
                            f"{detail}: expected {value_type}",
                        )
                    )
            for key in item_contract["nonempty_keys"]:
                if key in row and not _nonempty(row.get(key)):
                    issues.append(
                        _issue("array_item_required_value_empty", f"{prefix}.{key}")
                    )
            if not item_contract["additional_keys_allowed"] and required_keys:
                allowed = set(required_keys)
                for key in sorted(set(row) - allowed):
                    issues.append(_issue("array_item_unexpected_key", f"{prefix}.{key}"))
    return issues


def _normalise_target(value: Any) -> str:
    target = str(value or "content").strip().lower()
    if target not in {"content", "temporal_each_group"}:
        raise ValueError("output_contract.target must be content or temporal_each_group")
    return target


def _term_groups(value: Any) -> list[list[str]]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("required_term_groups must be a list")
    rows = []
    for raw in value:
        alternatives = raw if isinstance(raw, list) else [raw]
        clean = [str(item) for item in alternatives if str(item)]
        if not clean:
            raise ValueError("required_term_groups cannot contain empty groups")
        rows.append(clean)
    return rows


def _string_list(value: Any, *, field: str) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [str(item) for item in value if str(item)]


def _safe_key(value: Any, *, field: str) -> str:
    key = str(value or "").strip()
    if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", key):
        raise ValueError(f"invalid {field}: {value!r}")
    return key


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, dict, tuple, set)):
        return bool(value)
    return True


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _issue(key: str, detail: str) -> dict[str, str]:
    return {"key": key, "detail": detail}


def _load_contract(value: str) -> dict[str, Any]:
    path = Path(value).expanduser()
    data = read_json(path) if path.is_file() else json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("output contract must be an object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a saved VKP model execution report")
    parser.add_argument("execution_report")
    parser.add_argument("--contract", required=True, help="JSON object or JSON file path")
    args = parser.parse_args(argv)
    result = validate_execution_report(args.execution_report, _load_contract(args.contract))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["quality_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
