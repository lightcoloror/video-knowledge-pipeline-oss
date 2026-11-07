from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .file_hash import sha256_file as _sha256
from .model_output_contracts import validate_model_output
from .models import now_iso
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.model_candidate_benchmark.v1"
MANIFEST_SCHEMA = "video_knowledge_pipeline.model_candidate_benchmark_manifest.v1"
RUNTIME_SCHEMA = "video_knowledge_pipeline.model_runtime_result.v1"


def compare_model_candidates(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    source = Path(manifest_path).expanduser().resolve()
    manifest = read_json(source)
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("invalid model candidate benchmark manifest")
    cases = []
    for raw_case in manifest.get("cases") or []:
        case = dict(raw_case)
        rows = [_candidate_row(case, dict(row)) for row in case.get("candidates") or []]
        loaded_rows = [row for row in rows if row["loaded"]]
        artifact_hashes = {
            row["artifact_manifest_sha256"]
            for row in loaded_rows
            if row["artifact_manifest_sha256"]
        }
        instruction_hashes = {
            row["instructions_sha256"]
            for row in loaded_rows
            if row["instructions_sha256"]
        }
        same_artifacts = (
            bool(loaded_rows)
            and all(row["artifact_manifest_sha256"] for row in loaded_rows)
            and len(artifact_hashes) == 1
        )
        same_instructions = (
            bool(loaded_rows)
            and all(row["instructions_sha256"] for row in loaded_rows)
            and len(instruction_hashes) == 1
        )
        comparable = (
            len(rows) >= 2
            and all(row["loaded"] for row in rows)
            and same_artifacts
            and same_instructions
        )
        ranked = sorted(
            (row for row in rows if row["quality_gate_passed"]),
            key=lambda row: (-float(row["automatic_proxy_score"]), _latency_sort(row["latency_ms"])),
        )
        cases.append(
            {
                "id": str(case.get("id") or ""),
                "task": str(case.get("task") or ""),
                "model_type": str(case.get("model_type") or ""),
                "sample_id": str(case.get("sample_id") or ""),
                "status": "ready_for_review" if comparable else "incomplete",
                "comparable": comparable,
                "same_artifact_manifest": same_artifacts,
                "same_instructions": same_instructions,
                "candidate_count": len(rows),
                "successful_count": sum(bool(row["transport_ok"]) for row in rows),
                "failed_count": sum(not bool(row["transport_ok"]) for row in rows),
                "contract_passed_count": sum(bool(row["contract_ok"]) for row in rows),
                "quality_passed_count": sum(
                    bool(row["quality_gate_passed"]) for row in rows
                ),
                "automatic_proxy_winner": str(ranked[0]["candidate_id"]) if comparable and ranked else "",
                "candidates": rows,
                "limitations": _limitations(rows, comparable, same_artifacts, same_instructions),
            }
        )
    ready_count = sum(row["status"] == "ready_for_review" for row in cases)
    result = {
        "schema": SCHEMA,
        "status": "ready_for_review" if cases and ready_count == len(cases) else "incomplete",
        "manifest_path": str(source),
        "case_count": len(cases),
        "ready_case_count": ready_count,
        "incomplete_case_count": len(cases) - ready_count,
        "cases": cases,
        "operator_boundary": {
            "offline_comparison_only": True,
            "model_calls_made": 0,
            "source_artifacts_read": False,
            "model_content_copied_into_report": False,
            "automatic_score_is_not_human_quality_judgment": True,
            "no_default_route_promotion": True,
        },
        "updated_at": now_iso(),
    }
    destination = Path(output_dir).expanduser().resolve() if output_dir else source.parent
    if write:
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "model-candidate-benchmark.json"
        markdown_path = destination / "model-candidate-benchmark.md"
        write_json(json_path, result)
        markdown_path.write_text(_render_markdown(result), encoding="utf-8")
        result["artifacts"] = {"json": str(json_path), "markdown": str(markdown_path)}
    return result


def _candidate_row(case: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(candidate.get("result_path") or "")).expanduser().resolve()
    candidate_id = str(candidate.get("id") or path.parent.name)
    try:
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise ValueError("execution report is not a JSON object")
        model_result = raw.get("model_result") if isinstance(raw.get("model_result"), dict) else raw
        runtime = _runtime_result(model_result)
        route = raw.get("route") if isinstance(raw.get("route"), dict) else {}
        route_deployments = route.get("deployments") if isinstance(route.get("deployments"), list) else []
        route_deployment = dict(route_deployments[0]) if route_deployments and isinstance(route_deployments[0], dict) else {}
        deployment = runtime.get("deployment") if isinstance(runtime.get("deployment"), dict) else route_deployment
        content = model_result.get("content", runtime.get("content"))
        text = _content_text(content)
        consent = _consent(raw.get("consent_path"))
        reference_similarity = _similarity(text, str(case.get("reference_text") or ""))
        term_coverage = _term_coverage(text, case.get("required_term_groups") or [])
        format_ok = _format_ok(content, str(case.get("expected_format") or "any"))
        forbidden = [str(value) for value in case.get("forbidden_markers") or []]
        forbidden_hits = [value for value in forbidden if value.casefold() in text.casefold()]
        transport_ok = bool(raw.get("ok", runtime.get("ok")))
        contract_result = validate_model_output(
            content,
            _case_output_contract(case),
            transport_ok=transport_ok,
        )
        score = _proxy_score(
            ok=bool(contract_result["quality_gate_passed"]),
            reference_text=str(case.get("reference_text") or ""),
            reference_similarity=reference_similarity,
            required_groups=case.get("required_term_groups") or [],
            term_coverage=term_coverage,
            format_ok=format_ok,
            clean=not forbidden_hits,
        )
        usage = runtime.get("usage") if isinstance(runtime.get("usage"), dict) else {}
        top_usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return {
            "candidate_id": candidate_id,
            "result_path": str(path),
            "result_sha256": _sha256(path),
            "loaded": True,
            "ok": transport_ok,
            "transport_ok": transport_ok,
            "contract_ok": bool(contract_result["contract_ok"]),
            "quality_gate_passed": bool(contract_result["quality_gate_passed"]),
            "outcome_status": str(contract_result["status"]),
            "applied_aliases": list(contract_result["applied_aliases"]),
            "contract_issues": list(contract_result["contract_issues"]),
            "quality_issues": list(contract_result["quality_issues"]),
            "status": str(raw.get("status") or runtime.get("status") or ""),
            "provider": str(runtime.get("provider") or deployment.get("provider") or ""),
            "deployment_id": str(deployment.get("id") or ""),
            "model": str(deployment.get("model") or ""),
            "provider_response_model": str((runtime.get("response") or {}).get("model") or ""),
            "route_id": str(runtime.get("route_id") or route.get("route_id") or ""),
            "route_revision": str(runtime.get("route_revision") or route.get("route_revision") or ""),
            "artifact_manifest_sha256": str((raw.get("upload_manifest") or {}).get("manifest_sha256") or ""),
            "instructions_sha256": str(consent.get("instructions_sha256") or ""),
            "latency_ms": _number_or_none(runtime.get("latency_ms")),
            "usage": usage,
            "estimated_cost": runtime.get("estimated_cost"),
            "cost_reported_usd": top_usage.get("cost_reported_usd"),
            "cost_unreported": bool(top_usage.get("cost_unreported_calls")),
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "content_chars": len(text),
            "reference_similarity": reference_similarity,
            "required_term_coverage": term_coverage,
            "format_ok": format_ok,
            "forbidden_marker_hits": forbidden_hits,
            "automatic_proxy_score": score,
            "error": str(model_result.get("error") or runtime.get("error") or "")[:300],
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "candidate_id": candidate_id,
            "result_path": str(path),
            "loaded": False,
            "ok": False,
            "transport_ok": False,
            "contract_ok": False,
            "quality_gate_passed": False,
            "outcome_status": "transport_failed",
            "applied_aliases": [],
            "contract_issues": [],
            "quality_issues": [],
            "status": "unavailable",
            "artifact_manifest_sha256": "",
            "instructions_sha256": "",
            "latency_ms": None,
            "automatic_proxy_score": 0.0,
            "error": str(exc),
        }


def _case_output_contract(case: dict[str, Any]) -> dict[str, Any]:
    raw = (
        dict(case["output_contract"])
        if isinstance(case.get("output_contract"), dict)
        else {}
    )
    raw.setdefault("format", str(case.get("expected_format") or "any"))
    raw.setdefault("required_term_groups", list(case.get("required_term_groups") or []))
    raw.setdefault("forbidden_markers", list(case.get("forbidden_markers") or []))
    return raw


def _runtime_result(model_result: dict[str, Any]) -> dict[str, Any]:
    if model_result.get("schema") == RUNTIME_SCHEMA:
        return model_result
    nested = model_result.get("runtime_result")
    if isinstance(nested, dict):
        return nested
    calls = model_result.get("calls") if isinstance(model_result.get("calls"), list) else []
    for call in calls:
        if isinstance(call, dict) and isinstance(call.get("runtime_result"), dict):
            return dict(call["runtime_result"])
    return model_result


def _consent(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    path = Path(str(value)).expanduser()
    if not path.is_file():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) if value is not None else ""


def _normalised_text(value: str) -> str:
    return "".join(re.findall(r"[\w\u3400-\u9fff]+", str(value).casefold(), flags=re.UNICODE))


def _similarity(value: str, reference: str) -> float:
    if not reference:
        return 0.0
    return round(difflib.SequenceMatcher(None, _normalised_text(value), _normalised_text(reference)).ratio(), 4)


def _term_coverage(value: str, groups: Any) -> float:
    rows = list(groups) if isinstance(groups, list) else []
    if not rows:
        return 0.0
    haystack = value.casefold()
    matched = 0
    for raw in rows:
        alternatives = raw if isinstance(raw, list) else [raw]
        if any(str(item).casefold() in haystack for item in alternatives if str(item)):
            matched += 1
    return round(matched / len(rows), 4)


def _format_ok(content: Any, expected: str) -> bool:
    expected = expected.strip().lower()
    if expected in {"", "any"}:
        return True
    if expected == "text":
        return isinstance(content, str) and bool(content.strip())
    if expected == "ocr_pages":
        return isinstance(content, dict) and isinstance(content.get("pages"), list) and bool(content["pages"])
    if expected == "json":
        if isinstance(content, (dict, list)):
            return True
        text = str(content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            json.loads(text)
            return True
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
    raise ValueError(f"unsupported expected_format: {expected}")


def _proxy_score(*, ok: bool, reference_text: str, reference_similarity: float, required_groups: Any, term_coverage: float, format_ok: bool, clean: bool) -> float:
    if not ok:
        return 0.0
    score = 30.0
    score += 50.0 * (reference_similarity if reference_text else (term_coverage if required_groups else 1.0))
    score += 10.0 if format_ok else 0.0
    score += 10.0 if clean else 0.0
    return round(score, 2)


def _limitations(rows: list[dict[str, Any]], comparable: bool, same_artifacts: bool, same_instructions: bool) -> list[str]:
    values = []
    if len(rows) < 2:
        values.append("fewer_than_two_candidates")
    if not all(row["loaded"] for row in rows):
        values.append("missing_execution_report")
    if not same_artifacts:
        values.append("artifact_manifest_mismatch")
    if not same_instructions:
        values.append("instructions_mismatch")
    if comparable:
        values.append("automatic_proxy_score_requires_human_review")
    if any(row.get("transport_ok") and not row.get("contract_ok") for row in rows):
        values.append("transport_success_with_contract_failure")
    if any(row.get("contract_ok") and not row.get("quality_gate_passed") for row in rows):
        values.append("contract_success_with_quality_gate_failure")
    if any(row.get("cost_unreported") for row in rows):
        values.append("provider_cost_not_reported")
    return values


def _latency_sort(value: Any) -> float:
    number = _number_or_none(value)
    return number if number is not None else float("inf")


def _number_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None

def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# VKP Online Model Candidate Benchmark",
        "",
        f"- Status: `{result['status']}`",
        f"- Comparable cases: `{result['ready_case_count']}/{result['case_count']}`",
        "- Automatic proxy scores are screening aids, not human quality judgments.",
    ]
    for case in result["cases"]:
        lines.extend(["", f"## {case['id']}", "", f"- Task: `{case['task']}`", f"- Comparable: `{str(case['comparable']).lower()}`", f"- Proxy winner: `{case['automatic_proxy_winner'] or 'none'}`", "", "| Candidate | Model | Status | Score | Latency ms | Ref similarity | Term coverage | Format | Reasoning leak | Cost |", "|---|---|---|---:|---:|---:|---:|---|---|---|"])
        for row in case["candidates"]:
            hits = ", ".join(row.get("forbidden_marker_hits") or []) or "none"
            cost = row.get("estimated_cost") if row.get("estimated_cost") is not None else "unknown"
            lines.append(f"| `{row['candidate_id']}` | `{row.get('model', '')}` | `{row['status']}` | {row['automatic_proxy_score']} | {row.get('latency_ms') if row.get('latency_ms') is not None else 'unknown'} | {row.get('reference_similarity', 0)} | {row.get('required_term_coverage', 0)} | {'pass' if row.get('format_ok') else 'fail'} | {hits} | {cost} |")
        if case["limitations"]:
            lines.extend(["", "Limitations: " + ", ".join(f"`{value}`" for value in case["limitations"])])
    lines.extend(["", "This report reads saved execution evidence only. It does not call a model, upload data, or change task routes.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare saved VKP online model candidate results")
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    result = compare_model_candidates(args.manifest, output_dir=args.output_dir or None, write=not args.no_write)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready_for_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
