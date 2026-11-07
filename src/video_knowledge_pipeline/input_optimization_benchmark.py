from __future__ import annotations

# Offline comparator for network and token input optimization.

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .storage import read_json, write_json
from .transcript_stability_evaluation import evaluate_transcript_stability


SCHEMA = "video_knowledge_pipeline.input_optimization_benchmark.v1"


def compare_asr_input_optimization(
    *,
    baseline_audio_path: str | Path,
    optimized_audio_path: str | Path,
    baseline_transcript_path: str | Path,
    optimized_transcript_path: str | Path,
    baseline_execution_report_path: str | Path,
    optimized_execution_report_path: str | Path,
    critical_terms: Sequence[str] = (),
    max_content_distance: float = 0.05,
) -> dict[str, Any]:
    """Compare an ASR upload optimization without promoting either transcript."""
    baseline_audio = Path(baseline_audio_path).resolve()
    optimized_audio = Path(optimized_audio_path).resolve()
    baseline_transcript = read_json(Path(baseline_transcript_path).resolve())
    optimized_transcript = read_json(Path(optimized_transcript_path).resolve())
    baseline_report = read_json(Path(baseline_execution_report_path).resolve())
    optimized_report = read_json(Path(optimized_execution_report_path).resolve())
    strict = evaluate_transcript_stability(
        optimized_transcript,
        baseline_transcript,
        max_normalized_reference_edit_distance=max_content_distance,
        normalization_profile="strict_v1",
    )
    content = evaluate_transcript_stability(
        optimized_transcript,
        baseline_transcript,
        max_normalized_reference_edit_distance=max_content_distance,
        normalization_profile="content_vocal_fillers_v1",
    )
    baseline_identity = _route_identity(baseline_report)
    optimized_identity = _route_identity(optimized_report)
    baseline_quality = _asr_quality(baseline_report)
    optimized_quality = _asr_quality(optimized_report)
    optimized_text = _transcript_text(optimized_transcript)
    critical_term_results = {
        term: term.casefold() in optimized_text.casefold()
        for term in critical_terms
        if str(term).strip()
    }
    baseline_bytes = _uploaded_bytes(baseline_report) or baseline_audio.stat().st_size
    optimized_bytes = (
        _uploaded_bytes(optimized_report) or optimized_audio.stat().st_size
    )
    gates = {
        "same_route_and_model": baseline_identity == optimized_identity,
        "both_executions_completed": _completed(baseline_report)
        and _completed(optimized_report),
        "content_stability_passed": content["status"] == "passed",
        "no_failed_segment_regression": _quality_count(
            optimized_quality, "failed_segment_count"
        )
        <= _quality_count(baseline_quality, "failed_segment_count"),
        "no_review_segment_regression": _quality_count(
            optimized_quality, "review_segment_count"
        )
        <= _quality_count(baseline_quality, "review_segment_count"),
        "critical_terms_preserved": all(critical_term_results.values()),
        "network_input_reduced": optimized_bytes < baseline_bytes,
    }
    return {
        "schema": SCHEMA,
        "benchmark": "asr_input_optimization",
        "status": "passed" if all(gates.values()) else "failed",
        "evaluation_only": True,
        "provider_calls_made": 0,
        "transcript_promoted": False,
        "route_identity": {
            "baseline": baseline_identity,
            "optimized": optimized_identity,
        },
        "input_bytes": _number_comparison(baseline_bytes, optimized_bytes),
        "latency_ms": {
            "baseline": _latency_ms(baseline_report),
            "optimized": _latency_ms(optimized_report),
        },
        "quality": {
            "strict_surface": strict,
            "content_without_vocal_fillers": content,
            "baseline_asr_quality": _quality_summary(baseline_quality),
            "optimized_asr_quality": _quality_summary(optimized_quality),
            "critical_terms": critical_term_results,
        },
        "gates": gates,
        "decision": "eligible_for_explicit_default_change"
        if all(gates.values())
        else "keep_current_default",
    }


def compare_semantic_input_optimization(
    *,
    optimized_pack_path: str | Path,
    baseline_execution_report_path: str | Path,
    optimized_execution_report_path: str | Path,
) -> dict[str, Any]:
    """Compare compact semantic input with the historical full-pack execution."""
    pack_path = Path(optimized_pack_path).resolve()
    pack = read_json(pack_path)
    baseline_report = read_json(Path(baseline_execution_report_path).resolve())
    optimized_report = read_json(Path(optimized_execution_report_path).resolve())
    candidates = {
        str(row.get("candidate_id")): row
        for row in pack.get("candidates") or []
        if isinstance(row, dict) and row.get("candidate_id")
    }
    decisions = {
        str(row.get("candidate_id")): row
        for row in _semantic_decisions(optimized_report)
        if row.get("candidate_id")
    }
    missing = sorted(set(candidates) - set(decisions))
    extra = sorted(set(decisions) - set(candidates))
    original_text_mismatches: list[str] = []
    unknown_evidence: list[str] = []
    for candidate_id, candidate in candidates.items():
        decision = decisions.get(candidate_id)
        if decision is None:
            continue
        if str(decision.get("original_text") or "") != str(
            candidate.get("original_text") or ""
        ):
            original_text_mismatches.append(candidate_id)
        allowed_evidence = {str(value) for value in candidate.get("evidence_ids") or []}
        used_evidence = {str(value) for value in decision.get("evidence_ids") or []}
        if not used_evidence.issubset(allowed_evidence):
            unknown_evidence.append(candidate_id)
    baseline_identity = _route_identity(baseline_report)
    optimized_identity = _route_identity(optimized_report)
    baseline_bytes = _uploaded_bytes(baseline_report)
    optimized_bytes = _uploaded_bytes(optimized_report) or pack_path.stat().st_size
    baseline_tokens = _prompt_tokens(baseline_report)
    optimized_tokens = _prompt_tokens(optimized_report)
    gates = {
        "same_route_and_model": baseline_identity == optimized_identity,
        "optimized_execution_completed": _completed(optimized_report),
        "output_contract_passed": bool(optimized_report.get("contract_ok")),
        "task_quality_gate_passed": bool(optimized_report.get("quality_gate_passed")),
        "all_selected_candidates_returned": not missing,
        "no_unrequested_candidates_returned": not extra,
        "original_text_preserved": not original_text_mismatches,
        "evidence_ids_bound_to_pack": not unknown_evidence,
        "network_input_reduced": optimized_bytes < baseline_bytes,
        "prompt_tokens_reduced": optimized_tokens is not None
        and baseline_tokens is not None
        and optimized_tokens < baseline_tokens,
    }
    return {
        "schema": SCHEMA,
        "benchmark": "semantic_input_optimization",
        "status": "passed" if all(gates.values()) else "failed",
        "evaluation_only": True,
        "provider_calls_made": 0,
        "corrections_applied": False,
        "route_identity": {
            "baseline": baseline_identity,
            "optimized": optimized_identity,
        },
        "candidate_accounting": {
            "selected": len(candidates),
            "returned": len(decisions),
            "missing_candidate_ids": missing,
            "unexpected_candidate_ids": extra,
            "original_text_mismatch_ids": original_text_mismatches,
            "unknown_evidence_ids_for_candidates": unknown_evidence,
            "deferred_low_evidence": int(
                (pack.get("candidate_selection") or {}).get("deferred_candidate_count")
                or 0
            ),
        },
        "input_bytes": _number_comparison(baseline_bytes, optimized_bytes),
        "prompt_tokens": _number_comparison(baseline_tokens, optimized_tokens),
        "latency_ms": {
            "baseline": _latency_ms(baseline_report),
            "optimized": _latency_ms(optimized_report),
        },
        "gates": gates,
        "decision": "eligible_for_compact_gateway_default"
        if all(gates.values())
        else "keep_compact_pack_as_candidate",
    }


def combine_input_optimization_benchmarks(
    asr_result: dict[str, Any], semantic_result: dict[str, Any]
) -> dict[str, Any]:
    passed = asr_result.get("status") == semantic_result.get("status") == "passed"
    return {
        "schema": SCHEMA,
        "benchmark": "network_and_token_optimization",
        "status": "passed" if passed else "failed",
        "evaluation_only": True,
        "provider_calls_made": 0,
        "results": {"asr": asr_result, "semantic": semantic_result},
        "production_recommendation": "explicitly_review_and_change_defaults"
        if passed
        else "keep_existing_defaults",
    }


def _route_identity(report: dict[str, Any]) -> dict[str, Any]:
    route = report.get("route") or {}
    deployments = route.get("deployments") or []
    first = deployments[0] if deployments and isinstance(deployments[0], dict) else {}
    return {
        "route_id": route.get("route_id"),
        "route_revision": route.get("route_revision"),
        "provider": first.get("provider"),
        "model": first.get("model"),
    }


def _completed(report: dict[str, Any]) -> bool:
    runtime = (report.get("model_result") or {}).get("runtime_result") or {}
    return bool(report.get("ok")) and runtime.get("status") == "completed"


def _uploaded_bytes(report: dict[str, Any]) -> int:
    return int((report.get("upload_manifest") or {}).get("total_bytes") or 0)


def _latency_ms(report: dict[str, Any]) -> int | None:
    runtime = (report.get("model_result") or {}).get("runtime_result") or {}
    value = runtime.get("latency_ms")
    return int(value) if value is not None else None


def _prompt_tokens(report: dict[str, Any]) -> int | None:
    runtime = (report.get("model_result") or {}).get("runtime_result") or {}
    value = (runtime.get("usage") or {}).get("prompt_tokens")
    return int(value) if value is not None else None


def _asr_quality(report: dict[str, Any]) -> dict[str, Any]:
    return (report.get("model_result") or {}).get("asr_quality") or {}


def _quality_count(quality: dict[str, Any], key: str) -> int:
    return int(quality.get(key) or 0)


def _quality_summary(quality: dict[str, Any]) -> dict[str, Any]:
    return {
        key: quality.get(key)
        for key in (
            "status",
            "segment_count",
            "passed_segment_count",
            "review_segment_count",
            "failed_segment_count",
        )
    }


def _transcript_text(payload: dict[str, Any]) -> str:
    rows = payload.get("segments") or payload.get("cues") or []
    if rows:
        return "\n".join(
            str(row.get("text") or "") for row in rows if isinstance(row, dict)
        )
    return str(payload.get("text") or payload.get("content") or "")


def _semantic_decisions(report: dict[str, Any]) -> list[dict[str, Any]]:
    content = str((report.get("model_result") or {}).get("content") or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    payload = json.loads(content)
    return [row for row in payload.get("decisions") or [] if isinstance(row, dict)]


def _number_comparison(baseline: int | None, optimized: int | None) -> dict[str, Any]:
    if baseline is None or optimized is None:
        return {"baseline": baseline, "optimized": optimized, "reduction": None}
    reduction = max(0, baseline - optimized)
    return {
        "baseline": baseline,
        "optimized": optimized,
        "reduction": reduction,
        "reduction_ratio": round(reduction / baseline, 6) if baseline else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline input-optimization benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    asr = subparsers.add_parser("asr")
    asr.add_argument("baseline_audio")
    asr.add_argument("optimized_audio")
    asr.add_argument("baseline_transcript")
    asr.add_argument("optimized_transcript")
    asr.add_argument("baseline_report")
    asr.add_argument("optimized_report")
    asr.add_argument("output_json")
    asr.add_argument("--critical-term", action="append", default=[])
    semantic = subparsers.add_parser("semantic")
    semantic.add_argument("optimized_pack")
    semantic.add_argument("baseline_report")
    semantic.add_argument("optimized_report")
    semantic.add_argument("output_json")
    final = subparsers.add_parser("final")
    final.add_argument("asr_report")
    final.add_argument("semantic_report")
    final.add_argument("output_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "asr":
        result = compare_asr_input_optimization(
            baseline_audio_path=args.baseline_audio,
            optimized_audio_path=args.optimized_audio,
            baseline_transcript_path=args.baseline_transcript,
            optimized_transcript_path=args.optimized_transcript,
            baseline_execution_report_path=args.baseline_report,
            optimized_execution_report_path=args.optimized_report,
            critical_terms=args.critical_term,
        )
    elif args.command == "semantic":
        result = compare_semantic_input_optimization(
            optimized_pack_path=args.optimized_pack,
            baseline_execution_report_path=args.baseline_report,
            optimized_execution_report_path=args.optimized_report,
        )
    else:
        result = combine_input_optimization_benchmarks(
            read_json(Path(args.asr_report).resolve()),
            read_json(Path(args.semantic_report).resolve()),
        )
    destination = Path(args.output_json).resolve()
    write_json(destination, result)
    print(json.dumps({**result, "report_path": str(destination)}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
