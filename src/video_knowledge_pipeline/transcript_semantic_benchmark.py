from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.transcript_semantic_benchmark.v1"
HIGH_RISK_TYPES = {
    "negation",
    "condition",
    "exception",
    "person_name",
    "organization",
    "date_time",
    "money",
    "percentage",
    "number",
    "action",
    "punctuation",
    "segment_boundary",
}


def evaluate_transcript_semantic_gold(
    gold_json: str | Path,
    candidate_pack_json: str | Path,
    validation_json: str | Path,
    *,
    full_body_text: str = "",
    smart_summary_text: str = "",
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a de-identified semantic-correction run against a fixed gold set."""

    gold_path = Path(gold_json).expanduser().resolve()
    pack_path = Path(candidate_pack_json).expanduser().resolve()
    validation_path = Path(validation_json).expanduser().resolve()
    gold = _object(read_json(gold_path), "gold set")
    pack = _object(read_json(pack_path), "candidate pack")
    validation = _object(read_json(validation_path), "validation")
    cases = [row for row in gold.get("cases") or [] if isinstance(row, dict)]
    candidates = [row for row in pack.get("candidates") or [] if isinstance(row, dict)]
    accepted = [
        row
        for row in validation.get("accepted_decisions") or []
        if isinstance(row, dict)
    ]
    candidate_by_id = {
        str(row.get("candidate_id") or ""): row
        for row in candidates
        if str(row.get("candidate_id") or "")
    }

    expected_cases = [
        row
        for row in cases
        if bool(row.get("expect_candidate", True))
        and str(row.get("projection_disposition") or "retained_semantic")
        == "retained_semantic"
    ]
    matched_case_ids: set[str] = set()
    matched_candidate_ids: set[str] = set()
    case_by_candidate_id: dict[str, dict[str, Any]] = {}
    for case in expected_cases:
        match = next(
            (
                candidate
                for candidate in candidates
                if _candidate_matches_case(candidate, case)
            ),
            None,
        )
        if match is None:
            continue
        case_id = str(case.get("case_id") or "")
        candidate_id = str(match.get("candidate_id") or "")
        matched_case_ids.add(case_id)
        matched_candidate_ids.add(candidate_id)
        case_by_candidate_id[candidate_id] = case

    true_positive = len(matched_case_ids)
    false_positive = len(
        [
            row
            for row in candidates
            if str(row.get("candidate_id") or "") not in matched_candidate_ids
        ]
    )
    false_negative = max(len(expected_cases) - true_positive, 0)

    correct_auto = 0
    incorrect_auto = 0
    dangerous_auto = 0
    dangerous_incorrect = 0
    dangerous_type_counts: dict[str, dict[str, int]] = {}
    full_body_absorption_expected = 0
    full_body_absorption_passed = 0
    summary_freshness_expected = 0
    summary_freshness_passed = 0
    decision_rows: list[dict[str, Any]] = []
    for decision in accepted:
        candidate_id = str(decision.get("candidate_id") or "")
        candidate = candidate_by_id.get(candidate_id, {})
        case = case_by_candidate_id.get(candidate_id)
        correction_type = str(
            decision.get("correction_type")
            or candidate.get("correction_type")
            or ""
        )
        auto_applied = not bool(decision.get("human_confirmed"))
        correct = bool(
            case
            and str(decision.get("corrected_text") or "").strip()
            == str(case.get("corrected_text") or "").strip()
        )
        if correction_type in HIGH_RISK_TYPES:
            type_counts = dangerous_type_counts.setdefault(
                correction_type,
                {
                    "accepted_decisions": 0,
                    "auto_applied": 0,
                    "incorrect_auto": 0,
                },
            )
            type_counts["accepted_decisions"] += 1
        if auto_applied:
            if correct:
                correct_auto += 1
            else:
                incorrect_auto += 1
            if correction_type in HIGH_RISK_TYPES:
                dangerous_auto += 1
                dangerous_type_counts[correction_type]["auto_applied"] += 1
                if not correct:
                    dangerous_incorrect += 1
                    dangerous_type_counts[correction_type]["incorrect_auto"] += 1
        full_body_absorbed = None
        summary_has_no_stale_original = None
        if correct and case:
            original = str(case.get("original_text") or "").strip()
            corrected = str(case.get("corrected_text") or "").strip()
            if original and corrected and original != corrected:
                full_body_absorption_expected += 1
                full_body_absorbed = (
                    corrected in full_body_text
                    and original not in full_body_text.replace(corrected, "")
                )
                full_body_absorption_passed += int(full_body_absorbed)
                # A summary may legitimately omit a detail.  It only fails
                # freshness when it repeats the stale original outside an
                # occurrence of the corrected phrase.
                summary_freshness_expected += 1
                summary_has_no_stale_original = (
                    original not in smart_summary_text.replace(corrected, "")
                )
                summary_freshness_passed += int(summary_has_no_stale_original)
        decision_rows.append(
            {
                "candidate_id": candidate_id,
                "case_id": str(case.get("case_id") or "") if case else "",
                "correction_type": correction_type,
                "auto_applied": auto_applied,
                "correct": correct,
                "absorbed_by_full_body": full_body_absorbed,
                "summary_has_no_stale_original": summary_has_no_stale_original,
            }
        )

    excluded_cases = [
        row
        for row in cases
        if str(row.get("projection_disposition") or "") == "excluded_confirmed"
    ]
    excluded_budget_consumed = sum(
        1
        for case in excluded_cases
        if any(_candidate_matches_case(candidate, case) for candidate in candidates)
    )
    excluded_leaks = sum(
        1
        for case in excluded_cases
        if str(case.get("original_text") or "")
        and (
            str(case.get("original_text")) in full_body_text
            or str(case.get("original_text")) in smart_summary_text
        )
    )

    metrics = {
        "candidate_recall": _ratio(true_positive, true_positive + false_negative),
        "candidate_precision": _ratio(true_positive, true_positive + false_positive),
        "auto_apply_accuracy": _ratio(correct_auto, correct_auto + incorrect_auto),
        "dangerous_misedit_rate": _ratio(
            dangerous_incorrect, dangerous_auto, empty_value=0.0
        ),
        "full_body_absorption_rate": _ratio(
            full_body_absorption_passed, full_body_absorption_expected
        ),
        "summary_no_stale_original_rate": _ratio(
            summary_freshness_passed, summary_freshness_expected
        ),
        "excluded_content_candidate_budget_consumed": excluded_budget_consumed,
        "excluded_content_leak_count": excluded_leaks,
    }
    thresholds = gold.get("thresholds") if isinstance(gold.get("thresholds"), dict) else {}
    checks = {
        "candidate_recall": metrics["candidate_recall"]
        >= float(thresholds.get("candidate_recall_min", 0.8)),
        "candidate_precision": metrics["candidate_precision"]
        >= float(thresholds.get("candidate_precision_min", 0.8)),
        "auto_apply_accuracy": metrics["auto_apply_accuracy"]
        >= float(thresholds.get("auto_apply_accuracy_min", 0.98)),
        "dangerous_misedit_rate": metrics["dangerous_misedit_rate"]
        <= float(thresholds.get("dangerous_misedit_rate_max", 0.0)),
        "full_body_absorption_rate": metrics["full_body_absorption_rate"]
        >= float(
            thresholds.get(
                "full_body_absorption_rate_min",
                thresholds.get("downstream_absorption_rate_min", 1.0),
            )
        ),
        "summary_no_stale_original_rate": metrics["summary_no_stale_original_rate"]
        >= float(thresholds.get("summary_no_stale_original_rate_min", 1.0)),
        "excluded_content_candidate_budget": excluded_budget_consumed
        <= int(thresholds.get("excluded_content_candidate_budget_max", 0)),
        "excluded_content_leaks": excluded_leaks
        <= int(thresholds.get("excluded_content_leak_max", 0)),
    }
    result = {
        "schema": SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "ok": all(checks.values()),
        "gold_json": str(gold_path),
        "candidate_pack_json": str(pack_path),
        "validation_json": str(validation_path),
        "counts": {
            "gold_cases": len(cases),
            "expected_candidates": len(expected_cases),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "accepted_decisions": len(accepted),
        },
        "metrics": metrics,
        "dangerous_misedit_by_type": {
            correction_type: {
                **counts,
                "rate": _ratio(
                    counts["incorrect_auto"],
                    counts["auto_applied"],
                    empty_value=0.0,
                ),
            }
            for correction_type, counts in sorted(dangerous_type_counts.items())
        },
        "checks": checks,
        "decision_rows": decision_rows,
        "boundary": {
            "deidentified_fixture_only": bool(gold.get("deidentified_fixture_only")),
            "does_not_modify_bundle": True,
            "does_not_call_provider": True,
        },
        "updated_at": now_iso(),
    }
    if output_json is not None:
        write_json(Path(output_json).expanduser().resolve(), result)
    return result


def run_transcript_semantic_benchmark(
    gold_json: str | Path,
    candidate_pack_json: str | Path,
    validation_json: str | Path,
    *,
    full_body_path: str | Path | None = None,
    smart_summary_path: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run the gold evaluator using local projection artifact paths only."""

    full_body_text, full_body_identity = _read_projection_file(full_body_path)
    smart_summary_text, smart_summary_identity = _read_projection_file(
        smart_summary_path
    )
    result = evaluate_transcript_semantic_gold(
        gold_json,
        candidate_pack_json,
        validation_json,
        full_body_text=full_body_text,
        smart_summary_text=smart_summary_text,
    )
    result["projection_inputs"] = {
        "full_body": full_body_identity,
        "smart_summary": smart_summary_identity,
    }
    if output_json is not None and str(output_json).strip():
        write_json(Path(output_json).expanduser().resolve(), result)
    return result


def _candidate_matches_case(candidate: dict[str, Any], case: dict[str, Any]) -> bool:
    return (
        int(candidate.get("segment_index") or 0)
        == int(case.get("segment_index") or 0)
        and str(candidate.get("original_text") or "").strip()
        == str(case.get("original_text") or "").strip()
        and (
            not str(case.get("correction_type") or "")
            or str(candidate.get("correction_type") or "")
            == str(case.get("correction_type") or "")
        )
    )


def _read_projection_file(
    path: str | Path | None,
) -> tuple[str, dict[str, Any]]:
    if path is None or not str(path).strip():
        return "", {"path": "", "sha256": "", "bytes": 0}
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"projection input file not found: {resolved}")
    payload = resolved.read_bytes()
    return payload.decode("utf-8-sig"), {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _ratio(numerator: int, denominator: int, *, empty_value: float = 1.0) -> float:
    if denominator <= 0:
        return float(empty_value)
    return round(numerator / denominator, 4)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value
