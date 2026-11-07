"""Version-bound result recovery over the existing VKP run registry.

No producer is rerun. Immutable stage snapshots are installed into a run-scoped
projection; completion is derived from hashes, not another workflow ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from .artifact_freshness import build_dependency_snapshot, validate_dependency_snapshot
from .artifact_validation import sha256_file
from .canonical_json import canonical_json_sha256
from .models import now_iso
from .run_artifact_registry import RUN_SCHEMA, _write_run, register_bundle_run
from .stage_cache import atomic_copy_file
from .storage import bundle_write_lock, read_json

PROFILE = "versioned_result_recovery.v1"
MAX_BYTES = 4 * 1024 * 1024
MAX_STAGES = 64
KINDS = {"current_candidates", "historical_candidates", "ocr_inventory", "human_evidence", "discovery_prompts", "other"}


def _producer_versions() -> dict:
    base = Path(__file__).parent
    return {name: sha256_file(base / name) for name in
            ("run_artifact_recovery.py", "run_artifact_registry.py", "artifact_freshness.py", "stage_cache.py")}


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not value or path == root or not path.is_relative_to(root):
        raise ValueError("artifact_path_outside_bundle")
    return path


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", value):
        raise ValueError("invalid_run_or_stage_id")
    return value


def _json(path: Path) -> dict:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("artifact_too_large")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("artifact_object_required")
    return value


def _binding(run: dict) -> str:
    return canonical_json_sha256({key: run[key] for key in
                                  ("schema", "run_id", "run_type", "bundle_dir", "inputs", "parameters", "artifacts", "dependency_snapshot")})


def capture_result_run(bundle_dir: str | Path, request: dict) -> dict:
    """Pin existing result files, dependencies and versions; never infer history.

    Required request fields: run_id, inputs (bundle-relative metadata paths),
    dependency_versions (nonempty strings), stages [{stage_id, source, kind}].
    Optional declared_count belongs only to the historical source observation.
    Missing stage sources are retained as unreconstructable, not fabricated.
    """
    root = Path(bundle_dir).resolve()
    run_id = _identifier(request.get("run_id"))
    versions = request.get("dependency_versions")
    if not isinstance(versions, dict) or not versions or any(not isinstance(v, str) or not v.strip() for v in versions.values()):
        raise ValueError("dependency_versions_required")
    inputs = request.get("inputs")
    stages = request.get("stages")
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= MAX_STAGES or not all(isinstance(p, str) for p in inputs):
        raise ValueError("explicit_inputs_required")
    if not isinstance(stages, list) or not 1 <= len(stages) <= MAX_STAGES:
        raise ValueError("bounded_stages_required")
    run_dir = root / "runs" / run_id
    with bundle_write_lock(root, operation="capture_versioned_results"):
        if run_dir.exists():
            raise ValueError("run_identity_already_exists")
        for value in inputs:
            path = _inside(root, value)
            if path.suffix.lower() not in {".json", ".txt", ".srt", ".vtt", ".md", ".toml"}:
                raise ValueError("metadata_inputs_only")
            if not path.is_file() or path.stat().st_size > MAX_BYTES:
                raise ValueError(f"input_missing_or_too_large:{value}")
        dependency = build_dependency_snapshot(root, subject=PROFILE, inputs=inputs,
                                               source_run_id=run_id, producer_schema=PROFILE)
        plan, ids = [], set()
        # Validate all paths before creating a new run directory.
        for row in stages:
            sid = _identifier(row.get("stage_id"))
            if sid in ids or row.get("kind") not in KINDS:
                raise ValueError("duplicate_stage_or_unknown_kind")
            ids.add(sid)
            source = _inside(root, str(row.get("source") or ""))
            if source.suffix.lower() != ".json":
                raise ValueError("result_json_required")
            value = _json(source) if source.is_file() else None
            actual_producer_run = str((value or {}).get("run_id") or "")
            expected_producer_run = str(row.get("source_run_id") or "")
            if actual_producer_run and actual_producer_run != expected_producer_run:
                raise ValueError(f"explicit_source_run_identity_required:{sid}")
            plan.append({"stage_id": sid, "kind": row["kind"], "source": source.relative_to(root).as_posix(),
                         "snapshot": f"runs/{run_id}/snapshots/{sid}.json",
                         "target": f"exports/versioned-results/{run_id}/{sid}.json",
                         "source_sha256": sha256_file(source) if value is not None else "",
                         "declared_count": row.get("declared_count"),
                         "source_run_id": actual_producer_run or "not_recorded",
                         "snapshot_available": value is not None})
        artifacts = []
        for row in plan:
            if row["snapshot_available"]:
                source, target = _inside(root, row["source"]), _inside(root, row["snapshot"])
                atomic_copy_file(source, target)
                if sha256_file(target) != row["source_sha256"] or sha256_file(source) != row["source_sha256"]:
                    raise ValueError("source_changed_during_capture")
                artifacts.append({"key": row["stage_id"], "path": row["snapshot"]})
        if not validate_dependency_snapshot(root, dependency)["passed"]:
            raise ValueError("dependencies_changed_during_capture")
        run = register_bundle_run(root, run_type="versioned_results", run_id=run_id, status="captured",
                                  inputs={"explicit_paths": inputs},
                                  parameters={"recovery_profile": PROFILE, "dependency_versions": versions,
                                              "producer_code_versions": _producer_versions(), "stages": plan},
                                  artifacts=artifacts, dependency_snapshot=dependency,
                                  operator_boundary={"projection_only": True, "no_producer_execution": True,
                                                     "semantic_acceptance": False}, write=False)
        run["result_binding_sha256"] = _binding(run)
        # Existing run.json is the only identity record. Do not rewrite the
        # source manifest through registry refresh (it may itself be an input).
        _write_run(root, run)
    return inspect_result_run(root, run_id, dependency_versions=versions)


def _load(root: Path, run_id: str) -> dict:
    run = _json(_inside(root, f"runs/{_identifier(run_id)}/run.json"))
    if (run.get("schema") != RUN_SCHEMA or run.get("run_id") != run_id
            or run.get("run_type") != "versioned_results" or run.get("bundle_dir") != str(root)
            or (run.get("parameters") or {}).get("recovery_profile") != PROFILE
            or run.get("result_binding_sha256") != _binding(run)):
        raise ValueError("run_identity_or_binding_mismatch")
    snap = run.get("dependency_snapshot") or {}
    if snap.get("source_run_id") != run_id or snap.get("producer_schema") != PROFILE:
        raise ValueError("dependency_run_identity_mismatch")
    return run


def _projection(row: dict, payload: dict) -> dict:
    kind = row["kind"]
    if kind in {"current_candidates", "historical_candidates"}:
        candidates = payload.get("candidates")
        concrete = sum(isinstance(c, dict) and bool(c.get("candidate_id"))
                       and bool(c.get("original_text") or c.get("text")) for c in candidates or [])
        declared = row.get("declared_count")
        if declared is None:
            declared = payload.get("candidate_count")
        valid_count = isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0
        missing = max(0, declared - concrete) if valid_count else None
        return {"kind": kind, "concrete_candidate_count": concrete,
                "declared_count": declared, "missing_candidate_bodies": missing,
                "current_zero_candidates": kind == "current_candidates" and candidates == [],
                "candidate_list_present": isinstance(candidates, list),
                "historical_rebuild_status": "unreconstructable_missing_bodies" if missing else "not_attempted",
                "semantic_acceptance": False, "zero_candidates_means_zero_errors": False}
    if kind == "ocr_inventory":
        rows = [r for r in payload.get("items", []) if isinstance(r, dict)]
        counts = Counter(r.get("execution_status", "unknown") for r in rows)
        known = {"completed", "failed", "not_run", "running", "blocked", "human_confirmed_evidence", "existing_evidence"}
        return {"kind": kind, "required": len(rows),
                "machine_executed": counts["completed"] + counts["failed"],
                "machine_failed": counts["failed"],
                "machine_not_run": sum(counts[s] for s in ("not_run", "running", "blocked")),
                "human_evidence": counts["human_confirmed_evidence"],
                "existing_evidence": counts["existing_evidence"],
                "unknown_execution": sum(count for state, count in counts.items() if state not in known),
                "empty_results": sum(r.get("recognition_status") == "empty" for r in rows),
                "full_video_page_coverage_verified": False}
    if kind == "human_evidence":
        return {"kind": kind, "record_count": len(payload.get("items") or payload.get("decisions") or []),
                "included_in_machine_ocr_count": False}
    if kind == "discovery_prompts":
        return {"kind": kind, "coverage": payload.get("coverage") or {}, "semantic_execution_complete": False}
    return {"kind": kind}


def inspect_result_run(bundle_dir: str | Path, run_id: str, *, dependency_versions: dict | None = None) -> dict:
    root = Path(bundle_dir).resolve()
    run = _load(root, run_id)
    dep = validate_dependency_snapshot(root, run["dependency_snapshot"])
    expected_versions = run["parameters"]["dependency_versions"]
    version_state = "not_checked" if dependency_versions is None else ("matching" if dependency_versions == expected_versions else "changed")
    issues = list(dep["issues"])
    if run["parameters"].get("producer_code_versions") != _producer_versions():
        issues.append({"key": "producer_code_changed"})
    if version_state != "matching":
        issues.append({"key": "dependency_versions_" + version_state, "expected": expected_versions, "actual": dependency_versions})
    rows = []
    by_key = {r["key"]: r for r in run["artifacts"]}
    for stage in run["parameters"]["stages"]:
        row = dict(stage)
        snapshot = _inside(root, row["snapshot"])
        target = _inside(root, row["target"])
        evidence = by_key.get(row["stage_id"], {})
        # Enforce run namespaces even if a tampered record has a recomputed hash.
        if row["snapshot"] != f"runs/{run_id}/snapshots/{row['stage_id']}.json" or row["target"] != f"exports/versioned-results/{run_id}/{row['stage_id']}.json":
            raise ValueError("stage_run_namespace_mismatch")
        if not row["snapshot_available"] or not snapshot.is_file():
            row["state"] = "unreconstructable_snapshot_missing"
        elif (evidence.get("path", "").replace("\\", "/") != row["snapshot"]
              or sha256_file(snapshot) != row["source_sha256"] or evidence.get("sha256") != row["source_sha256"]):
            row["state"] = "snapshot_hash_mismatch"
        else:
            row["evidence"] = _projection(row, _json(snapshot))
            source = _inside(root, row["source"])
            row["source_state"] = "missing" if not source.is_file() else (
                "matching" if sha256_file(source) == row["source_sha256"] else "changed")
            row["evidence_freshness"] = ("historical" if row["kind"] == "historical_candidates" else
                                         "current" if not issues and row["source_state"] == "matching" else "stale_or_unverified")
            if row["kind"] == "current_candidates":
                row["evidence"]["captured_zero_candidates"] = row["evidence"]["current_zero_candidates"]
                row["evidence"]["current_zero_candidates"] &= row["evidence_freshness"] == "current"
            row["state"] = "pending" if not target.exists() else (
                "completed" if target.is_file() and sha256_file(target) == row["source_sha256"] else "target_conflict")
        rows.append(row)
    unsafe = any(r["state"] in {"snapshot_hash_mismatch", "target_conflict"} for r in rows)
    pending = [r["stage_id"] for r in rows if r["state"] == "pending"]
    missing = [r["source"] for r in rows if r["state"] == "unreconstructable_snapshot_missing"]
    gaps = [{"stage_id": r["stage_id"], "missing_candidate_bodies": r["evidence"]["missing_candidate_bodies"]}
            for r in rows if (r.get("evidence") or {}).get("missing_candidate_bodies")]
    ready = not issues and not unsafe
    return {"schema": PROFILE, "run_id": run_id, "binding_sha256": run["result_binding_sha256"],
            "dependency_validation": dep, "dependency_versions": version_state, "issues": issues,
            "stages": rows, "pending_stages": pending, "missing_snapshots": missing,
            "content_gaps": gaps, "content_recovery_complete": not gaps and not missing,
            "can_resume": ready and bool(pending),
            "status": "blocked" if issues or unsafe or missing else ("pending" if pending else "projection_complete"),
            "semantic_acceptance": False, "historical_snapshots_inferred": False}


def resume_result_run(bundle_dir: str | Path, run_id: str, *, dependency_versions: dict,
                      max_stages: int = 0) -> dict:
    root = Path(bundle_dir).resolve()
    restored = []
    with bundle_write_lock(root, operation="resume_versioned_results"):
        before = inspect_result_run(root, run_id, dependency_versions=dependency_versions)
        if before["can_resume"]:
            for row in before["stages"]:
                if row["state"] != "pending":
                    continue
                if max_stages and len(restored) >= max_stages:
                    break
                # Revalidate before every stage; a prior stage is an independent
                # atomic copy, so a crash cannot reset already matching outputs.
                current = inspect_result_run(root, run_id, dependency_versions=dependency_versions)
                if not current["can_resume"]:
                    break
                target = _inside(root, row["target"])
                if target.exists():
                    raise ValueError("target_changed_during_recovery")
                atomic_copy_file(_inside(root, row["snapshot"]), target)
                if sha256_file(target) != row["source_sha256"]:
                    raise ValueError("recovered_artifact_hash_mismatch")
                restored.append(row["stage_id"])
        result = inspect_result_run(root, run_id, dependency_versions=dependency_versions)
    return {**result, "restored_stages": restored, "restored_count": len(restored),
            "skipped_completed": [r["stage_id"] for r in before["stages"] if r["state"] == "completed"],
            "checked_at": now_iso(), "producer_calls": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "status", "resume"))
    parser.add_argument("bundle_dir")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--versions", type=Path)
    parser.add_argument("--max-stages", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.action == "capture":
            if args.request is None:
                raise ValueError("request_required")
            result = capture_result_run(args.bundle_dir, _json(args.request))
        else:
            if not args.run_id or args.versions is None or args.max_stages < 0:
                raise ValueError("run_id_and_current_versions_required")
            versions = _json(args.versions)
            result = (resume_result_run(args.bundle_dir, args.run_id, dependency_versions=versions, max_stages=args.max_stages)
                      if args.action == "resume" else inspect_result_run(args.bundle_dir, args.run_id, dependency_versions=versions))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result["status"] == "blocked" else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc), "semantic_acceptance": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
