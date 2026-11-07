from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from .file_hash import sha256_file as _sha256
from .asr_response_quality import assess_asr_response
from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .transcript_source_arbitration import _render_corrected_markdown, _render_srt


SCHEMA = "video_knowledge_pipeline.asr_targeted_retry_merge.v1"


def merge_asr_targeted_retry_reports(
    bundle_dir: str | Path,
    authorization_plan: str | Path,
    execution_reports: Sequence[str | Path],
    *,
    write: bool = False,
    refresh_downstream: bool = True,
) -> dict[str, Any]:
    """Validate consented retry reports and replace only their rejected source segments."""

    root = Path(bundle_dir).expanduser().resolve()
    plan_path = Path(authorization_plan).expanduser().resolve()
    plan = _object(read_json(plan_path), "authorization plan")
    canonical_path = root / "source-arbitrated-transcript.json"
    canonical = _object(read_json(canonical_path), "canonical transcript")
    original_segments = [dict(row) for row in canonical.get("segments") or [] if isinstance(row, dict)]
    if not original_segments:
        raise ValueError("canonical transcript contains no segments")

    expected = _expected_artifacts(plan)
    reports = [Path(value).expanduser().resolve() for value in execution_reports]
    report_rows = [_object(read_json(path), f"execution report {path}") for path in reports]
    by_artifact = _execution_reports_by_artifact(report_rows, reports)
    replacements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for artifact_path, artifact in expected.items():
        try:
            execution, execution_path = by_artifact[artifact_path]
            replacements.extend(
                _replacement_rows(
                    artifact,
                    execution,
                    execution_path=execution_path,
                    plan=plan,
                    canonical_segments=original_segments,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "artifact_path": artifact_path,
                    "retry_id": str(artifact.get("retry_id") or ""),
                    "source_segment_ids": list(artifact.get("source_segment_ids") or []),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    extra_reports = sorted(set(by_artifact) - set(expected))
    for path in extra_reports:
        failures.append(
            {
                "artifact_path": path,
                "retry_id": "",
                "source_segment_ids": [],
                "error": "ValueError: execution report artifact is not present in the authorization plan",
            }
        )

    updated = json.loads(json.dumps(canonical, ensure_ascii=False))
    updated_segments = [dict(row) for row in updated.get("segments") or [] if isinstance(row, dict)]
    applied: list[dict[str, Any]] = []
    for replacement in replacements:
        index = int(replacement["canonical_segment_position"])
        segment = updated_segments[index]
        old_text = str(segment.get("text") or "")
        new_text = str(replacement["replacement_text"])
        if old_text == new_text:
            failures.append(
                {
                    "artifact_path": replacement["artifact_path"],
                    "retry_id": replacement["retry_id"],
                    "source_segment_ids": replacement["source_segment_ids"],
                    "error": "ValueError: retry text is identical to the rejected original text",
                }
            )
            continue
        transformation = {
            "schema": "video_knowledge_pipeline.asr_targeted_retry_transformation.v1",
            "retry_id": replacement["retry_id"],
            "consent_id": replacement["consent_id"],
            "provider": replacement["provider"],
            "model": replacement["model"],
            "destination": replacement["destination"],
            "artifact_path": replacement["artifact_path"],
            "artifact_sha256": replacement["artifact_sha256"],
            "execution_report_path": replacement["execution_report_path"],
            "source_segment_ids": replacement["source_segment_ids"],
            "original_text": old_text,
            "replacement_text": new_text,
            "timestamp_boundary_preserved": True,
            "updated_at": now_iso(),
        }
        segment["text"] = new_text
        segment["corrected_text"] = new_text
        segment["changed"] = True
        transformations = [
            dict(row)
            for row in segment.get("asr_retry_transformations") or []
            if isinstance(row, dict)
        ]
        transformations.append(transformation)
        segment["asr_retry_transformations"] = transformations
        applied.append(
            {
                **replacement,
                "original_text": old_text,
                "replacement_text": new_text,
                "timestamp_boundary_preserved": True,
            }
        )
    updated["segments"] = updated_segments
    unresolved = len(expected) - len(applied)
    status = "completed" if not failures and unresolved == 0 else ("degraded" if applied else "failed")
    updated["source"] = "transcript_semantic_correction+targeted_asr_recovery"
    updated["updated_at"] = now_iso()
    previous_recovery = (
        canonical.get("targeted_asr_recovery")
        if isinstance(canonical.get("targeted_asr_recovery"), dict)
        else {}
    )
    recovery_history = [
        json.loads(json.dumps(row, ensure_ascii=False))
        for row in previous_recovery.get("history") or []
        if isinstance(row, dict)
    ]
    if previous_recovery:
        previous_snapshot = json.loads(json.dumps(previous_recovery, ensure_ascii=False))
        previous_snapshot.pop("history", None)
        recovery_history.append(previous_snapshot)
    cumulative_repaired_positions: list[int] = []
    cumulative_transformation_count = 0
    for position, segment in enumerate(updated_segments):
        transformations = [
            row
            for row in segment.get("asr_retry_transformations") or []
            if isinstance(row, dict)
        ]
        if transformations:
            cumulative_repaired_positions.append(position)
            cumulative_transformation_count += len(transformations)
    recovery = {
        "schema": SCHEMA,
        "status": status,
        "authorization_plan": str(plan_path),
        "authorization_plan_sha256": _sha256(plan_path),
        "expected_retry_count": len(expected),
        "applied_retry_count": len(applied),
        "failed_retry_count": len(failures),
        "applied": applied,
        "failures": failures,
        "cumulative_repaired_segment_count": len(cumulative_repaired_positions),
        "cumulative_repaired_segment_positions": cumulative_repaired_positions,
        "cumulative_transformation_count": cumulative_transformation_count,
    }
    if recovery_history:
        recovery["history"] = recovery_history
    updated["targeted_asr_recovery"] = recovery

    canonical_before_sha256 = _sha256(canonical_path)
    canonical_after_sha256 = _payload_sha256(updated)
    unchanged_positions = [
        index
        for index, (before, after) in enumerate(zip(original_segments, updated_segments, strict=True))
        if before == after
    ]
    result = {
        "schema": SCHEMA,
        "status": status,
        "ok": status == "completed",
        "write": bool(write),
        "bundle_dir": str(root),
        "authorization_plan": str(plan_path),
        "authorization_plan_sha256": _sha256(plan_path),
        "execution_reports": [str(path) for path in reports],
        "canonical_path": str(canonical_path),
        "canonical_before_sha256": canonical_before_sha256,
        "canonical_after_sha256": canonical_after_sha256,
        "canonical_segment_count": len(original_segments),
        "applied_retry_count": len(applied),
        "failed_retry_count": len(failures),
        "prior_recovery_count": len(recovery_history),
        "cumulative_repaired_segment_count": len(cumulative_repaired_positions),
        "cumulative_repaired_segment_positions": cumulative_repaired_positions,
        "cumulative_transformation_count": cumulative_transformation_count,
        "unchanged_segment_count": len(unchanged_positions),
        "unchanged_segment_positions": unchanged_positions,
        "applied": applied,
        "failures": failures,
        "production_qualified": status == "completed",
        "requires_export_refresh": bool(applied),
        "refresh_downstream": bool(refresh_downstream),
        "silent_fallback_performed": False,
        "raw_asr_modified": False,
        "updated_at": now_iso(),
    }
    if write and applied:
        report_path = root / "asr-targeted-retry-merge-report.json"
        manifest_path = root / "manifest.json"
        manifest = _object(read_json(manifest_path), "bundle manifest") if manifest_path.exists() else {}
        with bundle_write_lock(root, operation="asr_targeted_retry_merge", timeout_seconds=5):
            write_json(canonical_path, updated)
            (root / "source-arbitrated-transcript.srt").write_text(
                _render_srt(updated_segments), encoding="utf-8"
            )
            (root / "source-arbitrated-transcript.md").write_text(
                _render_corrected_markdown(updated), encoding="utf-8"
            )
            manifest["source_arbitrated_transcript_json"] = "source-arbitrated-transcript.json"
            manifest["source_arbitrated_transcript_srt"] = "source-arbitrated-transcript.srt"
            manifest["source_arbitrated_transcript_markdown"] = "source-arbitrated-transcript.md"
            manifest["asr_targeted_retry_merge_report"] = report_path.name
            manifest["asr_targeted_retry_status"] = status
            manifest["asr_targeted_retry_cumulative_repaired_segment_count"] = len(
                cumulative_repaired_positions
            )
            write_json(manifest_path, manifest)
            result["canonical_written_sha256"] = _sha256(canonical_path)
            result["report_path"] = str(report_path)
            write_json(report_path, result)
        if refresh_downstream:
            try:
                from .transcript_downstream_refresh import (
                    refresh_transcript_downstream_outputs,
                )

                downstream = refresh_transcript_downstream_outputs(
                    root,
                    canonical_before_sha256=canonical_before_sha256,
                    canonical_after_sha256=result["canonical_written_sha256"],
                    reason="targeted_asr_merge",
                    write=True,
                )
            except Exception as exc:  # noqa: BLE001 - canonical repair must remain committed.
                downstream = {
                    "schema": "video_knowledge_pipeline.transcript_downstream_refresh.v1",
                    "status": "degraded",
                    "ok": False,
                    "local_refresh_completed": False,
                    "requires_summary_regeneration": True,
                    "full_pipeline_production_qualified": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            downstream = {
                "schema": "video_knowledge_pipeline.transcript_downstream_refresh.v1",
                "status": "skipped",
                "ok": False,
                "local_refresh_completed": False,
                "requires_summary_regeneration": True,
                "full_pipeline_production_qualified": False,
                "reason": "disabled_by_caller",
            }
        result["downstream_refresh"] = downstream
        result["requires_export_refresh"] = not bool(
            downstream.get("local_refresh_completed")
        )
        result["requires_summary_regeneration"] = bool(
            downstream.get("requires_summary_regeneration", True)
        )
        result["full_pipeline_production_qualified"] = bool(
            downstream.get("full_pipeline_production_qualified")
        )
        with bundle_write_lock(
            root,
            operation="asr_targeted_retry_merge_refresh_status",
            timeout_seconds=5,
        ):
            manifest = (
                _object(read_json(manifest_path), "bundle manifest")
                if manifest_path.exists()
                else {}
            )
            manifest["asr_targeted_retry_downstream_refresh_status"] = str(
                downstream.get("status") or "unknown"
            )
            manifest["smart_summary_requires_regeneration"] = result[
                "requires_summary_regeneration"
            ]
            manifest["full_pipeline_production_qualified"] = result[
                "full_pipeline_production_qualified"
            ]
            write_json(manifest_path, manifest)
            write_json(report_path, result)
    return result


def _expected_artifacts(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [dict(row) for row in plan.get("artifact_manifest") or [] if isinstance(row, dict)]
    if not rows:
        raise ValueError("authorization plan contains no artifacts")
    expected: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = Path(str(row.get("path") or "")).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"authorized retry artifact not found: {path}")
        actual_hash = _sha256(path)
        actual_bytes = path.stat().st_size
        if actual_hash != str(row.get("sha256") or ""):
            raise ValueError(f"authorized retry artifact hash mismatch: {path}")
        if actual_bytes != int(row.get("bytes") or -1):
            raise ValueError(f"authorized retry artifact byte count mismatch: {path}")
        item = dict(row)
        item["path"] = str(path)
        expected[str(path)] = item
    return expected


def _execution_reports_by_artifact(
    reports: list[dict[str, Any]], paths: list[Path]
) -> dict[str, tuple[dict[str, Any], Path]]:
    result: dict[str, tuple[dict[str, Any], Path]] = {}
    for report, path in zip(reports, paths, strict=True):
        artifacts = [str(Path(str(value)).expanduser().resolve()) for value in report.get("artifact_paths") or []]
        if len(artifacts) != 1:
            raise ValueError(f"targeted retry execution must contain exactly one artifact: {path}")
        if artifacts[0] in result:
            raise ValueError(f"duplicate execution report for artifact: {artifacts[0]}")
        result[artifacts[0]] = (report, path)
    return result


def _replacement_rows(
    artifact: dict[str, Any],
    execution: dict[str, Any],
    *,
    execution_path: Path,
    plan: dict[str, Any],
    canonical_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if str(execution.get("task") or "") not in {"cloud_asr", "local_asr_service"}:
        raise ValueError("execution report is not an ASR task")
    if not execution.get("production_qualified"):
        raise ValueError("execution report did not pass the connector production quality gate")
    deployment = _deployment(execution)
    destination = str(plan.get("destination") or "")
    if urlparse(str(deployment.get("base_url") or "")).hostname != urlparse(destination).hostname:
        raise ValueError("execution destination does not match the authorization plan")
    if str(deployment.get("model") or "") != str(plan.get("model") or ""):
        raise ValueError("execution model does not match the authorization plan")
    provider_values = {
        str(deployment.get("provider") or ""),
        str(deployment.get("litellm_provider") or ""),
    }
    if str(plan.get("provider") or "") not in provider_values:
        raise ValueError("execution provider does not match the authorization plan")
    consent_id = str(execution.get("consent_id") or "")
    if not consent_id:
        raise ValueError("execution report is missing consent_id")
    locked_instructions, locked_prompt = _locked_asr_context(
        execution,
        consent_id=consent_id,
        artifact=artifact,
        plan=plan,
    )
    raw = _raw_asr_response(execution)
    quality = assess_asr_response(
        raw,
        task_instructions=locked_instructions,
        asr_prompt=locked_prompt,
    )
    if not quality.get("quality_gate_passed"):
        raise ValueError(f"retry ASR quality gate failed: {quality.get('status')}")

    window = artifact.get("source_window_seconds") if isinstance(artifact.get("source_window_seconds"), dict) else {}
    window_start = float(window.get("start") or 0.0)
    target_ids = [str(value) for value in artifact.get("source_segment_ids") or []]
    rows: list[dict[str, Any]] = []
    for source_id in target_ids:
        position = _canonical_position(canonical_segments, source_id)
        target = canonical_segments[position]
        target_start = float(target.get("start") or 0.0)
        target_end = float(target.get("end") or target_start)
        text, evidence = _crop_retry_text(
            raw,
            snippet_start=window_start,
            target_start=target_start,
            target_end=target_end,
        )
        rows.append(
            {
                "retry_id": str(artifact.get("retry_id") or ""),
                "source_segment_ids": [source_id],
                "canonical_segment_position": position,
                "target_start": target_start,
                "target_end": target_end,
                "replacement_text": text,
                "crop_evidence": evidence,
                "artifact_path": str(artifact["path"]),
                "artifact_sha256": str(artifact.get("sha256") or ""),
                "execution_report_path": str(execution_path),
                "consent_id": consent_id,
                "provider": str(plan.get("provider") or ""),
                "model": str(plan.get("model") or ""),
                "destination": destination,
                "quality": quality,
            }
        )
    return rows


def _crop_retry_text(
    raw: dict[str, Any],
    *,
    snippet_start: float,
    target_start: float,
    target_end: float,
) -> tuple[str, dict[str, Any]]:
    segments = [dict(row) for row in raw.get("segments") or [] if isinstance(row, dict)]
    selected: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    excluded_boundary_segments: list[dict[str, Any]] = []
    for row in segments:
        local_start = float(row.get("start") or 0.0)
        local_end = float(row.get("end") or local_start)
        absolute_start = snippet_start + local_start
        absolute_end = snippet_start + local_end
        midpoint = (absolute_start + absolute_end) / 2
        overlap = max(0.0, min(absolute_end, target_end) - max(absolute_start, target_start))
        duration = max(0.001, absolute_end - absolute_start)
        if target_start <= midpoint <= target_end or overlap / duration >= 0.8:
            extends_left = max(0.0, target_start - absolute_start)
            extends_right = max(0.0, absolute_end - target_end)
            boundary_evidence = {
                "start": absolute_start,
                "end": absolute_end,
                "overlap_ratio": overlap / duration,
                "extends_left": extends_left,
                "extends_right": extends_right,
            }
            if (extends_left > 0.75 or extends_right > 0.75) and not row.get("words"):
                if overlap / duration < 0.8:
                    excluded_boundary_segments.append(boundary_evidence)
                else:
                    ambiguous.append(boundary_evidence)
                continue
            selected.append(
                {
                    "start": absolute_start,
                    "end": absolute_end,
                    "text": str(row.get("text") or "").strip(),
                }
            )
    if ambiguous:
        raise ValueError(f"retry segment crosses target boundaries without word timestamps: {ambiguous}")
    text = "".join(str(row["text"]) for row in selected if str(row["text"]))
    if not text:
        if excluded_boundary_segments:
            raise ValueError(
                "retry segment crosses target boundaries without word timestamps: excluded-only"
            )
        raise ValueError("retry response contains no unambiguous text inside the target segment")
    return text, {
        "selected_segments": selected,
        "excluded_boundary_segments": excluded_boundary_segments,
        "boundary_mode": "segment_midpoint_or_80pct_overlap",
    }


def _canonical_position(segments: list[dict[str, Any]], source_id: str) -> int:
    def identity(value: Any) -> str:
        return "" if value is None else str(value)

    matches = [
        index
        for index, row in enumerate(segments)
        if source_id
        in {
            identity(row.get("index")),
            identity(row.get("source_segment_index")),
            identity(row.get("id")),
            identity(row.get("segment_id")),
            *[str(value) for value in row.get("source_segment_ids") or []],
        }
    ]
    if len(matches) != 1:
        raise ValueError(f"source segment id must match exactly one canonical segment: {source_id}")
    return matches[0]


def _locked_asr_context(
    execution: dict[str, Any],
    *,
    consent_id: str,
    artifact: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[str, str]:
    path = Path(str(execution.get("consent_path") or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"locked consent file not found: {path}")
    consent = _object(read_json(path), "consent")
    if str(consent.get("consent_id") or "") != consent_id:
        raise ValueError("execution report consent_id does not match its consent file")
    if str(consent.get("schema") or "") != "video_knowledge_pipeline.model_connector_consent.v2":
        raise ValueError("targeted remote ASR merge requires consent v2")
    if not consent.get("user_confirmed_data_export"):
        raise ValueError("targeted retry consent was not explicitly confirmed")
    confirmation = consent.get("operator_confirmation") if isinstance(consent.get("operator_confirmation"), dict) else {}
    if not confirmation.get("confirmed"):
        raise ValueError("targeted retry consent operator confirmation is missing")
    if str(consent.get("instruction_transport") or "") != "audit_only":
        raise ValueError("ASR task instructions were not locked as audit-only")
    if str(consent.get("asr_prompt_transport") or "") != "provider_audio_prompt":
        raise ValueError("ASR prompt transport is not locked to provider_audio_prompt")

    locked_artifacts = [dict(row) for row in consent.get("artifacts") or [] if isinstance(row, dict)]
    if len(locked_artifacts) != 1:
        raise ValueError("targeted retry consent must lock exactly one artifact")
    locked_artifact = locked_artifacts[0]
    if Path(str(locked_artifact.get("path") or "")).expanduser().resolve() != Path(str(artifact["path"])).resolve():
        raise ValueError("targeted retry consent artifact path differs from the authorization plan")
    if str(locked_artifact.get("sha256") or "") != str(artifact.get("sha256") or ""):
        raise ValueError("targeted retry consent artifact hash differs from the authorization plan")
    if int(locked_artifact.get("bytes") or -1) != int(artifact.get("bytes") or -1):
        raise ValueError("targeted retry consent artifact bytes differ from the authorization plan")

    destination = str(plan.get("destination") or "")
    destination_host = urlparse(destination).hostname
    locked_hosts = {
        urlparse(str(value or "")).hostname
        for value in consent.get("authorized_destinations") or []
    }
    if destination_host not in locked_hosts:
        raise ValueError("targeted retry consent destination differs from the authorization plan")
    plan_route = plan.get("route") if isinstance(plan.get("route"), dict) else {}
    consent_route = consent.get("route") if isinstance(consent.get("route"), dict) else {}
    execution_route = execution.get("route") if isinstance(execution.get("route"), dict) else {}
    for key in ("route_id", "route_revision", "virtual_model"):
        expected = str(plan_route.get(key) or "")
        if expected and str(consent_route.get(key) or "") != expected:
            raise ValueError(f"targeted retry consent {key} differs from the authorization plan")
        if expected and str(execution_route.get(key) or "") != expected:
            raise ValueError(f"targeted retry execution {key} differs from the authorization plan")

    limits = plan.get("limits") if isinstance(plan.get("limits"), dict) else {}
    scope = consent.get("scope") if isinstance(consent.get("scope"), dict) else {}
    allowed_calls = int(limits.get("max_calls_per_artifact") or 0)
    if allowed_calls < 1 or int(scope.get("max_calls") or 0) > allowed_calls:
        raise ValueError("targeted retry consent call limit exceeds the authorization plan")
    artifact_count = max(1, len(plan.get("artifact_manifest") or []))
    per_artifact_budget = float(limits.get("max_estimated_cost_usd") or 0) / artifact_count
    if float(scope.get("max_estimated_cost_usd") or 0) > per_artifact_budget + 1e-9:
        raise ValueError("targeted retry consent cost limit exceeds the authorization plan")
    expected_retries = int(limits.get("external_retries") or 0)
    if scope.get("max_retries_per_call") != expected_retries:
        raise ValueError("targeted retry consent retry limit differs from the authorization plan")
    model = execution.get("model_result") if isinstance(execution.get("model_result"), dict) else {}
    runtime = model.get("runtime_result") if isinstance(model.get("runtime_result"), dict) else {}
    request_options = runtime.get("request_options") if isinstance(runtime.get("request_options"), dict) else {}
    if request_options.get("max_retries") != expected_retries:
        raise ValueError("targeted retry runtime retry limit differs from the authorization plan")
    return str(consent.get("instructions") or ""), str(consent.get("asr_prompt") or "")


def _deployment(execution: dict[str, Any]) -> dict[str, Any]:
    route = execution.get("route") if isinstance(execution.get("route"), dict) else {}
    rows = [dict(row) for row in route.get("deployments") or [] if isinstance(row, dict)]
    if len(rows) != 1:
        raise ValueError("targeted retry route must lock exactly one deployment")
    return rows[0]


def _raw_asr_response(execution: dict[str, Any]) -> dict[str, Any]:
    model = execution.get("model_result") if isinstance(execution.get("model_result"), dict) else {}
    runtime = model.get("runtime_result") if isinstance(model.get("runtime_result"), dict) else {}
    raw = runtime.get("raw_output") if isinstance(runtime.get("raw_output"), dict) else model.get("raw_response")
    if not isinstance(raw, dict):
        raise ValueError("execution report contains no verbose ASR response")
    return raw


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _payload_sha256(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge consented targeted ASR retries into canonical transcript")
    parser.add_argument("bundle_dir")
    parser.add_argument("authorization_plan")
    parser.add_argument("execution_reports", nargs="+")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--no-refresh-downstream", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = merge_asr_targeted_retry_reports(
        args.bundle_dir,
        args.authorization_plan,
        args.execution_reports,
        write=args.write,
        refresh_downstream=not args.no_refresh_downstream,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
