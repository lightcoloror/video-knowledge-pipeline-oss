from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .file_hash import sha256_file as _sha256
from .models import now_iso
from .storage import read_json, write_json


COMPARISON_SCHEMA = "video_knowledge_pipeline.model_gateway_abc_comparison.v1"
TEMPORAL_MANIFEST_SCHEMA = "video_knowledge_pipeline.temporal_gateway_acceptance_manifest.v1"
RUNTIME_RESULT_SCHEMA = "video_knowledge_pipeline.model_runtime_result.v1"
LANES = (
    ("A", "legacy_adapter"),
    ("B", "litellm_proxy_remote"),
    ("C", "openai_compatible_local"),
)
LANE_FILENAMES = {
    "A": "lane-a-legacy.json",
    "B": "lane-b-proxy-remote.json",
    "C": "lane-c-proxy-local.json",
}
REQUIRED_RUNTIME_FIELDS = {
    "ok",
    "status",
    "task",
    "execution_location",
    "route_id",
    "route_revision",
    "deployment",
    "provider",
    "latency_ms",
    "usage",
    "estimated_cost",
    "content",
    "evidence",
    "consent_id",
}


def compare_model_gateway_results(
    lane_a: str | Path,
    lane_b: str | Path,
    lane_c: str | Path,
    *,
    output_dir: str | Path,
    sample_id: str = "",
    write: bool = True,
) -> dict[str, Any]:
    paths = [Path(value).expanduser().resolve() for value in (lane_a, lane_b, lane_c)]
    rows = [
        _lane_result(label, expected, path)
        for (label, expected), path in zip(LANES, paths, strict=True)
    ]
    tasks = sorted({str(row["task"]) for row in rows if row["task"]})
    schema_compatible = all(bool(row["schema_contract"]["compatible"]) for row in rows)
    result = {
        "schema": COMPARISON_SCHEMA,
        "status": "ready_for_review" if all(row["loaded"] for row in rows) else "incomplete",
        "sample_id": str(sample_id or ""),
        "tasks": tasks,
        "lanes": rows,
        "comparison": {
            "schema_compatible": schema_compatible,
            "quality_gate": {row["lane"]: row["quality_gate"] for row in rows},
            "latency_ms": {row["lane"]: row["latency_ms"] for row in rows},
            "call_count": {row["lane"]: row["call_count"] for row in rows},
            "estimated_cost": {row["lane"]: row["estimated_cost"] for row in rows},
            "failure_recovery": {row["lane"]: row["failure_recovery"] for row in rows},
        },
        "operator_boundary": {
            "offline_result_comparison_only": True,
            "does_not_call_models": True,
            "does_not_read_source_media": True,
            "does_not_promote_candidate_evidence": True,
            "remote_smoke_requires_separate_consent": True,
        },
        "updated_at": now_iso(),
    }
    out = Path(output_dir).expanduser().resolve()
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "model-gateway-abc-comparison.json", result)
        (out / "model-gateway-abc-comparison.md").write_text(
            _render_comparison_markdown(result), encoding="utf-8"
        )
        result["artifacts"] = {
            "json": str(out / "model-gateway-abc-comparison.json"),
            "markdown": str(out / "model-gateway-abc-comparison.md"),
        }
    return result


def capture_model_gateway_lane_result(
    lane: str,
    execution_result: str | Path,
    *,
    output_dir: str | Path,
    write: bool = True,
) -> dict[str, Any]:
    lane_key = str(lane or "").strip().upper()
    if lane_key not in LANE_FILENAMES:
        raise ValueError("lane must be A, B, or C")
    source = Path(execution_result).expanduser().resolve()
    raw = read_json(source)
    if not isinstance(raw, dict):
        raise ValueError("model gateway lane source must be a JSON object")
    runtime = dict(_unwrap_result(raw))
    missing = sorted(field for field in REQUIRED_RUNTIME_FIELDS if field not in runtime)
    if missing:
        raise ValueError(f"model gateway lane result is missing required fields: {', '.join(missing)}")
    route = raw.get("route") if isinstance(raw.get("route"), dict) else runtime.get("route")
    route = route if isinstance(route, dict) else {}
    deployments = route.get("deployments") if isinstance(route.get("deployments"), list) else []
    backends = {
        str(row.get("adapter_backend") or "").strip().lower()
        for row in deployments
        if isinstance(row, dict)
    }
    allowed_backends = {
        "A": {"legacy"},
        "B": {"proxy"},
        # local-production-v1 uses VKP's built-in OpenAI-compatible loopback
        # adapter; a local LiteLLM proxy remains valid for the same lane.
        "C": {"builtin", "proxy"},
    }[lane_key]
    if not backends or not backends.issubset(allowed_backends):
        expected_backend = " or ".join(sorted(allowed_backends))
        raise ValueError(
            f"lane {lane_key} requires only {expected_backend} deployments in the captured route"
        )
    expected_location = {"A": "remote", "B": "remote", "C": "local"}[lane_key]
    if str(runtime.get("execution_location") or "") != expected_location:
        raise ValueError(f"lane {lane_key} requires execution_location={expected_location}")
    runtime["acceptance_capture"] = {
        "lane": lane_key,
        "expected_runtime": dict(LANES)[lane_key],
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "captured_at": now_iso(),
    }
    out = Path(output_dir).expanduser().resolve()
    target = out / LANE_FILENAMES[lane_key]
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(target, runtime)
    return {
        "schema": "video_knowledge_pipeline.model_gateway_lane_capture.v1",
        "status": "captured" if write else "planned",
        "lane": lane_key,
        "expected_runtime": dict(LANES)[lane_key],
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "target_path": str(target),
        "route_id": str(runtime.get("route_id") or ""),
        "route_revision": str(runtime.get("route_revision") or ""),
        "execution_location": str(runtime.get("execution_location") or ""),
        "remote_requests_made": False,
        "updated_at": now_iso(),
    }

def build_temporal_gateway_acceptance_manifest(
    bundle_dir: str | Path,
    *,
    indexes: list[int],
    frame_count: int = 8,
    output_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    frames_root = root / "temporal-frames"
    groups: list[dict[str, Any]] = []
    for index in sorted({int(value) for value in indexes if int(value) > 0}):
        group_dir = frames_root / f"{index:04d}"
        frames = sorted(group_dir.glob("frame_*")) if group_dir.is_dir() else []
        records = [_file_record(path) for path in frames if path.is_file()]
        groups.append(
            {
                "index": index,
                "group_dir": str(group_dir),
                "frame_count": len(records),
                "expected_frame_count": int(frame_count),
                "ready": len(records) == int(frame_count),
                "frames": records,
            }
        )
    ready_count = sum(1 for group in groups if group["ready"])
    result = {
        "schema": TEMPORAL_MANIFEST_SCHEMA,
        "status": "ready" if groups and ready_count == len(groups) else "incomplete",
        "bundle_dir": str(root),
        "indexes": [group["index"] for group in groups],
        "group_count": len(groups),
        "ready_group_count": ready_count,
        "failed_group_count": len(groups) - ready_count,
        "groups": groups,
        "operator_boundary": {
            "local_inventory_only": True,
            "images_uploaded": False,
            "model_calls_made": 0,
            "separate_remote_consent_required": True,
            "not_an_automated_test_fixture": True,
        },
        "updated_at": now_iso(),
    }
    out = Path(output_dir).expanduser().resolve() if output_dir else root / "exports"
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "temporal-gateway-acceptance-manifest.json", result)
        (out / "temporal-gateway-acceptance-manifest.md").write_text(
            _render_temporal_markdown(result), encoding="utf-8"
        )
        result["artifacts"] = {
            "json": str(out / "temporal-gateway-acceptance-manifest.json"),
            "markdown": str(out / "temporal-gateway-acceptance-manifest.md"),
        }
    return result


def _lane_result(label: str, expected: str, path: Path) -> dict[str, Any]:
    try:
        raw = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "lane": label,
            "expected_runtime": expected,
            "result_path": str(path),
            "loaded": False,
            "error": str(exc),
            "task": "",
            "schema_contract": {"compatible": False, "missing_fields": sorted(REQUIRED_RUNTIME_FIELDS)},
            "quality_gate": "unavailable",
            "latency_ms": None,
            "call_count": 0,
            "estimated_cost": "unknown",
            "failure_recovery": "unavailable",
        }
    if not isinstance(raw, dict):
        raise ValueError(f"model result must be a JSON object: {path}")
    result = _unwrap_result(raw)
    missing = sorted(field for field in REQUIRED_RUNTIME_FIELDS if field not in result)
    content = result.get("content")
    content_bytes = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    quality = result.get("quality_gate")
    quality_gate = (
        str(quality.get("status") or quality.get("result") or "reported")
        if isinstance(quality, dict)
        else str(quality or "pending_human_review")
    )
    return {
        "lane": label,
        "expected_runtime": expected,
        "result_path": str(path),
        "result_sha256": _sha256(path),
        "loaded": True,
        "source_schema": str(result.get("schema") or raw.get("schema") or ""),
        "ok": bool(result.get("ok")),
        "status": str(result.get("status") or ""),
        "task": str(result.get("task") or raw.get("task") or ""),
        "execution_location": str(result.get("execution_location") or ""),
        "route_id": str(result.get("route_id") or ""),
        "route_revision": str(result.get("route_revision") or ""),
        "deployment": result.get("deployment") or {},
        "provider": result.get("provider") or {},
        "schema_contract": {
            "compatible": not missing,
            "missing_fields": missing,
            "expected_schema": RUNTIME_RESULT_SCHEMA,
        },
        "quality_gate": quality_gate,
        "latency_ms": _number_or_none(result.get("latency_ms")),
        "call_count": _call_count(result),
        "estimated_cost": result.get("estimated_cost") if result.get("estimated_cost") is not None else "unknown",
        "failure_recovery": result.get("failure_recovery") or result.get("recovery") or str(result.get("status") or "unknown"),
        "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
        "content_bytes": len(content_bytes),
        "evidence_count": len(result.get("evidence") or []) if isinstance(result.get("evidence"), list) else int(bool(result.get("evidence"))),
        "consent_id_present": bool(result.get("consent_id")),
    }


def _unwrap_result(raw: dict[str, Any]) -> dict[str, Any]:
    nested = raw.get("model_result")
    if isinstance(nested, dict) and (nested.get("schema") == RUNTIME_RESULT_SCHEMA or "route_revision" in nested):
        return nested
    return raw


def _call_count(result: dict[str, Any]) -> int:
    if result.get("call_count") is not None:
        try:
            return max(0, int(result["call_count"]))
        except (TypeError, ValueError):
            pass
    calls = result.get("calls")
    if isinstance(calls, list):
        return len(calls)
    return 1 if str(result.get("status") or "") not in {"", "planned"} else 0


def _number_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _render_comparison_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# VKP Model Gateway A/B/C Comparison",
        "",
        f"- Status: `{result['status']}`",
        f"- Sample: `{result.get('sample_id') or 'unspecified'}`",
        f"- Unified schema compatible: `{result['comparison']['schema_compatible']}`",
        "",
        "| Lane | Runtime | Status | Schema | Quality gate | Latency ms | Calls | Cost | Recovery |",
        "|---|---|---|---|---|---:|---:|---|---|",
    ]
    for row in result["lanes"]:
        lines.append(
            f"| {row['lane']} | {row['expected_runtime']} | {row.get('status') or 'unavailable'} | "
            f"{'pass' if row['schema_contract']['compatible'] else 'fail'} | {row['quality_gate']} | "
            f"{row['latency_ms'] if row['latency_ms'] is not None else 'unknown'} | {row['call_count']} | "
            f"{row['estimated_cost']} | {row['failure_recovery']} |"
        )
    lines.extend(
        [
            "",
            "This report compares saved results only. It performs no model call and does not promote candidate evidence.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_temporal_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Temporal Gateway Acceptance Manifest",
        "",
        f"- Status: `{result['status']}`",
        f"- Ready groups: `{result['ready_group_count']}/{result['group_count']}`",
        "- Model calls: `0`",
        "- Images uploaded: `false`",
        "",
        "| Index | Frames | Expected | Ready | Path |",
        "|---:|---:|---:|---|---|",
    ]
    for group in result["groups"]:
        lines.append(
            f"| {group['index']} | {group['frame_count']} | {group['expected_frame_count']} | "
            f"{str(group['ready']).lower()} | `{group['group_dir']}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build offline VKP model-gateway acceptance evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--c", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--sample-id", default="")
    temporal = sub.add_parser("temporal-manifest")
    temporal.add_argument("bundle_dir")
    temporal.add_argument("--indexes", required=True)
    temporal.add_argument("--frame-count", type=int, default=8)
    temporal.add_argument("--output-dir", default="")
    capture = sub.add_parser("capture-lane")
    capture.add_argument("--lane", required=True, choices=["A", "B", "C", "a", "b", "c"])
    capture.add_argument("--input", required=True)
    capture.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "compare":
        result = compare_model_gateway_results(
            args.a,
            args.b,
            args.c,
            output_dir=args.output_dir,
            sample_id=args.sample_id,
        )
    elif args.command == "capture-lane":
        result = capture_model_gateway_lane_result(
            args.lane,
            args.input,
            output_dir=args.output_dir,
        )
    else:
        indexes = [int(value.strip()) for value in str(args.indexes).split(",") if value.strip()]
        result = build_temporal_gateway_acceptance_manifest(
            args.bundle_dir,
            indexes=indexes,
            frame_count=args.frame_count,
            output_dir=args.output_dir or None,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"ready", "ready_for_review", "captured"} else 2


if __name__ == "__main__":
    raise SystemExit(main())