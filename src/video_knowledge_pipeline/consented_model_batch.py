from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import urllib.parse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Callable

from .file_hash import sha256_file as _file_sha256
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .time_utils import (
    parse_utc_datetime_or_none,
    utc_now_iso_seconds as _now_iso,
)
from .trusted_model_connector import (
    compact_model_execution_receipt,
    execute_consented_model_task,
)
from .trusted_model_connector_policy import TrustedModelConnectorPolicy

SCHEMA = "video_knowledge_pipeline.consented_model_batch.v1"
BATCH_LIST_SCHEMA = "video_knowledge_pipeline.consented_model_batch_list.v1"
DEFAULT_BATCH_DIRNAME = "model-connector-batches"
ACTIVE_STATES = frozenset({"accepted", "running"})

Executor = Callable[..., dict[str, Any]]



def _destination_key(consent: dict[str, Any]) -> str:
    destinations = {
        str(value or "").strip()
        for value in consent.get("authorized_destinations") or []
        if str(value or "").strip()
    }
    if not destinations:
        for deployment in consent.get("authorized_deployments") or []:
            if not isinstance(deployment, dict):
                continue
            base_url = str(deployment.get("base_url") or "").strip()
            parsed = urllib.parse.urlsplit(base_url)
            if parsed.scheme and parsed.netloc:
                destinations.add(f"{parsed.scheme}://{parsed.netloc}")
    if len(destinations) != 1:
        raise ValueError(
            "parallel batch execution requires each consent to lock exactly one destination"
        )
    return next(iter(destinations))


def _route_revision(consent: dict[str, Any]) -> str:
    route = consent.get("route") if isinstance(consent.get("route"), dict) else {}
    revision = str(route.get("route_revision") or "").strip()
    if not revision:
        raise ValueError("consent route_revision is missing")
    return revision


def _error_text(result: dict[str, Any]) -> str:
    values: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).casefold() in {"error", "message", "detail", "status"}:
                    if isinstance(nested, (str, int, float)):
                        values.append(str(nested))
                elif isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(result)
    return " | ".join(values)[:2000]


def classify_execution_result(result: dict[str, Any]) -> str:
    if bool(result.get("ok")):
        return "success"
    text = _error_text(result).casefold()
    if any(
        marker in text
        for marker in (
            "http 429",
            "status 429",
            "rate_limit",
            "rate limit",
            "too many requests",
        )
    ):
        return "rate_limited"
    if any(
        marker in text
        for marker in (
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "status 500",
            "status 502",
            "status 503",
            "status 504",
            "timeout",
            "timed out",
            "server disconnected",
            "provider_unavailable",
            "provider_exception",
            "connection reset",
        )
    ):
        return "transient_provider_failure"
    return "permanent_failure"


def _normalise_node_ids(value: Any, *, item_count: int) -> list[str]:
    if value is None:
        return [f"item-{index + 1:04d}" for index in range(item_count)]
    if not isinstance(value, list) or len(value) != item_count:
        raise ValueError("node_ids must be a list aligned with consent_paths")
    rows: list[str] = []
    for raw in value:
        node_id = str(raw or "").strip()
        if (
            not node_id
            or len(node_id) > 128
            or Path(node_id).name != node_id
            or "/" in node_id
            or "\\" in node_id
        ):
            raise ValueError("node_ids must be non-empty path-free identifiers")
        if node_id in rows:
            raise ValueError("node_ids must be unique")
        rows.append(node_id)
    return rows


def _normalise_dependencies(value: Any, *, item_count: int) -> list[list[int]]:
    """Validate a dependency graph with Python's reusable graphlib primitive."""

    if value is None:
        dependencies = [[] for _ in range(item_count)]
    elif not isinstance(value, list) or len(value) != item_count:
        raise ValueError("depends_on must be a list aligned with consent_paths")
    else:
        dependencies = []
        for index, raw in enumerate(value):
            if not isinstance(raw, list):
                raise ValueError("each depends_on item must be a list of item indexes")
            row: list[int] = []
            for dependency in raw:
                if isinstance(dependency, bool) or not isinstance(dependency, int):
                    raise ValueError("dependency indexes must be integers")
                if dependency < 0 or dependency >= item_count or dependency == index:
                    raise ValueError(
                        "dependency index is out of range or self-referential"
                    )
                if dependency not in row:
                    row.append(dependency)
            dependencies.append(sorted(row))
    try:
        TopologicalSorter(
            {index: set(row) for index, row in enumerate(dependencies)}
        ).prepare()
    except CycleError as exc:
        raise ValueError("depends_on contains a cycle") from exc
    return dependencies


def _safe_bundle_dir(project_root: Path, value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        path = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return ""
    if path == project_root or path.is_relative_to(project_root):
        return str(path)
    return ""


def _redacted_consent_allowance(
    project_root: Path, items: list[dict[str, Any]]
) -> dict[str, Any]:
    remaining_calls = 0
    remaining_cost = 0.0
    known_calls = 0
    known_cost = 0
    expiries: list[str] = []
    for item in items:
        raw = str(item.get("consent_path") or "").strip()
        if not raw:
            continue
        try:
            path = Path(raw).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if not (path == project_root or path.is_relative_to(project_root)):
            continue
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        try:
            max_calls = int(scope.get("max_calls"))
            calls_attempted = int(usage.get("calls_attempted") or 0)
        except (TypeError, ValueError):
            pass
        else:
            remaining_calls += max(0, max_calls - calls_attempted)
            known_calls += 1
        try:
            max_cost = float(scope.get("max_estimated_cost_usd"))
            committed_cost = float(usage.get("cost_committed_usd") or 0)
        except (TypeError, ValueError):
            pass
        else:
            remaining_cost += max(0.0, max_cost - committed_cost)
            known_cost += 1
        expires_at = str(payload.get("expires_at") or "").strip()
        if expires_at:
            expiries.append(expires_at)
    return {
        "consent_count": len(items),
        "known_call_allowances": known_calls,
        "remaining_calls": remaining_calls if known_calls else None,
        "known_cost_allowances": known_cost,
        "remaining_estimated_cost_usd": round(remaining_cost, 6)
        if known_cost
        else None,
        "earliest_expiry": min(expiries) if expiries else "",
    }


def list_consented_model_batches(
    project_root: str | Path, *, limit: int = 50
) -> dict[str, Any]:
    """Return redacted persisted batch status without constructing a manager."""
    bounded_limit = max(1, min(int(limit), 200))
    batch_root = (
        Path(project_root).expanduser().resolve()
        / ".local"
        / DEFAULT_BATCH_DIRNAME
    )
    project_path = Path(project_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    if batch_root.is_dir():
        for status_path in batch_root.glob("*/batch-execution.json"):
            try:
                payload = read_json(status_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
                continue
            items = payload.get("items") if isinstance(payload.get("items"), list) else []
            settings = (
                payload.get("settings")
                if isinstance(payload.get("settings"), dict)
                else {}
            )
            persisted_summary = (
                payload.get("summary")
                if isinstance(payload.get("summary"), dict)
                else {}
            )
            summary = dict(persisted_summary)
            if items:
                live_summary = ConsentedModelBatchManager._summary(items)
                if not summary:
                    summary = live_summary
                else:
                    summary["heartbeat_alive"] = live_summary[
                        "heartbeat_alive"
                    ]
                    summary["heartbeat_stale"] = live_summary[
                        "heartbeat_stale"
                    ]
            controllers = (
                payload.get("destination_controllers")
                if isinstance(payload.get("destination_controllers"), dict)
                else {}
            )
            safe_controllers: dict[str, dict[str, Any]] = {}
            for destination, controller in controllers.items():
                if not isinstance(controller, dict):
                    continue
                safe_controllers[str(destination)] = {
                    "configured_batch_hard_limit": controller.get(
                        "configured_batch_hard_limit"
                    ),
                    "provider_rate_limit_owner": str(
                        controller.get("provider_rate_limit_owner") or "litellm_proxy"
                    ),
                    "adaptive_limit_changed_by_batch": bool(
                        controller.get("adaptive_limit_changed_by_batch")
                    ),
                    "completed_calls": int(controller.get("completed_calls") or 0),
                    "outcomes": {
                        str(key): int(value or 0)
                        for key, value in dict(controller.get("outcomes") or {}).items()
                    },
                }
            rows.append(
                {
                    "job_id": str(payload.get("job_id") or status_path.parent.name),
                    "status": str(payload.get("status") or "unknown"),
                    "bundle_dir": _safe_bundle_dir(project_path, payload.get("bundle_dir")),
                    "consent_allowance": _redacted_consent_allowance(project_path, items),
                    "terminal": bool(payload.get("terminal")),
                    "submitted_at": str(payload.get("submitted_at") or ""),
                    "updated_at": str(payload.get("updated_at") or ""),
                    "completed_at": str(payload.get("completed_at") or ""),
                    "status_path": str(status_path.resolve()),
                    "summary": {
                        str(key): value
                        for key, value in summary.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    },
                    "tasks": sorted(
                        {
                            str(item.get("task") or "")
                            for item in items
                            if isinstance(item, dict) and item.get("task")
                        }
                    ),
                    "nodes": [
                        str(item.get("node_id") or f"item-{index + 1:04d}")
                        for index, item in enumerate(items)
                        if isinstance(item, dict)
                    ],
                    "destinations": sorted(
                        {
                            str(item.get("destination") or "")
                            for item in items
                            if isinstance(item, dict) and item.get("destination")
                        }
                    ),
                    "settings": {
                        "max_parallel_global": settings.get("max_parallel_global"),
                        "max_parallel_per_destination": settings.get(
                            "max_parallel_per_destination"
                        ),
                        "provider_rate_limit_owner": str(
                            settings.get("provider_rate_limit_owner") or "litellm_proxy"
                        ),
                        "dependency_engine": str(
                            settings.get("dependency_engine")
                            or "python.graphlib.TopologicalSorter"
                        ),
                        "heartbeat_interval_seconds": settings.get(
                            "heartbeat_interval_seconds"
                        ),
                        "heartbeat_stale_after_seconds": settings.get(
                            "heartbeat_stale_after_seconds"
                        ),
                        "retries_added_by_batch": int(
                            settings.get("retries_added_by_batch") or 0
                        ),
                        "automatic_fallback_added_by_batch": bool(
                            settings.get("automatic_fallback_added_by_batch")
                        ),
                    },
                    "destination_controllers": safe_controllers,
                }
            )
    rows.sort(
        key=lambda row: (str(row.get("updated_at") or ""), str(row["job_id"])),
        reverse=True,
    )
    return {
        "ok": True,
        "schema": BATCH_LIST_SCHEMA,
        "count": min(len(rows), bounded_limit),
        "items": rows[:bounded_limit],
        "provider_rate_limit_owner": "litellm_proxy",
        "provider_rate_limit_fields": ["rpm", "tpm", "max_parallel_requests"],
        "batch_adaptive_limiter_enabled": False,
        "secrets_exposed": False,
        "artifact_contents_exposed": False,
        "consent_paths_exposed": False,
    }


class ConsentedModelBatchManager:
    """Crash-visible background execution over the existing consent executor."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        policy: TrustedModelConnectorPolicy,
        executor: Executor = execute_consented_model_task,
        global_parallel_limit: int = 4,
        maximum_parallel_per_destination: int = 2,
        heartbeat_interval_seconds: float = 15.0,
        heartbeat_stale_multiplier: float = 3.0,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.batch_root = (
            self.project_root / ".local" / DEFAULT_BATCH_DIRNAME
        ).resolve()
        if not any(
            self.batch_root == root or self.batch_root.is_relative_to(root)
            for root in policy.allowed_roots
        ):
            raise ValueError("batch artifact root is outside the connector allowed roots")
        self.batch_root.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.executor = executor
        self.global_parallel_limit = max(1, int(global_parallel_limit))
        self.maximum_parallel_per_destination = max(
            1, int(maximum_parallel_per_destination)
        )
        self.heartbeat_interval_seconds = max(
            0.01, float(heartbeat_interval_seconds)
        )
        self.heartbeat_stale_multiplier = max(
            2.0, float(heartbeat_stale_multiplier)
        )
        self._global_gate = threading.BoundedSemaphore(self.global_parallel_limit)
        self._coordinator = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="vkp-consented-batch"
        )
        self._lock = threading.RLock()
        self._futures: dict[str, Future[dict[str, Any]]] = {}

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "batch_root": str(self.batch_root),
            "global_parallel_limit": self.global_parallel_limit,
            "maximum_parallel_per_destination": self.maximum_parallel_per_destination,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "heartbeat_stale_after_seconds": round(
                self.heartbeat_interval_seconds * self.heartbeat_stale_multiplier,
                3,
            ),
            "retries_added_by_batch": 0,
            "automatic_fallback_added_by_batch": False,
            "batch_adaptive_limiter_enabled": False,
            "provider_rate_limit_owner": "litellm_proxy",
            "provider_rate_limit_fields": ["rpm", "tpm", "max_parallel_requests"],
            "dependency_engine": "python.graphlib.TopologicalSorter",
        }

    def submit_workflow(
        self,
        *,
        bundle_dir: str,
        nodes: list[dict[str, Any]],
        write: bool = True,
        max_parallel_global: int | None = None,
        max_parallel_per_destination: int | None = None,
    ) -> dict[str, Any]:
        """Map named production nodes onto the existing consent batch DAG."""
        if not isinstance(nodes, list) or not nodes or len(nodes) > 64:
            raise ValueError("nodes must contain between 1 and 64 workflow items")
        node_ids = _normalise_node_ids(
            [node.get("id") if isinstance(node, dict) else "" for node in nodes],
            item_count=len(nodes),
        )
        index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
        consent_paths: list[str] = []
        depends_on: list[list[int]] = []
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError("each workflow node must be an object")
            consent_path = str(node.get("consent_path") or "").strip()
            if not consent_path:
                raise ValueError("each workflow node must provide consent_path")
            consent_paths.append(consent_path)
            raw_dependencies = node.get("depends_on") or []
            if not isinstance(raw_dependencies, list):
                raise ValueError("workflow depends_on must be a list of node ids")
            indexes: list[int] = []
            for raw_dependency in raw_dependencies:
                dependency_id = str(raw_dependency or "").strip()
                if dependency_id not in index_by_id:
                    raise ValueError(
                        f"workflow dependency does not exist: {dependency_id}"
                    )
                indexes.append(index_by_id[dependency_id])
            depends_on.append(indexes)
        return self.submit(
            consent_paths,
            write=write,
            max_parallel_global=max_parallel_global,
            max_parallel_per_destination=max_parallel_per_destination,
            depends_on=depends_on,
            node_ids=node_ids,
            bundle_dir=bundle_dir,
        )

    def submit(
        self,
        consent_paths: list[str],
        *,
        write: bool = True,
        max_parallel_global: int | None = None,
        max_parallel_per_destination: int | None = None,
        depends_on: list[list[int]] | None = None,
        node_ids: list[str] | None = None,
        bundle_dir: str = "",
    ) -> dict[str, Any]:
        if not consent_paths or len(consent_paths) > 64:
            raise ValueError("consent_paths must contain between 1 and 64 items")
        resolved = [
            self.policy.require_path(value, label="consent_path")
            for value in consent_paths
        ]
        if len({str(path).casefold() for path in resolved}) != len(resolved):
            raise ValueError("consent_paths must not contain duplicates")
        normalized_node_ids = _normalise_node_ids(node_ids, item_count=len(resolved))
        dependencies = _normalise_dependencies(
            depends_on, item_count=len(resolved)
        )
        bundle_path: Path | None = None
        if str(bundle_dir or "").strip():
            bundle_manifest = self.policy.require_path(
                Path(bundle_dir) / "manifest.json", label="bundle manifest"
            )
            self.policy.require_path(
                bundle_manifest.parent / "timeline.json", label="bundle timeline"
            )
            bundle_path = bundle_manifest.parent
        requested_global = (
            self.global_parallel_limit
            if max_parallel_global is None
            else max(1, int(max_parallel_global))
        )
        requested_destination = (
            self.maximum_parallel_per_destination
            if max_parallel_per_destination is None
            else max(1, int(max_parallel_per_destination))
        )
        if requested_global > self.global_parallel_limit:
            raise ValueError("requested global concurrency exceeds the Broker limit")
        if requested_destination > self.maximum_parallel_per_destination:
            raise ValueError("requested destination concurrency exceeds the Broker limit")

        items: list[dict[str, Any]] = []
        identity_rows: list[dict[str, Any]] = []
        for index, path in enumerate(resolved):
            consent = self.policy.require_consent_scope(
                path, require_execution_contract=True
            )
            destination = _destination_key(consent)
            revision = _route_revision(consent)
            consent_sha256 = _file_sha256(path)
            row = {
                "index": index,
                "node_id": normalized_node_ids[index],
                "consent_path": str(path),
                "consent_sha256": consent_sha256,
                "consent_id": str(consent.get("consent_id") or ""),
                "task": str(consent.get("task") or ""),
                "destination": destination,
                "route_revision": revision,
                "depends_on": dependencies[index],
                "state": "queued",
                "outcome": "pending",
                "started_at": "",
                "completed_at": "",
                "latency_ms": None,
                "execution_status": "",
                "execution_report": "",
                "heartbeat_at": "",
                "heartbeat_at_unix_ms": None,
                "heartbeat_count": 0,
                "heartbeat_state": "not_started",
                "heartbeat_stale_after_seconds": round(
                    self.heartbeat_interval_seconds
                    * self.heartbeat_stale_multiplier,
                    3,
                ),
                "error": "",
            }
            items.append(row)
            identity_rows.append(
                {
                    "consent_path": str(path),
                    "node_id": normalized_node_ids[index],
                    "consent_id": str(consent.get("consent_id") or ""),
                    "upload_manifest_sha256": str(
                        (consent.get("upload_manifest") or {}).get("manifest_sha256")
                        or ""
                    ),
                    "route_revision": revision,
                    "depends_on": dependencies[index],
                    "write": bool(write),
                }
            )
        identity_rows.append({"bundle_dir": str(bundle_path or "")})
        identity = json.dumps(identity_rows, ensure_ascii=False, sort_keys=True)
        job_id = f"model_batch_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
        status_path = self._status_path(job_id)

        with self._lock:
            existing = self._read_existing(job_id)
            if existing is not None:
                return self._submission_result(existing, existing=True)
            submitted_at = _now_iso()
            payload = {
                "schema": SCHEMA,
                "job_id": job_id,
                "status": "accepted",
                "terminal": False,
                "submitted_at": submitted_at,
                "started_at": "",
                "completed_at": "",
                "bundle_dir": str(bundle_path or ""),
                "updated_at": submitted_at,
                "status_path": str(status_path),
                "settings": {
                    "write_execution_reports": bool(write),
                    "max_parallel_global": requested_global,
                    "max_parallel_per_destination": requested_destination,
                    "broker_global_parallel_limit": self.global_parallel_limit,
                    "broker_maximum_parallel_per_destination": self.maximum_parallel_per_destination,
                    "retries_added_by_batch": 0,
                    "automatic_fallback_added_by_batch": False,
                    "batch_adaptive_limiter_enabled": False,
                    "provider_rate_limit_owner": "litellm_proxy",
                    "dependency_engine": "python.graphlib.TopologicalSorter",
                    "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
                    "heartbeat_stale_after_seconds": round(
                        self.heartbeat_interval_seconds
                        * self.heartbeat_stale_multiplier,
                        3,
                    ),
                },
                "summary": self._summary(items),
                "destination_controllers": {},
                "events": [
                    {
                        "at": submitted_at,
                        "type": "batch_accepted",
                        "message": "Validated consent-locked batch accepted for background execution",
                    }
                ],
                "items": items,
            }
            self._write_payload(payload)
            self._sync_bundle_run(payload)
            future = self._coordinator.submit(self._run_job, job_id)
            self._futures[job_id] = future
            return self._submission_result(payload, existing=False)

    def status(self, job_id: str) -> dict[str, Any]:
        clean_id = str(job_id or "").strip()
        if not clean_id.startswith("model_batch_") or Path(clean_id).name != clean_id:
            raise ValueError("invalid batch job_id")
        with self._lock:
            payload = self._read_payload(clean_id)
            future = self._futures.get(clean_id)
            if payload.get("status") in ACTIVE_STATES and future is None:
                payload = self._mark_interrupted(payload)
            elif future is not None and future.done() and future.exception() is not None:
                payload = self._mark_interrupted(
                    payload,
                    error=f"{type(future.exception()).__name__}: {future.exception()}",
                )
            payload["summary"] = self._summary(payload.get("items") or [])
            return payload

    def wait(self, job_id: str, timeout: float = 10.0) -> dict[str, Any]:
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            return self.status(job_id)
        future.result(timeout=timeout)
        return self.status(job_id)

    def _run_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            payload = self._read_payload(job_id)
            payload["status"] = "running"
            payload["started_at"] = _now_iso()
            payload["updated_at"] = payload["started_at"]
            payload["events"].append(
                {
                    "at": payload["started_at"],
                    "type": "batch_started",
                    "message": "Background workers started",
                }
            )
            self._write_payload(payload)
        settings = payload["settings"]
        graph = TopologicalSorter(
            {
                int(item["index"]): set(item.get("depends_on") or [])
                for item in payload["items"]
            }
        )
        graph.prepare()
        local_destination_gates: dict[str, threading.BoundedSemaphore] = {}
        for item in payload["items"]:
            local_destination_gates.setdefault(
                item["destination"],
                threading.BoundedSemaphore(settings["max_parallel_per_destination"]),
            )
        workers = min(settings["max_parallel_global"], len(payload["items"]))
        with ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix=f"vkp-{job_id[-8:]}"
        ) as pool:
            running: dict[Future[None], int] = {}
            while graph.is_active():
                for index in graph.get_ready():
                    with self._lock:
                        current = self._read_payload(job_id)
                    item = current["items"][index]
                    failed_dependencies = [
                        dependency
                        for dependency in item.get("depends_on") or []
                        if current["items"][dependency].get("state") != "completed"
                    ]
                    if failed_dependencies:
                        self._update_item(
                            job_id,
                            index,
                            {
                                "state": "dependency_blocked",
                                "outcome": "dependency_blocked",
                                "completed_at": _now_iso(),
                                "execution_status": "dependency_blocked",
                                "error": "upstream items did not complete: "
                                + ", ".join(
                                    str(value) for value in failed_dependencies
                                ),
                            },
                        )
                        graph.done(index)
                        continue
                    running[
                        pool.submit(
                            self._execute_item,
                            job_id,
                            index,
                            local_destination_gates[item["destination"]],
                        )
                    ] = index
                if not running:
                    continue
                completed, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in completed:
                    index = running.pop(future)
                    failure = future.exception()
                    if failure is not None:
                        self._update_item(
                            job_id,
                            index,
                            {
                                "state": "failed",
                                "outcome": "batch_internal_failure",
                                "completed_at": _now_iso(),
                                "execution_status": "batch_internal_failure",
                                "error": f"{type(failure).__name__}: {failure}",
                            },
                        )
                    graph.done(index)
        with self._lock:
            payload = self._read_payload(job_id)
            summary = self._summary(payload["items"])
            payload["summary"] = summary
            unresolved = sum(
                summary[key]
                for key in ("queued", "running", "unknown_after_restart")
            )
            failures = summary["failed"] + summary["dependency_blocked"]
            if unresolved:
                terminal_status = "failed" if summary["completed"] == 0 else "degraded"
            elif failures == 0:
                terminal_status = "completed"
            elif summary["completed"] == 0:
                terminal_status = "failed"
            else:
                terminal_status = "degraded"
            payload["status"] = terminal_status
            payload["terminal"] = True
            payload["completed_at"] = _now_iso()
            payload["updated_at"] = payload["completed_at"]
            payload["events"].append(
                {
                    "at": payload["completed_at"],
                    "type": "batch_terminal",
                    "message": f"Batch reached terminal state: {terminal_status}",
                }
            )
            self._write_payload(payload)
            self._sync_bundle_run(payload)
        return payload

    def _execute_item(
        self,
        job_id: str,
        index: int,
        local_destination_gate: threading.BoundedSemaphore,
    ) -> None:
        with self._lock:
            payload = self._read_payload(job_id)
        item = payload["items"][index]
        local_destination_gate.acquire()
        self._global_gate.acquire()
        started_monotonic = time.monotonic()
        started_at = _now_iso()
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_item,
            args=(job_id, index, heartbeat_stop),
            name=f"vkp-heartbeat-{job_id[-8:]}-{index}",
            daemon=True,
        )
        heartbeat_started = False
        try:
            self._update_item(
                job_id,
                index,
                {
                    "state": "running",
                    "started_at": started_at,
                    "heartbeat_at": started_at,
                    "heartbeat_at_unix_ms": int(time.time() * 1000),
                    "heartbeat_count": 0,
                    "heartbeat_state": "active",
                },
            )
            heartbeat_thread.start()
            heartbeat_started = True
            try:
                result = self.executor(
                    item["consent_path"],
                    expected_route_revision=item["route_revision"],
                    write=bool(payload["settings"]["write_execution_reports"]),
                )
                if not isinstance(result, dict):
                    result = {
                        "ok": False,
                        "status": "invalid_executor_result",
                        "error": "executor returned a non-object result",
                    }
            except Exception as exc:  # Keep the rest of the batch running.
                result = {
                    "ok": False,
                    "status": "executor_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            outcome = classify_execution_result(result)
            completed_at = _now_iso()
            artifacts = (
                result.get("artifacts")
                if isinstance(result.get("artifacts"), dict)
                else {}
            )
            compact_receipt = compact_model_execution_receipt(result)
            self._update_item(
                job_id,
                index,
                {
                    "state": "completed" if outcome == "success" else "failed",
                    "outcome": outcome,
                    "completed_at": completed_at,
                    "latency_ms": round(
                        (time.monotonic() - started_monotonic) * 1000, 3
                    ),
                    "execution_status": str(result.get("status") or ""),
                    "execution_report": str(artifacts.get("execution_report") or ""),
                    "network_accounting": dict(
                        compact_receipt.get("network_accounting") or {}
                    ),
                    "usage": dict(compact_receipt.get("usage") or {}),
                    "cost_control": dict(
                        compact_receipt.get("cost_control") or {}
                    ),
                    "heartbeat_state": "stopped",
                    "heartbeat_stopped_at": completed_at,
                    "error": "" if outcome == "success" else _error_text(result),
                },
            )
        finally:
            heartbeat_stop.set()
            if heartbeat_started:
                heartbeat_thread.join(timeout=1.0)
            self._global_gate.release()
            local_destination_gate.release()
            self._record_destination_outcome(
                job_id,
                item["destination"],
                outcome if "outcome" in locals() else "transient_provider_failure",
            )

    def _heartbeat_item(
        self, job_id: str, index: int, stop: threading.Event
    ) -> None:
        while not stop.wait(self.heartbeat_interval_seconds):
            with self._lock:
                payload = self._read_payload(job_id)
                item = payload["items"][index]
                if item.get("state") != "running":
                    return
                heartbeat_at = _now_iso()
                item["heartbeat_at"] = heartbeat_at
                item["heartbeat_at_unix_ms"] = int(time.time() * 1000)
                item["heartbeat_count"] = int(item.get("heartbeat_count") or 0) + 1
                item["heartbeat_state"] = "active"
                payload["summary"] = self._summary(payload["items"])
                payload["updated_at"] = heartbeat_at
                self._write_payload(payload)

    def _update_item(self, job_id: str, index: int, values: dict[str, Any]) -> None:
        with self._lock:
            payload = self._read_payload(job_id)
            payload["items"][index].update(values)
            payload["summary"] = self._summary(payload["items"])
            payload["updated_at"] = _now_iso()
            self._write_payload(payload)

    def _record_destination_outcome(
        self, job_id: str, destination: str, outcome: str
    ) -> None:
        with self._lock:
            payload = self._read_payload(job_id)
            controller = dict(
                payload["destination_controllers"].get(destination) or {}
            )
            controller.setdefault(
                "configured_batch_hard_limit",
                int(payload["settings"]["max_parallel_per_destination"]),
            )
            controller["provider_rate_limit_owner"] = "litellm_proxy"
            controller["adaptive_limit_changed_by_batch"] = False
            controller["completed_calls"] = int(
                controller.get("completed_calls") or 0
            ) + 1
            outcomes = dict(controller.get("outcomes") or {})
            outcomes[outcome] = int(outcomes.get(outcome) or 0) + 1
            controller["outcomes"] = outcomes
            payload["destination_controllers"][destination] = controller
            payload["updated_at"] = _now_iso()
            self._write_payload(payload)

    def _read_existing(self, job_id: str) -> dict[str, Any] | None:
        path = self._status_path(job_id)
        if not path.is_file():
            return None
        payload = self._read_payload(job_id)
        future = self._futures.get(job_id)
        if payload.get("status") in ACTIVE_STATES and future is None:
            return self._mark_interrupted(payload)
        return payload

    def _mark_interrupted(
        self, payload: dict[str, Any], *, error: str = ""
    ) -> dict[str, Any]:
        if payload.get("status") not in ACTIVE_STATES:
            return payload
        for item in payload.get("items") or []:
            if item.get("state") in {"queued", "running"}:
                item["state"] = "unknown_after_restart"
                item["outcome"] = "not_replayed"
                item["heartbeat_state"] = "interrupted"
                item["heartbeat_stopped_at"] = _now_iso()
        payload["status"] = "interrupted"
        payload["terminal"] = True
        payload["completed_at"] = _now_iso()
        payload["updated_at"] = payload["completed_at"]
        payload["summary"] = self._summary(payload.get("items") or [])
        payload["resume_requires_explicit_action"] = True
        payload["events"].append(
            {
                "at": payload["completed_at"],
                "type": "batch_interrupted",
                "message": error
                or "Broker restarted before a terminal artifact was written; no item was replayed",
            }
        )
        self._write_payload(payload)
        self._sync_bundle_run(payload)
        return payload

    def _submission_result(
        self, payload: dict[str, Any], *, existing: bool
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "job_id": payload["job_id"],
            "status": "existing_result" if existing else "accepted",
            "batch_status": payload["status"],
            "terminal": bool(payload.get("terminal")),
            "status_path": payload["status_path"],
            "summary": payload["summary"],
            "bundle_dir": str(payload.get("bundle_dir") or ""),
            "background_execution_started": not existing,
            "retries_added_by_batch": 0,
            "automatic_fallback_added_by_batch": False,
        }

    def _status_path(self, job_id: str) -> Path:
        return self.batch_root / job_id / "batch-execution.json"

    def _read_payload(self, job_id: str) -> dict[str, Any]:
        path = self._status_path(job_id)
        if not path.is_file():
            raise FileNotFoundError(f"batch job does not exist: {job_id}")
        payload = read_json(path)
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError(f"invalid batch artifact: {path}")
        return payload

    def _write_payload(self, payload: dict[str, Any]) -> None:
        write_json(self._status_path(str(payload["job_id"])), payload)

    def _sync_bundle_run(self, payload: dict[str, Any]) -> None:
        bundle_dir = str(payload.get("bundle_dir") or "").strip()
        if not bundle_dir:
            return
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        status = str(payload.get("status") or "accepted")
        run_status = "running" if status in ACTIVE_STATES else status
        failed_items = [
            {
                "id": str(item.get("node_id") or item.get("index") or ""),
                "reason": str(item.get("outcome") or item.get("state") or "failed"),
                "detail": str(item.get("error") or ""),
            }
            for item in payload.get("items") or []
            if isinstance(item, dict)
            and item.get("state")
            in {"failed", "dependency_blocked", "unknown_after_restart"}
        ]
        try:
            register_bundle_run(
                bundle_dir,
                run_type="consented_model_batch",
                run_id=str(payload.get("job_id") or "consented-model-batch"),
                status=run_status,
                title="Consent-locked online model batch",
                summary=(
                    f"{int(summary.get('completed') or 0)}/{int(summary.get('total') or 0)} "
                    f"items completed; provider throttling is owned by LiteLLM."
                ),
                inputs={"job_id": str(payload.get("job_id") or "")},
                parameters={
                    "tasks": sorted(
                        {
                            str(item.get("task") or "")
                            for item in payload.get("items") or []
                            if isinstance(item, dict) and item.get("task")
                        }
                    ),
                    "provider_rate_limit_owner": "litellm_proxy",
                    "dependency_engine": "python.graphlib.TopologicalSorter",
                },
                failed_items=failed_items,
                next_actions=["Refresh Task Console to inspect the redacted batch status."],
                operator_boundary={
                    "consent_v2_required": True,
                    "automatic_retry": False,
                    "automatic_fallback": False,
                    "model_content_copied_to_registry": False,
                },
                resource_requirements={"network": 1},
                write=True,
            )
        except Exception as exc:
            payload.setdefault("events", []).append(
                {
                    "at": _now_iso(),
                    "type": "run_registry_sync_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            self._write_payload(payload)

    @staticmethod
    def _summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        latencies = sorted(
            float(row["latency_ms"])
            for row in items
            if isinstance(row.get("latency_ms"), (int, float))
            and not isinstance(row.get("latency_ms"), bool)
        )

        def percentile(q: float) -> float | None:
            if not latencies:
                return None
            position = (len(latencies) - 1) * q
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return round(latencies[lower], 3)
            fraction = position - lower
            return round(
                latencies[lower]
                + (latencies[upper] - latencies[lower]) * fraction,
                3,
            )

        now_unix_ms = int(time.time() * 1000)

        def heartbeat_is_stale(row: dict[str, Any]) -> bool:
            if row.get("state") != "running":
                return False
            try:
                heartbeat_unix_ms = int(row.get("heartbeat_at_unix_ms"))
            except (TypeError, ValueError):
                parsed = parse_utc_datetime_or_none(
                    row.get("heartbeat_at") or row.get("started_at")
                )
                heartbeat_unix_ms = (
                    int(parsed.timestamp() * 1000) if parsed is not None else 0
                )
            try:
                stale_after_ms = int(
                    float(row.get("heartbeat_stale_after_seconds") or 45.0)
                    * 1000
                )
            except (TypeError, ValueError):
                stale_after_ms = 45_000
            return (
                heartbeat_unix_ms <= 0
                or now_unix_ms - heartbeat_unix_ms > stale_after_ms
            )

        running_rows = [row for row in items if row.get("state") == "running"]
        stale_heartbeat_count = sum(
            1 for row in running_rows if heartbeat_is_stale(row)
        )
        network_rows = [
            row.get("network_accounting")
            for row in items
            if isinstance(row.get("network_accounting"), dict)
        ]
        return {
            "total": len(items),
            "queued": sum(1 for row in items if row.get("state") == "queued"),
            "running": len(running_rows),
            "heartbeat_alive": len(running_rows) - stale_heartbeat_count,
            "heartbeat_stale": stale_heartbeat_count,
            "completed": sum(1 for row in items if row.get("state") == "completed"),
            "failed": sum(1 for row in items if row.get("state") == "failed"),
            "dependency_blocked": sum(
                1 for row in items if row.get("state") == "dependency_blocked"
            ),
            "rate_limited": sum(
                1 for row in items if row.get("outcome") == "rate_limited"
            ),
            "transient_provider_failure": sum(
                1
                for row in items
                if row.get("outcome") == "transient_provider_failure"
            ),
            "unknown_after_restart": sum(
                1 for row in items if row.get("state") == "unknown_after_restart"
            ),
            "latency_sample_count": len(latencies),
            "latency_p50_ms": percentile(0.50),
            "latency_p95_ms": percentile(0.95),
            "gateway_request_bytes": sum(
                int(row.get("gateway_request_bytes") or 0)
                for row in network_rows
            ),
            "gateway_response_bytes": sum(
                int(row.get("gateway_response_bytes") or 0)
                for row in network_rows
            ),
            "source_artifact_bytes": sum(
                int(row.get("source_artifact_bytes") or 0)
                for row in network_rows
            ),
        }
