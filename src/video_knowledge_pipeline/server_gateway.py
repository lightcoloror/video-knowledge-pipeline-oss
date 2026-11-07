"""API-stage glue over the canonical shared gateway, never a provider client."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .canonical_json import canonical_json_sha256
from .model_provider_gateway_adapter import _shared


class ServerAdmissionRejected(RuntimeError):
    """The hosting workbench refused the call before provider execution."""


def bind_workbench_gateway(client, job, *, ledger_path, outbox, executor=None):
    """Reuse the workbench guard lazily; neither import nor binding makes a call."""
    from workbench.gateway_guard import ExecutionAdmissionError, execute_guarded_gateway

    def call(pack, context):
        try:
            return execute_guarded_gateway(client, job, context["stage"], pack["request"],
                profile=pack["profile"], route=pack["route"], consent=pack["consent"],
                ledger_path=ledger_path, outbox=outbox, executor=executor)
        except ExecutionAdmissionError as exc:
            raise ServerAdmissionRejected("workbench_call_not_admitted; reconcile request_id") from exc
    return call


@dataclass(frozen=True)
class ApiStageBackend:
    """Host-owned builders and guarded call; no Python import paths in job JSON.

    The workbench supplies a closure around execute_guarded_gateway. It owns
    budget, lease, owner consent and ledger/outbox. There is no default network
    executor here, and merely registering one does not assert safety_v1.
    """

    descriptor: dict[str, Any]
    prepare_request: Callable[[dict[str, Any]], dict[str, Any]]
    guarded_call: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

    @property
    def revision(self) -> str:
        return canonical_json_sha256(self.descriptor)

    def validate(self) -> None:
        if (self.descriptor.get("mode") != "api" or self.descriptor.get("location") != "remote"
                or self.descriptor.get("fallback") is not False
                or self.descriptor.get("capability") not in {"text", "vision"}
                or not all(self.descriptor.get(k) for k in ("provider_id", "model", "profile_revision", "route_revision"))
                or not callable(self.prepare_request) or not callable(self.guarded_call)):
            raise ValueError("API backend requires explicit remote shared-gateway text/vision route; no fallback")

    def prepare(self, context: dict[str, Any]) -> dict[str, Any]:
        self.validate()
        pack = self.prepare_request(context)
        if not isinstance(pack, dict) or set(pack) != {"request", "profile", "route", "consent"}:
            raise ValueError("gateway pack must contain request/profile/route/consent")
        _shared("audit").assert_secretless(pack)
        request = pack["request"]
        _shared("execution").validate_execution_request(
            request, profile=pack["profile"], route=pack["route"], consent=pack["consent"])
        expected = {"request_id": context["request_id"], "consumer_id": "vkp",
                    **{k: self.descriptor[k] for k in ("provider_id", "model", "capability", "profile_revision", "route_revision")}}
        if any(request.get(k) != v for k, v in expected.items()):
            raise ValueError("gateway request identity differs from pinned backend/job")
        if pack["profile"].get("retry_policy", {}).get("max_attempts") != 1 or pack["profile"].get("fallback_pool"):
            raise ValueError("server stages require one guarded call; no hidden retries/fallback")
        if pack["route"].get("fallback_policy", {}).get("enabled"):
            raise ValueError("server stages forbid implicit route fallback")
        return pack

    def validate_result(self, result: dict[str, Any], pack: dict[str, Any]) -> None:
        schema = json.loads(resources.files("model_provider_gateway.schemas").joinpath(
            "execution-result.v1.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(result)
        request = pack["request"]
        for key in ("request_id", "provider_id", "model", "capability", "consent_hash", "input_manifest_hash"):
            if result.get(key) != request.get(key):
                raise ValueError("gateway result binding mismatch: " + key)
        if result.get("route_hash") != request["route_revision"]:
            raise ValueError("gateway result route binding mismatch")
        if any(result["redacted_audit"].get(k) != request.get(k) for k in ("consumer_id", "task", "profile_id")):
            raise ValueError("gateway result consumer/profile audit binding mismatch")
        if result["redacted_audit"].get("automatic_fallback_used") is not False:
            raise ValueError("unexpected gateway fallback")

    def parse(self, result: dict[str, Any], pack: dict[str, Any], output_schema: dict[str, Any]) -> dict[str, Any]:
        self.validate_result(result, pack)
        if not result["ok"] or result["status"] != "completed":
            raise ValueError("gateway did not complete the stage")
        value = json.loads(result["output"]["text"])
        Draft202012Validator(output_schema).validate(value)
        if not isinstance(value, dict):
            raise ValueError("stage output must be an object")
        return value
