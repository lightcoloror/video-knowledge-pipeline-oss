from __future__ import annotations

from video_knowledge_pipeline.consented_model_batch import ConsentedModelBatchManager


def test_batch_summary_reports_latency_and_network_accounting() -> None:
    """Batch status exposes stable aggregates without another telemetry stack.

    Intent: make slow-but-completing batches and network-heavy batches visible.
    Decision: aggregate the already redacted compact connector receipts.
    Reason: LiteLLM remains the rate-limit owner; VKP only needs execution
    evidence rather than a second limiter or metrics database.
    Evidence: connector receipts already record loopback gateway request,
    response, and source-artifact bytes for every call.
    Effective scope: persisted batch status only; no payload content, API key,
    provider wire estimate, retry, or fallback is introduced.
    """

    summary = ConsentedModelBatchManager._summary(
        [
            {
                "state": "completed",
                "outcome": "success",
                "latency_ms": 100.0,
                "network_accounting": {
                    "gateway_request_bytes": 10,
                    "gateway_response_bytes": 20,
                    "source_artifact_bytes": 30,
                },
            },
            {
                "state": "completed",
                "outcome": "success",
                "latency_ms": 200.0,
                "network_accounting": {
                    "gateway_request_bytes": 11,
                    "gateway_response_bytes": 21,
                    "source_artifact_bytes": 31,
                },
            },
            {
                "state": "failed",
                "outcome": "rate_limited",
                "latency_ms": 300.0,
                "network_accounting": {
                    "gateway_request_bytes": 12,
                    "gateway_response_bytes": 22,
                    "source_artifact_bytes": 32,
                },
            },
            {
                "state": "failed",
                "outcome": "transient_provider_failure",
                "latency_ms": 400.0,
                "network_accounting": {
                    "gateway_request_bytes": 13,
                    "gateway_response_bytes": 23,
                    "source_artifact_bytes": 33,
                },
            },
        ]
    )

    assert summary["latency_sample_count"] == 4
    assert summary["latency_p50_ms"] == 250.0
    assert summary["latency_p95_ms"] == 385.0
    assert summary["gateway_request_bytes"] == 46
    assert summary["gateway_response_bytes"] == 86
    assert summary["source_artifact_bytes"] == 126
    assert summary["rate_limited"] == 1
    assert summary["transient_provider_failure"] == 1
