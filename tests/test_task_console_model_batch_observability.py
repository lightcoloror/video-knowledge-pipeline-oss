from __future__ import annotations

from video_knowledge_pipeline.task_console import _model_batches_html


def test_model_batch_card_shows_heartbeat_latency_and_network_bytes() -> None:
    rendered = _model_batches_html(
        {
            "items": [
                {
                    "job_id": "model_batch_fixture",
                    "status": "running",
                    "terminal": False,
                    "nodes": ["summary-0001"],
                    "destinations": ["https://api.example"],
                    "summary": {
                        "total": 1,
                        "completed": 0,
                        "failed": 0,
                        "dependency_blocked": 0,
                        "heartbeat_alive": 1,
                        "heartbeat_stale": 0,
                        "latency_p50_ms": 1234.4,
                        "latency_p95_ms": 2345.6,
                        "gateway_request_bytes": 4096,
                        "gateway_response_bytes": 1024,
                    },
                    "consent_allowance": {
                        "remaining_calls": 1,
                        "remaining_estimated_cost_usd": 0.01,
                    },
                }
            ]
        }
    )

    assert "心跳正常 1；心跳过期 0" in rendered
    assert "延迟 P50/P95：1234 ms / 2346 ms" in rendered
    assert "网络请求/响应：4096 / 1024 bytes" in rendered
