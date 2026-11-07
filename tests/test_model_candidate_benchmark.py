from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.model_candidate_benchmark import (
    MANIFEST_SCHEMA,
    compare_model_candidates,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _connector_report(
    root: Path,
    candidate_id: str,
    *,
    content: object,
    artifact_manifest_sha256: str = "a" * 64,
    instructions_sha256: str = "b" * 64,
    latency_ms: int = 100,
    nested_ocr: bool = False,
) -> Path:
    consent = _write_json(
        root / f"{candidate_id}-consent.json",
        {
            "schema": "video_knowledge_pipeline.model_connector_consent.v2",
            "instructions_sha256": instructions_sha256,
            "upload_manifest": {"manifest_sha256": artifact_manifest_sha256},
        },
    )
    runtime = {
        "schema": "video_knowledge_pipeline.model_runtime_result.v1",
        "ok": True,
        "status": "completed",
        "task": "online_ocr" if nested_ocr else "summary_rewrite",
        "execution_location": "remote",
        "route_id": "candidate-route",
        "route_revision": "c" * 64,
        "deployment": {
            "id": candidate_id,
            "provider": "fake-provider",
            "model": f"model-{candidate_id}",
        },
        "provider": "fake-provider",
        "latency_ms": latency_ms,
        "usage": {"total_tokens": 10},
        "estimated_cost": None,
        "response": {"model": f"served-{candidate_id}"},
        "content": content,
    }
    model_result = (
        {
            "schema": "video_knowledge_pipeline.online_ocr.v1",
            "ok": True,
            "status": "completed",
            "calls": [{"runtime_result": runtime}],
            "content": content,
        }
        if nested_ocr
        else {
            "schema": "video_knowledge_pipeline.model_task_gateway.v1",
            "ok": True,
            "status": "completed",
            "content": content,
            "runtime_result": runtime,
        }
    )
    return _write_json(
        root / candidate_id / "connector-execution.json",
        {
            "schema": "video_knowledge_pipeline.trusted_model_connector.v1",
            "ok": True,
            "status": "completed",
            "consent_path": str(consent),
            "upload_manifest": {"manifest_sha256": artifact_manifest_sha256},
            "route": {
                "route_id": "candidate-route",
                "route_revision": "c" * 64,
                "deployments": [runtime["deployment"]],
            },
            "model_result": model_result,
            "usage": {"cost_unreported_calls": 1},
        },
    )


def _manifest(path: Path, cases: list[dict[str, object]]) -> Path:
    return _write_json(path, {"schema": MANIFEST_SCHEMA, "cases": cases})


def test_fixed_sample_ranks_clean_candidate_without_copying_content(tmp_path: Path) -> None:
    clean = _connector_report(
        tmp_path,
        "clean",
        content='{"title":"明亚领航计划","speaker":"王菲"}',
        latency_ms=200,
    )
    reasoning = _connector_report(
        tmp_path,
        "reasoning",
        content="<think>reasoning</think> 明亚领航计划",
        latency_ms=50,
    )
    manifest = _manifest(
        tmp_path / "manifest.json",
        [
            {
                "id": "summary-fixed-1",
                "task": "summary_rewrite",
                "model_type": "summary_rewrite",
                "sample_id": "fixed-1",
                "expected_format": "json",
                "required_term_groups": [["明亚"], ["领航计划"], ["王菲"]],
                "forbidden_markers": ["<think>"],
                "candidates": [
                    {"id": "clean", "result_path": str(clean)},
                    {"id": "reasoning", "result_path": str(reasoning)},
                ],
            }
        ],
    )

    result = compare_model_candidates(manifest, output_dir=tmp_path / "report")

    case = result["cases"][0]
    assert result["status"] == "ready_for_review"
    assert case["automatic_proxy_winner"] == "clean"
    assert case["candidates"][1]["forbidden_marker_hits"] == ["<think>"]
    assert case["candidates"][0]["transport_ok"] is True
    assert case["candidates"][0]["provider_response_model"] == "served-clean"
    assert case["candidates"][0]["contract_ok"] is True
    assert case["candidates"][0]["quality_gate_passed"] is True
    assert case["candidates"][1]["transport_ok"] is True
    assert case["candidates"][1]["contract_ok"] is False
    assert case["quality_passed_count"] == 1
    assert all("content" not in row for row in case["candidates"])
    saved = (tmp_path / "report" / "model-candidate-benchmark.json").read_text(
        encoding="utf-8"
    )
    assert "reasoning</think>" not in saved


def test_mismatched_prompt_or_artifact_is_not_comparable(tmp_path: Path) -> None:
    first = _connector_report(tmp_path, "first", content="one")
    second = _connector_report(
        tmp_path,
        "second",
        content="two",
        artifact_manifest_sha256="d" * 64,
        instructions_sha256="e" * 64,
    )
    manifest = _manifest(
        tmp_path / "manifest.json",
        [
            {
                "id": "mismatch",
                "task": "text_llm",
                "model_type": "text_llm",
                "sample_id": "fixed-1",
                "candidates": [
                    {"id": "first", "result_path": str(first)},
                    {"id": "second", "result_path": str(second)},
                ],
            }
        ],
    )

    result = compare_model_candidates(manifest, write=False)

    case = result["cases"][0]
    assert result["status"] == "incomplete"
    assert case["comparable"] is False
    assert case["automatic_proxy_winner"] == ""
    assert "artifact_manifest_mismatch" in case["limitations"]
    assert "instructions_mismatch" in case["limitations"]


def test_missing_prompt_and_artifact_hashes_are_not_comparable(tmp_path: Path) -> None:
    first = _connector_report(tmp_path, "first", content="one")
    second = _connector_report(tmp_path, "second", content="two")
    for path in (first, second):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["upload_manifest"] = {}
        consent_path = Path(report["consent_path"])
        consent = json.loads(consent_path.read_text(encoding="utf-8"))
        consent.pop("instructions_sha256", None)
        _write_json(consent_path, consent)
        _write_json(path, report)
    manifest = _manifest(
        tmp_path / "manifest.json",
        [
            {
                "id": "missing-hashes",
                "task": "text_llm",
                "model_type": "text_llm",
                "sample_id": "fixed-1",
                "candidates": [
                    {"id": "first", "result_path": str(first)},
                    {"id": "second", "result_path": str(second)},
                ],
            }
        ],
    )

    result = compare_model_candidates(manifest, write=False)

    case = result["cases"][0]
    assert case["comparable"] is False
    assert "artifact_manifest_mismatch" in case["limitations"]
    assert "instructions_mismatch" in case["limitations"]


def test_nested_ocr_runtime_and_page_format_are_supported(tmp_path: Path) -> None:
    pages = {"pages": [{"page": 1, "visual_text": "高频问题"}]}
    first = _connector_report(tmp_path, "ocr-a", content=pages, nested_ocr=True)
    second = _connector_report(
        tmp_path,
        "ocr-b",
        content=pages,
        nested_ocr=True,
        latency_ms=120,
    )
    manifest = _manifest(
        tmp_path / "manifest.json",
        [
            {
                "id": "ocr-fixed-1",
                "task": "online_ocr",
                "model_type": "ocr",
                "sample_id": "fixed-1",
                "expected_format": "ocr_pages",
                "required_term_groups": [["高频问题"]],
                "candidates": [
                    {"id": "ocr-a", "result_path": str(first)},
                    {"id": "ocr-b", "result_path": str(second)},
                ],
            }
        ],
    )

    result = compare_model_candidates(manifest, write=False)

    case = result["cases"][0]
    assert case["comparable"] is True
    assert all(row["format_ok"] for row in case["candidates"])
    assert all(row["required_term_coverage"] == 1.0 for row in case["candidates"])
