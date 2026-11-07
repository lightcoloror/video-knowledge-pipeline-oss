from __future__ import annotations

import hashlib
import json
from pathlib import Path

from video_knowledge_pipeline.model_task_gateway import MODEL_TASKS
from video_knowledge_pipeline.ocr_route import normalize_online_ocr_result, run_ocr_route


from video_knowledge_pipeline.trusted_model_connector import trusted_model_connector_capabilities

def test_online_ocr_is_a_consented_multi_image_model_task() -> None:
    spec = MODEL_TASKS["online_ocr"]
    assert spec["model_type"] == "ocr"
    assert spec["modality"] == "multi_image"
    assert spec["execution"] == "online_via_consented_connector"
    capabilities = trusted_model_connector_capabilities()
    advertised = {row["task"]: row for row in capabilities["tasks"]}
    assert advertised["online_ocr"]["modality"] == "multi_image"


def test_online_route_prepares_exact_artifacts_without_executing_provider(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    image = bundle / "slide-6.jpg"
    image.write_bytes(b"jpeg")
    provider_calls: list[dict] = []

    def fake_visual_structure(bundle_dir, **kwargs):
        return {
            "items": [{"index": 6, "image_path": str(image), "visual_route": "document_visual"}],
            "summary": {"execute_ebook_pipeline": kwargs.get("execute_ebook_pipeline")},
        }

    def fake_model_task(task, **kwargs):
        provider_calls.append({"task": task, **kwargs})
        return {"request_plan": {"interface": "vision", "image_paths": kwargs["image_paths"]}}

    monkeypatch.setattr("video_knowledge_pipeline.ocr_route.run_visual_structure_plan", fake_visual_structure)
    monkeypatch.setattr("video_knowledge_pipeline.ocr_route.model_task_api_call", fake_model_task)

    result = run_ocr_route(bundle, backend="online", provider_config={"provider": "fixture"})

    assert result["status"] == "planned_requires_consent"
    assert result["image_paths"] == [str(image.resolve())]
    assert provider_calls[0]["task"] == "online_ocr"
    assert provider_calls[0]["execute"] is False
    assert result["operator_boundary"]["online_execution_exposed_here"] is False
    request = json.loads((bundle / "online-ocr-consent-request.json").read_text(encoding="utf-8"))
    assert request["artifact_paths"] == [str(image.resolve())]
    assert request["confirm_data_export"] is False
    assert request["max_calls"] == 1


def test_local_route_keeps_ebook_markdown_pipeline_as_default(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    calls: list[dict] = []

    def fake_visual_structure(bundle_dir, **kwargs):
        calls.append(kwargs)
        return {"items": [], "summary": {"execute_ebook_pipeline": kwargs["execute_ebook_pipeline"]}}

    monkeypatch.setattr("video_knowledge_pipeline.ocr_route.run_visual_structure_plan", fake_visual_structure)
    result = run_ocr_route(bundle, backend="local", execute_local=True)

    assert result["backend_id"] == "ebook_markdown_pipeline"
    assert result["status"] == "completed"
    assert calls[0]["execute_ebook_pipeline"] is True
    assert result["operator_boundary"]["network_call"] is False


def test_connector_result_normalizes_to_visual_structure_rows(tmp_path: Path) -> None:
    image = tmp_path / "slide-6.jpg"
    image.write_bytes(b"jpeg")
    connector = tmp_path / "connector-execution.json"
    connector.write_text(
        json.dumps(
            {
                "task": "online_ocr",
                "ok": True,
                "model_result": {
                    "ok": True,
                    "content": json.dumps(
                        {
                            "pages": [
                                {
                                    "index": 6,
                                    "image_path": image.name,
                                    "markdown": "# 高频问题\n\n客户不回复",
                                    "confidence": 0.97,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows = normalize_online_ocr_result(
        connector,
        candidates=[{"index": 6, "image_path": str(image)}],
    )

    assert len(rows) == 1
    assert rows[0]["index"] == 6
    assert rows[0]["source"] == "online_ocr"
    assert rows[0]["visual_text"] == "# 高频问题\n\n客户不回复"
    assert rows[0]["confidence"] == 0.97
    assert rows[0]["image_path"] == str(image.resolve())
    assert rows[0]["evidence_paths"] == [str(image.resolve())]


def test_partial_online_ocr_keeps_successful_candidate_evidence(tmp_path: Path) -> None:
    image = tmp_path / "slide-8.jpg"
    image.write_bytes(b"jpeg")
    connector = tmp_path / "connector-partial.json"
    connector.write_text(
        json.dumps(
            {
                "task": "online_ocr",
                "ok": False,
                "model_result": {
                    "ok": False,
                    "status": "partial_ocr_failure",
                    "content": {
                        "pages": [
                            {
                                "index": 0,
                                "image_path": str(image),
                                "markdown": "# Partial OCR",
                                "dimensions": {"width": 1280, "height": 720},
                                "source_artifact_sha256": "a" * 64,
                            }
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    rows = normalize_online_ocr_result(
        connector,
        candidates=[{"index": 8, "image_path": str(image)}],
    )

    assert rows[0]["index"] == 8
    assert rows[0]["evidence_status"] == "candidate"
    assert rows[0]["source_artifact_sha256"] == "a" * 64
    assert rows[0]["structured_visual"][0]["page_index"] == 0


def test_online_import_maps_audited_artifact_to_explicit_reviewed_index(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    image = bundle / "slide-reviewed.jpg"
    image.write_bytes(b"reviewed-jpeg")
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    connector = tmp_path / "connector-execution.json"
    connector.write_text(
        json.dumps(
            {
                "task": "online_ocr",
                "ok": True,
                "artifact_paths": [str(image.resolve())],
                "upload_manifest": {
                    "files": [
                        {
                            "path": str(image.resolve()),
                            "data_type": "image",
                            "bytes": image.stat().st_size,
                            "sha256": image_sha256,
                        }
                    ]
                },
                "model_result": {
                    "ok": True,
                    "status": "completed",
                    "content": {
                        "pages": [
                            {
                                "index": 0,
                                "image_path": str(image.resolve()),
                                "markdown": "# 已审核页面\n\n在线 OCR 候选",
                                "source_artifact_sha256": image_sha256,
                            }
                        ]
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls: list[dict] = []

    def fake_visual_structure(bundle_dir, **kwargs):
        calls.append(kwargs)
        if kwargs.get("input_json"):
            imported = json.loads(Path(kwargs["input_json"]).read_text(encoding="utf-8"))
            assert imported["items"][0]["index"] == 6
            return {"items": [], "summary": {"updated": 1}}
        return {"items": [], "summary": {"updated": 0}}

    monkeypatch.setattr("video_knowledge_pipeline.ocr_route.run_visual_structure_plan", fake_visual_structure)
    monkeypatch.setattr(
        "video_knowledge_pipeline.ocr_route.model_task_api_call",
        lambda *args, **kwargs: {"request_plan": {}},
    )

    result = run_ocr_route(
        bundle,
        backend="online",
        connector_result_json=connector,
        indexes=[6],
        limit=1,
    )

    assert result["status"] == "imported"
    assert result["candidate_count"] == 0
    assert result["import_candidate_count"] == 1
    assert result["imported_row_count"] == 1
    assert len(calls) == 2