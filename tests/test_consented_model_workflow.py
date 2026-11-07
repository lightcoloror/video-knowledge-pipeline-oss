from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from video_knowledge_pipeline import task_console
from video_knowledge_pipeline.consented_model_batch import (
    ConsentedModelBatchManager,
    list_consented_model_batches,
)
from video_knowledge_pipeline.model_connector_consent import (
    create_model_connector_consent,
)
from video_knowledge_pipeline.task_console import (
    _model_batches_for_bundle,
    _model_batches_html,
)
from video_knowledge_pipeline.trusted_model_connector_policy import (
    TrustedModelConnectorPolicy,
)


class WorkflowExecutor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[str] = []

    def __call__(
        self,
        consent_path: str,
        *,
        expected_route_revision: str,
        write: bool,
    ) -> dict[str, object]:
        payload = json.loads(Path(consent_path).read_text(encoding="utf-8"))
        assert expected_route_revision == payload["route"]["route_revision"]
        assert write is True
        with self.lock:
            self.calls.append(Path(consent_path).stem)
        time.sleep(0.02)
        return {"ok": True, "status": "completed"}


def _consent(root: Path, name: str, destination: str) -> Path:
    artifact = root / f"{name}.txt"
    artifact.write_text(f"fixture {name}", encoding="utf-8")
    result = create_model_connector_consent(
        root,
        task="smart_summary_rewrite",
        artifact_paths=[artifact],
        provider_config={
            "provider": "custom_openai_compatible",
            "base_url": f"https://{destination}/v1",
            "model": "fixture-model",
        },
        output_path=root / "consents" / f"{name}.json",
        max_calls=1,
        max_estimated_cost_usd=0.01,
        max_cost_per_call_usd=0.01,
        max_retries_per_call=0,
        confirm_data_export=True,
    )
    return Path(result["consent_path"])


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"title": "Workflow fixture"}), encoding="utf-8"
    )
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    return bundle


def test_named_workflow_reuses_batch_dag_and_bundle_run_registry(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    asr = _consent(tmp_path, "asr", "a.example")
    ocr = _consent(tmp_path, "ocr", "b.example")
    summary = _consent(tmp_path, "summary", "a.example")
    executor = WorkflowExecutor()
    policy = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),), frozenset({"a.example", "b.example"})
    )
    manager = ConsentedModelBatchManager(
        project_root=tmp_path,
        policy=policy,
        executor=executor,
        global_parallel_limit=3,
        maximum_parallel_per_destination=2,
    )

    submitted = manager.submit_workflow(
        bundle_dir=str(bundle),
        nodes=[
            {"id": "asr", "consent_path": str(asr), "depends_on": []},
            {"id": "ocr", "consent_path": str(ocr), "depends_on": []},
            {
                "id": "summary",
                "consent_path": str(summary),
                "depends_on": ["asr", "ocr"],
            },
        ],
    )
    result = manager.wait(submitted["job_id"])

    assert result["status"] == "completed"
    assert [row["node_id"] for row in result["items"]] == ["asr", "ocr", "summary"]
    assert executor.calls.index("summary") > executor.calls.index("asr")
    assert executor.calls.index("summary") > executor.calls.index("ocr")
    registry = json.loads(
        (bundle / "run-artifact-registry.json").read_text(encoding="utf-8")
    )
    run = next(row for row in registry["runs"] if row["run_id"] == submitted["job_id"])
    assert run["status"] == "completed"
    assert run["resource_requirements"] == {"network": 1}


def test_bundle_console_reads_redacted_batch_and_current_consent_allowance(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = _bundle(tmp_path)
    consent = _consent(tmp_path, "visual", "a.example")
    policy = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),), frozenset({"a.example"})
    )
    manager = ConsentedModelBatchManager(
        project_root=tmp_path,
        policy=policy,
        executor=WorkflowExecutor(),
        global_parallel_limit=1,
        maximum_parallel_per_destination=1,
    )
    submitted = manager.submit_workflow(
        bundle_dir=str(bundle),
        nodes=[
            {
                "id": "semantic-frame-0001",
                "consent_path": str(consent),
                "depends_on": [],
            }
        ],
    )
    manager.wait(submitted["job_id"])

    monkeypatch.setattr(task_console, "MODEL_BATCH_PROJECT_ROOT", tmp_path)
    batches = _model_batches_for_bundle(bundle.resolve())
    encoded = json.dumps(batches, ensure_ascii=False)
    html = _model_batches_html(batches)

    assert batches["count"] == 1
    assert batches["items"][0]["nodes"] == ["semantic-frame-0001"]
    assert batches["items"][0]["consent_allowance"]["remaining_calls"] == 1
    assert str(consent) not in encoded
    assert "semantic-frame-0001" in html
    assert "Consent 剩余调用" in html


def test_workflow_rejects_unknown_dependency_before_execution(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    consent = _consent(tmp_path, "summary", "a.example")
    executor = WorkflowExecutor()
    policy = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),), frozenset({"a.example"})
    )
    manager = ConsentedModelBatchManager(
        project_root=tmp_path,
        policy=policy,
        executor=executor,
    )

    try:
        manager.submit_workflow(
            bundle_dir=str(bundle),
            nodes=[
                {
                    "id": "summary",
                    "consent_path": str(consent),
                    "depends_on": ["missing-ocr"],
                }
            ],
        )
    except ValueError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("unknown dependency should be rejected")
    assert executor.calls == []


def test_batch_list_does_not_expose_consent_paths_or_model_content(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    consent = _consent(tmp_path, "ocr", "a.example")
    status_path = (
        tmp_path
        / ".local"
        / "model-connector-batches"
        / "model_batch_redacted"
        / "batch-execution.json"
    )
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.consented_model_batch.v1",
                "job_id": "model_batch_redacted",
                "status": "completed",
                "terminal": True,
                "bundle_dir": str(bundle),
                "summary": {"total": 1, "completed": 1},
                "settings": {},
                "destination_controllers": {},
                "items": [
                    {
                        "node_id": "ocr-0001",
                        "task": "online_ocr",
                        "destination": "https://a.example",
                        "consent_path": str(consent),
                        "content": "private model content",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = list_consented_model_batches(tmp_path)
    encoded = json.dumps(result, ensure_ascii=False)

    assert result["count"] == 1
    assert result["items"][0]["consent_allowance"]["remaining_calls"] == 1
    assert str(consent) not in encoded
    assert "private model content" not in encoded
