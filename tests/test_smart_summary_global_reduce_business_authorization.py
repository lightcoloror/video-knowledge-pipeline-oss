from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.smart_summary_global_reduce as global_reduce


def _bundle(tmp_path: Path) -> tuple[Path, list[Path]]:
    root = tmp_path / "bundle"
    exports = root / "exports"
    exports.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"title": "summary fixture"}), encoding="utf-8"
    )
    (root / "timeline.json").write_text("[]", encoding="utf-8")
    workflow = exports / "smart-summary-section-workflow.json"
    revisions = exports / "smart-summary-section-llm-revisions.json"
    course_map = exports / "course-map.json"
    row = {
        "section_id": "chapter-0001",
        "title": "topic",
        "time_range": "00:00:00.000 - 00:01:00.000",
        "final_markdown": "evidence-bound chapter text",
    }
    workflow.write_text(json.dumps({"sections": [row]}), encoding="utf-8")
    revisions.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    course_map.write_text(json.dumps({"mainline": "topic"}), encoding="utf-8")
    return root, [workflow, revisions, course_map]


def _remote_config() -> dict[str, object]:
    return {
        "provider": "fake_remote",
        "model": "reader-model",
        "base_url": "http://127.0.0.1:18776/v1",
        "execution_location": "remote",
        "adapter_backend": "proxy",
        "route_id": "summary-route",
        "route_revision": "a" * 64,
    }


def _patch_reduce_quality(monkeypatch) -> None:
    monkeypatch.setattr(
        global_reduce,
        "_chapter_fact_pack",
        lambda *args, **kwargs: {
            "schema": global_reduce.CHAPTER_FACT_PACK_SCHEMA,
            "revision": "b" * 64,
            "sections": [],
            "summary": {
                "evidence_bound_sections": 1,
                "review_only_sections": 0,
                "unbound_section_ids": [],
                "evidence_reference_count": 1,
                "review_only_evidence_count": 0,
                "source_kinds": ["asr"],
            },
        },
    )
    monkeypatch.setattr(
        global_reduce,
        "parse_reader_plan",
        lambda value: {
            "ok": True,
            "plan": {"schema": "reader", "source_section_ids": ["chapter-0001"]},
            "errors": [],
        },
    )
    monkeypatch.setattr(
        global_reduce,
        "validate_reader_plan",
        lambda *args, **kwargs: {"passed": True, "errors": []},
    )
    monkeypatch.setattr(
        global_reduce,
        "render_reader_summary",
        lambda *args, **kwargs: "# final reader summary",
    )
    monkeypatch.setattr(
        global_reduce,
        "_shape_quality",
        lambda *args, **kwargs: {"passed": True, "checks": []},
    )


def test_remote_reduce_fails_closed_without_business_authorization(
    tmp_path: Path, monkeypatch
) -> None:
    root, _inputs = _bundle(tmp_path)
    cfg = _remote_config()
    monkeypatch.setattr(
        global_reduce, "resolve_model_api_provider_config", lambda *args, **kwargs: cfg
    )
    monkeypatch.setattr(global_reduce, "resolve_text_provider_config", lambda value: value)
    monkeypatch.setattr(
        global_reduce,
        "model_task_api_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct remote call must not run")
        ),
    )

    result = global_reduce.run_smart_summary_global_reduce(
        root, execute=True, install=False, write=False
    )

    assert result["status"] == "business_authorization_required"
    assert result["ok"] is False


def test_remote_reduce_uses_parent_child_and_broker_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    root, inputs = _bundle(tmp_path)
    cfg = _remote_config()
    authorization_path = tmp_path / "parent.json"
    authorization_path.write_text(
        json.dumps(
            {
                "sources": [{"path": str(path)} for path in inputs],
                "stages": [
                    {
                        "id": "global-reduce",
                        "task": "smart_summary_global_reduce",
                        "route_snapshot": {
                            "route_id": cfg["route_id"],
                            "route_revision": cfg["route_revision"],
                        },
                        "allowed_producers": [
                            "smart_summary_global_reduce_request"
                        ],
                    }
                ],
                "admissions": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        global_reduce, "resolve_model_api_provider_config", lambda *args, **kwargs: cfg
    )
    monkeypatch.setattr(global_reduce, "resolve_text_provider_config", lambda value: value)
    monkeypatch.setattr(
        global_reduce,
        "validate_model_business_authorization",
        lambda path: {
            "valid": True,
            "bundle_dirs": [str(root)],
            "blockers": [],
        },
    )
    _patch_reduce_quality(monkeypatch)
    captured: dict[str, object] = {}

    def fake_child(path, **kwargs):
        captured["child"] = {"path": str(path), **kwargs}
        return {
            "status": "created",
            "consent_path": str(tmp_path / "child.json"),
            "consent_id": "consent-1",
            "route_revision": cfg["route_revision"],
            "admission_id": "admission-1",
        }

    def fake_execute(path, **kwargs):
        captured["execute"] = {"path": str(path), **kwargs}
        return {
            "ok": True,
            "status": "completed",
            "model_result": {
                "ok": True,
                "runtime_result": {"content": "{}"},
            },
        }

    monkeypatch.setattr(global_reduce, "create_business_child_consent", fake_child)
    monkeypatch.setattr(global_reduce, "execute_consented_model_task", fake_execute)
    monkeypatch.setattr(
        global_reduce,
        "model_task_api_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct proxy call must not run")
        ),
    )

    result = global_reduce.run_smart_summary_global_reduce(
        root,
        execute=True,
        install=False,
        write=True,
        business_authorization_path=authorization_path,
    )

    assert result["status"] == "waiting_claim_review"
    assert result["ok"] is False
    assert result["model_call"]["connector_status"] == "completed"
    assert captured["child"]["producer"] == "smart_summary_global_reduce_request"
    assert captured["child"]["input_paths"] == [str(path.resolve()) for path in inputs]
    assert captured["execute"]["expected_route_revision"] == cfg["route_revision"]
    request = root / "exports" / "business-authorized-summary-requests" / "smart-summary-global-reduce.json"
    assert request.is_file()


def test_completed_execution_report_can_be_recovered_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    root, _inputs = _bundle(tmp_path)
    cfg = _remote_config()
    monkeypatch.setattr(
        global_reduce, "resolve_model_api_provider_config", lambda *args, **kwargs: cfg
    )
    monkeypatch.setattr(global_reduce, "resolve_text_provider_config", lambda value: value)
    _patch_reduce_quality(monkeypatch)
    report = (
        root
        / "exports"
        / "business-child-consents"
        / "model-connector-runs"
        / "completed"
        / "connector-execution.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "task": "smart_summary_global_reduce",
                "ok": True,
                "status": "completed",
                "model_result": {
                    "ok": True,
                    "runtime_result": {"content": "{}"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        global_reduce,
        "model_task_api_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("recovery must not call a provider")
        ),
    )
    monkeypatch.setattr(
        global_reduce,
        "execute_consented_model_task",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("recovery must not reserve or execute consent")
        ),
    )

    result = global_reduce.run_smart_summary_global_reduce(
        root,
        recovery_execution_report=report,
        install=False,
        write=True,
    )

    assert result["status"] == "waiting_claim_review"
    assert result["ok"] is False
    assert result["recovery"]["network_requests_made"] is False
    assert result["recovery"]["execution_report"] == str(report.resolve())
    assert (root / "exports" / "smart-summary-reader-plan.json").is_file()
    assert (
        root / "exports" / "smart-summary-global-reduce-raw-response.txt"
    ).read_text(encoding="utf-8") == "{}"
