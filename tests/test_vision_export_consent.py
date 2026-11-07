from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.multimodal_frame_analyzer as analyzer
from video_knowledge_pipeline.vision_export_consent import (
    create_vision_export_consent,
    revoke_vision_export_consent,
    vision_export_consent_status,
)
from video_knowledge_pipeline.visual_ab_benchmark import build_visual_ab_benchmark_plan


PROVIDER = {
    "provider": "openai_compatible",
    "base_url": "https://example.invalid/v1",
    "model": "vision-test-model",
    "api_key": "must-not-be-persisted",
}


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    assets = root / "assets"
    assets.mkdir(parents=True)
    for index in range(1, 5):
        (assets / f"frame-{index}.jpg").write_bytes(f"frame-{index}".encode())
    (root / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}), encoding="utf-8")
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 8,
            "text": "大家看这个界面，公司名称可能识别不准。",
            "visual_route": "semantic_frame",
            "frame_paths": ["assets/frame-1.jpg"],
            "quality_issues": ["missing_visual_text", "semantic_frame_without_analysis"],
        },
        {
            "index": 2,
            "start": 60,
            "end": 68,
            "text": "接下来演示三个步骤。",
            "visual_route": "temporal_sequence",
            "frame_paths": ["assets/frame-2.jpg"],
            "temporal_frame_paths": ["assets/frame-2.jpg", "assets/frame-3.jpg"],
            "quality_issues": ["temporal_sequence_without_analysis"],
        },
        {
            "index": 3,
            "start": 120,
            "end": 128,
            "text": "课件写了产品名称 Alpha Pro。",
            "visual_route": "document_visual",
            "frame_paths": ["assets/frame-3.jpg"],
            "visual_text": "Alpha Pro 产品说明",
            "structured_visual": {"type": "slide"},
        },
        {
            "index": 4,
            "start": 180,
            "end": 188,
            "text": "这里的金额是 12800 元。",
            "visual_route": "mixed",
            "frame_paths": ["assets/frame-4.jpg"],
            "quality_issues": ["ocr_text_low_information"],
        },
    ]
    (root / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    return root


def test_consent_requires_explicit_confirmation_and_never_persists_key(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    preview = create_vision_export_consent(
        root,
        provider_config=PROVIDER,
        semantic_indexes=[1],
        confirm_data_export=False,
    )
    assert preview["status"] == "confirmation_required"
    assert preview["user_confirmed_data_export"] is False

    active = create_vision_export_consent(
        root,
        provider_config=PROVIDER,
        semantic_indexes=[1, 4],
        temporal_indexes=[2],
        max_calls=3,
        confirm_data_export=True,
    )
    persisted = (root / "vision-export-consent.json").read_text(encoding="utf-8")
    assert active["status"] == "active"
    assert "must-not-be-persisted" not in persisted
    assert "api_key" not in active["provider"]


def test_consent_validates_bundle_provider_indexes_and_export_evidence(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    create_vision_export_consent(
        root,
        provider_config=PROVIDER,
        semantic_indexes=[1, 4],
        max_calls=2,
        confirm_data_export=True,
    )
    ok = vision_export_consent_status(
        root,
        provider_config=PROVIDER,
        semantic_indexes=[1],
        expected_calls=1,
    )
    assert ok["valid"] is True

    wrong_model = dict(PROVIDER, model="other-model")
    mismatch = vision_export_consent_status(
        root,
        provider_config=wrong_model,
        semantic_indexes=[1],
        expected_calls=1,
    )
    assert mismatch["valid"] is False
    assert "consent_provider_model_mismatch" in {row["key"] for row in mismatch["blockers"]}

    timeline_path = root / "timeline.json"
    timeline_path.write_text(timeline_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    formatting_only = vision_export_consent_status(
        root,
        provider_config=PROVIDER,
        semantic_indexes=[1],
        expected_calls=1,
    )
    assert formatting_only["valid"] is True

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline[0]["visual_understanding"] = {"objects": ["screen"]}
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    model_writeback = vision_export_consent_status(
        root,
        provider_config=PROVIDER,
        semantic_indexes=[1],
        expected_calls=1,
    )
    assert model_writeback["valid"] is True

    timeline[0]["transcript"] = "The transcript context changed."
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    drift = vision_export_consent_status(root, provider_config=PROVIDER, semantic_indexes=[1], expected_calls=1)
    assert "consent_export_evidence_changed" in {row["key"] for row in drift["blockers"]}


def test_agent_execution_requires_scoped_consent(monkeypatch, tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    preflight = {
        "ready_to_execute": True,
        "expected_api_calls": 1,
        "selected_indexes": {"semantic": [1], "temporal": []},
        "blockers": [],
        "commands": {"confirmed_run_semantic": "manual-command"},
    }
    monkeypatch.setattr(analyzer, "vision_execution_preflight", lambda *args, **kwargs: dict(preflight))
    gate, control = analyzer._execution_control(
        root,
        True,
        PROVIDER,
        semantic_limit=1,
        temporal_limit=0,
        frame_count=8,
        include_semantic=True,
        include_temporal=False,
        semantic_indexes=[1],
        confirm_vision_calls=1,
        confirm_vision_indexes="1",
        image_probe_max_edge=512,
        image_probe_jpeg_quality=55,
        execution_actor="agent",
    )
    assert gate["status"] == "vision_export_consent_required"
    assert control["confirmed"] is False
    assert gate["fallbacks"]["visible_powershell"] == "manual-command"

    consent = create_vision_export_consent(
        root,
        provider_config=PROVIDER,
        semantic_indexes=[1],
        confirm_data_export=True,
    )
    gate, control = analyzer._execution_control(
        root,
        True,
        PROVIDER,
        semantic_limit=1,
        temporal_limit=0,
        frame_count=8,
        include_semantic=True,
        include_temporal=False,
        semantic_indexes=[1],
        confirm_vision_calls=1,
        confirm_vision_indexes="1",
        image_probe_max_edge=512,
        image_probe_jpeg_quality=55,
        execution_actor="agent",
        export_consent=consent["artifacts"]["consent_json"],
    )
    assert gate == {}
    assert control["confirmed"] is True
    assert control["export_consent"]["valid"] is True


def test_revoke_and_visual_ab_plan(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    create_vision_export_consent(
        root,
        provider_config=PROVIDER,
        semantic_indexes=[1],
        confirm_data_export=True,
    )
    revoked = revoke_vision_export_consent(root)
    assert revoked["status"] == "revoked"

    plan = build_visual_ab_benchmark_plan(root, limit=3, min_score=4)
    assert plan["status"] == "ready"
    assert 1 <= len(plan["items"]) <= 3
    assert plan["operator_boundary"]["does_not_call_online_model"] is True
    assert (root / "visual-ab-benchmark-plan.md").exists()

def test_volcengine_runner_exposes_actor_and_consent_contract() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run-volcengine-vision-batch.ps1"
    script = script_path.read_text(encoding="utf-8")
    assert '[ValidateSet("operator", "agent")]' in script
    assert '[string]$ExecutionActor = "operator"' in script
    assert '[string]$ExportConsent = ""' in script
    assert '"--execution-actor", $ExecutionActor' in script
    assert '"--export-consent", $resolvedConsent' in script
