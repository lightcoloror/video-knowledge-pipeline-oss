from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from video_knowledge_pipeline.cli import build_parser, run_mcp_call
from video_knowledge_pipeline.openclaw_http import build_handler
from video_knowledge_pipeline.openclaw_docker_contract import openclaw_docker_contract_check
from video_knowledge_pipeline.openclaw_integration import openclaw_video_ingest, openclaw_video_link, openclaw_video_plan
from video_knowledge_pipeline.openclaw_bridge_status import openclaw_bridge_status
from video_knowledge_pipeline.openclaw_live_smoke import openclaw_live_smoke
from video_knowledge_pipeline.path_defaults import workspace_root
from video_knowledge_pipeline.vdo_handoff import ingest_vdo_handoff, vdo_handoff_plan


def _completed(payload: dict) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["fake"], returncode=0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")



def _write_semantic_accepted_bundle(bundle: Path) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "Semantic accepted bundle",
                "normalized_transcript_json": "normalized-transcript.json",
                "corrected_transcript_json": "source-arbitrated-transcript.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    (bundle / "normalized-transcript.json").write_text(json.dumps({"segments": []}, ensure_ascii=False), encoding="utf-8")
    (bundle / "transcript-semantic-correction-pack.json").write_text(
        json.dumps(
            {
                "schema": "pack",
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_id": "semcorr-0001",
                        "correction_type": "proper_noun",
                        "risk_level": "medium",
                        "evidence_source_types": ["ocr_ebook"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-correction-validation.json").write_text(
        json.dumps(
            {
                "accepted_decision_count": 1,
                "review_required_count": 0,
                "accepted_decisions": [
                    {
                        "candidate_id": "semcorr-0001",
                        "original_text": "play right",
                        "corrected_text": "Playwright",
                        "correction_type": "proper_noun",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-correction-closure.json").write_text(json.dumps({"status": "completed", "ok": True, "applied_correction_count": 1}, ensure_ascii=False), encoding="utf-8")
    (bundle / "transcript-semantic-correction-impact-report.json").write_text(
        json.dumps({"status": "passed", "ok": True, "final_residual_error_total": 0, "accepted_decision_count": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-readable-impact-report.json").write_text(
        json.dumps({"status": "passed", "ok": True, "required_readable_residual_total": 0}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "transcript-semantic-summary-impact-report.json").write_text(
        json.dumps({"status": "passed", "ok": True, "summary_absorption_rate": 1.0, "summary_residual_original_total": 0}, ensure_ascii=False),
        encoding="utf-8",
    )
    canonical = bundle / "source-arbitrated-transcript.json"
    canonical.write_text(
        json.dumps({"segments": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    canonical_hash = hashlib.sha256(canonical.read_bytes()).hexdigest()
    exports = bundle / "exports"
    exports.mkdir(exist_ok=True)
    (exports / "full-transcript.md").write_text(
        f"Canonical source SHA-256: `{canonical_hash}`\n\nPlaywright",
        encoding="utf-8",
    )
    (exports / "knowledge-note.md").write_text(
        f"Canonical transcript SHA-256: `{canonical_hash}`\n\nPlaywright",
        encoding="utf-8",
    )
    (exports / "smart-summary-input-pack.json").write_text(
        json.dumps(
            {
                "transcript_source": str(canonical),
                "transcript_source_sha256": canonical_hash,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (exports / "smart-summary.md").write_text("Playwright", encoding="utf-8")


def test_openclaw_video_plan_delegates_to_video_download_orchestrator_without_downloading(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], env: dict[str, str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert "video_orchestrator.cli" in command
        assert "openclaw-plan" in command
        assert env["PYTHONPATH"]
        return _completed(
            {
                "service": "video-download-orchestrator",
                "contract": "openclaw_telegram_link_plan_v1",
                "status": "planned",
                "will_download": False,
                "batch_dir": str(tmp_path / "vdo"),
                "items": [{"url": "https://example.com/video", "manifest_path": str(tmp_path / "vdo" / "manifest.json")}],
            }
        )

    result = openclaw_video_plan(
        "https://example.com/video",
        output_dir=tmp_path / "openclaw",
        vdo_root=tmp_path / "video-download-orchestrator",
        runner=runner,
    )

    assert result["ok"] is True
    assert result["input_type"] == "video_url"
    assert result["will_download"] is False
    assert result["download_plan"]["contract"] == "openclaw_telegram_link_plan_v1"
    assert result["artifacts"]["download_manifest_paths"] == [str(tmp_path / "vdo" / "manifest.json")]
    assert result["operator_boundary"]["requires_human_confirmation_for_download"] is True
    assert "execute_download_in_video_download_orchestrator_after_confirmation" in result["next_actions"]
    assert calls


def test_openclaw_video_ingest_wraps_prepare_local_video_run_for_existing_file(tmp_path: Path) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    def prepare_runner(*args, **kwargs) -> dict:
        workspace = Path(args[1])
        bundle = workspace / "webui-bundle"
        return {
            "schema": "video_knowledge_local_video_run.v1",
            "workspace_dir": str(workspace),
            "media_path": str(media),
            "selected_media_path": str(media),
            "json_path": str(workspace / "video-knowledge-run.json"),
            "markdown_path": str(workspace / "video-knowledge-run.md"),
            "asr_plan": {"plan_path": str(workspace / "asr-plan.json")},
            "transcript_path": "",
            "initial_bundle": {
                "status": "ok",
                "bundle_dir": str(bundle),
                "review_html": str(bundle / "review.html"),
                "manifest_path": str(bundle / "manifest.json"),
            },
            "next_steps": [{"key": "run_frame_router"}, {"key": "run_visual_branches"}],
        }

    result = openclaw_video_ingest(media, workspace=tmp_path / "run", prepare_runner=prepare_runner)

    assert result["ok"] is True
    assert result["input_type"] == "local_video_path"
    assert result["will_download"] is False
    assert result["workspace"] == str((tmp_path / "run").resolve())
    assert result["review_url_or_file"].endswith("review.html")
    assert result["artifacts"]["bundle_dir"].endswith("webui-bundle")
    assert result["next_actions"] == ["run_frame_router", "run_visual_branches"]


def test_openclaw_video_link_defaults_to_plan_only(tmp_path: Path) -> None:
    executed: list[str] = []

    def runner(command: list[str], env: dict[str, str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        executed.append(command[3])
        return _completed({"status": "planned", "will_download": False, "items": [{"url": "https://example.com/video"}]})

    result = openclaw_video_link(
        "https://example.com/video",
        output_dir=tmp_path / "openclaw",
        vdo_root=tmp_path / "video-download-orchestrator",
        allow_download=False,
        runner=runner,
    )

    assert result["status"] == "planned_download_not_executed"
    assert result["will_download"] is False
    assert executed == ["openclaw-plan"]
    assert "explicitly_execute_download_in_video_download_orchestrator" in result["next_actions"]


def test_openclaw_video_link_requires_confirmation_before_execute(tmp_path: Path) -> None:
    executed: list[str] = []

    def runner(command: list[str], env: dict[str, str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        executed.append(command[3])
        return _completed({"status": "planned", "will_download": False, "items": [{"url": "https://example.com/video"}]})

    result = openclaw_video_link(
        "https://example.com/video",
        output_dir=tmp_path / "openclaw",
        vdo_root=tmp_path / "video-download-orchestrator",
        allow_download=True,
        runner=runner,
    )

    assert result["ok"] is False
    assert result["status"] == "download_confirmation_required"
    assert result["will_download"] is False
    assert executed == ["openclaw-plan"]


def test_openclaw_cli_contracts() -> None:
    plan_args = build_parser().parse_args(["openclaw-video-plan", "https://example.com/video", "--include-manifests"])
    assert plan_args.command == "openclaw-video-plan"
    assert plan_args.url_or_text == "https://example.com/video"
    assert plan_args.include_manifests is True

    ingest_args = build_parser().parse_args(["openclaw-video-ingest", "D:\\videos\\lesson.mp4", "--no-build-initial-bundle"])
    assert ingest_args.command == "openclaw-video-ingest"
    assert ingest_args.media_path == "D:\\videos\\lesson.mp4"
    assert ingest_args.no_build_initial_bundle is True

    link_args = build_parser().parse_args(["openclaw-video-link", "https://example.com/video", "--allow-download", "--confirm-download"])
    assert link_args.command == "openclaw-video-link"
    assert link_args.allow_download is True
    assert link_args.confirm_download is True

    status_args = build_parser().parse_args(["openclaw-bridge-status", "--no-health"])
    assert status_args.command == "openclaw-bridge-status"
    assert status_args.no_health is True

    docker_args = build_parser().parse_args(["openclaw-docker-contract-check", "--compose-path", "docker-compose.yml"])
    assert docker_args.command == "openclaw-docker-contract-check"
    assert docker_args.compose_path == "docker-compose.yml"

    handoff_args = build_parser().parse_args(["openclaw-video-from-vdo-handoff", "--summary-path", "summary.json"])
    assert handoff_args.command == "openclaw-video-from-vdo-handoff"
    assert handoff_args.summary_path == "summary.json"

    ingest_handoff_args = build_parser().parse_args(["openclaw-video-ingest-vdo-handoff", "--handoff-path", "handoff.json", "--execute"])
    assert ingest_handoff_args.command == "openclaw-video-ingest-vdo-handoff"
    assert ingest_handoff_args.handoff_path == "handoff.json"
    assert ingest_handoff_args.execute is True

    content_asset_args = build_parser().parse_args(["content-asset-status", "D:\\runs\\bundle"])
    assert content_asset_args.command == "content-asset-status"
    assert content_asset_args.bundle_dir == "D:\\runs\\bundle"

    doctor_args = build_parser().parse_args(["openclaw-bridge-doctor", "--timeout-seconds", "0.1"])
    assert doctor_args.command == "openclaw-bridge-doctor"
    assert doctor_args.timeout_seconds == 0.1

    live_smoke_args = build_parser().parse_args([
        "openclaw-live-smoke",
        "--bundle-dir",
        r"D:\runs\bundle",
        "--semantic-batch-input",
        r"D:\runs",
        "--semantic-target-bundle-count",
        "5",
        "--semantic-limit",
        "5",
        "--write-report",
    ])
    assert live_smoke_args.command == "openclaw-live-smoke"
    assert live_smoke_args.bundle_dir == r"D:\runs\bundle"
    assert live_smoke_args.semantic_batch_input == r"D:\runs"
    assert live_smoke_args.semantic_target_bundle_count == 5
    assert live_smoke_args.semantic_limit == 5
    assert live_smoke_args.write_report is True

    batch_content_args = build_parser().parse_args(["batch-content-asset-status", "batch.json"])
    assert batch_content_args.command == "batch-content-asset-status"

    handoff_pack_args = build_parser().parse_args(["content-handoff-pack", "batch.json"])
    assert handoff_pack_args.command == "content-handoff-pack"


def test_openclaw_ingest_is_available_through_mcp_call_bridge(tmp_path: Path) -> None:
    args_json = tmp_path / "openclaw-ingest.args.json"
    args_json.write_text(json.dumps({"media_path": str(tmp_path / "missing.mp4")}), encoding="utf-8")

    result = run_mcp_call("openclaw_video_ingest", args_json)

    assert result["status"] == "media_not_found"
    assert result["input_type"] == "local_video_path"
    assert result["mcp_call"]["tool"] == "openclaw_video_ingest"


def test_openclaw_bridge_status_reports_configured_but_not_running(monkeypatch, tmp_path: Path) -> None:
    port = _free_port()
    config_path = tmp_path / "video-knowledge-pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.config.v1",
                "services": {
                    "review_webui": {"type": "static_file", "entrypoint": "webui-bundle/review.html"},
                    "ebook_markdown_pipeline_http": {"host": "127.0.0.1", "port": 9876, "path": "/call"},
                    "openclaw_http": {"host": "127.0.0.1", "port": port, "path": "/call", "docker_host": "host.docker.internal"},
                    "mcp": {"transport": "stdio"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))

    result = openclaw_bridge_status(timeout_seconds=0.1, check_health=False, check_task=False)

    assert result["configured"] is True
    assert result["running"] is False
    assert result["status"] == "not_running"
    assert result["runtime_disposition"] == "stopped_by_design"
    assert result["auto_start_attempted"] is False
    assert result["scheduled_task"]["checked"] is False
    assert result["host_call_url"] == f"http://127.0.0.1:{port}/call"
    assert "register_or_start_openclaw_http_bridge" in result["next_actions"]


def test_openclaw_docker_contract_check_reports_missing_mount_and_env(tmp_path: Path) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        """
services:
  openclaw-gateway:
    volumes:
      - ${OPENCLAW_CONFIG_DIR}:/home/node/.openclaw
""",
        encoding="utf-8",
    )

    result = openclaw_docker_contract_check(compose)

    assert result["ok"] is False
    issue_keys = {issue["key"] for issue in result["issues"]}
    assert "container_root_mounted" in issue_keys
    assert "vkp_api_base" in issue_keys
    assert result["recommended_override"]["volume"] == f"{workspace_root()}:/mnt/used-by-codex"


def test_openclaw_live_smoke_writes_report_without_processing_video(tmp_path: Path) -> None:
    bundle = tmp_path / "missing-bundle"
    out = tmp_path / "reports"

    result = openclaw_live_smoke(bundle_dir=bundle, output_dir=out, timeout_seconds=0.1, write_report=True)

    assert result["schema"] == "video_knowledge_pipeline.openclaw_live_smoke.v1"
    assert result["ok"] is False
    assert result["operator_boundary"]["no_video_processing"] is True
    assert Path(result["report_json_path"]).exists()
    assert Path(result["report_markdown_path"]).exists()
    assert "OpenClaw Live Smoke Report" in Path(result["report_markdown_path"]).read_text(encoding="utf-8")



def test_openclaw_live_smoke_reports_transcript_semantic_batch_acceptance(tmp_path: Path) -> None:
    bundle = tmp_path / "semantic-bundle" / "webui-bundle"
    out = tmp_path / "reports"
    _write_semantic_accepted_bundle(bundle)

    result = openclaw_live_smoke(bundle_dir=bundle, output_dir=out, timeout_seconds=0.1, write_report=True)

    semantic = result["transcript_semantic_batch_acceptance"]
    assert semantic["checked"] is True
    assert semantic["status"] == "accepted"
    assert semantic["target_bundle_count"] == 1
    assert semantic["summary"]["accepted_count"] == 1
    queue = result["transcript_semantic_repair_queue"]
    assert queue["checked"] is True
    assert queue["status"] == "complete"
    assert queue["summary"]["action_required_count"] == 0
    assert queue["items"][0]["action_key"] == "none"
    assert result["operator_boundary"]["no_cloud_calls"] is True
    markdown = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
    assert "Transcript Semantic Correction" in markdown
    assert "Repair queue action required" in markdown
    assert "accepted" in markdown

def test_vdo_handoff_ready_for_ingest_with_reviewed_media(tmp_path: Path) -> None:
    media = tmp_path / "lesson.mp4"
    sidecar = tmp_path / "lesson.info.json"
    media.write_bytes(b"fake media")
    sidecar.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.json"
    review = tmp_path / "review-checklist.json"
    manifest.write_text(
        json.dumps({"url": "https://www.bilibili.com/video/BV123", "title": "课程", "route": {"kind": "bilibili"}, "selected_backend": "yt-dlp"}),
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "status": "finished",
                "task_id": "task-1",
                "selected_backend": "yt-dlp",
                "backend_result": {"output_file": str(media)},
                "archive_files": [{"path": str(sidecar)}],
                "review_checklist_path": str(review),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    review.write_text(
        json.dumps(
            {
                "status": "ok",
                "needs_review": False,
                "manual_review_required": False,
                "checks": [{"name": "output_files", "status": "ok", "files": [{"path": str(media)}]}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = vdo_handoff_plan(manifest_path=manifest, summary_path=summary, review_checklist_path=review, host_root=tmp_path, container_root="/mnt/test")

    assert result["ok"] is True
    assert result["status"] == "ready_for_ingest"
    assert result["media_path"] == str(media.resolve())
    assert result["media_path_container"] == "/mnt/test/lesson.mp4"
    assert result["platform"] == "bilibili"
    assert result["ingestion"]["recommended_tool"] == "openclaw_video_ingest"
    assert result["ingestion"]["execute_asr"] is False
    assert result["ingestion"]["execute_vision"] is False
    assert result["content_assets"]["review_required"] is True
    assert result["content_assets"]["publication_allowed"] is False
    assert result["content_assets"]["material_card_contract"]["field_mapping"]["content_stage"] == "candidate"
    assert result["content_assets"]["material_card_contract"]["allowed_as_inspiration"] is False
    assert result["content_assets"]["consumer_rules"]["circle_of_friends"]["allowed_status"] == "not_allowed_until_vkp_export"
    assert "vdo_review_before_ingest" in result["content_assets"]["human_confirmation_required"]
    assert result["sidecars"][0]["kind"] == "info_json"


def test_vdo_handoff_blocks_ingest_when_review_requires_human(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    review = tmp_path / "review-checklist.json"
    summary.write_text(json.dumps({"status": "finished", "review_checklist_path": str(review)}), encoding="utf-8")
    review.write_text(
        json.dumps(
            {
                "status": "ok",
                "needs_review": True,
                "manual_review_required": True,
                "checks": [{"name": "output_files", "status": "warning", "message": "no output files were detected", "files": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = vdo_handoff_plan(summary_path=summary, review_checklist_path=review)

    assert result["ok"] is False
    assert result["status"] == "needs_review"
    assert result["media_path"] == ""
    assert result["ingestion"]["next_action"] == "download_or_media_path_required"
    assert "media_missing" in result["review"]["reasons"]
    assert "output_files_warning" in result["review"]["reasons"]


def test_ingest_vdo_handoff_preview_and_review_gate(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    review = tmp_path / "review-checklist.json"
    summary.write_text(json.dumps({"status": "finished", "review_checklist_path": str(review)}), encoding="utf-8")
    review.write_text(json.dumps({"needs_review": True, "manual_review_required": True, "checks": []}), encoding="utf-8")

    result = ingest_vdo_handoff(summary_path=summary, review_checklist_path=review, execute=True)

    assert result["ok"] is False
    assert result["status"] == "operator_review_required"
    assert result["operator_boundary"]["no_video_processing"] is True


def test_ingest_vdo_handoff_execute_uses_existing_ingest_without_asr_execution(tmp_path: Path) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake media")
    handoff = {
        "ok": True,
        "status": "ready_for_ingest",
        "media_path": str(media),
        "title": "课程",
        "ingestion": {"workspace": str(tmp_path / "run"), "title": "课程"},
    }
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(handoff, ensure_ascii=False), encoding="utf-8")
    calls: list[dict] = []

    def prepare_runner(*args, **kwargs) -> dict:
        calls.append(kwargs)
        workspace = Path(args[1])
        return {
            "workspace_dir": str(workspace),
            "media_path": str(media),
            "json_path": str(workspace / "video-knowledge-run.json"),
            "markdown_path": str(workspace / "video-knowledge-run.md"),
            "initial_bundle": {"bundle_dir": str(workspace / "webui-bundle"), "review_html": str(workspace / "webui-bundle" / "review.html")},
            "next_steps": [{"key": "review_bundle"}],
        }

    result = ingest_vdo_handoff(handoff_path=handoff_path, execute=True, prepare_runner=prepare_runner)

    assert result["ok"] is True
    assert result["status"] == "ingested"
    assert calls
    assert calls[0]["execute_asr"] is False


def test_openclaw_http_bridge_health_tools_and_call(tmp_path: Path) -> None:
    handler = build_handler()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        health = _http_json(f"{base_url}/health")
        assert health["ok"] is True
        assert "openclaw_video_ingest" in health["tools"]
        assert "openclaw_bridge_status" in health["tools"]
        assert "openclaw_docker_contract_check" in health["tools"]
        assert "openclaw_bridge_doctor" in health["tools"]
        assert "openclaw_live_smoke" in health["tools"]
        assert "openclaw_video_ingest_vdo_handoff" in health["tools"]
        assert "content_asset_status" in health["tools"]
        assert "batch_content_asset_status" in health["tools"]
        assert "content_handoff_pack" in health["tools"]
        assert "transcript_semantic_repair_run" in health["tools"]

        tools = _http_json(f"{base_url}/tools")
        assert {tool["name"] for tool in tools["tools"]} >= {
            "openclaw_bridge_status",
            "openclaw_bridge_doctor",
            "openclaw_live_smoke",
            "openclaw_docker_contract_check",
            "openclaw_video_plan",
            "openclaw_video_ingest",
            "openclaw_video_link",
            "openclaw_video_from_vdo_handoff",
            "openclaw_video_ingest_vdo_handoff",
            "content_asset_status",
            "batch_content_asset_status",
            "content_handoff_pack",
            "transcript_semantic_repair_run",
        }

        result = _http_json(
            f"{base_url}/call",
            {
                "name": "openclaw_video_ingest",
                "arguments": {"media_path": str(tmp_path / "missing.mp4")},
            },
        )
        assert result["ok"] is False
        assert result["status"] == "media_not_found"
        assert result["result"]["input_type"] == "local_video_path"

        status_result = _http_json(f"{base_url}/call", {"name": "openclaw_bridge_status", "arguments": {"check_health": False, "check_task": False}})
        assert status_result["result"]["schema"] == "video_knowledge_pipeline.openclaw_bridge_status.v1"
        assert "running" in status_result["result"]

        content_status = _http_json(f"{base_url}/call", {"name": "content_asset_status", "arguments": {"bundle_dir": str(tmp_path / "missing-bundle")}})
        assert content_status["result"]["status"] == "bundle_missing"
        assert content_status["result"]["schema"] == "video_knowledge_pipeline.content_asset_status.v1"

        doctor = _http_json(f"{base_url}/call", {"name": "openclaw_bridge_doctor", "arguments": {"timeout_seconds": 0.1, "project_root": str(tmp_path)}})
        assert doctor["result"]["schema"] == "video_knowledge_pipeline.openclaw_bridge_doctor.v1"

        live_smoke = _http_json(f"{base_url}/call", {"name": "openclaw_live_smoke", "arguments": {"bundle_dir": str(tmp_path / "missing-bundle"), "timeout_seconds": 0.1}})
        assert live_smoke["result"]["schema"] == "video_knowledge_pipeline.openclaw_live_smoke.v1"

        batch_status = _http_json(f"{base_url}/call", {"name": "batch_content_asset_status", "arguments": {"batch_input": str(tmp_path), "write": False}})
        assert batch_status["result"]["schema"] == "video_knowledge_pipeline.batch_content_asset_status.v1"

        handoff_pack = _http_json(f"{base_url}/call", {"name": "content_handoff_pack", "arguments": {"batch_input": str(tmp_path), "write": False}})
        assert handoff_pack["result"]["schema"] == "video_knowledge_pipeline.content_handoff_pack.v1"

        repair_run = _http_json(
            f"{base_url}/call",
            {
                "name": "transcript_semantic_repair_run",
                "arguments": {"batch_input": str(tmp_path), "write": False, "execute_safe_actions": False},
            },
        )
        assert repair_run["result"]["schema"] == "video_knowledge_pipeline.transcript_semantic_repair_run.v1"
        assert repair_run["result"]["operator_boundary"]["preview_by_default"] is True
        assert repair_run["result"]["operator_boundary"]["llm_provider_call_requires_allow_llm"] is True

        options_request = urllib.request.Request(f"{base_url}/call", method="OPTIONS")
        with urllib.request.urlopen(options_request, timeout=10) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == "*"
    finally:
        server.shutdown()
        server.server_close()


def test_openclaw_http_bridge_rejects_unknown_tool() -> None:
    handler = build_handler()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = _http_json(
            f"http://127.0.0.1:{server.server_port}/call",
            {"name": "missing_tool", "arguments": {}},
            allow_http_error=True,
        )
        assert result["ok"] is False
        assert result["code"] == "invalid_request"
        assert "unsupported tool" in result["message"]
    finally:
        server.shutdown()
        server.server_close()


def _http_json(url: str, payload: dict | None = None, *, allow_http_error: bool = False) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if not allow_http_error:
            raise
        return json.loads(exc.read().decode("utf-8"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
