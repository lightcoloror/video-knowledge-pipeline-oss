from __future__ import annotations

import importlib.util
from pathlib import Path


CLIENT_PATH = Path(__file__).resolve().parents[1] / "examples" / "openclaw" / "openclaw_video_knowledge_call.py"


def load_client():
    spec = importlib.util.spec_from_file_location("openclaw_video_knowledge_call", CLIENT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_translate_container_path_to_windows_host_path() -> None:
    client = load_client()

    result = client.translate_container_path(
        "/mnt/used-by-codex/video-download-orchestrator/downloads/demo/download.mp4",
        container_root="/mnt/used-by-codex",
        host_root=r"D:\workspace",
    )

    assert result == r"D:\workspace\video-download-orchestrator\downloads\demo\download.mp4"


def test_build_ingest_payload_translates_media_and_workspace() -> None:
    client = load_client()

    payload = client.build_call_payload(
        "ingest",
        "/mnt/used-by-codex/videos/lesson.mp4",
        {
            "mount_container_root": "/mnt/used-by-codex",
            "mount_host_root": r"D:\workspace",
            "workspace": "/mnt/used-by-codex/video-knowledge-pipeline/openclaw-runs/lesson",
            "title": "课程",
            "max_frames": 12,
            "sample_interval": 3.0,
        },
    )

    assert payload["name"] == "openclaw_video_ingest"
    assert payload["arguments"]["media_path"] == r"D:\workspace\videos\lesson.mp4"
    assert payload["arguments"]["workspace"] == r"D:\workspace\video-knowledge-pipeline\openclaw-runs\lesson"
    assert payload["arguments"]["title"] == "课程"
    assert payload["arguments"]["max_frames"] == 12


def test_build_link_payload_keeps_download_confirmation_explicit() -> None:
    client = load_client()

    payload = client.build_call_payload(
        "link",
        "https://example.com/video",
        {
            "allow_download": False,
            "actor_id": "",
            "confirm_download": False,
            "confirm_sensitive": False,
            "ingest_after_download": False,
            "downloaded_media_path": "",
            "workspace": "",
            "title": "",
            "max_frames": 80,
            "sample_interval": 5.0,
        },
    )

    assert payload["name"] == "openclaw_video_link"
    assert payload["arguments"]["url_or_text"] == "https://example.com/video"
    assert payload["arguments"]["allow_download"] is False
    assert payload["arguments"]["confirm_download"] is False


def test_build_status_payload() -> None:
    client = load_client()

    payload = client.build_call_payload("status", "", {})

    assert payload["name"] == "openclaw_bridge_status"
    assert payload["arguments"]["check_health"] is True


def test_build_contract_payload() -> None:
    client = load_client()

    payload = client.build_call_payload(
        "contract",
        "",
        {"mount_container_root": "/mnt/used-by-codex", "mount_host_root": r"D:\workspace"},
    )

    assert payload["name"] == "openclaw_docker_contract_check"
    assert payload["arguments"]["container_root"] == "/mnt/used-by-codex"
    assert payload["arguments"]["host_root"] == r"D:\workspace"


def test_build_live_smoke_payload_translates_bundle_path() -> None:
    client = load_client()

    payload = client.build_call_payload(
        "live-smoke",
        "/mnt/used-by-codex/video-knowledge-pipeline/openclaw-runs/demo/webui-bundle",
        {
            "mount_container_root": "/mnt/used-by-codex",
            "mount_host_root": r"D:\workspace",
            "semantic_batch_input": "/mnt/used-by-codex/video-knowledge-pipeline/openclaw-runs",
            "semantic_target_bundle_count": 5,
            "semantic_limit": 5,
        },
    )

    assert payload["name"] == "openclaw_live_smoke"
    assert payload["arguments"]["bundle_dir"] == r"D:\workspace\video-knowledge-pipeline\openclaw-runs\demo\webui-bundle"
    assert payload["arguments"]["container_root"] == "/mnt/used-by-codex"
    assert payload["arguments"]["semantic_batch_input"] == r"D:\workspace\video-knowledge-pipeline\openclaw-runs"
    assert payload["arguments"]["semantic_target_bundle_count"] == 5
    assert payload["arguments"]["semantic_limit"] == 5


def test_build_handoff_payload_translates_vdo_paths() -> None:
    client = load_client()

    payload = client.build_call_payload(
        "handoff",
        "",
        {
            "mount_container_root": "/mnt/used-by-codex",
            "mount_host_root": r"D:\workspace",
            "manifest_path": "/mnt/used-by-codex/video-download-orchestrator/downloads/demo/.vdo/manifests/task.json",
            "summary_path": "/mnt/used-by-codex/video-download-orchestrator/downloads/demo/.vdo/reports/1/summary.json",
            "review_checklist_path": "/mnt/used-by-codex/video-download-orchestrator/downloads/demo/.vdo/reports/1/review-checklist.json",
            "media_path": "/mnt/used-by-codex/video-download-orchestrator/downloads/demo/download.mp4",
            "workspace": "/mnt/used-by-codex/video-knowledge-pipeline/openclaw-runs/demo",
            "title": "课程",
        },
    )

    assert payload["name"] == "openclaw_video_from_vdo_handoff"
    assert payload["arguments"]["manifest_path"] == r"D:\workspace\video-download-orchestrator\downloads\demo\.vdo\manifests\task.json"
    assert payload["arguments"]["summary_path"] == r"D:\workspace\video-download-orchestrator\downloads\demo\.vdo\reports\1\summary.json"
    assert payload["arguments"]["review_checklist_path"] == r"D:\workspace\video-download-orchestrator\downloads\demo\.vdo\reports\1\review-checklist.json"
    assert payload["arguments"]["media_path"] == r"D:\workspace\video-download-orchestrator\downloads\demo\download.mp4"
    assert payload["arguments"]["workspace"] == r"D:\workspace\video-knowledge-pipeline\openclaw-runs\demo"


def test_build_ingest_handoff_payload_translates_paths_and_keeps_execute_explicit() -> None:
    client = load_client()

    payload = client.build_call_payload(
        "ingest-handoff",
        "",
        {
            "mount_container_root": "/mnt/used-by-codex",
            "mount_host_root": r"D:\workspace",
            "handoff_path": "/mnt/used-by-codex/video-download-orchestrator/downloads/demo/handoff.json",
            "summary_path": "/mnt/used-by-codex/video-download-orchestrator/downloads/demo/.vdo/reports/1/summary.json",
            "review_checklist_path": "/mnt/used-by-codex/video-download-orchestrator/downloads/demo/.vdo/reports/1/review-checklist.json",
            "media_path": "/mnt/used-by-codex/video-download-orchestrator/downloads/demo/download.mp4",
            "workspace": "/mnt/used-by-codex/video-knowledge-pipeline/openclaw-runs/demo",
            "title": "课程",
            "execute": False,
            "max_frames": 9,
            "sample_interval": 4.0,
        },
    )

    assert payload["name"] == "openclaw_video_ingest_vdo_handoff"
    assert payload["arguments"]["execute"] is False
    assert payload["arguments"]["handoff_path"] == r"D:\workspace\video-download-orchestrator\downloads\demo\handoff.json"
    assert payload["arguments"]["media_path"] == r"D:\workspace\video-download-orchestrator\downloads\demo\download.mp4"
    assert payload["arguments"]["max_frames"] == 9


def test_build_content_status_payload_translates_bundle_path() -> None:
    client = load_client()

    payload = client.build_call_payload(
        "content-status",
        "/mnt/used-by-codex/video-knowledge-pipeline/openclaw-runs/demo/webui-bundle",
        {"mount_container_root": "/mnt/used-by-codex", "mount_host_root": r"D:\workspace"},
    )

    assert payload["name"] == "content_asset_status"
    assert payload["arguments"]["bundle_dir"] == r"D:\workspace\video-knowledge-pipeline\openclaw-runs\demo\webui-bundle"


def test_build_batch_content_status_payload_translates_batch_path() -> None:
    client = load_client()

    payload = client.build_call_payload(
        "batch-content-status",
        "/mnt/used-by-codex/video-knowledge-pipeline/openclaw-runs",
        {"mount_container_root": "/mnt/used-by-codex", "mount_host_root": r"D:\workspace"},
    )

    assert payload["name"] == "batch_content_asset_status"
    assert payload["arguments"]["batch_input"] == r"D:\workspace\video-knowledge-pipeline\openclaw-runs"


def test_build_content_handoff_pack_payload_translates_batch_path() -> None:
    client = load_client()

    payload = client.build_call_payload(
        "content-handoff-pack",
        "/mnt/used-by-codex/video-knowledge-pipeline/openclaw-runs",
        {"mount_container_root": "/mnt/used-by-codex", "mount_host_root": r"D:\workspace"},
    )

    assert payload["name"] == "content_handoff_pack"
    assert payload["arguments"]["batch_input"] == r"D:\workspace\video-knowledge-pipeline\openclaw-runs"
