from __future__ import annotations

from pathlib import Path
from typing import Any

from .content_asset_status import content_asset_status
from .models import now_iso
from .openclaw_bridge_status import openclaw_bridge_status
from .openclaw_docker_contract import openclaw_docker_contract_check
from .path_defaults import workspace_root
from .storage import write_json
from .transcript_semantic_batch import transcript_semantic_batch_acceptance, transcript_semantic_repair_queue


def openclaw_live_smoke(
    *,
    bundle_dir: str | Path = "",
    compose_path: str | Path = "",
    host_root: str | Path = str(workspace_root()),
    container_root: str = "/mnt/used-by-codex",
    timeout_seconds: float = 2.0,
    output_dir: str | Path = "",
    semantic_batch_input: str | Path = "",
    semantic_target_bundle_count: int = 3,
    semantic_limit: int = 0,
    write_report: bool = False,
) -> dict[str, Any]:
    """Read-only live smoke for OpenClaw/VKP handoff readiness."""

    bridge = openclaw_bridge_status(timeout_seconds=timeout_seconds, check_health=True, check_task=False)
    docker = openclaw_docker_contract_check(compose_path, host_root=host_root, container_root=container_root) if compose_path else {"checked": False}
    content = content_asset_status(bundle_dir, write=False) if str(bundle_dir or "").strip() else {"checked": False}
    semantic = _semantic_batch_status(bundle_dir=bundle_dir, semantic_batch_input=semantic_batch_input, target_bundle_count=semantic_target_bundle_count, limit=semantic_limit)
    semantic_queue = _semantic_repair_queue_status(bundle_dir=bundle_dir, semantic_batch_input=semantic_batch_input, target_bundle_count=semantic_target_bundle_count, limit=semantic_limit)
    ok = bool(bridge.get("ok"))
    if docker.get("checked", True):
        ok = ok and bool(docker.get("ok"))
    if content.get("checked", True):
        ok = ok and bool(content.get("ok"))
    if semantic.get("checked", True):
        ok = ok and bool(semantic.get("ok"))
    result = {
        "schema": "video_knowledge_pipeline.openclaw_live_smoke.v1",
        "created_at": now_iso(),
        "ok": ok,
        "status": "ok" if ok else "not_ready",
        "bridge": bridge,
        "docker_contract": docker,
        "content_asset_status": content,
        "transcript_semantic_batch_acceptance": semantic,
        "transcript_semantic_repair_queue": semantic_queue,
        "operator_boundary": {
            "no_video_processing": True,
            "no_download": True,
            "no_cloud_calls": True,
            "read_only": True,
        },
        "next_actions": _next_actions(bridge=bridge, docker=docker, content=content, semantic=semantic, semantic_queue=semantic_queue),
        "write_report": write_report,
    }
    if write_report:
        out_dir = _report_output_dir(output_dir=output_dir, bundle_dir=bundle_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "openclaw-live-smoke-report.json"
        markdown_path = out_dir / "openclaw-live-smoke-report.md"
        result["report_json_path"] = str(json_path)
        result["report_markdown_path"] = str(markdown_path)
        write_json(json_path, result)
        markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    return result


def _semantic_batch_status(*, bundle_dir: str | Path, semantic_batch_input: str | Path, target_bundle_count: int, limit: int = 0) -> dict[str, Any]:
    source = semantic_batch_input or bundle_dir
    if not str(source or "").strip():
        return {"checked": False}
    target = int(target_bundle_count or 0) if semantic_batch_input else 1
    result = transcript_semantic_batch_acceptance(source, target_bundle_count=target, limit=limit, write=False)
    return {**result, "checked": True, "source": str(Path(source).expanduser())}



def _semantic_repair_queue_status(*, bundle_dir: str | Path, semantic_batch_input: str | Path, target_bundle_count: int, limit: int = 0) -> dict[str, Any]:
    source = semantic_batch_input or bundle_dir
    if not str(source or "").strip():
        return {"checked": False}
    target = int(target_bundle_count or 0) if semantic_batch_input else 1
    result = transcript_semantic_repair_queue(source, target_bundle_count=target, limit=limit, write=False)
    return {**result, "checked": True, "source": str(Path(source).expanduser())}
def _next_actions(*, bridge: dict[str, Any], docker: dict[str, Any], content: dict[str, Any], semantic: dict[str, Any], semantic_queue: dict[str, Any] | None = None) -> list[str]:
    actions: list[str] = []
    if not bridge.get("ok"):
        actions.extend(bridge.get("next_actions") or ["start_openclaw_http_bridge"])
    if docker.get("checked", True) and not docker.get("ok"):
        actions.append("fix_openclaw_docker_mount_or_env_contract")
    if content.get("checked", True) and not content.get("ok"):
        actions.extend(content.get("next_actions") or ["run_export_knowledge_note"])
    if semantic.get("checked", True) and not semantic.get("ok"):
        actions.extend(semantic.get("next_actions") or ["run_transcript_semantic_batch_acceptance_actions"])
    return _dedupe(actions) or ["openclaw_can_call_vkp_content_asset_status"]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _report_output_dir(*, output_dir: str | Path, bundle_dir: str | Path) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    if str(bundle_dir or "").strip():
        path = Path(bundle_dir).expanduser()
        if path.name == "webui-bundle":
            return (path / "exports").resolve()
        return path.resolve()
    return Path.cwd().resolve()


def _render_markdown(result: dict[str, Any]) -> str:
    bridge = result.get("bridge") if isinstance(result.get("bridge"), dict) else {}
    docker = result.get("docker_contract") if isinstance(result.get("docker_contract"), dict) else {}
    content = result.get("content_asset_status") if isinstance(result.get("content_asset_status"), dict) else {}
    semantic = result.get("transcript_semantic_batch_acceptance") if isinstance(result.get("transcript_semantic_batch_acceptance"), dict) else {}
    semantic_summary = semantic.get("summary") if isinstance(semantic.get("summary"), dict) else {}
    semantic_queue = result.get("transcript_semantic_repair_queue") if isinstance(result.get("transcript_semantic_repair_queue"), dict) else {}
    semantic_queue_summary = semantic_queue.get("summary") if isinstance(semantic_queue.get("summary"), dict) else {}
    lines = [
        "# OpenClaw Live Smoke Report",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Status: `{result.get('status')}`",
        f"- Bridge running: `{str(bool(bridge.get('running'))).lower()}`",
        f"- Bridge health URL: `{bridge.get('health_url', '')}`",
        f"- Docker contract: `{docker.get('status', 'not_checked')}`",
        f"- Content asset status: `{content.get('status', 'not_checked')}`",
        f"- Transcript semantic batch: `{semantic.get('status', 'not_checked')}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in result.get("next_actions") or []:
        lines.append(f"- `{action}`")
    lines.extend(
        [
            "",
            "## Operator Boundary",
            "",
            "- No video processing.",
            "- No download.",
            "- No cloud ASR or vision call.",
            "- No publication or knowledge-base writeback.",
            "",
            "## Content Card",
            "",
            f"- Material card: `{content.get('content_material_card_path', '')}`",
            f"- Allowed as inspiration: `{str(bool(content.get('allowed_as_inspiration'))).lower()}`",
            f"- Allowed as fact: `{str(bool(content.get('allowed_as_fact'))).lower()}`",
            f"- Publication allowed: `{str(bool(content.get('publication_allowed'))).lower()}`",
            "",
            "## Transcript Semantic Correction",
            "",
            f"- Status: `{semantic.get('status', 'not_checked')}`",
            f"- Source: `{semantic.get('source', '')}`",
            f"- Accepted: `{semantic_summary.get('accepted_count', 0)}`",
            f"- Not accepted: `{semantic_summary.get('not_accepted_count', 0)}`",
            f"- Candidates: `{semantic_summary.get('candidate_count', 0)}`",
            f"- Review required: `{semantic_summary.get('review_required_count', 0)}`",
            f"- Residual errors: `{semantic_summary.get('final_residual_error_total', 0)}`",
            f"- Repair queue action required: `{semantic_queue_summary.get('action_required_count', 0)}`",
            f"- Repair queue machine actions: `{semantic_queue_summary.get('machine_action_available_count', 0)}`",
            f"- Repair queue human review: `{semantic_queue_summary.get('human_review_required_count', 0)}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
