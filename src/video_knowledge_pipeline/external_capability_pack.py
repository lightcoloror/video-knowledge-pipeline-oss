from __future__ import annotations

from pathlib import Path
from typing import Any

from .content_asset_status import content_asset_status
from .local_vlm_server_adapter import local_vlm_adapter_plan
from .long_video_memory_pack import build_long_video_memory_pack
from .models import now_iso
from .external_reuse_run_artifacts import ps_quote, register_external_reuse_run
from .storage import read_json, write_json
from .video_moment_index import build_video_moment_index
from .video_rag_pack import build_video_rag_pack

SCHEMA = "video_knowledge_pipeline.external_capability_pack.v1"


def build_external_capability_pack(
    bundle_dir: str | Path,
    *,
    query: str = "",
    write: bool = True,
) -> dict[str, Any]:
    """Bundle the reusable external-project-inspired capabilities for one VKP bundle."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")

    moment_index = build_video_moment_index(root, query=query, write=write)
    long_memory = build_long_video_memory_pack(root, write=write)
    rag_pack = build_video_rag_pack(root, query=query, write=write)
    vlm_plan = local_vlm_adapter_plan(output_dir=str(root / "exports"), write=write)
    content_status = content_asset_status(root, write=False)
    capabilities = [
        _capability(
            key="long_video_layered_summary",
            label="长视频分层总结",
            source_projects=["MovieChat"],
            status="ready",
            artifacts=[
                str(root / "exports" / "long-video-memory-pack.json"),
                str(root / "exports" / "long-video-memory-pack.md"),
            ],
            commands=[f".\\scripts\\video-knowledge.ps1 long-video-memory-pack {_ps(root)}"],
            notes=[
                "Short memories preserve local timeline detail; long memories compress course flow for smart-summary.",
                "No online LLM is required for this pack; final prose can later use Codex or provider LLM.",
            ],
        ),
        _capability(
            key="time_localization",
            label="时间定位 / Query-to-moment",
            source_projects=["VTimeLLM", "VideoRAG"],
            status="ready",
            artifacts=[
                str(root / "exports" / "video-moment-index.json"),
                str(root / "exports" / "video-moment-index.md"),
            ],
            commands=[
                f".\\scripts\\video-knowledge.ps1 video-moment-index {_ps(root)} --query \"<关键词或疑难点>\"",
                f".\\scripts\\video-knowledge.ps1 export-task-console {_ps(root)}",
            ],
            notes=[
                "Moment index keeps start/end timestamps, timeline indexes, keywords, and evidence paths.",
                "Task console uses this index for local browser-side search.",
            ],
        ),
        _capability(
            key="video_rag",
            label="视频 RAG 检索包",
            source_projects=["VideoRAG"],
            status="ready",
            artifacts=[
                str(root / "exports" / "video-rag-pack.json"),
                str(root / "exports" / "video-rag-pack.md"),
                str(root / "exports" / "video-rag-chunks.jsonl"),
            ],
            commands=[f".\\scripts\\video-knowledge.ps1 video-rag-pack {_ps(root)} --query \"<问题或术语>\""],
            notes=[
                "Outputs JSONL retrieval units for future vector or graph backends.",
                "Does not start a vector DB and does not call cloud models.",
            ],
        ),
        _capability(
            key="local_vlm_adapter",
            label="本地 VLM adapter",
            source_projects=["Qwen2.5-VL/Qwen3-VL", "InternVL", "LLaVA-OneVision"],
            status="adapter_contract_ready",
            artifacts=[str(root / "exports" / "local-vlm-adapter-plan.json"), str(root / "exports" / "local-vlm-adapter-plan.md")],
            commands=[
                ".\\scripts\\video-knowledge.ps1 local-vlm-adapter-plan --output-dir " + _ps(root / "exports") + " --write",
                f".\\scripts\\video-knowledge.ps1 vision-provider-smoke --provider local_qwen_vl --bundle-dir {_ps(root)}",
            ],
            notes=[
                "VKP reuses the existing OpenAI-compatible vision provider layer.",
                "Model repositories stay outside VKP; serve them over HTTP or subprocess adapter.",
            ],
        ),
        _capability(
            key="content_material_generation",
            label="内容素材生成",
            source_projects=["VSummary", "BiliNote", "VKP content asset contract"],
            status=str(content_status.get("status") or "export_required"),
            artifacts=_content_artifacts(content_status),
            commands=[f".\\scripts\\video-knowledge.ps1 export-knowledge-note {_ps(root)}"],
            notes=[
                "Reuses VKP export layer: key segments, short-video script drafts, highlight post drafts, material card.",
                "All material remains review_required=true and publication_allowed=false.",
            ],
        ),
    ]
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "created_at": now_iso(),
        "query": query,
        "status": "ready",
        "capabilities": capabilities,
        "artifacts": {
            "external_capability_pack_json": str(root / "exports" / "external-capability-pack.json"),
            "external_capability_pack_markdown": str(root / "exports" / "external-capability-pack.md"),
            "video_moment_index": str(root / "exports" / "video-moment-index.json"),
            "long_video_memory_pack": str(root / "exports" / "long-video-memory-pack.json"),
            "video_rag_pack": str(root / "exports" / "video-rag-pack.json"),
            "content_material_card": content_status.get("content_material_card_path", ""),
        },
        "summaries": {
            "moment_index": moment_index.get("summary", {}),
            "long_video_memory_pack": long_memory.get("summary", {}),
            "video_rag_pack": rag_pack.get("summary", {}),
            "content_asset_status": {
                "status": content_status.get("status"),
                "allowed_as_inspiration": content_status.get("allowed_as_inspiration", False),
                "allowed_as_fact": content_status.get("allowed_as_fact", False),
                "publication_allowed": content_status.get("publication_allowed", False),
            },
            "local_vlm_adapter": {
                "default_recommendation": vlm_plan.get("default_recommendation"),
                "implemented_provider_profiles": vlm_plan.get("implemented_provider_profiles", {}),
            },
        },
        "operator_boundary": {
            "no_download": True,
            "no_cloud_call": True,
            "no_model_server_started": True,
            "content_assets_review_only": True,
        },
        "write": bool(write),
    }
    if write:
        latest_manifest = read_json(manifest_path)
        if isinstance(latest_manifest, dict):
            manifest = latest_manifest
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "external-capability-pack.json", result)
        (exports / "external-capability-pack.md").write_text(_render_markdown(result), encoding="utf-8")
        manifest["external_capability_pack"] = "exports/external-capability-pack.json"
        manifest["external_capability_pack_markdown"] = "exports/external-capability-pack.md"
        manifest["mcp_external_capability_pack_args"] = "mcp-external-capability-pack.args.json"
        write_json(root / "mcp-external-capability-pack.args.json", {"bundle_dir": str(root), "query": query, "write": True})
        write_json(manifest_path, manifest)
        content_ready = str(content_status.get("status") or "") == "ready_for_inspiration_review"
        failed_items = [] if content_ready else [{"id": "content_material_generation", "reason": str(content_status.get("status") or "content_asset_not_ready"), "detail": "Content assets are not ready for inspiration review; run export-knowledge-note if needed."}]
        register_external_reuse_run(
            root,
            run_type="external_capability_pack",
            title="External capability reuse pack",
            result=result,
            status="needs_input" if failed_items else "completed",
            failed_items=failed_items,
            retry_command=f".\\scripts\\video-knowledge.ps1 external-capability-pack {ps_quote(root)}",
            next_actions=[] if not failed_items else ["Run export-knowledge-note, then rebuild external-capability-pack."],
            write=True,
        )
    return result


def _capability(
    *,
    key: str,
    label: str,
    source_projects: list[str],
    status: str,
    artifacts: list[str],
    commands: list[str],
    notes: list[str],
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "source_projects": source_projects,
        "status": status,
        "artifacts": artifacts,
        "commands": commands,
        "notes": notes,
    }


def _content_artifacts(content_status: dict[str, Any]) -> list[str]:
    paths = content_status.get("content_asset_paths") if isinstance(content_status.get("content_asset_paths"), dict) else {}
    preferred = [
        "smart_summary_path",
        "key_segments_path",
        "short_video_script_drafts_path",
        "highlight_post_drafts_path",
        "content_material_card_path",
        "content_material_card_markdown_path",
    ]
    values = [paths.get(key, "") for key in preferred]
    values.extend(
        [
            content_status.get("content_material_card_path", ""),
            content_status.get("content_material_card_markdown_path", ""),
        ]
    )
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result

def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# External Video Capability Pack",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Title: {result.get('title')}",
        f"- Bundle: `{result.get('bundle_dir')}`",
        f"- Query: `{result.get('query') or ''}`",
        "",
    ]
    for cap in result.get("capabilities") or []:
        lines.extend(
            [
                f"## {cap.get('label')}",
                "",
                f"- Key: `{cap.get('key')}`",
                f"- Status: `{cap.get('status')}`",
                f"- Source projects: `{', '.join(cap.get('source_projects') or [])}`",
                "",
                "Artifacts:",
            ]
        )
        for artifact in cap.get("artifacts") or []:
            lines.append(f"- `{artifact}`")
        lines.extend(["", "Commands:", ""])
        for command in cap.get("commands") or []:
            lines.append(f"```powershell\n{command}\n```")
        lines.extend(["Notes:", ""])
        for note in cap.get("notes") or []:
            lines.append(f"- {note}")
        lines.append("")
    lines.extend(
        [
            "## Operator Boundary",
            "",
            "- 不下载视频。",
            "- 不调用云模型。",
            "- 不启动本地模型服务。",
            "- 内容素材只作为待审核灵感/证据，不允许自动发布。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _ps(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"
