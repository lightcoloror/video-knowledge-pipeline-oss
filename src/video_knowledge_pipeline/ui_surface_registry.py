from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import now_iso
from .powershell import quote_powershell_literal
from .storage import write_json, write_text_atomic


SCHEMA = "video_knowledge_pipeline.ui_surface_registry.v1"
REGISTRY_JSON_PATH = Path("ui-surface-registry.json")
REGISTRY_MARKDOWN_PATH = Path("ui-surface-registry.md")


@dataclass(frozen=True, slots=True)
class UISurface:
    surface_id: str
    artifact_key: str
    title: str
    group: str
    kind: str
    manifest_key: str
    default_path: str
    write_mode: str
    service_requirement: str
    stability: str
    description: str
    generation_command: str | None = None
    show_in_sidebar: bool = True
    show_in_artifacts: bool = True


GROUPS: tuple[tuple[str, str], ...] = (
    ("overview", "总览"),
    ("processing", "处理"),
    ("review", "审核"),
    ("editing", "编辑"),
    ("evaluation", "评测"),
    ("settings", "设置"),
)


SURFACES: tuple[UISurface, ...] = (
    UISurface(
        surface_id="video_workbench",
        artifact_key="video_workbench_html",
        title="视频知识工作台",
        group="overview",
        kind="hub",
        manifest_key="video_workbench_html",
        default_path="video-workbench.html",
        write_mode="read_only",
        service_requirement="none",
        stability="primary",
        description="唯一首页；汇总视频、时间线、证据、队列和专业页面入口。",
        generation_command=r".\scripts\video-knowledge.ps1 export-video-workbench {bundle}",
        show_in_artifacts=False,
    ),
    UISurface(
        surface_id="task_console",
        artifact_key="task_console",
        title="任务控制台",
        group="processing",
        kind="console",
        manifest_key="task_console",
        default_path="task-console.html",
        write_mode="read_only",
        service_requirement="none",
        stability="stable",
        description="查看任务历史、处理队列、失败项和恢复命令。",
        generation_command=r".\scripts\video-knowledge.ps1 export-task-console {bundle}",
    ),
    UISurface(
        surface_id="review",
        artifact_key="review_html",
        title="证据审核",
        group="review",
        kind="review",
        manifest_key="review_html",
        default_path="review.html",
        write_mode="loopback_writeback",
        service_requirement="review_loopback",
        stability="stable",
        description="通用时间线、OCR 与多模态证据审核；正式写回需要本地审核服务。",
    ),
    UISurface(
        surface_id="targeted_review",
        artifact_key="targeted_review_html",
        title="精确区间审核",
        group="review",
        kind="review",
        manifest_key="targeted_review_html",
        default_path="targeted-review.html",
        write_mode="loopback_writeback",
        service_requirement="targeted_review_loopback",
        stability="specialized",
        description="针对音频、字幕、画面、隐私、合规与发布问题的精确时间段审核。",
    ),
    UISurface(
        surface_id="subtitle_editor",
        artifact_key="subtitle_editor_html",
        title="双轨字幕编辑器",
        group="editing",
        kind="editor",
        manifest_key="subtitle_editor_html",
        default_path="subtitle-editor.html",
        write_mode="loopback_writeback",
        service_requirement="review_loopback",
        stability="stable",
        description="复用 MOYS 的播放器、波形、时间轴与双轨字幕编辑能力。",
        generation_command=r".\scripts\video-knowledge.ps1 prepare-subtitle-editor {bundle}",
    ),
    UISurface(
        surface_id="transcript_editor",
        artifact_key="transcript_editor_html",
        title="旧版逐字稿编辑器",
        group="editing",
        kind="editor",
        manifest_key="transcript_editor_html",
        default_path="transcript-editor.html",
        write_mode="download_then_apply",
        service_requirement="none",
        stability="compatibility",
        description="BiliNote 风格兼容界面；新字幕时序编辑优先使用双轨字幕编辑器。",
        generation_command=r".\scripts\video-knowledge.ps1 prepare-transcript-edit-session {bundle}",
    ),
    UISurface(
        surface_id="smart_summary_section_editor",
        artifact_key="smart_summary_section_editor_html",
        title="智能总结章节编辑器",
        group="editing",
        kind="editor",
        manifest_key="smart_summary_section_editor_html",
        default_path="smart-summary-section-editor.html",
        write_mode="download_then_apply",
        service_requirement="none",
        stability="stable",
        description="按证据章节修订智能总结，导出后通过显式 apply 流程安装。",
        generation_command=r".\scripts\video-knowledge.ps1 smart-summary-section-editor {bundle}",
    ),
    UISurface(
        surface_id="quality_console",
        artifact_key="quality_console",
        title="质量控制台",
        group="evaluation",
        kind="console",
        manifest_key="quality_console",
        default_path="quality-console.html",
        write_mode="read_only",
        service_requirement="none",
        stability="stable",
        description="汇总 ASR、时间戳与总结质量状态以及安全恢复命令。",
        generation_command=r".\scripts\video-knowledge.ps1 export-quality-console {bundle}",
    ),
    UISurface(
        surface_id="vision_review_queue",
        artifact_key="vision_review_queue_html",
        title="多模态疑难队列",
        group="evaluation",
        kind="queue",
        manifest_key="vision_review_queue_html",
        default_path="vision-review-queue.html",
        write_mode="read_only",
        service_requirement="none",
        stability="specialized",
        description="查看多模态候选批次、失败项和显式重试命令。",
        generation_command=r".\scripts\video-knowledge.ps1 vision-review-queue {bundle} --min-score 10 --batch-size 10 --max-items 0",
    ),
    UISurface(
        surface_id="multimodal_sample_review",
        artifact_key="multimodal_sample_review_html",
        title="多模态抽样评估",
        group="evaluation",
        kind="evaluation",
        manifest_key="multimodal_sample_review_html",
        default_path="multimodal-sample-review.html",
        write_mode="download_then_apply",
        service_requirement="none",
        stability="specialized",
        description="人工抽样评估多模态准确性、帮助率和证据充分率。",
        generation_command=r".\scripts\video-knowledge.ps1 multimodal-sample-review {bundle} --sample-size 30",
    ),
    UISurface(
        surface_id="quality_benchmark",
        artifact_key="quality_benchmark_html",
        title="ASR 质量基准",
        group="evaluation",
        kind="evaluation",
        manifest_key="quality_benchmark_html",
        default_path="quality-benchmark.html",
        write_mode="read_only",
        service_requirement="none",
        stability="specialized",
        description="展示固定抽样集的 ASR 与时间戳质量指标。",
    ),
    UISurface(
        surface_id="quality_benchmark_review",
        artifact_key="quality_benchmark_review_html",
        title="ASR 人工参考评分",
        group="evaluation",
        kind="evaluation",
        manifest_key="quality_benchmark_review_html",
        default_path="quality-benchmark-review.html",
        write_mode="download_then_apply",
        service_requirement="none",
        stability="specialized",
        description="填写基准片段人工真值并导出评分文件。",
    ),
    UISurface(
        surface_id="summary_blind_review",
        artifact_key="summary_blind_review_html",
        title="总结匿名盲评",
        group="evaluation",
        kind="evaluation",
        manifest_key="summary_blind_review_html",
        default_path="summary-blind-review.html",
        write_mode="download_then_apply",
        service_requirement="none",
        stability="specialized",
        description="匿名比较多套总结候选，避免模型或来源标签影响判断。",
    ),
    UISurface(
        surface_id="model_settings",
        artifact_key="model_settings",
        title="模型与 API 设置",
        group="settings",
        kind="settings",
        manifest_key="model_settings",
        default_path="model-settings.html",
        write_mode="loopback_writeback",
        service_requirement="model_settings_loopback",
        stability="stable",
        description="查看已脱敏的 Provider、本地模型能力与配置状态。",
        generation_command=r".\scripts\video-knowledge.ps1 export-task-console {bundle}",
    ),
)


def ui_surface_definitions() -> list[dict[str, Any]]:
    return [asdict(surface) for surface in SURFACES]


def resolve_ui_surface_registry(
    bundle_dir: str | Path,
    manifest: dict[str, Any],
    *,
    current_surface: str | None = None,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    bundle_literal = quote_powershell_literal(str(root))
    resolved: list[dict[str, Any]] = []
    for surface in SURFACES:
        configured = str(manifest.get(surface.manifest_key) or surface.default_path).strip()
        target = _bundle_path(root, configured)
        is_current = surface.surface_id == current_surface
        exists = target.exists() and target.is_file()
        command = surface.generation_command.format(bundle=bundle_literal) if surface.generation_command else None
        status = "current" if is_current else ("ready" if exists else "not_generated")
        resolved.append(
            {
                **asdict(surface),
                "configured_path": configured,
                "absolute_path": str(target),
                "href": _surface_href(root, target),
                "exists": exists,
                "is_current": is_current,
                "status": status,
                "status_label": _status_label(status),
                "write_mode_label": _write_mode_label(surface.write_mode),
                "service_label": _service_label(surface.service_requirement),
                "resolved_generation_command": command,
            }
        )

    grouped = []
    for group_key, group_label in GROUPS:
        items = [item for item in resolved if item["group"] == group_key and item["show_in_sidebar"]]
        grouped.append(
            {
                "key": group_key,
                "label": group_label,
                "surface_count": len(items),
                "ready_count": sum(1 for item in items if item["exists"] or item["is_current"]),
                "items": items,
            }
        )
    ready_count = sum(1 for item in resolved if item["exists"] or item["is_current"])
    return {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "current_surface": current_surface,
        "surface_count": len(resolved),
        "ready_count": ready_count,
        "summary": {
            "ready_count": ready_count,
            "not_generated_count": sum(1 for item in resolved if item["status"] == "not_generated"),
            "write_capable_count": sum(1 for item in resolved if item["write_mode"] != "read_only"),
            "service_required_count": sum(1 for item in resolved if item["service_requirement"] != "none"),
        },
        "groups": grouped,
        "surfaces": resolved,
    }


def export_ui_surface_registry(
    bundle_dir: str | Path,
    manifest: dict[str, Any] | None = None,
    *,
    current_surface: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Persist the authoritative UI inventory without generating specialist pages.

    The registry describes existing pages, their generation commands and their
    writeback boundaries. Exporting it never starts a service or executes a
    page-generation command.
    """

    root = Path(bundle_dir).expanduser().resolve()
    registry = resolve_ui_surface_registry(root, manifest or {}, current_surface=current_surface)
    registry.update(
        {
            "generated_at": now_iso(),
            "paths": {
                "json": str(root / REGISTRY_JSON_PATH),
                "markdown": str(root / REGISTRY_MARKDOWN_PATH),
            },
            "operator_boundary": {
                "inventory_only": True,
                "no_surface_generation": True,
                "no_service_started": True,
                "no_cloud_call": True,
                "no_media_processing": True,
            },
        }
    )
    if write:
        write_json(root / REGISTRY_JSON_PATH, registry)
        write_text_atomic(root / REGISTRY_MARKDOWN_PATH, render_ui_surface_registry_markdown(registry))
    return registry


def render_ui_surface_registry_markdown(registry: dict[str, Any]) -> str:
    summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    lines = [
        "# VKP UI Surface Registry",
        "",
        f"- Schema: `{registry.get('schema', '')}`",
        f"- Bundle: `{registry.get('bundle_dir', '')}`",
        f"- Current surface: `{registry.get('current_surface') or 'none'}`",
        f"- Ready: `{summary.get('ready_count', 0)}/{registry.get('surface_count', 0)}`",
        f"- Not generated: `{summary.get('not_generated_count', 0)}`",
        f"- Write-capable: `{summary.get('write_capable_count', 0)}`",
        f"- Service required: `{summary.get('service_required_count', 0)}`",
        "",
        "The registry is inventory-only. It does not generate pages, start loopback services, or execute recovery commands.",
        "",
        "| Group | Surface | Status | Stability | Write mode | Service | Path / command |",
        "|---|---|---|---|---|---|---|",
    ]
    for group in registry.get("groups") or []:
        if not isinstance(group, dict):
            continue
        group_label = str(group.get("label") or group.get("key") or "")
        for surface in group.get("items") or []:
            if not isinstance(surface, dict):
                continue
            command = str(surface.get("resolved_generation_command") or "")
            target = str(surface.get("href") or surface.get("configured_path") or "")
            action = target if surface.get("exists") or surface.get("is_current") else (command or target)
            lines.append(
                "| "
                + " | ".join(
                    _md_cell(value)
                    for value in (
                        group_label,
                        surface.get("title") or surface.get("surface_id") or "",
                        surface.get("status_label") or surface.get("status") or "",
                        surface.get("stability") or "",
                        surface.get("write_mode_label") or surface.get("write_mode") or "",
                        surface.get("service_label") or surface.get("service_requirement") or "",
                        action,
                    )
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def surface_artifact_cards(registry: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for surface in registry.get("surfaces") or []:
        if not isinstance(surface, dict) or not surface.get("show_in_artifacts"):
            continue
        cards.append(
            {
                "key": str(surface.get("artifact_key") or ""),
                "label": str(surface.get("title") or ""),
                "path": str(surface.get("absolute_path") or ""),
                "href": str(surface.get("href") or ""),
                "exists": bool(surface.get("exists")),
                "status": str(surface.get("status_label") or ""),
                "surface_id": str(surface.get("surface_id") or ""),
                "surface_group": str(surface.get("group") or ""),
                "write_mode": str(surface.get("write_mode") or ""),
                "service_requirement": str(surface.get("service_requirement") or ""),
                "generation_command": surface.get("resolved_generation_command"),
            }
        )
    return cards


def _bundle_path(root: Path, configured: str) -> Path:
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _surface_href(root: Path, target: Path) -> str:
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return target.as_uri()


def _status_label(status: str) -> str:
    return {"current": "当前首页", "ready": "已生成", "not_generated": "未生成"}.get(status, status)


def _write_mode_label(mode: str) -> str:
    return {
        "read_only": "只读",
        "loopback_writeback": "本地服务写回",
        "download_then_apply": "下载后显式应用",
    }.get(mode, mode)


def _service_label(service: str) -> str:
    return {
        "none": "无需服务",
        "review_loopback": "审核服务",
        "targeted_review_loopback": "精确审核服务",
        "model_settings_loopback": "设置服务",
    }.get(service, service)


def _md_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
