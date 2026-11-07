from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact_freshness import validate_dependency_snapshot
from .bundle_status import bundle_status_report
from .knowledge_coverage import audit_knowledge_coverage
from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .review_session import review_closure_status
from .storage import bundle_write_lock, read_json, write_json

ACCEPTANCE_SCHEMA = "lecture_acceptance_check.v1"


def acceptance_check(bundle_dir: str | Path, *, refresh: bool = True, write: bool = True) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    report_path = root / "acceptance-check.json"
    markdown_path = root / "acceptance-check.md"
    args_path = root / "mcp-acceptance-check.args.json"

    setup_blockers = _setup_blockers(root, manifest_path, timeline_path)
    if setup_blockers:
        report = _report(
            root=root,
            status="incomplete",
            summary={"setup": "incomplete"},
            blockers=setup_blockers,
            next_action={
                "key": "repair_bundle_setup",
                "label": "修复 bundle 基础文件",
                "mcp_tool": "",
                "hint": "manifest.json 和 timeline.json 是验收的最低输入。",
            },
            report_path=report_path,
            markdown_path=markdown_path,
            args_path=args_path,
        )
        if write:
            _write_acceptance_outputs(root, manifest_path, report, report_path, markdown_path, args_path)
        return report

    coverage_result = audit_knowledge_coverage(root, write=write) if refresh else _read_or_audit_coverage(root, write=write)
    coverage = coverage_result.get("coverage") if isinstance(coverage_result.get("coverage"), dict) else {}
    status_result = bundle_status_report(root, refresh=refresh, write=write)
    controlled = status_result.get("controlled_execution") if isinstance(status_result.get("controlled_execution"), dict) else {}
    note_quality = _note_quality(root)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest = manifest if isinstance(manifest, dict) else {}
    review_lifecycle = _review_lifecycle(root, manifest)
    review_closure = _review_closure(root)
    provider_matrix = _provider_matrix(root, manifest)
    quality_gates = _quality_gate_evidence(root)

    blockers = _acceptance_blockers(
        coverage,
        controlled,
        note_quality,
        review_lifecycle,
        quality_gates,
    )
    status = _acceptance_status(
        coverage,
        controlled,
        blockers,
        note_quality,
        review_lifecycle,
        quality_gates,
    )
    next_action = _acceptance_next_action(
        status,
        coverage,
        controlled,
        note_quality,
        review_lifecycle,
        provider_matrix,
        quality_gates,
    )
    report = _report(
        root=root,
        status=status,
        summary=_summary(
            coverage,
            controlled,
            note_quality,
            review_lifecycle,
            review_closure,
            quality_gates,
        ),
        blockers=blockers,
        next_action=next_action,
        report_path=report_path,
        markdown_path=markdown_path,
        args_path=args_path,
        coverage=coverage,
        controlled_execution=controlled,
        note_quality=note_quality,
        review_lifecycle=review_lifecycle,
        review_closure=review_closure,
        provider_matrix=provider_matrix,
        quality_gates=quality_gates,
    )
    if write:
        _write_acceptance_outputs(root, manifest_path, report, report_path, markdown_path, args_path)
    return report


def render_acceptance_check_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    next_action = report.get("next_action") if isinstance(report.get("next_action"), dict) else {}
    lines = [
        "---",
        "type: lecture-acceptance-check",
        f'created: "{report.get("checked_at", now_iso())}"',
        "---",
        "",
        "# Bundle 验收检查",
        "",
        f"- 状态：`{report.get('status', 'unknown')}`",
        f"- Bundle：`{report.get('bundle_dir', '')}`",
        f"- 下一步：`{next_action.get('key', '')}` / {next_action.get('label', '')}",
        f"- 建议工具：`{next_action.get('mcp_tool', '')}`",
        f"- 命令：`{_md_cell(str(next_action.get('command') or ''))}`",
        "",
        "## 摘要",
        "",
        "| 项目 | 状态 |",
        "|---|---|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{_md_cell(str(value))}` |")
    if summary.get("visual_applicability") == "not_applicable":
        lines.extend(["", "## OCR 执行与画面覆盖", "",
                      "- 不适用：已核验源文件仅含音轨，且未提供附加视觉材料。未执行 OCR，不作为识别成功或页面覆盖证明。"])
    lines.extend(["", "## Blockers", "", "| Key | Severity | Detail | Next Action |", "|---|---|---|---|"])
    blockers = [item for item in report.get("blockers") or [] if isinstance(item, dict)]
    if blockers:
        for item in blockers:
            lines.append(
                "| `{key}` | `{severity}` | {detail} | {next_action} |".format(
                    key=item.get("key", ""),
                    severity=item.get("severity", ""),
                    detail=_md_cell(str(item.get("detail") or "")),
                    next_action=_md_cell(str(item.get("next_action") or "")),
                )
            )
    else:
        lines.append("| none |  |  |  |")
    lines.extend(
        [
            "",
            "## 覆盖缺口",
            "",
            f"- Semantic missing：`{summary.get('semantic_missing', 0)}`",
            f"- Temporal missing：`{summary.get('temporal_missing', 0)}`",
            f"- Provider health：`{summary.get('provider_health', 'not_checked')}` / safe `{summary.get('provider_safe_to_execute', None)}`",
            f"- Export freshness：`{summary.get('export_freshness', 'unknown')}`",
        ]
    )
    review_queue = summary.get("review_queue") if isinstance(summary.get("review_queue"), dict) else {}
    if review_queue:
        lines.extend(
            [
                "",
                "## 分组复核队列",
                "",
                f"- Transcript review targets：`{review_queue.get('transcript_review_targets', 0)}`",
                f"- OCR review targets：`{review_queue.get('ocr_review_targets', 0)}`",
                f"- Semantic visual missing：`{review_queue.get('semantic_visual_missing', 0)}`",
                f"- Temporal visual missing：`{review_queue.get('temporal_visual_missing', 0)}`",
                f"- Deduplicated queue sample：`{review_queue.get('deduplicated_sample_indexes', [])}`",
            ]
        )
    provider_matrix = report.get("provider_matrix") if isinstance(report.get("provider_matrix"), dict) else {}
    matrix_rows = provider_matrix.get("providers") if isinstance(provider_matrix.get("providers"), list) else []
    if provider_matrix:
        lines.extend(
            [
                "",
                "## Provider Matrix",
                "",
                f"- Status：`{provider_matrix.get('status', '')}`",
                f"- Recommended：`{provider_matrix.get('recommended_provider', '')}`",
                f"- Report：`{provider_matrix.get('report_path', '')}`",
                "",
                "| Provider | Model | Ready | Status | Error |",
                "|---|---|---:|---|---|",
            ]
        )
        for row in matrix_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| `{row.get('provider', '')}` | `{row.get('model', '')}` | `{row.get('safe_to_execute', False)}` | `{row.get('status', '')}` | `{row.get('error_class', '')}` |"
            )
    review_lifecycle = report.get("review_lifecycle") if isinstance(report.get("review_lifecycle"), dict) else {}
    review_closure = report.get("review_closure") if isinstance(report.get("review_closure"), dict) else {}
    lines.extend(
        [
            "",
            "## Review Lifecycle",
            "",
            f"- State：`{review_lifecycle.get('state', 'not_prepared')}`",
            f"- Template prepared：`{review_lifecycle.get('review_template_prepared', False)}`",
            f"- Review notes imported：`{review_lifecycle.get('review_notes_imported', False)}`",
            f"- Open targets：`{review_lifecycle.get('review_targets_open', 0)}`",
            f"- Listed targets：`{review_lifecycle.get('review_targets_listed', 0)}`",
            f"- Template：`{review_lifecycle.get('review_template_path', '')}`",
            f"- Review notes：`{review_lifecycle.get('review_notes_path', '')}`",
            f"- Closure open：`{review_closure.get('open', 0)}`",
            f"- Closure closed：`{review_closure.get('closed', 0)}`",
            f"- Closure status：`{review_closure.get('report_markdown_path', '')}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _setup_blockers(root: Path, manifest_path: Path, timeline_path: Path) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not root.exists():
        blockers.append(_blocker("missing_bundle_dir", "blocking", str(root), "确认 bundle_dir 路径。"))
    if not manifest_path.exists():
        blockers.append(_blocker("missing_manifest", "blocking", str(manifest_path), "重新生成 WebUI bundle。"))
    if not timeline_path.exists():
        blockers.append(_blocker("missing_timeline", "blocking", str(timeline_path), "重新刷新 lecture review outputs。"))
    return blockers


def _read_or_audit_coverage(root: Path, *, write: bool) -> dict[str, Any]:
    path = root / "knowledge-coverage.json"
    data = read_json(path) if path.exists() else {}
    if isinstance(data, dict) and data.get("schema") == "lecture_knowledge_coverage.v1":
        return {"coverage": data}
    return audit_knowledge_coverage(root, write=write)


def _acceptance_blockers(
    coverage: dict[str, Any],
    controlled: dict[str, Any],
    note_quality: dict[str, Any],
    review_lifecycle: dict[str, Any],
    quality_gates: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    provider_blocked = _provider_blocked(controlled)
    semantic_missing = int(coverage.get("semantic_frame_without_analysis") or 0)
    temporal_missing = int(coverage.get("temporal_sequence_without_analysis") or 0)
    coverage_blockers = coverage.get("blockers") if isinstance(coverage.get("blockers"), list) else []
    transcript_quality = (
        quality_gates.get("transcript")
        if isinstance(quality_gates.get("transcript"), dict)
        else {}
    )
    summary_quality = (
        quality_gates.get("smart_summary")
        if isinstance(quality_gates.get("smart_summary"), dict)
        else {}
    )
    semantic_correction = (
        coverage.get("transcript_semantic_correction")
        if isinstance(coverage.get("transcript_semantic_correction"), dict)
        else {}
    )

    if transcript_quality.get("classification") in {"failed", "invalid"}:
        blockers.append(
            _blocker(
                "transcript_quality_failed",
                "blocking",
                str(transcript_quality.get("detail") or "transcript quality gate failed"),
                "修复逐字稿缺口并重新运行 transcript-quality-gate。",
            )
        )
    elif transcript_quality.get("classification") == "warning":
        blockers.append(
            _blocker(
                "transcript_quality_warning",
                "known_gap",
                str(transcript_quality.get("detail") or "transcript quality has warnings"),
                "保留候选稿并复核列明的逐字稿警告。",
            )
        )
    if semantic_correction.get("required"):
        semantic_status = str(semantic_correction.get("status") or "not_started")
        discovery_status = str(
            semantic_correction.get("candidate_discovery_status") or "not_planned"
        )
        semantic_review_count = int(
            semantic_correction.get("review_required_count") or 0
        )
        completion_gate = (
            semantic_correction.get("completion_gate")
            if isinstance(semantic_correction.get("completion_gate"), dict)
            else {}
        )
        required_next_action_key = str(
            semantic_correction.get("required_next_action_key")
            or semantic_correction.get("next_action_key")
            or "inspect_bundle"
        )
        required_action_kind = str(
            semantic_correction.get("required_action_kind")
            or completion_gate.get("required_action_kind")
            or "machine"
        )
        semantic_incomplete = not bool(semantic_correction.get("ok"))
        if semantic_incomplete and required_action_kind != "human":
            blockers.append(
                _blocker(
                    "transcript_semantic_correction_incomplete",
                    "machine_action_available",
                    (
                        f"semantic_status={semantic_status}; "
                        f"candidate_discovery={discovery_status}; "
                        f"required_next_action={required_next_action_key}"
                    ),
                    "继续有界候选扫描、强校验、闭合和下游重生成；不得把未扫描或陈旧结果视为完成。",
                )
            )
        elif semantic_incomplete or semantic_review_count:
            blockers.append(
                _blocker(
                    "transcript_semantic_high_risk_review",
                    "human_review_required",
                    (
                        f"{semantic_review_count} semantic correction candidates still require review; "
                        f"required_next_action={required_next_action_key}."
                    ),
                    "复核否定、条件、例外、实体、日期、金额和比例等高风险候选。",
                )
            )
    if summary_quality.get("classification") in {"failed", "invalid"}:
        blockers.append(
            _blocker(
                "smart_summary_quality_failed",
                "blocking",
                str(summary_quality.get("detail") or "smart summary quality gate failed"),
                "修复证据或总结后重新运行 smart-summary-quality-check。",
            )
        )
    elif summary_quality.get("classification") == "human_review_required":
        blockers.append(
            _blocker(
                "smart_summary_human_key_points_missing",
                "human_review_required",
                str(summary_quality.get("detail") or "human key point recall is not evaluated"),
                "补充人工关键点基准并重新运行 smart-summary-quality-check。",
            )
        )

    if provider_blocked:
        blockers.append(
            _blocker(
                "provider_health_failed",
                "blocking",
                f"provider={controlled.get('provider_health_status', 'not_checked')}; safe={controlled.get('provider_health_safe_to_execute')}",
                "修复 provider 或切换 provider 后重新运行 vision-execution-preflight --check-provider。",
            )
        )

    if semantic_missing:
        severity = "blocked_by_provider" if provider_blocked else "machine_action_available"
        review_hint = _review_resolution_hint(review_lifecycle)
        blockers.append(
            _blocker(
                "semantic_frame_without_analysis",
                severity,
                f"{semantic_missing} semantic/mixed timeline items still need visual_understanding.",
                f"运行 run_multimodal_frame_analysis，或导入人工审核结果。{review_hint}",
            )
        )

    if temporal_missing:
        severity = "blocked_by_provider" if provider_blocked else "machine_action_available"
        review_hint = _review_resolution_hint(review_lifecycle)
        blockers.append(
            _blocker(
                "temporal_sequence_without_analysis",
                severity,
                f"{temporal_missing} temporal/mixed timeline items still need temporal_visual_understanding.",
                f"运行 run_temporal_visual_analysis，或导入人工审核结果。{review_hint}",
            )
        )

    for channel in coverage_blockers:
        if not isinstance(channel, dict):
            continue
        key = str(channel.get("key") or "")
        if key in {
            "semantic_frame_understanding",
            "temporal_visual_understanding",
            "transcript_semantic_correction",
        }:
            continue
        if int(channel.get("blocker_count") or 0) <= 0:
            continue
        blockers.append(
            _blocker(
                key or "coverage_blocker",
                "blocking",
                f"{channel.get('label', key)} blocker_count={channel.get('blocker_count', 0)}",
                str(channel.get("hint") or "补齐该覆盖通道。"),
            )
        )

    if note_quality.get("export_freshness") == "stale":
        blockers.append(
            _blocker(
                "export_stale",
                "machine_action_available",
                "knowledge-note.md is older than timeline/review/coverage inputs.",
                "重新运行 export-knowledge-note。",
            )
        )
    elif note_quality.get("export_freshness") == "missing":
        blockers.append(
            _blocker(
                "export_missing",
                "machine_action_available",
                "knowledge-note.md or full-transcript.md is missing.",
                "运行 export-knowledge-note。",
            )
        )
    return blockers


def _acceptance_status(
    coverage: dict[str, Any],
    controlled: dict[str, Any],
    blockers: list[dict[str, Any]],
    note_quality: dict[str, Any],
    review_lifecycle: dict[str, Any],
    quality_gates: dict[str, Any],
) -> str:
    if any(item.get("severity") == "blocking" and item.get("key") not in {"provider_health_failed"} for item in blockers):
        return "incomplete"
    if _provider_blocked(controlled):
        return "provider_blocked"
    if any(item.get("severity") == "machine_action_available" for item in blockers):
        return "machine_action_available"
    if any(item.get("severity") == "human_review_required" for item in blockers):
        return "human_review_required"
    if any(item.get("severity") == "known_gap" for item in blockers):
        return "accepted_with_known_gaps"
    if _has_human_review_acceptance(coverage) and not blockers:
        return "accepted_with_known_gaps"
    if note_quality.get("export_freshness") in {"missing", "stale"}:
        return "machine_action_available"
    if str(coverage.get("status") or "") in {"ok", "weak"}:
        return "complete" if str(coverage.get("status") or "") == "ok" else "accepted_with_known_gaps"
    return "human_review_required"


def _acceptance_next_action(
    status: str,
    coverage: dict[str, Any],
    controlled: dict[str, Any],
    note_quality: dict[str, Any],
    review_lifecycle: dict[str, Any],
    provider_matrix: dict[str, Any],
    quality_gates: dict[str, Any],
) -> dict[str, Any]:
    transcript_quality = quality_gates.get("transcript") or {}
    summary_quality = quality_gates.get("smart_summary") or {}
    semantic_correction = (
        coverage.get("transcript_semantic_correction")
        if isinstance(coverage.get("transcript_semantic_correction"), dict)
        else {}
    )
    if transcript_quality.get("classification") in {"failed", "invalid"}:
        return {
            "key": "repair_transcript_quality",
            "label": "修复逐字稿完整性",
            "mcp_tool": "transcript_quality_gate",
            "hint": str(transcript_quality.get("detail") or ""),
            "command": "run transcript-quality-gate",
        }
    semantic_completion_gate = (
        semantic_correction.get("completion_gate")
        if isinstance(semantic_correction.get("completion_gate"), dict)
        else {}
    )
    semantic_required_action_kind = str(
        semantic_correction.get("required_action_kind")
        or semantic_completion_gate.get("required_action_kind")
        or "machine"
    )
    if (
        semantic_correction.get("required")
        and not bool(semantic_correction.get("ok"))
        and semantic_required_action_kind == "human"
    ):
        return {
            "key": str(
                semantic_correction.get("required_next_action_key")
                or "review_transcript_semantic_high_risk"
            ),
            "label": "复核高风险语义纠错候选",
            "mcp_tool": "transcript_semantic_correction_status",
            "hint": "只修改目标 cue；优先复核否定、条件、例外、实体、日期、金额和比例。",
            "command": "review transcript-semantic-correction-review.md",
        }
    if semantic_correction.get("required") and not bool(
        semantic_correction.get("ok")
    ):
        return {
            "key": str(
                semantic_correction.get("required_next_action_key")
                or semantic_correction.get("next_action_key")
                or "continue_transcript_semantic_correction"
            ),
            "label": "继续转写语义纠错闭环",
            "mcp_tool": "transcript_semantic_correction_status",
            "hint": "候选扫描、裁决、Hash 新鲜度、正式纠错稿和下游影响报告必须全部有明确回执。",
            "command": "run transcript-semantic-correction-status and follow required_next_action_key",
        }
    if int(semantic_correction.get("review_required_count") or 0):
        return {
            "key": "review_transcript_semantic_high_risk",
            "label": "复核高风险语义纠错候选",
            "mcp_tool": "transcript_semantic_correction_status",
            "hint": "只修改目标 cue；优先复核否定、条件、例外、实体、日期、金额和比例。",
            "command": "review transcript-semantic-correction-review.md",
        }
    if summary_quality.get("classification") in {"failed", "invalid"}:
        return {
            "key": "repair_smart_summary_quality",
            "label": "修复智能总结质量",
            "mcp_tool": "smart_summary_quality_check",
            "hint": str(summary_quality.get("detail") or ""),
            "command": "run smart-summary-quality-check",
        }
    if summary_quality.get("classification") == "human_review_required":
        return {
            "key": "provide_human_summary_key_points",
            "label": "补充人工关键点并复核智能总结",
            "mcp_tool": "smart_summary_quality_check",
            "hint": str(summary_quality.get("detail") or ""),
            "command": "provide human-key-points then run smart-summary-quality-check",
        }
    if status == "provider_blocked":
        matrix_action = _provider_matrix_next_action(provider_matrix)
        if review_lifecycle.get("state") == "human_review_ready":
            return {
                "key": "provider_repair_or_apply_review_notes",
                "label": "修复 Provider 或导入人工审核结果",
                "mcp_tool": matrix_action.get("mcp_tool") or "vision_provider_matrix",
                "fallback_mcp_tool": "apply_review_notes",
                "hint": f"{matrix_action.get('hint') or '当前 provider 不安全，先比较 provider matrix。'} review-notes.template.json 已准备好；也可以填好模板后导入人工审核结果。",
                "command": f"{matrix_action.get('command') or 'run vision-provider-matrix'}; or fill review-notes.template.json then run apply-review-notes",
                "provider_matrix": matrix_action.get("provider_matrix", {}),
            }
        return {
            "key": matrix_action.get("key") or "provider_repair",
            "label": matrix_action.get("label") or "修复或切换多模态 Provider",
            "mcp_tool": matrix_action.get("mcp_tool") or "vision_provider_matrix",
            "hint": matrix_action.get("hint") or "当前 provider 不安全，先比较 provider matrix，再执行带 --check-provider 的 preflight 和真实多模态写回。",
            "command": matrix_action.get("command") or "run vision-provider-matrix",
            "provider_matrix": matrix_action.get("provider_matrix", {}),
        }
    if note_quality.get("export_freshness") in {"missing", "stale"}:
        return {
            "key": "export_knowledge_note",
            "label": "重新导出人类可读 Markdown",
            "mcp_tool": "export_knowledge_note",
            "hint": "验收前需要刷新 knowledge-note.md 和 full-transcript.md。",
            "command": "run export-knowledge-note",
        }
    next_action = coverage.get("next_action") if isinstance(coverage.get("next_action"), dict) else {}
    if status == "machine_action_available" and next_action:
        return {
            "key": str(next_action.get("key") or "machine_action"),
            "label": str(next_action.get("label") or "继续机器处理"),
            "mcp_tool": str(next_action.get("mcp_tool") or ""),
            "mcp_args_path": str(next_action.get("mcp_args_path") or ""),
            "hint": str(next_action.get("hint") or ""),
            "command": str(next_action.get("command") or ""),
        }
    if status == "human_review_required":
        return {
            "key": "prepare_review_session",
            "label": "准备人工审核",
            "mcp_tool": "prepare_review_session",
            "hint": "机器路径无法自动补齐时，生成 review notes 模板。",
        }
    return {"key": "none", "label": "无需下一步", "mcp_tool": "", "hint": "当前验收状态无需机器动作。"}


def _summary(
    coverage: dict[str, Any],
    controlled: dict[str, Any],
    note_quality: dict[str, Any],
    review_lifecycle: dict[str, Any],
    review_closure: dict[str, Any] | None = None,
    quality_gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    channels = {str(item.get("key") or ""): item for item in coverage.get("channels") or [] if isinstance(item, dict)}
    closure = review_closure or {}
    gates = quality_gates or {}
    transcript_quality = gates.get("transcript") or {}
    summary_quality = gates.get("smart_summary") or {}
    semantic_correction = (
        coverage.get("transcript_semantic_correction")
        if isinstance(coverage.get("transcript_semantic_correction"), dict)
        else {}
    )
    review_queue = _review_queue_summary(coverage, channels)
    return {
        "timeline_items": int(coverage.get("timeline_items") or 0),
        "media_kind": (coverage.get("media_applicability") or {}).get("media_kind", "unknown"),
        "visual_applicability": (coverage.get("media_applicability") or {}).get("visual_status", "required"),
        "ocr_execution_summary": (coverage.get("ocr_execution") or {}).get("summary"),
        "coverage_status": coverage.get("status", "unknown"),
        "transcript_quality_status": transcript_quality.get("status", "not_available"),
        "transcript_quality_classification": transcript_quality.get("classification", "not_available"),
        "smart_summary_quality_status": summary_quality.get("status", "not_available"),
        "smart_summary_quality_classification": summary_quality.get("classification", "not_available"),
        "smart_summary_production_ready": summary_quality.get("production_ready"),
        "speech": _channel_status(channels, "speech"),
        "frames": _channel_status(channels, "visual_frames"),
        "visual_route": _channel_status(channels, "visual_route"),
        "document_visual": _channel_status(channels, "structured_visual"),
        "screen_text": _channel_status(channels, "screen_text"),
        "semantic_visual": _channel_status(channels, "semantic_frame_understanding"),
        "temporal_visual": _channel_status(channels, "temporal_visual_understanding"),
        "transcript_semantic_correction": _channel_status(
            channels, "transcript_semantic_correction"
        ),
        "transcript_semantic_correction_status": semantic_correction.get(
            "status", "not_evaluated"
        ),
        "transcript_semantic_candidate_discovery_status": semantic_correction.get(
            "candidate_discovery_status", "not_evaluated"
        ),
        "transcript_semantic_review_required": int(
            semantic_correction.get("review_required_count") or 0
        ),
        "transcript_semantic_unadjudicated": int(
            semantic_correction.get("unadjudicated_candidate_count") or 0
        ),
        "transcript_semantic_completion_gate": semantic_correction.get(
            "completion_gate", {}
        ),
        "transcript_semantic_required_next_action": semantic_correction.get(
            "required_next_action_key", "none"
        ),
        "semantic_missing": int(coverage.get("semantic_frame_without_analysis") or 0),
        "temporal_missing": int(coverage.get("temporal_sequence_without_analysis") or 0),
        "provider_health": controlled.get("provider_health_status", "not_checked"),
        "provider_safe_to_execute": controlled.get("provider_health_safe_to_execute"),
        "controlled_execution": controlled.get("status", "unknown"),
        "export_freshness": note_quality.get("export_freshness", "unknown"),
        "review_state": review_lifecycle.get("state", "not_prepared"),
        "review_template_prepared": bool(review_lifecycle.get("review_template_prepared")),
        "review_notes_imported": bool(review_lifecycle.get("review_notes_imported")),
        "review_targets_open": int(review_lifecycle.get("review_targets_open") or 0),
        "review_targets_listed": int(review_lifecycle.get("review_targets_listed") or 0),
        "review_closure_open": int(closure.get("open") or 0),
        "review_closure_closed": int(closure.get("closed") or 0),
        "review_closure_invalid": int(closure.get("invalid") or 0),
        "review_queue": review_queue,
    }


def _review_queue_summary(
    coverage: dict[str, Any], channels: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    samples = coverage.get("samples") if isinstance(coverage.get("samples"), dict) else {}
    grouped_samples = {
        "transcript": _int_list(samples.get("missing_transcript")),
        "ocr": _int_list(samples.get("ocr_gap")),
        "semantic_visual": _int_list(samples.get("missing_visual_understanding")),
        "temporal_visual": _int_list(samples.get("temporal_sequence_without_analysis")),
    }
    deduplicated = sorted(
        {
            index
            for values in grouped_samples.values()
            for index in values
        }
    )
    return {
        "transcript_review_targets": _channel_gap_count(channels, "speech"),
        "ocr_review_targets": _channel_gap_count(channels, "screen_text"),
        "semantic_visual_missing": int(
            coverage.get("semantic_frame_without_analysis") or 0
        ),
        "temporal_visual_missing": int(
            coverage.get("temporal_sequence_without_analysis") or 0
        ),
        "sample_indexes_by_group": grouped_samples,
        "deduplicated_sample_indexes": deduplicated,
        "deduplicated_sample_count": len(deduplicated),
        "note": "sample indexes are bounded by the coverage report; counts remain full-channel counts",
    }


def _channel_gap_count(
    channels: dict[str, dict[str, Any]], key: str
) -> int:
    channel = channels.get(key) if isinstance(channels.get(key), dict) else {}
    expected = int(channel.get("expected_count") or 0)
    covered = int(channel.get("covered_count") or 0)
    blockers = int(channel.get("blocker_count") or 0)
    return max(blockers, expected - covered, 0)


def _int_list(value: Any) -> list[int]:
    result: list[int] = []
    for item in value if isinstance(value, list) else []:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in result:
            result.append(number)
    return result


def _quality_gate_evidence(root: Path) -> dict[str, Any]:
    """Read existing quality artifacts without creating another state machine.

    Intent: prevent final acceptance from claiming complete when the existing
    transcript or smart-summary quality gate disagrees.
    Decision: aggregate only persisted gate artifacts; legacy bundles with
    neither file preserve the historical acceptance behavior.
    Reason: the two production bundles were reported complete despite warning,
    failed, or human-benchmark-blocked quality evidence.
    Evidence: transcript-quality-gate.json and exports/smart-summary-quality.json.
    Effective scope: final acceptance reports when either artifact exists.
    """

    transcript_path = root / "transcript-quality-gate.json"
    summary_path = root / "exports" / "smart-summary-quality.json"
    transcript = _read_quality_artifact(transcript_path, kind="transcript")
    summary = _read_quality_artifact(summary_path, kind="smart_summary")
    return {
        "schema": "video_knowledge_pipeline.acceptance_quality_gates.v1",
        "present": transcript["present"] or summary["present"],
        "transcript": transcript,
        "smart_summary": summary,
    }


def _read_quality_artifact(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "path": str(path),
            "status": "not_available",
            "classification": "not_available",
            "detail": "quality artifact does not exist; legacy behavior preserved",
        }
    try:
        value = read_json(path)
    except Exception as exc:
        return {
            "present": True,
            "path": str(path),
            "status": "invalid",
            "classification": "invalid",
            "detail": f"quality artifact is unreadable: {exc}",
        }
    if not isinstance(value, dict):
        return {
            "present": True,
            "path": str(path),
            "status": "invalid",
            "classification": "invalid",
            "detail": "quality artifact is not a JSON object",
        }
    status = str(value.get("status") or "").strip().lower()
    if kind == "transcript":
        if status == "failed" or value.get("ok") is False or int(value.get("fail_count") or 0) > 0:
            classification = "failed"
        elif status == "warning":
            classification = "warning"
        elif status == "passed" and value.get("ok") is not False:
            classification = "passed"
        else:
            classification = "invalid"
        detail = (
            f"status={status or 'missing'}; fail_count={int(value.get('fail_count') or 0)}; "
            f"warning_count={int(value.get('warning_count') or 0)}"
        )
        return {
            "present": True,
            "path": str(path),
            "status": status or "invalid",
            "classification": classification,
            "ok": value.get("ok"),
            "fail_count": int(value.get("fail_count") or 0),
            "warning_count": int(value.get("warning_count") or 0),
            "detail": detail,
        }
    automated = value.get("automated_checks_passed")
    production_ready = value.get("production_ready")
    if status == "failed" or automated is False:
        classification = "failed"
    elif status == "blocked_missing_human_key_points" or (
        automated is True and production_ready is False
    ):
        classification = "human_review_required"
    elif status == "passed" and production_ready is True:
        classification = "passed"
    else:
        classification = "invalid"
    return {
        "present": True,
        "path": str(path),
        "status": status or "invalid",
        "classification": classification,
        "passed": value.get("passed"),
        "automated_checks_passed": automated,
        "production_ready": production_ready,
        "detail": (
            f"status={status or 'missing'}; automated_checks_passed={automated}; "
            f"production_ready={production_ready}"
        ),
    }


def _note_quality(root: Path) -> dict[str, Any]:
    export_dir = root / "exports"
    note_path = export_dir / "knowledge-note.md"
    transcript_path = export_dir / "full-transcript.md"
    full_body_path = export_dir / "full-body.md"
    audit_path = export_dir / "extraction-audit.md"
    # Watch source inputs only. Derived reports such as knowledge-coverage.json
    # are refreshed by acceptance-check itself and would otherwise make every
    # freshly exported note look stale.
    watched = [
        root / "timeline.json",
        root / "review-notes.json",
        root / "review-notes.template.json",
        root / "review-session.json",
        root / "knowledge-coverage.json",
        root / "vision-execution-preflight.json",
    ]
    derived = {root / "knowledge-coverage.json"}
    watched = [path for path in watched if path not in derived]
    snapshot_path = export_dir / "knowledge-export-dependency-snapshot.json"
    snapshot_validation: dict[str, Any] = {
        "status": "missing",
        "passed": False,
        "issues": [{"key": "dependency_snapshot_missing"}],
    }
    if snapshot_path.is_file():
        try:
            snapshot_value = read_json(snapshot_path)
        except Exception as exc:
            snapshot_validation = {
                "status": "invalid",
                "passed": False,
                "issues": [{"key": "dependency_snapshot_unreadable", "detail": str(exc)}],
            }
        else:
            snapshot_validation = validate_dependency_snapshot(
                root,
                snapshot_value if isinstance(snapshot_value, dict) else {},
            )
    if not note_path.exists() or not transcript_path.exists() or not full_body_path.exists() or not audit_path.exists():
        freshness = "missing"
    elif snapshot_validation.get("status") in {"fresh", "stale", "missing", "invalid"}:
        freshness = str(snapshot_validation["status"])
    else:
        note_mtime = min(
            note_path.stat().st_mtime,
            transcript_path.stat().st_mtime,
            full_body_path.stat().st_mtime,
            audit_path.stat().st_mtime,
        )
        newest_input = max((path.stat().st_mtime for path in watched if path.exists()), default=0)
        freshness = "stale" if newest_input > note_mtime else "fresh"
    return {
        "knowledge_note_path": str(note_path),
        "full_transcript_path": str(transcript_path),
        "full_body_path": str(full_body_path),
        "extraction_audit_path": str(audit_path),
        "export_freshness": freshness,
        "dependency_snapshot_path": str(snapshot_path),
        "dependency_snapshot_validation": snapshot_validation,
    }


def _write_acceptance_outputs(
    root: Path,
    manifest_path: Path,
    report: dict[str, Any],
    report_path: Path,
    markdown_path: Path,
    args_path: Path,
) -> None:
    with bundle_write_lock(root, operation="acceptance_check"):
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if isinstance(manifest, dict):
            manifest["acceptance_check"] = "acceptance-check.md"
            manifest["acceptance_check_json"] = "acceptance-check.json"
            manifest["mcp_acceptance_check_args"] = "mcp-acceptance-check.args.json"
            write_json(manifest_path, manifest)
        write_json(report_path, report)
        markdown_path.write_text(render_acceptance_check_markdown(report), encoding="utf-8")
        write_json(args_path, {"bundle_dir": str(root), "refresh": True, "write": True})


def _report(
    *,
    root: Path,
    status: str,
    summary: dict[str, Any],
    blockers: list[dict[str, Any]],
    next_action: dict[str, Any],
    report_path: Path,
    markdown_path: Path,
    args_path: Path,
    coverage: dict[str, Any] | None = None,
    controlled_execution: dict[str, Any] | None = None,
    note_quality: dict[str, Any] | None = None,
    review_lifecycle: dict[str, Any] | None = None,
    review_closure: dict[str, Any] | None = None,
    provider_matrix: dict[str, Any] | None = None,
    quality_gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "checked_at": now_iso(),
        "bundle_dir": str(root),
        "status": status,
        "summary": summary,
        "blockers": blockers,
        "next_action": next_action,
        "coverage": coverage or {},
        "controlled_execution": controlled_execution or {},
        "note_quality": note_quality or {},
        "review_lifecycle": review_lifecycle or {},
        "review_closure": review_closure or {},
        "provider_matrix": provider_matrix or {},
        "quality_gates": quality_gates or {},
        "report_path": str(report_path),
        "report_markdown_path": str(markdown_path),
        "mcp_args_path": str(args_path),
    }


def _review_lifecycle(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    template_path = root / str(manifest.get("review_notes_template") or "review-notes.template.json")
    session_path = root / str(manifest.get("review_session_json") or "review-session.json")
    review_notes_path = root / str(manifest.get("review_notes") or "review-notes.json")

    template_rows = _read_review_rows(template_path)
    imported_rows = _read_review_rows(review_notes_path)
    session_targets = _read_session_targets(session_path)
    last_import = manifest.get("review_notes_last_import") if isinstance(manifest.get("review_notes_last_import"), dict) else {}
    imported_count = int(last_import.get("updated") or 0)
    if imported_count <= 0:
        imported_count = len([row for row in imported_rows if _review_row_has_resolution(row)])

    template_prepared = template_path.exists() and bool(template_rows)
    review_imported = imported_count > 0
    if review_imported:
        state = "human_review_imported"
    elif template_prepared:
        state = "human_review_ready"
    else:
        state = "not_prepared"

    return {
        "schema": "lecture_review_lifecycle.v1",
        "state": state,
        "review_template_prepared": template_prepared,
        "review_notes_imported": review_imported,
        "review_targets_open": int(session_targets.get("total_open") or len(template_rows)),
        "review_targets_listed": int(session_targets.get("listed_count") or len(template_rows)),
        "review_template_rows": len(template_rows),
        "review_notes_rows": len(imported_rows),
        "review_notes_imported_count": imported_count,
        "review_template_path": str(template_path),
        "review_session_path": str(session_path),
        "review_notes_path": str(review_notes_path),
        "review_resolution_mode": "provider_or_human_review" if template_prepared or review_imported else "provider_or_prepare_review",
        "template_suggested_statuses": _status_counts(template_rows, key="suggested_status"),
    }


def _review_closure(root: Path) -> dict[str, Any]:
    try:
        result = review_closure_status(root, write=False)
    except Exception:
        return {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    return {
        "open": int(summary.get("open") or 0),
        "closed": int(summary.get("closed") or 0),
        "imported": int(summary.get("imported") or 0),
        "invalid": int(summary.get("invalid") or 0),
        "report_path": result.get("report_path", ""),
        "report_markdown_path": result.get("report_markdown_path", ""),
        "next_batch": result.get("next_batch") if isinstance(result.get("next_batch"), dict) else {},
    }


def _read_review_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("reviews", "items", "notes"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]
    return []


def _read_session_targets(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {}
    targets = payload.get("review_targets")
    return targets if isinstance(targets, dict) else {}


def _review_row_has_resolution(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or row.get("suggested_status") or "").strip().lower()
    if status in {"", "needs_human_review", "needs_fix", "todo"}:
        return False
    if row.get("comment") or row.get("corrected_transcript") or row.get("corrected_visual_text"):
        return True
    for key in ("corrected_visual_understanding", "corrected_temporal_visual_understanding"):
        value = row.get(key)
        if isinstance(value, dict) and value:
            return True
    return "status" in row


def _status_counts(rows: list[dict[str, Any]], *, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get(key) or "unknown").strip() or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _provider_matrix(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    matrix_path = root / str(manifest.get("vision_provider_matrix_json") or "vision-provider-matrix.json")
    if not matrix_path.exists():
        return {
            "schema": "lecture_provider_matrix_summary.v1",
            "status": "missing",
            "recommended_provider": "",
            "providers": [],
            "report_path": str(matrix_path),
            "mcp_args_path": str(root / str(manifest.get("mcp_vision_provider_matrix_args") or "mcp-vision-provider-matrix.args.json")),
        }
    payload = read_json(matrix_path)
    if not isinstance(payload, dict):
        return {"schema": "lecture_provider_matrix_summary.v1", "status": "invalid", "recommended_provider": "", "providers": [], "report_path": str(matrix_path)}
    rows: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider") if isinstance(item.get("provider"), dict) else {}
        rows.append(
            {
                "provider": str(provider.get("provider") or ""),
                "model": str(provider.get("model") or ""),
                "safe_to_execute": bool(item.get("safe_to_execute")),
                "status": str(item.get("status") or ""),
                "error_class": str(item.get("error_class") or ""),
                "recovery_suggestion": str(item.get("recovery_suggestion") or ""),
            }
        )
    return {
        "schema": "lecture_provider_matrix_summary.v1",
        "status": str(payload.get("status") or "unknown"),
        "recommended_provider": str(payload.get("recommended_provider") or ""),
        "providers": rows,
        "report_path": str(matrix_path),
        "report_markdown_path": str(root / str(manifest.get("vision_provider_matrix") or "vision-provider-matrix.md")),
        "mcp_args_path": str(root / str(manifest.get("mcp_vision_provider_matrix_args") or "mcp-vision-provider-matrix.args.json")),
    }


def _provider_matrix_next_action(provider_matrix: dict[str, Any]) -> dict[str, Any]:
    recommended = str(provider_matrix.get("recommended_provider") or "").strip()
    providers = provider_matrix.get("providers") if isinstance(provider_matrix.get("providers"), list) else []
    matrix_summary = {
        "status": provider_matrix.get("status", ""),
        "recommended_provider": recommended,
        "providers": providers,
        "report_path": provider_matrix.get("report_path", ""),
    }
    if recommended:
        return {
            "key": "provider_preflight_with_recommended",
            "label": f"使用可用 Provider 运行 preflight：{recommended}",
            "mcp_tool": "vision_execution_preflight",
            "hint": f"Provider matrix 已找到可用 provider `{recommended}`；用该 provider 重新运行 vision-execution-preflight --check-provider，再执行确认批次。",
            "command": f"run vision-execution-preflight with provider={recommended}",
            "provider_matrix": matrix_summary,
        }
    if providers:
        statuses = ", ".join(f"{row.get('provider')}={row.get('status')}" for row in providers if isinstance(row, dict))
        return {
            "key": "provider_matrix_repair",
            "label": "修复或补齐 Provider Matrix",
            "mcp_tool": "vision_provider_matrix",
            "hint": f"当前 provider matrix 没有可用 provider：{statuses}。补 key、修网络/proxy/TLS，或改用人工审核导入。",
            "command": "run vision-provider-matrix",
            "provider_matrix": matrix_summary,
        }
    return {
        "key": "provider_matrix_missing",
        "label": "运行 Provider Matrix",
        "mcp_tool": "vision_provider_matrix",
        "hint": "当前缺少 provider matrix；先比较 Agnes/Gemini/OpenAI-compatible，而不是重复单个 provider smoke。",
        "command": "run vision-provider-matrix",
        "provider_matrix": matrix_summary,
    }


def _review_resolution_hint(review_lifecycle: dict[str, Any]) -> str:
    if review_lifecycle.get("state") == "human_review_ready":
        return f" 已有可填写模板：{review_lifecycle.get('review_template_path', '')}"
    if review_lifecycle.get("state") == "human_review_imported":
        return " 已检测到人工审核导入记录；请刷新 coverage/export。"
    return " 如 provider 继续不可用，先运行 prepare-review-session 生成人工审核模板。"


def _provider_blocked(controlled: dict[str, Any]) -> bool:
    blockers = controlled.get("blockers") if isinstance(controlled.get("blockers"), list) else []
    return "provider_health_failed" in blockers or controlled.get("provider_health_safe_to_execute") is False


def _has_human_review_acceptance(coverage: dict[str, Any]) -> bool:
    return bool(coverage.get("weak_channels")) and not coverage.get("blockers")


def _channel_status(channels: dict[str, Any], key: str) -> str:
    channel = channels.get(key)
    return str(channel.get("status") or "unknown") if isinstance(channel, dict) else "unknown"


def _blocker(key: str, severity: str, detail: str, next_action: str) -> dict[str, Any]:
    return {"key": key, "severity": severity, "detail": detail, "next_action": next_action}
