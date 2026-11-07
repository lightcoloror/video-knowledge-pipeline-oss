from __future__ import annotations

from pathlib import Path
from typing import Any

from .companion_courseware_text import load_companion_courseware_text
from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .ocr_inventory import build_ocr_inventory
from .media_applicability import VISUAL_CHANNELS, audio_only_ocr_status, project_timeline_for_media, resolve_media_applicability
from .panorama_evidence import panorama_bundle_coverage
from .source_artifacts import summarize_manifest_source_artifacts
from .storage import read_json, write_json
from .timeline_alignment_audit import (
    resolve_canonical_transcript_path,
    timeline_alignment_audit_is_current,
)
from .visual_evidence import (
    continuous_tag_evidence_state as _continuous_tag_evidence_state,
    evidence_level_counts as _shared_evidence_level_counts,
    has_temporal_evidence as _shared_has_temporal_evidence,
    has_visual_evidence as _shared_has_visual_evidence,
    human_review_accepted as _shared_human_review_accepted,
)

COVERAGE_SCHEMA = "lecture_knowledge_coverage.v1"


def audit_knowledge_coverage(
    bundle_dir: str | Path, *, write: bool = True
) -> dict[str, Any]:
    """Audit whether a lecture WebUI bundle covers the major knowledge channels.

    This is intentionally a glue-layer report. It does not run ASR, OCR, or visual
    models; it reads the bundle artifacts produced by reused tools and points to
    the next existing tool to call when a channel is weak.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline_data, list):
        raise ValueError("timeline.json must be a JSON array")
    timeline = [item for item in timeline_data if isinstance(item, dict)]
    report = build_knowledge_coverage(manifest, timeline, bundle_dir=root)
    json_path = root / "knowledge-coverage.json"
    markdown_path = root / "knowledge-coverage.md"
    args_path = root / "mcp-audit-knowledge-coverage.args.json"
    report["next_action"] = _executable_next_action(
        report.get("next_action"), root=root, fallback_args_path=args_path
    )
    result = {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "coverage_path": str(json_path),
        "coverage_markdown_path": str(markdown_path),
        "mcp_args_path": str(args_path),
        "coverage": report,
    }
    if write:
        manifest["media_applicability"] = report["media_applicability"]
        write_json(root / "media-applicability.json", report["media_applicability"])
        manifest["knowledge_coverage"] = report
        manifest["knowledge_coverage_json"] = "knowledge-coverage.json"
        manifest["knowledge_coverage_markdown"] = "knowledge-coverage.md"
        manifest["mcp_knowledge_coverage_args"] = (
            "mcp-audit-knowledge-coverage.args.json"
        )
        write_json(manifest_path, manifest)
        write_json(json_path, report)
        markdown_path.write_text(
            render_knowledge_coverage_markdown(report), encoding="utf-8"
        )
        write_json(args_path, {"bundle_dir": str(root), "write": True})
    return result


def build_knowledge_coverage(
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    *,
    bundle_dir: str | Path | None = None,
) -> dict[str, Any]:
    applicability = resolve_media_applicability(bundle_dir, manifest, timeline)
    timeline = project_timeline_for_media(timeline, applicability)
    total = len(timeline)
    triage_policy = _triage_requirement_policy(bundle_dir, timeline_count=total)
    canonical_transcript_indexes = _canonical_transcript_coverage_indexes(
        bundle_dir,
        timeline_count=total,
    )
    transcript_items = [
        item
        for position, item in enumerate(timeline, start=1)
        if _text(item.get("transcript"))
        or _text(item.get("original_transcript"))
        or _timeline_index(item, position) in canonical_transcript_indexes
    ]
    visual_text_items = [item for item in timeline if _has_meaningful_visual_text(item)]
    visual_asset_items = [item for item in timeline if _asset_paths(item)]
    structured_expected_items = [
        item
        for item in timeline
        if _structured_visual(item) or _has_structured_type(item)
    ]
    structured_items = [
        item
        for item in structured_expected_items
        if _structured_visual(item) or _human_review_structured_fallback(item)
    ]
    raw_structured_without_structure = [
        item
        for item in structured_expected_items
        if not (_structured_visual(item) or _human_review_structured_fallback(item))
    ]
    asset_gap_items = [
        item for item in timeline if _has_asset_gap(item, bundle_dir=bundle_dir)
    ]
    frame_gap_items = [
        item
        for item in timeline
        if set(_quality_issues(item))
        & {
            "missing_frame",
            "structured_visual_without_frame",
            "keep_image_without_frame",
        }
    ]
    raw_ocr_gap_items = [item for item in timeline if _needs_ocr(item)]
    screen_text_low_confidence_items = [
        item
        for item in timeline
        if "screen_text_low_confidence" in set(_quality_issues(item))
    ]
    ocr_text_empty_items = [
        item for item in timeline if "ocr_text_empty" in set(_quality_issues(item))
    ]
    visual_route_items = [item for item in timeline if _text(item.get("visual_route"))]
    visual_understanding_items = [
        item for item in timeline if _has_visual_understanding(item)
    ]
    temporal_understanding_items = [
        item for item in timeline if _has_temporal_understanding(item)
    ]
    temporal_tag_weak_items = [
        item for item in timeline if _continuous_tag_evidence_state(item)["consumable_weak"]
    ]
    evidence_levels = _shared_evidence_level_counts(timeline)
    panorama_coverage = panorama_bundle_coverage(manifest, timeline)
    semantic_frame_items = [
        item
        for item in timeline
        if str(item.get("visual_route") or "") in {"semantic_frame", "mixed"}
    ]
    temporal_sequence_items = [
        item
        for item in timeline
        if str(item.get("visual_route") or "") in {"temporal_sequence", "mixed"}
    ]
    if triage_policy:
        semantic_indexes = set(triage_policy["semantic_indexes"])
        temporal_indexes = set(triage_policy["temporal_indexes"])
        visual_structure_indexes = set(triage_policy["visual_structure_first_indexes"])
        semantic_without_analysis = [
            item
            for position, item in enumerate(timeline, start=1)
            if _timeline_index(item, position) in semantic_indexes
            and not _has_visual_understanding(item)
        ]
        temporal_without_analysis = [
            item
            for position, item in enumerate(timeline, start=1)
            if _timeline_index(item, position) in temporal_indexes
            and not (_has_temporal_understanding(item) or _continuous_tag_evidence_state(item)["consumable_weak"])
        ]
        ocr_gap_items = [
            item
            for position, item in enumerate(timeline, start=1)
            if _timeline_index(item, position) in visual_structure_indexes
            and _needs_ocr(item)
        ]
        structured_without_structure = [
            item
            for position, item in enumerate(timeline, start=1)
            if _timeline_index(item, position) in visual_structure_indexes
            and item in raw_structured_without_structure
        ]
        screen_expected_count = len(_unique_items([*visual_text_items, *ocr_gap_items]))
        structured_expected_count = len(structured_items) + len(
            structured_without_structure
        )
        semantic_expected_count = len(visual_understanding_items) + len(
            semantic_without_analysis
        )
        semantic_covered_count = len(visual_understanding_items)
        temporal_expected_count = len(temporal_understanding_items) + len(
            temporal_without_analysis
        )
        temporal_covered_count = len(_unique_items([*temporal_understanding_items, *temporal_tag_weak_items]))
    else:
        semantic_without_analysis = [
            item for item in semantic_frame_items if not _has_visual_understanding(item)
        ]
        temporal_without_analysis = [
            item
            for item in temporal_sequence_items
            if not (_has_temporal_understanding(item) or _continuous_tag_evidence_state(item)["consumable_weak"])
        ]
        ocr_gap_items = raw_ocr_gap_items
        structured_without_structure = raw_structured_without_structure
        screen_expected_count = total
        structured_expected_count = len(structured_expected_items)
        semantic_expected_count = len(semantic_frame_items)
        semantic_covered_count = len(
            [item for item in semantic_frame_items if _has_visual_understanding(item)]
        )
        temporal_expected_count = len(temporal_sequence_items)
        temporal_covered_count = len(
            [
                item
                for item in temporal_sequence_items
                if _has_temporal_understanding(item) or _continuous_tag_evidence_state(item)["consumable_weak"]
            ]
        )
    ocr_execution = build_ocr_inventory(Path(bundle_dir), timeline, applicability=applicability) if bundle_dir else None
    if ocr_execution:
        pending = set(ocr_execution["pending_indexes"] + ocr_execution["empty_indexes"])
        pending.update(row["index"] for row in ocr_execution["items"] if row.get("quality_suspicious"))
        ocr_gap_items = _unique_items([*ocr_gap_items, *[
            item for position, item in enumerate(timeline, 1)
            if _timeline_index(item, position) in pending
        ]])
        screen_expected_count = max(screen_expected_count, ocr_execution["summary"]["required"])
    coverage = (
        manifest.get("coverage") if isinstance(manifest.get("coverage"), dict) else {}
    )
    source_summary = summarize_manifest_source_artifacts(manifest)
    semantic_correction_channel, semantic_correction_summary = (
        _transcript_semantic_correction_channel(
            bundle_dir,
            manifest,
            transcript_item_count=len(transcript_items),
        )
    )
    companion_courseware = load_companion_courseware_text(bundle_dir, manifest) if bundle_dir else None
    speech_tool, speech_args, speech_hint = _speech_recovery_action(
        manifest,
        bundle_dir=bundle_dir,
    )
    time_gap_count = int(_number(coverage.get("time_gap_count")))
    panorama_channel = _channel(
        "panorama_spherical_evidence",
        "全景球面几何/语义/时序证据",
        int(panorama_coverage.get("expected_segments") or 0),
        min(
            int(panorama_coverage.get("geometric_complete_segments") or 0),
            int(panorama_coverage.get("semantic_covered_segments") or 0),
        ),
        "panorama_status",
        str(manifest.get("mcp_panorama_status_args") or "mcp-panorama-status.args.json"),
        "全景几何覆盖、语义观察和方向时序分别统计；局部视口不得冒充全局事实，缺口进入复核而不阻断 ASR 等其他产出。",
        samples=list(panorama_coverage.get("missing_segment_indexes") or []),
    )
    if not panorama_coverage.get("required"):
        panorama_channel["status"] = "not_required"
    elif panorama_coverage.get("status") != "complete":
        panorama_channel["status"] = "weak"
        panorama_channel["review_gap_count"] = len(panorama_coverage.get("review_gaps") or [])
    channels = [
        _channel(
            "speech",
            "语言转写",
            total,
            len(transcript_items),
            speech_tool,
            speech_args,
            speech_hint,
            samples=_sample_indexes(
                [item for item in timeline if item not in transcript_items]
            ),
        ),
        _channel(
            "screen_text",
            "屏幕文字/图文截图解析",
            screen_expected_count,
            len(visual_text_items),
            "run_visual_structure_plan",
            str(manifest.get("mcp_visual_structure_args") or ""),
            "屏幕文字缺口会漏掉课件标题、公式旁注、代码、表格标签和软件界面小字；主通道复用 ebook_markdown_pipeline，直接 OCR/crop OCR 只作备用，低置信度必须显式复核。",
            blockers=len(ocr_gap_items),
            samples=_sample_indexes(ocr_gap_items),
        ),
        _channel(
            "visual_frames",
            "关键帧/板书截图",
            total,
            len(visual_asset_items),
            "run_frame_recapture_plan",
            str(manifest.get("mcp_frame_recapture_args") or ""),
            "需要保留图片的片段必须有可追溯关键帧；否则无法核对图表、板书和空间关系。",
            blockers=len(asset_gap_items) + len(frame_gap_items),
            samples=_sample_indexes(asset_gap_items + frame_gap_items),
        ),
        _channel(
            "structured_visual",
            "图表/公式/代码结构化",
            structured_expected_count,
            len(structured_items),
            "run_visual_structure_plan",
            str(manifest.get("mcp_visual_structure_args") or ""),
            "公式、表格、代码不应只被压成自然语言；能降维成文字就结构化，必须保留图片就保留图片。",
            blockers=len(structured_without_structure),
            samples=_sample_indexes(structured_without_structure),
        ),
        _channel(
            "visual_route",
            "画面类型路由",
            total,
            len(visual_route_items),
            "run_video_frame_router",
            str(manifest.get("mcp_video_frame_router_args") or ""),
            "每个时间片应先路由到图文、语义画面或连续变化分支，否则后续工具无法选择正确处理方式。",
            samples=_sample_indexes(
                [item for item in timeline if item not in visual_route_items]
            ),
        ),
        _channel(
            "semantic_frame_understanding",
            "多模态单帧理解",
            semantic_expected_count,
            semantic_covered_count,
            "run_multimodal_frame_analysis",
            str(manifest.get("mcp_multimodal_frame_analysis_args") or ""),
            "实物、界面状态、人物动作、空间关系和讲师指向不能只靠 OCR；需要多模态单帧理解并保留证据帧。",
            blockers=len(semantic_without_analysis),
            samples=_sample_indexes(semantic_without_analysis),
        ),
        _channel(
            "temporal_visual_understanding",
            "连续片段理解",
            temporal_expected_count,
            temporal_covered_count,
            "run_temporal_visual_analysis",
            str(manifest.get("mcp_temporal_visual_analysis_args") or ""),
            "静态且标签充分的片段可用 candidate_weak 关闭机器缺口；软件操作、流程演示和动态变化仍必须升级 5-12 帧顺序理解。",
            blockers=len(temporal_without_analysis),
            samples=_sample_indexes(temporal_without_analysis),
        ),
        panorama_channel,
        semantic_correction_channel,
        _source_channel(source_summary, manifest),
        _time_gap_channel(coverage, manifest, time_gap_count),
    ]
    if applicability["visual_status"] == "not_applicable":
        for channel in channels:
            if channel.get("key") in VISUAL_CHANNELS:
                channel.update(status="not_applicable", expected_count=0, covered_count=0,
                               coverage_percent=None, blocker_count=0, sample_indexes=[],
                               reason=applicability["reason"], mcp_tool="", mcp_args_path="",
                               hint="纯音频无附加视觉材料，此通道不适用。")
    if companion_courseware:
        for channel in channels:
            if channel.get("key") in {"screen_text", "structured_visual"}:
                channel.update({
                    "status": "covered_by_external_courseware",
                    "expected_count": 0,
                    "covered_count": 1,
                    "coverage_percent": 100.0,
                    "blocker_count": 0,
                    "coverage_source": "companion_courseware_text",
                    "coverage_scope": "external_courseware_not_video_frame",
                })
    temporal_channel = next((channel for channel in channels if channel.get("key") == "temporal_visual_understanding"), None)
    if temporal_channel and temporal_tag_weak_items and temporal_channel.get("status") == "ok":
        temporal_channel["status"] = "candidate_weak"
        temporal_channel["weak_evidence_count"] = len(temporal_tag_weak_items)
        temporal_channel["evidence_level"] = "candidate_weak"
    blockers = [channel for channel in channels if channel["status"] == "blocked"]
    weak = [channel for channel in channels if channel["status"] == "weak"]
    next_action = _next_action(blockers or weak)
    from .summary_stage_coverage import current_summary_stage_coverage
    summary_stage = current_summary_stage_coverage(bundle_dir) if bundle_dir else {"status": "not_assessed", "passed": None}
    from .citation_digest import audit_citation_digest
    citation_digest = audit_citation_digest(bundle_dir) if bundle_dir else {"status": "not_assessed", "passed": None}
    return {
        "schema": COVERAGE_SCHEMA,
        "media_applicability": applicability,
        "ocr_execution": audio_only_ocr_status(applicability) or ocr_execution,
        "checked_at": now_iso(),
        "status": "blocked" if blockers else ("weak" if weak else "ok"),
        "visual_requirement_policy": triage_policy or {"mode": "route_full_legacy", "fresh": False},
        "companion_courseware": {"status": "covered_by_external_courseware", "path": str(companion_courseware.get("bundle_copy_path") or ""), "source_sha256": str(companion_courseware.get("source_sha256") or "")} if companion_courseware else {"status": "not_present"},
        "timeline_items": total,
        "canonical_transcript_aligned_items": len(canonical_transcript_indexes),
        "items_with_visual_route": len(visual_route_items),
        "items_with_visual_understanding": len(visual_understanding_items),
        "items_with_temporal_understanding": len(temporal_understanding_items),
        "items_with_continuous_tag_weak_evidence": len(temporal_tag_weak_items),
        "visual_evidence_levels": evidence_levels["visual"],
        "temporal_evidence_levels": evidence_levels["temporal"],
        "continuous_tag_evidence_levels": evidence_levels["continuous_tag"],

        "panorama_coverage": panorama_coverage,
        "transcript_semantic_correction": semantic_correction_summary,
        # Distinct from extraction channels: output gaps must not rerun OCR/ASR.
        # Existing summary quality/acceptance gates own final draft acceptance.
        "summary_stage_coverage": summary_stage,
        "citation_digest_integrity": citation_digest,
        "semantic_frame_without_analysis": len(semantic_without_analysis),
        "temporal_sequence_without_analysis": len(temporal_without_analysis),
        "missing_visual_understanding": len(semantic_without_analysis)
        + len(temporal_without_analysis),
        "screen_text_low_confidence": len(screen_text_low_confidence_items),
        "ocr_text_empty": len(ocr_text_empty_items),
        "channels": channels,
        "blockers": blockers,
        "weak_channels": weak,
        "next_action": next_action,
        "samples": {
            "missing_transcript": _sample_indexes(
                [item for item in timeline if item not in transcript_items]
            ),
            "ocr_gap": _sample_indexes(ocr_gap_items),
            "screen_text_low_confidence": _sample_indexes(
                screen_text_low_confidence_items
            ),
            "ocr_text_empty": _sample_indexes(ocr_text_empty_items),
            "frame_gap": _sample_indexes(asset_gap_items + frame_gap_items),
            "structured_gap": _sample_indexes(structured_without_structure),
            "missing_visual_understanding": _sample_indexes(semantic_without_analysis),
            "temporal_sequence_without_analysis": _sample_indexes(
                temporal_without_analysis
            ),
        },
    }


def _transcript_semantic_correction_channel(
    bundle_dir: str | Path | None,
    manifest: dict[str, Any],
    *,
    transcript_item_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project semantic-correction closure into coverage without mutating a Bundle."""

    if bundle_dir is None:
        summary = {"status": "not_evaluated", "required": False}
        return (
            _channel(
                "transcript_semantic_correction",
                "转写语义准确性",
                0,
                0,
                "transcript_semantic_correction_status",
                "",
                "传入 Bundle 路径后才能核对语义候选扫描、裁决、闭合与新鲜度。",
            ),
            summary,
        )
    root = Path(bundle_dir).expanduser().resolve()
    canonical = resolve_canonical_transcript_path(root, manifest)
    required = bool(
        transcript_item_count
        and canonical is not None
        and canonical.is_file()
    )
    if not required:
        summary = {
            "status": "not_required_without_bound_transcript",
            "required": False,
            "canonical_transcript": str(canonical or ""),
        }
        return (
            _channel(
                "transcript_semantic_correction",
                "转写语义准确性",
                0,
                0,
                "transcript_semantic_correction_status",
                str(
                    manifest.get("mcp_transcript_semantic_correction_status_args")
                    or "mcp-transcript-semantic-correction-status.args.json"
                ),
                "尚无绑定的规范化转写；先完成转写绑定，再核对语义准确性。",
            ),
            summary,
        )

    try:
        from .transcript_semantic_correction import (
            transcript_semantic_correction_status,
        )

        status = transcript_semantic_correction_status(root, write=False)
    except (FileNotFoundError, OSError, ValueError) as exc:
        status = {
            "status": "status_unavailable",
            "ok": False,
            "candidate_discovery_status": "unknown",
            "candidate_discovery_complete": False,
            "review_required_count": 0,
            "unadjudicated_candidate_count": 0,
            "error": str(exc),
        }

    workflow_status = str(status.get("status") or "not_started")
    discovery_status = str(
        status.get("candidate_discovery_status") or "not_planned"
    )
    review_required_count = int(status.get("review_required_count") or 0)
    unadjudicated_candidate_count = int(
        status.get("unadjudicated_candidate_count") or 0
    )
    completion_gate = (
        status.get("completion_gate")
        if isinstance(status.get("completion_gate"), dict)
        else {}
    )
    completion_ok = bool(status.get("ok")) and bool(completion_gate.get("ok"))
    required_next_action_key = str(
        status.get("required_next_action_key")
        or status.get("next_action_key")
        or "inspect_bundle"
    )
    required_action_kind = str(
        status.get("required_action_kind")
        or completion_gate.get("required_action_kind")
        or "machine"
    )
    accepted_decision_count = int(status.get("accepted_decision_count") or 0)
    corrected_transcript_missing = not bool(status.get("corrected_transcript_exists"))
    artifact_identity = (
        status.get("artifact_identity")
        if isinstance(status.get("artifact_identity"), dict)
        else {}
    )
    stale = bool(artifact_identity) and not bool(artifact_identity.get("current"))
    machine_discovery_pending = not bool(
        status.get("candidate_discovery_complete")
    )
    machine_closure_pending = workflow_status in {
        "not_started",
        "needs_pack",
        "needs_validation",
        "needs_closure",
        "needs_impact_report",
        "needs_readable_impact_report",
        "needs_summary_impact_report",
        "readable_impact_needs_fix",
        "summary_impact_needs_fix",
        "stale_validation_pack",
        "needs_candidate_discovery",
        "quality_gate_needs_fix",
    }
    machine_completion_pending = bool(
        not completion_ok and required_action_kind != "human"
    )
    complete_statuses = {
        "impact_passed",
        "no_candidates",
        "arbitrated_no_change",
    }
    covered = int(
        workflow_status in complete_statuses and completion_ok
    )
    review_only = bool(
        (review_required_count > 0 or required_action_kind == "human")
        and not machine_discovery_pending
        and not machine_closure_pending
        and not machine_completion_pending
        and not stale
        and not corrected_transcript_missing
        and unadjudicated_candidate_count == 0
    )
    blocker_count = int(
        machine_completion_pending
    )
    channel = _channel(
        "transcript_semantic_correction",
        "转写语义准确性",
        1,
        covered,
        "transcript_semantic_correction_status",
        str(
            manifest.get("mcp_transcript_semantic_correction_status_args")
            or "mcp-transcript-semantic-correction-status.args.json"
        ),
        (
            "完整性、语义准确性和正文投影分别验收；候选未扫描、未裁决、Hash 漂移或闭合未完成时不得宣称已完成。"
        ),
        blockers=blocker_count,
    )
    if review_only:
        channel["status"] = "weak"
        channel["blocker_count"] = 0
        channel["review_required_count"] = review_required_count
    channel["unadjudicated_candidate_count"] = unadjudicated_candidate_count
    channel["completion_gate"] = completion_gate
    channel["required_next_action_key"] = required_next_action_key
    summary = {
        "required": True,
        "status": workflow_status,
        "ok": completion_ok,
        "completion_gate": completion_gate,
        "required_next_action_key": required_next_action_key,
        "required_action_kind": required_action_kind,
        "candidate_discovery_status": discovery_status,
        "candidate_count": int(status.get("candidate_count") or 0),
        "accepted_decision_count": accepted_decision_count,
        "review_required_count": review_required_count,
        "unadjudicated_candidate_count": unadjudicated_candidate_count,
        "candidate_discovery_complete": bool(
            status.get("candidate_discovery_complete")
        ),
        "artifact_identity_current": not stale,
        "corrected_transcript_exists": bool(
            status.get("corrected_transcript_exists")
        ),
        "corrected_transcript_missing_after_acceptance": corrected_transcript_missing,
        "next_action_key": str(status.get("next_action_key") or ""),
        "canonical_transcript": str(canonical),
        "error": str(status.get("error") or ""),
    }
    return channel, summary


def _canonical_transcript_coverage_indexes(
    bundle_dir: str | Path | None,
    *,
    timeline_count: int,
) -> set[int]:
    """Reuse the canonical transcript alignment audit for speech coverage."""

    if bundle_dir is None or timeline_count <= 0:
        return set()
    root = Path(bundle_dir).expanduser().resolve()
    audit_path = root / "timeline-alignment-audit.json"
    timeline_path = root / "timeline.json"
    if not audit_path.is_file():
        return set()
    try:
        if (
            timeline_path.is_file()
            and audit_path.stat().st_mtime_ns < timeline_path.stat().st_mtime_ns
        ):
            return set()
        payload = read_json(audit_path)
    except (OSError, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    if not timeline_alignment_audit_is_current(root, payload):
        return set()
    summary = (
        payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    )
    if not bool(summary.get("transcript_available")):
        return set()
    if int(_number(summary.get("items"))) != timeline_count:
        return set()
    rows = payload.get("items") if isinstance(payload.get("items"), list) else []
    covered: set[int] = set()
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or int(_number(row.get("asr_overlap_count"))) <= 0:
            continue
        issues = {str(value) for value in row.get("issues") or []}
        if "missing_asr_overlap" in issues:
            continue
        index = int(_number(row.get("index"))) or position
        if 1 <= index <= timeline_count:
            covered.add(index)
    return covered


def _speech_recovery_action(
    manifest: dict[str, Any],
    *,
    bundle_dir: str | Path | None,
) -> tuple[str, str, str]:
    if bundle_dir is not None:
        root = Path(bundle_dir).expanduser().resolve()
        transcript_path = resolve_canonical_transcript_path(root, manifest)
        if transcript_path is not None and transcript_path.is_file():
            return (
                "timeline_alignment_audit",
                str(
                    manifest.get("mcp_timeline_alignment_audit_args")
                    or "mcp-timeline-alignment-audit.args.json"
                ),
                "Bundle 已有规范化转写，但当前时间线对齐审计缺失或与转写/时间线 Hash 不一致；只刷新本地绑定与覆盖审计，不重跑 ASR。",
            )
    return (
        "normalize_asr_output",
        "",
        "没有可复用的规范化转写会导致口头知识丢失；先生成或导入 FunASR/SenseVoice/WhisperX/faster-whisper 规范化输出。",
    )


def _triage_requirement_policy(
    bundle_dir: str | Path | None,
    *,
    timeline_count: int,
) -> dict[str, Any] | None:
    if bundle_dir is None:
        return None
    root = Path(bundle_dir).expanduser().resolve()
    triage_path = root / "vision-review-triage.json"
    timeline_path = root / "timeline.json"
    if not triage_path.is_file():
        return None
    try:
        if (
            timeline_path.is_file()
            and triage_path.stat().st_mtime_ns < timeline_path.stat().st_mtime_ns
        ):
            return None
        payload = read_json(triage_path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        str(payload.get("status") or "") != "ok"
        or str(payload.get("mode") or "") != "triage"
    ):
        return None
    if int(_number(payload.get("total_items"))) != timeline_count:
        return None

    def indexes(key: str) -> list[int]:
        return sorted(
            {
                int(_number(value))
                for value in payload.get(key) or []
                if 1 <= int(_number(value)) <= timeline_count
            }
        )

    return {
        "mode": "risk_based_triage",
        "fresh": True,
        "source": str(triage_path),
        "semantic_indexes": indexes("semantic_indexes"),
        "temporal_indexes": indexes("temporal_indexes"),
        "visual_structure_first_indexes": indexes("visual_structure_first_indexes"),
        "suppressed_count": int(
            _number((payload.get("selected_counts") or {}).get("suppressed"))
        ),
    }


def _timeline_index(item: dict[str, Any], position: int) -> int:
    value = int(_number(item.get("index")))
    return value if value > 0 else position


def _unique_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def render_knowledge_coverage_markdown(report: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: lecture-knowledge-coverage",
        f'created: "{report.get("checked_at", now_iso())}"',
        "---",
        "",
        "# 知识覆盖审计",
        "",
        "这个报告用于检查知识类视频是否尽量保留了语言、屏幕文字、关键帧、结构化视觉材料和原始证据。它不是摘要质量评分，而是漏项地图。",
        "",
        f"- 状态：`{report.get('status', 'unknown')}`",
        f"- 时间线片段：`{report.get('timeline_items', 0)}`",
        f"- 已路由片段：`{report.get('items_with_visual_route', 0)}`",
        f"- 单帧视觉理解：`{report.get('items_with_visual_understanding', 0)}`",
        f"- 连续片段理解：`{report.get('items_with_temporal_understanding', 0)}`",
        f"- 屏幕小字/低置信度：`{report.get('screen_text_low_confidence', 0)}`",
        f"- OCR 空壳/无有效文字：`{report.get('ocr_text_empty', 0)}`",
        f"- 下一步：`{(report.get('next_action') or {}).get('key', '')}` / {(report.get('next_action') or {}).get('label', '')}",
        "",
        "## 覆盖矩阵",
        "",
        "| 通道 | 状态 | 覆盖 | Blockers | 建议工具 | MCP args |",
        "|---|---|---:|---:|---|---|",
    ]
    for channel in report.get("channels") or []:
        if not isinstance(channel, dict):
            continue
        lines.append(
            "| {label} | `{status}` | {covered}/{expected} ({percent}%) | {blockers} | `{tool}` | `{args}` |".format(
                label=_md_cell(str(channel.get("label") or channel.get("key") or "")),
                status=channel.get("status", ""),
                covered=channel.get("covered_count", 0),
                expected=channel.get("expected_count", 0),
                percent=channel.get("coverage_percent", 0),
                blockers=channel.get("blocker_count", 0),
                tool=channel.get("mcp_tool", ""),
                args=channel.get("mcp_args_path", ""),
            )
        )
    lines.extend(["", "## 说明", ""])
    stage = report.get("summary_stage_coverage") or {}
    if stage.get("required"):
        lines.extend([f"- 重要信息保留：`{_md_cell(stage.get('status', 'not_assessed'))}`；来源→章节材料→准备输入→最终正文。",
                      "- 该项不改变抽取通道的下一步；总结质量门负责拒绝漏点，不能因此重复运行 OCR/ASR。",
                      "- 准备输入不代表模型实际已读；全片完整性和现实事实真伪未由此证明。"])
    for channel in report.get("channels") or []:
        if not isinstance(channel, dict):
            continue
        samples = ", ".join(str(value) for value in channel.get("sample_indexes") or [])
        suffix = f" 样例片段：{samples}" if samples else ""
        lines.append(
            f"- **{channel.get('label', '')}**：{channel.get('hint', '')}{suffix}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _channel(
    key: str,
    label: str,
    expected: int,
    covered: int,
    mcp_tool: str,
    mcp_args_path: str,
    hint: str,
    *,
    blockers: int = 0,
    samples: list[int] | None = None,
) -> dict[str, Any]:
    percent = round((covered / expected * 100), 1) if expected else 100.0
    status = "ok"
    if blockers > 0:
        status = "blocked"
    elif expected and covered == 0:
        status = "blocked"
    elif expected and percent < 80:
        status = "weak"
    return {
        "key": key,
        "label": label,
        "status": status,
        "expected_count": expected,
        "covered_count": covered,
        "coverage_percent": percent,
        "blocker_count": int(blockers),
        "mcp_tool": mcp_tool,
        "mcp_args_path": mcp_args_path,
        "hint": hint,
        "sample_indexes": (samples or [])[:10],
    }


def _source_channel(
    source_summary: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    expected = int(_number(source_summary.get("source_count")))
    covered = int(_number(source_summary.get("sources_with_artifacts")))
    missing = max(expected - covered, 0)
    channel = _channel(
        "source_artifacts",
        "原始抽取物回溯",
        expected,
        covered,
        "bundle_source_artifacts",
        str(manifest.get("mcp_source_artifacts_args") or ""),
        "结构化笔记可疑时必须能回到 vidclaude/peepshow/vidwise 的真实输出核对。",
        blockers=0,
    )
    if missing > 0:
        channel["status"] = "weak"
        channel["missing_count"] = missing
        channel["blocker_count"] = missing
    return channel


def _time_gap_channel(
    coverage: dict[str, Any], manifest: dict[str, Any], time_gap_count: int
) -> dict[str, Any]:
    percent = _number(coverage.get("timeline_coverage_percent"))
    expected = 100
    covered = int(round(percent)) if percent else (0 if time_gap_count else 100)
    channel = _channel(
        "time_axis",
        "时间轴连续性",
        expected,
        covered,
        "run_frame_recapture_plan",
        str(manifest.get("mcp_frame_recapture_args") or ""),
        "时间轴空白段需要抽帧或人工确认，避免中间内容完全没有进入知识库。",
        blockers=time_gap_count,
    )
    channel["time_gap_count"] = time_gap_count
    channel["max_time_gap_seconds"] = coverage.get("max_time_gap_seconds", 0)
    return channel


def _next_action(channels: list[dict[str, Any]]) -> dict[str, Any]:
    if not channels:
        return {
            "key": "ready_for_review",
            "label": "覆盖审计通过",
            "hint": "进入人工复核和最终导出 gate。",
        }
    first = channels[0]
    return {
        "key": str(first.get("key") or "coverage_gap"),
        "label": str(first.get("label") or "补齐覆盖缺口"),
        "mcp_tool": str(first.get("mcp_tool") or ""),
        "mcp_args_path": str(first.get("mcp_args_path") or ""),
        "hint": str(first.get("hint") or ""),
    }


def _executable_next_action(
    action: Any, *, root: Path, fallback_args_path: Path
) -> dict[str, Any]:
    base = action if isinstance(action, dict) else {}
    tool = str(base.get("mcp_tool") or "audit_knowledge_coverage")
    args_path = _resolve_bundle_path(
        root, base.get("mcp_args_path") or fallback_args_path
    )
    return {
        **base,
        "mcp_tool": tool,
        "mcp_args_path": str(args_path),
        "command": _mcp_command(tool, args_path),
    }


def _resolve_bundle_path(root: Path, value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        return root
    path = Path(text)
    return path if path.is_absolute() else root / path


def _mcp_command(tool: str, args_path: Path) -> str:
    escaped_args_path = str(args_path).replace("'", "''")
    return f".\\scripts\\video-knowledge.ps1 mcp-call {tool} '{escaped_args_path}'"


def _needs_ocr(item: dict[str, Any]) -> bool:
    if _human_review_accepted(item) and (
        _human_keep_image(item) or _has_meaningful_visual_text(item)
    ):
        return False
    issues = set(_quality_issues(item))
    return bool(
        issues
        & {
            "missing_ocr",
            "low_ocr_confidence",
            "screen_text_low_confidence",
            "structured_visual_without_structure",
            "ocr_text_empty",
        }
    ) or (_has_structured_type(item) and not _has_meaningful_visual_text(item))


def _has_structured_type(item: dict[str, Any]) -> bool:
    material_types = {str(value) for value in item.get("material_types") or []}
    return bool(material_types & {"formula", "table", "code"})


def _structured_visual(item: dict[str, Any]) -> list[Any]:
    value = item.get("structured_visual")
    if not isinstance(value, list):
        return []
    return [entry for entry in value if _has_meaningful_structured_entry(entry, item)]


def _has_meaningful_visual_text(item: dict[str, Any]) -> bool:
    return bool(
        _meaningful_visual_text(item.get("visual_text"), item)
        or _meaningful_visual_text(item.get("original_visual_text"), item)
        or _meaningful_human_text(item.get("human_corrected_visual_text"))
        or _meaningful_human_text(_human_review(item).get("corrected_visual_text"))
        or (_human_review_accepted(item) and _human_keep_image(item))
    )


def _has_meaningful_structured_entry(entry: Any, item: dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return bool(_meaningful_human_text(entry))
    for key in ("markdown", "text", "content", "value"):
        if _meaningful_visual_text(entry.get(key), item):
            return True
    return False


def _meaningful_visual_text(value: Any, item: dict[str, Any]) -> str:
    text = _text(value)
    if not text:
        return ""
    stems = _frame_stems(item)
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            stripped.startswith("<!--")
            and stripped.endswith("-->")
            and "source:" in stripped.lower()
        ):
            continue
        if stripped.startswith("# ") and stripped[2:].strip() in stems:
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()


def _meaningful_human_text(value: Any) -> str:
    return _text(value)


def _frame_stems(item: dict[str, Any]) -> set[str]:
    stems: set[str] = set()
    for key in ("frame_paths", "evidence_paths"):
        values = item.get(key)
        if isinstance(values, list):
            for value in values:
                text = str(value or "").strip()
                if text:
                    stems.add(Path(text).stem)
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            text = str(asset.get("path") or asset.get("source") or "").strip()
            if text:
                stems.add(Path(text).stem)
    for entry in item.get("structured_visual") or []:
        if isinstance(entry, dict):
            for key in ("image_path", "source"):
                text = str(entry.get(key) or "").strip()
                if text and Path(text).suffix:
                    stems.add(Path(text).stem)
    return {stem for stem in stems if stem}


def _asset_paths(item: dict[str, Any]) -> list[str]:
    paths = []
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            path = str(asset.get("path") or asset.get("source") or "").strip()
            if path:
                paths.append(path)
    return paths


def _has_asset_gap(
    item: dict[str, Any], *, bundle_dir: str | Path | None = None
) -> bool:
    for asset in item.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        if (
            str(asset.get("copied") or "").lower() == "false"
            or asset.get("exists") is False
        ):
            return True
        asset_path = str(asset.get("path") or "").strip()
        if (
            bundle_dir
            and asset_path
            and not asset_path.startswith(("http://", "https://"))
        ):
            path = Path(asset_path)
            candidate = path if path.is_absolute() else Path(bundle_dir) / path
            if not candidate.exists():
                return True
    return False


def _quality_issues(item: dict[str, Any]) -> list[str]:
    return _dedupe(
        [str(issue) for issue in item.get("quality_issues") or [] if str(issue)]
        + _inferred_screen_text_issues(item)
    )


def _inferred_screen_text_issues(item: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if _is_wrapper_only_visual_text(item):
        issues.append("ocr_text_empty")
    route = str(item.get("visual_route") or "")
    material_types = {str(value) for value in item.get("material_types") or []}
    signals = {str(value) for value in item.get("signals") or []}
    ui_tokens = {
        "ui",
        "software",
        "browser",
        "editor",
        "code_editor",
        "terminal",
        "screen",
        "interface",
        "app",
        "operation",
        "mouse",
        "workflow",
    }
    if (
        route in {"semantic_frame", "temporal_sequence", "mixed"}
        and bool((material_types | signals) & ui_tokens)
        and not _human_review_accepted(item)
    ):
        issues.append("screen_text_low_confidence")
    return issues


def _is_wrapper_only_visual_text(item: dict[str, Any]) -> bool:
    text = _text(item.get("visual_text"))
    if not text:
        return False
    return not bool(_meaningful_visual_text(text, item))


def _sample_indexes(items: list[dict[str, Any]], limit: int = 10) -> list[int]:
    return [
        int(_number(item.get("index")))
        for item in items[:limit]
        if _number(item.get("index")) > 0
    ]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _non_empty_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _valid_visual_understanding(value: Any) -> bool:
    if not _non_empty_mapping(value):
        return False
    if value.get("parse_failed") or value.get("validation_status") == "incomplete":
        return False
    has_content = bool(
        _as_list(value.get("objects"))
        or _as_list(value.get("actions"))
        or _text(value.get("interface_state"))
        or _as_list(value.get("spatial_relations"))
        or _as_list(value.get("non_text_information"))
        or _text(value.get("instructor_focus"))
    )
    return has_content and bool(_as_list(value.get("evidence_frame_paths")))


def _has_visual_understanding(item: dict[str, Any]) -> bool:
    return _shared_has_visual_evidence(item)


def _has_temporal_understanding(item: dict[str, Any]) -> bool:
    return _shared_has_temporal_evidence(item)


def _human_review_accepted(item: dict[str, Any]) -> bool:
    return _shared_human_review_accepted(item)


def _human_review(item: dict[str, Any]) -> dict[str, Any]:
    return (
        item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    )


def _human_keep_image(item: dict[str, Any]) -> bool:
    human_review = _human_review(item)
    return bool(item.get("human_keep_image") or human_review.get("keep_image"))


def _human_review_structured_fallback(item: dict[str, Any]) -> bool:
    return (
        _has_structured_type(item)
        and _human_review_accepted(item)
        and (
            _human_keep_image(item)
            or _has_meaningful_visual_text(item)
            or _valid_visual_understanding(
                item.get("human_corrected_visual_understanding")
            )
        )
    )


def _as_list(value: Any) -> list[Any]:
    return (
        value if isinstance(value, list) else ([] if value in (None, "") else [value])
    )


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
