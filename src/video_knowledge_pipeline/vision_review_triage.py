from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .scene_detection_adapter import scene_boundary_candidates
from .storage import bundle_write_lock, read_json, write_json
from .transcript import parse_transcript, transcript_excerpt
from .vision_escalation_gate import (
    apply_duplicate_suppression,
    evaluate_escalation_signals,
    execution_estimate,
)

SCHEMA = "video_knowledge_pipeline.vision_review_triage.v2"
LEGACY_SCHEMA = "video_knowledge_pipeline.vision_review_triage.v1"
TAGGER_PRODUCER_FIELDS = ('executor','model_actual','run_id','frame_id','timestamp',
                         'recognition_binding','recognition_provenance','recognition_origin')

DOCUMENT_ROUTES = {"document_visual", "mixed"}
SEMANTIC_ROUTES = {"semantic_frame", "mixed"}
TEMPORAL_ROUTES = {"temporal_sequence", "mixed"}

DEICTIC_TERMS = [
    "这个",
    "这里",
    "上面",
    "下面",
    "左边",
    "右边",
    "屏幕",
    "画面",
    "看一下",
    "展示",
    "表格",
    "二维码",
]
TEMPORAL_TERMS = [
    "点击",
    "打开",
    "切换",
    "选择",
    "输入",
    "提交",
    "拖动",
    "滚动",
    "上传",
    "演示",
    "操作",
    "流程",
    "后台",
    "页面",
    "按钮",
]


def vision_review_triage(
    bundle_dir: str | Path,
    *,
    mode: str = "triage",
    tagger_json: str | Path | None = None,
    semantic_limit: int | None = None,
    temporal_limit: int | None = None,
    visual_structure_limit: int | None = None,
    min_score: int = 3,
    write: bool = True,
) -> dict[str, Any]:
    """Plan multimodal and ebook/document parsing work for extracted frames.

    `fast` is the production mode: it keeps a time-distributed subset of risky
    frames so transcript delivery is not blocked by exhaustive visual enrichment.
    `triage` selects every suspicious ASR/OCR/frame-route conflict. `full` selects
    every timeline item with frame evidence for ebook/document parsing and
    single-frame multimodal review; temporal/mixed routes are also selected.
    `tagger_json` can carry Qinglong/manual tags by index or timestamp.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")

    mode = str(mode or "triage").strip().lower()
    if mode not in {"fast", "triage", "full"}:
        raise ValueError("mode must be 'fast', 'triage', or 'full'")

    manifest = _as_dict(read_json(manifest_path)) if manifest_path.exists() else {}
    timeline = _as_timeline(read_json(timeline_path))
    _overlay_canonical_transcript(root, manifest, timeline)
    tagger_annotations = _load_tagger_annotations(tagger_json, timeline)
    scene_boundaries = scene_boundary_candidates(root)
    ranked = []
    for position, item in enumerate(timeline, start=1):
        index = _int(item.get("index")) or position
        ranked.append(
            _score_item(
                item,
                position=position,
                root=root,
                scene_boundaries=scene_boundaries,
                tagger_annotations=tagger_annotations.get(index, []),
            )
        )
    if mode in {"fast", "triage"}:
        apply_duplicate_suppression(ranked)

    if mode == "full":
        semantic_rows = [row for row in ranked if row["has_frame_evidence"]]
        visual_rows = [row for row in ranked if row["has_frame_evidence"]]
        temporal_rows = [
            row
            for row in ranked
            if row["has_frame_evidence"]
            and (
                row["visual_route"] in TEMPORAL_ROUTES
                or row["has_temporal_frame_evidence"]
            )
        ]
        for row in semantic_rows:
            row["recommended_action"] = "semantic_multimodal"
            row["reasons"] = _unique(
                [*row["reasons"], "full_mode_all_frame_multimodal"]
            )
        for row in visual_rows:
            row["visual_structure_reasons"] = _unique(
                [*row["visual_structure_reasons"], "full_mode_all_frame_ebook_markdown"]
            )
        for row in temporal_rows:
            row["temporal_reasons"] = _unique(
                [*row["temporal_reasons"], "full_mode_temporal_or_mixed_frame_group"]
            )
    else:
        threshold = max(0, int(min_score or 0))
        semantic_rows = [
            row
            for row in ranked
            if row["recommended_action"] == "semantic_multimodal"
            and row["has_frame_evidence"]
            and row["score"] >= threshold
        ]
        temporal_rows = [
            row
            for row in ranked
            if row["recommended_action"] == "temporal_multimodal"
            and row["has_frame_evidence"]
            and row["score"] >= threshold
        ]
        visual_rows = [
            row
            for row in ranked
            if row["recommended_action"] == "visual_structure_first"
            and row["has_frame_evidence"]
            and row["score"] >= threshold
        ]
    recapture_rows = [
        row
        for row in ranked
        if row.get("local_prerequisite_action") == "capture_temporal_frames"
    ]

    if mode == "fast":
        semantic_limit = 4 if semantic_limit is None else semantic_limit
        temporal_limit = 1 if temporal_limit is None else temporal_limit
        visual_structure_limit = 6 if visual_structure_limit is None else visual_structure_limit
        semantic_rows = _spread_limit(semantic_rows, semantic_limit)
        temporal_rows = _spread_limit(temporal_rows, temporal_limit)
        visual_rows = _spread_limit(visual_rows, visual_structure_limit)
    else:
        sorter = _sort_rows if mode == "triage" else _sort_by_index
        semantic_rows = _limit(sorter(semantic_rows), semantic_limit)
        temporal_rows = _limit(sorter(temporal_rows), temporal_limit)
        visual_rows = _limit(sorter(visual_rows), visual_structure_limit)

    result = {
        "schema": SCHEMA,
        "compatible_schemas": [LEGACY_SCHEMA],
        "bundle_dir": str(root),
        "status": "ok",
        "mode": mode,
        "default_mode": "fast",
        "write": bool(write),
        "tagger_json": str(Path(tagger_json).expanduser().resolve())
        if tagger_json
        else "",
        "tagger_annotations_count": sum(
            len(rows) for rows in tagger_annotations.values()
        ),
        "tagger_indexes": sorted(tagger_annotations),
        "min_score": int(min_score or 0),
        "total_items": len(timeline),
        "frame_items": sum(1 for row in ranked if row["has_frame_evidence"]),
        "selected_counts": {
            "semantic": len(semantic_rows),
            "temporal": len(temporal_rows),
            "visual_structure_first": len(visual_rows),
            "suppressed": sum(
                1
                for row in ranked
                if row.get("suppression_reasons")
                and row.get("recommended_action") == "none"
            ),
            "temporal_recapture": len(recapture_rows),
        },
        "selection_policy": {
            "strategy": "balanced_progressive_escalation",
            "local_evidence_first": True,
            "scene_boundaries_source": "exports/scene-detection.json",
            "static_main_region_threshold": 0.012,
            "dynamic_main_region_threshold": 0.04,
            "ocr_high_confidence_threshold": 0.85,
            "presenter_or_overlay_region_policy": "explicit_normalized_regions_or_adaptive_localized_motion",
            "fixed_presenter_position_assumed": False,
            "temporal_multimodal_requires": "broad_dynamic_or_scene_boundary_or_localized_motion_with_operation_evidence",
            "near_duplicate_text_similarity": 0.97,
            "near_duplicate_dhash_distance": 4,
            "remote_execution_requires_preflight_and_consent": True,
            "delivery_policy": "transcript_first_visual_enrichment_nonblocking"
            if mode == "fast"
            else "visual_enrichment_before_quality_closure",
            "effective_limits": {
                "semantic": semantic_limit,
                "temporal": temporal_limit,
                "visual_structure_first": visual_structure_limit,
            },
        },
        "semantic_indexes": [row["index"] for row in semantic_rows],
        "temporal_indexes": [row["index"] for row in temporal_rows],
        "visual_structure_first_indexes": [row["index"] for row in visual_rows],
        "suppressed_indexes": [
            row["index"]
            for row in ranked
            if row.get("suppression_reasons")
            and row.get("recommended_action") == "none"
        ],
        "temporal_recapture_indexes": [row["index"] for row in recapture_rows],
        "estimated_totals": {
            "model_calls": sum(
                int(row.get("estimated_model_calls") or 0)
                for row in [*semantic_rows, *temporal_rows, *visual_rows]
            ),
            "images": sum(
                int(row.get("estimated_images") or 0)
                for row in [*semantic_rows, *temporal_rows, *visual_rows]
            ),
        },
        "semantic_candidates": semantic_rows,
        "temporal_candidates": temporal_rows,
        "visual_structure_first_candidates": visual_rows,
        "temporal_recapture_candidates": recapture_rows,
        "all_ranked_candidates": [
            row
            for row in _sort_rows(ranked)
            if row["score"] > 0
            or row.get("tagger_annotations")
            or row.get("suppression_reasons")
            or mode == "full"
        ],
    }
    result["next_actions"] = _next_actions(root, result)

    if write:
        with bundle_write_lock(
            root, operation="vision_review_triage", timeout_seconds=1.0
        ):
            _write_outputs(root, manifest, result)
    return result


def _score_item(
    item: dict[str, Any],
    *,
    position: int,
    root: Path,
    scene_boundaries: list[dict[str, Any]],
    tagger_annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    index = _int(item.get("index")) or position
    route = str(item.get("visual_route") or "unknown")
    transcript = _text(
        item.get("transcript")
        or item.get("text")
        or item.get("asr_text")
        or item.get("transcript_excerpt")
    )
    visual_text = _text(item.get("visual_text") or item.get("ocr_text"))
    issues = _issue_keys(item)
    # Timeline coverage flags may predate the canonical transcript overlay.
    # Do not keep reporting a speech gap after a time-aligned transcript excerpt
    # has been supplied from the corrected/normalized transcript sidecar.
    if transcript:
        issues.discard("missing_transcript")
    tagger_annotations = tagger_annotations or []
    tags = _unique(
        [
            *_tags(item),
            *[
                tag
                for annotation in tagger_annotations
                for tag in annotation.get("tags", [])
            ],
        ]
    )
    frame_paths = _frame_paths(item, root=root)
    temporal_paths = [
        str(value) for value in item.get("temporal_frame_paths") or [] if value
    ]
    gate = evaluate_escalation_signals(
        root,
        item,
        transcript=transcript,
        visual_text=visual_text,
        tags=tags,
        frame_paths=frame_paths,
        scene_boundaries=scene_boundaries,
    )
    frame_change_status = str(
        gate["frame_change_evidence"].get("status") or "not_available"
    )
    scene_boundary_matched = bool(gate["scene_boundary_evidence"]["matched"])
    static_sequence = (
        frame_change_status in {"static", "presenter_only", "explicit_overlay_only"}
        and not scene_boundary_matched
    )
    localized_change_hint = _contains_any(transcript, TEMPORAL_TERMS) or _tag_contains(
        tags, ["操作", "步骤", "流程", "演示", "动态"]
    )
    temporal_hint = (
        route in TEMPORAL_ROUTES
        or localized_change_hint
        or "temporal_sequence_without_analysis" in issues
    )
    semantic_analysis_ready = _analysis_has_content(
        item.get("visual_understanding"),
        fields=(
            "objects",
            "actions",
            "interface_state",
            "spatial_relations",
            "instructor_focus",
            "non_text_information",
        ),
    )
    temporal_analysis_ready = _analysis_has_content(
        item.get("temporal_visual_understanding"),
        fields=(
            "event_sequence",
            "state_changes",
            "operation_steps",
            "causal_links",
        ),
    )
    verified_temporal_change = (
        frame_change_status == "dynamic"
        or scene_boundary_matched
        or (frame_change_status == "localized_motion" and localized_change_hint)
    )
    local_prerequisite_action = (
        "capture_temporal_frames"
        if temporal_hint
        and not temporal_analysis_ready
        and frame_change_status == "not_available"
        else "none"
    )
    suppression_reasons: list[str] = []

    semantic_score = 0
    temporal_score = 0
    visual_score = 0
    semantic_reasons: list[str] = []
    temporal_reasons: list[str] = []
    visual_reasons: list[str] = []

    if route in SEMANTIC_ROUTES and not item.get("visual_understanding"):
        semantic_reasons.append("semantic_frame_without_analysis")
    if route in TEMPORAL_ROUTES and not item.get("temporal_visual_understanding"):
        if static_sequence:
            suppression_reasons.append("static_sequence_suppresses_temporal_multimodal")
        elif verified_temporal_change:
            temporal_score += 4
            temporal_reasons.append("temporal_sequence_without_analysis")
        elif frame_change_status == "not_available":
            suppression_reasons.append("temporal_evidence_missing_requires_recapture")
        elif frame_change_status == "localized_motion":
            suppression_reasons.append("localized_motion_without_operation_evidence")
        else:
            suppression_reasons.append("low_change_requires_local_review_or_recapture")
    if route in DOCUMENT_ROUTES and not item.get("structured_visual"):
        visual_score += 2
        visual_reasons.append("document_visual_without_structure")
    if not visual_text:
        if (
            route in DOCUMENT_ROUTES
            or "missing_visual_text" in issues
            or "ocr_text_empty" in issues
        ):
            visual_score += 3
            visual_reasons.append("missing_visual_text")
        if _contains_any(transcript, DEICTIC_TERMS):
            semantic_score += 2
            semantic_reasons.append("transcript_points_to_screen_but_ocr_missing")
    elif _weak_visual_text(visual_text):
        visual_score += 2
        visual_reasons.append("visual_text_low_information")
    if gate["ocr_evidence"]["status"] == "low_confidence":
        visual_score += 3
        visual_reasons.append("ocr_confidence_below_threshold")
    if _contains_any(transcript, DEICTIC_TERMS):
        semantic_score += 2
        semantic_reasons.append("deictic_transcript_requires_screen_check")
    if _contains_any(transcript, TEMPORAL_TERMS):
        if static_sequence:
            suppression_reasons.append("operation_language_without_main_region_change")
        elif verified_temporal_change:
            temporal_score += 2
            temporal_reasons.append("operation_language_requires_sequence_check")
        else:
            suppression_reasons.append("operation_language_requires_temporal_recapture")
    if frame_change_status == "dynamic":
        temporal_score += 2
        temporal_reasons.append("verified_main_region_change")
    if scene_boundary_matched:
        temporal_score += 2
        temporal_reasons.append("pyscenedetect_boundary_in_item")
    if gate["complex_layout_signals"]:
        semantic_score += 4
        semantic_reasons.append("complex_relational_layout")
    if gate["non_text_signals"]:
        semantic_score += 2
        semantic_reasons.append("non_text_visual_information")
    if _tag_contains(tags, ["疑难", "易错", "重点", "工具名", "术语", "名称", "复核"]):
        semantic_score += 2
        semantic_reasons.append("tagged_for_multimodal_review")
    if _tag_contains(tags, ["操作", "步骤", "流程", "演示", "动态"]):
        if verified_temporal_change:
            temporal_score += 2
            temporal_reasons.append("tagged_for_temporal_review")
        else:
            suppression_reasons.append(
                "temporal_tag_requires_verified_change_or_recapture"
            )
    if _tag_contains(tags, ["OCR", "屏幕文字", "课件", "表格", "公式", "代码"]):
        visual_score += 2
        visual_reasons.append("tagged_for_document_visual_review")
    tag_suppression_reasons: list[str] = []
    if _tag_contains(
        tags, ["闲聊", "过渡", "重复", "铺垫", "口水", "低价值", "无信息"]
    ):
        semantic_score = max(0, semantic_score - 3)
        temporal_score = max(0, temporal_score - 3)
        visual_score = max(0, visual_score - 3)
        tag_suppression_reasons.append("tagged_as_low_value_or_repetitive")
    if _number_mismatch(transcript, visual_text):
        semantic_score += 3
        visual_score += 1
        semantic_reasons.append("number_mismatch_between_asr_and_ocr")
    if _term_mismatch(transcript, visual_text):
        semantic_score += 2
        semantic_reasons.append("term_mismatch_between_asr_and_ocr")
    term_candidates = (
        item.get("term_candidates")
        if isinstance(item.get("term_candidates"), list)
        else []
    )
    if any(
        isinstance(row, dict) and row.get("needs_human_review")
        for row in term_candidates
    ):
        semantic_score += 3
        semantic_reasons.append("term_resolution_needs_review")
    for issue in issues:
        if issue in {
            "ocr_text_empty",
            "screen_text_low_confidence",
            "missing_visual_text",
            "structured_visual_without_structure",
        }:
            visual_score += 2
            visual_reasons.append(issue)
        if issue in {"missing_visual_understanding", "semantic_frame_without_analysis"}:
            semantic_reasons.append(issue)
        if issue in {"term_resolution_needs_review", "tagger_marked_term_sensitive"}:
            semantic_score += 2
            semantic_reasons.append(issue)
        if issue == "temporal_sequence_without_analysis":
            if verified_temporal_change:
                temporal_score += 2
                temporal_reasons.append(issue)
            elif frame_change_status == "not_available":
                suppression_reasons.append("temporal_issue_requires_recapture")

    if semantic_score <= 0 and set(semantic_reasons).intersection(
        {"semantic_frame_without_analysis", "missing_visual_understanding"}
    ):
        suppression_reasons.append(
            "semantic_analysis_absence_without_positive_visual_signal"
        )
    if gate["ocr_sufficient_simple_layout"] and route in DOCUMENT_ROUTES:
        visual_score = 0
        suppression_reasons.append("ocr_sufficient_simple_layout")
    if not verified_temporal_change:
        temporal_score = 0

    if semantic_analysis_ready:
        semantic_score = 0
        suppression_reasons.append("semantic_analysis_already_available")
    if temporal_analysis_ready:
        temporal_score = 0
        suppression_reasons.append("temporal_analysis_already_available")
    action, score, reasons = sorted(
        [
            ("temporal_multimodal", temporal_score, temporal_reasons),
            ("semantic_multimodal", semantic_score, semantic_reasons),
            ("visual_structure_first", visual_score, visual_reasons),
        ],
        key=lambda row: (-row[1], row[0]),
    )[0]
    if score <= 0:
        action = "none"
        reasons = []
    estimated_calls, estimated_images, execution_location = execution_estimate(
        action, temporal_paths
    )
    benefit_reasons = _unique(
        [
            *reasons,
            *gate["cross_source_conflicts"],
            *(["complex_relational_layout"] if gate["complex_layout_signals"] else []),
            *(["non_text_visual_information"] if gate["non_text_signals"] else []),
        ]
    )

    return {
        "index": index,
        "start": item.get("start"),
        "end": item.get("end"),
        "visual_route": route,
        "score": int(score),
        "priority": _priority(score),
        "recommended_action": action,
        "selected_action": action,
        "reasons": _unique(reasons),
        "benefit_reasons": benefit_reasons,
        "suppression_reasons": _unique(
            [*suppression_reasons, *tag_suppression_reasons]
        ),
        "semantic_reasons": _unique(semantic_reasons),
        "temporal_reasons": _unique(temporal_reasons),
        "visual_structure_reasons": _unique(visual_reasons),
        "all_scores": {
            "semantic": semantic_score,
            "temporal": temporal_score,
            "visual_structure_first": visual_score,
        },
        "quality_issues": sorted(issues),
        "cross_source_conflicts": gate["cross_source_conflicts"],
        "complex_layout_signals": gate["complex_layout_signals"],
        "non_text_signals": gate["non_text_signals"],
        "ocr_evidence": gate["ocr_evidence"],
        "frame_change_evidence": gate["frame_change_evidence"],
        "scene_boundary_evidence": gate["scene_boundary_evidence"],
        "duplicate_of_index": None,
        "representative_fingerprint": gate["representative_fingerprint"],
        "tags": tags,
        "tagger_annotations": tagger_annotations,
        "tag_suppression_reasons": tag_suppression_reasons,
        "has_frame_evidence": bool(frame_paths),
        "has_temporal_frame_evidence": bool(item.get("temporal_frame_paths")),
        "has_structured_visual_evidence": bool(item.get("structured_visual")),
        "has_valid_visual_understanding": semantic_analysis_ready,
        "has_valid_temporal_visual_understanding": temporal_analysis_ready,
        "frame_paths": frame_paths[:12],
        "transcript_excerpt": _excerpt(transcript),
        "visual_text_excerpt": _excerpt(visual_text),
        "estimated_model_calls": estimated_calls,
        "estimated_images": estimated_images,
        "recommended_execution_location": execution_location,
        "local_prerequisite_action": local_prerequisite_action,
    }


def _analysis_has_content(value: Any, *, fields: tuple[str, ...]) -> bool:
    if not isinstance(value, dict) or value.get("parse_failed"):
        return False
    if str(value.get("validation_status") or "").strip().lower() in {
        "blocked",
        "failed",
        "incomplete",
        "invalid",
    }:
        return False
    return any(bool(value.get(field)) for field in fields)


def _write_outputs(
    root: Path, manifest: dict[str, Any], result: dict[str, Any]
) -> None:
    json_path = root / "vision-review-triage.json"
    md_path = root / "vision-review-triage.md"
    args_path = root / "mcp-vision-review-triage.args.json"
    preflight_path = root / "mcp-vision-review-triage-preflight.args.json"

    write_json(json_path, result)
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    write_json(
        args_path,
        {
            "bundle_dir": str(root),
            "mode": result["mode"],
            "tagger_json": result.get("tagger_json") or "",
            "min_score": result["min_score"],
            "write": True,
        },
    )
    write_json(
        preflight_path,
        {
            "bundle_dir": str(root),
            "semantic_indexes": result["semantic_indexes"],
            "temporal_indexes": result["temporal_indexes"],
            "semantic_limit": len(result["semantic_indexes"]),
            "temporal_limit": len(result["temporal_indexes"]),
            "include_semantic": bool(result["semantic_indexes"]),
            "include_temporal": bool(result["temporal_indexes"]),
            "write": True,
        },
    )
    manifest.update(
        {
            "vision_review_triage_json": str(json_path.relative_to(root)),
            "vision_review_triage_report": str(md_path.relative_to(root)),
            "mcp_vision_review_triage_args": str(args_path.relative_to(root)),
            "mcp_vision_review_triage_preflight_args": str(
                preflight_path.relative_to(root)
            ),
        }
    )
    write_json(root / "manifest.json", manifest)


def _render_markdown(result: dict[str, Any]) -> str:
    bundle = result["bundle_dir"]
    semantic = _csv(result["semantic_indexes"])
    temporal = _csv(result["temporal_indexes"])
    visual = _csv(result["visual_structure_first_indexes"])
    recapture = _csv(result["temporal_recapture_indexes"])
    lines = [
        "# Vision Review Triage",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Default mode: `{result['default_mode']}`",
        f"- Timeline items: `{result['total_items']}`",
        f"- Frame items: `{result['frame_items']}`",
        f"- Tagger annotations: `{result.get('tagger_annotations_count', 0)}`",
        "",
        "| Bucket | Count | Indexes |",
        "| --- | ---: | --- |",
        f"| multimodal single-frame | {result['selected_counts']['semantic']} | `{semantic or '-'}` |",
        f"| multimodal temporal | {result['selected_counts']['temporal']} | `{temporal or '-'}` |",
        f"| ebook/document visual | {result['selected_counts']['visual_structure_first']} | `{visual or '-'}` |",
        f"| local temporal recapture | {result['selected_counts']['temporal_recapture']} | `{recapture or '-'}` |",
        "",
        "## Commands",
        "",
    ]
    if semantic or temporal:
        command = [
            f'.\\scripts\\video-knowledge.ps1 vision-execution-preflight "{bundle}"'
        ]
        command.append(
            f'--semantic-indexes "{semantic}"' if semantic else "--no-semantic"
        )
        command.append(
            f'--temporal-indexes "{temporal}"' if temporal else "--no-temporal"
        )
        lines.extend(["```powershell", " `\n  ".join(command), "```", ""])
    if visual:
        lines.extend(
            [
                "```powershell",
                f'.\\scripts\\video-knowledge.ps1 run-visual-structure "{bundle}" --indexes "{visual}"',
                "```",
                "",
            ]
        )
    if recapture:
        lines.extend(
            [
                "```powershell",
                f'.\\scripts\\video-knowledge.ps1 run-temporal-frame-groups "{bundle}" --indexes "{recapture}" --frame-count 8',
                "```",
                "",
            ]
        )
    for title, key in [
        ("Multimodal Single-Frame", "semantic_candidates"),
        ("Multimodal Temporal", "temporal_candidates"),
        ("Ebook/Document Visual", "visual_structure_first_candidates"),
        ("Local Temporal Recapture", "temporal_recapture_candidates"),
    ]:
        lines.extend(
            [
                f"## {title}",
                "",
                "| Index | Time | Score | Reasons | ASR | OCR |",
                "| ---: | --- | ---: | --- | --- | --- |",
            ]
        )
        rows = result.get(key) or []
        for row in rows:
            reasons = ", ".join(
                row.get("reasons")
                or row.get("semantic_reasons")
                or row.get("visual_structure_reasons")
                or []
            )
            lines.append(
                f"| {row.get('index')} | {_time_range(row)} | {row.get('score')} | {_md(reasons)} | {_md(row.get('transcript_excerpt'))} | {_md(row.get('visual_text_excerpt'))} |"
            )
        if not rows:
            lines.append("| - | - | - | - | - | - |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _next_actions(root: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if result["temporal_recapture_indexes"]:
        actions.append(
            {
                "key": "run_temporal_frame_groups",
                "indexes": result["temporal_recapture_indexes"],
            }
        )
    if result["semantic_indexes"] or result["temporal_indexes"]:
        actions.append(
            {
                "key": "vision_execution_preflight",
                "mcp_args_path": str(
                    root / "mcp-vision-review-triage-preflight.args.json"
                ),
            }
        )
    if result["visual_structure_first_indexes"]:
        actions.append(
            {
                "key": "run_visual_structure",
                "indexes": result["visual_structure_first_indexes"],
            }
        )
    return actions


def _load_tagger_annotations(
    tagger_json: str | Path | None, timeline: list[dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    if not tagger_json:
        return {}
    path = Path(tagger_json).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"tagger_json not found: {path}")
    data = read_json(path)
    rows = _annotation_rows(data)
    if isinstance(data,dict) and 'recognition_contract' in data:
        from .general_tagger_adapter import RECOGNITION_CONTRACT
        if (data['recognition_contract'] != RECOGNITION_CONTRACT
                or any(not isinstance(row.get('recognition_binding'),dict) for row in rows)):
            raise ValueError('tagger producer binding missing/invalid')
    by_index: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        annotation = _normalise_annotation(row)
        if not annotation.get("tags") and not annotation.get("text"):
            continue
        index = _annotation_target_index(row, timeline)
        if index <= 0:
            continue
        annotation["source_path"] = str(path)
        by_index.setdefault(index, []).append(annotation)
    return by_index


def _annotation_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    for key in (
        "items",
        "segments",
        "timeline",
        "annotations",
        "labels",
        "tags",
        "results",
    ):
        value = data.get(key)
        if isinstance(value, list):
            rows = []
            for row in value:
                if isinstance(row, dict):
                    rows.append(row)
                elif isinstance(row, str):
                    rows.append({"tag": row})
            return rows
    rows: list[dict[str, Any]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            row = dict(value)
            row.setdefault("index", key)
            rows.append(row)
        elif isinstance(value, list):
            rows.append({"index": key, "tags": value})
        elif isinstance(value, str):
            rows.append({"index": key, "tag": value})
    return rows


def _normalise_annotation(row: dict[str, Any]) -> dict[str, Any]:
    tags: list[str] = []
    for key in ("tags", "labels", "material_types", "categories"):
        raw = row.get(key)
        if isinstance(raw, list):
            tags.extend(str(value) for value in raw if value)
        elif isinstance(raw, str):
            tags.extend(
                part.strip() for part in re.split(r"[,，;；/\s]+", raw) if part.strip()
            )
    for key in ("tag", "label", "category", "type", "reason", "topic"):
        if row.get(key):
            tags.append(str(row[key]))
    return {
        "schema": "video_knowledge_pipeline.tagger_annotation.v1",
        **{key:row[key] for key in TAGGER_PRODUCER_FIELDS if key in row},
        "tags": _unique(tags),
        "text": _text(row.get("text") or row.get("note") or row.get("description")),
        "priority": _text(
            row.get("priority") or row.get("importance") or row.get("risk")
        ),
        "raw_index": row.get("index")
        or row.get("timeline_index")
        or row.get("segment_index"),
        "start": row.get("start"),
        "end": row.get("end"),
        "time": row.get("time") or row.get("timestamp") or row.get("midpoint"),
        "model": _text(row.get("model")),
        "model_revision": _text(row.get("model_revision")),
        "artifact_path": _text(row.get("artifact_path")),
        "artifact_sha256": _text(row.get("artifact_sha256")),
        "tag_vocabulary": _text(row.get("tag_vocabulary")),
        "labels_en": _unique(
            [str(value) for value in row.get("labels_en", []) if str(value)]
        )
        if isinstance(row.get("labels_en"), list)
        else [],
        "labels_zh": _unique(
            [str(value) for value in row.get("labels_zh", []) if str(value)]
        )
        if isinstance(row.get("labels_zh"), list)
        else [],
        "candidate_only": bool(row.get("candidate_only", True)),
        "human_review_required": bool(row.get("human_review_required", True)),
    }


def _annotation_target_index(
    row: dict[str, Any], timeline: list[dict[str, Any]]
) -> int:
    for key in ("index", "timeline_index", "item_index", "segment_index"):
        if row.get(key) is not None:
            value = _int(row.get(key))
            if value > 0:
                return value
    start = _seconds(row.get("start"))
    end = _seconds(row.get("end"))
    point = _seconds(row.get("time") or row.get("timestamp") or row.get("midpoint"))
    if point is None and start is not None and end is not None:
        point = (start + end) / 2
    if point is None and start is None and end is None:
        return 0
    best_index = 0
    best_score = -1.0
    for position, item in enumerate(timeline, start=1):
        item_index = _int(item.get("index")) or position
        item_start = _seconds(item.get("start")) or 0.0
        item_end = _seconds(item.get("end"))
        if item_end is None:
            item_end = item_start
        score = 0.0
        if point is not None and item_start <= point <= item_end:
            score = 1.0
        elif start is not None and end is not None:
            overlap = max(0.0, min(end, item_end) - max(start, item_start))
            denom = max(0.001, end - start, item_end - item_start)
            score = overlap / denom
        if score > best_score:
            best_score = score
            best_index = item_index
    return best_index if best_score > 0 else 0


def _seconds(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    if ":" in text:
        parts = text.split(":")
        try:
            total = 0.0
            for part in parts:
                total = total * 60 + float(part.replace(",", "."))
            return total
        except ValueError:
            return None
    return None


def _issue_keys(item: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    raw = (
        item.get("quality_issues")
        or item.get("coverage_issues")
        or item.get("issues")
        or []
    )
    if isinstance(raw, dict):
        raw = raw.get("issues") or raw.get("gaps") or raw.get("reasons") or []
    if isinstance(raw, list):
        for issue in raw:
            if isinstance(issue, str):
                keys.add(issue)
            elif isinstance(issue, dict):
                key = issue.get("key") or issue.get("reason") or issue.get("type")
                if key:
                    keys.add(str(key))
    for key in [
        "missing_visual_text",
        "screen_text_low_confidence",
        "ocr_text_empty",
        "structured_visual_without_structure",
        "missing_visual_understanding",
        "semantic_frame_without_analysis",
        "temporal_sequence_without_analysis",
        "term_resolution_needs_review",
        "tagger_marked_term_sensitive",
    ]:
        if item.get(key):
            keys.add(key)
    return keys


def _tags(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("tags", "labels", "material_types"):
        raw = item.get(key)
        if isinstance(raw, list):
            values.extend(str(value) for value in raw if value)
    review = item.get("human_review")
    if isinstance(review, dict) and isinstance(review.get("tags"), list):
        values.extend(str(value) for value in review["tags"] if value)
    return _unique(values)


def _tag_contains(tags: list[str], needles: list[str]) -> bool:
    joined = " ".join(tags).lower()
    return any(str(needle).lower() in joined for needle in needles)


def _frame_paths(item: dict[str, Any], *, root: Path | None = None) -> list[str]:
    paths: list[str] = []
    for key in ("frame_paths", "temporal_frame_paths"):
        value = item.get(key)
        if isinstance(value, list):
            paths.extend(str(path) for path in value if path)
    if isinstance(item.get("assets"), list):
        for asset in item["assets"]:
            if isinstance(asset, dict) and asset.get("path"):
                paths.append(str(asset["path"]))
    for key in ("frame_path", "image_path", "screenshot_path", "recaptured_frame_path"):
        if item.get(key):
            paths.append(str(item[key]))
    unique = _unique(paths)
    if root is None:
        return unique
    existing: list[str] = []
    for value in unique:
        path = Path(value)
        candidate = path if path.is_absolute() else root / path
        if candidate.is_file():
            existing.append(str(candidate.resolve()))
    return existing


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _number_mismatch(transcript: str, visual_text: str) -> bool:
    if not transcript or not visual_text:
        return False
    left = set(re.findall(r"\d+(?:\.\d+)?", transcript))
    right = set(re.findall(r"\d+(?:\.\d+)?", visual_text))
    return bool(left and right and not left.intersection(right))


def _term_mismatch(transcript: str, visual_text: str) -> bool:
    if not transcript or not visual_text:
        return False
    left = _terms(transcript)
    right = _terms(visual_text)
    return bool(left and right and left.symmetric_difference(right))


def _terms(text: str) -> set[str]:
    return {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)}


def _weak_visual_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 8:
        return True
    useful = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", compact)
    return len(useful) < max(4, len(compact) // 3)


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows, key=lambda row: (-int(row.get("score") or 0), int(row.get("index") or 0))
    )


def _sort_by_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row.get("index") or 0))


def _spread_limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    size = int(limit or 0)
    ordered = _sort_by_index(rows)
    if size <= 0 or len(ordered) <= size:
        return ordered
    if size == 1:
        return [_sort_rows(ordered)[0]]
    positions = [
        round(offset * (len(ordered) - 1) / (size - 1))
        for offset in range(size)
    ]
    return [ordered[position] for position in positions]


def _limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit and int(limit) > 0:
        return rows[: int(limit)]
    return rows


def _priority(score: int) -> str:
    if score >= 7:
        return "high"
    if score >= 4:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def _time_range(row: dict[str, Any]) -> str:
    return f"{row.get('start', '?')}-{row.get('end', '?')}"


def _excerpt(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _overlay_canonical_transcript(
    root: Path,
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> None:
    candidates = [
        manifest.get("source_arbitrated_transcript_json"),
        manifest.get("corrected_transcript_json"),
        manifest.get("normalized_transcript_json"),
        manifest.get("transcript_json"),
        "source-arbitrated-transcript.json",
        "corrected-transcript.json",
        "normalized-transcript.json",
        "transcript.json",
    ]
    transcript_path: Path | None = None
    for value in candidates:
        if not value:
            continue
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            transcript_path = candidate.resolve()
            break
    if transcript_path is None:
        return
    try:
        cues = parse_transcript(transcript_path)
    except Exception:
        return
    if not cues:
        return
    for item in timeline:
        if _text(
            item.get("transcript")
            or item.get("corrected_transcript")
            or item.get("asr_text")
            or item.get("transcript_excerpt")
        ):
            continue
        try:
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
        except (TypeError, ValueError):
            continue
        excerpt = transcript_excerpt(cues, start, max(start, end)).strip()
        if excerpt:
            item["transcript"] = excerpt
            item["transcript_source"] = "canonical_transcript_overlay"



def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_timeline(value: Any) -> list[dict[str, Any]]:
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _csv(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def _md(value: Any) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "\\|")
    return text or "-"
