from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .artifact_freshness import build_dependency_snapshot, validate_dependency_snapshot
from .artifact_validation import artifact_evidence
from .canonical_json import canonical_json_sha256
from .models import now_iso
from .smart_summary_codex import CODEX_FILENAMES, _section_text
from .smart_summary_input_pack import select_canonical_transcript_path
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic


REPORT_SCHEMA = "video_knowledge_pipeline.video_decomposition_report.v1"
STATUS_SCHEMA = "video_knowledge_pipeline.video_decomposition_report_status.v1"
COMPARISON_SCHEMA = "video_knowledge_pipeline.video_decomposition_comparison.v1"

FINDING_STATUSES = {"confirmed", "inferred", "unavailable"}
MODALITIES = (
    "video",
    "audio",
    "transcript",
    "platform_metadata",
    "user_material",
    "link_acquisition",
)
FINDING_DIMENSIONS = {
    "positioning", "hook", "body_structure", "payoff", "content_value",
    "emotion", "expression", "visual_language", "subtitle_packaging",
    "voice_delivery", "bgm_style", "bgm_identity", "audiovisual_coordination",
    "reusable_framework", "technique", "non_copyable_factor", "imitation_risk",
    "differentiation", "author_identity", "behind_the_scenes", "performance_metrics",
}
CREATIVE_CATEGORIES = (
    "reusable_frameworks",
    "techniques",
    "non_copyable_factors",
    "imitation_risks",
    "differentiation",
)

UPSTREAM_REFERENCE = {
    "project": "liuliu-66-create/ll-video-decomposer",
    "commit": "8b4d57ce0dc8475751c372c8dc49c1088cee1e69",
    "license": "MIT",
    "reuse_mode": "independent_contract_adaptation",
    "copied_upstream_source": False,
}

_SENSITIVE_DIRECT_EVIDENCE = {
    "bgm_identity": {"platform_metadata", "reliable_music_recognition", "user_material"},
    "author_identity": {"platform_metadata", "user_material"},
    "behind_the_scenes": {"user_material"},
    "performance_metrics": {"platform_metadata", "user_material"},
}
_DIMENSION_MODALITY_RULES = {
    "visual_language": {"video"},
    "subtitle_packaging": {"video"},
    "voice_delivery": {"audio"},
    "bgm_style": {"audio"},
    "bgm_identity": {"audio", "platform_metadata", "user_material"},
    "audiovisual_coordination": {"video", "audio"},
    "author_identity": {"platform_metadata", "user_material"},
    "behind_the_scenes": {"user_material"},
    "performance_metrics": {"platform_metadata", "user_material"},
}
_VISUAL_KEYS = (
    "corrected_visual_text", "visual_text", "ocr_text", "structured_visual",
    "corrected_visual_understanding", "human_corrected_visual_understanding",
    "visual_understanding", "corrected_temporal_visual_understanding",
    "human_corrected_temporal_visual_understanding", "temporal_visual_understanding",
    "tagger_tags", "shot_type", "camera_movement", "composition",
)
_FRAME_KEYS = ("frame_path", "frame_paths", "temporal_frame_paths", "evidence_frame_paths")


class VideoDecompositionContractError(ValueError):
    pass


def build_video_decomposition_report(
    bundle_dir: str | Path,
    *,
    title: str = "",
    write: bool = True,
) -> dict[str, Any]:
    """Build a read-only, evidence-bound decomposition projection.

    Intent: expose VKP evidence downstream without creating a second truth
    source. Decision: reuse canonical transcript selection, canonical hashing,
    atomic storage, and dependency-snapshot freshness. Effective scope:
    Bundle-derived JSON/Markdown and Workbench links only.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_object(root / "manifest.json")
    if not manifest:
        raise FileNotFoundError(f"manifest.json not found or invalid: {root / 'manifest.json'}")
    timeline_path = root / "timeline.json"
    timeline = _read_list(timeline_path)
    transcript_path = select_canonical_transcript_path(root, manifest)
    summary_path = _canonical_summary_path(root)
    sources = _discover_sources(
        root,
        manifest=manifest,
        timeline_path=timeline_path if timeline_path.is_file() else None,
        timeline=timeline,
        transcript_path=transcript_path,
        summary_path=summary_path,
    )
    if not sources:
        raise ValueError("video decomposition requires at least one existing Bundle evidence artifact")

    source_artifacts = _source_artifact_references(sources)
    source_index = _source_index(source_artifacts)
    coverage = _modality_coverage(timeline, source_index)
    segments_source = timeline or _transcript_segments(transcript_path)
    findings, structure_segments, creative_strategy = _build_findings_and_structure(
        title=str(title or manifest.get("title") or root.name),
        rows=segments_source,
        summary_path=summary_path,
        source_index=source_index,
        coverage=coverage,
    )
    dependency_snapshot = build_dependency_snapshot(
        root,
        subject="video_decomposition_report",
        inputs=[{"role": str(row["role"]), "path": str(row["path"])} for row in source_artifacts],
        producer_schema=REPORT_SCHEMA,
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "report_id": f"decomposition-{dependency_snapshot['snapshot_sha256'][:16]}",
        "title": str(title or manifest.get("title") or root.name),
        "source_artifacts": source_artifacts,
        "modality_coverage": coverage,
        "findings": findings,
        "structure_segments": structure_segments,
        "creative_strategy": creative_strategy,
        "capability_gate": _capability_gate(coverage),
        "single_video_layers": _single_video_layers(findings, structure_segments),
        "dependency_snapshot": dependency_snapshot,
        "bundle_dir": str(root),
        "source_reuse": {
            "upstream": UPSTREAM_REFERENCE,
            "reused_vkp_modules": [
                "artifact_freshness.build_dependency_snapshot",
                "artifact_freshness.validate_dependency_snapshot",
                "canonical_json.canonical_json_sha256",
                "smart_summary_input_pack.select_canonical_transcript_path",
                "storage.bundle_write_lock/write_json/write_text_atomic",
            ],
            "rejected_upstream_modules": [
                "ASR routing and backend cache",
                "FFmpeg/ffprobe frame and audio extraction",
                "tool scanning and private virtualenv",
                "second state machine or index",
                "yt-dlp Chrome-cookie acquisition",
            ],
        },
        "operator_boundary": {
            "derived_projection_only": True,
            "timeline_mutated": False,
            "canonical_transcript_mutated": False,
            "raw_evidence_mutated": False,
            "run_registry_mutated": False,
            "media_decoded": False,
            "model_calls_made": 0,
            "network_calls_made": 0,
            "automatic_publish": False,
        },
        "artifacts": {
            "report_json": "exports/video-decomposition-report.json",
            "report_markdown": "exports/video-decomposition-report.md",
            "status_json": "exports/video-decomposition-report-status.json",
            "status_markdown": "exports/video-decomposition-report-status.md",
        },
        "generated_at": now_iso(),
    }
    report["report_sha256"] = _payload_sha256(report, "report_sha256")
    validate_video_decomposition_report(report, check_source_artifacts=True)
    if write:
        _write_report(root, report, manifest)
    return report


def validate_video_decomposition_report(
    report: dict[str, Any],
    *,
    check_source_artifacts: bool = True,
) -> None:
    if report.get("schema") != REPORT_SCHEMA:
        raise VideoDecompositionContractError("unsupported video decomposition report schema")
    if report.get("report_sha256") != _payload_sha256(report, "report_sha256"):
        raise VideoDecompositionContractError("video decomposition report integrity check failed")
    if not str(report.get("report_id") or "").strip():
        raise VideoDecompositionContractError("video decomposition report id is required")
    if not str(report.get("title") or "").strip():
        raise VideoDecompositionContractError("video decomposition report title is required")
    source_ids = _validate_source_artifacts(report.get("source_artifacts"), check_files=check_source_artifacts)
    coverage = _validate_modality_coverage(report.get("modality_coverage"), source_ids)
    findings = _validate_findings(report.get("findings"), coverage, source_ids)
    _validate_inference_precedence(findings)
    _validate_structure_segments(report.get("structure_segments"), findings)
    _validate_creative_strategy(report.get("creative_strategy"), findings)
    snapshot = report.get("dependency_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("producer_schema") != REPORT_SCHEMA:
        raise VideoDecompositionContractError("video decomposition dependency snapshot is missing or incompatible")


def video_decomposition_report_status(
    bundle_dir: str | Path,
    *,
    report_path: str | Path | None = None,
    write: bool = False,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    path = _resolve_report_path(root, report_path)
    if not path.is_file():
        result = _status_result(root, path, status="missing", issues=[{"key": "report_missing"}])
    else:
        try:
            report = _read_object(path)
            validate_video_decomposition_report(report, check_source_artifacts=False)
        except (OSError, ValueError, VideoDecompositionContractError) as exc:
            result = _status_result(root, path, status="invalid", issues=[{"key": "report_invalid", "detail": str(exc)}])
        else:
            freshness = validate_dependency_snapshot(root, report["dependency_snapshot"])
            result = _status_result(
                root,
                path,
                status=str(freshness.get("status") or "invalid"),
                report=report,
                freshness=freshness,
                issues=list(freshness.get("issues") or []),
            )
    if write:
        exports = root / "exports"
        write_json(exports / "video-decomposition-report-status.json", result)
        write_text_atomic(exports / "video-decomposition-report-status.md", _render_status_markdown(result))
    return result


def compare_video_decomposition_reports(
    report_paths: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
    title: str = "",
    write: bool = True,
) -> dict[str, Any]:
    paths = [Path(value).expanduser().resolve() for value in report_paths]
    if len(paths) < 2:
        raise ValueError("video decomposition comparison requires at least two reports")
    reports: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        report = _read_object(path)
        validate_video_decomposition_report(report, check_source_artifacts=True)
        reports.append(report)
        source_reports.append({
            "artifact_id": f"report-{index:03d}",
            "role": "video_decomposition_report",
            **_artifact_reference(path),
            "report_id": report["report_id"],
            "report_sha256": report["report_sha256"],
        })
    layout = "cards_and_matrix" if len(reports) >= 5 else "wide_table"
    comparison: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "comparison_id": "comparison-" + canonical_json_sha256([row["report_sha256"] for row in reports])[:16],
        "title": str(title or "视频拆解同尺度比较"),
        "layout": layout,
        "source_reports": source_reports,
        "cards": [_comparison_card(report) for report in reports],
        "uniform_matrix": _uniform_comparison_matrix(reports),
        "operator_boundary": {
            "derived_projection_only": True,
            "source_reports_mutated": False,
            "timeline_mutated": False,
            "network_calls_made": 0,
            "automatic_publish": False,
        },
        "generated_at": now_iso(),
    }
    comparison["comparison_sha256"] = _payload_sha256(comparison, "comparison_sha256")
    if write:
        if output_dir is None:
            raise ValueError("output_dir is required when write=True")
        target = Path(output_dir).expanduser().resolve()
        write_json(target / "video-decomposition-comparison.json", comparison)
        write_text_atomic(target / "video-decomposition-comparison.md", _render_comparison_markdown(comparison))
    return comparison


def _discover_sources(
    root: Path,
    *,
    manifest: dict[str, Any],
    timeline_path: Path | None,
    timeline: list[dict[str, Any]],
    transcript_path: Path | None,
    summary_path: Path | None,
) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []

    def add(role: str, value: str | Path | None) -> None:
        if value is None:
            return
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file() or not path.is_relative_to(root):
            return
        if any(existing == path for _, existing in sources):
            return
        sources.append((role, path))

    add("timeline", timeline_path)
    add("canonical_transcript", transcript_path)
    add("smart_summary", summary_path)
    add("platform_metadata", manifest.get("page_metadata_json"))
    add("platform_metadata", root / "source" / "page-metadata.json")
    add("companion_courseware", manifest.get("companion_courseware_text"))
    add("companion_courseware", manifest.get("companion_courseware_text_markdown"))
    add("companion_courseware", root / "exports" / "companion-courseware-text.md")
    for key in ("link_acquisition_result", "acquisition_result_json", "source_acquisition_json"):
        add("link_acquisition", manifest.get(key))
    for candidate in (
        root / "source" / "link-acquisition-result.json",
        root / "link-acquisition-result.json",
        root / "exports" / "speech-execution-receipt.json",
        root / "exports" / "audio-evidence.json",
    ):
        role = "link_acquisition" if "acquisition" in candidate.name else "audio_evidence"
        add(role, candidate)
    media_execution = root / "exports" / "media-execution"
    if media_execution.is_dir():
        for candidate in sorted(media_execution.glob("*audio*receipt.json")):
            add("audio_evidence", candidate)
        for candidate in sorted(media_execution.glob("*speech*receipt.json")):
            add("audio_evidence", candidate)
    for frame in _timeline_frame_paths(root, timeline):
        add("visual_frame", frame)
    return sources


def _source_artifact_references(sources: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for role, path in sources:
        counts[role] = counts.get(role, 0) + 1
        rows.append({
            "artifact_id": f"{_slug(role)}-{counts[role]:03d}",
            "role": role,
            **_artifact_reference(path),
        })
    return rows


def _source_index(source_artifacts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in source_artifacts:
        result.setdefault(str(row["role"]), []).append(row)
    return result


def _modality_coverage(
    timeline: list[dict[str, Any]],
    source_index: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    timeline_ids = _source_ids(source_index, "timeline")
    visual_ids = _source_ids(source_index, "visual_frame")
    has_visual = bool(visual_ids or any(_row_has_visual(row) for row in timeline))
    transcript_ids = _source_ids(source_index, "canonical_transcript")
    if not transcript_ids and any(_row_transcript(row) for row in timeline):
        transcript_ids = timeline_ids
    platform_ids = _source_ids(source_index, "platform_metadata")
    user_ids = _source_ids(source_index, "companion_courseware")
    audio_ids = _source_ids(source_index, "audio_evidence")
    link_ids = _source_ids(source_index, "link_acquisition")
    return {
        "video": _coverage_row(
            [*timeline_ids, *visual_ids] if has_visual else [],
            "no derived frame, OCR, visual, or temporal evidence is present in the Bundle",
        ),
        "audio": _coverage_row(
            audio_ids,
            "no audio-analysis or speech execution receipt is present in the Bundle",
        ),
        "transcript": _coverage_row(
            transcript_ids,
            "no canonical transcript or timestamped Timeline transcript is present",
        ),
        "platform_metadata": _coverage_row(platform_ids, "no imported platform/page metadata is present"),
        "user_material": _coverage_row(user_ids, "no user-supplied companion material is present"),
        "link_acquisition": _coverage_row(link_ids, "no explicit link acquisition result is present"),
    }


def _coverage_row(evidence_ids: list[str], missing: str) -> dict[str, Any]:
    unique = _unique(evidence_ids)
    return {
        "status": "available" if unique else "unavailable",
        "evidence_ids": unique,
        "missing_evidence": [] if unique else [missing],
    }


def _build_findings_and_structure(
    *,
    title: str,
    rows: list[dict[str, Any]],
    summary_path: Path | None,
    source_index: dict[str, list[dict[str, Any]]],
    coverage: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    findings: list[dict[str, Any]] = []
    structure: list[dict[str, Any]] = []
    timeline_ids = _source_ids(source_index, "timeline")
    transcript_ids = _source_ids(source_index, "canonical_transcript")
    transcript_evidence = _unique([*transcript_ids, *timeline_ids])
    video_evidence = _unique([*timeline_ids, *_source_ids(source_index, "visual_frame")])
    user_evidence = _source_ids(source_index, "companion_courseware")
    content_ids = transcript_evidence or user_evidence
    content_modality = "transcript" if transcript_evidence else "user_material"
    summary_ids = _source_ids(source_index, "smart_summary")

    selected = _select_nonoverlapping_rows(rows, limit=5)
    positioning_excerpt = _summary_section_excerpt(
        summary_path,
        ("## 核心主题 / 课程主线", "## 核心主题", "## 一句话概览"),
    )
    if content_ids:
        positioning_claim = (
            f"现有 Smart Summary 对《{title}》的内容定位：{positioning_excerpt}"
            if positioning_excerpt
            else f"Bundle 中已有《{title}》的文本证据，但没有独立确认的内容定位摘要。"
        )
        findings.append(_finding(
            "positioning-confirmed", "positioning", "confirmed", positioning_claim,
            [content_modality], _unique([*summary_ids, *content_ids])[:3],
        ))
    else:
        findings.append(_unavailable_finding(
            "positioning-unavailable", "positioning", "当前证据无法确认内容定位。",
            ["transcript", "user_material"], ["canonical transcript or user-supplied material"],
        ))

    hook_id = ""
    findings.append(_unavailable_finding(
        "hook-unavailable", "hook",
        "当前证据只有开头时间顺序，不能证明该片段承担钩子作用。",
        ["video", "transcript"],
        ["explicit hook analysis or human confirmation"],
    ))

    payoff_id = ""
    findings.append(_unavailable_finding(
        "payoff-unavailable", "payoff",
        "当前证据只有结尾时间顺序，不能证明该片段兑现开头或承担收束作用。",
        ["video", "transcript"],
        ["explicit payoff/CTA analysis or human confirmation"],
    ))

    structure_id = ""
    if selected and content_ids:
        structure_id = "body-structure-confirmed"
        findings.append(_finding(
            structure_id, "body_structure", "confirmed",
            f"现有时间证据形成 {len(selected)} 个可回看的覆盖锚点；它们不等同于已确认的语义章节。",
            [content_modality], content_ids[:2],
        ))

    content_value_id = ""
    summary_excerpt = _summary_section_excerpt(
        summary_path,
        ("## 一句话概览", "## 关键观点 / 方法论", "## 关键观点"),
    )
    if summary_excerpt and content_ids:
        content_value_id = "content-value-confirmed"
        findings.append(_finding(
            content_value_id, "content_value", "confirmed",
            "现有 Smart Summary 提炼：" + summary_excerpt,
            [content_modality], _unique([*summary_ids, *content_ids])[:3],
        ))

    visual_id = ""
    visual_row = next((row for row in selected if _row_has_visual(row)), None)
    if visual_row is not None and coverage["video"]["status"] == "available":
        visual_id = "visual-language-confirmed"
        findings.append(_finding(
            visual_id, "visual_language", "confirmed",
            "既有视觉证据：" + _row_visual_excerpt(visual_row), ["video"], video_evidence[:3],
            time_range=_row_time_range(visual_row),
        ))
    else:
        findings.append(_unavailable_finding(
            "visual-language-unavailable", "visual_language",
            "当前 Bundle 没有足以描述视觉语言的已分析画面证据。",
            ["video"], ["derived visual or temporal evidence"],
        ))

    subtitle_id = ""
    subtitle_row = next((row for row in selected if _row_visual_text(row)), None)
    if subtitle_row is not None and coverage["video"]["status"] == "available":
        subtitle_id = "subtitle-packaging-confirmed"
        findings.append(_finding(
            subtitle_id, "subtitle_packaging", "confirmed",
            "屏幕文字直接证据：" + _clip(_row_visual_text(subtitle_row), 180),
            ["video"], video_evidence[:3], time_range=_row_time_range(subtitle_row),
        ))

    framework_id = ""
    if structure_id and hook_id and payoff_id:
        framework_id = "reusable-framework-confirmed"
        findings.append(_finding(
            framework_id, "reusable_framework", "confirmed",
            "可复用结构由开头入口、正文节点和结尾收束三部分组成。",
            [content_modality], content_ids[:2],
        ))

    risk_id = ""
    if transcript_evidence and video_evidence and coverage["video"]["status"] == "available":
        risk_evidence = _unique([transcript_evidence[0], video_evidence[-1]])
        if len(risk_evidence) >= 2:
            risk_id = "imitation-risk-inferred"
            findings.append(_finding(
                risk_id, "imitation_risk", "inferred",
                "复用时可能需要同时保持内容顺序与视觉呈现；该项仅为跨模态候选判断。",
                ["transcript", "video"], risk_evidence,
            ))

    for dimension, label, modalities, missing in (
        ("voice_delivery", "人声表达方式", ["audio"], "direct prosody or voice-delivery analysis"),
        ("bgm_style", "BGM 风格与作用", ["audio"], "readable audio analysis"),
        ("bgm_identity", "BGM 曲名与作者", ["audio", "platform_metadata"], "reliable music recognition or direct metadata"),
        ("author_identity", "作者身份", ["platform_metadata", "user_material"], "direct author metadata or user confirmation"),
        ("behind_the_scenes", "幕后信息", ["user_material"], "direct behind-the-scenes material"),
        ("performance_metrics", "播放与互动数据", ["platform_metadata", "user_material"], "direct platform metrics or user-supplied data"),
    ):
        findings.append(_unavailable_finding(
            f"{_slug(dimension)}-unavailable", dimension, f"当前证据无法确认{label}。",
            modalities, [missing],
        ))

    for index, row in enumerate(selected, start=1):
        modalities, evidence_ids = _row_evidence(row, coverage, transcript_evidence, video_evidence)
        excerpt = _row_excerpt(row)
        time_range = _row_time_range(row)
        if not modalities or not evidence_ids or not excerpt or time_range["scope"] != "local":
            continue
        finding_id = f"structure-segment-{index:03d}"
        findings.append(_finding(
            finding_id, "body_structure", "confirmed", excerpt,
            modalities, evidence_ids, time_range=time_range,
        ))
        structure.append({
            "segment_id": f"segment-{index:03d}",
            "label": f"时间证据锚点 {index}",
            "start_s": time_range["start_s"],
            "end_s": time_range["end_s"],
            "summary": excerpt,
            "status": "confirmed",
            "finding_ids": [finding_id],
            "confirmed_requirements": {"scene": "", "action": "", "shot_size": "", "avoid": []},
        })

    creative = {
        "reusable_frameworks": [framework_id] if framework_id else [],
        "techniques": [value for value in (visual_id, subtitle_id) if value],
        "non_copyable_factors": [],
        "imitation_risks": [risk_id] if risk_id else [],
        "differentiation": [content_value_id] if content_value_id else [],
    }
    return findings, structure, creative


def _finding(
    finding_id: str,
    dimension: str,
    status: str,
    claim: str,
    source_modalities: list[str],
    evidence_ids: list[str],
    *,
    time_range: dict[str, Any] | None = None,
    direct_evidence_kind: str = "",
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "dimension": dimension,
        "status": status,
        "claim": _clip(claim, 280),
        "time_range": time_range or {"scope": "global", "start_s": None, "end_s": None},
        "source_modalities": _unique(source_modalities),
        "evidence_ids": _unique(evidence_ids),
        "evidence_paths": [],
        "missing_evidence": [],
        "direct_evidence_kind": direct_evidence_kind,
    }


def _unavailable_finding(
    finding_id: str,
    dimension: str,
    claim: str,
    source_modalities: list[str],
    missing_evidence: list[str],
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "dimension": dimension,
        "status": "unavailable",
        "claim": claim,
        "time_range": {"scope": "global", "start_s": None, "end_s": None},
        "source_modalities": _unique(source_modalities),
        "evidence_ids": [],
        "evidence_paths": [],
        "missing_evidence": _unique(missing_evidence),
        "direct_evidence_kind": "",
    }


def _single_video_layers(
    findings: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_dimension: dict[str, list[str]] = {}
    for row in findings:
        by_dimension.setdefault(str(row["dimension"]), []).append(str(row["finding_id"]))
    return [
        {"layer": 1, "key": "positioning_and_value", "finding_ids": _ids_for(by_dimension, "positioning", "content_value")},
        {
            "layer": 2,
            "key": "hook_structure_payoff",
            "finding_ids": _ids_for(by_dimension, "hook", "body_structure", "payoff"),
            "segment_ids": [str(row["segment_id"]) for row in segments],
        },
        {
            "layer": 3,
            "key": "audiovisual_expression",
            "finding_ids": _ids_for(
                by_dimension, "visual_language", "subtitle_packaging", "voice_delivery",
                "bgm_style", "bgm_identity", "audiovisual_coordination",
            ),
        },
        {
            "layer": 4,
            "key": "creative_strategy",
            "finding_ids": _ids_for(
                by_dimension, "reusable_framework", "technique", "non_copyable_factor",
                "imitation_risk", "differentiation",
            ),
        },
        {
            "layer": 5,
            "key": "verification_boundaries",
            "finding_ids": [
                str(row["finding_id"])
                for row in findings
                if row["status"] == "unavailable" or row["dimension"] in _SENSITIVE_DIRECT_EVIDENCE
            ],
        },
    ]


def _capability_gate(coverage: dict[str, dict[str, Any]]) -> dict[str, Any]:
    matrix = {
        "video": {
            "allows": ["visual_language", "subtitle_packaging", "audiovisual_coordination"],
            "forbids_without_direct_evidence": ["bgm_identity", "author_identity", "behind_the_scenes", "performance_metrics"],
        },
        "audio": {
            "allows": ["voice_delivery", "bgm_style"],
            "forbids_without_direct_evidence": ["visual_language", "subtitle_packaging", "bgm_identity"],
        },
        "transcript": {
            "allows": ["positioning", "hook", "body_structure", "payoff", "content_value", "expression"],
            "forbids_without_direct_evidence": ["visual_language", "voice_delivery", "bgm_style", "bgm_identity"],
        },
        "platform_metadata": {
            "allows": ["author_identity", "performance_metrics"],
            "forbids_without_direct_evidence": ["behind_the_scenes"],
        },
        "user_material": {
            "allows": ["positioning", "content_value", "author_identity", "behind_the_scenes", "performance_metrics"],
            "forbids_without_direct_evidence": ["voice_delivery", "bgm_identity"],
        },
        "link_acquisition": {
            "allows": ["acquisition provenance only"],
            "forbids_without_direct_evidence": ["all audiovisual conclusions until media evidence exists"],
        },
    }
    return {
        "policy": "only already-extracted Bundle evidence may support findings",
        "modalities": {
            modality: {
                **matrix[modality],
                "status": coverage[modality]["status"],
                "evidence_ids": list(coverage[modality]["evidence_ids"]),
            }
            for modality in MODALITIES
        },
    }


def _write_report(root: Path, report: dict[str, Any], manifest: dict[str, Any]) -> None:
    exports = root / "exports"
    with bundle_write_lock(root, operation="video_decomposition_report", timeout_seconds=5.0):
        write_json(exports / "video-decomposition-report.json", report)
        write_text_atomic(exports / "video-decomposition-report.md", _render_report_markdown(report))
        write_json(root / "mcp-video-decomposition-report.args.json", {"bundle_dir": str(root), "write": True})
        current = _read_object(root / "manifest.json") or dict(manifest)
        current.update({
            "video_decomposition_report_json": "exports/video-decomposition-report.json",
            "video_decomposition_report_markdown": "exports/video-decomposition-report.md",
            "video_decomposition_report_status_json": "exports/video-decomposition-report-status.json",
            "video_decomposition_report_status_markdown": "exports/video-decomposition-report-status.md",
            "mcp_video_decomposition_report_args": "mcp-video-decomposition-report.args.json",
            "video_decomposition_report_sha256": report["report_sha256"],
            "video_decomposition_report_updated_at": report["generated_at"],
        })
        write_json(root / "manifest.json", current)
        status = video_decomposition_report_status(root, write=False)
        write_json(exports / "video-decomposition-report-status.json", status)
        write_text_atomic(exports / "video-decomposition-report-status.md", _render_status_markdown(status))


def _validate_source_artifacts(value: Any, *, check_files: bool) -> set[str]:
    if not isinstance(value, list) or not value:
        raise VideoDecompositionContractError("video decomposition report requires source artifacts")
    ids: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise VideoDecompositionContractError("source artifact must be an object")
        artifact_id = str(raw.get("artifact_id") or "").strip()
        if not artifact_id or artifact_id in ids:
            raise VideoDecompositionContractError("source artifact ids must be non-empty and unique")
        ids.add(artifact_id)
        if not str(raw.get("role") or "").strip():
            raise VideoDecompositionContractError("source artifact role is required")
        for key in ("path", "sha256", "content_kind"):
            if raw.get(key) is None or str(raw.get(key)) == "":
                raise VideoDecompositionContractError(f"source artifact {artifact_id} lacks {key}")
        if str(raw.get("content_kind")) not in {"binary", "json"}:
            raise VideoDecompositionContractError(f"source artifact {artifact_id} content kind is invalid")
        if check_files:
            current = _artifact_reference(Path(str(raw["path"])))
            for key in ("bytes", "sha256", "content_kind", "canonical_sha256"):
                if raw.get(key) != current.get(key):
                    raise VideoDecompositionContractError(f"source artifact {artifact_id} changed after report creation")
    return ids


def _validate_modality_coverage(value: Any, source_ids: set[str]) -> dict[str, dict[str, Any]]:
    coverage = value if isinstance(value, dict) else {}
    if set(coverage) != set(MODALITIES):
        raise VideoDecompositionContractError("modality coverage must declare every supported modality")
    result: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        row = coverage.get(modality)
        if not isinstance(row, dict):
            raise VideoDecompositionContractError(f"modality coverage {modality} must be an object")
        status = str(row.get("status") or "")
        evidence = _validate_string_list(row.get("evidence_ids"), f"{modality} evidence ids")
        missing = _validate_string_list(row.get("missing_evidence"), f"{modality} missing evidence")
        if any(item not in source_ids for item in evidence):
            raise VideoDecompositionContractError(f"modality {modality} cites unknown evidence")
        if status == "available" and not evidence:
            raise VideoDecompositionContractError(f"available modality {modality} requires evidence ids")
        if status == "unavailable" and (evidence or not missing):
            raise VideoDecompositionContractError(f"unavailable modality {modality} must expose only missing evidence")
        if status not in {"available", "unavailable"}:
            raise VideoDecompositionContractError(f"unsupported modality status for {modality}: {status}")
        result[modality] = row
    return result


def _validate_findings(
    value: Any,
    coverage: dict[str, dict[str, Any]],
    source_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise VideoDecompositionContractError("video decomposition report requires findings")
    findings: dict[str, dict[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, dict):
            raise VideoDecompositionContractError("decomposition finding must be an object")
        finding_id = str(raw.get("finding_id") or "").strip()
        if not finding_id or finding_id in findings:
            raise VideoDecompositionContractError("finding ids must be non-empty and unique")
        dimension = str(raw.get("dimension") or "")
        status = str(raw.get("status") or "")
        if dimension not in FINDING_DIMENSIONS:
            raise VideoDecompositionContractError(f"unsupported decomposition finding dimension: {dimension}")
        if status not in FINDING_STATUSES:
            raise VideoDecompositionContractError(f"unsupported finding status: {status}")
        if not str(raw.get("claim") or "").strip():
            raise VideoDecompositionContractError(f"finding {finding_id} claim is required")
        modalities = _validate_string_list(raw.get("source_modalities"), f"{finding_id} modalities")
        evidence_ids = _validate_string_list(raw.get("evidence_ids"), f"{finding_id} evidence ids")
        evidence_paths = _validate_string_list(raw.get("evidence_paths"), f"{finding_id} evidence paths")
        missing = _validate_string_list(raw.get("missing_evidence"), f"{finding_id} missing evidence")
        _validate_time_range(raw.get("time_range"), finding_id)
        if any(modality not in MODALITIES for modality in modalities):
            raise VideoDecompositionContractError(f"finding {finding_id} uses an unsupported modality")
        if any(evidence_id not in source_ids for evidence_id in evidence_ids):
            raise VideoDecompositionContractError(f"finding {finding_id} cites unknown evidence")
        if status == "unavailable":
            if evidence_ids or evidence_paths or not missing:
                raise VideoDecompositionContractError(f"unavailable finding {finding_id} must expose only missing evidence")
        else:
            if not modalities or not (evidence_ids or evidence_paths):
                raise VideoDecompositionContractError(f"finding {finding_id} requires source modalities and evidence")
            if any(coverage[modality]["status"] != "available" for modality in modalities):
                raise VideoDecompositionContractError(f"finding {finding_id} cites unavailable modalities")
            if status == "inferred" and len(set(evidence_ids + evidence_paths)) < 2:
                raise VideoDecompositionContractError(f"inferred finding {finding_id} requires multiple evidence items")
            _validate_dimension_modalities(finding_id, dimension, modalities)
        direct_allowed = _SENSITIVE_DIRECT_EVIDENCE.get(dimension)
        if direct_allowed:
            direct_kind = str(raw.get("direct_evidence_kind") or "")
            if status == "inferred":
                raise VideoDecompositionContractError(f"sensitive finding {finding_id} cannot be inferred")
            if status == "confirmed" and direct_kind not in direct_allowed:
                raise VideoDecompositionContractError(f"finding {finding_id} lacks accepted direct evidence")
        findings[finding_id] = raw
    return findings


def _validate_inference_precedence(findings: dict[str, dict[str, Any]]) -> None:
    confirmed_keys = {
        (str(row["dimension"]), canonical_json_sha256(row["time_range"]))
        for row in findings.values()
        if row["status"] == "confirmed"
    }
    for row in findings.values():
        key = (str(row["dimension"]), canonical_json_sha256(row["time_range"]))
        if row["status"] == "inferred" and key in confirmed_keys:
            raise VideoDecompositionContractError(
                f"inferred finding {row['finding_id']} cannot override a confirmed finding"
            )


def _validate_structure_segments(value: Any, findings: dict[str, dict[str, Any]]) -> None:
    if not isinstance(value, list):
        raise VideoDecompositionContractError("structure segments must be a list")
    seen: set[str] = set()
    previous_end = -1.0
    for raw in value:
        if not isinstance(raw, dict):
            raise VideoDecompositionContractError("structure segment must be an object")
        segment_id = str(raw.get("segment_id") or "").strip()
        if not segment_id or segment_id in seen:
            raise VideoDecompositionContractError("structure segment ids must be non-empty and unique")
        seen.add(segment_id)
        status = str(raw.get("status") or "")
        if status not in FINDING_STATUSES:
            raise VideoDecompositionContractError(f"unsupported structure segment status: {status}")
        if not str(raw.get("label") or "").strip() or not str(raw.get("summary") or "").strip():
            raise VideoDecompositionContractError(f"structure segment {segment_id} requires label and summary")
        start = _finite_seconds(raw.get("start_s"), f"{segment_id}.start_s")
        end = _finite_seconds(raw.get("end_s"), f"{segment_id}.end_s")
        if end <= start or start < previous_end:
            raise VideoDecompositionContractError("structure segments must be ordered, non-overlapping intervals")
        previous_end = end
        finding_ids = _validate_string_list(raw.get("finding_ids"), f"{segment_id} finding ids")
        if not finding_ids or any(item not in findings for item in finding_ids):
            raise VideoDecompositionContractError(f"structure segment {segment_id} has invalid finding ids")
        linked = {findings[item]["status"] for item in finding_ids}
        if status == "confirmed" and linked != {"confirmed"}:
            raise VideoDecompositionContractError(f"confirmed segment {segment_id} requires confirmed findings")
        if status == "inferred" and "unavailable" in linked:
            raise VideoDecompositionContractError(f"inferred segment {segment_id} cannot cite unavailable findings")
        requirements = raw.get("confirmed_requirements")
        if not isinstance(requirements, dict) or set(requirements) != {"scene", "action", "shot_size", "avoid"}:
            raise VideoDecompositionContractError(f"structure segment {segment_id} confirmed requirements are invalid")
        avoid = _validate_string_list(requirements.get("avoid"), f"{segment_id} avoid")
        if status != "confirmed" and any(
            str(requirements.get(key) or "").strip() for key in ("scene", "action", "shot_size")
        ):
            raise VideoDecompositionContractError(f"non-confirmed segment {segment_id} cannot contain instructions")
        if status != "confirmed" and avoid:
            raise VideoDecompositionContractError(f"non-confirmed segment {segment_id} cannot contain avoid instructions")


def _validate_creative_strategy(value: Any, findings: dict[str, dict[str, Any]]) -> None:
    strategy = value if isinstance(value, dict) else {}
    if set(strategy) != set(CREATIVE_CATEGORIES):
        raise VideoDecompositionContractError("creative strategy must declare every supported category")
    for category in CREATIVE_CATEGORIES:
        ids = _validate_string_list(strategy.get(category), f"creative strategy {category}")
        for finding_id in ids:
            if finding_id not in findings:
                raise VideoDecompositionContractError(f"creative strategy cites unknown finding {finding_id}")
            if findings[finding_id]["status"] == "unavailable":
                raise VideoDecompositionContractError(f"creative strategy cannot adopt unavailable finding {finding_id}")


def _validate_dimension_modalities(finding_id: str, dimension: str, modalities: list[str]) -> None:
    allowed = _DIMENSION_MODALITY_RULES.get(dimension)
    if not allowed:
        if not set(modalities).intersection({"video", "audio", "transcript", "user_material"}):
            raise VideoDecompositionContractError(f"finding {finding_id} lacks a content evidence modality")
        return
    if dimension == "audiovisual_coordination":
        if not allowed.issubset(set(modalities)):
            raise VideoDecompositionContractError(f"finding {finding_id} requires video and audio evidence")
        return
    if not set(modalities).intersection(allowed):
        raise VideoDecompositionContractError(f"finding {finding_id} uses the wrong evidence modality")


def _validate_time_range(value: Any, finding_id: str) -> None:
    row = value if isinstance(value, dict) else {}
    scope = str(row.get("scope") or "")
    if scope == "global":
        if row.get("start_s") is not None or row.get("end_s") is not None:
            raise VideoDecompositionContractError(f"global finding {finding_id} cannot claim a local interval")
        return
    if scope != "local":
        raise VideoDecompositionContractError(f"finding {finding_id} time range scope is invalid")
    start = _finite_seconds(row.get("start_s"), f"{finding_id}.start_s")
    end = _finite_seconds(row.get("end_s"), f"{finding_id}.end_s")
    if end <= start:
        raise VideoDecompositionContractError(f"finding {finding_id} time range must have positive duration")


def _artifact_reference(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"decomposition source artifact not found: {resolved}")
    evidence = artifact_evidence(resolved)
    reference: dict[str, Any] = {
        "path": str(resolved),
        "bytes": int(evidence["bytes"]),
        "sha256": str(evidence["sha256"]),
        "content_kind": "binary",
        "canonical_sha256": None,
    }
    if resolved.suffix.lower() == ".json":
        try:
            value = json.loads(resolved.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return reference
        reference["content_kind"] = "json"
        reference["canonical_sha256"] = canonical_json_sha256(value)
    return reference


def _resolve_report_path(root: Path, value: str | Path | None) -> Path:
    if value:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    manifest = _read_object(root / "manifest.json")
    raw = str(manifest.get("video_decomposition_report_json") or "").strip()
    return ((root / raw) if raw else (root / "exports" / "video-decomposition-report.json")).resolve()


def _status_result(
    root: Path,
    path: Path,
    *,
    status: str,
    report: dict[str, Any] | None = None,
    freshness: dict[str, Any] | None = None,
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": STATUS_SCHEMA,
        "status": status,
        "passed": status == "fresh",
        "bundle_dir": str(root),
        "report_path": str(path),
        "report_id": str((report or {}).get("report_id") or ""),
        "report_sha256": str((report or {}).get("report_sha256") or ""),
        "freshness": freshness or {},
        "issues": issues or [],
        "checked_at": now_iso(),
    }


def _comparison_card(report: dict[str, Any]) -> dict[str, Any]:
    counts = {
        status: sum(row.get("status") == status for row in report["findings"])
        for status in sorted(FINDING_STATUSES)
    }
    return {
        "report_id": report["report_id"],
        "title": report["title"],
        "report_sha256": report["report_sha256"],
        "status_counts": counts,
        "available_modalities": [
            modality for modality in MODALITIES
            if report["modality_coverage"][modality]["status"] == "available"
        ],
        "structure_segment_count": len(report["structure_segments"]),
    }


def _uniform_comparison_matrix(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in sorted(FINDING_DIMENSIONS):
        cells = []
        for report in reports:
            candidates = [row for row in report["findings"] if row.get("dimension") == dimension]
            candidates.sort(key=lambda row: {"confirmed": 0, "inferred": 1, "unavailable": 2}[row["status"]])
            selected = candidates[0] if candidates else None
            cells.append({
                "report_id": report["report_id"],
                "status": str((selected or {}).get("status") or "unavailable"),
                "claim": str((selected or {}).get("claim") or "该报告未输出此维度。"),
                "time_range": (selected or {}).get(
                    "time_range", {"scope": "global", "start_s": None, "end_s": None},
                ),
            })
        rows.append({"dimension": dimension, "videos": cells})
    return rows


def _render_report_markdown(report: dict[str, Any]) -> str:
    findings = {str(row["finding_id"]): row for row in report["findings"]}
    lines = [
        f"# {report['title']} - 证据化视频拆解",
        "",
        f"- Contract: `{report['schema']}`",
        f"- Report SHA-256: `{report['report_sha256']}`",
        "- 性质：Bundle/Timeline/逐字稿/Smart Summary 的只读派生投影",
        "",
        "## 输入模态能力门",
        "",
    ]
    for modality in MODALITIES:
        row = report["modality_coverage"][modality]
        lines.append(f"- {modality}: **{row['status']}**")
        for missing in row["missing_evidence"]:
            lines.append(f"  - 缺失：{missing}")
    lines.extend(["", "## 单视频五层报告", ""])
    for layer in report["single_video_layers"]:
        lines.extend([f"### Layer {layer['layer']} · {layer['key']}", ""])
        for finding_id in layer.get("finding_ids") or []:
            row = findings.get(finding_id)
            if not row:
                continue
            lines.append(f"- **{row['status']}** `{row['dimension']}`：{row['claim']}")
            if row["missing_evidence"]:
                lines.append("  - 缺失：" + "；".join(row["missing_evidence"]))
        lines.append("")
    lines.extend(["## 结构片段", ""])
    if report["structure_segments"]:
        for row in report["structure_segments"]:
            lines.extend([
                f"### {row['label']} · {row['start_s']:.3f}s–{row['end_s']:.3f}s",
                "",
                f"- 状态：`{row['status']}`",
                f"- 摘要：{row['summary']}",
                f"- 证据 finding：{', '.join(row['finding_ids'])}",
                "",
            ])
    else:
        lines.extend(["- 当前没有可由时间戳证据确认的结构片段。", ""])
    lines.extend([
        "## 边界",
        "",
        "- inferred 不得覆盖 confirmed。",
        "- unavailable 不补写猜测。",
        "- BGM 曲名、作者身份、幕后信息和播放数据必须有直接证据。",
        "- 本报告不修改 Timeline、规范逐字稿、原始 evidence 或 run registry。",
        "",
    ])
    return "\n".join(lines)


def _render_status_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# Video Decomposition Report Status",
        "",
        f"- Status: `{status['status']}`",
        f"- Report: `{status['report_path']}`",
        f"- Report SHA-256: `{status['report_sha256']}`",
        "",
        "## Issues",
        "",
    ]
    issues = status.get("issues") or []
    if issues:
        for row in issues:
            detail = row.get("path") or row.get("detail") or ""
            lines.append(f"- `{row.get('key', 'unknown')}` {detail}".rstrip())
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def _render_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        f"# {comparison['title']}",
        "",
        f"- Layout: `{comparison['layout']}`",
        f"- Reports: {len(comparison['cards'])}",
        f"- Comparison SHA-256: `{comparison['comparison_sha256']}`",
        "",
    ]
    if comparison["layout"] == "cards_and_matrix":
        lines.extend(["## 视频卡片", ""])
        for card in comparison["cards"]:
            lines.extend([
                f"### {card['title']}",
                "",
                f"- Report: `{card['report_id']}`",
                f"- 可用模态：{', '.join(card['available_modalities']) or 'none'}",
                "- confirmed / inferred / unavailable: "
                f"{card['status_counts']['confirmed']} / "
                f"{card['status_counts']['inferred']} / "
                f"{card['status_counts']['unavailable']}",
                "",
            ])
        lines.extend(["## 统一尺度矩阵", ""])
        for row in comparison["uniform_matrix"]:
            lines.extend([f"### {row['dimension']}", ""])
            for cell in row["videos"]:
                lines.append(f"- `{cell['report_id']}` · **{cell['status']}**：{cell['claim']}")
            lines.append("")
    else:
        headers = ["维度", *[card["title"] for card in comparison["cards"]]]
        lines.extend([
            "## 同尺度表",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ])
        for row in comparison["uniform_matrix"]:
            cells = [f"{cell['status']}: {_clip(cell['claim'], 80)}" for cell in row["videos"]]
            lines.append("| " + " | ".join([row["dimension"], *cells]) + " |")
        lines.append("")
    return "\n".join(lines)


def _canonical_summary_path(root: Path) -> Path | None:
    for value in (*CODEX_FILENAMES, "exports/smart-summary.md"):
        path = (root / value).resolve()
        if path.is_file():
            return path
    return None


def _transcript_segments(path: Path | None) -> list[dict[str, Any]]:
    if path is None or path.suffix.lower() != ".json":
        return []
    value = _read_json_value(path)
    rows = value.get("segments") or value.get("items") or [] if isinstance(value, dict) else value
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _timeline_frame_paths(root: Path, timeline: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for row in timeline:
        for key in _FRAME_KEYS:
            raw = row.get(key)
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    continue
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = root / path
                resolved = path.resolve()
                if resolved.is_file() and resolved.is_relative_to(root) and resolved not in paths:
                    paths.append(resolved)
        for key in (
            "visual_understanding", "temporal_visual_understanding",
            "corrected_visual_understanding", "corrected_temporal_visual_understanding",
        ):
            nested = row.get(key)
            if not isinstance(nested, dict):
                continue
            for value in nested.get("evidence_frame_paths") or []:
                path = Path(str(value)).expanduser()
                if not path.is_absolute():
                    path = root / path
                resolved = path.resolve()
                if resolved.is_file() and resolved.is_relative_to(root) and resolved not in paths:
                    paths.append(resolved)
    return paths


def _select_nonoverlapping_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if _row_end(row) > _row_start(row) >= 0 and (_row_transcript(row) or _row_has_visual(row))
    ]
    candidates.sort(key=lambda row: (_row_start(row), _row_end(row)))
    nonoverlapping: list[dict[str, Any]] = []
    previous_end = -1.0
    for row in candidates:
        if _row_start(row) < previous_end:
            continue
        nonoverlapping.append(row)
        previous_end = _row_end(row)
    if len(nonoverlapping) <= limit:
        return nonoverlapping
    indexes = sorted({
        round(position * (len(nonoverlapping) - 1) / (limit - 1))
        for position in range(limit)
    })
    return [nonoverlapping[index] for index in indexes]


def _row_evidence(
    row: dict[str, Any],
    coverage: dict[str, dict[str, Any]],
    transcript_ids: list[str],
    video_ids: list[str],
) -> tuple[list[str], list[str]]:
    modalities: list[str] = []
    evidence: list[str] = []
    if _row_transcript(row) and coverage["transcript"]["status"] == "available":
        modalities.append("transcript")
        evidence.extend(transcript_ids[:2])
    if _row_has_visual(row) and coverage["video"]["status"] == "available":
        modalities.append("video")
        evidence.extend(video_ids[:2])
    return _unique(modalities), _unique(evidence)


def _row_time_range(row: dict[str, Any]) -> dict[str, Any]:
    start, end = _row_start(row), _row_end(row)
    if end > start >= 0:
        return {"scope": "local", "start_s": round(start, 3), "end_s": round(end, 3)}
    return {"scope": "global", "start_s": None, "end_s": None}


def _row_start(row: dict[str, Any]) -> float:
    return _seconds(row.get("start_s", row.get("start")))


def _row_end(row: dict[str, Any]) -> float:
    return _seconds(row.get("end_s", row.get("end")))


def _row_excerpt(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    return _clip(_row_transcript(row) or _row_visual_excerpt(row), 220)


def _row_transcript(row: dict[str, Any]) -> str:
    for key in ("corrected_transcript", "transcript", "corrected_text", "text", "raw_text"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _row_visual_text(row: dict[str, Any]) -> str:
    for key in ("corrected_visual_text", "visual_text", "ocr_text"):
        value = _flatten_text(row.get(key))
        if value:
            return value
    return ""


def _row_visual_excerpt(row: dict[str, Any]) -> str:
    values = [_flatten_text(row.get(key)) for key in _VISUAL_KEYS]
    return _clip("；".join(_unique(value for value in values if value)), 220)


def _row_has_visual(row: dict[str, Any]) -> bool:
    return any(_flatten_text(row.get(key)) for key in _VISUAL_KEYS) or any(row.get(key) for key in _FRAME_KEYS)


def _flatten_text(value: Any, *, limit: int = 8) -> str:
    values: list[str] = []

    def visit(item: Any) -> None:
        if len(values) >= limit or item is None or item is False:
            return
        if isinstance(item, str):
            text = item.strip()
            if text and text not in values:
                values.append(text)
        elif isinstance(item, (int, float)):
            values.append(str(item))
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for key, child in item.items():
                if key in {"schema", "confidence", "status", "evidence_frame_paths"}:
                    continue
                visit(child)

    visit(value)
    return "；".join(values)


def _artifact_text_excerpt(source_index: dict[str, list[dict[str, Any]]], role: str) -> str:
    rows = source_index.get(role) or []
    return _read_text_excerpt(Path(str(rows[0]["path"]))) if rows else ""


def _read_text_excerpt(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        if path.suffix.lower() == ".json":
            value = _read_json_value(path)
            text = _flatten_text(value, limit=20)
        else:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return _clip(text, 220)


def _summary_section_excerpt(path: Path | None, headings: Iterable[str]) -> str:
    """Reuse Smart Summary's section parser and return one reader-facing line.

    Intent: prevent report positioning/value from ingesting generation metadata.
    Decision: delegate section boundaries to ``smart_summary_codex._section_text``
    and keep only minimal Markdown cleanup here. Effective scope: derived report
    wording only; the Smart Summary and its evidence remain unchanged.
    """

    if path is None or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""
    for heading in headings:
        excerpt = _markdown_section_excerpt(_section_text(text, heading))
        if excerpt:
            return excerpt
    return ""


def _markdown_section_excerpt(value: str) -> str:
    for raw_line in str(value or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line).replace("`", "")
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            return _clip(line, 220)
    return ""


def _source_ids(source_index: dict[str, list[dict[str, Any]]], role: str) -> list[str]:
    return [str(row["artifact_id"]) for row in source_index.get(role) or []]


def _ids_for(by_dimension: dict[str, list[str]], *dimensions: str) -> list[str]:
    return [finding_id for dimension in dimensions for finding_id in by_dimension.get(dimension) or []]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _read_json_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _payload_sha256(payload: dict[str, Any], hash_field: str) -> str:
    value = dict(payload)
    value.pop(hash_field, None)
    return canonical_json_sha256(value)


def _validate_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise VideoDecompositionContractError(f"{label} must be a list")
    result: list[str] = []
    for raw in value:
        item = str(raw or "").strip()
        if not item:
            raise VideoDecompositionContractError(f"{label} cannot contain empty values")
        if item in result:
            raise VideoDecompositionContractError(f"{label} cannot contain duplicates")
        result.append(item)
    return result


def _finite_seconds(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise VideoDecompositionContractError(f"{label} must be a non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VideoDecompositionContractError(f"{label} must be a non-negative number") from exc
    if result < 0 or result != result or result in {float("inf"), float("-inf")}:
        raise VideoDecompositionContractError(f"{label} must be a non-negative finite number")
    return result


def _seconds(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, result)


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result


def _clip(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "artifact"
