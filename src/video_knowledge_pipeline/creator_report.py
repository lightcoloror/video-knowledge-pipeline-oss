"""Creator-facing projection of existing VKP evidence, never a second analysis truth.

Intent / decision / reason / evidence / scope are recorded in the adapter document.
This layer exports candidates; it cannot approve, publish, download, or run models.
"""
from __future__ import annotations

import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .canonical_json import canonical_json_sha256
from .creator_report_renderers import bundle_file, export_format, frame_preview, html_document
from .creator_scriptbreak import export_prompt_pack, project_candidate, validate_brief
from .file_hash import sha256_file
from .shot_breakdown import build_shot_breakdown
from .smart_summary_input_pack import select_canonical_transcript_path
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic
from .transcript import format_timestamp, parse_transcript
from .video_decomposition import build_video_decomposition_report
from .workflow_completion import workflow_completion_status


SCHEMA = "video_knowledge_pipeline.creator_report.v1"
EXPORT_ROOT = "exports/creator-report"
PRACTICAL_EXPORT_ROOT = "exports/creator-report-practical"
REPORT_VIEWS = ("strict", "practical")
VIDEO_TYPES = ("auto", "tutorial", "talking_head", "visual_story", "general")
FORMATS = ("md", "html", "docx", "pdf", "srt", "csv")
STATUS_LABELS = {"completed": "已完成", "not-applicable": "不适用", "review-required": "待复核",
                 "awaiting-agent": "等待总结接力", "blocked": "尚有阻塞"}
STAGE_LABELS = {"input_validation": "输入材料", "primary_transcript": "主路逐字稿",
                "secondary_asr_and_differences": "双路语音对照", "speech_completeness": "语音覆盖",
                "visual_evidence": "画面与 OCR", "smart_summary": "智能总结",
                "summary_validation": "总结验收", "readable_export": "正式可读产物"}
FACT_LABELS = {"shot_type": "景别", "camera_movement": "运镜", "subject_action": "画面动作",
               "composition": "构图", "lighting": "光线", "dialogue_or_narration": "口播／对白",
               "screen_text": "画面文字"}
DIMENSION_LABELS = dict(zip(
    ("positioning", "hook", "body_structure", "payoff", "content_value", "emotion", "expression",
     "visual_language", "subtitle_packaging", "voice_delivery", "bgm_style", "bgm_identity",
     "audiovisual_coordination", "reusable_framework", "technique", "non_copyable_factor",
     "imitation_risk", "differentiation", "author_identity", "behind_the_scenes", "performance_metrics"),
    ("内容定位", "开场吸引点", "主体结构", "结尾兑现", "内容价值", "情绪", "表达方式",
     "镜头语言", "字幕包装", "声音表达", "音乐风格", "音乐身份", "视听配合", "可复用结构",
     "表现技巧", "不可照搬因素", "复刻风险", "差异化", "创作者身份", "幕后条件", "传播数据"),
))


def build_creator_report(bundle_dir: str | Path, *, title: str = "", brief: dict[str, Any] | None = None,
                         view: str = "strict", video_type: str = "talking_head",
                         ocr_results_json: str | Path | None = None) -> dict[str, Any]:
    if view not in REPORT_VIEWS or video_type not in VIDEO_TYPES:
        raise ValueError("invalid report view or video type")
    if view == "practical" and brief:
        raise ValueError("practical reports do not accept a creative brief")
    if ocr_results_json is not None and view != "practical":
        raise ValueError("explicit OCR candidate projection requires the practical view")
    root = Path(bundle_dir).resolve()
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("Bundle manifest must be an object")
    brief = validate_brief(brief)
    decomposition = build_video_decomposition_report(root, title=title, write=False)
    breakdown = build_shot_breakdown(root, title=title, write=False)
    completion = workflow_completion_status(root, include_readable_export=False)
    stages = {row["key"]: row for row in completion["stages"]}
    transcript_path = select_canonical_transcript_path(root, manifest)
    cues = parse_transcript(transcript_path) if transcript_path else []
    transcript = [{"start": cue.start, "end": cue.end, "text": cue.text, "speaker": cue.speaker,
                   "source_cue_index": index, "source_cue_id": cue.segment_id,
                   "source_segment_ids": list(cue.source_segment_ids)} for index, cue in enumerate(cues)]
    # A timeline fallback is visibly a fragment collection, never a complete transcript.
    if not transcript:
        timeline = read_json(root / "timeline.json") if (root / "timeline.json").is_file() else []
        transcript = [{"start": row.get("start", 0), "end": row.get("end", 0),
                       "text": row.get("corrected_transcript") or row.get("transcript") or "", "speaker": row.get("speaker", "")}
                      for row in timeline if isinstance(row, dict) and (row.get("corrected_transcript") or row.get("transcript"))]
    summary_stage = stages.get("summary_validation", {})
    installed_summary = (stages.get("smart_summary", {}).get("evidence") or {}).get("path")
    summary = Path(installed_summary).read_text(encoding="utf-8-sig") if (
        summary_stage.get("status") == "completed" and installed_summary) else ""
    sources = list(decomposition["source_artifacts"])
    summary_ids = {row["artifact_id"] for row in sources if row.get("role") == "smart_summary"}
    # The decomposition selector may see an older filename than the validated summary.
    # Permit its summary-derived claims only when that exact source is the approved one.
    summary_findings_current = bool(summary and summary_ids) and all(
        Path(row["path"]).resolve() == Path(installed_summary).resolve()
        and row["sha256"] == sha256_file(Path(installed_summary))
        for row in sources if row.get("role") == "smart_summary")
    findings = [row for row in decomposition["findings"] if summary_findings_current or not summary_ids.intersection(row.get("evidence_ids", []))]
    if summary and not summary_findings_current:
        sources.append({"artifact_id": "validated-summary-001", "role": "validated_smart_summary",
                        "path": str(Path(installed_summary).resolve()), "sha256": sha256_file(Path(installed_summary))})
    shots = breakdown["shots"]
    previews, issues = {}, []
    oral_reading = view == "practical" and video_type in {"talking_head", "auto"}
    if len(shots) > 500 and not oral_reading:
        raise ValueError("creator report supports at most 500 shots; split the Bundle first")
    for shot in ([] if oral_reading else shots):
        for frame in shot.get("reference_frames", []):
            try:
                previews[shot["shot_id"]] = frame_preview(root, str(frame))
                break
            except (OSError, ValueError, Warning) as exc:
                issues.append({"shot_id": shot["shot_id"], "reason": type(exc).__name__,
                               "message": "代表帧不可用或不在当前 Bundle 内；保留文字与时间范围。"})
    projection = {
        "schema": SCHEMA, "title": title or decomposition["title"], "brief": brief,
        "candidate_only": True, "publication_allowed": False,
        "workflow_status": completion["status"],
        "stages": [{"key": row["key"], "status": row["status"]} for row in completion["stages"]],
        "summary": summary, "summary_status": summary_stage.get("status", "blocked"),
        "transcript": transcript, "transcript_scope": "canonical" if cues else "timeline_fragments",
        "findings": findings, "shots": shots, "style_fingerprint": breakdown["style_fingerprint"],
        "shot_readiness": breakdown["readiness"], "previews": previews, "preview_issues": issues,
        "sources": sources, "technical_sources": breakdown["input_provenance"],
        "operator_boundary": {"model_calls_made": 0, "network_calls_made": 0,
                              "timeline_mutated": False, "quality_gate_bypassed": False,
                              "formal_artifact_created": False},
    }
    if view == "practical":
        from .creator_report_practical_view import build_practical_view
        from .creator_report_reference_summary import load_reference_summary
        from .creator_report_review_notes import collect_review_notes
        reference = load_reference_summary(root, strict_summary=summary,
            strict_summary_path=Path(installed_summary) if summary else None)
        notes = collect_review_notes(root, transcript)
        timeline = read_json(root / "timeline.json") if (root / "timeline.json").is_file() else []
        if ocr_results_json is not None:
            from .ocr_block_candidates import project_ocr_block_candidates
            ocr_source = Path(ocr_results_json).expanduser().resolve()
            if not ocr_source.is_file() or ocr_source.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("explicit OCR results must be an existing JSON file within 64 MiB")
            ocr_payload = read_json(ocr_source)
            items = ocr_payload.get("items") if isinstance(ocr_payload, dict) else None
            if not isinstance(items, list):
                raise ValueError("explicit OCR results must contain an items list")
            timeline = project_ocr_block_candidates(root, timeline, items)
            projection["sources"].append({"artifact_id": "oral-ocr-candidate-input", "role": "ocr_candidate_projection",
                                          "path": str(ocr_source), "sha256": sha256_file(ocr_source)})
        from .ocr_block_candidates import bind_ocr_candidates_for_display
        timeline = bind_ocr_candidates_for_display(root, timeline)
        practical = build_practical_view(manifest=manifest, transcript=notes["transcript"],
            timeline=timeline, shots=shots, reference_summary=reference,
            video_type="talking_head" if video_type == "auto" else video_type)
        if video_type == "auto":
            practical["basis"].update(source="scope_default_no_classification",
                note="本轮按口播类范围展示，未执行自动类型判断。")
        practical["reference_summary"] = reference
        if oral_reading:
            from .creator_report_oral_chapters import build_oral_chapters
            oral = build_oral_chapters(root, transcript=notes["transcript"], timeline=timeline,
                                      visuals=practical["visual_evidence"], reference=reference)
            practical["chapters"] = oral["chapters"]
            practical["oral_visual_summary"] = oral["summary"]
            projection["chapter_previews"] = oral["previews"]
        practical["issues"].extend(notes["issues"])
        secondary = stages.get("secondary_asr_and_differences", {})
        if (secondary.get("status") != "completed"
                and not any(issue.get("code") in {"asr_difference_pending", "asr_review_stale"}
                            for issue in practical["issues"])):
            practical["issues"].append({"code": "aggregate_asr_review_pending",
                "message": "双路转写核对尚未完成；当前记录不足以定位到具体字词，不能把未标记处当作全部核实。",
                "evidence_refs": ["dual-asr-review.json"]})
        if reference["status"] == "unavailable":
            practical["issues"].append({"code": "reference_summary_unavailable",
                "message": "参考总结未登记、来源已变化或文件不可用，本次没有套用旧总结。", "evidence_refs": []})
        projection.update(report_view="practical", video_type_requested=video_type, practical=practical)
        projection["sources"].extend(reference["sources"])
    projection["input_fingerprint"] = canonical_json_sha256(projection)
    return projection


def _text(value: Any) -> str:
    if value in (None, "", "unknown", [], {}):
        return "未知／尚无证据"
    if isinstance(value, dict):
        return "；".join(f"{k}：{_text(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "；".join(_text(v) for v in value)
    return str(value)


def _safe(value: Any) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#+!|])", r"\\\1", _text(value)).replace("\r", "")


def _safe_summary_markdown(text: str) -> str:
    """Retain heading/list structure while escaping untrusted inline content.

    Reuse the renderer's HTML-disabled MarkdownIt; never admit source HTML,
    external images, arbitrary links, or internal generation-mode metadata.
    """
    lines = []
    for line in text.splitlines():
        if re.match(r'^\s*生成方式\s*[：:]', line):
            continue
        heading = re.match(r'^(#{1,6})\s+(.+)$', line)
        bullet = re.match(r'^\s*[-*]\s+(.+)$', line)
        if heading:
            lines.append('#' * min(6,len(heading[1])+2) + ' ' + _safe(heading[2]))
        elif bullet:
            lines.append('- ' + _safe(bullet[1]))
        else:
            lines.append(_safe(line) if line.strip() else '')
    return '\n'.join(lines)


def render_creator_markdown(report: dict[str, Any]) -> str:
    if report.get("report_view") == "practical":
        from .creator_report_practical_renderer import render_practical_markdown
        return render_practical_markdown(report)
    lines = [f"# {_safe(report['title'])}", "", "> 创作者参考报告 · 候选版。下载成功不代表分析通过或可以发布。", "",
             f"输入版本：`{report['input_fingerprint'][:16]}`", "", "## 完成情况", ""]
    for row in report["stages"]:
        lines.append(f"- {STAGE_LABELS.get(row['key'], row['key'])}：{STATUS_LABELS.get(row['status'], _safe(row['status']))}")
    lines += ["", "## 智能总结", "", _safe_summary_markdown(report["summary"]) if report["summary"] else
              "智能总结尚未通过现有质量验收；本报告不把候选或旧总结当作正式总结。", "", "## 内容拆解", ""]
    for row in report["findings"]:
        if row["status"] == "unavailable":
            continue
        label = DIMENSION_LABELS.get(row["dimension"], row["dimension"])
        status = {"confirmed": "有直接证据", "inferred": "推断，待核实"}.get(row["status"], row["status"])
        lines += [f"- **{_safe(label)}**（{_safe(status)}）：{_safe(row['claim'])}",
                  "  证据：" + _safe(", ".join(row.get("evidence_ids", [])))]
    missing = list(dict.fromkeys(DIMENSION_LABELS.get(r["dimension"], r["dimension"]) for r in report["findings"] if r["status"] == "unavailable"))
    if missing:
        lines += ["", "尚无足够证据：" + _safe("、".join(missing)) + "。"]
    lines += ["", "## 逐镜头拉片", "", "按已验证的技术镜头边界展示。未知机位、景别与光线保持未知；单张代表帧不证明 OCR 无遗漏。", ""]
    if not report["shots"]:
        lines += ["缺少可验证的镜头边界，本节未生成。", ""]
    for shot in report["shots"]:
        lines += [f"### 镜头 {shot['index']} · {shot['start_time']} – {shot['end_time']}", ""]
        preview = report["previews"].get(shot["shot_id"])
        if preview:
            lines += [f"![镜头 {shot['index']} 代表帧]({preview['data_uri']})", ""]
        else:
            lines += ["代表帧缺失或不可安全导出。", ""]
        for key, label in FACT_LABELS.items():
            lines += [f"- **{label}**：{_safe(shot['facts'].get(key))}"]
        lines += ["- 来源：" + _safe(", ".join(str(n) for n in shot.get("timeline_indexes", []))) + "（时间线片段）", ""]
    lines += ["## 复刻准备", "", "本节为新作品的创作准备，不是对原片拍摄参数的确认。", ""]
    if report["brief"]:
        labels = {"product": "你的产品／主题", "audience": "目标观众", "goal": "视频目标", "resources": "拍摄条件"}
        lines += [f"- {labels[k]}：{_safe(v)}" for k, v in report["brief"].items()]
        lines += ["", "同包 ScriptBreak 项目保留原片时间结构，供生成新作品的分镜提示词。模板补全仅为建议，需复核后拍摄。"]
    else:
        lines += ["尚未提供产品、观众、目标和拍摄条件，未生成冒充定制方案的文案。补齐创作需求后可导出 ScriptBreak 候选项目。"]
    lines += ["", "## 逐字稿", ""]
    if report["transcript_scope"] == "timeline_fragments":
        lines += ["> 当前仅有时间线文字片段，不能作为全文逐字稿。", ""]
    for cue in report["transcript"]:
        lines += [f"**{format_timestamp(float(cue['start']))} – {format_timestamp(float(cue['end']))}** "
                  + (_safe(cue["speaker"]) + "：" if cue.get("speaker") else "") + _safe(cue["text"]), ""]
    if not report["transcript"]:
        lines += ["尚无逐字稿。", ""]
    lines += ["## 证据索引", "", "此处只列证据编号与摘要校验值；本机绝对路径不进入分享文档。", ""]
    for row in report["sources"]:
        lines += [f"- {_safe(row.get('artifact_id'))} · {_safe(row.get('role'))} · `{str(row.get('sha256', ''))[:16]}`"]
    return "\n".join(lines).rstrip() + "\n"


def _check_external_destination(root: Path, destination: Path) -> None:
    """An explicit read-only-source export cannot write into the source tree.

    Reject ancestors as well: a source's parent may contain protected analysis
    entrypoints. Existing external indexes are reusable only when an unmodified
    receipt proves ownership by this exact source and destination.
    """
    if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("explicit output_dir must not overlap the source Bundle tree")
    index, latest = destination / "index.html", destination / "latest.json"
    if not index.exists() and not latest.exists():
        return
    if not latest.is_file():
        raise ValueError("external output_dir contains an unowned index; existing files preserved")
    receipt = read_json(bundle_file(destination, latest))
    if (not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA + ".receipt"
            or receipt.get("source_bundle") != str(root) or receipt.get("output_dir") != str(destination)):
        raise ValueError("external output_dir belongs to another source; existing files preserved")
    if index.exists() and (not isinstance(receipt.get("index_sha256"), str)
                          or sha256_file(bundle_file(destination, index)) != receipt["index_sha256"]):
        raise ValueError("external output index ownership/content changed; existing file preserved")


def export_creator_report(bundle_dir: str | Path, *, title: str = "", brief: dict[str, Any] | None = None,
                          formats: list[str] | None = None, pandoc: str = "", scriptbreak_script: str = "",
                          scriptbreak_sha256: str = "", write: bool = True,
                          view: str = "strict", video_type: str = "talking_head", interactive_html: bool = False,
                          output_dir: str | Path | None = None, ocr_results_json: str | Path | None = None) -> dict[str, Any]:
    formats = list(dict.fromkeys(FORMATS[:2] if formats is None else formats))
    if not formats or any(value not in FORMATS for value in formats):
        raise ValueError("formats must be a nonempty subset of md, html, docx, pdf, srt, csv")
    if type(interactive_html) is not bool:
        raise ValueError("interactive_html must be a boolean")
    root = Path(bundle_dir).resolve()
    external_destination = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if external_destination is not None:
        _check_external_destination(root, external_destination)
    if view == "practical" and (scriptbreak_script or scriptbreak_sha256):
        raise ValueError("practical reports do not execute ScriptBreak")
    report = build_creator_report(bundle_dir, title=title, brief=brief, view=view, video_type=video_type,
                                  ocr_results_json=ocr_results_json)
    if scriptbreak_script and not report["brief"]:
        raise ValueError("a creative brief is required for ScriptBreak")
    if not write:
        return {"status": "preview", "written": False, "formats": formats, "report": report}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_title = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", report["title"]).strip(" .")[:64] or "video"
    revision = f"{stamp}_{report['input_fingerprint'][:8]}_{uuid.uuid4().hex[:6]}"
    name = f"{safe_title}_{revision}"
    export_root = PRACTICAL_EXPORT_ROOT if view == "practical" else EXPORT_ROOT
    destination = external_destination if external_destination is not None else root / export_root
    if output_dir is None and not destination.resolve().is_relative_to(root):
        raise ValueError("export directory escapes this Bundle")
    folder = destination / revision
    # Reject pre-existing export-tree junctions that would escape the Bundle.
    if not folder.resolve().is_relative_to(destination.resolve()):
        raise ValueError("export directory escapes this Bundle")
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "projection.json", report)
    markdown = render_creator_markdown(report)
    artifacts, failures = [], []

    def record(path: Path, kind: str) -> None:
        artifacts.append({"format": kind, "path": str(path) if output_dir is not None else path.relative_to(root).as_posix(),
                          "sha256": sha256_file(path), "bytes": path.stat().st_size})

    for fmt in formats:
        target = folder / (name + "." + fmt)
        try:
            if fmt == "html" and interactive_html:
                from .creator_report_interactive import interactive_html_document
                write_text_atomic(target, interactive_html_document(markdown, report["title"], report))
            elif fmt == "srt":
                from .creator_report_timed_exports import render_report_srt
                srt, source_map = render_report_srt(report)
                write_text_atomic(target, srt)
                map_target = folder / "subtitle-source-map.json"
                write_json(map_target, source_map)
                record(map_target, "subtitle-source-map")
            elif fmt == "csv" and report.get("practical", {}).get("video_type") == "talking_head":
                from .creator_report_timed_exports import render_report_chapter_csv
                write_text_atomic(target, render_report_chapter_csv(report))
            elif fmt == "csv":
                from .shot_breakdown_exports import render_shot_breakdown_csv
                write_text_atomic(target, render_shot_breakdown_csv(report))
            else:
                export_format(markdown, target, title=report["title"], pandoc=pandoc)
            record(target, fmt)
        except (OSError, ValueError, RuntimeError, ImportError, subprocess.TimeoutExpired) as exc:
            failures.append({"format": fmt, "error_type": type(exc).__name__, "message": str(exc)[:600]})
    if report.get("practical", {}).get("video_type") == "talking_head":
        from .shot_breakdown_exports import render_shot_breakdown_csv
        technical_csv = folder / "technical-shots.csv"
        write_text_atomic(technical_csv, render_shot_breakdown_csv(report))
        record(technical_csv, "technical-shots-csv")
    if report["brief"]:
        try:
            target = folder / (name + ".scriptbreak")
            write_json(target, project_candidate(report))
            record(target, "scriptbreak")
            if scriptbreak_script:
                pack = export_prompt_pack(target, Path(scriptbreak_script), expected_sha256=scriptbreak_sha256)
                prompt_path = folder / (name + ".prompt-pack.md")
                write_text_atomic(prompt_path, pack["text"])
                record(prompt_path, "prompt-pack")
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            failures.append({"format": "scriptbreak", "error_type": type(exc).__name__, "message": str(exc)[:600]})
    record(folder / "projection.json", "projection")
    receipt = {"schema": SCHEMA + ".receipt", "status": "partial_failed" if failures else "exported_candidate",
               "candidate_only": True, "publication_allowed": False, "input_fingerprint": report["input_fingerprint"],
               "title": title, "brief": report["brief"], "view": view, "video_type": video_type,
               "formats": formats, "artifacts": artifacts, "interactive_html": interactive_html,
               "source_bundle": str(root), "output_dir": str(destination) if output_dir is not None else "",
               "ocr_results_json": str(Path(ocr_results_json).expanduser().resolve()) if ocr_results_json is not None else "",
               "failures": failures, "generated_at": stamp, "workflow_status": report["workflow_status"],
               "preview_issues": report["preview_issues"], "operator_boundary": report["operator_boundary"]}
    # Recheck immediately before advancing the pointer; keep failed/stale revisions for diagnosis.
    with bundle_write_lock(destination if output_dir is not None else root, operation="creator_report", timeout_seconds=1.0):
        if output_dir is not None:
            _check_external_destination(root, destination)
        try:
            current = build_creator_report(root, title=title, brief=brief, view=view, video_type=video_type,
                                           ocr_results_json=ocr_results_json)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            receipt["status"] = "stale_during_export"
            receipt["input_revalidation_error"] = {"error_type": type(exc).__name__, "message": str(exc)[:300]}
        else:
            if current["input_fingerprint"] != report["input_fingerprint"]:
                receipt["status"] = "stale_during_export"
        if receipt["status"] != "stale_during_export":
            links = [f"# {_safe(report['title'])} · 报告下载", "", "> 候选报告。缺口与质量状态见正文。", ""]
            links += [f"- [{a['format']}]({quote((Path(a['path']).relative_to(destination) if output_dir is not None else Path(a['path']).relative_to(export_root)).as_posix())})" for a in artifacts if a["format"] != "projection"]
            links += ["", f"生成结果：{receipt['status']}。失败格式：" + ", ".join(f["format"] for f in failures)]
            write_text_atomic(destination / "index.html", html_document("\n".join(links), report["title"]))
            receipt["index_sha256"] = sha256_file(destination / "index.html")
        write_json(folder / "receipt.json", receipt)
        if receipt["status"] != "stale_during_export":
            write_json(destination / "latest.json", receipt)
    return receipt


def creator_report_status(bundle_dir: str | Path, *, view: str = "strict", output_dir: str | Path | None = None) -> dict[str, Any]:
    if view not in REPORT_VIEWS:
        raise ValueError("invalid report view")
    root = Path(bundle_dir).resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir is not None else root / (PRACTICAL_EXPORT_ROOT if view == "practical" else EXPORT_ROOT)
    path = destination / "latest.json"
    if not path.is_file():
        return {"status": "missing", "candidate_only": True}
    try:
        receipt = read_json(bundle_file(destination if output_dir is not None else root, path))
        if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA + ".receipt":
            raise ValueError("invalid receipt schema")
        if not isinstance(receipt.get("artifacts"), list) or not receipt["artifacts"]:
            raise ValueError("missing artifact list")
        if any(not isinstance(a, dict)
               or not all(isinstance(a.get(k), str) and a[k] for k in ("format", "path", "sha256"))
               or a["format"] not in {*FORMATS, "scriptbreak", "prompt-pack", "projection", "technical-shots-csv", "subtitle-source-map"}
               or re.fullmatch(r"[0-9a-f]{64}", a["sha256"]) is None
               for a in receipt["artifacts"]):
            raise ValueError("invalid artifact entry")
        if not isinstance(receipt.get("title"), str) or not isinstance(receipt.get("brief"), dict):
            raise ValueError("invalid report parameters")
        if receipt.get("view", "strict") != view or receipt.get("video_type", "auto") not in VIDEO_TYPES:
            raise ValueError("invalid report view parameters")
        if output_dir is not None and (receipt.get("source_bundle") != str(root) or receipt.get("output_dir") != str(destination)):
            raise ValueError("external export belongs to a different source Bundle or output directory")
        validate_brief(receipt["brief"] or None)
        if (not isinstance(receipt.get("status"), str)
                or receipt["status"] not in {"exported_candidate", "partial_failed", "stale_during_export"}
                or not isinstance(receipt.get("input_fingerprint"), str)
                or re.fullmatch(r"[0-9a-f]{64}", receipt["input_fingerprint"]) is None):
            raise ValueError("incomplete receipt")
    except (OSError, ValueError, TypeError):
        return {"status": "invalid", "issues": ["invalid_receipt"], "candidate_only": True, "publication_allowed": False}
    issues = []
    if output_dir is not None:
        try:
            if sha256_file(bundle_file(destination, "index.html")) != receipt.get("index_sha256"):
                issues.append("index_changed")
        except (OSError, ValueError):
            issues.append("index_missing_or_unsafe")
    for artifact in receipt["artifacts"]:
        try:
            if sha256_file(bundle_file(destination if output_dir is not None else root, artifact["path"])) != artifact["sha256"]:
                issues.append("artifact_changed:" + artifact["format"])
        except (OSError, ValueError):
            issues.append("artifact_missing_or_unsafe:" + artifact["format"])
    try:
        current = build_creator_report(root, title=receipt["title"], brief=receipt["brief"] or None,
            view=view, video_type=receipt.get("video_type", "auto"), ocr_results_json=receipt.get("ocr_results_json") or None)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return {"status": "stale", "issues": [*issues, "source_evidence_unavailable"], "candidate_only": True,
                "publication_allowed": False, "export_status": receipt["status"]}
    if current["input_fingerprint"] != receipt["input_fingerprint"]:
        issues.append("inputs_changed")
    return {"status": "stale" if issues else "fresh", "export_status": receipt["status"], "issues": issues,
            "candidate_only": True, "publication_allowed": False, "workflow_status": current["workflow_status"]}
