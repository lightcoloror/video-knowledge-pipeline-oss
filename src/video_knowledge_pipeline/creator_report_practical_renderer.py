"""Render the derived reading view through the existing safe Markdown exporters."""
from __future__ import annotations

from typing import Any
import math
import re

from .transcript import format_timestamp


TYPE_LABELS = {"tutorial": "教程步骤", "talking_head": "口播内容", "visual_story": "画面叙事", "general": "内容参考"}


def _reading_summary(text: str) -> str:
    """Select existing reader sections; keep the full candidate in provenance."""
    selected, current = [], []
    allowed = {"一句话概览", "核心主题", "分段总结", "关键观点", "可执行动作清单", "核心观点", "总结"}
    keep = False
    for line in text.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            if keep:
                selected.extend(current)
            current = [line]
            keep = heading[1] in allowed
        else:
            current.append(line)
    if keep:
        selected.extend(current)
    return "\n".join(selected).strip() or text


def render_practical_markdown(report: dict[str, Any]) -> str:
    from .creator_report import FACT_LABELS, STAGE_LABELS, STATUS_LABELS, _safe, _safe_summary_markdown

    view = report["practical"]
    reference = view["reference_summary"]
    kind = view["video_type"]

    def cell(value: Any) -> str:
        return _safe(value).replace("\n", "；")

    def when(row: dict[str, Any]) -> str:
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, ValueError, TypeError, OverflowError):
            return "时间待核"
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start
                or any(i.get("code") in {"invalid_time", "invalid_visual_time"}
                       for i in row.get("issues", []) if isinstance(i, dict))):
            return "时间待核"
        return f"{format_timestamp(start)} – {format_timestamp(end)}"

    def doubtful(row: dict[str, Any]) -> bool:
        return bool(row.get("issues")) or row.get("status") == "review-required"

    lines = [f"# {_safe(report['title'])}", "", "> 实用参考报告。用于阅读、学习和回找素材；标为“待核”的字词、参数与画面信息请结合原片判断。", "",
             f"内容类型：{TYPE_LABELS.get(kind, '内容参考')}。", "", "## 总结", ""]
    if reference["text"]:
        if reference["status"] == "reference":
            lines += ["> 参考总结：来自当前已登记的机器候选，尚未通过原严格验收。", ""]
        lines += [_safe_summary_markdown(_reading_summary(reference["text"])), ""]
    else:
        lines += ["当前没有可绑定到本次内容的总结；以下整理已有转写与画面信息。", ""]

    if kind == "tutorial":
        lines += ["## 教程步骤：步骤—按钮—参数—结果", "",
                  "操作顺序依据原片文字整理；按钮和参数不等于已独立核验的实际设置，缺少结果证据时保留待核。", "",
                  "| 时间 | 步骤 | 按钮／入口 | 参数 | 结果／待核 |", "| --- | --- | --- | --- | --- |"]
        for step in view["steps"]:
            note = "（待核）" if doubtful(step) else ""
            lines.append(f"| {when(step)} | {cell(step['step'])}{note} | {cell(step['button'])} | "
                         f"{cell('；'.join(step['parameters']))} | {cell(step['result'])} |")
        if not view["steps"]:
            lines += ["", "尚无可定位的操作步骤；保留下面的文字与截图参考。"]
        lines += [""]

    heading = {"tutorial": "图文操作参考", "talking_head": "口播与画面分段", "visual_story": "画面与镜头拆解"}.get(kind, "图文分段")
    lines += [f"## {heading}", "", "代表帧与分段用于回找内容；单帧没有展示的动作、参数或结果保持待核。", ""]
    if kind == "talking_head":
        for chapter in view.get("chapters", []):
            lines += [f"### 章节 {chapter['index']} · {when(chapter)} · {_safe(chapter['title'])}", ""]
            for key in chapter["images"]:
                preview = report.get("chapter_previews", {}).get(key)
                if preview:
                    lines += [f"![{_safe(chapter['title'])} · {format_timestamp(preview['time'])} 代表帧]({preview['data_uri']})", ""]
            seen_observations = set()
            for observation in chapter["observations"]:
                value = str(observation.get("text") or "").strip()
                if value and value not in seen_observations:
                    seen_observations.add(value)
                    lines += [f"- **画面观察**：{_safe(value)}"]
            for candidate in chapter.get("screen_text_candidates", []):
                lines += [f"- **画面文字候选（待核）**：{_safe(candidate['text'])}"]
            if not chapter["images"]:
                lines += ["- 本章没有可绑定的代表帧；保留文字导航，不补画面事实。"]
            lines += [""]
        lines += ["### 技术分镜详情", "",
                  f"原有 {len(view['shots'])} 个技术候选分段保留在结构化投影及本次分镜 CSV 附件中。"
                  "章节用于内容导航，不是镜头边界；自动边界尚未逐项人工确认。", ""]
    for shot in ([] if kind == "talking_head" else view["shots"]):
        lines += [f"### 分段 {shot['index']} · {when(shot)}", ""]
        preview = report["previews"].get(shot["shot_id"])
        if preview:
            lines += [f"![分段 {shot['index']} 代表帧]({preview['data_uri']})", ""]
        facts = shot["facts"]
        keys = ["subject_action", "screen_text"]
        if kind == "visual_story":
            keys = ["shot_type", "camera_movement", "subject_action", "composition", "lighting", "screen_text"]
        for key in keys:
            value = facts.get(key)
            if value not in (None, "", "unknown", [], {}):
                lines.append(f"- **{FACT_LABELS[key]}**：{_safe(value)}")
        if doubtful(shot):
            lines.append("- **待核**：本段部分文字或画面信息不完整；未展示的信息不补成事实。")
        lines += [""]

    lines += ["## 整理转写", "", "仅合并可确认的重复投影；原始逐字稿及其时间信息保留。不同时间的重复表达、数字和否定词不会因相似而合并。", ""]
    if report["transcript_scope"] == "timeline_fragments":
        lines += ["> 当前来源为时间线摘录，并非完整逐字稿。", ""]
    for cue in view["transcript"]:
        text = _safe(cue["text"]) if str(cue["text"]).strip() else "〔此处原转写为空，待核〕"
        speaker = _safe(cue["speaker"]) + "：" if cue.get("speaker") else ""
        note = " **〔待核〕**" if doubtful(cue) else ""
        lines += [f"**{when(cue)}** {speaker}{text}{note}", ""]

    lines += ["## 局部待核", ""]
    technical_count = 0
    seen = set()
    for issue in view.get("issues", []):
        if not isinstance(issue, dict) or not issue.get("message"):
            continue
        refs = issue.get("evidence_refs") or []
        if kind == "talking_head" and refs and all(isinstance(ref, str) and re.fullmatch(r"shots:\d+(?::.*)?", ref) for ref in refs):
            technical_count += 1
            continue
        key = (issue.get("code"), issue.get("message"), issue.get("start"), issue.get("end"))
        if key in seen:
            continue
        seen.add(key)
        clock = f"{when(issue)}：" if "start" in issue and "end" in issue else ""
        lines.append(f"- {clock}{_safe(issue['message'])}")
    if not seen:
        lines += ["逐项待核标记见相应步骤和分段；没有标记不代表已经逐句人工核验。"]
    if technical_count:
        lines += [f"- 技术候选分镜另有 {technical_count} 项局部标记，完整记录保留在结构化投影／分镜附件；本阅读视图未关闭这些待核项。"]

    lines += ["", "## 严格验收记录（保留）", "", "本参考视图不替代严格验收，也不把参考总结安装为正式总结。", ""]
    for stage in report["stages"]:
        lines.append(f"- {STAGE_LABELS.get(stage['key'], stage['key'])}：{STATUS_LABELS.get(stage['status'], _safe(stage['status']))}")
    return "\n".join(lines).rstrip() + "\n"
