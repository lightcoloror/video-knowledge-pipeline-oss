"""Deterministic, evidence-linked reading view; source and strict gates stay intact.

Intent: make an existing creator report useful without another model pass.
Decision: derive display rows from canonical cues, timeline observations and shots.
Scope: no I/O, source mutation, correction, acceptance or publication decisions.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any

from .content_clip_query_profiles import get_content_clip_query_profile
from .semantic_chapter_plan import TOPIC_MARKERS
from .text_normalization import compact_ascii_cjk


SCHEMA = "video_knowledge_pipeline.creator_report_practical_view.v1"
VIDEO_TYPES = ("auto", "tutorial", "talking_head", "visual_story", "general")
PENDING = "待核／原片未明确展示"
_ACTION_WORDS = ("点击", "单击", "双击", "右键", "按下", "选择", "打开", "进入", "设置", "设为",
                 "调整", "输入", "填写", "拖动", "勾选", "保存", "导出", "导入", "上传", "下载",
                 "安装", "运行", "新建", "复制", "粘贴", "切换", "点亮", "调到", "调至", "保留", "点", "选")
_ACTION = re.compile("|".join(_ACTION_WORDS))
_SCREEN = re.compile(r"屏幕录制|软件界面|屏幕操作|操作界面|screen.?record|screencast|screen.?capture|按钮|菜单", re.I)
_TALKING = re.compile(r"真人口播|对着镜头|面对镜头|talking.?head|speaking to camera", re.I)
_RESULT = re.compile(r"界面显示|页面显示|屏幕显示|看到|出现|生成了|得到|导出完成|导出成功|保存成功|安装成功|完成了")
_NUMBER = r"[-+]?\d+(?:\.\d+)?(?:\s*(?:到|至|比|[-~～])\s*\d+(?:\.\d+)?)?"
_VALUE = _NUMBER + r"(?:[A-Za-z0-9.%:/×xX_\-]*)(?:\s*(?:fps|kbps|Mbps|px|倍|秒|毫秒|分钟))?(?:左右)?"
_PARAMETER_FIELDS = r"分辨率|帧率|码率|宽度|高度|音量|速度|阈值|质量|时长|格式|尺寸|参数|比例|温度|变速|叠画|滤镜|强度|fps|resolution|bitrate"
_PARAMETER = re.compile(
    r"(?:" + _PARAMETER_FIELDS + r")(?:的)?(?:强度|速度|数值|时长)?\s*"
    r"(?:设置为|调整为|调整到|调到|调至|给到|设为|设置成|为|是|[:：=])?\s*(?:大概|大约|约)?\s*" + _VALUE,
    re.I,
)
_BOUND_PARAMETER = re.compile(
    r"(?:" + _PARAMETER_FIELDS + r")[，,\s]*"
    r"(?:(?:这里|然后|我们|再|接着|就)\s*)*(?:把|将)?(?:它|这个|该项|数值|强度|速度|时长)?"
    r"(?:的)?\s*(?:大概|大约|约)?\s*(?:调到|调至|调整到|设为|设置为|设置成|选择|选|给到)\s*" + _VALUE,
    re.I,
)
_DURATION = re.compile(r"(?:大概|大约|约)?\s*(?:保留|留出|时长(?:为|设为|调到)?)\s*" + _NUMBER + r"\s*(?:毫秒|秒|分钟)(?:左右)?")
_FORMAT = re.compile(r"(?:格式|编码)\s*(?:设置为|设为|为|是|[:：=])?\s*(?:mp4|mov|mkv|png|jpg|h264|h265|hevc)\b", re.I)
_UNIT_VALUE = re.compile(r"(?<![A-Za-z0-9])" + _NUMBER + r"\s*(?:fps|kbps|Mbps|px|p|%|倍|毫秒|秒|分钟)(?:左右)?(?![A-Za-z0-9])", re.I)
_INPUT_VALUE = re.compile(r"(?:输入|填写)\s*[“\"「]?[-+]?\d+(?:\.\d+)?(?:\s*(?:%|秒|fps))?[”\"」]?")
_QUOTED_BUTTON = re.compile(r"(?:点击|单击|双击|按下|选择|打开|进入|切换到?|点亮|点|选)\s*[“\"「『《]([^”\"」』》\n]{1,40})[”\"」』》]")
_NAMED_BUTTON = re.compile(r"(?:点击|单击|双击|按下|点亮|点|选)\s*(?:右上角|左上角|右侧|左侧|上方|下方)?的?([A-Za-z0-9一-鿿 _\-]{1,18}?)(?:按钮|菜单|选项)")
_TARGET_VERBS = {"点击", "单击", "双击", "按下", "选择", "打开", "进入", "切换", "点亮", "点", "选"}
_SENTENCE_END = re.compile(r"[。！？!?；;\n]|\.(?!\d)")
_CONTINUED_SCALAR = re.compile(r"(?:将|把)(?:它|这个|该项)(?:调到|调至|调整到|设为|设置为|设置成)\s*" + _VALUE)
_CONTINUED_RATIO = re.compile(r"(?:选择|选)\s*\d+\s*比\s*\d+")
_IRRELEVANT_LINES = {"为你推荐", "猜你喜欢", "相关推荐", "热门视频", "更多推荐", "点赞 评论 收藏 转发", "关注 点赞 评论 收藏 转发"}
# Same flags as ocr_inventory.build_ocr_inventory. That function performs I/O;
# this pure projection reuses its flag and native-engine score semantics only.
_OCR_SUSPICIOUS_FLAGS = frozenset({"screen_text_low_confidence", "low_ocr_confidence", "ocr_text_low_information", "ocr_wrapper_only"})
_OCR_EMPTY_FLAGS = frozenset({"ocr_text_empty", "missing_ocr", "missing_visual_text"})
_UI_STATS = re.compile(r"^(?:\d{1,2}:\d{2}\s*[·•|]?\s*)?(?:使用量|播放量|点赞量|收藏量|转发量|评论量|浏览量)\s*[:：]?\s*\d+(?:\.\d+)?\s*(?:万|亿|次|人)?$")
_METRIC_CONTEXT = re.compile(r"使用量|播放量|点赞[量数]|浏览量|收藏量|评论量|转发量|流量|指标|数据|点击量|点击率")
_TYPE_ALIASES = {"教程": "tutorial", "操作教程": "tutorial", "口播": "talking_head", "真人口播": "talking_head",
                 "剧情": "visual_story", "视觉叙事": "visual_story", "通用": "general", "talking-head": "talking_head",
                 "visual-story": "visual_story"}


def _is_instruction(text: str) -> bool:
    result = _RESULT.search(text)
    for action in _ACTION.finditer(text):
        # A discussion of 点击量/下载率 is not an operation instruction.
        if not _valid_action(text, action):
            continue
        if not result or action.start() < result.start():
            return True
    return False


def _valid_action(text: str, action: re.Match) -> bool:
    if re.match(r"量|率|次数", text[action.end():]):
        return False
    if action.group() in {"点", "选"}:
        prefix = text[:action.start()]
        return not prefix or bool(re.search(r"[，,。；;：:\s]$|(?:这里|我们|再|先|然后|接着|接下来|最后|直接|就|可以|需要|要|去|里|中|的)$", prefix))
    return True


def _buttons(text: str) -> list[str]:
    labels = [match.group(1).strip() for pattern in (_QUOTED_BUTTON, _NAMED_BUTTON) for match in pattern.finditer(text)]
    for action in _ACTION.finditer(text):
        if action.group() not in _TARGET_VERBS or not _valid_action(text, action):
            continue
        target = re.split(r"[，,。；;！!？?\n]", text[action.end():], maxsplit=1)[0].strip()
        target = re.sub(r"^(?:右上角|左上角|右侧|左侧|上方|下方)(?:的)?", "", target)
        target = re.split(r"按钮|菜单|选项|之后|以后|然后|这里|进行|调到|设置为|设为|后|将|把", target, maxsplit=1)[0].strip()
        # An unquoted bare number is a value, not evidence of a button label.
        if re.fullmatch(r"[A-Za-z一-鿿][A-Za-z0-9一-鿿 _\-]{0,17}", target):
            labels.append(target)
    return list(dict.fromkeys(label for label in labels if label))


def _parameters(text: str) -> list[str]:
    matches = [match for pattern in (_PARAMETER, _BOUND_PARAMETER, _DURATION, _FORMAT, _UNIT_VALUE, _INPUT_VALUE)
               for match in pattern.finditer(text)]
    # Containment is positional within one source, not substring similarity:
    # distinct values 30fps and 130fps must both survive across evidence rows.
    selected: list[re.Match] = []
    for match in sorted(matches, key=lambda item: (item.start(), -(item.end() - item.start()))):
        if not any(previous.start() <= match.start() and match.end() <= previous.end() for previous in selected):
            selected.append(match)
    return list(dict.fromkeys(match.group().strip() for match in selected))


def build_practical_view(*, manifest: dict, transcript: list[dict], timeline: list[dict],
                         shots: list[dict], reference_summary: dict | None = None,
                         video_type: str = "auto") -> dict:
    """Project a machine-reference reading view, retaining exact source locators.

    ``reference_summary`` may provide chapter boundaries, never new factual claims.
    A title alone cannot select tutorial mode. All returned objects are detached
    from the inputs; an uncertain local cue does not block the rest of the view.
    """
    if video_type not in VIDEO_TYPES:
        raise ValueError("video_type must be auto, tutorial, talking_head, visual_story, or general")
    cleanup: dict[str, Any] = {"preserves_canonical": True, "policy": "conservative_display_only",
                               "removed": [], "transcript_input_count": len(transcript),
                               "transcript_display_count": 0, "ocr_hidden_line_count": 0}
    issues: list[dict] = []
    cues = _display_transcript(transcript, cleanup, issues)
    context = "\n".join(row["text"] for row in cues)
    visuals = _visual_rows(timeline, context, cleanup, issues)
    display_shots = _display_shots(shots, visuals, context, cleanup, issues)
    selected, basis = _select_type(manifest, cues, visuals, display_shots, video_type)
    if video_type == "talking_head":
        _filter_oral_ocr(visuals, display_shots, context, cleanup)
    basis["boundary_strategy"] = (get_content_clip_query_profile("tutorial-step-v1")["boundary_strategy"]
                                  if selected == "tutorial" else "semantic_paragraphs_and_existing_shots")
    steps = _tutorial_steps(cues, visuals, display_shots, reference_summary, issues) if selected == "tutorial" else []
    cleanup["transcript_display_count"] = len(cues)
    cleanup["transcript_hidden_duplicate_count"] = sum(row["kind"] == "transcript" for row in cleanup["removed"])
    cleanup["ocr_hidden_line_count"] = sum(row["kind"] == "ocr" for row in cleanup["removed"])
    return {"schema": SCHEMA, "candidate_only": True, "human_confirmed": False,
            "video_type": selected, "basis": basis, "transcript": cues, "steps": steps,
            "shots": display_shots, "visual_evidence": visuals, "issues": issues, "cleanup": cleanup,
            "operator_boundary": {"model_calls_made": 0, "network_calls_made": 0,
                                  "source_mutated": False, "strict_gate_modified": False}}


def _issue(code: str, message: str, refs: list[str]) -> dict:
    return {"code": code, "message": message, "evidence_refs": list(dict.fromkeys(refs))}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def _range(row: dict) -> tuple[float, float, bool]:
    start, end = _number(row.get("start")), _number(row.get("end"))
    valid = start is not None and end is not None and end >= start
    return (start if start is not None else 0.0, end if end is not None else start or 0.0, valid)


def _local_issues(row: dict, refs: list[str], *, ocr: bool = False) -> list[dict]:
    result = deepcopy([issue for issue in row.get("issues", []) if isinstance(issue, dict)]) if isinstance(row.get("issues"), list) else []
    confidence = row.get("ocr_confidence") if ocr and "ocr_confidence" in row else row.get("confidence")
    if isinstance(confidence, dict):
        # Native OCR scores describe engine confidence, not factual accuracy.
        mean = _number(confidence.get("mean_confidence"))
        threshold = _number(confidence.get("minimum_mean_confidence"))
        low = (confidence.get("quality_threshold_passed") is False
               or confidence.get("valid_tsv") is False
               or (_number(confidence.get("low_confidence_block_count")) or 0) > 0
               or (mean is not None and mean < (threshold if threshold is not None else 70.0)))
        if confidence.get("word_count") == 0 or confidence.get("block_count") == 0:
            result.append(_issue("ocr_text_empty", "本处OCR记录为空结果，保留待核。", refs))
    else:
        try:
            value = float(confidence) if confidence is not None else None
            if ocr and value is not None and 1 < value <= 100:
                value /= 100
            low = value is not None and value < 0.6
        except (ValueError, TypeError):
            low = str(confidence).lower() == "low"
    if low:
        result.append(_issue("low_confidence", "本处识别置信度较低，待核。", refs))
    if row.get("review_required") or row.get("uncertain") or row.get("status") in {"review-required", "review_required", "uncertain"}:
        result.append(_issue("source_review_required", "源证据在本处标记待核。", refs))
    if ocr:
        flags = row.get("quality_issues") if isinstance(row.get("quality_issues"), list) else []
        result.extend(_issue(flag, f"源OCR质量标记：{flag}；本处待核。", refs)
                      for flag in flags if flag in _OCR_SUSPICIOUS_FLAGS or flag in _OCR_EMPTY_FLAGS)
        if row.get("quality_suspicious") or row.get("recognition_status") == "suspicious":
            result.append(_issue("ocr_quality_suspicious", "源OCR记录可疑，展示清理不解除本处待核。", refs))
        if row.get("recognition_status") == "empty":
            result.append(_issue("ocr_text_empty", "源OCR识别为空结果，保留待核。", refs))
    return result


def _source_identity(row: dict) -> set[str]:
    result = {f"{key}:{row[key]}" for key in ("source_cue_id", "cue_id", "source_segment_id") if row.get(key) not in (None, "")}
    for key in ("source_cue_indexes", "source_cue_ids"):
        if isinstance(row.get(key), list):
            result.update(f"{key}:{value}" for value in row[key])
    return result


def _display_transcript(transcript: list[dict], cleanup: dict, all_issues: list[dict]) -> list[dict]:
    output: list[dict] = []
    identities: list[set[str]] = []
    valid_ranges: list[bool] = []
    for index, source in enumerate(transcript):
        if not isinstance(source, dict):
            continue
        refs = list(dict.fromkeys([f"transcript:{index}", *[ref for ref in source.get("evidence_refs", []) if isinstance(ref, str)]]))
        raw = _text(source.get("text") or source.get("corrected_transcript") or source.get("transcript"))
        start, end, valid = _range(source)
        local = _local_issues(source, refs)
        if not raw:
            local.append(_issue("empty_transcript_cue", "该条逐字稿为空，保留原字幕引用待核。", refs))
        if not valid:
            local.append(_issue("invalid_time", "该条时间范围不可核定，展示时间待核。", refs))
        if "\ufffd" in raw:
            local.append(_issue("transcript_character_uncertain", "本处含不可识别字符；保留原文待核。", refs))
        identity = _source_identity(source)
        # Equality is deliberately stricter than compact_ascii_cjk: punctuation,
        # digits, negation and intentional repeated characters remain meaningful.
        key = re.sub(r"\s+", " ", raw)
        duplicate = None
        reason = ""
        for previous_index, previous in enumerate(output):
            if not valid or not valid_ranges[previous_index] or not key:
                continue
            if previous["_comparison_text"] != key or previous["speaker"] != _text(source.get("speaker")):
                continue
            same_time = start == previous["start"] and end == previous["end"]
            overlap = max(0.0, min(end, previous["end"]) - max(start, previous["start"]))
            shorter = min(end - start, previous["end"] - previous["start"])
            same_source = bool(identity.intersection(identities[previous_index]))
            adjacent_overlap = (previous["source_cue_indexes"][-1] == index - 1 and shorter > 0 and overlap / shorter >= 0.8)
            if same_time or (overlap > 0 and (same_source or adjacent_overlap)):
                duplicate, reason = previous, "same_text_same_time" if same_time else "same_text_explicit_overlap"
                break
        if duplicate is not None:
            duplicate["evidence_refs"].extend(refs)
            duplicate["source_cue_indexes"].append(index)
            duplicate["start"], duplicate["end"] = min(start, duplicate["start"]), max(end, duplicate["end"])
            duplicate["issues"].extend(local)
            duplicate["status"] = "review-required" if duplicate["issues"] else "evidence-present"
            cleanup["removed"].append({"kind": "transcript", "reason": reason, "text": raw,
                                       "evidence_refs": refs, "retained_in": duplicate["evidence_refs"][0]})
            identities[output.index(duplicate)].update(identity)
        else:
            output.append({"start": start, "end": end, "text": raw, "speaker": _text(source.get("speaker")),
                           "evidence_refs": refs, "source_cue_indexes": [index], "issues": local,
                           "status": "review-required" if local else "evidence-present", "_comparison_text": key})
            identities.append(identity)
            valid_ranges.append(valid)
        all_issues.extend(local)
    for row in output:
        row.pop("_comparison_text", None)
    return output


def _clean_ocr(value: str, ref: str, context: str, cleanup: dict) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    issues: list[dict] = []
    seen: dict[str, dict] = {}
    context_key = compact_ascii_cjk(context)
    for index, line in enumerate(value.splitlines()):
        line = line.strip()
        if not line:
            continue
        refs = [f"{ref}:line:{index}"]
        useful = compact_ascii_cjk(line)
        referenced = bool(useful) and useful in context_key
        protected = referenced or bool(re.search(r"\d", line)) or bool(_ACTION.search(line)) or bool(re.search(r"菜单|按钮|分辨率|帧率|码率|文件|编辑|视图|工具|设置|格式|帮助", line))
        reason = ""
        if _UI_STATS.fullmatch(line) and not referenced and not _METRIC_CONTEXT.search(context):
            reason = "unrelated_ui_statistics"
        elif not useful or (len(useful) / max(1, len(re.sub(r"\s", "", line))) < 0.3 and not protected):
            reason = "obvious_ocr_symbol_noise"
        elif line in _IRRELEVANT_LINES and not protected:
            reason = "unrelated_recommendation_ui"
        elif line in seen:
            reason = "repeated_ocr_display"
        if reason:
            retained = seen.get(line)
            if retained:
                retained["evidence_refs"].extend(refs)
            cleanup["removed"].append({"kind": "ocr", "reason": reason, "text": line, "evidence_refs": refs,
                                       **({"retained_in": retained["evidence_refs"][0]} if retained else {})})
            continue
        local = []
        if "\ufffd" in line or (len(line) >= 8 and line.count("\ufffd") / len(line) >= 0.2):
            local.append(_issue("ocr_character_uncertain", "本处画面文字含乱码；有用内容保留待核。", refs))
        row = {"text": line, "evidence_refs": refs, "issues": local}
        seen[line] = row
        kept.append(row)
        issues.extend(local)
    return kept, issues


def _observation_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [text for item in value for text in _observation_texts(item)]
    if isinstance(value, dict):
        return [text for key in ("summary", "description", "action", "observation", "observed_action", "events", "changes", "actions")
                for text in _observation_texts(value.get(key))]
    return []


def _filter_oral_ocr(visuals: list[dict], shots: list[dict], context: str, cleanup: dict) -> None:
    """Hide background writing only in the explicitly requested oral view.

    Matching is direct text correspondence, never a spelling correction or
    semantic relevance judgement. Raw timeline and strict shot facts stay with
    their caller; these rows are already detached display projections.
    """
    oral_text = compact_ascii_cjk(context)
    for row in [*visuals, *shots]:
        kept = []
        for line in row["ocr_lines"]:
            key = compact_ascii_cjk(line["text"])
            foreground_candidate = (line.get("candidate_only") is True
                and line.get("reference_usable") is True
                and line.get("role") in {"title", "subtitle", "tablet"})
            if (key and key in oral_text) or foreground_candidate:
                kept.append(line)
            else:
                cleanup["removed"].append({"kind": "ocr", "reason": "not_referenced_in_oral_transcript",
                                           "text": line["text"], "evidence_refs": line["evidence_refs"][:]})
        row["ocr_lines"] = kept
        shown = "\n".join(line["text"] for line in kept)
        if "facts" in row:
            row["facts"]["screen_text"] = shown
        else:
            row["screen_text"] = shown


def _visual_rows(timeline: list[dict], context: str, cleanup: dict, all_issues: list[dict]) -> list[dict]:
    result = []
    for index, source in enumerate(timeline):
        if not isinstance(source, dict):
            continue
        start, end, valid = _range(source)
        refs = [f"timeline:{index}"]
        key = next((key for key in ("corrected_visual_text", "visual_text", "ocr_text", "screen_text") if _text(source.get(key))), "")
        lines, local = _clean_ocr(_text(source.get(key)), f"timeline:{index}:{key}", context, cleanup) if key else ([], [])
        block_payload = source.get("ocr_block_candidates")
        if isinstance(block_payload, dict) and block_payload.get("source_verified") is True:
            seen_text = {line["text"] for line in lines}
            for block in block_payload.get("blocks", []):
                if not isinstance(block, dict) or block.get("reference_usable") is not True:
                    continue
                text = _text(block.get("text"))
                if not text or text in seen_text:
                    continue
                seen_text.add(text)
                lines.append({"text": text,
                    "evidence_refs": [f"timeline:{index}:ocr_block_candidates:{block.get('block_index')}"] ,
                    "candidate_only": True, "human_confirmed": False, "reference_usable": True,
                    "role": block.get("role", "unclassified"), "role_basis": block.get("role_basis"),
                    "score": block.get("score"), "source_image_sha256": block_payload.get("source_image_sha256")})
            if any(line.get("candidate_only") for line in lines):
                local.append(_issue("ocr_block_reference_candidate",
                    "本处展示带图像来源的OCR文字块候选；整体文字识别仍待核。", refs))
        local.extend(_local_issues(source, refs, ocr=True))
        if not key and not lines and (any(field in source for field in ("visual_text", "ocr_text", "screen_text", "ocr_confidence"))):
            local.append(_issue("ocr_text_empty", "本处画面文字为空；保留对应帧与待核标记。", refs))
        observations = []
        for field in ("temporal_visual_understanding", "visual_understanding", "subject_action"):
            observations.extend({"text": value, "evidence_refs": [f"timeline:{index}:{field}"]}
                                for value in _observation_texts(source.get(field)))
        if not lines and not observations and not local:
            continue
        if not valid:
            local.append(_issue("invalid_visual_time", "该处画面证据时间不完整，仅供独立查阅。", refs))
        result.append({"start": start, "end": end, "screen_text": "\n".join(row["text"] for row in lines),
                       "ocr_lines": lines, "observations": observations, "evidence_refs": refs,
                       "issues": local, "status": "review-required" if local else "evidence-present", "time_valid": valid,
                       "quality_issues": deepcopy(source.get("quality_issues", [])),
                       "ocr_confidence": deepcopy(source.get("ocr_confidence"))})
        all_issues.extend(local)
    return result


def _overlap(left: dict, right: dict) -> bool:
    return min(left["end"], right["end"]) > max(left["start"], right["start"]) or (
        left["start"] == left["end"] and right["start"] <= left["start"] <= right["end"])


def _display_shots(shots: list[dict], visuals: list[dict], context: str, cleanup: dict, all_issues: list[dict]) -> list[dict]:
    result = []
    for index, source in enumerate(shots):
        if not isinstance(source, dict):
            continue
        row = deepcopy(source)
        row["start"], row["end"], valid = _range(source)
        row.setdefault("index", index + 1)
        row.setdefault("shot_id", f"shot-{index + 1:04d}")
        refs = [f"shots:{index}"]
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        lines, local = _clean_ocr(_text(facts.get("screen_text") or source.get("screen_text")), f"shots:{index}:facts:screen_text", context, cleanup)
        local.extend(_local_issues(source, refs, ocr=True))
        # Timeline OCR has precise locators; use it before a broad shot projection.
        related_visuals = [visual for visual in visuals if visual["time_valid"] and valid and _overlap(visual, row)]
        precise = [line for visual in related_visuals for line in visual["ocr_lines"]]
        local.extend(issue for visual in related_visuals for issue in visual["issues"])
        combined: dict[str, dict] = {}
        for line in [*precise, *lines]:
            if line["text"] in combined:
                retained = combined[line["text"]]
                cleanup["removed"].append({"kind": "ocr", "reason": "repeated_ocr_projection", "text": line["text"],
                                           "evidence_refs": line["evidence_refs"][:],
                                           "retained_in": retained["evidence_refs"][0]})
                retained["evidence_refs"] = list(dict.fromkeys([*retained["evidence_refs"], *line["evidence_refs"]]))
            else:
                combined[line["text"]] = deepcopy(line)
        row["facts"] = {**facts, "screen_text": "\n".join(combined)}
        row["ocr_lines"] = list(combined.values())
        row["evidence_refs"], row["issues"] = refs, local
        row["quality_issues"] = list(dict.fromkeys([*[flag for flag in row.get("quality_issues", []) if isinstance(flag, str)],
                                                  *[flag for visual in related_visuals for flag in visual["quality_issues"] if isinstance(flag, str)]]))
        row["status"] = "review-required" if local else "evidence-present"
        row["candidate_only"], row["human_confirmed"] = True, False
        result.append(row)
        all_issues.extend(local)
    return result


def _select_type(manifest: dict, cues: list[dict], visuals: list[dict], shots: list[dict], requested: str) -> tuple[str, dict]:
    signals = []
    declared = []
    for prefix, source in (("manifest", manifest), ("manifest.metadata", manifest.get("metadata")),
                           ("manifest.analysis", manifest.get("analysis")), ("manifest.creator_report", manifest.get("creator_report"))):
        if not isinstance(source, dict):
            continue
        for key in ("video_type", "content_type"):
            raw = _text(source.get(key))
            value = _TYPE_ALIASES.get(raw, raw)
            if value in VIDEO_TYPES and value != "auto":
                declared.append((value, f"{prefix}.{key}"))
    operation_cues = [row for row in cues if _is_instruction(row["text"])]
    visual_screen = [row for row in visuals if _SCREEN.search(row["screen_text"] + " ".join(item["text"] for item in row["observations"]))]
    shot_screen = [row for row in shots if _SCREEN.search(" ".join(_observation_texts((row.get("facts") or {}).get("subject_action")))
                                                            + " " + _text((row.get("facts") or {}).get("shot_type")))]
    if operation_cues:
        signals.append({"kind": "operation_cues", "count": len(operation_cues), "evidence_refs": [ref for row in operation_cues for ref in row["evidence_refs"]]})
    if visual_screen or shot_screen:
        signals.append({"kind": "screen_operation_evidence", "count": len(visual_screen) + len(shot_screen),
                        "evidence_refs": [ref for row in [*visual_screen, *shot_screen] for ref in row["evidence_refs"]]})
    title = _text(manifest.get("title"))
    if "教程" in title or "tutorial" in title.lower():
        signals.append({"kind": "weak_title_hint", "value": title, "evidence_refs": ["manifest.title"]})
    if requested != "auto":
        return requested, {"requested": requested, "source": "explicit_request", "confidence": "user_selected",
                           "signals": signals, "note": "类型用于组织展示，不代表已核实内容。"}
    if declared:
        values = {value for value, _ in declared}
        signals.extend({"kind": "declared_type", "value": value, "evidence_refs": [ref]} for value, ref in declared)
        if len(values) == 1:
            return declared[0][0], {"requested": "auto", "source": "manifest_declaration", "confidence": "declared",
                                     "signals": signals, "note": "沿用输入声明；未独立核定视频类型。"}
        return "general", {"requested": "auto", "source": "conflicting_declarations", "confidence": "low",
                           "signals": signals, "note": "类型声明冲突，按通用内容展示。"}
    if len(operation_cues) >= 2 and (visual_screen or shot_screen):
        return "tutorial", {"requested": "auto", "source": "operation_and_visual_evidence", "confidence": "moderate",
                            "signals": signals, "note": "操作语句与画面记录共同支持教程候选；步骤仍需逐项核对。"}
    talking = [row for row in visuals if any(_TALKING.search(item["text"]) for item in row["observations"])]
    if len(talking) >= 2 and cues:
        signals.append({"kind": "talking_head_observations", "count": len(talking), "evidence_refs": [ref for row in talking for ref in row["evidence_refs"]]})
        return "talking_head", {"requested": "auto", "source": "repeated_visual_observations", "confidence": "moderate", "signals": signals}
    return "general", {"requested": "auto", "source": "insufficient_type_evidence", "confidence": "low",
                       "signals": signals, "note": "证据不足以稳定区分类型，按通用内容展示。"}


def _clauses(text: str) -> list[str]:
    # Clause timestamps remain their source cue's range; never invent word timing.
    split = r"(?<=[。；;！!？?\n])|[，,](?=(?:(?:然后|接着|再|接下来|最后)\s*)?(?:" + "|".join(_ACTION_WORDS) + r"))"
    return [value.strip(" ，,\n") for value in re.split(split, text) if value.strip(" ，,\n")]


def _chapter_starts(summary: dict | None) -> set[float]:
    if not isinstance(summary, dict):
        return set()
    chapters = summary.get("chapters")
    return {value for row in chapters or [] if isinstance(row, dict) and (value := _number(row.get("start"))) is not None} if isinstance(chapters, list) else set()


def _sentence_contexts(cues: list[dict]) -> dict[str, dict]:
    """Carry only a nearby unfinished spoken sentence across ASR cue splits."""
    result, previous = {}, None
    carry, carry_refs, carry_start = "", [], 0.0
    for cue in cues:
        adjacent = bool(previous and -0.1 <= cue["start"] - previous["end"] <= 1.0
                        and cue["end"] - carry_start <= 6.0 and cue["speaker"] == previous["speaker"])
        if carry and adjacent:
            result[cue["evidence_refs"][0]] = {"text": carry, "evidence_refs": carry_refs[:]}
            combined = carry + cue["text"]
            refs = list(dict.fromkeys([*carry_refs, *cue["evidence_refs"]]))
        else:
            combined, refs, carry_start = cue["text"], cue["evidence_refs"][:], cue["start"]
        ends = list(_SENTENCE_END.finditer(combined))
        carry = combined[ends[-1].end():] if ends else combined
        if ends:
            refs, carry_start = cue["evidence_refs"][:], cue["start"]
        carry_refs = refs if carry.strip() else []
        previous = cue
    return result


def _continued_parameter(context: dict | None, clause: str, refs: list[str]) -> dict | None:
    if not context:
        return None
    scalar, ratio = _CONTINUED_SCALAR.search(clause), _CONTINUED_RATIO.search(clause)
    numeric = scalar or ratio
    if not numeric:
        return None
    anchors = list(re.finditer(r"(?:" + _PARAMETER_FIELDS + r")", context["text"], re.I))
    # Conflicting candidate objects make a pronoun ambiguous; do not choose one.
    if not anchors or len({match.group() for match in anchors}) != 1:
        return None
    anchor = anchors[-1]
    if ratio and not scalar and anchor.group() != "比例":
        return None
    if scalar and anchor.group() in {"格式", "比例", "fps", "resolution", "bitrate"}:
        return None
    text = context["text"][anchor.start():] + clause[:numeric.end()]
    if len(text) > 64:
        return None
    return {"text": text, "evidence_refs": list(dict.fromkeys([*context["evidence_refs"], *refs])),
            "kind": "transcript", "parameter_only": True}


def _tutorial_steps(cues: list[dict], visuals: list[dict], shots: list[dict], summary: dict | None, all_issues: list[dict]) -> list[dict]:
    groups: list[list[dict]] = []
    chapter_starts = _chapter_starts(summary)
    sentence_contexts = _sentence_contexts(cues)
    for cue in cues:
        clauses = _clauses(cue["text"])
        actions = sum(_is_instruction(clause) for clause in clauses)
        continuing_sentence = sentence_contexts.get(cue["evidence_refs"][0])
        for clause in clauses:
            local = list(cue["issues"])
            if actions > 1:
                local.append(_issue("shared_cue_timing", "同一字幕含多个操作，步骤共用原字幕时间；精确动作时点待核。", cue["evidence_refs"]))
            row = {**cue, "text": clause, "issues": local, "_sentence_context": continuing_sentence}
            if _SENTENCE_END.search(clause):
                continuing_sentence = None
            semantic_break = (clause.startswith(TOPIC_MARKERS) or cue["start"] in chapter_starts)
            if _is_instruction(clause) or (groups and semantic_break and not _RESULT.search(clause)):
                groups.append([row])
            elif groups:
                groups[-1].append(row)
    output = []
    for index, group in enumerate(groups):
        start, end = min(row["start"] for row in group), max(row["end"] for row in group)
        # Include the interval until the next instruction, but never extend past
        # the supplied transcript simply because a technical shot lasts longer.
        if index + 1 < len(groups) and groups[index + 1][0]["start"] > end:
            end = groups[index + 1][0]["start"]
        step_range = {"start": start, "end": end}
        local_visuals = [row for row in visuals if row["time_valid"] and _overlap(row, step_range)
                         and row["end"] - row["start"] <= max((end - start) * 2, 8)]
        local_shots = [row for row in shots if _overlap(row, step_range)
                       and row["end"] - row["start"] <= max((end - start) * 1.25, 4)]
        sources = [{"text": row["text"], "evidence_refs": row["evidence_refs"], "kind": "transcript"} for row in group]
        sources += [continued for row in group
                    if (continued := _continued_parameter(row.get("_sentence_context"), row["text"], row["evidence_refs"]))]
        sources += [{**line, "kind": "ocr"} for row in local_visuals for line in row["ocr_lines"]]
        sources += [{**item, "kind": "observation"} for row in local_visuals for item in row["observations"]]
        sources += [{**line, "kind": "ocr"} for row in local_shots for line in row.get("ocr_lines", [])]
        buttons, parameters, results = [], [], []
        fields = {key: [] for key in ("step", "button", "parameters", "result")}
        for source in sources:
            text, refs = source["text"], source["evidence_refs"]
            labels = [] if source.get("parameter_only") else _buttons(text)
            if labels:
                buttons.extend(label for label in labels if label not in buttons)
                fields["button"].extend(refs)
            values = [text] if source.get("parameter_only") else _parameters(text)
            if values:
                parameters.extend(value for value in values if value not in parameters)
                fields["parameters"].extend(refs)
            if not source.get("parameter_only") and _RESULT.search(text):
                label = {"transcript": "口述", "ocr": "画面文字", "observation": "观察记录"}[source["kind"]]
                value = f"{label}：{text}"
                if value not in results:
                    results.append(value)
                fields["result"].extend(refs)
        fields["step"] = group[0]["evidence_refs"][:]
        fields = {key: list(dict.fromkeys(value)) for key, value in fields.items()}
        refs = list(dict.fromkeys(ref for source in sources for ref in source["evidence_refs"]))
        local = [issue for row in [*group, *local_visuals, *local_shots] for issue in row["issues"]]
        for key, label, found in (("button", "按钮／菜单", buttons), ("parameters", "参数", parameters), ("result", "结果", results)):
            if not found:
                local.append(_issue(f"step_{key}_unavailable", f"本步骤{label}待核；原片输入证据未明确展示。", group[0]["evidence_refs"]))
        output.append({"index": index + 1, "start": start, "end": end, "step": group[0]["text"],
                       "button": "／".join(dict.fromkeys(buttons)) or PENDING,
                       "parameters": parameters or [PENDING], "result": "；".join(results) or PENDING,
                       "status": "review-required" if local else "evidence-present", "evidence_refs": refs,
                       "field_evidence": fields, "issues": local, "candidate_only": True})
        all_issues.extend(local)
    if not output:
        all_issues.append(_issue("tutorial_steps_unavailable", "已选教程展示，但没有足够操作语句可拆成步骤；保留逐字稿与镜头。", []))
    return output
