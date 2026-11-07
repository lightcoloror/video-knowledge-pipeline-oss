from __future__ import annotations

import csv
import io
import json
from typing import Any


def render_shot_breakdown_csv(result: dict[str, Any]) -> str:
    """Render a flat interchange projection without becoming a new truth source."""

    stream = io.StringIO(newline="")
    fields = [
        "shot_id",
        "start_time",
        "end_time",
        "duration_seconds",
        "shot_type",
        "camera_movement",
        "subject_action",
        "composition",
        "lighting",
        "dialogue_or_narration",
        "screen_text",
        "evidence_ids",
        "unknown_fields",
        "human_confirmed",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for shot in result.get("shots") or []:
        facts = shot.get("facts") or {}
        evidence_ids = sorted(
            {
                str(evidence_id)
                for field in (shot.get("fact_fields") or {}).values()
                if isinstance(field, dict)
                for evidence_id in (field.get("evidence_ids") or [])
                if evidence_id
            }
        )
        writer.writerow(
            {
                "shot_id": shot.get("shot_id", ""),
                "start_time": shot.get("start_time", ""),
                "end_time": shot.get("end_time", ""),
                "duration_seconds": shot.get("duration", 0),
                "shot_type": facts.get("shot_type") or "待确认",
                "camera_movement": facts.get("camera_movement") or "待确认",
                "subject_action": facts.get("subject_action") or "待确认",
                "composition": _text(facts.get("composition")) or "待确认",
                "lighting": _text(facts.get("lighting")) or _text(facts.get("color_profile")) or "待确认",
                "dialogue_or_narration": facts.get("dialogue_or_narration") or "",
                "screen_text": facts.get("screen_text") or "",
                "evidence_ids": json.dumps(evidence_ids, ensure_ascii=False),
                "unknown_fields": json.dumps(shot.get("unknown_fields") or [], ensure_ascii=False),
                "human_confirmed": bool(shot.get("human_confirmed")),
            }
        )
    return stream.getvalue()


def render_shot_breakdown_logseq(result: dict[str, Any]) -> str:
    """Render Logseq-native nested blocks; never opt into collapsed-by-default UI."""

    lines = [
        f"- 逐镜头拉片：{result.get('title') or '未命名视频'}",
        f"  - 状态：{result.get('status', 'unknown')}",
        f"  - 镜头数：{result.get('shot_count', 0)}",
        f"  - 人工复核准备度：{(result.get('readiness') or {}).get('ready_count', 0)}/{result.get('shot_count', 0)}",
        "  - 使用边界：所有内容均为证据绑定候选；正式应用前须人工确认",
        "  - 镜头",
    ]
    for shot in result.get("shots") or []:
        facts = shot.get("facts") or {}
        lines.extend(
            [
                f"    - {shot.get('shot_id', 'shot')} · {shot.get('start_time', '')} - {shot.get('end_time', '')}",
                f"      - 时长：{float(shot.get('duration') or 0):.2f} 秒",
                f"      - 景别：{facts.get('shot_type') or '待确认'}",
                f"      - 主运镜：{facts.get('camera_movement') or '待确认'}",
                f"      - 主体与动作：{facts.get('subject_action') or '待确认'}",
                f"      - 构图：{_text(facts.get('composition')) or '待确认'}",
                f"      - 灯光与色彩：{_text(facts.get('lighting')) or _text(facts.get('color_profile')) or '待确认'}",
                f"      - 对白或旁白：{_line(facts.get('dialogue_or_narration')) or '无'}",
                f"      - 屏幕文字：{_line(facts.get('screen_text')) or '无'}",
                f"      - 未确认字段：{', '.join(shot.get('unknown_fields') or []) or '无'}",
                "      - 证据",
            ]
        )
        evidence_rows = []
        for name, field in (shot.get("fact_fields") or {}).items():
            if not isinstance(field, dict):
                continue
            evidence_ids = [str(value) for value in (field.get("evidence_ids") or []) if value]
            if evidence_ids:
                evidence_rows.append(
                    f"        - {name}：{', '.join(evidence_ids)}（{field.get('status', 'unknown')}）"
                )
        lines.extend(evidence_rows or ["        - 暂无可绑定证据"])
    return "\n".join(lines).rstrip() + "\n"


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or value.get("summary") or "").strip()
    return str(value or "").strip()


def _line(value: Any) -> str:
    return _text(value).replace("\r", " ").replace("\n", " ").strip()
