from __future__ import annotations

import csv
import io
from datetime import date
from pathlib import Path
from typing import Any

from .lecture_study_index import generate_lecture_study_cards, generate_lecture_study_index
from .models import now_iso
from .storage import ensure_project_dirs, read_json, write_json


REVIEW_QUEUE_SCHEMA = "lecture_review_queue.v1"


def build_lecture_review_queue(root: str | Path, *, today: str | None = None) -> dict[str, Any]:
    """Build a review queue from lecture-study-cards or lecture-package."""
    paths = ensure_project_dirs(root)
    package_path = paths["lecture_packages"] / "lecture-package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"lecture package not found: {package_path}")
    package = read_json(package_path)
    if not isinstance(package, dict):
        raise ValueError("lecture package must be a JSON object")
    return write_lecture_review_queue(paths, package, today=today)


def write_lecture_review_queue(paths: dict[str, Path], package: dict[str, Any], *, today: str | None = None) -> dict[str, Any]:
    study_index = generate_lecture_study_index(package)
    study_cards = generate_lecture_study_cards(study_index)
    queue = generate_lecture_review_queue(study_cards, today=today)
    json_path = paths["lecture_packages"] / "lecture-review-queue.json"
    markdown_path = paths["notes"] / "lecture-review-queue.md"
    tasks_path = paths["notes"] / "lecture-review-tasks.md"
    anki_path = paths["lecture_packages"] / "lecture-review-anki.csv"
    write_json(json_path, queue)
    markdown_path.write_text(render_lecture_review_queue_markdown(queue), encoding="utf-8")
    tasks_path.write_text(render_lecture_review_tasks_markdown(queue), encoding="utf-8")
    anki_path.write_text(render_lecture_review_anki_csv(queue), encoding="utf-8-sig")
    return {
        "review_queue_path": str(json_path),
        "review_queue_markdown_path": str(markdown_path),
        "review_tasks_markdown_path": str(tasks_path),
        "review_anki_csv_path": str(anki_path),
        "review_queue": queue,
    }


def generate_lecture_review_queue(study_cards: dict[str, Any], *, today: str | None = None) -> dict[str, Any]:
    today_value = _date_or_today(today)
    cards = study_cards.get("cards") if isinstance(study_cards.get("cards"), list) else []
    items = [_queue_item(card, today_value) for card in cards if isinstance(card, dict)]
    items.sort(key=_queue_sort_key)
    buckets = {
        "due_now": [item for item in items if item["queue_status"] == "due_now"],
        "ready_unscheduled": [item for item in items if item["queue_status"] == "ready_unscheduled"],
        "scheduled": [item for item in items if item["queue_status"] == "scheduled"],
        "draft": [item for item in items if item["queue_status"] == "draft"],
        "needs_review": [item for item in items if item["queue_status"] == "needs_review"],
    }
    return {
        "schema": REVIEW_QUEUE_SCHEMA,
        "title": str(study_cards.get("title") or "Lecture Review Queue"),
        "created_at": now_iso(),
        "today": today_value.isoformat(),
        "source_cards_schema": str(study_cards.get("schema") or ""),
        "item_count": len(items),
        "summary": {key: len(value) for key, value in buckets.items()},
        "buckets": buckets,
        "items": items,
    }


def render_lecture_review_queue_markdown(queue: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: lecture-review-queue",
        f'title: "{queue.get("title", "Lecture Review Queue")}"',
        "tags: [lecture-video, review-queue, study-cards, knowledge-package]",
        f'created: "{queue.get("created_at", now_iso())}"',
        f'today: "{queue.get("today", "")}"',
        "---",
        "",
        f"# 复习队列：{queue.get('title', 'Lecture Review Queue')}",
        "",
        "> 这个队列来自人工整理后的学习卡片字段，不重新摘要视频内容。优先处理到期、已就绪、混淆点多或仍需人工确认的卡片。",
        "",
        "## 总览",
        "",
        f"- 队列日期：{queue.get('today', '')}",
        f"- 卡片数量：{queue.get('item_count', 0)}",
    ]
    summary = queue.get("summary") if isinstance(queue.get("summary"), dict) else {}
    for key in ("due_now", "ready_unscheduled", "scheduled", "draft", "needs_review"):
        lines.append(f"- `{key}`: {summary.get(key, 0)}")
    buckets = queue.get("buckets") if isinstance(queue.get("buckets"), dict) else {}
    for key, title in [
        ("due_now", "今日到期"),
        ("ready_unscheduled", "已整理但未排期"),
        ("scheduled", "未来复习"),
        ("draft", "草稿 / 待整理"),
        ("needs_review", "仍需人工核对"),
    ]:
        lines.extend(["", f"## {title}", ""])
        entries = buckets.get(key) if isinstance(buckets.get(key), list) else []
        if not entries:
            lines.append("暂无。")
            continue
        for item in entries:
            lines.extend(_queue_item_markdown(item))
    return "\n".join(lines).rstrip() + "\n"


def render_lecture_review_tasks_markdown(queue: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: lecture-review-tasks",
        f'title: "{queue.get("title", "Lecture Review Tasks")}"',
        "tags: [lecture-video, review-tasks, obsidian-tasks]",
        f'created: "{queue.get("created_at", now_iso())}"',
        f'today: "{queue.get("today", "")}"',
        "---",
        "",
        f"# 复习任务：{queue.get('title', 'Lecture Review Tasks')}",
        "",
        "> 面向 Obsidian Tasks 的任务视图。任务内容来自学习卡片人工字段，不重新摘要课程内容。",
        "",
    ]
    task_count = 0
    for item in queue.get("items") or []:
        if not isinstance(item, dict) or item.get("queue_status") == "draft":
            continue
        due = str(item.get("due_date") or item.get("next_review") or queue.get("today") or "").strip()
        due_part = f" 📅 {due[:10]}" if due else ""
        tags = _task_tags(item)
        title = _task_title(item)
        lines.append(f"- [ ] {title}{due_part} {tags}".rstrip())
        lines.append(f"  - 来源：{item.get('time_label', '')} / {', '.join(item.get('source_segment_ids') or [])}")
        if item.get("questions"):
            lines.append(f"  - 问题：{'; '.join(item.get('questions') or [])}")
        if item.get("confusion_points"):
            lines.append(f"  - 混淆点：{'; '.join(item.get('confusion_points') or [])}")
        if item.get("links"):
            lines.append(f"  - 关联：{'; '.join(item.get('links') or [])}")
        task_count += 1
    if task_count == 0:
        lines.append("暂无需要导入 Obsidian Tasks 的卡片。")
    return "\n".join(lines).rstrip() + "\n"


def render_lecture_review_anki_csv(queue: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Front", "Back", "Tags", "Source", "NextReview", "CardId"])
    for item in queue.get("items") or []:
        if not isinstance(item, dict) or item.get("queue_status") == "draft":
            continue
        front = _anki_front(item)
        back = _anki_back(item)
        tags = " ".join(_anki_tags(item))
        source = f"{item.get('time_label', '')} {'; '.join(item.get('source_segment_ids') or [])}".strip()
        writer.writerow([front, back, tags, source, item.get("next_review", ""), item.get("id", "")])
    return buffer.getvalue()


def _queue_item(card: dict[str, Any], today: date) -> dict[str, Any]:
    fields = card.get("human_fields") if isinstance(card.get("human_fields"), dict) else {}
    next_review = str(fields.get("next_review") or "").strip()
    review_state = str(fields.get("review_state") or "").strip()
    mastery_level = str(fields.get("mastery_level") or "").strip()
    final_card = str(fields.get("final_card") or "").strip()
    questions = [str(value) for value in fields.get("questions") or [] if str(value)]
    confusion_points = [str(value) for value in fields.get("confusion_points") or [] if str(value)]
    due_date = _parse_date(next_review)
    queue_status = _queue_status(
        review_state=review_state,
        final_card=final_card,
        questions=questions,
        next_review=next_review,
        due_date=due_date,
        today=today,
        needs_human_review=bool(card.get("needs_human_review")),
    )
    priority = _priority_score(
        queue_status=queue_status,
        mastery_level=mastery_level,
        confusion_points=confusion_points,
        needs_human_review=bool(card.get("needs_human_review")),
    )
    return {
        "id": str(card.get("id") or ""),
        "timeline_index": int(card.get("timeline_index") or 0),
        "title": str(card.get("title") or ""),
        "card_type": str(fields.get("card_kind") or card.get("card_type") or "evidence"),
        "time_label": str(card.get("time_label") or ""),
        "source_segment_ids": card.get("source_segment_ids") or [],
        "queue_status": queue_status,
        "priority": priority,
        "review_state": review_state,
        "mastery_level": mastery_level,
        "next_review": next_review,
        "due_date": due_date.isoformat() if due_date else "",
        "final_card": final_card,
        "questions": questions,
        "links": fields.get("links") or [],
        "confusion_points": confusion_points,
        "needs_human_review": bool(card.get("needs_human_review")),
        "review_status": str(card.get("review_status") or "pending"),
        "quality_issues": card.get("quality_issues") or [],
    }


def _queue_status(
    *,
    review_state: str,
    final_card: str,
    questions: list[str],
    next_review: str,
    due_date: date | None,
    today: date,
    needs_human_review: bool,
) -> str:
    state = review_state.lower()
    if state in {"draft", "blocked"} or not final_card:
        return "draft"
    if needs_human_review and state not in {"ready", "scheduled", "done"}:
        return "needs_review"
    if due_date and due_date <= today:
        return "due_now"
    if due_date or next_review or state == "scheduled":
        return "scheduled"
    if state == "ready" or questions:
        return "ready_unscheduled"
    return "needs_review"


def _priority_score(*, queue_status: str, mastery_level: str, confusion_points: list[str], needs_human_review: bool) -> int:
    score = {
        "due_now": 100,
        "ready_unscheduled": 80,
        "needs_review": 60,
        "draft": 40,
        "scheduled": 20,
    }.get(queue_status, 10)
    if mastery_level.lower() in {"new", "learning", "weak"}:
        score += 10
    if confusion_points:
        score += min(len(confusion_points) * 5, 20)
    if needs_human_review:
        score += 3
    return score


def _queue_sort_key(item: dict[str, Any]) -> tuple[int, str, int]:
    due = str(item.get("due_date") or "9999-12-31")
    return (-int(item.get("priority") or 0), due, int(item.get("timeline_index") or 0))


def _queue_item_markdown(item: dict[str, Any]) -> list[str]:
    lines = [
        f"### {item.get('id', '')} {item.get('title', '')}",
        "",
        f"- 状态：`{item.get('queue_status', '')}`",
        f"- 优先级：{item.get('priority', 0)}",
        f"- 卡片类型：`{item.get('card_type', 'evidence')}`",
        f"- 掌握度：`{item.get('mastery_level', '')}`",
        f"- 复习状态：`{item.get('review_state', '')}`",
        f"- 下次复习：{item.get('next_review', '')}",
        f"- 时间：{item.get('time_label', '')}",
        f"- 来源片段：{', '.join(item.get('source_segment_ids') or [])}",
        f"- 仍需人工核对：{bool(item.get('needs_human_review'))}",
        "",
        "最终卡片：",
        "",
        str(item.get("final_card") or "> "),
        "",
    ]
    if item.get("questions"):
        lines.extend(["复习问题：", ""])
        lines.extend([f"- {question}" for question in item.get("questions") or []])
        lines.append("")
    if item.get("confusion_points"):
        lines.extend(["混淆点 / 错因：", ""])
        lines.extend([f"- {point}" for point in item.get("confusion_points") or []])
        lines.append("")
    if item.get("links"):
        lines.extend(["关联链接：", ""])
        lines.extend([f"- {link}" for link in item.get("links") or []])
        lines.append("")
    if item.get("quality_issues"):
        lines.extend(["质量缺口：", ""])
        lines.extend([f"- {issue}" for issue in item.get("quality_issues") or []])
        lines.append("")
    return lines


def _task_title(item: dict[str, Any]) -> str:
    card = " ".join(str(item.get("final_card") or item.get("title") or item.get("id") or "").split())
    if len(card) > 72:
        card = card[:69].rstrip() + "..."
    return f"复习 {item.get('id', '')}: {card}"


def _task_tags(item: dict[str, Any]) -> str:
    values = ["#lecture-review", f"#lecture-review/{item.get('queue_status', 'review')}"]
    card_type = str(item.get("card_type") or "").strip()
    if card_type:
        values.append(f"#lecture-card/{_tag_slug(card_type)}")
    mastery = str(item.get("mastery_level") or "").strip()
    if mastery:
        values.append(f"#mastery/{_tag_slug(mastery)}")
    return " ".join(values)


def _anki_front(item: dict[str, Any]) -> str:
    questions = [str(value) for value in item.get("questions") or [] if str(value).strip()]
    if questions:
        return "\n".join(questions)
    title = str(item.get("title") or item.get("id") or "").strip()
    return title or "请回忆这一段课程内容。"


def _anki_back(item: dict[str, Any]) -> str:
    parts = [str(item.get("final_card") or "").strip()]
    if item.get("confusion_points"):
        parts.extend(["", "混淆点 / 错因："])
        parts.extend([f"- {point}" for point in item.get("confusion_points") or []])
    if item.get("links"):
        parts.extend(["", "关联链接："])
        parts.extend([f"- {link}" for link in item.get("links") or []])
    parts.extend(["", f"来源：{item.get('time_label', '')} / {', '.join(item.get('source_segment_ids') or [])}"])
    return "\n".join(part for part in parts if part is not None).strip()


def _anki_tags(item: dict[str, Any]) -> list[str]:
    values = ["lecture_review", str(item.get("queue_status") or "review")]
    for key in ("card_type", "mastery_level", "review_state"):
        value = str(item.get(key) or "").strip()
        if value:
            values.append(_tag_slug(value))
    return values


def _tag_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower()).strip("_") or "unknown"


def _date_or_today(value: str | None) -> date:
    parsed = _parse_date(value or "")
    return parsed or date.today()


def _parse_date(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None
