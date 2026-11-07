from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .models import now_iso
from .production_artifact_gate import evaluate_production_artifact_gate
from .run_artifact_registry import register_bundle_run
from .smart_summary_codex import generate_smart_summary_with_codex, numbered_summary_heading
from .storage import read_json, write_json, write_text_atomic
from .transcript import format_timestamp

SCHEMA = "video_knowledge_pipeline.smart_summary_section_apply.v1"


def apply_smart_summary_sections(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    write: bool = True,
    require_all_sections: bool = False,
) -> dict[str, Any]:
    """Install a staged section rewrite pack as a Codex smart-summary candidate.

    This is the second half of the BiliNote/vsummary-style section workflow:
    humans, Codex, or a future LLM provider fill section-level Markdown in
    `smart-summary-section-todo.json`; this command stitches accepted sections
    into a separate candidate and reuses the existing Codex installation/quality
    gate. It does not call a cloud model.
    """

    root = Path(bundle_dir).expanduser().resolve()
    production_gate = evaluate_production_artifact_gate(
        root,
        artifact_kind="smart_summary",
        write=write,
    )
    if not production_gate.get("formal_generation_allowed"):
        return {
            "schema": SCHEMA,
            "bundle_dir": str(root),
            "input_json": str(input_json or ""),
            "status": "blocked_by_production_artifact_gate",
            "section_count": 0,
            "installed_section_count": 0,
            "missing_section_count": 0,
            "require_all_sections": bool(require_all_sections),
            "smart_summary_codex_path": str(root / "exports" / "smart-summary.codex.md"),
            "quality_status": "blocked_review_required",
            "quality_passed": False,
            "production_artifact_gate": production_gate,
            "operator_boundary": {
                "local_only": True,
                "no_cloud_call": True,
                "formal_filename_not_written": True,
                "purpose": "Prevent a section pack from bypassing the formal Smart Summary quality gate.",
            },
            "updated_at": now_iso(),
        }
    exports = root / "exports"
    if write:
        exports.mkdir(parents=True, exist_ok=True)
    workflow = _read_mapping(exports / "smart-summary-section-workflow.json")
    if not workflow.get("sections"):
        from .smart_summary_section_workflow import build_smart_summary_section_workflow

        workflow = build_smart_summary_section_workflow(root, write=write)
    source_path = _resolve_input(root, input_json)
    payload = _read_mapping(source_path)
    sections = _merge_section_revisions(workflow, payload)
    missing = [row for row in sections if not row.get("final_markdown")]
    ready_sections = [row for row in sections if row.get("final_markdown")]
    status = "ready_to_install" if ready_sections and (not missing or not require_all_sections) else "needs_section_revisions"
    target = exports / "smart-summary.codex.md"
    from .courseware_material import safe_path
    candidate = safe_path(root, "exports/smart-summary-section-candidate.md", exists=False)
    draft_text = _render_codex_summary(root, workflow, sections, source_path=source_path)
    quality: dict[str, Any]
    codex_status: dict[str, Any] | None = None
    if write and status == "ready_to_install":
        write_text_atomic(candidate, draft_text)
        codex_status = generate_smart_summary_with_codex(root, input_md=candidate, write=True)
        quality = codex_status.get("quality") if isinstance(codex_status.get("quality"), dict) else {}
        if not quality.get("passed") or codex_status.get("ok") is False:
            status = str(codex_status.get("status") or "blocked_quality")
    else:
        quality = _missing_quality(root)
    installed = bool(write and quality.get("passed") and (codex_status or {}).get("status") in {"ready", "installed_registry_pending"})
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "input_json": str(source_path),
        "status": status,
        "ok": bool(installed and (codex_status or {}).get("ok", True)),
        "section_count": len(sections),
        "installed_section_count": len(ready_sections) if installed else 0,
        "candidate_section_count": len(ready_sections),
        "missing_section_count": len(missing),
        "require_all_sections": bool(require_all_sections),
        "smart_summary_codex_path": str(target),
        "quality_status": quality.get("status", ""),
        "quality_passed": bool(quality.get("passed")),
        "sections": sections,
        "missing_sections": [
            {"section_id": row.get("section_id"), "title": row.get("title"), "reason": "missing_final_markdown"}
            for row in missing
        ],
        "artifacts": {
            "json": str(exports / "smart-summary-section-apply.json"),
            "markdown": str(exports / "smart-summary-section-apply.md"),
            "codex_summary": str(target),
            "candidate": str(candidate),
            "quality": str(exports / "smart-summary-quality.md"),
        },
        "codex_status": codex_status,
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "does_not_process_media": True,
            "purpose": "Install staged section rewrites as a Codex/LLM smart-summary candidate and run the existing quality gate.",
        },
        "updated_at": now_iso(),
    }
    if write:
        write_json(exports / "smart-summary-section-apply.json", result)
        (exports / "smart-summary-section-apply.md").write_text(_render_report(result), encoding="utf-8")
        write_json(root / "mcp-smart-summary-section-apply.args.json", {"bundle_dir": str(root), "input_json": str(source_path), "write": True, "require_all_sections": bool(require_all_sections)})
        manifest_path = root / "manifest.json"
        manifest = _read_mapping(manifest_path)
        manifest["smart_summary_section_apply"] = "exports/smart-summary-section-apply.json"
        manifest["smart_summary_section_apply_markdown"] = "exports/smart-summary-section-apply.md"
        manifest["mcp_smart_summary_section_apply_args"] = "mcp-smart-summary-section-apply.args.json"
        write_json(manifest_path, manifest)
        result["run_artifact"] = _register_run(root, result, source_path=source_path, write=True)
        write_json(exports / "smart-summary-section-apply.json", result)
        (exports / "smart-summary-section-apply.md").write_text(_render_report(result), encoding="utf-8")
    return result


def _resolve_input(root: Path, input_json: str | Path | None) -> Path:
    if input_json:
        path = Path(input_json).expanduser()
        if path.is_absolute():
            return path.resolve()
        if path.exists():
            return path.resolve()
        return (root / path).resolve()
    return (root / "exports" / "smart-summary-section-todo.json").resolve()


def _merge_section_revisions(workflow: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    revisions = _revision_map(payload)
    sections: list[dict[str, Any]] = []
    for section in workflow.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("section_id") or "").strip()
        revision = revisions.get(section_id, {})
        markdown = _revision_markdown(revision)
        status = "installed" if markdown else "missing_revision"
        sections.append(
            {
                "section_id": section_id,
                "chapter_index": section.get("chapter_index"),
                "title": section.get("title"),
                "start": float(section.get("start") or 0.0),
                "end": float(section.get("end") or 0.0),
                "start_time": section.get("start_time") or format_timestamp(float(section.get("start") or 0.0)),
                "end_time": section.get("end_time") or format_timestamp(float(section.get("end") or 0.0)),
                "source_status": section.get("status"),
                "apply_status": status,
                "reasons": section.get("reasons") or [],
                "final_markdown": markdown,
                "revision_status": revision.get("status") if isinstance(revision, dict) else "",
            }
        )
    return sections


def _revision_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else None
    if rows is None:
        rows = payload.get("sections") if isinstance(payload.get("sections"), list) else None
    if rows is None:
        rows = payload.get("revised_sections") if isinstance(payload.get("revised_sections"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        section_id = str(row.get("section_id") or row.get("id") or "").strip()
        if not section_id:
            continue
        out[section_id] = row
    return out


def _revision_markdown(row: dict[str, Any]) -> str:
    for key in ("final_markdown", "revised_markdown", "draft_markdown", "markdown", "content"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _render_codex_summary(root: Path, workflow: dict[str, Any], sections: list[dict[str, Any]], *, source_path: Path) -> str:
    manifest = _read_mapping(root / "manifest.json")
    title = str(workflow.get("title") or manifest.get("title") or root.name).strip()
    ready = [row for row in sections if row.get("final_markdown")]
    duration = max((float(row.get("end") or 0.0) for row in sections), default=0.0)
    overview = _overview(title, ready)
    has_online_llm_sections = any(str(row.get("revision_status") or "") == "llm_rewritten" for row in ready)
    if has_online_llm_sections:
        generation_note = "生成方式：`online_llm_section_rewrite` + `section_staged_apply`。本稿包含在线 API 章节改写；未通过模型输出质量门禁的章节不会被伪装成在线改写完成。"
    else:
        generation_note = "生成方式：`codex_first_llm_substitute` + `section_staged_apply`。本稿由章节级修订稿拼装，并复用 VKP 既有 smart-summary 质量门禁；没有在此步骤调用云端模型。"
    lines = [
        f"# {title} - 智能总结",
        "",
        generation_note,
        "",
        numbered_summary_heading("基本信息"),
        "",
        f"- 视频名：{title}",
        f"- 时长：`{format_timestamp(duration)}`",
        f"- 处理时间：`{now_iso()}`",
        f"- 来源路径：`{manifest.get('media_path') or manifest.get('source_path') or '(unknown)'}`",
        f"- 章节修订来源：`{source_path}`",
        "- 视觉证据状态：视觉证据未执行/待复核的章节仍必须以原视频、OCR/ebook、多模态结果和人工审核为准。",
        "",
        numbered_summary_heading("一句话概览"),
        "",
        overview,
        "",
        numbered_summary_heading("核心主题 / 课程主线", number="3"),
        "",
        f"- 课程主线：围绕《{title}》按时间顺序整理讲师的关键判断、方法步骤、案例解释和后续行动。",
        "- 章节来源：每节来自 `smart-summary-section-workflow` 的 section state；未修订章节不会被伪装成已完成总结。",
        "- 证据边界：本稿可读性来自章节修订，事实准确性仍以 bundle 内 transcript、timeline、视觉证据和 review 记录为准。",
        "",
        numbered_summary_heading("分段总结"),
        "",
    ]
    if ready:
        for index, row in enumerate(ready, start=1):
            lines.extend(_section_markdown(row, number=f"4.1.{index}"))
    else:
        lines.append("（没有可安装的章节修订稿。请先填写 `exports/smart-summary-section-todo.json` 的 `draft_markdown` / `revised_markdown` / `final_markdown`。）")
    lines.extend(["", numbered_summary_heading("关键观点 / 方法论", number="5"), ""])
    lines.extend(_extract_bullets(ready, fallback="- 暂无已修订章节可抽取关键观点；请先完成章节修订。", mode="points", numbering_prefix="5.1"))
    lines.extend(["", numbered_summary_heading("可执行动作清单"), ""])
    lines.extend(_extract_bullets(ready, fallback="- 暂无已修订章节可抽取行动项；请先完成章节修订。", mode="actions", numbering_prefix="7.1"))
    lines.extend(["", numbered_summary_heading("高频话术 / 可复用表达", number="8"), ""])
    lines.extend(_extract_bullets(ready, fallback="- 暂无已修订章节可抽取话术；请先完成章节修订。", mode="expressions", numbering_prefix="8.1"))
    lines.extend(["", numbered_summary_heading("待复核点 / 低置信内容", number="9"), ""])
    missing = [row for row in sections if not row.get("final_markdown")]
    if missing:
        lines.append("- 以下章节还没有修订稿，不能视为最终智能总结覆盖：" + "、".join(str(row.get("section_id")) for row in missing) + "。")
    lines.append("- 视觉证据未执行/待复核：章节修订若涉及屏幕文字、图表、公式、软件操作或画面状态，仍需回到 evidence frame / OCR / 多模态 / 人工审核核对。")
    return "\n".join(lines).rstrip() + "\n"


def _section_title(row: dict[str, Any], raw_markdown: str) -> str:
    fallback = str(row.get("title") or row.get("section_id") or "chapter").strip()
    if not fallback.lower().startswith("schema:"):
        return fallback
    headings = re.findall(r"(?m)^#{1,4}\s+(.+)$", str(raw_markdown or ""))
    for heading in reversed(headings):
        value = re.sub(r"`[^`]*`", "", heading).strip()
        value = re.sub(r"\s*[\(\uff08][^\)\uff09]*[\)\uff09]\s*$", "", value).strip()
        if value and not value.lower().startswith("schema:"):
            return value
    return "chapter points"


def _section_markdown(row: dict[str, Any], *, number: str = "") -> list[str]:
    raw_markdown = str(row.get("final_markdown") or "")
    title = _section_title(row, raw_markdown)
    start = str(row.get("start_time") or format_timestamp(float(row.get("start") or 0.0)))
    end = str(row.get("end_time") or format_timestamp(float(row.get("end") or 0.0)))
    body = _normalize_section_body(raw_markdown)
    prefix = f"{number} " if number else ""
    lines = [f"### {prefix}`{start} - {end}` {title}", ""]
    lines.append(body or "（该章节修订为空。）")
    lines.append("")
    return lines


def _normalize_section_body(text: str) -> str:
    value = str(text or "").strip()
    # Every staged section may already have its own heading. Strip it first so
    # a provider preamble that follows can be recognized deterministically.
    value = re.sub(r"^#{1,4}\s+.*$", "", value, count=1, flags=re.MULTILINE).strip()
    value, wrapper_count = re.subn(
        r"^\s*好的[，,]?\s*这是根据.*?重写的\s*`?smart-summary\.md`?\s*章节[。.!！]?\s*",
        "",
        value,
        flags=re.DOTALL,
    )
    value, generic_wrapper_count = re.subn(
        r"^\s*(?:\u597d\u7684[，,]?\s*)?(?:\u8fd9\u662f|\u4ee5\u4e0b\u662f)(?:\u6839\u636e|\u6309|\u57fa\u4e8e).*?(?:\u91cd\u5199|\u6574\u7406|\u751f\u6210)(?:\u7684)?\s*`?(?:smart-summary(?:\.md)?|\u667a\u80fd\u603b\u7ed3)`?(?:\s*(?:\u7ae0\u8282|\u5c0f\u8282|\u5185\u5bb9))?[\u3002.!\uFF01:\uFF1A]?\s*",
        "",
        value,
        flags=re.DOTALL,
    )
    wrapper_count += generic_wrapper_count
    if wrapper_count:
        # The known provider wrapper is followed by a second generated heading;
        # VKP renders its canonical time/title heading itself.
        value = re.sub(r"^#{1,4}\s+.*$", "", value, count=1, flags=re.MULTILINE).strip()
    value = re.sub(r"(?m)^\s*[-*]\s*(?:\*\*)?\u89c6\u89c9\u8bc1\u636e\u8fb9\u754c(?:\*\*)?[\uff1a:].*(?:\n|$)", "", value).strip()
    return value

def _overview(title: str, sections: list[dict[str, Any]]) -> str:
    if not sections:
        return "本视频尚未完成章节级修订，当前只生成了待处理框架，不能作为最终智能总结使用。"
    first = _compact_text(sections[0].get("final_markdown"), 42)
    last = _compact_text(sections[-1].get("final_markdown"), 42)
    return f"这份总结围绕《{title}》的完整章节修订展开，从“{first}”推进到“{last}”，重点保留可复习的课程主线、关键判断和行动线索。"


def _extract_bullets(sections: list[dict[str, Any]], *, fallback: str, mode: str, numbering_prefix: str = "") -> list[str]:
    keywords = {
        "points": ("关键", "核心", "原则", "判断", "方法", "因为", "所以"),
        "actions": ("要", "需要", "先", "再", "复盘", "记录", "确认", "整理", "避免"),
        "expressions": ("话术", "可以说", "你可以", "客户", "对方", "怎么", "为什么"),
    }.get(mode, ())
    per_section: list[list[str]] = []
    for section in sections:
        start = str(section.get("start_time") or "")
        candidates: list[str] = []
        for sentence in _sentences(str(section.get("final_markdown") or "")):
            if keywords and not any(key in sentence for key in keywords):
                continue
            candidates.append(f"- `{start}` {_compact_text(sentence, 120)}")
            if len(candidates) >= 3:
                break
        if candidates:
            per_section.append(candidates)
    # These lists are navigation aids derived from the full section prose above.
    # Cap them conservatively so a long lecture does not repeat the same claims in
    # three different summary sections and fail the independent compression gate.
    max_rows = {"points": 6, "actions": 7, "expressions": 7}.get(mode, 6)
    rows: list[str] = []
    while len(rows) < max_rows and any(per_section):
        for candidates in per_section:
            if not candidates:
                continue
            row = candidates.pop(0)
            if numbering_prefix:
                rows.append(f"- {numbering_prefix}.{len(rows) + 1} " + row[2:])
            else:
                rows.append(row)
            if len(rows) >= max_rows:
                break
    return rows or [fallback]


def _sentences(text: str) -> list[str]:
    value = re.sub(r"[`*_>#\-]+", " ", str(text or ""))
    parts = re.split(r"[。！？!?；;\n]+", value)
    return [part.strip(" ，,、：:") for part in parts if len(part.strip()) >= 8]


def _compact_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"[`*_>#\-]+", "", text).strip(" ，,、。")
    if not text:
        return "章节修订内容待补齐"
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip(" ，,、；;") + "…"


def _missing_quality(root: Path) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.smart_summary_quality.v1",
        "bundle_dir": str(root),
        "status": "missing_codex_summary",
        "passed": False,
        "checks": [{"key": "codex_final", "passed": False, "detail": "smart-summary.codex.md is missing or sections are incomplete"}],
        "checked_at": now_iso(),
    }


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Smart Summary Section Apply",
        "",
        f"- Bundle: `{result.get('bundle_dir')}`",
        f"- Status: `{result.get('status')}`",
        f"- Installed sections: `{result.get('installed_section_count')}` / `{result.get('section_count')}`",
        f"- Quality: `{result.get('quality_status')}`",
        f"- Codex summary: `{result.get('smart_summary_codex_path')}`",
        "",
        "| Section | Status | Title |",
        "| --- | --- | --- |",
    ]
    for row in result.get("sections") or []:
        if not isinstance(row, dict):
            continue
        lines.append(f"| `{row.get('section_id')}` | `{row.get('apply_status')}` | {_md(str(row.get('title') or ''))} |")
    missing = result.get("missing_sections") if isinstance(result.get("missing_sections"), list) else []
    if missing:
        lines.extend(["", "## Missing Sections", ""])
        for row in missing:
            lines.append(f"- `{row.get('section_id')}` {row.get('title')}")
    lines.extend(["", "## Next Actions", ""])
    for action in _next_actions(result):
        lines.append(f"- {action}")
    return "\n".join(lines).rstrip() + "\n"


def _register_run(root: Path, result: dict[str, Any], *, source_path: Path, write: bool) -> dict[str, Any]:
    failed_items = [
        {"id": row.get("section_id"), "reason": row.get("reason") or "missing_final_markdown", "detail": row.get("title", "")}
        for row in result.get("missing_sections") or []
        if isinstance(row, dict)
    ]
    if not result.get("quality_passed") and not failed_items:
        failed_items.append({"id": "quality", "reason": str(result.get("quality_status") or "quality_failed"), "detail": "smart-summary quality gate did not pass"})
    if result.get("status") == "needs_section_revisions":
        status = "needs_input"
    elif result.get("quality_passed") and result.get("ok"):
        status = "completed"
    else:
        status = "needs_retry"
    return register_bundle_run(
        root,
        run_type="smart_summary_section_apply",
        run_id="smart-summary-section-apply",
        status=status,
        title="Smart summary section apply",
        summary=f"Installed {result.get('installed_section_count', 0)} of {result.get('section_count', 0)} smart-summary sections; quality={result.get('quality_status', '')}.",
        inputs={"input_json": str(source_path), "workflow_json": str(root / "exports" / "smart-summary-section-workflow.json")},
        parameters={"require_all_sections": bool(result.get("require_all_sections")), "write": bool(write)},
        artifacts=[
            {"key": "section_apply_json", "path": str(root / "exports" / "smart-summary-section-apply.json")},
            {"key": "section_apply_markdown", "path": str(root / "exports" / "smart-summary-section-apply.md")},
            {"key": "smart_summary_codex", "path": str(root / "exports" / "smart-summary.codex.md")},
            {"key": "quality", "path": str(root / "exports" / "smart-summary-quality.md")},
            {"key": "mcp_args", "path": str(root / "mcp-smart-summary-section-apply.args.json")},
        ],
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 smart-summary-section-apply {_ps_quote(str(root))} --input-json {_ps_quote(str(source_path))}",
        next_actions=_next_actions(result),
        operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {},
        write=write,
    )


def _next_actions(result: dict[str, Any]) -> list[str]:
    if result.get("status") == "needs_section_revisions":
        return [
            "Fill draft_markdown/revised_markdown/final_markdown in exports/smart-summary-section-todo.json.",
            "Rerun smart-summary-section-apply after section revisions are available.",
        ]
    if not result.get("quality_passed"):
        return [
            "Revise exports/smart-summary.codex.md until smart-summary-quality passes.",
            "Rerun generate-smart-summary-with-codex or smart-summary-section-apply.",
        ]
    return ["Run export-knowledge-note to install the final smart-summary.md into the human-readable export set."]


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}



def _md(value: str) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")
