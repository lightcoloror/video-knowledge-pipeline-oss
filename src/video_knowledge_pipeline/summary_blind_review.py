from __future__ import annotations

import hashlib
import html
import json
import re
import statistics
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from .models import now_iso
from .storage import read_json, write_json


PRIVATE_SCHEMA = "video_knowledge_pipeline.summary_blind_review_private.v1"
PUBLIC_SCHEMA = "video_knowledge_pipeline.summary_blind_review_public.v1"
SCORES_SCHEMA = "video_knowledge_pipeline.summary_blind_review_scores.v1"
RESULT_SCHEMA = "video_knowledge_pipeline.summary_blind_review_result.v1"
CRITERIA = [
    ("coverage", "内容覆盖"),
    ("accuracy", "事实与术语准确"),
    ("structure", "结构清晰"),
    ("actionability", "可执行性"),
    ("readability", "阅读自然度"),
    ("evidence_discipline", "证据边界"),
]


def build_summary_blind_review(
    quality_manifest_json: str | Path,
    *,
    output_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(quality_manifest_json).expanduser().resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("quality benchmark manifest must be a JSON object")
    out = Path(output_dir).expanduser().resolve() if output_dir else manifest_path.parent
    review = manifest.get("summary_blind_review") if isinstance(manifest.get("summary_blind_review"), dict) else {}
    public_items: list[dict[str, Any]] = []
    private_items: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, row in enumerate(review.get("items") or []):
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("item_id") or f"summary-{index + 1:02d}")
        role_paths = _resolved_review_paths(row)
        for role, path in role_paths.items():
            if path is not None:
                row[f"{role}_summary_path"] = str(path)
        row["external_reference_is_evaluation_only"] = bool(role_paths.get("reference"))
        excluded_roles: list[dict[str, Any]] = []
        for role in ("baseline", "candidate"):
            source = role_paths.get(role)
            reason = _non_llm_draft_reason(source)
            if reason:
                excluded_roles.append({"role": role, "path": str(source), "reason": reason})
                role_paths[role] = None
        required_roles = ["candidate"] + (["reference"] if row.get("reference_summary_path") else [])
        missing_roles = [role for role in required_roles if role_paths.get(role) is None]
        if missing_roles:
            missing.append({"item_id": item_id, "missing_roles": missing_roles, "excluded_roles": excluded_roles})
            continue
        roles = [role for role in ("baseline", "candidate", "reference") if role_paths.get(role)]
        roles.sort(key=lambda role: hashlib.sha256(f"{item_id}:{role}".encode("utf-8")).hexdigest())
        labels = {chr(ord("A") + offset): role for offset, role in enumerate(roles)}
        public_versions = []
        private_labels: dict[str, Any] = {}
        for label, role in labels.items():
            source = role_paths[role]
            assert source is not None
            public_versions.append({"label": label, "content": _blind_text(source.read_text(encoding="utf-8"))})
            private_labels[label] = {"role": role, "path": str(source)}
        video_title = _review_video_title(row, role_paths, index=index)
        public_items.append(
            {
                "item_id": item_id,
                "video_title": video_title,
                "display_title": f"视频 {index + 1} | {video_title}",
                "versions": public_versions,
                "criteria": [{"key": key, "label": label} for key, label in CRITERIA],
                "review_status": "todo",
                "excluded_non_llm_versions": len(excluded_roles),
            }
        )
        private_items.append(
            {
                "item_id": item_id,
                "bundle_dir": str(row.get("bundle_dir") or ""),
                "labels": private_labels,
                "excluded_roles": excluded_roles,
            }
        )
    public = {
        "schema": PUBLIC_SCHEMA,
        "quality_manifest_json": "hidden_in_public_review",
        "criteria": [{"key": key, "label": label} for key, label in CRITERIA],
        "items": public_items,
        "operator_boundary": {
            "anonymous_labels_only": True,
            "evaluation_only": True,
            "does_not_modify_summaries": True,
            "get_note_reference_never_enters_production_evidence": True,
            "non_llm_rule_drafts_excluded": True,
        },
        "updated_at": now_iso(),
    }
    private = {
        "schema": PRIVATE_SCHEMA,
        "quality_manifest_json": str(manifest_path),
        "public_review_json": str(out / "summary-blind-review.json"),
        "items": private_items,
        "missing": missing,
        "updated_at": now_iso(),
    }
    result = {
        "schema": "video_knowledge_pipeline.summary_blind_review_build.v1",
        "status": "ready" if public_items and not missing else ("partial" if public_items else "blocked"),
        "ok": bool(public_items),
        "item_count": len(public_items),
        "missing": missing,
        "artifacts": {
            "html": str(out / "summary-blind-review.html"),
            "public_json": str(out / "summary-blind-review.json"),
            "private_json": str(out / "summary-blind-review.private.json"),
            "scores_filename": "summary-blind-review-scores.json",
        },
        "operator_boundary": public["operator_boundary"],
        "updated_at": now_iso(),
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        write_json(out / "summary-blind-review.json", public)
        write_json(out / "summary-blind-review.private.json", private)
        (out / "summary-blind-review.html").write_text(_render_review_html(public), encoding="utf-8")
        write_json(out / "summary-blind-review-build.json", result)
    return result


def apply_summary_blind_review(
    private_json: str | Path,
    scores_json: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    private_path = Path(private_json).expanduser().resolve()
    scores_path = Path(scores_json).expanduser().resolve()
    private = read_json(private_path)
    scores = read_json(scores_path)
    if not isinstance(private, dict) or private.get("schema") != PRIVATE_SCHEMA:
        raise ValueError("invalid private summary blind review manifest")
    if not isinstance(scores, dict) or scores.get("schema") != SCORES_SCHEMA:
        raise ValueError("invalid summary blind review scores")
    quality_manifest_path = Path(str(private.get("quality_manifest_json") or "")).expanduser().resolve()
    quality_manifest = read_json(quality_manifest_path)
    if not isinstance(quality_manifest, dict):
        raise ValueError("quality benchmark manifest must be a JSON object")
    review = quality_manifest.get("summary_blind_review") if isinstance(quality_manifest.get("summary_blind_review"), dict) else {}
    review_items = [row for row in review.get("items") or [] if isinstance(row, dict)]
    score_by_id = {str(row.get("item_id") or ""): row for row in scores.get("items") or [] if isinstance(row, dict)}
    completed = 0
    result_items: list[dict[str, Any]] = []
    for private_item in private.get("items") or []:
        if not isinstance(private_item, dict):
            continue
        item_id = str(private_item.get("item_id") or "")
        score_row = score_by_id.get(item_id) or {}
        label_scores = score_row.get("label_scores") if isinstance(score_row.get("label_scores"), dict) else {}
        role_scores: dict[str, float] = {}
        role_dimensions: dict[str, dict[str, float]] = {}
        for label, mapping in (private_item.get("labels") or {}).items():
            if not isinstance(mapping, dict):
                continue
            role = str(mapping.get("role") or "")
            dimensions = _valid_dimension_scores(label_scores.get(label))
            if dimensions:
                role_dimensions[role] = dimensions
                role_scores[role] = round(statistics.mean(dimensions.values()), 6)
        target = next((row for row in review_items if str(row.get("item_id") or "") == item_id), None)
        if target is None:
            target = next((row for row in review_items if str(row.get("bundle_dir") or "") == str(private_item.get("bundle_dir") or "")), None)
        available_roles = {
            str(mapping.get("role") or "")
            for mapping in (private_item.get("labels") or {}).values()
            if isinstance(mapping, dict)
        }
        required = {"candidate"}
        if "baseline" in available_roles:
            required.add("baseline")
        if "reference" in available_roles:
            required.add("reference")
        item_complete = bool(target) and required.issubset(role_scores)
        if target is not None:
            for role in ("baseline", "candidate", "reference"):
                if role not in available_roles:
                    continue
                target[f"{role}_score"] = role_scores.get(role)
                target[f"{role}_dimension_scores"] = role_dimensions.get(role) or {}
            target["review_status"] = "completed" if item_complete else "incomplete"
            target["blind_review_winner_label"] = str(score_row.get("winner_label") or "")
            target["blind_review_notes"] = str(score_row.get("notes") or "")
        completed += int(item_complete)
        result_items.append({"item_id": item_id, "completed": item_complete, "role_scores": role_scores})
    review["items"] = review_items
    review["completed_count"] = completed
    review["updated_at"] = now_iso()
    quality_manifest["summary_blind_review"] = review
    result = {
        "schema": RESULT_SCHEMA,
        "status": "completed" if review_items and completed == len(review_items) else "incomplete",
        "ok": bool(review_items) and completed == len(review_items),
        "quality_manifest_json": str(quality_manifest_path),
        "completed": completed,
        "total": len(review_items),
        "items": result_items,
        "updated_at": now_iso(),
    }
    if write:
        write_json(quality_manifest_path, quality_manifest)
        write_json(private_path.parent / "summary-blind-review-result.json", result)
        (private_path.parent / "summary-blind-review-result.md").write_text(_render_result(result), encoding="utf-8")
        from .quality_benchmark import run_quality_benchmark

        run_quality_benchmark(quality_manifest_path, write=True)
    return result


def _resolved_review_paths(row: dict[str, Any]) -> dict[str, Path | None]:
    paths = {
        "baseline": _existing_path(row.get("baseline_summary_path")),
        "candidate": _existing_path(row.get("candidate_summary_path")),
        "reference": _existing_path(row.get("reference_summary_path")),
    }
    bundle = _existing_directory(row.get("bundle_dir"))
    if bundle and bundle.parent.name.lower() == "local-asr-vkp":
        source_root = bundle.parent.parent
        paths["candidate"] = paths["candidate"] or _existing_path(bundle / "exports" / "smart-summary.md")
        paths["baseline"] = _existing_path(source_root / "webui-bundle" / "exports" / "smart-summary.md") or paths["baseline"]
        paths["reference"] = paths["reference"] or _existing_path(source_root / "getbrain-smart-summary.md")
    return paths


def _non_llm_draft_reason(path: Path | None) -> str:
    if path is None:
        return ""
    text = path.read_text(encoding="utf-8-sig")
    if "local_scaffold_not_llm" in text:
        return "local_scaffold_not_llm"
    if "codex_assisted_draft" in text:
        return "codex_assisted_draft"
    return ""

def _existing_directory(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.exists() and path.is_dir() else None


def _existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.exists() and path.is_file() else None


def _valid_dimension_scores(value: Any) -> dict[str, float]:
    source = value if isinstance(value, dict) else {}
    result: dict[str, float] = {}
    for key, _label in CRITERIA:
        try:
            score = float(source.get(key))
        except (TypeError, ValueError):
            continue
        if 1.0 <= score <= 5.0:
            result[key] = score
    return result if len(result) == len(CRITERIA) else {}


def _blind_text(value: str) -> str:
    hidden_line_markers = ("生成方式", "处理时间", "来源路径", "转写来源", "章节修订来源", "相关产物", "证据审计笔记", "完整逐字稿", "智能总结输入包", "质量报告")
    lines = [
        line
        for line in str(value or "").splitlines()
        if not any(marker in line for marker in hidden_line_markers)
        and not re.search(r"[A-Za-z]:\\", line)
    ]
    text = "\n".join(lines)
    text = re.sub(r"[（(](?:本地\s*)?SenseVoice\s*ASR\s*版[）)]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[（(]得到大脑转写输入版[）)]", "", text)
    text = re.sub(r"\b(?:SenseVoice|VKP)\b", "来源已隐藏", text, flags=re.IGNORECASE)
    text = text.replace("得到大脑", "外部对照").replace("Get笔记", "外部对照")
    return text.strip()


def _review_video_title(row: dict[str, Any], role_paths: dict[str, Path | None], *, index: int) -> str:
    explicit = str(row.get("video_title") or row.get("display_title") or "").strip()
    if explicit:
        return explicit

    source_markers = ("sensevoice", "vkp", "得到大脑", "get笔记", "baseline", "candidate", "reference", "基线", "候选")
    for role in ("candidate", "baseline", "reference"):
        path = role_paths.get(role)
        if path is None:
            continue
        try:
            first_heading = next(
                (line[2:].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("# ")),
                "",
            )
        except OSError:
            first_heading = ""
        title = re.sub(r"\s*[-–—]\s*智能总结\s*$", "", first_heading).strip()
        if title and not any(marker in title.lower() for marker in source_markers):
            return title

    bundle = _existing_directory(row.get("bundle_dir"))
    if bundle is not None:
        parent = bundle.parent
        if parent.name.lower() == "local-asr-vkp":
            parent = parent.parent
        if parent.name and parent.name.lower() not in {"webui-bundle", "bundle"}:
            return parent.name
    return f"未命名视频 {index + 1}"


def _render_review_html(public: dict[str, Any]) -> str:
    cards = []
    for item in public.get("items") or []:
        versions = []
        for version in item.get("versions") or []:
            label = str(version.get("label") or "")
            score_rows = "".join(
                f'<label>{html.escape(str(row["label"]))}<select data-item="{html.escape(str(item["item_id"]))}" data-label="{label}" data-criterion="{row["key"]}"><option value="">未评分</option>'
                + "".join(f'<option value="{score}">{score}</option>' for score in range(1, 6))
                + "</select></label>"
                for row in public.get("criteria") or []
            )
            versions.append(
                f'<article class="version"><h3>版本 {label}</h3><div class="summary">{_simple_markdown(str(version.get("content") or ""))}</div>'
                f'<div class="scores">{score_rows}</div></article>'
            )
        labels = [str(row.get("label") or "") for row in item.get("versions") or []]
        winner_options = '<option value="">未选择</option>' + "".join(f'<option value="{label}">版本 {label}</option>' for label in labels)
        section_id = f'review-{html.escape(str(item.get("item_id") or ""))}'
        cards.append(
            f'<section id="{section_id}" class="item" data-item="{html.escape(str(item.get("item_id") or ""))}"><h2>{html.escape(str(item.get("display_title") or ""))}</h2>'
            f'<div class="versions">{"".join(versions)}</div><div class="decision"><label>整体最佳 <select class="winner">{winner_options}</select></label>'
            '<label>备注 <textarea class="notes" placeholder="可选：遗漏、错词、无证据主张、结构问题"></textarea></label></div></section>'
        )
    video_links = "".join(
        f'<a href="#review-{html.escape(str(item.get("item_id") or ""))}">{html.escape(str(item.get("display_title") or ""))}</a>'
        for item in public.get("items") or []
    )
    video_index = f'<nav class="video-index"><strong>本页评测视频</strong>{video_links}</nav>'
    public_json = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>智能总结匿名盲评</title><style>
body{{font-family:Segoe UI,"Microsoft YaHei",sans-serif;margin:0;background:#f4f6f8;color:#17202a}}header{{position:sticky;top:0;background:#fff;padding:14px 24px;border-bottom:1px solid #ccd3da;z-index:2}}main{{padding:20px}}.video-index{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:#fff;border:1px solid #ccd3da;padding:12px;margin-bottom:20px}}.video-index a{{color:#205d61;text-decoration:none;border-bottom:1px solid #9ab8ba}}.item{{scroll-margin-top:72px;margin-bottom:28px;border-top:3px solid #2f6f73}}.versions{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}}.version{{background:#fff;border:1px solid #ccd3da;border-radius:6px;padding:14px;min-width:0}}.summary{{height:52vh;overflow:auto;border:1px solid #e1e5e8;padding:12px;background:#fbfcfd}}.summary h1{{font-size:1.35rem;margin:0 0 1rem;padding-bottom:.55rem;border-bottom:2px solid #c9d8d9}}.summary h2{{font-size:1.16rem;margin:1.4rem 0 .65rem;color:#205d61}}.summary h3{{font-size:1.04rem;margin:1.1rem 0 .5rem}}.summary p{{line-height:1.65}}.summary ul{{padding-left:1.35rem}}.summary ul ul{{margin:.35rem 0 .45rem .2rem;padding-left:1.25rem;border-left:2px solid #dbe5e6}}.summary li{{margin:.35rem 0;line-height:1.6}}.summary table{{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.94rem}}.summary th,.summary td{{border:1px solid #ccd3da;padding:.45rem .55rem;text-align:left;vertical-align:top}}.summary th{{background:#edf3f3}}.scores{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:12px}}.scores label,.decision label{{display:flex;justify-content:space-between;gap:8px}}select,textarea,button{{font:inherit}}.decision{{display:flex;gap:18px;align-items:flex-start;background:#fff;padding:12px;margin-top:10px}}.decision textarea{{width:420px;height:70px}}button{{padding:8px 14px}}@media(max-width:720px){{.versions{{grid-template-columns:1fr}}.decision{{flex-direction:column}}.decision textarea{{width:90vw}}}}
</style></head><body><header><strong>智能总结匿名盲评</strong>　<span>视频名称公开；各版本的生成来源保持匿名。每项 1–5 分，评分仅用于质量验收。</span> <button id="download">导出评分</button></header><main>{video_index}{''.join(cards)}</main><script id="review-data" type="application/json">{public_json}</script><script>
const schema="{SCORES_SCHEMA}";document.getElementById("download").addEventListener("click",()=>{{const items=[...document.querySelectorAll("section.item")].map(section=>{{const labelScores={{}};section.querySelectorAll("select[data-label]").forEach(select=>{{const label=select.dataset.label;labelScores[label]??={{}};labelScores[label][select.dataset.criterion]=select.value?Number(select.value):null;}});return{{item_id:section.dataset.item,label_scores:labelScores,winner_label:section.querySelector(".winner").value,notes:section.querySelector(".notes").value.trim()}};}});const payload={{schema,items,updated_at:new Date().toISOString()}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:"application/json"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="summary-blind-review-scores.json";a.click();}});
</script></body></html>'''


def _simple_markdown(value: str) -> str:
    source = _normalize_logseq_markdown(value)
    parser = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False}).enable("table")
    return parser.render(source)


def _normalize_logseq_markdown(value: str) -> str:
    lines = str(value or "").splitlines()
    list_indents: list[int] = []
    list_pattern = re.compile(r"^\s*(?:[-*+](?:\s+|$)|•(?:\s+|$))")
    for raw in lines:
        expanded = raw.expandtabs(4)
        if list_pattern.match(expanded):
            list_indents.append(len(expanded) - len(expanded.lstrip(" ")))
    common_indent = min(list_indents) if list_indents else 0
    normalized: list[str] = []
    for raw in lines:
        expanded = raw.expandtabs(4)
        stripped = expanded.lstrip(" ")
        if list_pattern.match(expanded):
            if stripped in {"-", "*", "+", "•"}:
                continue
            indent = max(0, len(expanded) - len(stripped) - common_indent)
            marker_normalized = "- " + stripped[2:] if stripped.startswith("• ") else stripped
            normalized.append(" " * indent + marker_normalized)
        else:
            normalized.append(expanded)
    first_content = next((index for index, line in enumerate(normalized) if line.strip()), None)
    if first_content is not None:
        first = normalized[first_content].strip()
        if not re.match(r"^(?:#{1,6}\s+|[-*+]\s+|\||```)", first):
            normalized[first_content] = f"# {first}"
    return "\n".join(normalized).strip() + "\n"

def _render_result(result: dict[str, Any]) -> str:
    lines = ["# Summary Blind Review Result", "", f"- Status: `{result.get('status', '')}`", f"- Completed: `{result.get('completed', 0)}/{result.get('total', 0)}`", "", "| Item | Completed | Baseline | Candidate | Reference |", "| --- | --- | ---: | ---: | ---: |"]
    for row in result.get("items") or []:
        scores = row.get("role_scores") or {}
        lines.append(f"| {row.get('item_id', '')} | {row.get('completed', False)} | {scores.get('baseline', '')} | {scores.get('candidate', '')} | {scores.get('reference', '')} |")
    return "\n".join(lines).rstrip() + "\n"