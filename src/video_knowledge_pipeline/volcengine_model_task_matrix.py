from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import write_json
from .model_task_gateway import model_task_api_call
from .text_llm_gateway import openai_compatible_chat_completions_url, resolve_text_provider_config
from .vision_api import redact_url_secrets

SCHEMA = "video_knowledge_pipeline.volcengine_model_task_matrix.v1"
BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"

MODEL_TASK_PROFILES: list[dict[str, Any]] = [
    {
        "task_key": "smart_summary_final",
        "task_label": "智能总结最终成稿",
        "model": "doubao-seed-2.0-pro",
        "role": "primary",
        "why": "通用理解和中文成稿能力优先，适合作为 smart-summary.md 默认改写模型。",
        "max_tokens": 900,
        "prompt": "把下面课程证据改写成可读的中文智能总结小节，保留时间范围，不要编造视觉证据。\n证据：00:04:28-00:06:42 讲师讲解电话邀约保险客户：先降低陌生电话压力，说明只需要十几二十分钟；用二择一约时间；强调保单整理、查缺补漏；如果客户拒绝电话，则顺应客户，先用微信文字结合语音沟通。",
        "required_markers": ["邀约", "客户", "时间", "保单"],
    },
    {
        "task_key": "batch_triage",
        "task_label": "批量疑难点初筛",
        "model": "doubao-seed-2.0-lite",
        "role": "primary",
        "why": "低成本快速判断是否需要高质量模型复核。",
        "max_tokens": 500,
        "prompt": "判断下面转写片段是否需要进一步语义纠错，只输出 JSON：{needs_review:boolean, reasons:string[], suspected_terms:string[]}。\n片段：这里我们用 play right MCP 和 chrome dive tooth 做浏览器自动化，另一个叫 browser honeys。",
        "expect_json": True,
        "required_markers": ["needs_review", "suspected_terms"],
    },
    {
        "task_key": "evidence_arbitration",
        "task_label": "多证据冲突仲裁",
        "model": "deepseek-v4-pro",
        "role": "primary",
        "why": "强推理复核，用于 ASR、字幕、OCR、多模态互相冲突时的最终判断。",
        "max_tokens": 700,
        "prompt": "根据证据判断正确词，输出 JSON：{winner:string, confidence:number, rationale:string}。\nASR：chrome dive tooth。OCR：Chrome DevTools。网页标题：浏览器自动化工具横评。上下文：讲师在评价谷歌亲儿子的浏览器调试/自动化工具。",
        "expect_json": True,
        "required_markers": ["Chrome", "confidence"],
    },
    {
        "task_key": "fast_evidence_arbitration",
        "task_label": "快速冲突仲裁备选",
        "model": "deepseek-v4-flash",
        "role": "fallback_fast",
        "why": "低延迟/低成本冲突复核备选。",
        "max_tokens": 600,
        "prompt": "根据证据判断正确词，输出 JSON：{winner:string, confidence:number, rationale:string}。\nASR：brother mc p。OCR：browser mcp。上下文：讲师在评价浏览器插件走 MCP 协议的自动化工具。",
        "expect_json": True,
        "required_markers": ["browser", "confidence"],
    },
    {
        "task_key": "tool_code_terms",
        "task_label": "工具名/代码名纠错",
        "model": "doubao-seed-2.0-code",
        "role": "primary",
        "why": "工具名、API、代码框架、CLI 名称密集内容优先用代码向模型。",
        "max_tokens": 700,
        "prompt": "纠正下面浏览器自动化工具名，输出 Markdown 表格：误识别|推荐写法|理由。\n误识别：play right MCP、chrome dive tooth、Page agent、u i tars、Browser honeys、open client。",
        "required_markers": ["Playwright", "Chrome", "Browser"],
    },
    {
        "task_key": "tool_code_terms_deepseek",
        "task_label": "工具名/代码名纠错（非豆包）",
        "model": "deepseek-v4-pro",
        "role": "fallback",
        "why": "完全避开豆包时，用强推理模型结合上下文、OCR 和候选词典做工具名/代码名仲裁。",
        "max_tokens": 2400,
        "prompt": "你在修正一段浏览器自动化横评视频的 ASR 错词。请结合上下文、OCR 候选和常见工具名，输出 Markdown 表格：误识别|推荐写法|置信度|理由。\n上下文：讲师在横评浏览器自动化、MCP、CDP、Agent 浏览器和桌面操作工具。\nOCR/候选词典：Playwright MCP, Chrome DevTools, Browserbase, Browser Use, Stagehand, OpenAI Operator, UI-TARS, PageAgent, Browser MCP。\n误识别：play right MCP、chrome dive tooth、Page agent、u i tars、Browser honeys、open client。",
        "required_markers": ["Playwright", "Chrome", "UI-TARS"],
    },
    {
        "task_key": "tool_code_terms_deepseek_flash",
        "task_label": "工具名/代码名纠错（DeepSeek Flash）",
        "model": "deepseek-v4-flash",
        "role": "candidate_fast",
        "why": "完全避开豆包时的低成本工具名/代码名纠错候选，适合先跑批量草判。",
        "max_tokens": 1800,
        "prompt": "你在修正一段浏览器自动化横评视频的 ASR 错词。请结合上下文、OCR 候选和常见工具名，输出 Markdown 表格：误识别|推荐写法|置信度|理由。\n上下文：讲师在横评浏览器自动化、MCP、CDP、Agent 浏览器和桌面操作工具。\nOCR/候选词典：Playwright MCP, Chrome DevTools, Browserbase, Browser Use, Stagehand, OpenAI Operator, UI-TARS, PageAgent, Browser MCP。\n误识别：play right MCP、chrome dive tooth、Page agent、u i tars、Browser honeys、open client。",
        "required_markers": ["Playwright", "Chrome", "UI-TARS"],
    },
    {
        "task_key": "tool_code_terms_glm",
        "task_label": "工具名/代码名纠错（GLM）",
        "model": "glm-5.2",
        "role": "candidate_needs_adapter_check",
        "why": "GLM 具备代码/通用推理能力，但需要在 Coding Plan OpenAI-compatible 接口下验证是否能稳定返回 content。",
        "max_tokens": 2400,
        "prompt": "你在修正一段浏览器自动化横评视频的 ASR 错词。请结合上下文、OCR 候选和常见工具名，输出 Markdown 表格：误识别|推荐写法|置信度|理由。\n上下文：讲师在横评浏览器自动化、MCP、CDP、Agent 浏览器和桌面操作工具。\nOCR/候选词典：Playwright MCP, Chrome DevTools, Browserbase, Browser Use, Stagehand, OpenAI Operator, UI-TARS, PageAgent, Browser MCP。\n误识别：play right MCP、chrome dive tooth、Page agent、u i tars、Browser honeys、open client。",
        "required_markers": ["Playwright", "Chrome", "UI-TARS"],
    },
    {
        "task_key": "tool_code_terms_kimi",
        "task_label": "工具名/代码名纠错（Kimi 2.6）",
        "model": "kimi-k2.6",
        "role": "candidate_long_context",
        "why": "Kimi 2.6 适合长文本和结构化，测试其在工具名/代码名纠错中的稳定性。",
        "max_tokens": 2400,
        "prompt": "你在修正一段浏览器自动化横评视频的 ASR 错词。请结合上下文、OCR 候选和常见工具名，输出 Markdown 表格：误识别|推荐写法|置信度|理由。\n上下文：讲师在横评浏览器自动化、MCP、CDP、Agent 浏览器和桌面操作工具。\nOCR/候选词典：Playwright MCP, Chrome DevTools, Browserbase, Browser Use, Stagehand, OpenAI Operator, UI-TARS, PageAgent, Browser MCP。\n误识别：play right MCP、chrome dive tooth、Page agent、u i tars、Browser honeys、open client。",
        "required_markers": ["Playwright", "Chrome", "UI-TARS"],
    },
    {
        "task_key": "tool_code_terms_kimi_code",
        "task_label": "工具名/代码名纠错（Kimi Code）",
        "model": "kimi-k2.7-code",
        "catalog_status": "coding_plan_alias_verified_by_execution",
        "role": "candidate_code",
        "why": "Kimi code 模型理论上适合代码/工具链术语，但需要验证 Coding Plan 接口是否能给出最终 content。",
        "max_tokens": 3200,
        "prompt": "你在修正一段浏览器自动化横评视频的 ASR 错词。请结合上下文、OCR 候选和常见工具名，输出 Markdown 表格：误识别|推荐写法|置信度|理由。\n上下文：讲师在横评浏览器自动化、MCP、CDP、Agent 浏览器和桌面操作工具。\nOCR/候选词典：Playwright MCP, Chrome DevTools, Browserbase, Browser Use, Stagehand, OpenAI Operator, UI-TARS, PageAgent, Browser MCP。\n误识别：play right MCP、chrome dive tooth、Page agent、u i tars、Browser honeys、open client。",
        "required_markers": ["Playwright", "Chrome", "UI-TARS"],
    },
    {
        "task_key": "legacy_code_terms",
        "task_label": "旧代码模型备选",
        "model": "doubao-seed-code",
        "role": "fallback_code",
        "why": "旧版代码向模型，仅作为工具名纠错备选。",
        "max_tokens": 700,
        "prompt": "纠正下面技术工具名，输出 JSON 数组，每项包含 wrong, corrected, confidence。\nplay right client, chrome dive tools, browser use, stage hand",
        "expect_json": True,
        "required_markers": ["corrected", "Playwright"],
    },
    {
        "task_key": "long_section_summary",
        "task_label": "长章节总结备选",
        "model": "minimax-m3",
        "catalog_status": "not_visible_in_account_catalog",
        "role": "fallback_long_context",
        "why": "长章节、长文本改写备选。",
        "max_tokens": 900,
        "prompt": "把下面长章节证据压缩成课程笔记，包含：本章主线、关键判断、可执行动作、待复核边界。\n证据：讲师先讲 AI 浪潮下跨境电商和独立站机会，再讲华强北供应链、TikTok 官方仓、AI 建站、SEO 内容矩阵、课程权益和简历修改需求。视觉证据包含 PPT 和网页操作，但不是所有画面都经过人工复核。",
        "required_markers": ["主线", "动作", "复核"],
    },
    {
        "task_key": "cheap_long_summary",
        "task_label": "轻量长文本备选",
        "model": "minimax-m2.7",
        "catalog_status": "not_visible_in_account_catalog",
        "role": "fallback_light_long",
        "why": "长文本低成本备选，适合非最终稿初稿。",
        "max_tokens": 700,
        "prompt": "请把这段课程内容整理成 3 条关键观点和 3 条行动建议：AI 可以降低独立站建站成本，外贸获客从展会转向短视频和搜索，供应链选品仍需要线下验厂和安全认证。",
        "required_markers": ["观点", "建议"],
    },
    {
        "task_key": "general_arbitration",
        "task_label": "通用语义仲裁备选",
        "model": "glm-5.2",
        "role": "needs_adapter",
        "why": "通用中文理解与仲裁候选；实测当前接口可能只返回 reasoning_content，未适配前不进默认生产链路。",
        "max_tokens": 700,
        "prompt": "判断下面 ASR 是否存在语义错词，并给出修正：\nASR：客户说二十分钟的话，那我需要是明天上午。\n上下文：顾问在用二择一方式约客户明天上午/中午/晚上电话沟通。",
        "required_markers": ["客户", "明天", "修正"],
    },
    {
        "task_key": "long_transcript_structure",
        "task_label": "长转写结构化备选",
        "model": "kimi-k2.6",
        "role": "needs_adapter",
        "why": "长文本阅读和结构化候选；实测当前接口可能只返回 reasoning_content，未适配前不进默认生产链路。",
        "max_tokens": 800,
        "prompt": "把下面视频转写整理成章节结构：第一段讲获客邀约前置同理心，第二段讲二择一约时间，第三段讲保单整理建立专家感，第四段讲客户拒绝电话时改用微信文字加语音。",
        "required_markers": ["章节", "邀约", "保单"],
    },
    {
        "task_key": "code_tool_chain_terms",
        "task_label": "代码/工具链视频备选",
        "model": "kimi-k2.7-code",
        "catalog_status": "coding_plan_alias_verified_by_execution",
        "role": "needs_adapter",
        "why": "代码/工具链术语候选；实测当前接口可能只返回 reasoning_content，未适配前不进默认生产链路。",
        "max_tokens": 700,
        "prompt": "请识别并纠正这段浏览器自动化横评中的工具名：play right MCP、stage hand、page agent、browser honeys、u i tars。输出 Markdown 表格。",
        "required_markers": ["Playwright", "Stagehand", "Page"],
    },
]

TASK_RECOMMENDATIONS = {
    "smart_summary": ["kimi-k2.6", "deepseek-v4-pro", "glm-5.2"],
    "transcript_correction": ["deepseek-v4-pro", "deepseek-v4-flash", "glm-5.2"],
    "triage": ["deepseek-v4-flash", "glm-5.2"],
    "tool_terms": ["doubao-seed-2.0-pro", "deepseek-v4-pro", "glm-5.2", "kimi-k2.6"],
    # Legacy key is retained for callers, but it is no longer a model-name filter.
    "tool_terms_no_doubao": ["doubao-seed-2.0-pro", "deepseek-v4-pro", "glm-5.2", "kimi-k2.6", "deepseek-v4-flash"],
    "evidence_arbitration": ["deepseek-v4-pro", "deepseek-v4-flash"],
    "long_context": ["kimi-k2.6", "glm-5.2", "deepseek-v4-pro"],
}


def run_volcengine_model_task_matrix(
    *,
    execute: bool = False,
    output_dir: str | Path | None = None,
    models: list[str] | str | None = None,
    tasks: list[str] | str | None = None,
    timeout_seconds: int = 120,
    write: bool = True,
) -> dict[str, Any]:
    selected_models = _normalise_filter(models)
    selected_tasks = _normalise_filter(tasks)
    rows = [
        dict(row) for row in MODEL_TASK_PROFILES
        if str(row.get("catalog_status") or "verified_visible")
        in {"verified_visible", "coding_plan_alias_verified_by_execution"}
    ]
    if selected_models:
        rows = [row for row in rows if str(row.get("model") or "") in selected_models]
    if selected_tasks:
        rows = [row for row in rows if str(row.get("task_key") or "") in selected_tasks]

    results: list[dict[str, Any]] = []
    for row in rows:
        provider_config = {
            "provider": "volcengine_coding_plan",
            "base_url": BASE_URL,
            "model": row["model"],
            "timeout_seconds": int(timeout_seconds),
        }
        if isinstance(row.get("thinking"), dict):
            provider_config["thinking"] = row["thinking"]
        if isinstance(row.get("extra_body"), dict):
            provider_config["extra_body"] = row["extra_body"]
        cfg = resolve_text_provider_config(provider_config)
        item = {
            "task_key": row["task_key"],
            "task_label": row["task_label"],
            "model": row["model"],
            "role": row["role"],
            "why": row["why"],
            "execute": bool(execute),
            "provider": _public_provider(cfg),
            "request_plan": {
                "url": redact_url_secrets(openai_compatible_chat_completions_url(cfg)),
                "model": row["model"],
                "max_tokens": int(row.get("max_tokens") or 800),
                "thinking": row.get("thinking") if isinstance(row.get("thinking"), dict) else None,
            },
            "quality_gate": {
                "expect_json": bool(row.get("expect_json")),
                "required_markers": row.get("required_markers") or [],
            },
        }
        if execute:
            call = model_task_api_call("provider_task_benchmark", execute=True, write=False,
                provider_config=cfg,
                messages=[{"role": "user", "content": str(row.get("prompt") or "")}],
                temperature=0,
                max_tokens=int(row.get("max_tokens") or 800),
            )
            assessment = _assess_output(str(call.get("content") or ""), row)
            item.update({
                "ok": bool(call.get("ok")) and bool(assessment.get("passed")),
                "status": "passed" if call.get("ok") and assessment.get("passed") else "failed",
                "error": str(call.get("error") or ""),
                "content_chars": len(str(call.get("content") or "")),
                "assessment": assessment,
            })
        else:
            item.update({"ok": True, "status": "planned"})
        results.append(item)

    result = {
        "schema": SCHEMA,
        "execute": bool(execute),
        "status": _overall_status(results, execute=execute),
        "ok": all(bool(row.get("ok")) for row in results) if results else False,
        "base_url": BASE_URL,
        "base_url_boundary": {
            "use": BASE_URL,
            "do_not_use": "https://ark.cn-beijing.volces.com/api/v3",
            "reason": "api/v3 is not the Coding Plan endpoint and may create extra charges.",
        },
        "recommendations": TASK_RECOMMENDATIONS,
        "model_profiles": results,
        "selected_count": len(results),
        "operator_boundary": {
            "preview_by_default": True,
            "execute_required_for_network_call": True,
            "secrets_redacted": True,
            "does_not_send_media": True,
            "text_samples_only": True,
        },
        "updated_at": now_iso(),
    }
    return _write_outputs(result, output_dir=output_dir, write=write)


def _normalise_filter(value: list[str] | str | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).replace(";", ",").split(","))
    return {part.strip() for part in parts if part.strip()}


def _public_provider(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": cfg.get("provider"),
        "base_url": redact_url_secrets(str(cfg.get("base_url") or "")),
        "model": cfg.get("model"),
        "api_key_required": True,
        "api_key_configured": bool(cfg.get("api_key")),
        "interface": "openai_chat_completions",
    }


def _assess_output(content: str, row: dict[str, Any]) -> dict[str, Any]:
    text = str(content or "")
    markers = [str(marker) for marker in row.get("required_markers") or []]
    marker_hits = [marker for marker in markers if marker.lower() in text.lower()]
    checks = [
        {"key": "non_empty_content", "passed": bool(text.strip()), "detail": f"content_chars={len(text)}"},
        {"key": "required_markers", "passed": len(marker_hits) >= max(1, min(len(markers), 2)) if markers else True, "detail": ",".join(marker_hits)},
    ]
    if row.get("expect_json"):
        checks.append({"key": "json_like", "passed": _looks_json_like(text), "detail": "requires JSON-like output"})
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "preview": text[:320],
    }


def _looks_json_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("```") and "{" in stripped:
        return True
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            return True
    return "{" in stripped and "}" in stripped and ":" in stripped


def _overall_status(rows: list[dict[str, Any]], *, execute: bool) -> str:
    if not rows:
        return "no_matching_profiles"
    if not execute:
        return "planned"
    failed = [row for row in rows if not row.get("ok")]
    if not failed:
        return "passed"
    if len(failed) == len(rows):
        return "failed"
    return "partial_failed"


def _write_outputs(result: dict[str, Any], *, output_dir: str | Path | None, write: bool) -> dict[str, Any]:
    if not write or not output_dir:
        return result
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "volcengine-model-task-matrix.json"
    md_path = root / "volcengine-model-task-matrix.md"
    write_json(json_path, result)
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    result["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
    write_json(json_path, result)
    return result


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Volcengine Coding Plan Model Task Matrix",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Execute: `{result.get('execute')}`",
        f"- Base URL: `{result.get('base_url')}`",
        "- Boundary: do not use `https://ark.cn-beijing.volces.com/api/v3` for Coding Plan quota.",
        "",
        "## Recommendations",
        "",
    ]
    recommendations = result.get("recommendations") if isinstance(result.get("recommendations"), dict) else {}
    for task, models in recommendations.items():
        lines.append(f"- `{task}`: " + ", ".join(f"`{model}`" for model in models))
    lines.extend(["", "## Test Results", "", "| Task | Model | Role | Status | Chars | Why |", "| --- | --- | --- | --- | ---: | --- |"])
    for row in result.get("model_profiles") or []:
        lines.append(
            f"| {row.get('task_label')} | `{row.get('model')}` | `{row.get('role')}` | `{row.get('status')}` | {row.get('content_chars', '')} | {str(row.get('why') or '').replace('|', '/')} |"
        )
    failed = [row for row in result.get("model_profiles") or [] if row.get("status") == "failed"]
    if failed:
        lines.extend(["", "## Failed / Needs Review", ""])
        for row in failed:
            assessment = row.get("assessment") if isinstance(row.get("assessment"), dict) else {}
            lines.append(f"- `{row.get('model')}` / `{row.get('task_key')}`: {row.get('error') or assessment.get('checks')}")
    return "\n".join(lines).rstrip() + "\n"
