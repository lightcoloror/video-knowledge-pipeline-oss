from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import write_json

SCHEMA = "video_knowledge_pipeline.volcengine_model_routing.v1"
BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"

TOOL_TERMS_ROUTE: list[dict[str, Any]] = [
    {
        "stage": "final_arbitration_doubao",
        "label": "豆包最终术语仲裁",
        "model": "doubao-seed-2.0-pro",
        "role": "primary",
        "max_tokens": 2400,
        "thinking": None,
        "when_to_use": "已配置且经授权时，可作为中文工具名与语义纠错的主模型。",
        "quality_note": "由实际 provider 响应和 consent 约束决定可用性，不按模型名称预先封禁。",
    },
    {
        "stage": "batch_triage",
        "label": "批量初筛 / 低成本草判",
        "model": "deepseek-v4-flash",
        "role": "fast_triage",
        "max_tokens": 1800,
        "thinking": None,
        "when_to_use": "大批量工具名/代码术语疑难点筛选；先判断是否值得高成本仲裁。",
        "quality_note": "实测能纠出 Playwright、Chrome DevTools、UI-TARS；置信度校准较保守。",
    },
    {
        "stage": "final_arbitration",
        "label": "最终高风险工具名仲裁",
        "model": "deepseek-v4-pro",
        "role": "primary_alternative",
        "max_tokens": 2400,
        "thinking": None,
        "when_to_use": "ASR、OCR、字幕、画面文字、候选词典互相冲突，且会影响最终逐字稿/总结的专名。",
        "quality_note": "同题公平测试中最稳，纠错准确、理由够用，适合最终判定。",
    },
    {
        "stage": "second_opinion_long_explanation",
        "label": "长解释/结构化第二意见",
        "model": "kimi-k2.6",
        "role": "second_opinion",
        "max_tokens": 2400,
        "when_to_use": "需要更长中文解释、更完整表格说明，或 DeepSeek 结果需要独立复核。",
        "quality_note": "必须关闭 thinking；关闭后输出完整，可读性好，但疑难候选会犹豫。",
    },
    {
        "stage": "second_opinion_independent",
        "label": "独立判断第二意见",
        "model": "glm-5.2",
        "role": "second_opinion",
        "max_tokens": 2400,
        "when_to_use": "需要不同模型族给出独立判断，辅助发现 DeepSeek/Kimi 可能遗漏的解释。",
        "quality_note": "必须关闭 thinking；输出清楚，但要防轻微幻觉。",
    },
]

MIXED_ROUTE_NOTES = {
    "kimi-k2.7-code": {
        "status": "coding_plan_alias_verified_by_execution",
        "why": "Coding Plan Model Name 已由 consented Stage B 实测；通用 /models 不作为 Coding Plan 别名真源。",
    },
}

ROUTES = {
    "tool_terms": TOOL_TERMS_ROUTE,
    # Backward-compatible name only. It deliberately resolves to the same full route.
    "tool_terms_no_doubao": TOOL_TERMS_ROUTE,
}


def volcengine_model_routing(
    *,
    route: str = "tool_terms",
    output_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    key = str(route or "tool_terms").strip()
    rows = [dict(row) for row in ROUTES.get(key, [])]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "route": key,
        "status": "ok" if rows else "unknown_route",
        "ok": bool(rows),
        "base_url": BASE_URL,
        "do_not_use_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "operator_boundary": {
            "does_not_call_network": True,
            "secrets_redacted": True,
            "provider_config_runtime_only": True,
        },
        "provider_configs": [_provider_config_for(row) for row in rows],
        "route_steps": rows,
        "mixed_route_notes": MIXED_ROUTE_NOTES,
        "input_requirements": [
            "ASR/subtitle suspect term",
            "nearby transcript context",
            "OCR/ebook/visual text evidence when available",
            "screen/multimodal context when available",
            "candidate dictionary for tool names/code terms",
            "source URLs/page title/description when relevant",
        ],
        "quality_gates": [
            "must preserve evidence source names",
            "must output confidence per term",
            "low-confidence or conflicting results go to review instead of overwriting transcript",
            "use only exact Coding Plan Model Names documented by the subscription or verified by consented execution",
            "generic /models absence does not invalidate a Coding Plan alias; unverified aliases still require a consented capability gate",
        ],
        "updated_at": now_iso(),
    }
    return _write_outputs(result, output_dir=output_dir, write=write)


def _provider_config_for(row: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "provider": "volcengine_coding_plan",
        "base_url": BASE_URL,
        "model": row.get("model"),
        "max_tokens": row.get("max_tokens"),
    }
    if isinstance(row.get("thinking"), dict):
        cfg["thinking"] = row["thinking"]
    return cfg


def _write_outputs(result: dict[str, Any], *, output_dir: str | Path | None, write: bool) -> dict[str, Any]:
    if not write or not output_dir:
        return result
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "volcengine-model-routing.json"
    md_path = root / "volcengine-model-routing.md"
    write_json(json_path, result)
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    result["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
    write_json(json_path, result)
    return result


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Volcengine Model Routing",
        "",
        f"- Route: `{result.get('route')}`",
        f"- Status: `{result.get('status')}`",
        f"- Base URL: `{result.get('base_url')}`",
        "- Boundary: do not use `https://ark.cn-beijing.volces.com/api/v3` for Coding Plan quota.",
        "",
        "## Route Steps",
        "",
        "| Stage | Model | Role | Max tokens | Thinking | When to use |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in result.get("route_steps") or []:
        thinking = row.get("thinking") if isinstance(row.get("thinking"), dict) else None
        thinking_text = "disabled" if thinking else "default"
        lines.append(
            f"| {row.get('label')} | `{row.get('model')}` | `{row.get('role')}` | {row.get('max_tokens')} | {thinking_text} | {str(row.get('when_to_use') or '').replace('|', '/')} |"
        )
    lines.extend(["", "## Input Requirements", ""])
    for item in result.get("input_requirements") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Quality Gates", ""])
    for item in result.get("quality_gates") or []:
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"
