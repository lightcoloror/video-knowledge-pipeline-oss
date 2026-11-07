from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

from .config import processing_profile
from .content_profile import resolve_content_profile
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .smart_summary_section_apply import apply_smart_summary_sections
from .smart_summary_section_workflow import build_smart_summary_section_workflow
from .storage import read_json, write_json
from .model_task_gateway import model_task_api_call
from .model_api_settings import resolve_model_api_provider_config
from .production_artifact_gate import evaluate_production_artifact_gate
from .companion_courseware_text import load_companion_courseware_text
from .courseware_material import assert_material_fresh, courseware_prompt
from .model_business_authorization import create_business_child_consent, validate_model_business_authorization
from .trusted_model_connector import execute_consented_model_task
from .text_llm_gateway import openai_compatible_chat_completions_url, resolve_text_provider_config
from .vision_api import provider_requires_api_key, redact_url_secrets


def call_openai_compatible_text(*, provider_config, messages, temperature=0, max_tokens=None):
    return model_task_api_call("smart_summary_section_rewrite", provider_config=provider_config, messages=messages, execute=True, temperature=temperature, max_tokens=max_tokens, write=False)

SCHEMA = "video_knowledge_pipeline.smart_summary_section_llm_rewrite.v1"


def run_smart_summary_section_llm_rewrite(
    bundle_dir: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    execute: bool = False,
    auto_from_profile: bool = False,
    quality_profile: str = "quality",
    target_chapters: int = 8,
    limit: int = 0,
    section_ids: list[str] | str | None = None,
    only_needing_rewrite: bool = False,
    max_prompt_chars: int = 6000,
    max_tokens: int = 1200,
    min_section_chars: int = 120,
    temperature: float = 0,
    install: bool = True,
    require_all_sections: bool = True,
    write: bool = True,
    business_authorization_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rewrite smart-summary section by section, then aggregate with section apply.

    The old whole-summary LLM rewrite can time out on long videos. This command
    reuses the existing section workflow and calls the configured text model once
    per chapter with a small evidence pack. Provider config remains runtime-only.
    """

    root = Path(bundle_dir).expanduser().resolve()
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    production_gate = evaluate_production_artifact_gate(
        root,
        artifact_kind="smart_summary_sections",
        write=write,
    )
    if execute and not production_gate.get("formal_generation_allowed"):
        result = {
            "schema": SCHEMA,
            "bundle_dir": str(root),
            "execute": True,
            "status": "blocked_by_production_artifact_gate",
            "ok": False,
            "production_artifact_gate": production_gate,
            "provider_call_performed": False,
            "next_actions": ["Resolve production-artifact-gate.md; no Provider call was made."],
        }
        return _write_result(root, result, revisions=[], failed_items=[], calls=[], write=write)
    workflow = build_smart_summary_section_workflow(root, target_chapters=target_chapters, write=write)
    cfg = resolve_text_provider_config(
        resolve_model_api_provider_config("summary_rewrite", provider_config)
    )
    public_provider = _public_provider_config(cfg)
    all_sections = [row for row in workflow.get("sections") or [] if isinstance(row, dict)]
    requested_section_ids = _normalise_section_ids(section_ids)
    if requested_section_ids:
        candidates = [row for row in all_sections if str(row.get("section_id") or "") in requested_section_ids]
    elif only_needing_rewrite:
        candidates = [row for row in all_sections if row.get("status") != "ready"]
    else:
        candidates = list(all_sections)
    if limit and limit > 0:
        candidates = candidates[: int(limit)]
    selected_ids = {str(row.get("section_id") or "") for row in candidates}
    profile_execution = _profile_execution(
        root,
        profile_name=quality_profile,
        cfg=cfg,
        candidates=candidates,
        max_prompt_chars=max_prompt_chars,
        explicit_execute=execute,
        auto_from_profile=auto_from_profile,
    )
    if profile_execution.get("auto_execute"):
        execute = True
    revisions: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "execute": bool(execute),
        "auto_from_profile": bool(auto_from_profile),
        "quality_profile": quality_profile,
        "profile_execution": profile_execution,
        "status": "planned",
        "ok": True,
        "provider": public_provider,
        "section_count": len(all_sections),
        "selected_section_count": len(candidates),
        "selected_section_ids": [str(row.get("section_id") or "") for row in candidates],
        "parameters": {
            "target_chapters": int(target_chapters),
            "auto_from_profile": bool(auto_from_profile),
            "quality_profile": quality_profile,
            "limit": int(limit or 0),
            "section_ids": sorted(requested_section_ids),
            "only_needing_rewrite": bool(only_needing_rewrite),
            "max_prompt_chars": int(max_prompt_chars),
            "max_tokens": int(max_tokens),
            "min_section_chars": int(min_section_chars),
            "temperature": float(temperature),
            "install": bool(install),
            "require_all_sections": bool(require_all_sections),
        },
        "business_authorization": {
            "path": str(Path(business_authorization_path).expanduser().resolve()) if business_authorization_path else "",
            "required_for_remote_proxy": _uses_remote_proxy(cfg),
            "execution_mode": "business_child_consent" if business_authorization_path else "direct_or_local",
        },
        "request_plan": {
            "url": redact_url_secrets(openai_compatible_chat_completions_url(cfg)),
            "model": cfg.get("model"),
            "call_count": len(candidates),
            "per_section_prompt_budget": int(max_prompt_chars),
        },
        "artifacts": {
            "workflow_json": str(exports / "smart-summary-section-workflow.json"),
            "revisions_json": str(exports / "smart-summary-section-llm-revisions.json"),
            "report_markdown": str(exports / "smart-summary-section-llm-rewrite.md"),
            "status_json": str(exports / "smart-summary-section-llm-rewrite.json"),
            "apply_markdown": str(exports / "smart-summary-section-apply.md"),
        },
        "operator_boundary": {
            "preview_by_default": True,
            "execute_required_for_network_call": not bool(profile_execution.get("auto_execute")),
            "profile_auto_execution_requires_data_export_allowed": True,
            "provider_config_runtime_only": True,
            "secrets_redacted": True,
            "does_not_process_media": True,
        },
        "updated_at": now_iso(),
    }
    if not execute:
        result["next_actions"] = [
            "Rerun with --execute after provider credentials and data boundary are configured.",
            "Use --limit for small batches, or leave limit=0 to process all chapters.",
        ]
        return _write_result(root, result, revisions=[], failed_items=[], calls=calls, write=write)
    if _uses_remote_proxy(cfg) and not business_authorization_path:
        result.update({"status": "business_authorization_required", "ok": False, "next_actions": ["Create or reuse a matching business authorization.", "Rerun with --business-authorization <active authorization JSON>; direct remote proxy execution is forbidden."]})
        return _write_result(root, result, revisions=[], failed_items=[], calls=calls, write=write)
    if _uses_remote_proxy(cfg) and not write:
        result.update({"status": "business_authorization_write_required", "ok": False, "next_actions": ["Remote business-child execution writes an immutable child consent and Broker receipt; omit --no-write."]})
        return _write_result(root, result, revisions=[], failed_items=[], calls=calls, write=write)
    business_context = _prepare_business_summary_context(root, cfg, business_authorization_path) if business_authorization_path else None
    content_profile = resolve_content_profile(root)
    interview_profile = content_profile.get("profile_id") in {
        "interview-v1",
        "medical-insurance-interview-v1",
    }

    for section in candidates:
        section_id = str(section.get("section_id") or "").strip()
        try:
            assert_material_fresh(root, workflow.get("courseware_material", {}))
            prompt = _section_prompt(root, section, max_prompt_chars=max_prompt_chars)
        except ValueError as exc:
            failed_items.append({"id": section_id, "reason": "courseware_input_blocked", "detail": str(exc)})
            continue
        messages = [
            {
                "role": "system",
                "content": (
                    "你是采访事实摘要的章节编辑器。只还原受访者原话、经历、感受、已确认事实和待核实项；不得生成方法论、面向读者的医疗保险建议、高频话术或可复用表达。只输出本章 Markdown，不要编造视觉证据。"
                    if interview_profile
                    else "你是课程视频智能总结的章节编辑器。只输出这一章的 Markdown 小节正文，不要输出代码块，不要复制 ASR 流水账，不要编造视觉证据。"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        child_consent: dict[str, Any] | None = None
        if business_context is not None:
            request_path = _write_business_section_request(root, section, prompt)
            child_consent = create_business_child_consent(business_context["authorization_path"], stage_id=business_context["stage_id"], artifact_paths=[request_path], producer=business_context["producer"], input_paths=business_context["lineage_input_paths"], max_calls=1, write=True)
            execution = execute_consented_model_task(child_consent["consent_path"], expected_route_revision=business_context["route_revision"], write=True)
            model_result = execution.get("model_result") if isinstance(execution.get("model_result"), dict) else {}
            call = {"ok": bool(execution.get("ok")), "error": str(execution.get("error") or model_result.get("error") or ""), "content": model_result.get("content") or "", "business_child_consent": _public_child_consent(child_consent), "connector_status": str(execution.get("status") or "")}
        else:
            call = call_openai_compatible_text(provider_config=cfg, messages=messages, temperature=temperature, max_tokens=max_tokens)
        raw_content = str(call.get("content") or "")
        assessment = _assess_model_section_output(raw_content, section, min_chars=min_section_chars)
        call_row = {
            "section_id": section_id,
            "ok": bool(call.get("ok")),
            "error": str(call.get("error") or ""),
            "content_chars": len(raw_content),
            "accepted": bool(assessment.get("accepted")),
            "reject_reason": str(assessment.get("reason") or ""),
        }
        if child_consent is not None:
            call_row["business_child_consent"] = _public_child_consent(child_consent)
            call_row["connector_status"] = str(call.get("connector_status") or "")
            call_row["request_artifact"] = str(request_path)
        calls.append(call_row)
        if not call.get("ok"):
            failed_items.append({"id": section_id, "reason": "llm_call_failed", "detail": str(call.get("error") or "failed")})
            continue
        if not assessment.get("accepted"):
            failed_items.append(
                {
                    "id": section_id,
                    "reason": str(assessment.get("reason") or "invalid_model_output"),
                    "detail": str(assessment.get("detail") or "model output did not meet section rewrite quality gate"),
                }
            )
            continue
        markdown = _normalise_section_markdown(raw_content, section)
        revisions.append(
            {
                "section_id": section_id,
                "status": "llm_rewritten",
                "title": section.get("title") or section_id,
                "time_range": f"{section.get('start_time', '')} - {section.get('end_time', '')}",
                "final_markdown": markdown,
                "source": "section_llm_rewrite",
                "courseware_revision": workflow.get("courseware_material", {}).get("revision", ""),
            }
        )

    # Preserve non-selected ready sections only when installing a full summary would otherwise lose coverage.
    revision_ids = {str(row.get("section_id") or "") for row in revisions}
    if install and not require_all_sections:
        for section in all_sections:
            section_id = str(section.get("section_id") or "")
            if section_id in revision_ids or section_id in selected_ids:
                continue
            fallback = _fallback_section_markdown(section)
            if fallback:
                revisions.append({"section_id": section_id, "status": "fallback_from_section_evidence", "final_markdown": fallback})

    status = "completed" if revisions and not failed_items else ("partial_failed" if revisions else "failed")
    result["status"] = status
    result["ok"] = bool(revisions) and not failed_items
    result["rewritten_section_count"] = len(revisions)
    result["failed_section_count"] = len(failed_items)
    result["calls"] = calls
    result["failed_items"] = failed_items
    revisions = _merge_existing_revisions(exports / "smart-summary-section-llm-revisions.json", revisions, selected_ids)
    revisions_payload = {
        "schema": "video_knowledge_pipeline.smart_summary_section_llm_revisions.v1",
        "bundle_dir": str(root),
        "created_at": now_iso(),
        "rows": revisions,
        "courseware_material": workflow.get("courseware_material", {"active": False}),
    }
    apply_result: dict[str, Any] | None = None
    if write:
        write_json(exports / "smart-summary-section-llm-revisions.json", revisions_payload)
    should_install = bool(install and revisions and (not failed_items) and (len(revisions) >= len(all_sections) or not require_all_sections))
    if should_install:
        apply_result = apply_smart_summary_sections(root, input_json=exports / "smart-summary-section-llm-revisions.json", write=write, require_all_sections=require_all_sections)
        result["apply_result"] = apply_result
        result["quality"] = apply_result.get("codex_status", {}).get("quality", {}) if isinstance(apply_result.get("codex_status"), dict) else {}
        if not bool(result.get("quality", {}).get("passed")) or apply_result.get("ok") is False:
            result["status"] = "installed_registry_pending" if apply_result.get("status") == "installed_registry_pending" else "installed_quality_failed"
            result["ok"] = False
    elif install:
        result["next_actions"] = [
            "Some sections failed or were not selected; rerun with a smaller --limit or without --require-all-sections.",
            "Inspect exports/smart-summary-section-llm-rewrite.md and retry failed sections.",
        ]
    return _write_result(root, result, revisions=revisions, failed_items=failed_items, calls=calls, write=write)


def _profile_execution(
    root: Path,
    *,
    profile_name: str,
    cfg: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_prompt_chars: int,
    explicit_execute: bool,
    auto_from_profile: bool,
) -> dict[str, Any]:
    if not auto_from_profile:
        return {"status": "explicit_flags", "auto_execute": False, "preflight_required": False}
    profile = processing_profile(profile_name)
    prompts = [_section_prompt(root, section, max_prompt_chars=max_prompt_chars) for section in candidates]
    input_chars = sum(len(prompt) for prompt in prompts)
    calls = len(candidates)
    call_threshold = max(1, int(profile.get("llm_preflight_call_threshold") or 20))
    char_threshold = max(1000, int(profile.get("llm_preflight_input_char_threshold") or 120000))
    provider_ready = bool(cfg.get("model") and cfg.get("base_url")) and (not provider_requires_api_key(cfg) or bool(cfg.get("api_key")))
    local_execution = _is_loopback_model_config(cfg)
    data_export_allowed = bool(profile.get("data_export_allowed"))
    allowed = data_export_allowed or local_execution
    over_threshold = calls > call_threshold or input_chars > char_threshold
    auto_execute = bool(not explicit_execute and profile.get("text_llm_auto_execute") and allowed and provider_ready and not over_threshold)
    if explicit_execute:
        status = "explicit_execution"
    elif not allowed:
        status = "data_export_not_allowed"
    elif not provider_ready:
        status = "provider_not_ready"
    elif over_threshold:
        status = "batch_preflight_required"
    elif auto_execute:
        status = "auto_execution_ready"
    else:
        status = "auto_execution_disabled"
    return {
        "status": status,
        "auto_execute": auto_execute,
        "data_export_allowed": data_export_allowed,
        "local_execution": local_execution,
        "data_export_required": not local_execution,
        "provider_ready": provider_ready,
        "estimated_calls": calls,
        "estimated_input_chars": input_chars,
        "call_threshold": call_threshold,
        "input_char_threshold": char_threshold,
        "preflight_required": over_threshold,
    }


def _is_loopback_model_config(cfg: dict[str, Any]) -> bool:
    if str(cfg.get("execution_location") or "").strip().lower() == "local":
        return True
    host = str(urlsplit(str(cfg.get("base_url") or "")).hostname or "").lower()
    return host in {"127.0.0.1", "::1", "localhost"}

def _section_prompt(root: Path, section: dict[str, Any], *, max_prompt_chars: int) -> str:
    lines = [
        f"Bundle: {root}",
        f"章节：{section.get('start_time', '')} - {section.get('end_time', '')} {section.get('title', '')}",
        "",
        "请把下面证据改写成 smart-summary.md 的一个章节。要求：",
        "- 只写这一节，不要写整篇总结。",
        "- 输出 1 个三级标题 + 2-5 个要点段落。",
        "- 保留时间范围和可执行动作。",
        "- 压缩口语和重复，不要复制 ASR 流水账。",
        "- 忠实还原说话人的原意；目标不是判断其观点在外部世界是否真实。",
        "- 对主观评价或产品说法使用“讲者表示/客户认为”等归因，不因缺少外部核验而删除。",
        "- 只有音频不清、证据冲突或模型新增内容才进入待复核；不得补造原话之外的事实。",
        "- 若视觉证据未执行或低置信，明确写入边界，不要编造画面细节。",
        "",
        "## 章节改写提示",
        str(section.get("rewrite_prompt") or ""),
        "",
        "## 证据摘要",
    ]
    evidence = section.get("evidence") if isinstance(section.get("evidence"), dict) else {}
    for key, label in (
        ("summary_sentences", "候选摘要"),
        ("key_points", "关键观点"),
        ("actions", "动作"),
        ("reusable_expressions", "话术"),
        ("visual_notes", "视觉证据"),
    ):
        values = evidence.get(key) if isinstance(evidence.get(key), list) else []
        if values:
            lines.append(f"### {label}")
            lines.extend([f"- {str(value)}" for value in values[:8]])
    citations = section.get("citations") if isinstance(section.get("citations"), list) else []
    if citations:
        lines.append("### Citation Digest")
        for row in citations[:8]:
            lines.append(f"- {row.get('time_range', '')} / {row.get('source', '')} / {row.get('snippet', '')}")
    companion = _companion_courseware_context(root, max_chars=min(1600, max(400, int(max_prompt_chars) // 3)))
    if companion:
        lines.append("### 外部课件转写（非视频帧 OCR；不含逐帧时间对应）")
        lines.append(companion)

    text = "\n".join(lines)
    limit = max(1200, int(max_prompt_chars or 6000))
    material = evidence.get("courseware_material", {"active": False})
    if material.get("active"):
        # Never squeeze full page text into the legacy preview truncation path.
        assert_material_fresh(root, material)
        text += "\n\n" + courseware_prompt(material, max_chars=limit)
        if len(text) > limit:
            raise ValueError(f"courseware_input_budget: {len(text)} > {limit}; split section or increase budget")
        return text
    return text if len(text) <= limit else text[:limit].rstrip() + "\n\n[VKP: section prompt truncated by max_prompt_chars]"


def _normalise_section_markdown(content: str, section: dict[str, Any]) -> str:
    text = _strip_fence(str(content or "").strip())
    title = str(section.get("title") or section.get("section_id") or "章节").strip()
    start = str(section.get("start_time") or "").strip()
    end = str(section.get("end_time") or "").strip()
    if not text:
        return _fallback_section_markdown(section)
    if not text.lstrip().startswith("###"):
        text = f"### `{start} - {end}` {title}\n\n" + text
    if "视觉" not in text:
        text = text.rstrip() + "\n\n- 视觉证据边界：如本节涉及屏幕/课件/图表，仍以 bundle 内 OCR、ebook、多模态或人工审核证据为准。"
    return text.rstrip()



def _normalise_section_ids(value: list[str] | str | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).replace(";", ",").split(","))
    return {part.strip() for part in parts if part.strip()}


def _merge_existing_revisions(path: Path, new_revisions: list[dict[str, Any]], selected_ids: set[str]) -> list[dict[str, Any]]:
    """Merge retry output with existing accepted section revisions."""

    merged: dict[str, dict[str, Any]] = {}
    if path.exists():
        existing = read_json(path)
        rows = existing.get("rows") if isinstance(existing, dict) else []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                section_id = str(row.get("section_id") or "").strip()
                if section_id and section_id not in selected_ids:
                    merged[section_id] = row
    for row in new_revisions:
        section_id = str(row.get("section_id") or "").strip()
        if section_id:
            merged[section_id] = row
    return [merged[key] for key in sorted(merged)]

def _assess_model_section_output(content: str, section: dict[str, Any], *, min_chars: int = 120) -> dict[str, Any]:
    text = _strip_fence(str(content or "").strip())
    compact = "".join(text.split())
    min_len = max(40, int(min_chars or 120))
    if not compact:
        return {"accepted": False, "reason": "empty_model_output", "detail": "provider returned empty content"}
    if len(compact) < min_len:
        return {
            "accepted": False,
            "reason": "model_output_too_short",
            "detail": f"provider returned {len(compact)} compact chars; minimum is {min_len}",
        }
    substantive_markers = ("关键", "动作", "建议", "客户", "问题", "流程", "方法", "复核", "证据", "边界")
    if not any(marker in text for marker in substantive_markers):
        return {
            "accepted": False,
            "reason": "model_output_low_information",
            "detail": "provider output lacks expected summary/action/evidence markers",
        }
    title = str(section.get("title") or section.get("section_id") or "").strip()
    if title and compact in "".join(title.split()):
        return {"accepted": False, "reason": "model_output_title_only", "detail": "provider output only repeats the section title"}
    return {"accepted": True, "reason": "", "detail": ""}

def _fallback_section_markdown(section: dict[str, Any]) -> str:
    title = str(section.get("title") or section.get("section_id") or "章节").strip()
    start = str(section.get("start_time") or "").strip()
    end = str(section.get("end_time") or "").strip()
    evidence = section.get("evidence") if isinstance(section.get("evidence"), dict) else {}
    points: list[str] = []
    for key in ("summary_sentences", "key_points", "actions"):
        for value in evidence.get(key) or []:
            text = str(value).strip()
            if text:
                points.append(text)
            if len(points) >= 5:
                break
        if len(points) >= 5:
            break
    if not points:
        return ""
    lines = [f"### `{start} - {end}` {title}", ""]
    lines.extend([f"- {point}" for point in points[:5]])
    lines.append("- 视觉证据边界：本节未经过独立视觉复核时，屏幕/课件细节仍需回看证据。")
    return "\n".join(lines).rstrip()


def _strip_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return value


def _public_provider_config(cfg: dict[str, Any]) -> dict[str, Any]:

    return {
        "provider": cfg.get("provider", ""),
        "model": cfg.get("model", ""),
        "base_url": redact_url_secrets(str(cfg.get("base_url") or "")),
        "api_key_required": provider_requires_api_key(cfg),
        "api_key_configured": bool(cfg.get("api_key")),
        "interface": cfg.get("interface", "openai_chat_completions"),
    }


def _write_result(root: Path, result: dict[str, Any], *, revisions: list[dict[str, Any]], failed_items: list[dict[str, Any]], calls: list[dict[str, Any]], write: bool) -> dict[str, Any]:
    if not write:
        return result
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    write_json(exports / "smart-summary-section-llm-rewrite.json", result)
    (exports / "smart-summary-section-llm-rewrite.md").write_text(_render_markdown(result, revisions, failed_items, calls), encoding="utf-8")
    write_json(root / "mcp-smart-summary-section-llm-rewrite.args.json", {"bundle_dir": str(root), "provider_config": {}, "execute": False, "target_chapters": int(result.get("parameters", {}).get("target_chapters") or 8), "limit": 0, "write": True})
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if isinstance(manifest, dict):
        manifest["smart_summary_section_llm_rewrite"] = "exports/smart-summary-section-llm-rewrite.json"
        manifest["smart_summary_section_llm_rewrite_markdown"] = "exports/smart-summary-section-llm-rewrite.md"
        manifest["smart_summary_section_llm_revisions"] = "exports/smart-summary-section-llm-revisions.json"
        manifest["mcp_smart_summary_section_llm_rewrite_args"] = "mcp-smart-summary-section-llm-rewrite.args.json"
        write_json(manifest_path, manifest)
    run_status = _run_artifact_status(result)
    result["run_artifact"] = register_bundle_run(
        root,
        run_type="smart_summary_section_llm_rewrite",
        run_id="smart-summary-section-llm-rewrite",
        status=run_status,
        title="Smart summary section LLM rewrite",
        summary=f"status={result.get('status')}; selected={result.get('selected_section_count')}; failed={result.get('failed_section_count', 0)}",
        inputs={"workflow_json": str(exports / "smart-summary-section-workflow.json")},
        parameters=result.get("parameters") if isinstance(result.get("parameters"), dict) else {},
        artifacts=[
            {"key": "section_llm_rewrite_json", "path": str(exports / "smart-summary-section-llm-rewrite.json")},
            {"key": "section_llm_rewrite_markdown", "path": str(exports / "smart-summary-section-llm-rewrite.md")},
            {"key": "section_llm_revisions", "path": str(exports / "smart-summary-section-llm-revisions.json")},
        ],
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 run-smart-summary-section-llm-rewrite '{root}'",
        next_actions=result.get("next_actions") or [],
        operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {},
        write=write,
    )
    write_json(exports / "smart-summary-section-llm-rewrite.json", result)
    return result

def _uses_remote_proxy(cfg: dict[str, Any]) -> bool:
    return str(cfg.get("execution_location") or "").strip().lower() == "remote" and str(cfg.get("adapter_backend") or "").strip().lower() == "proxy"


def _prepare_business_summary_context(root: Path, cfg: dict[str, Any], authorization_path: str | Path) -> dict[str, str]:
    path = Path(authorization_path).expanduser().resolve()
    status = validate_model_business_authorization(path)
    if not status.get("valid"):
        blockers = [str(row.get("key") or "blocked") for row in status.get("blockers") or [] if isinstance(row, dict)]
        raise ValueError("business authorization is not active: " + (",".join(blockers) or "unknown"))
    bound_bundle_dirs = {
        Path(str(value or "")).expanduser().resolve()
        for value in (status.get("bundle_dirs") or [status.get("bundle_dir")])
        if str(value or "").strip()
    }
    if root not in bound_bundle_dirs:
        raise ValueError("business authorization bundle does not match bundle_dir")
    payload = read_json(path)
    route_id = str(cfg.get("route_id") or "").strip()
    route_revision = str(cfg.get("route_revision") or "").strip()
    matches = []
    for stage in payload.get("stages") or []:
        if not isinstance(stage, dict) or str(stage.get("task") or "") != "smart_summary_section_rewrite":
            continue
        route = stage.get("route_snapshot") if isinstance(stage.get("route_snapshot"), dict) else {}
        if route_id and str(route.get("route_id") or "") != route_id:
            continue
        if route_revision and str(route.get("route_revision") or "") != route_revision:
            continue
        matches.append(stage)
    if len(matches) != 1:
        raise ValueError("business authorization must contain exactly one matching smart_summary_section_rewrite route stage")
    stage = matches[0]
    if "smart_summary_input_pack" not in [str(value) for value in stage.get("allowed_producers") or []]:
        raise ValueError("business authorization summary stage does not allow smart_summary_input_pack")
    route = stage.get("route_snapshot") if isinstance(stage.get("route_snapshot"), dict) else {}
    lineage_input_paths = [str(_authorised_summary_lineage(root, payload))]
    companion = _companion_courseware_lineage(root, payload)
    if companion is not None:
        lineage_input_paths.append(str(companion))
    return {"authorization_path": str(path), "stage_id": str(stage.get("id") or ""), "producer": "smart_summary_input_pack", "lineage_input_paths": lineage_input_paths, "route_revision": str(route.get("route_revision") or "")}


def _authorised_summary_lineage(root: Path, authorization: dict[str, Any]) -> Path:
    manifest = read_json(root / "manifest.json") if (root / "manifest.json").is_file() else {}
    candidates: list[Path] = []
    for key in ("source_arbitrated_transcript_json", "corrected_transcript_json", "transcript_json"):
        value = str(manifest.get(key) or "") if isinstance(manifest, dict) else ""
        if value:
            candidates.append((root / value).resolve())
    pack = root / "exports" / "smart-summary-input-pack.json"
    if pack.is_file():
        try:
            packed = read_json(pack)
            value = str(packed.get("transcript_source") or "") if isinstance(packed, dict) else ""
            if value:
                candidates.append(Path(value).expanduser().resolve())
        except (OSError, ValueError):
            pass
    known = {str(row.get("path") or "") for row in authorization.get("sources") or [] if isinstance(row, dict)}
    known.update(str(row.get("path") or "") for admission in authorization.get("admissions") or [] if isinstance(admission, dict) for row in admission.get("artifacts") or [] if isinstance(row, dict))
    for candidate in candidates:
        if candidate.is_file() and str(candidate) in known:
            return candidate
    raise ValueError("business authorization does not bind a current canonical transcript lineage input")


def _companion_courseware_lineage(root: Path, authorization: dict[str, Any]) -> Path | None:
    manifest = read_json(root / "manifest.json") if (root / "manifest.json").is_file() else {}
    payload = load_companion_courseware_text(root, manifest) if isinstance(manifest, dict) else None
    if not payload:
        return None
    path = Path(str(payload.get("bundle_copy_path") or "")).expanduser().resolve()
    known = {str(row.get("path") or "") for row in authorization.get("sources") or [] if isinstance(row, dict)}
    known.update(str(row.get("path") or "") for admission in authorization.get("admissions") or [] if isinstance(admission, dict) for row in admission.get("artifacts") or [] if isinstance(row, dict))
    if not path.is_file() or str(path) not in known:
        raise ValueError("business authorization does not bind the imported companion courseware text")
    return path

def _write_business_section_request(root: Path, section: dict[str, Any], prompt: str) -> Path:
    section_id = str(section.get("section_id") or "").strip()
    if not section_id:
        raise ValueError("smart summary section id is required for business authorization")
    path = root / "exports" / "business-authorized-summary-requests" / f"summary-section-{section_id}.json"
    write_json(path, {"schema": "video_knowledge_pipeline.smart_summary_section_request.v1", "section_id": section_id, "title": str(section.get("title") or ""), "time_range": f"{section.get('start_time', '')} - {section.get('end_time', '')}", "instructions": prompt, "output_contract": {"format": "markdown", "scope": "one_section_only", "forbidden": ["code_fence", "invented_visual_facts", "whole_video_summary"]}})
    return path


def _public_child_consent(child: dict[str, Any]) -> dict[str, Any]:
    return {key: str(child.get(key) or "") for key in ("status", "consent_path", "consent_id", "route_revision", "admission_id")}


def _run_artifact_status(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "").strip()
    if status == "planned":
        return "needs_execution"
    if status in {"completed", "installed"} and bool(result.get("ok")):
        return "completed"
    if status in {"failed", "partial_failed", "installed_quality_failed"}:
        return "needs_retry"
    return "needs_retry" if not result.get("ok") else "completed"


def _companion_courseware_context(root: Path, *, max_chars: int) -> str:
    manifest = read_json(root / "manifest.json") if (root / "manifest.json").is_file() else {}
    payload = load_companion_courseware_text(root, manifest) if isinstance(manifest, dict) else None
    if not payload:
        return ""
    text = str(payload.get("text") or "").strip()
    if not text:
        return ""
    limit = max(200, int(max_chars))
    suffix = "\n[VKP: companion courseware excerpt truncated]"
    return text if len(text) <= limit else text[:limit].rstrip() + suffix


def _render_markdown(result: dict[str, Any], revisions: list[dict[str, Any]], failed_items: list[dict[str, Any]], calls: list[dict[str, Any]]) -> str:
    lines = [
        "# Smart Summary Section LLM Rewrite",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Execute: `{result.get('execute')}`",
        f"- Selected sections: `{result.get('selected_section_count')}`",
        f"- Rewritten sections: `{len(revisions)}`",
        f"- Failed sections: `{len(failed_items)}`",
        "",
        "## Calls",
        "",
    ]
    for call in calls:
        lines.append(f"- `{call.get('section_id')}` ok={call.get('ok')} chars={call.get('content_chars')} error={call.get('error')}")
    if failed_items:
        lines.extend(["", "## Failed Items", ""])
        for item in failed_items:
            lines.append(f"- `{item.get('id')}` {item.get('reason')}: {item.get('detail')}")
    lines.extend(["", "## Artifacts", ""])
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines).rstrip() + "\n"
