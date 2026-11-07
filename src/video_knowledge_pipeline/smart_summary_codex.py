from __future__ import annotations

import re
import json
import hashlib
import os
from pathlib import Path
from typing import Any

from .artifact_freshness import build_dependency_snapshot, validate_dependency_snapshot
from .content_profile import resolve_content_profile
from .content_scope import apply_content_scope_to_cues, load_content_scope
from .powershell import quote_powershell_literal as _ps_quote
from .file_hash import sha256_file
from .models import now_iso
from .numeric_normalization import number_evidence_map
from .production_artifact_gate import (
    evaluate_production_artifact_gate,
    transcript_completeness_status,
)
from .run_artifact_registry import register_bundle_run
from .long_video_memory_pack import build_long_video_memory_pack
from .storage import read_json, write_json
from .term_impact_gate import load_term_correction_impact_gate
from .transcript_semantic_correction import transcript_semantic_correction_status
from .smart_summary_chapters import build_smart_summary_chapter_pack
from .smart_summary_input_pack import build_smart_summary_input_pack
from .courseware_material import assert_material_fresh, build_courseware_material, render_courseware_model_input
from .smart_summary_keypoint_eval import evaluate_human_key_point_recall
from .smart_summary_reader_plan import evaluate_reader_markdown_semantics
from .term_text import apply_term_replacement_pairs, load_bundle_term_replacements
from .model_task_gateway import model_task_api_call
from .text_llm_gateway import openai_compatible_chat_completions_url, resolve_text_provider_config
from .transcript import format_timestamp, parse_transcript
from .vision_api import provider_requires_api_key, redact_url_secrets
from .visual_evidence import continuous_tag_language_gate, temporal_evidence_state, visual_evidence_state


def call_openai_compatible_text(*, provider_config, messages, temperature=0):
    return model_task_api_call("smart_summary_rewrite", provider_config=provider_config, messages=messages, execute=True, temperature=temperature, write=False)

SCHEMA = "video_knowledge_pipeline.smart_summary_codex.v1"
QUALITY_SCHEMA = "video_knowledge_pipeline.smart_summary_quality.v1"

CODEX_FILENAMES = (
    "exports/smart-summary.codex.md",
    "smart-summary.codex.md",
    "codex-smart-summary.md",
)
REQUIRED_HEADINGS = (
    "## 基本信息",
    "## 一句话概览",
    "## 核心主题",
    "## 分段总结",
    "## 关键观点",
    "## 可执行动作清单",
    "## 高频话术",
    "## 待复核点",
)
SUMMARY_MAIN_SECTION_NUMBERS = {
    "基本信息": "1",
    "一句话概览": "2",
    "核心主题": "3",
    "分段总结": "4",
    "关键观点": "5",
    "证据引用": "6",
    "可执行动作清单": "7",
    "高频话术": "8",
    "待复核点": "9",
}
INTERVIEW_REQUIRED_HEADINGS = (
    "## 基本信息",
    "## 一句话概览",
    "## 核心主题 / 事实主线",
    "## 事实时间线",
    "## 受访者原话与感受",
    "## 明确后续事项",
    "## 原话摘录",
    "## 待核实事项 / 隐私与发布边界",
)
BAD_OVERVIEW_FRAGMENTS = ("所以啊、", "如说、", "一下这个", "一个共同点就", "customer、", "方案、customer")
EVIDENCE_BOUNDARY = (
    "> 证据边界：本总结仅依据已入库的文本、时间轴与视觉证据；"
    "未入库或未核验的画面、课件和条款细节均为待复核，不应作为确定事实。"
)


def numbered_summary_heading(title: str, *, level: int = 2, number: str | None = None) -> str:
    """Render an Arabic-numbered summary heading with a stable hierarchy."""

    clean_title = str(title or "").strip()
    clean_number = str(number or SUMMARY_MAIN_SECTION_NUMBERS.get(clean_title) or "").strip()
    prefix = "#" * max(1, int(level))
    return f"{prefix} {clean_number + ' ' if clean_number else ''}{clean_title}".rstrip()


def generate_smart_summary_with_codex(
    bundle_dir: str | Path,
    *,
    input_md: str | Path | None = None,
    handoff_id: str = "",
    agent_available: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Generate, install, and validate a local Codex-style smart summary.

    This command is the stable handoff for using Codex as the first LLM layer.
    It does not prohibit online LLMs; provider integrations consume the same
    evidence pack and quality gate. If input_md is provided it installs that
    Codex/LLM Markdown. Without input_md it only prepares the handoff/status and
    never fabricates a rule-composed summary candidate.
    """

    root = Path(bundle_dir).expanduser().resolve()
    from .transcript_semantic_transaction import summary_transaction_status
    pending = summary_transaction_status(root)
    if not pending['passed']:
        return {'schema': SCHEMA, 'bundle_dir': str(root), 'status': 'blocked_install_transaction',
                'ok': False, 'installation_transaction': pending,
                'quality': {'passed': False, 'status': 'blocked_install_transaction'},
                'next_actions': ['Explicitly restore the interrupted summary transaction first.']}
    from .smart_summary_handoff import (
        handoff_status, prepare_summary_handoff, record_summary_install,
        register_review_candidate, validate_summary_submission,
    )
    submission = {}
    if input_md:
        source = Path(input_md).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"input markdown not found: {source}")
        submission = validate_summary_submission(root, source, handoff_id=handoff_id)
        if not submission["passed"]:
            return {"schema": SCHEMA, "bundle_dir": str(root), "status": submission["status"],
                    "workflow_status": "blocked", "ok": False, "candidate_binding": submission,
                    "next_actions": ["Prepare a fresh handoff and review the candidate against its bound inputs."],
                    "write": write}
    else:
        current = handoff_status(root)
        if current.get("freshness") == "fresh" and not _existing_codex_summary(root):
            current = {**current, "agent_available": bool(agent_available)}
            return {"schema": SCHEMA, "bundle_dir": str(root), "status": "needs_llm_rewrite",
                    "workflow_status": current["status"], "ok": False, "handoff": current,
                    "quality": _missing_quality(root), "next_actions": current.get("next_actions", []),
                    "reused": True, "write": write}
    production_gate = evaluate_production_artifact_gate(
        root,
        artifact_kind="smart_summary",
        write=write,
    )
    if input_md and not production_gate.get("formal_generation_allowed"):
        candidate = register_review_candidate(root, source, gate=production_gate, write=write)
        return {
            "schema": SCHEMA,
            "bundle_dir": str(root),
            "status": "blocked_by_production_artifact_gate",
            "ok": False,
            "workflow_status": "review-required",
            "candidate": candidate,
            "installed_from": "",
            "production_artifact_gate": production_gate,
            "quality": {"passed": False, "status": "blocked_review_required"},
            "next_actions": candidate["next_actions"],
        }
    exports = root / "exports"
    prompt_path = exports / "smart-summary-codex-prompt.md"
    target = exports / "smart-summary.codex.md"
    installed_from = ""
    if input_md:
        source = Path(input_md).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"input markdown not found: {source}")
        from .summary_claim_review import summary_claim_review_quality_gate
        claim_gate = summary_claim_review_quality_gate(root, source.read_text(encoding="utf-8-sig"))
        if claim_gate.get("evaluated") and not claim_gate.get("passed"):
            candidate = register_review_candidate(root, source, gate=production_gate,
                validation=claim_gate, write=write)
            return {"schema": SCHEMA, "bundle_dir": str(root), "status": "blocked_claim_review", "ok": False,
                    "workflow_status": "review-required", "candidate": candidate,
                    "installed_from": "", "claim_review": claim_gate,
                    "quality": {"passed": False, "status": "blocked_claim_review"},
                    "next_actions": [claim_gate["detail"]], "write": write}
        result = _install_summary_candidate(root, source=source, write=write,
            handoff_snapshot=submission.get("dependency_snapshot"))
        if result.get("status") == "blocked_quality":
            result["candidate"] = register_review_candidate(root, source, gate=production_gate,
                validation=result.get("quality"), write=write)
            result["workflow_status"] = "review-required"
        record_summary_install(root, result, write=write)
        return result
    else:
        installed_from = "llm_output_required"
    codex_path = _existing_codex_summary(root)
    status = "ready" if codex_path else "needs_llm_rewrite"
    input_pack_refresh: dict[str, Any] = {}
    chapter_pack_refresh: dict[str, Any] = {}
    dependency_snapshot: dict[str, Any] = {}
    if write:
        # Always prepare the evidence inputs for the LLM. These deterministic stages
        # may organize evidence, but they do not compose final summary prose.
        input_pack_refresh = _refresh_smart_summary_input_pack(root, required=False)
        try:
            chapter_pack_refresh = build_smart_summary_chapter_pack(root, write=True)
        except Exception as exc:
            chapter_pack_refresh = {"status": "unavailable", "error": str(exc)}
        # A status refresh must never rebind an installed old summary to new
        # evidence. Only the validated installation transaction writes a snapshot.
    quality = smart_summary_quality_check(root, summary_path=codex_path or target, require_codex=True, write=write) if codex_path else _missing_quality(root)
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": status,
        "installed_from": installed_from,
        "smart_summary_codex_path": str(codex_path or target),
        "prompt_path": str(prompt_path),
        "prompt_exists": prompt_path.exists(),
        "quality": quality,
        "input_pack_refresh": input_pack_refresh,
        "chapter_pack_refresh": chapter_pack_refresh,
        "dependency_snapshot": dependency_snapshot,
        "model_strategy": "codex_first_llm_layer",
        "llm_policy": "Smart-summary generation is LLM-only. Rules may prepare evidence and validate facts, but must not compose the final summary. Use Codex/agent output via --input-md or an online LLM provider.",
        "next_actions": _next_actions(root, status=status, quality=quality),
        "write": write,
        "updated_at": now_iso(),
    }
    if not codex_path:
        try:
            from uuid import uuid4
            candidate_path = root / "runs" / "smart-summary-candidates" / uuid4().hex / "candidate.md"
            if write:
                prepare_smart_summary_llm_rewrite(root, provider="codex_manual", output_md=candidate_path, write=True)
            handoff = prepare_summary_handoff(root, agent_available=agent_available,
                candidate_path=candidate_path, write=write)
        except (OSError, ValueError) as exc:
            handoff = {"status": "blocked", "ok": False, "reason": "summary_handoff_preparation_failed",
                       "detail": str(exc), "next_actions": ["Repair the evidence pack before writing a summary candidate."]}
        result.update(handoff=handoff, workflow_status=handoff["status"], ok=False,
                      next_actions=handoff["next_actions"])
    else:
        result.update(workflow_status="completed" if quality.get("passed") else "review-required",
                      ok=quality.get("passed") is True)
    if write:
        write_json(exports / "smart-summary-codex-status.json", result)
        (exports / "smart-summary-codex-status.md").write_text(_render_status_markdown(result), encoding="utf-8")
    run_artifact = _register_smart_summary_codex_run(root, result, input_md=input_md, write=write)
    result["run_artifact"] = run_artifact
    if write:
        write_json(exports / "smart-summary-codex-status.json", result)
        (exports / "smart-summary-codex-status.md").write_text(_render_status_markdown(result), encoding="utf-8")
    return result


def _install_summary_candidate(root: Path, *, source: Path, write: bool,
                               handoff_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate one immutable candidate, then reuse the existing six-file journal."""
    from .courseware_material import safe_path
    from .codex_vision_session import _write_bytes_atomic
    from .storage import bundle_write_lock
    from .transcript_semantic_transaction import commit_transcript_semantic_artifacts, _profile_targets
    if source.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('summary_candidate_budget')
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    checked = source
    if write:
        # Identical prose can be reviewed against new evidence. Retain the old
        # immutable admission copy; bind the new copy to this exact handoff so
        # its older file timestamp cannot invalidate the fresh submission.
        admission_key = source_sha
        if handoff_snapshot:
            admission_key = hashlib.sha256((source_sha + ':' + str(handoff_snapshot['snapshot_sha256'])).encode()).hexdigest()
        checked = safe_path(root, f'runs/summary-install-inputs/{admission_key}/smart-summary.codex.md', exists=False)
        if checked.exists():
            if checked.read_bytes() != raw:
                raise ValueError('summary_install_input_drift')
        else:
            _write_bytes_atomic(checked, raw)
            stat = source.stat()
            os.utime(checked, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with bundle_write_lock(root, operation='summary_candidate_install') if write else _no_install_lock():
        if handoff_snapshot and not validate_dependency_snapshot(root, handoff_snapshot)["passed"]:
            raise ValueError("summary_handoff_input_drift")
        input_refresh = _refresh_smart_summary_input_pack(root, required=False) if write and not handoff_snapshot else {}
        chapter_refresh = {}
        if write and not handoff_snapshot:
            try:
                chapter_refresh = build_smart_summary_chapter_pack(root, write=True)
            except Exception as exc:
                chapter_refresh = {'status': 'unavailable', 'error': str(exc)}
        try:
            dependency = write_smart_summary_dependency_snapshot(root, write=False)
        except (FileNotFoundError, ValueError) as exc:
            dependency = {'status': 'unavailable', 'detail': str(exc)}
        expected = {row['path']: row['sha256'] for row in dependency.get('snapshot', {}).get('inputs', [])}
        for name in ('manifest.json', 'summary-claim-review.json', 'exports/smart-summary-reader-plan.json',
                     'exports/smart-summary-chapter-fact-pack.json', 'exports/smart-summary-input-pack.json',
                     'exports/human-key-points.json', 'quality-benchmark-key-points.json',
                     'transcript-quality-gate.json', 'production-artifact-gate.json'):
            path = safe_path(root, name, exists=False)
            expected[name] = sha256_file(path) if path.is_file() else ''
        for name in _profile_targets('summary_install'):
            path = safe_path(root, name, exists=False)
            expected[name] = sha256_file(path) if path.is_file() else ''
        # The prepared dependency hashes take precedence over values sampled
        # during admission. A race must not silently rebind old candidate prose.
        expected.update({row['path']: row['sha256'] for row in (handoff_snapshot or {}).get('inputs', [])})
        quality = smart_summary_quality_check(root, summary_path=checked, require_codex=True, write=False,
            dependency_snapshot=dependency.get('snapshot'))
        result = {'schema': SCHEMA, 'bundle_dir': str(root), 'status': 'ready' if quality.get('passed') else 'blocked_quality',
            'ok': quality.get('passed') is True, 'installed_from': str(source),
            'smart_summary_codex_path': str(root / 'exports/smart-summary.codex.md'),
            'prompt_path': str(root / 'exports/smart-summary-codex-prompt.md'),
            'quality': quality, 'input_pack_refresh': input_refresh, 'chapter_pack_refresh': chapter_refresh,
            'dependency_snapshot': dependency, 'candidate_sha256': source_sha,
            'candidate_checked_path': str(checked), 'model_strategy': 'codex_first_llm_layer',
            'next_actions': _next_actions(root, status='ready', quality=quality), 'write': write, 'updated_at': now_iso()}
        if not quality.get('passed') or not write:
            if write:
                # Preserve rejected validation beside the immutable original,
                # never replace the previous installed quality/status reports.
                validation_id = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                rejection = safe_path(root, f'runs/summary-install-inputs/{source_sha}/{validation_id}.json', exists=False)
                if not rejection.exists():
                    write_json(rejection, result)
                result['validation_receipt'] = str(rejection)
            return result
        if checked.read_bytes() != raw:
            raise ValueError('summary_install_input_drift')
        def encode(value):
            return json.dumps(value, ensure_ascii=False, indent=2)
        artifacts = {'exports/smart-summary.codex.md': raw,
            'exports/smart-summary-quality.json': encode(quality),
            'exports/smart-summary-quality.md': _render_quality_markdown(quality),
            'exports/smart-summary-codex-status.json': encode(result),
            'exports/smart-summary-codex-status.md': _render_status_markdown(result)}
        if dependency.get('snapshot'):
            artifacts['exports/smart-summary-dependency-snapshot.json'] = encode(dependency['snapshot'])
        expected[checked.relative_to(root).as_posix()] = source_sha
        transaction = commit_transcript_semantic_artifacts(root, artifacts=artifacts, expected_inputs=expected,
            operation_identity={'pack_sha256': source_sha, 'quality_sha256': hashlib.sha256(encode(quality).encode()).hexdigest()},
            profile='summary_install')
        result['installation_transaction'] = transaction
    try:
        result['run_artifact'] = _register_smart_summary_codex_run(root, result, input_md=source, write=True)
    except Exception as exc:
        result.update(status='installed_registry_pending', ok=False,
                      next_actions=[f'Candidate committed; repair run registry without regeneration: {exc}'])
    return result


def _no_install_lock():
    from contextlib import nullcontext
    return nullcontext()


def _refresh_smart_summary_input_pack(root: Path, *, required: bool) -> dict[str, Any]:
    """Refresh the smart-summary input pack when possible without hiding quality gates.

    Section/manual LLM installs can happen in small test or handoff bundles that
    already contain a corrected transcript pointer but not a full timeline. In
    those cases the summary quality check should still run and fail or pass on
    its own gates instead of crashing before it can report a useful status.
    """

    try:
        pack = build_smart_summary_input_pack(root, write=True)
    except (FileNotFoundError, ValueError) as exc:
        if required:
            raise
        return {"status": "skipped_incomplete_bundle", "ok": False, "detail": str(exc)}
    return {
        "status": "refreshed",
        "ok": True,
        "path": str(root / "exports" / "smart-summary-input-pack.json"),
        "transcript_source": pack.get("transcript_source") if isinstance(pack, dict) else "",
    }



def run_smart_summary_llm_rewrite(
    bundle_dir: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    execute: bool = False,
    max_input_chars: int = 60000,
    temperature: float = 0,
    install: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Run the real text LLM rewrite layer for smart-summary.md.

    This is the executable counterpart to prepare_smart_summary_llm_rewrite:
    it reuses the same evidence pack, calls the existing OpenAI-compatible text
    gateway only when execute=True, writes exports/smart-summary.llm.md, and then
    installs it through generate_smart_summary_with_codex so the same quality
    gate and final-export logic apply.
    """

    root = Path(bundle_dir).expanduser().resolve()
    exports = root / "exports"
    production_gate = evaluate_production_artifact_gate(
        root,
        artifact_kind="smart_summary",
        write=write,
    )
    if execute and not production_gate.get("formal_generation_allowed"):
        result = {
            "schema": "video_knowledge_pipeline.smart_summary_llm_rewrite_run.v1",
            "bundle_dir": str(root),
            "execute": True,
            "status": "blocked_by_production_artifact_gate",
            "ok": False,
            "production_artifact_gate": production_gate,
            "provider_call_performed": False,
            "next_actions": ["Resolve production-artifact-gate.md; no Provider call was made."],
        }
        return _write_llm_run_result(root, result, write=write)
    cfg = resolve_text_provider_config(provider_config)
    title = _smart_summary_title(root)
    content_profile = resolve_content_profile(root)
    interview_profile = content_profile.get("profile_id") in {
        "interview-v1",
        "medical-insurance-interview-v1",
    }
    handoff = prepare_smart_summary_llm_rewrite(root, provider=str(cfg.get("provider") or "openai_compatible"), write=write)
    prompt_path = Path(str(handoff.get("prompt_path") or exports / "smart-summary-llm-rewrite-pack.md"))
    output_path = exports / "smart-summary.llm.md"
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
    if build_courseware_material(root, write=False).get("active"):
        if len(prompt_text) > max(8000, int(max_input_chars or 60000)):
            return _write_llm_run_result(root, {
                "schema": "video_knowledge_pipeline.smart_summary_llm_rewrite_run.v1",
                "ok": False, "status": "blocked_courseware_input_budget", "provider_call_performed": False,
                "next_actions": ["Use section workflow or increase explicit input budget; courseware must not be truncated."],
            }, write=write)
    else:
        prompt_text = _trim_llm_prompt(prompt_text, max_input_chars=max_input_chars)
    messages = [
        {
            "role": "system",
            "content": (
                "你是采访事实摘要编辑器。必须忠实区分采访者与受访者，只写事实时间线、原话与感受、已确认信息、待核实事项和隐私发布边界；不得把个人医疗保险经历扩写为建议、方法论、高频话术或可复用表达。输出 Markdown，不要编造视觉证据。"
                if interview_profile
                else "你是视频课程智能总结编辑器。必须输出可直接保存的 Markdown 成品，不要解释过程，不要包代码块。分段时间请统一使用 HH:MM:SS。必须保留不确定性边界，不要编造视觉证据。"
            ),
        },
        {"role": "user", "content": prompt_text},
    ]
    public_provider = _public_text_provider_config(cfg)
    result: dict[str, Any] = {
        "schema": "video_knowledge_pipeline.smart_summary_llm_rewrite_run.v1",
        "bundle_dir": str(root),
        "execute": bool(execute),
        "status": "planned",
        "ok": True,
        "provider": public_provider,
        "request_plan": {
            "url": redact_url_secrets(openai_compatible_chat_completions_url(cfg)),
            "model": cfg.get("model"),
            "message_count": len(messages),
            "temperature": temperature,
            "max_input_chars": int(max_input_chars),
        },
        "artifacts": {
            "prompt_path": str(prompt_path),
            "expected_output_path": str(output_path),
            "status_json": str(exports / "smart-summary-llm-run-status.json"),
            "status_markdown": str(exports / "smart-summary-llm-run-status.md"),
        },
        "operator_boundary": {
            "default_preview_only": True,
            "execute_required_for_network_call": True,
            "provider_config_runtime_only": True,
            "secrets_redacted": True,
            "installs_final_only_with_llm_marker": True,
        },
        "updated_at": now_iso(),
    }
    if not execute:
        result["next_actions"] = [
            "Rerun with --execute after provider credentials and data boundary are configured.",
            "If the provider is unavailable, open smart-summary-llm-rewrite-pack.md in Codex and write exports/smart-summary.llm.md manually.",
        ]
        return _write_llm_run_result(root, result, write=write)

    try:
        call = call_openai_compatible_text(provider_config=cfg, messages=messages, temperature=temperature)
    except (OSError, TimeoutError) as exc:
        call = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    result["ok"] = bool(call.get("ok"))
    result["status"] = "ok" if call.get("ok") else str(call.get("error") or "failed")
    result["error"] = str(call.get("error") or "")
    if not call.get("ok"):
        recovery = generate_smart_summary_with_codex(root, write=write)
        result["handoff"] = recovery.get("handoff") or {
            "status": recovery.get("workflow_status", "blocked"), "next_actions": recovery.get("next_actions", [])}
        result["workflow_status"] = result["handoff"]["status"]
        result["next_actions"] = result["handoff"]["next_actions"]
        return _write_llm_run_result(root, result, write=write)

    content = _normalise_llm_markdown(str(call.get("content") or ""), title=title)
    if write:
        exports.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    result["content_chars"] = len(content)
    result["artifacts"]["output_path"] = str(output_path)
    if install and not write:
        result.update(status="validated_not_installed", ok=False,
            install_result={"status": "not_attempted", "reason": "write_disabled", "ok": False},
            next_actions=["Save and review the generated candidate before installation; no stored candidate was installed or checked."])
        return _write_llm_run_result(root, result, write=False)
    if install:
        install_result = generate_smart_summary_with_codex(root, input_md=output_path, write=write)
        result["install_result"] = install_result
        result["quality"] = install_result.get("quality", {}) if isinstance(install_result, dict) else {}
        result["ok"] = bool(result["quality"].get("passed") and (install_result or {}).get("ok", True))
        result["status"] = "installed" if result["ok"] else str((install_result or {}).get("status") or "installed_quality_failed")
        result["next_actions"] = _llm_run_next_actions(root, install_result)
    else:
        result["next_actions"] = [f"Install with generate-smart-summary-with-codex --input-md {output_path}"]
    return _write_llm_run_result(root, result, write=write)


def _smart_summary_title(root: Path) -> str:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if isinstance(manifest, dict) and str(manifest.get("title") or "").strip():
        return str(manifest.get("title")).strip()
    return root.name


def _trim_llm_prompt(prompt_text: str, *, max_input_chars: int) -> str:
    limit = max(8000, int(max_input_chars or 60000))
    text = str(prompt_text or "")
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.62)]
    tail = text[-int(limit * 0.33) :]
    return head.rstrip() + "\n\n## Middle Omitted By VKP Budget Guard\n\n中间证据因输入预算被截断；请优先保持全片结构、时间覆盖和待复核边界。\n\n" + tail.lstrip()


def _normalise_llm_markdown(content: str, *, title: str) -> str:
    text = _strip_markdown_fence(str(content or "").strip())
    if not text:
        text = f"# {title} - 智能总结\n"
    if not text.lstrip().startswith("#"):
        text = f"# {title} - 智能总结\n\n" + text
    if EVIDENCE_BOUNDARY not in text:
        lines = text.splitlines()
        insert_at = 1 if lines and lines[0].startswith("#") else 0
        lines.insert(insert_at, "")
        lines.insert(insert_at + 1, EVIDENCE_BOUNDARY)
        text = "\n".join(lines)
    if "生成方式：`codex_llm_rewrite_final`" not in text and "生成方式：`codex_final`" not in text:
        lines = text.splitlines()
        insert_at = 1 if lines and lines[0].startswith("#") else 0
        lines.insert(insert_at, "")
        lines.insert(insert_at + 1, "生成方式：`codex_llm_rewrite_final`。")
        text = "\n".join(lines)
    return text.rstrip() + "\n"


def _strip_markdown_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return value


def _public_text_provider_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": cfg.get("provider", ""),
        "model": cfg.get("model", ""),
        "base_url": redact_url_secrets(str(cfg.get("base_url") or "")),
        "api_key_required": provider_requires_api_key(cfg),
        "api_key_configured": bool(cfg.get("api_key")),
        "interface": cfg.get("interface", "openai_chat_completions"),
    }


def _write_llm_run_result(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    if not write:
        return result
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    write_json(exports / "smart-summary-llm-run-status.json", result)
    (exports / "smart-summary-llm-run-status.md").write_text(_render_llm_run_status_markdown(result), encoding="utf-8")
    return result


def _render_llm_run_status_markdown(result: dict[str, Any]) -> str:
    provider = result.get("provider") if isinstance(result.get("provider"), dict) else {}
    lines = [
        "# Smart Summary LLM Run Status",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Execute: `{result.get('execute')}`",
        f"- Provider: `{provider.get('provider', '')}` / `{provider.get('model', '')}`",
        f"- API key configured: `{provider.get('api_key_configured', False)}`",
        "",
        "## Artifacts",
        "",
    ]
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")
    if result.get("error"):
        lines.extend(["", "## Error", "", str(result.get("error"))])
    actions = [str(value) for value in (result.get("next_actions") or []) if str(value)]
    if actions:
        lines.extend(["", "## Next Actions", ""])
        lines.extend([f"- {action}" for action in actions])
    return "\n".join(lines).rstrip() + "\n"


def _llm_run_next_actions(root: Path, install_result: dict[str, Any] | None) -> list[str]:
    quality = install_result.get("quality") if isinstance(install_result, dict) and isinstance(install_result.get("quality"), dict) else {}
    if isinstance(install_result, dict) and install_result.get("ok") is False and install_result.get("next_actions"):
        return list(install_result["next_actions"])
    if quality.get("passed"):
        return [f"Run .\\scripts\\video-knowledge.ps1 export-knowledge-note {_ps_quote(str(root))} to refresh final readable exports."]
    return [
        "Review exports/smart-summary-quality.md; revise exports/smart-summary.llm.md if needed.",
        f"Run .\\scripts\\video-knowledge.ps1 generate-smart-summary-with-codex {_ps_quote(str(root))} --input-md {_ps_quote(str(root / 'exports' / 'smart-summary.llm.md'))}",
        f"Run .\\scripts\\video-knowledge.ps1 export-knowledge-note {_ps_quote(str(root))}",
    ]

def prepare_smart_summary_llm_rewrite(
    bundle_dir: str | Path,
    *,
    provider: str = "codex_manual",
    output_md: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Prepare the real LLM rewrite handoff for final smart-summary generation.

    The default provider is codex_manual: Codex reads the generated pack and writes
    exports/smart-summary.llm.md. The existing generate-smart-summary-with-codex
    command then installs that Markdown through the same quality gate used by
    online LLM providers later.
    """
    root = Path(bundle_dir).expanduser().resolve()
    exports = root / "exports"
    pack = _load_or_build_input_pack(root)
    memory_pack = _load_or_build_long_memory_pack(root)
    chapter_pack = build_smart_summary_chapter_pack(root, title=str(pack.get("title") or root.name), write=write)
    baseline_path = exports / "smart-summary.md"
    prompt_path = exports / "smart-summary-llm-rewrite-pack.md"
    template_path = exports / "smart-summary.llm.todo.md"
    output_path = (root / output_md).resolve() if output_md else exports / "smart-summary.llm.md"
    if not output_path.is_relative_to(root):
        raise ValueError("summary candidate must be inside the Bundle")
    status_path = exports / "smart-summary-llm-rewrite-status.json"
    markdown_status_path = exports / "smart-summary-llm-rewrite-status.md"
    title = str(pack.get("title") or root.name).strip()
    prompt = _render_llm_rewrite_pack(root, title=title, provider=provider, pack=pack, memory_pack=memory_pack, chapter_pack=chapter_pack, baseline_path=baseline_path, output_path=output_path)
    template = _render_llm_rewrite_template(title)
    result = {
        "schema": "video_knowledge_pipeline.smart_summary_llm_rewrite.v1",
        "bundle_dir": str(root),
        "provider": provider,
        "status": "ready_for_codex_rewrite" if provider == "codex_manual" else "ready_for_llm_provider",
        "prompt_path": str(prompt_path),
        "template_path": str(template_path),
        "expected_output_path": str(output_path),
        "install_command": f".\\scripts\\video-knowledge.ps1 generate-smart-summary-with-codex {_ps_quote(str(root))} --input-md {_ps_quote(str(output_path))}",
        "quality_command": f".\\scripts\\video-knowledge.ps1 smart-summary-quality-check {_ps_quote(str(root))} --require-codex",
        "export_command": f".\\scripts\\video-knowledge.ps1 export-knowledge-note {_ps_quote(str(root))}",
        "operator_boundary": {
            "default_provider": "codex_manual",
            "cloud_llm_calls": "not_executed_by_this_command",
            "writes_final_summary": False,
            "requires_human_or_codex_output": True,
        },
        "write": write,
        "updated_at": now_iso(),
    }
    if write:
        exports.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
        template_path.write_text(template, encoding="utf-8")
        write_json(status_path, result)
        markdown_status_path.write_text(_render_llm_rewrite_status_markdown(result), encoding="utf-8")
    result["run_artifact"] = register_bundle_run(
        root,
        run_type="smart_summary_llm_rewrite",
        run_id="smart-summary-llm-rewrite",
        status="needs_input",
        title="Smart summary LLM rewrite handoff",
        summary=f"provider={provider}; prompt ready; waiting for {output_path.name}.",
        inputs={"baseline_summary": str(baseline_path), "input_pack": str(exports / "smart-summary-input-pack.md"), "chapter_pack": str(exports / "smart-summary-chapters.md")},
        parameters={"provider": provider, "write": bool(write)},
        artifacts=[
            {"key": "llm_rewrite_pack", "path": str(prompt_path), "description": "Prompt and evidence pack for Codex/LLM final rewrite."},
            {"key": "llm_rewrite_template", "path": str(template_path), "description": "Markdown template Codex/LLM should fill."},
            {"key": "llm_expected_output", "path": str(output_path), "description": "Expected final LLM rewrite output to install."},
            {"key": "status_json", "path": str(status_path), "description": "Machine-readable LLM rewrite handoff status."},
        ],
        failed_items=[{"id": "smart_summary_llm_output", "reason": "needs_codex_or_llm_rewrite", "detail": f"Write {output_path}, then install it with generate-smart-summary-with-codex --input-md."}],
        retry_command=f".\\scripts\\video-knowledge.ps1 prepare-smart-summary-llm-rewrite {_ps_quote(str(root))}",
        next_actions=[
            f"Open {prompt_path}",
            f"Write final Markdown to {output_path}",
            f"Run {result['install_command']}",
            f"Run {result['quality_command']}",
        ],
        operator_boundary=result["operator_boundary"],
        write=write,
    )
    if write:
        write_json(status_path, result)
        markdown_status_path.write_text(_render_llm_rewrite_status_markdown(result), encoding="utf-8")
    return result


def _render_llm_rewrite_pack(root: Path, *, title: str, provider: str, pack: dict[str, Any], memory_pack: dict[str, Any], chapter_pack: dict[str, Any], baseline_path: Path, output_path: Path) -> str:
    chapters = chapter_pack.get("chapters") if isinstance(chapter_pack.get("chapters"), list) else []
    course_map = chapter_pack.get("course_map") if isinstance(chapter_pack.get("course_map"), dict) else {}
    transcript_semantic = pack.get("transcript_semantic_correction") if isinstance(pack.get("transcript_semantic_correction"), dict) else {}
    transcript_chars = _transcript_char_count(pack.get("transcript_source"))
    minimum_output_chars = int(transcript_chars * 0.10) if transcript_chars >= 12000 else 0
    baseline_excerpt = baseline_path.read_text(encoding="utf-8")[:6000] if baseline_path.exists() else ""
    lines = [
        f"# Smart Summary LLM Rewrite Pack - {title}",
        "",
        f"Provider mode: `{provider}`. 当前默认由 Codex 手动扮演 LLM 改写层；不要调用云端，除非用户显式批准 provider 执行。",
        "",
        "## Output Contract",
        "",
        f"Write the final Markdown to: `{output_path}`",
        "",
        "Required headings:",
        "",
    ]
    for heading in REQUIRED_HEADINGS:
        lines.append(f"- `{heading}`")
    lines.extend([
        "",
        "## Rewrite Rules",
        "",
        "- 这是成品智能总结，不是证据流水账，也不是 ASR 摘录。",
        "- 主文不要出现 document_visual、semantic_frame、timeline= 等工程标签；这些只能留在 Citation Digest 附录。",
        "- 分段总结要用编辑后的中文概括，每段说明主题、讲师判断、例子/方法、行动含义。",
        "- 关键观点和动作清单要跨全片均衡，不能只抽前几分钟。",
        "- 章节证据必须覆盖视频结尾；不要因章节数量而静默截断长视频。",
        "- 默认使用阿拉伯数字层级编号：主栏目使用 `1`、`2`、`3`，栏目内条目使用 `1.1.1`、`1.1.2` 这类结构；禁止使用甲乙丙丁或章节一/二/三作为默认序号。",
        "- 关键观点、可执行动作清单和高频话术的每一项都要带 `HH:MM:SS` 来源时间；每个栏目至少覆盖视频的两个不同时间区间，并包含后半段证据。",
        "- 保留时间戳用于导航，但不要让时间戳压过内容。",
        "- 不要依据第一人称、语气或模糊上下文推断讲师姓名、机构、平台归属或资历；转写未明确说明时标为待复核。",
        "- 视觉证据未执行或待复核时必须如实说明，不能把屏幕细节写成确定事实。",
        "- 使用纠正版转写中的高置信术语；低置信内容放到待复核点。",
        "",
        "## Current Evidence Status",
        "",
        f"- Transcript source: `{pack.get('transcript_source') or 'unknown'}`",
        f"- Corrected transcript characters: `{transcript_chars or 'unknown'}`",
        f"- Semantic correction status: `{transcript_semantic.get('final_status') or transcript_semantic.get('status') or 'unknown'}`",
        f"- Long memory: `{memory_pack.get('summary', {}).get('short_memories', 0)}` short / `{memory_pack.get('summary', {}).get('long_memories', 0)}` long",
        "",
        "## Course Map",
        "",
        f"- Main question: {course_map.get('main_question') or ''}",
        f"- Mainline: {course_map.get('mainline') or ''}",
        "",
        "## Chapter Evidence",
        "",
    ])
    if minimum_output_chars:
        lines.extend([
            "## Long-video Content Density Requirement",
            "",
            f"- This corrected transcript contains about {transcript_chars} characters. Write at least {minimum_output_chars} Chinese characters of substantive final-summary prose so the result remains useful for a long video; expand evidence-grounded analysis rather than padding.",
            "",
        ])
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        time_range = f"{format_timestamp(_seconds(chapter.get('start')))} - {format_timestamp(_seconds(chapter.get('end')))}"
        title_text = _clean_llm_main_text(chapter.get("title") or "", limit=80)
        sentences = [_clean_llm_main_text(value, limit=160) for value in chapter.get("summary_sentences") or []]
        sentences = [value for value in sentences if value]
        lines.append(f"### {time_range} {title_text}")
        for sentence in sentences[:5]:
            lines.append(f"- {sentence}")
        lines.append("")
    if baseline_excerpt:
        lines.extend(["## Current Baseline Summary Excerpt", "", baseline_excerpt, ""])
    material = pack.get("courseware_material", {"active": False})
    if material.get("active"):
        assert_material_fresh(root, material)
        lines.extend([render_courseware_model_input(material), ""])
    lines.extend([
        "## Final Instruction For Codex",
        "",
        "请直接输出最终 smart-summary Markdown，不要解释过程，不要包代码块。",
    ])
    return "\n".join(lines).rstrip() + "\n"



def _transcript_char_count(source: Any) -> int:
    value = str(source or "").strip()
    if not value:
        return 0
    path = Path(value).expanduser()
    if not path.exists():
        return 0
    try:
        return len("".join(str(cue.text or "") for cue in parse_transcript(path)))
    except Exception:
        return 0

def _render_llm_rewrite_template(title: str) -> str:
    lines = [f"# {title} - 智能总结", "", "生成方式：`codex_llm_rewrite_final`。", ""]
    for heading in REQUIRED_HEADINGS:
        lines.extend([heading, "", "TODO", ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_llm_rewrite_status_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Smart Summary LLM Rewrite Status",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Provider: `{result.get('provider')}`",
        f"- Prompt: `{result.get('prompt_path')}`",
        f"- Template: `{result.get('template_path')}`",
        f"- Expected output: `{result.get('expected_output_path')}`",
        "",
        "## Commands",
        "",
        f"- Install: `{result.get('install_command')}`",
        f"- Quality: `{result.get('quality_command')}`",
        f"- Export: `{result.get('export_command')}`",
    ]
    return "\n".join(lines).rstrip() + "\n"

def _register_smart_summary_codex_run(root: Path, result: dict[str, Any], *, input_md: str | Path | None, write: bool) -> dict[str, Any]:
    quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
    status = str(result.get("status") or "")
    if status != "ready":
        run_status = "needs_execution"
    elif bool(quality.get("passed")):
        run_status = "completed"
    else:
        run_status = "needs_retry"
    failed_items = _smart_summary_failed_items(quality)
    summary = (
        f"status={status}; quality={quality.get('status', '')}; "
        f"passed={bool(quality.get('passed'))}; installed_from={result.get('installed_from', '')}."
    )
    return register_bundle_run(
        root,
        run_type="smart_summary_codex",
        run_id="smart-summary-codex",
        status=run_status,
        title="Smart summary Codex generation",
        summary=summary,
        inputs={
            "input_md": str(Path(input_md).expanduser().resolve()) if input_md else "",
            "prompt_path": str(result.get("prompt_path") or ""),
            "model_strategy": str(result.get("model_strategy") or ""),
        },
        parameters={
            "require_codex": True,
            "write": bool(write),
        },
        artifacts=[
            {"key": "smart_summary_codex", "path": str(result.get("smart_summary_codex_path") or root / "exports" / "smart-summary.codex.md"), "description": "Final Codex/LLM smart summary candidate."},
            {"key": "codex_prompt", "path": str(result.get("prompt_path") or root / "exports" / "smart-summary-codex-prompt.md"), "description": "Prompt/evidence handoff for Codex smart-summary rewriting."},
            {"key": "status_json", "path": str(root / "exports" / "smart-summary-codex-status.json"), "description": "Machine-readable Codex summary generation status."},
            {"key": "status_markdown", "path": str(root / "exports" / "smart-summary-codex-status.md"), "description": "Human-readable Codex summary generation status."},
            {"key": "quality_json", "path": str(root / "exports" / "smart-summary-quality.json"), "description": "Machine-readable smart-summary quality gate."},
            {"key": "quality_markdown", "path": str(root / "exports" / "smart-summary-quality.md"), "description": "Human-readable smart-summary quality gate."},
            {"key": "input_pack", "path": str(root / "exports" / "smart-summary-input-pack.md"), "description": "Corrected transcript, term, and visual evidence input pack."},
            {"key": "long_video_memory_pack", "path": str(root / "exports" / "long-video-memory-pack.md"), "description": "MovieChat-style long-video memory pack for summary coverage."},
            {"key": "chapter_pack", "path": str(root / "exports" / "smart-summary-chapters.md"), "description": "Chapter-level evidence and course map for summary generation."},
            {"key": "course_map", "path": str(root / "exports" / "course-map.md"), "description": "Course map generated from chapter evidence."},
        ],
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 generate-smart-summary-with-codex {_ps_quote(str(root))}",
        next_actions=list(result.get("next_actions") or []),
        operator_boundary={
            "local_only": True,
            "no_cloud_call": True,
            "does_not_process_media": True,
            "codex_first_llm_layer": True,
            "online_llm_allowed_later_behind_same_quality_gate": True,
        },
        write=write,
    )


def _smart_summary_failed_items(quality: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for row in quality.get("checks") or []:
        if not isinstance(row, dict) or row.get("passed"):
            continue
        failed.append(
            {
                "id": str(row.get("key") or "quality_check"),
                "reason": str(row.get("key") or "quality_check_failed"),
                "detail": str(row.get("detail") or ""),
            }
        )
    if not failed and not quality.get("passed"):
        failed.append({"id": "smart_summary_quality", "reason": str(quality.get("status") or "failed"), "detail": "smart-summary quality gate did not pass"})
    return failed


def _semantic_correction_quality_gate(root: Path) -> dict[str, Any]:
    status = transcript_semantic_correction_status(root, write=False)
    targeted_evidence = _targeted_semantic_evidence_gate(root)
    value = str(status.get("status") or "missing_pack")
    candidate_count = int(status.get("candidate_count") or 0)
    accepted_count = int(status.get("accepted_decision_count") or 0)
    residual_total = int(status.get("final_residual_error_total") or 0)
    readable_status = str(status.get("readable_impact_status") or "missing")
    readable_residual_total = int(status.get("readable_required_residual_total") or 0)
    if targeted_evidence["required"]:
        passed = False
        detail = (
            f"{targeted_evidence['pending_candidate_count']} high-risk semantic candidate(s) still lack "
            "independent local ASR evidence; smart summary remains a draft"
        )
    elif value == "no_candidates":
        passed = True
        detail = "semantic correction found no suspicious ASR/subtitle candidates"
    elif value == "missing_pack":
        has_transcript = (root / "normalized-transcript.json").exists() or (root / "source-arbitrated-transcript.json").exists() or (root / "transcript.json").exists()
        passed = False
        detail = "semantic correction pack missing; smart summary may be exported as a draft, but run transcript-semantic-correction-pack before treating transcript correction as closed" if has_transcript else "semantic correction pack missing and no transcript source is available"
    elif value == "impact_passed" and residual_total == 0 and readable_status == "passed" and readable_residual_total == 0:
        passed = True
        detail = "semantic correction impact passed; accepted suspicious-word corrections reached full-transcript and smart-summary"
    elif accepted_count == 0 and residual_total == 0 and readable_status == "no_accepted_decisions" and readable_residual_total == 0:
        passed = True
        detail = (
            "semantic correction produced review candidates but no high-confidence accepted corrections; "
            f"candidates={candidate_count} remain optional review items and do not block final summary export"
        )
    elif value in {"needs_readable_impact_report", "readable_impact_needs_fix"} or readable_status in {"missing", "needs_fix"}:
        passed = False
        detail = f"semantic correction readable impact status={readable_status}; readable_residual={readable_residual_total}; run transcript-semantic-readable-impact-report before treating smart summary as final"
    else:
        passed = False
        detail = f"semantic correction status={value}; candidates={candidate_count}; accepted={accepted_count}; residual={residual_total}; run pack/validate/closure/export/impact before treating smart summary as final"
    return {
        "passed": passed,
        "status": value,
        "candidate_count": candidate_count,
        "accepted_decision_count": accepted_count,
        "review_required_count": int(status.get("review_required_count") or 0),
        "final_residual_error_total": residual_total,
        "readable_impact_status": readable_status,
        "readable_required_residual_total": readable_residual_total,
        "next_action_key": status.get("next_action_key", ""),
        "artifacts": status.get("artifacts", {}),
        "targeted_evidence": targeted_evidence,
        "detail": detail,
    }


def _targeted_semantic_evidence_gate(root: Path) -> dict[str, Any]:
    path = root / "local-targeted-asr-plan.json"
    value = read_json(path) if path.is_file() else {}
    if not isinstance(value, dict):
        return {"required": False, "status": "not_planned", "pending_candidate_count": 0}
    retry_plan = value.get("retry_plan") if isinstance(value.get("retry_plan"), dict) else {}
    windows = int(retry_plan.get("window_count") or value.get("selected_candidate_count") or 0)
    plan_status = str(value.get("status") or "missing")
    required = windows > 0 and plan_status in {"planned", "in_progress", "degraded", "failed"}
    return {"required": required, "status": plan_status, "pending_candidate_count": int(value.get("selected_candidate_count") or windows) if required else 0, "window_count": windows, "path": str(path)}


def _compression_target(
    *,
    transcript_chars: int,
    transcript_max: float,
    coverage_ratio: float,
    section_coverage_passed: bool,
) -> tuple[float, float]:
    """Use a modest minimum for short sources while allowing concise, fully navigable long summaries."""
    long_video_with_complete_navigation = (
        transcript_chars >= 12000
        and transcript_max >= 1800
        and coverage_ratio >= 0.85
        and section_coverage_passed
    )
    if transcript_chars < 3000 or transcript_max < 900:
        return (0.12, 0.55)
    return (0.05, 0.30) if long_video_with_complete_navigation else (0.12, 0.30)


def _smart_summary_content_scope_gate(
    root: Path,
    summary_path: Path,
    summary_text: str,
) -> dict[str, Any]:
    """Require a summary to be generated after and from the active scope projection."""

    scope_path = root / "content-scope.json"
    if not scope_path.is_file():
        return {
            "status": "not_configured",
            "passed": True,
            "detail": "content scope is not configured",
            "forbidden_phrase_hits": [],
        }
    try:
        scope = load_content_scope(root)
    except Exception as exc:
        return {
            "status": "blocked_invalid_content_scope",
            "passed": False,
            "detail": str(exc),
            "forbidden_phrase_hits": [],
        }
    summary_intervals = [
        row
        for row in scope.get("excluded_intervals") or []
        if isinstance(row, dict) and "smart_summary" in (row.get("exclude_from") or [])
    ]
    if not summary_intervals:
        return {
            "status": "not_applicable",
            "passed": True,
            "detail": "content scope has no Smart Summary exclusions",
            "forbidden_phrase_hits": [],
        }

    pack_path = root / "exports" / "smart-summary-input-pack.json"
    pack = read_json(pack_path) if pack_path.is_file() else {}
    pack_scope = pack.get("content_scope") if isinstance(pack, dict) and isinstance(pack.get("content_scope"), dict) else {}
    current_scope_sha256 = str(scope.get("content_scope_sha256") or "")
    pack_scope_sha256 = str(pack_scope.get("content_scope_sha256") or "")
    pack_matches = bool(current_scope_sha256 and pack_scope_sha256 == current_scope_sha256)
    summary_is_fresh = bool(
        summary_path.is_file()
        and summary_path.stat().st_mtime_ns >= scope_path.stat().st_mtime_ns
    )

    source_path = Path(str(scope.get("transcript_source") or ""))
    forbidden_phrases: list[dict[str, Any]] = []
    try:
        cues = parse_transcript(source_path) if source_path.is_file() else []
        _kept, scope_report = apply_content_scope_to_cues(
            cues,
            scope,
            projection="smart_summary",
        )
        excluded_ids = {
            str(row.get("id") or "")
            for row in scope_report.get("excluded_items") or []
            if isinstance(row, dict)
        }
        for cue in cues:
            identity = str(getattr(cue, "segment_id", "") or "")
            if identity not in excluded_ids:
                continue
            normalized = _scope_comparison_text(str(getattr(cue, "text", "") or ""))
            if len(normalized) < 8:
                continue
            forbidden_phrases.append(
                {
                    "segment_id": identity,
                    "start": float(getattr(cue, "start", 0.0) or 0.0),
                    "text": normalized,
                }
            )
    except Exception as exc:
        return {
            "status": "blocked_scope_evidence_unreadable",
            "passed": False,
            "detail": str(exc),
            "pack_scope_matches": pack_matches,
            "summary_after_scope": summary_is_fresh,
            "forbidden_phrase_hits": [],
        }

    normalized_summary = _scope_comparison_text(summary_text)
    hits = [
        {"segment_id": row["segment_id"], "start": row["start"]}
        for row in forbidden_phrases
        if row["text"] and row["text"] in normalized_summary
    ]
    passed = pack_matches and summary_is_fresh and not hits
    return {
        "status": "passed" if passed else "blocked_content_scope_projection",
        "passed": passed,
        "detail": (
            f"pack_scope_matches={pack_matches}; summary_after_scope={summary_is_fresh}; "
            f"forbidden_phrase_hits={len(hits)}"
        ),
        "content_scope_sha256": current_scope_sha256,
        "pack_scope_sha256": pack_scope_sha256,
        "pack_scope_matches": pack_matches,
        "summary_after_scope": summary_is_fresh,
        "forbidden_phrase_hits": hits[:50],
        "exact_phrase_check_only": True,
    }


def _scope_comparison_text(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value or "")).lower()


def smart_summary_quality_check(
    bundle_dir: str | Path,
    *,
    summary_path: str | Path | None = None,
    require_codex: bool = False,
    write: bool = True,
    dependency_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    exports = root / "exports"
    requested_summary_path = str(summary_path or "")
    if summary_path:
        path = _resolve_summary_path(root, summary_path)
    else:
        codex_summary = exports / "smart-summary.codex.md"
        legacy_summary = exports / "smart-summary.md"
        # The installed section/global-reduce artifact is the final candidate.
        path = codex_summary if codex_summary.is_file() else legacy_summary
    text = _normalize_markdown_newlines(
        path.read_text(encoding="utf-8-sig") if path.exists() else ""
    )
    content_profile = resolve_content_profile(root)
    interview_profile = content_profile.get("profile_id") in {
        "interview-v1",
        "medical-insurance-interview-v1",
    }
    required_headings = INTERVIEW_REQUIRED_HEADINGS if interview_profile else REQUIRED_HEADINGS
    transcript_max = _transcript_max_seconds(root)
    summary_times = _timestamps(text)
    checks: list[dict[str, Any]] = []

    is_draft = "codex_assisted_draft" in text
    is_codex = _is_codex_summary(path, text)
    checks.append(_check("exists", path.exists() and bool(text.strip()), f"summary_path={path}"))
    checks.append(_check("codex_final", (not require_codex or is_codex) and not is_draft, "requires codex_final or codex_llm_rewrite_final marker"))
    checks.append(_check("required_headings", all(heading in text for heading in required_headings), f"required sections for {content_profile.get('profile_id')} are present"))
    term_gate = load_term_correction_impact_gate(root)
    checks.append(_check("term_correction_impact", bool(term_gate.get("passed")), str(term_gate.get("detail") or "")))
    semantic_gate = _semantic_correction_quality_gate(root)
    checks.append(_check("transcript_semantic_correction_impact", bool(semantic_gate.get("passed")), str(semantic_gate.get("detail") or "")))
    transcript_source_gate = _smart_summary_transcript_source_gate(root)
    checks.append(_check("corrected_transcript_input", bool(transcript_source_gate.get("passed")), str(transcript_source_gate.get("detail") or "")))
    transcript_completeness_gate = _summary_transcript_completeness_gate(root)
    checks.append(
        _check(
            "transcript_speech_completeness",
            bool(transcript_completeness_gate.get("passed")),
            str(transcript_completeness_gate.get("detail") or ""),
        )
    )
    invalidation_gate = _smart_summary_invalidation_gate(root, path)
    checks.append(_check("summary_not_invalidated", bool(invalidation_gate.get("passed")), str(invalidation_gate.get("detail") or "")))
    freshness_gate = _smart_summary_freshness_gate(path, transcript_source_gate)
    checks.append(_check("summary_after_corrected_transcript", bool(freshness_gate.get("passed")), str(freshness_gate.get("detail") or "")))
    evidence_freshness_gate = (_smart_summary_evidence_freshness_gate(root, snapshot=dependency_snapshot)
                               if dependency_snapshot is not None else _smart_summary_evidence_freshness_gate(root))
    checks.append(_check("summary_after_evidence_update", bool(evidence_freshness_gate.get("passed")), str(evidence_freshness_gate.get("detail") or "")))
    content_scope_gate = _smart_summary_content_scope_gate(root, path, text)
    checks.append(
        _check(
            "content_scope_projection",
            bool(content_scope_gate.get("passed")),
            str(content_scope_gate.get("detail") or ""),
        )
    )

    from .summary_claim_review import summary_claim_review_quality_gate
    claim_review_gate = summary_claim_review_quality_gate(root, text)
    checks.append({"key": "summary_claim_review", "passed": claim_review_gate["passed"],
                   "evaluated": claim_review_gate["evaluated"], "status": claim_review_gate["status"],
                   "detail": claim_review_gate["detail"]})
    from .citation_digest import audit_citation_digest
    digest_gate = audit_citation_digest(root)
    checks.append({"key": "citation_digest_integrity", "passed": digest_gate["passed"],
                   "evaluated": digest_gate["evaluated"], "status": digest_gate["status"],
                   "detail": digest_gate["detail"]})

    overview = _section_text(text, "## 一句话概览")
    overview_line = next((line.strip() for line in overview.splitlines() if line.strip()), "")
    overview_ok = 35 <= len(overview_line) <= 260 and not any(fragment in overview_line for fragment in BAD_OVERVIEW_FRAGMENTS)
    checks.append(_check("overview_readable", overview_ok, overview_line[:180]))

    segment_text = _section_text(text, "## 事实时间线" if interview_profile else "## 分段总结")
    long_paragraphs = [para for para in re.split(r"\n\s*\n", segment_text) if len(para.strip()) > 900]
    checks.append(_check("segment_not_asr_dump", not long_paragraphs, f"long_paragraphs={len(long_paragraphs)}"))

    coverage_ratio = (max(summary_times) / transcript_max) if summary_times and transcript_max > 0 else 0.0
    checks.append(_check("time_coverage", coverage_ratio >= 0.85, f"summary_max={_fmt(max(summary_times) if summary_times else 0)}, transcript_max={_fmt(transcript_max)}, ratio={coverage_ratio:.2f}"))

    timeline_anchor_quality = _summary_timeline_anchor_quality(summary_times, transcript_max)
    checks.append(
        _check(
            "timestamp_range",
            bool(timeline_anchor_quality["timestamp_range_passed"]),
            str(timeline_anchor_quality["detail"]),
        )
    )
    checks.append(
        _check(
            "first_last_timeline_coverage",
            bool(timeline_anchor_quality["first_last_passed"]),
            str(timeline_anchor_quality["detail"]),
        )
    )

    section_coverage = _section_coverage(text, transcript_max, interview_profile=interview_profile)
    checks.append(_check("balanced_sections", section_coverage["passed"], section_coverage["detail"]))
    visual_boundary = "视觉" in text and ("未执行" in text or "待复核" in text or "未可靠" in text)
    checks.append(_check("visual_boundary", visual_boundary, "must preserve visual evidence not executed / needs review boundary"))
    inferred_language_gate = _inferred_visual_language_gate(root, text)
    checks.append(_check("inferred_visual_language", bool(inferred_language_gate["passed"]), str(inferred_language_gate["detail"])))

    transcript_text = _quality_transcript_text(transcript_source_gate)
    summary_body = _quality_summary_body(text, interview_profile=interview_profile)
    # Intent: validate reader structure against the same Markdown the reader receives.
    # Decision: pass the full summary to the semantic gate; keep summary_body for metrics.
    # Reason: _quality_summary_body removes headings required by the semantic parser.
    # Evidence: a valid reader plan was falsely reported as overview_too_thin.
    # Effective scope: semantic maturity only; compression and evidence gates are unchanged.
    reader_semantics = evaluate_reader_markdown_semantics(text)
    checks.append(
        _check(
            "reader_semantic_maturity",
            bool(reader_semantics.get("passed")),
            f"problems={list(reader_semantics.get('problems') or [])[:12]}",
        )
    )
    transcript_chars = len(_compact_quality_text(transcript_text))
    summary_chars = len(_compact_quality_text(summary_body))
    compression_ratio = round(summary_chars / max(1, transcript_chars), 6) if transcript_chars else None
    compression_evaluated = transcript_chars >= 2000
    compression_target = _compression_target(
        transcript_chars=transcript_chars,
        transcript_max=transcript_max,
        coverage_ratio=coverage_ratio,
        section_coverage_passed=bool(section_coverage["passed"]),
    )
    compression_ok = not compression_evaluated or (
        compression_ratio is not None
        and compression_target[0] <= compression_ratio <= compression_target[1]
    )
    checks.append(
        _check(
            "compression_ratio",
            compression_ok,
            f"ratio={compression_ratio}; target={compression_target[0]:.2f}-{compression_target[1]:.2f}; evaluated={compression_evaluated}",
        )
    )

    repetition_rate = _summary_repetition_rate(summary_body)
    checks.append(_check("summary_repetition", repetition_rate <= 0.18, f"duplicate_paragraph_rate={repetition_rate}; target<=0.18"))

    transcript_number_map = _quality_number_evidence(transcript_text)
    visual_number_evidence = _trusted_visual_number_evidence(root)
    number_evidence: dict[str, list[dict[str, Any]]] = {
        key: [
            {
                "source_kind": "asr",
                "transcript_source": str(
                    transcript_source_gate.get("transcript_source") or ""
                ),
                "source_mentions": sorted(mentions),
            }
        ]
        for key, mentions in transcript_number_map.items()
    }
    for key, evidence in visual_number_evidence.items():
        number_evidence.setdefault(key, []).extend(evidence)
    stripped_summary_body = _strip_quality_metadata_identifiers(summary_body, root)
    structural_numbers = _structural_heading_numbers(stripped_summary_body)
    summary_number_map = _quality_number_evidence(stripped_summary_body)
    unsupported_number_keys = sorted(set(summary_number_map) - set(number_evidence))
    unsupported_numbers = sorted(
        {
            mention
            for key in unsupported_number_keys
            for mention in _number_display_mentions(
                summary_number_map.get(key, set())
            )
        }
    )
    supporting_claims = {
        mention: number_evidence[key][:8]
        for key, mentions in summary_number_map.items()
        if key in number_evidence
        for mention in _number_display_mentions(mentions)
    }
    number_evaluated = transcript_chars >= 2000
    checks.append(_check("number_consistency", (not number_evaluated) or not unsupported_numbers, f"unsupported_numbers={unsupported_numbers[:12]}; evaluated={number_evaluated}"))

    key_point_recall = _human_key_point_recall(root, summary_body)
    if key_point_recall["evaluated"]:
        key_point_check = _check(
            "human_key_point_recall",
            float(key_point_recall["recall"]) >= 0.85,
            key_point_recall["detail"],
        )
        key_point_check.update(
            {
                "evaluated": True,
                "status": (
                    "passed"
                    if key_point_check["passed"]
                    else "failed"
                ),
            }
        )
    else:
        key_point_check = {
            "key": "human_key_point_recall",
            "passed": None,
            "evaluated": False,
            "status": "not_evaluated",
            "detail": key_point_recall["detail"],
        }
    checks.append(key_point_check)

    from .summary_stage_coverage import stored_summary_stage_coverage
    stage_coverage = stored_summary_stage_coverage(root, output_text=summary_body)
    stage_evaluated = bool(stage_coverage.get("required") and
                           (stage_coverage.get("evaluated") or stage_coverage.get("status") == "blocked"))
    checks.append({"key": "declared_important_point_stage_retention",
                   "passed": stage_coverage.get("passed") if stage_evaluated else None,
                   "evaluated": stage_evaluated,
                   "detail": f"status={stage_coverage.get('status')}; input_basis={stage_coverage.get('input_basis')}"})

    from .courseware_material import courseware_summary_gate
    courseware_gate = courseware_summary_gate(root, path)
    checks.append(_check("courseware_material_complete_and_consumed", courseware_gate["passed"], courseware_gate["detail"]))

    automated_checks_passed = all(
        bool(row["passed"])
        for row in checks
        if row.get("evaluated", True)
    )
    quality_evidence_complete = bool(key_point_recall["evaluated"])
    production_ready = automated_checks_passed and quality_evidence_complete
    passed = production_ready
    status = (
        "failed"
        if not automated_checks_passed
        else (
            "passed"
            if quality_evidence_complete
            else "blocked_missing_human_key_points"
        )
    )
    result = {
        "schema": QUALITY_SCHEMA,
        "bundle_dir": str(root),
        "requested_summary_path": requested_summary_path,
        "resolved_summary_path": str(path),
        "summary_path": str(path),
        "status": status,
        "passed": passed,
        "quality_evidence_complete": quality_evidence_complete,
        "automated_checks_passed": automated_checks_passed,
        "production_ready": production_ready,
        "require_codex": require_codex,
        "is_codex_summary": is_codex,
        "is_draft": is_draft,
        "transcript_max_seconds": transcript_max,
        "summary_max_seconds": max(summary_times) if summary_times else 0.0,
        "term_correction_impact_gate": term_gate,
        "transcript_semantic_correction_gate": semantic_gate,
        "transcript_source_gate": transcript_source_gate,
        "transcript_completeness_gate": transcript_completeness_gate,
        "summary_invalidation_gate": invalidation_gate,
        "summary_freshness_gate": freshness_gate,
        "summary_evidence_freshness_gate": evidence_freshness_gate,
        "content_scope_gate": content_scope_gate,
        "summary_claim_review": claim_review_gate,
        "summary_stage_coverage": stage_coverage,
        "quality_metrics": {
            "transcript_chars": transcript_chars,
            "summary_chars": summary_chars,
            "compression_ratio": compression_ratio,
            "compression_target": list(compression_target),
            "repetition_rate": repetition_rate,
            "unsupported_numbers": unsupported_numbers,
            "structural_numbers": structural_numbers,
            "number_evidence": {
                "supporting_claims": supporting_claims,
                "unsupported_claims": _unsupported_number_claims(
                    summary_body, unsupported_numbers
                ),
                "canonical_comparison": True,
                "trusted_visual_evidence_count": sum(len(rows) for rows in visual_number_evidence.values()),
                "trusted_visual_item_count": len({(row.get("timeline_index"), row.get("artifact_path")) for rows in visual_number_evidence.values() for row in rows}),
            },
            "human_key_point_recall": key_point_recall,
        },
        "checks": checks,
        "checked_at": now_iso(),
    }
    if write:
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "smart-summary-quality.json", result)
        (exports / "smart-summary-quality.md").write_text(_render_quality_markdown(result), encoding="utf-8")
    return result



TEACHING_KEYWORDS = (
    "核心", "关键", "原则", "方法", "步骤", "流程", "问题", "需求", "客户", "信任", "成交", "复盘", "动作", "案例", "工具", "策略", "注意", "总结", "因为", "所以", "但是", "如果", "一定", "必须", "不要", "不能", "应该", "可以",
)
ACTION_KEYWORDS = ("要", "需要", "可以", "先", "再", "最后", "准备", "记录", "复盘", "确认", "整理", "提问", "跟进", "判断", "避免", "建立", "设计")
SPEECH_KEYWORDS = ("我", "你", "客户", "我们", "对方", "怎么", "为什么", "是不是", "能不能", "先", "不用", "不要")
FILLERS = (
    "然后呢", "就是说", "也就是说", "对吧", "是不是", "对不对", "其实", "那么", "这个", "那个", "就是", "啊", "呃", "嗯", "哈",
)


def _generate_local_codex_summary(root: Path) -> str:
    pack = _load_or_build_input_pack(root)
    memory_pack = _load_or_build_long_memory_pack(root)
    title = str(pack.get("title") or root.name).strip()
    manifest = _read_mapping(root / "manifest.json")
    segments = [row for row in (pack.get("transcript_segments") or []) if isinstance(row, dict) and _segment_text(row)]
    visual_digest = pack.get("visual_digest") if isinstance(pack.get("visual_digest"), dict) else {}
    term_summary = pack.get("term_summary") if isinstance(pack.get("term_summary"), dict) else {}
    term_arbitration_codex = pack.get("term_arbitration_codex") if isinstance(pack.get("term_arbitration_codex"), dict) else {}
    transcript_semantic = pack.get("transcript_semantic_correction") if isinstance(pack.get("transcript_semantic_correction"), dict) else {}
    codex_terms = _codex_arbitrated_terms(term_arbitration_codex)
    chapter_pack = build_smart_summary_chapter_pack(root, title=title, write=True)
    chapters = chapter_pack.get("chapters") if isinstance(chapter_pack.get("chapters"), list) else []
    course_map = chapter_pack.get("course_map") if isinstance(chapter_pack.get("course_map"), dict) else {}
    duration = max((_seconds(row.get("end")) for row in segments), default=0.0)
    source_path = str(manifest.get("media_path") or manifest.get("source_path") or manifest.get("path") or "").strip()
    chunks = _chunks_from_long_memories(memory_pack) or _chunks_from_chapters(chapters) or _balanced_segment_chunks(segments, target_chunks=8)
    key_points = _chapter_snippets(chapters, "key_points") or _select_ranked_snippets(segments, TEACHING_KEYWORDS, target=8, require_balanced=True)
    actions = _chapter_snippets(chapters, "actions") or _select_ranked_snippets(segments, ACTION_KEYWORDS, target=8, require_balanced=True)
    expressions = _chapter_snippets(chapters, "reusable_expressions") or _select_ranked_snippets(segments, SPEECH_KEYWORDS, target=6, require_balanced=True)
    citations = _chapter_citation_rows(chapters)
    overview = _local_overview(title, chunks, key_points, codex_terms)
    lines = [
        f"# {title} - 智能总结",
        "",
        "生成方式：`local_scaffold_not_llm`。这是 VKP 自动生成的本地结构化草稿，不等于真实 Codex/LLM 改写成品；默认导出必须继续进入 Codex/LLM 改写层，或者明确标记为待改写。",
        "",
        numbered_summary_heading("基本信息"),
        "",
        f"- 视频名：{title}",
        f"- 时长：`{format_timestamp(duration)}`",
        f"- 处理时间：`{now_iso()}`",
        f"- 来源路径：`{source_path or '(unknown)'}`",
        f"- 转写来源：`{pack.get('transcript_source') or 'unknown'}`",
        f"- 总结输入包：`{root / 'exports' / 'smart-summary-input-pack.md'}`",
        f"- 长视频记忆包：`{root / 'exports' / 'long-video-memory-pack.md'}`",
        f"- 长视频记忆：`{memory_pack.get('summary', {}).get('short_memories', 0)}` short / `{memory_pack.get('summary', {}).get('long_memories', 0)}` long",
        f"- 视觉证据状态：{_local_visual_status(visual_digest)}",
        f"- ASR/字幕语义纠错状态：`{transcript_semantic.get('final_status', 'not_started')}`",
        "",
        numbered_summary_heading("一句话概览"),
        "",
        overview,
        "",
        numbered_summary_heading("核心主题 / 课程主线", number="3"),
        "",
    ]
    lines.extend(_local_mainline_lines(title, chunks, term_summary, term_arbitration_codex, codex_terms))
    lines.extend(_course_map_lines(course_map))
    # Long-video memory remains an evidence source; do not dump memory rows into the main readable narrative.
    lines.extend(["", numbered_summary_heading("分段总结"), ""])
    lines.extend(_local_chunk_summary_lines(chunks, visual_digest))
    lines.extend(["", numbered_summary_heading("关键观点 / 方法论", number="5"), ""])
    lines.extend(_snippet_lines(key_points, fallback="暂无足够稳定的观点句，请回看完整逐字稿。", numbering_prefix="5.1"))
    lines.extend(["", numbered_summary_heading("证据引用 / Citation Digest", number="6"), ""])
    lines.extend(_citation_lines(citations, numbering_prefix="6.1"))
    lines.extend(["", numbered_summary_heading("可执行动作清单"), ""])
    lines.extend(_action_lines(actions, numbering_prefix="7.1"))
    lines.extend(["", numbered_summary_heading("高频话术 / 可复用表达", number="8"), ""])
    lines.extend(_expression_lines(expressions, numbering_prefix="8.1"))
    lines.extend(["", numbered_summary_heading("待复核点 / 低置信内容", number="9"), ""])
    lines.extend(_local_review_lines(pack, visual_digest, term_summary, transcript_semantic))
    summary_text = "\n".join(lines).rstrip() + "\n"
    return apply_term_replacement_pairs(summary_text, load_bundle_term_replacements(root))



def _chunks_from_chapters(chapters: list[Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        sentences = [str(value).strip() for value in chapter.get("summary_sentences") or [] if str(value).strip()]
        text = " ".join(sentences)
        chunks.append(
            {
                "start": _seconds(chapter.get("start")),
                "end": _seconds(chapter.get("end")),
                "title": chapter.get("title") or "",
                "text": text,
                "sentences": sentences or _split_sentences(text),
                "visual_notes": chapter.get("visual_notes") if isinstance(chapter.get("visual_notes"), list) else [],
            }
        )
    return chunks



def _chapter_citation_rows(chapters: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_index = chapter.get("index")
        for citation in chapter.get("citation_digest") or []:
            if not isinstance(citation, dict):
                continue
            text = str(citation.get("text") or "").strip()
            if not text:
                continue
            source_type = str(citation.get("source_type") or "evidence").strip()
            rows.append(
                {
                    **citation,
                    "chapter_index": chapter_index,
                    "source_type": source_type,
                    "time": str(citation.get("time") or format_timestamp(_seconds(chapter.get("start")))).strip(),
                    "timeline_indexes": citation.get("timeline_indexes") if isinstance(citation.get("timeline_indexes"), list) else [],
                    "text": text,
                    "evidence_paths": [str(value) for value in citation.get("evidence_paths") or [] if str(value)],
                }
            )
    return _dedupe_citation_rows(rows)


def _dedupe_citation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .citation_digest import digest_records
    return digest_records(rows)

def _chapter_snippets(chapters: list[Any], key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        for row in chapter.get(key) or []:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            out.append({"start": _seconds(row.get("seconds") or chapter.get("start")), "end": _seconds(chapter.get("end")), "text": text})
    return out


def _course_map_lines(course_map: dict[str, Any]) -> list[str]:
    if not course_map:
        return []
    lines: list[str] = []
    main_question = str(course_map.get("main_question") or "").strip()
    mainline = str(course_map.get("mainline") or "").strip()
    topics = course_map.get("topics") if isinstance(course_map.get("topics"), list) else []
    if main_question:
        lines.append(f"- 课程问题：{main_question}")
    if mainline:
        lines.append(f"- 课程地图：{mainline}")
    if topics:
        lines.append("- 章节角色：" + "；".join(_format_topic_line(topic) for topic in topics[:8] if isinstance(topic, dict)))
    return lines


def _format_topic_line(topic: dict[str, Any]) -> str:
    title = _clean_llm_main_text(topic.get("title") or "", limit=70)
    time_range = str(topic.get("time_range") or "").strip()
    role = str(topic.get("role") or "").strip()
    return f"`{time_range}` {title or '未识别主题'} ({role})"
def _load_or_build_long_memory_pack(root: Path) -> dict[str, Any]:
    path = root / "exports" / "long-video-memory-pack.json"
    if path.exists():
        data = read_json(path)
        if isinstance(data, dict):
            try:
                assert_material_fresh(root, data.get("courseware_material", {}))
                return data
            except ValueError:
                pass  # Rebuild stale opt-in material; never silently reuse a lossy older pack.
    return build_long_video_memory_pack(root, write=True)


def _chunks_from_long_memories(memory_pack: dict[str, Any]) -> list[dict[str, Any]]:
    memories = memory_pack.get("long_memories") if isinstance(memory_pack.get("long_memories"), list) else []
    chunks: list[dict[str, Any]] = []
    for memory in memories:
        if not isinstance(memory, dict):
            continue
        bullets = [str(value).strip() for value in memory.get("merged_bullets") or [] if str(value).strip()]
        topic = str(memory.get("topic_hint") or "").strip()
        text = " ".join(bullets)
        chunks.append(
            {
                "start": _seconds(memory.get("start")),
                "end": _seconds(memory.get("end")),
                "title": topic,
                "text": text,
                "sentences": bullets or _split_sentences(text),
                "visual_notes": [],
            }
        )
    return chunks


def _memory_map_lines(memory_pack: dict[str, Any]) -> list[str]:
    final_map = memory_pack.get("final_memory_map") if isinstance(memory_pack.get("final_memory_map"), dict) else {}
    flow = final_map.get("course_flow") if isinstance(final_map.get("course_flow"), list) else []
    if not flow:
        return []
    lines = ["- 长视频记忆主线："]
    for row in flow[:10]:
        if not isinstance(row, dict):
            continue
        time_range = str(row.get("time") or "").strip()
        topic = _clean_llm_main_text(row.get("topic_hint") or "未识别主题", limit=72) or "未识别主题"
        points = row.get("key_points") if isinstance(row.get("key_points"), list) else []
        clean_points = [_clean_llm_main_text(point, limit=88) for point in points]
        clean_points = [point for point in _dedupe(clean_points) if point and not _looks_like_evidence_label_list(point)]
        point_text = "；".join(clean_points[:2])
        lines.append(f"  - `{time_range}` {topic}" + (f"：{point_text}" if point_text else ""))
    return lines

def _load_or_build_input_pack(root: Path) -> dict[str, Any]:
    path = root / "exports" / "smart-summary-input-pack.json"
    if path.exists():
        data = read_json(path)
        if isinstance(data, dict):
            return data
    return build_smart_summary_input_pack(root, write=True)


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _segment_text(row: dict[str, Any]) -> str:
    return _clean_summary_text(row.get("punctuated_text") or row.get("corrected_text") or row.get("raw_text"))


def _balanced_segment_chunks(segments: list[dict[str, Any]], *, target_chunks: int) -> list[dict[str, Any]]:
    if not segments:
        return []
    duration = max((_seconds(row.get("end")) for row in segments), default=0.0)
    if duration <= 0:
        duration = float(len(segments))
    target = max(1, min(target_chunks, len(segments)))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(target)]
    for pos, row in enumerate(segments):
        start = _seconds(row.get("start"))
        bucket = min(target - 1, int((start / duration) * target)) if duration else min(target - 1, pos)
        buckets[bucket].append(row)
    chunks = []
    for bucket in buckets:
        if not bucket:
            continue
        text = " ".join(_segment_text(row) for row in bucket if _segment_text(row))
        chunks.append({
            "start": _seconds(bucket[0].get("start")),
            "end": _seconds(bucket[-1].get("end")),
            "text": text,
            "sentences": _split_sentences(text),
        })
    return chunks


def _local_overview(title: str, chunks: list[dict[str, Any]], key_points: list[dict[str, Any]], codex_terms: list[str]) -> str:
    topic_hint = _title_topic_hint(title)
    term_hint = "，并已把" + "、".join(codex_terms[:4]) + "等术语按语义仲裁结果统一" if codex_terms else ""
    stage_count = len(chunks)
    if topic_hint:
        return f"这节课围绕{topic_hint}展开{term_hint}，把讲师的核心判断、分段讲解、可执行动作和待复核证据整理成一份覆盖全片的学习笔记。"
    if stage_count:
        return f"这节课按时间推进整理为 {stage_count} 个内容段{term_hint}，重点保留讲师反复强调的判断依据、操作步骤、复盘要求和可复用表达。"
    return "这节课已基于完整转写整理为分段总结、关键观点、行动清单和待复核点，适合后续复习与人工校对。"

def _local_mainline_lines(title: str, chunks: list[dict[str, Any]], term_summary: dict[str, Any], term_arbitration_codex: dict[str, Any], codex_terms: list[str]) -> list[str]:
    lines = []
    terms = [str(row.get("canonical_term") or "").strip() for row in (term_summary.get("high_confidence_terms") or []) if isinstance(row, dict)]
    if terms:
        lines.append("- 本稿已自动采用高置信术语：" + "、".join(_dedupe([term for term in terms if term])[:12]) + "。")
    if codex_terms:
        source = str(term_arbitration_codex.get("status") or "imported")
        lines.append(f"- Codex 语义仲裁术语：`{source}`，最终采用 " + "、".join(codex_terms[:12]) + "；这些名称已优先用于纠正版转写和最终总结。")
    lines.append(f"- 课程主线：围绕《{title}》的核心问题，按时间推进梳理讲师提出的判断、解释、案例和行动要求。")
    if chunks:
        first_title = _clean_llm_main_text(chunks[0].get("title") or "开场背景", limit=42) or "开场背景"
        middle_title = _clean_llm_main_text(chunks[len(chunks) // 2].get("title") or "案例和方法展开", limit=42) or "案例和方法展开"
        last_title = _clean_llm_main_text(chunks[-1].get("title") or "行动建议和答疑收束", limit=42) or "行动建议和答疑收束"
        lines.append(f"- 内容推进：开头先交代{first_title}，中段围绕{middle_title}展开案例、判断和方法，最后收束到{last_title}以及后续行动。")
    lines.append("- 阅读方式：先看分段总结建立全局，再用关键观点和动作清单复习，最后检查待复核点。")
    return lines


def _local_chunk_summary_lines(chunks: list[dict[str, Any]], visual_digest: dict[str, Any] | None = None) -> list[str]:
    if not chunks:
        return ["（没有可用转写，无法生成分段总结。）"]
    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        summary = _chunk_summary_text(chunk, max_sentences=3, max_chars=260)
        visual_note = _chunk_visual_note(chunk) or _visual_note_for_chunk(chunk, visual_digest or {})
        lines.extend([
            f"### 4.1.{idx} `{format_timestamp(chunk['start'])} - {format_timestamp(chunk['end'])}` {str(chunk.get('title') or '内容段').strip()}",
            "",
        ])
        if visual_note:
            lines.extend([f"- 课件/画面补充：{visual_note}", ""])
        lines.extend([summary or "这一段转写信息较少，建议回看原视频核对。", ""])
    return lines


def _chunk_visual_note(chunk: dict[str, Any]) -> str:
    notes = chunk.get("visual_notes") if isinstance(chunk.get("visual_notes"), list) else []
    values: list[str] = []
    for note in notes:
        if isinstance(note, dict) and note.get("text"):
            values.append(str(note.get("text")))
        elif note:
            values.append(str(note))
    return "；".join(_dedupe([_clip_sentence(value, 90) for value in values])[:3])
def _visual_note_for_chunk(chunk: dict[str, Any], visual_digest: dict[str, Any]) -> str:
    items = visual_digest.get("items") if isinstance(visual_digest.get("items"), list) else []
    if not items:
        return ""
    start = _seconds(chunk.get("start"))
    end = _seconds(chunk.get("end"))
    notes: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_start = _seconds(item.get("start"))
        item_end = _seconds(item.get("end"))
        if item_end < start or item_start > end:
            continue
        note = _visual_item_note(item)
        if note:
            notes.append(note)
    return "；".join(_dedupe(notes)[:3])


def _visual_item_note(item: dict[str, Any]) -> str:
    for key in ("structured_visual", "visual_text", "visual_understanding", "temporal_visual_understanding"):
        text = _clean_visual_note_text(item.get(key))
        if text:
            return text
    return ""


def _clean_visual_note_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    headings = re.findall(r"#{1,4}\s*([^#|]{2,50})", text)
    if headings:
        return "、".join(_dedupe([_clip_sentence(h, 42) for h in headings if h.strip()])[:3])
    text = re.sub(r"source:\s*ebook_markdown_pipeline[；;]?", "", text)
    text = re.sub(r"type:\s*structured_visual[；;]?", "", text)
    text = re.sub(r"markdown:\s*", "", text)
    return _clip_sentence(text, 120)
def _chunk_summary_text(chunk: dict[str, Any], *, max_sentences: int, max_chars: int) -> str:
    sentences = chunk.get("sentences") if isinstance(chunk.get("sentences"), list) else []
    ranked = sorted(sentences, key=_sentence_score, reverse=True)
    chosen = _dedupe([_clip_sentence(sentence, 120) for sentence in ranked if sentence])[:max_sentences]
    if not chosen and chunk.get("text"):
        chosen = [_clip_sentence(str(chunk.get("text") or ""), 120)]
    text = "；".join(chosen)
    if text and text[-1] not in "。！？!?":
        text += "。"
    return _clip_sentence(text, max_chars)


def _select_ranked_snippets(segments: list[dict[str, Any]], keywords: tuple[str, ...], *, target: int, require_balanced: bool) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    duration = max((_seconds(row.get("end")) for row in segments), default=0.0)
    for row in segments:
        text = _segment_text(row)
        for sentence in _split_sentences(text):
            score = _sentence_score(sentence, keywords=keywords)
            if score <= 0:
                continue
            candidates.append({"start": _seconds(row.get("start")), "end": _seconds(row.get("end")), "text": _clip_sentence(sentence, 120), "score": score})
    candidates.sort(key=lambda row: (-float(row.get("score") or 0), _seconds(row.get("start"))))
    if not require_balanced or duration <= 0:
        return _dedupe_snippets(candidates)[:target]
    selected: list[dict[str, Any]] = []
    buckets = max(1, min(4, target))
    for bucket in range(buckets):
        left = duration * bucket / buckets
        right = duration * (bucket + 1) / buckets
        bucket_rows = [row for row in candidates if left <= _seconds(row.get("start")) <= right]
        selected.extend(_dedupe_snippets(bucket_rows)[: max(1, target // buckets)])
    if len(selected) < target:
        selected.extend(row for row in _dedupe_snippets(candidates) if row not in selected)
    return _dedupe_snippets(selected)[:target]


def _snippet_lines(rows: list[dict[str, Any]], *, fallback: str, numbering_prefix: str = "") -> list[str]:
    if not rows:
        prefix = f"{numbering_prefix}.1 " if numbering_prefix else ""
        return [f"- {prefix}{fallback}"]
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        text = _clean_llm_main_text(row.get("text"), limit=150)
        if text:
            prefix = f"{numbering_prefix}.{index} " if numbering_prefix else ""
            lines.append(f"- {prefix}`{_time_label(row)}` {text}")
    return lines or [f"- {(numbering_prefix + '.1 ') if numbering_prefix else ''}{fallback}"]



def _citation_lines(rows: list[dict[str, Any]], numbering_prefix: str = "") -> list[str]:
    if not rows:
        prefix = f"{numbering_prefix}.1 " if numbering_prefix else ""
        return [f"- {prefix}暂无可用 citation digest；最终总结仍需回到 smart-summary-chapters.md / full-transcript.md 核对证据。"]
    from .citation_digest import preview_note, evidence_level as citation_level
    lines = ["这些引用来自章节级 `citation_digest`，用于把总结观点回链到转写、画面、OCR/ebook、连续片段或待复核缺口；它们是证据导航，不等于事实已确认。", "",
             preview_note(min(24, len(rows)), len(rows))]
    for index, row in enumerate(rows[:24], start=1):
        source_type = str(row.get("source_type") or "evidence")
        evidence_level = citation_level(row)
        time = str(row.get("time") or "")
        timeline = ",".join(str(value) for value in row.get("timeline_indexes") or [])
        evidence = "; ".join(str(value) for value in row.get("evidence_paths") or [])
        suffix = []
        if timeline:
            suffix.append(f"timeline={timeline}")
        if evidence:
            suffix.append(f"evidence={evidence}")
        suffix_text = "；" + "；".join(suffix) if suffix else ""
        prefix = f"{numbering_prefix}.{index} " if numbering_prefix else ""
        claim = _clip_sentence(str(row.get("text") or ""), 240)
        if evidence_level == "inferred" and not any(term in claim for term in ("可能", "推断", "疑似", "似乎", "看起来")):
            claim = "推断：可能" + claim
        if evidence_level == "candidate_weak":
            claim = "弱证据（仅候选）：" + claim
        lines.append(f"- {prefix}`{time}` `{source_type}` `evidence_level={evidence_level}` {claim}{suffix_text}")
    return lines


def _inferred_visual_language_gate(root: Path, summary_text: str) -> dict[str, Any]:
    timeline_path = root / "timeline.json"
    if not timeline_path.is_file():
        return {"passed": True, "detail": "timeline_missing; no inferred visual claims evaluated", "violations": []}
    try:
        payload = read_json(timeline_path)
    except Exception as exc:
        return {"passed": False, "detail": f"timeline_unreadable:{type(exc).__name__}", "violations": []}
    timeline = payload if isinstance(payload, list) else []
    inferred: list[dict[str, Any]] = []
    confirmed_claims: set[str] = set()
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        for kind, state in (("visual", visual_evidence_state(item)), ("temporal", temporal_evidence_state(item))):
            claims = _visual_claim_strings(state.get("understanding"))
            if state.get("evidence_level") == "confirmed":
                confirmed_claims.update(_compact_quality_text(value) for value in claims)
            elif state.get("evidence_level") == "inferred":
                for claim in claims:
                    inferred.append({"timeline_index": int(item.get("index") or position), "kind": kind, "claim": claim})
    inferred = [row for row in inferred if _compact_quality_text(row["claim"]) not in confirmed_claims]
    review_match = re.search(r"(?m)^##\s*(?:\d+(?:\.\d+)?\s*)?待复核点", summary_text)
    main_text = summary_text[: review_match.start()] if review_match else summary_text
    qualifiers = ("可能", "推断", "疑似", "似乎", "看起来")
    violations: list[dict[str, Any]] = []
    for row in inferred:
        claim = str(row["claim"])
        for line in main_text.splitlines():
            if claim not in line:
                continue
            if not any(term in line for term in qualifiers):
                violations.append({**row, "reason": "inferred_visual_unqualified", "line": line.strip()[:240]})
                break
    tag_gate = continuous_tag_language_gate(timeline, summary_text)
    violations.extend(tag_gate.get("violations") or [])
    return {
        "passed": not violations,
        "detail": (
            f"inferred_only_claims={len(inferred)}; "
            f"continuous_tag_weak_claims={tag_gate.get('weak_claims', 0)}; "
            f"violations_before_review={len(violations)}"
        ),
        "violations": violations[:20],
    }


def _visual_claim_strings(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    keys = (
        "objects",
        "actions",
        "interface_state",
        "spatial_relations",
        "non_text_information",
        "instructor_focus",
        "event_sequence",
        "state_changes",
        "operation_steps",
        "causal_links",
    )
    claims: list[str] = []
    for key in keys:
        raw = value.get(key)
        values = raw if isinstance(raw, list) else [raw]
        for child in values:
            text = re.sub(r"\s+", " ", str(child or "").strip())
            if len(text) >= 4:
                claims.append(text)
    return list(dict.fromkeys(claims))

def _action_lines(rows: list[dict[str, Any]], numbering_prefix: str = "") -> list[str]:
    if not rows:
        prefix = f"{numbering_prefix}.1 " if numbering_prefix else ""
        return [f"- {prefix}回看完整逐字稿，先人工标出可执行步骤，再补进动作清单。"]
    lines = []
    for index, row in enumerate(rows, start=1):
        text = _clean_llm_main_text(row.get("text"), limit=150)
        if not text:
            continue
        if not text.startswith(("先", "把", "用", "记录", "准备", "确认", "整理", "复盘", "避免", "建立")):
            text = "落实这一点：" + text
        prefix = f"{numbering_prefix}.{index} " if numbering_prefix else ""
        lines.append(f"- {prefix}`{_time_label(row)}` {text}")
    return lines or [f"- {(numbering_prefix + '.1 ') if numbering_prefix else ''}回看完整逐字稿，先人工标出可执行步骤，再补进动作清单。"]


def _expression_lines(rows: list[dict[str, Any]], numbering_prefix: str = "") -> list[str]:
    if not rows:
        prefix = f"{numbering_prefix}.1 " if numbering_prefix else ""
        return [f"- {prefix}暂未抽出稳定话术；建议人工从逐字稿中挑选原句。"]
    lines = ["以下是从转写中整理出的可复述表达，发布或引用前建议回看原视频核对原话："]
    for index, row in enumerate(rows, start=1):
        text = _clean_llm_main_text(row.get("text"), limit=150)
        if text:
            prefix = f"{numbering_prefix}.{index} " if numbering_prefix else ""
            lines.append(f"- {prefix}`{_time_label(row)}` “{text}”")
    return lines if len(lines) > 1 else [f"- {(numbering_prefix + '.1 ') if numbering_prefix else ''}暂未抽出稳定话术；建议人工从逐字稿中挑选原句。"]


def _time_label(row: dict[str, Any]) -> str:
    start = _seconds(row.get("start"))
    end = _seconds(row.get("end"))
    if end > start and end - start >= 1.0:
        return f"{format_timestamp(start)} - {format_timestamp(end)}"
    return format_timestamp(start)


def _codex_arbitrated_terms(term_arbitration_codex: dict[str, Any]) -> list[str]:
    glossary_path = Path(str(term_arbitration_codex.get("glossary_path") or "")).expanduser()
    terms: list[str] = []
    if glossary_path.exists():
        try:
            glossary = read_json(glossary_path)
        except Exception:
            glossary = {}
        glossary_terms = glossary.get("terms") if isinstance(glossary, dict) and isinstance(glossary.get("terms"), list) else []
        for row in glossary_terms:
            if not isinstance(row, dict):
                continue
            if bool(row.get("review_required")):
                continue
            confidence = _seconds(row.get("confidence"))
            if confidence and confidence < 0.88:
                continue
            term = str(row.get("canonical") or row.get("canonical_term") or "").strip()
            if term:
                terms.append(term)
    return _dedupe(terms)


def _local_review_lines(pack: dict[str, Any], visual_digest: dict[str, Any], term_summary: dict[str, Any], transcript_semantic: dict[str, Any] | None = None) -> list[str]:
    lines = []
    if not visual_digest.get("items"):
        lines.append("- 视觉证据未执行/待复核：当前智能总结主要依据完整 ASR 和纠正版转写，屏幕文字、课件页、板书和软件界面不能当作已确认事实。")
    else:
        lines.append("- 视觉证据待复核：已纳入部分 OCR/图文结构/多模态摘要，但仍应以证据帧和人工审核为准。")
    semantic = transcript_semantic if isinstance(transcript_semantic, dict) else {}
    semantic_status = str(semantic.get("final_status") or "not_started")
    if semantic_status in {"needs_codex_or_llm_review", "needs_human_review", "needs_closure", "needs_readable_export_fix", "needs_smart_summary_refresh"}:
        detail = f"候选 {semantic.get('candidate_count', 0)} 个，重点复核 {semantic.get('semantic_attention_count', 0)} 个，已接受 {semantic.get('accepted_decision_count', 0)} 个，待人工 {semantic.get('review_required_count', 0)} 个。"
        lines.append(f"- ASR/字幕语义纠错待复核：状态 `{semantic_status}`，{detail}不要把这些疑似错词写成确定事实；需要时回到 transcript-semantic-correction-review.md 或 source-arbitrated transcript 核对。")
    elif semantic_status == "ready_for_summary_input":
        lines.append(f"- ASR/字幕语义纠错已闭合：已采用纠正版转写作为总结输入，修正 {semantic.get('applied_correction_count', 0)} 处，影响 {semantic.get('changed_segment_count', 0)} 个片段；仍建议抽样核对原视频。")
    elif semantic_status == "not_started":
        lines.append("- ASR/字幕语义纠错未执行：本总结不应被视为已完成错词排查，疑似错词需要后续跑 transcript-semantic-correction-pack。")
    needs_review = term_summary.get("needs_review_terms") if isinstance(term_summary.get("needs_review_terms"), list) else []
    if needs_review:
        terms = [str(row.get("canonical_term") or row.get("raw_mentions") or "").strip() for row in needs_review if isinstance(row, dict)]
        lines.append("- 术语待复核：" + "、".join(_dedupe([term for term in terms if term])[:20]) + "。")
    for note in pack.get("quality_notes") or []:
        note_text = str(note or "").strip()
        if note_text and "visual" in note_text.lower():
            lines.append(f"- {note_text}")
    return _dedupe(lines) or ["- 暂无额外低置信提示。"]


def _local_visual_status(visual_digest: dict[str, Any]) -> str:
    count = int(visual_digest.get("total_items_with_visual_digest") or 0)
    if count <= 0:
        return "视觉证据未执行/待复核；不要把屏幕细节写成确定事实。"
    route_counts = visual_digest.get("route_counts") if isinstance(visual_digest.get("route_counts"), dict) else {}
    route_label_map = {
        "document_visual": "图文截图",
        "semantic_frame": "语义画面",
        "temporal_sequence": "连续片段",
        "mixed": "混合画面",
        "unknown": "未知画面",
    }
    route_text = "、".join(f"{route_label_map.get(str(key), str(key))} {value}" for key, value in route_counts.items() if value)
    suffix = f"，其中 {route_text}" if route_text else ""
    return f"已纳入 `{count}` 条视觉/课件摘要{suffix}；仍需人工抽样核对。"


def _title_topic_hint(title: str) -> str:
    value = str(title or "").strip()
    value = re.sub(r"[\-_]+", "，", value)
    value = re.sub(r"\s+", "", value)
    value = value.strip(" ，,、。")
    if not value:
        return ""
    if len(value) <= 42:
        return f"“{value}”"
    return f"“{value[:41].rstrip()}…”"

def _split_sentences(text: str) -> list[str]:
    cleaned = _clean_summary_text(text)
    if not cleaned:
        return []
    expanded = _insert_soft_sentence_boundaries(cleaned)
    parts = re.split(r"[。！？!?；;\n]+", expanded)
    sentences = []
    for part in parts:
        value = part.strip(" ，,、：:")
        if 8 <= len(value) <= 180:
            sentences.append(value)
        elif len(value) > 180:
            sentences.extend(_fixed_width_sentence_chunks(value, max_chars=86))
    if not sentences and cleaned:
        sentences.append(_clip_sentence(cleaned, 120))
    return _dedupe(sentences)


def _insert_soft_sentence_boundaries(text: str) -> str:
    value = str(text or "")
    for term in ("首先", "第二", "第三", "第四", "最后", "接下来", "另外", "同时", "因为", "所以", "但是", "比如", "如果", "那么", "我们要", "大家要", "客户会", "客户可能", "这一步", "这个时候"):
        value = value.replace(term, "。" + term)
    return re.sub(r"。+", "。", value).strip("。")


def _fixed_width_sentence_chunks(text: str, *, max_chars: int) -> list[str]:
    value = str(text or "").strip()
    chunks: list[str] = []
    while len(value) > max_chars:
        cut = max(value.rfind("，", 0, max_chars), value.rfind("、", 0, max_chars), value.rfind(" ", 0, max_chars))
        if cut < max_chars // 2:
            cut = max_chars
        chunks.append(value[:cut].strip(" ，,、"))
        value = value[cut:].strip(" ，,、")
    if value:
        chunks.append(value)
    return [chunk for chunk in chunks if len(chunk) >= 8]


EVIDENCE_LABEL_PATTERN = re.compile(r"\b(document_visual|semantic_frame|temporal_sequence|mixed|image|table|text|code|review_gap)\b", re.IGNORECASE)


def _clean_llm_main_text(value: Any, *, limit: int = 160) -> str:
    """Clean evidence/navigation tokens before text enters the readable summary body."""
    text = _strip_evidence_prefix(str(value or ""))
    text = EVIDENCE_LABEL_PATTERN.sub("", text)
    text = re.sub(r"\btimeline=\d+(?:,\d+)*", "", text)
    text = re.sub(r"\bevidence=[^；;\n]+", "", text)
    text = re.sub(r"[,，、；;：:]{2,}", "，", text)
    text = _clean_summary_text(text)
    text = re.sub(r"^(ok|record|hold|pass|cloud|premiere|offer|facebook|youtube|TikTok|Shopify)[,，、：:\s]+", "", text, flags=re.IGNORECASE)
    if _looks_like_evidence_label_list(text):
        return ""
    return _clip_sentence(text, limit)


def _strip_evidence_prefix(value: str) -> str:
    text = str(value or "").strip()
    for sep in ("：", ":"):
        if sep not in text:
            continue
        prefix, rest = text.split(sep, 1)
        if len(prefix) <= 260 and len(EVIDENCE_LABEL_PATTERN.findall(prefix)) >= 2:
            return rest.strip()
    return text


def _looks_like_evidence_label_list(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    labels = len(EVIDENCE_LABEL_PATTERN.findall(text))
    tokens = [part for part in re.split(r"[、,，\s/]+", text) if part]
    if labels >= 2 and labels >= max(1, len(tokens) // 2):
        return True
    if labels >= 3 and len(text) < 120:
        return True
    return False


def _clean_summary_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    for filler in FILLERS:
        text = text.replace(filler, "")
    text = re.sub(r"[，,、]{2,}", "，", text)
    text = re.sub(r"^(所以|但是|然后|那么|那|比如说|注意|其实|就是|这个|这个呢|那么呢|好)+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ，,、。")
    return text

def _sentence_score(sentence: str, *, keywords: tuple[str, ...] = TEACHING_KEYWORDS) -> int:
    text = str(sentence or "").strip()
    if not text:
        return -10
    score = sum(2 for key in keywords if key and key in text)
    score += sum(1 for key in TEACHING_KEYWORDS if key in text)
    if 18 <= len(text) <= 90:
        score += 3
    if len(text) < 10 or len(text) > 150:
        score -= 2
    return score


def _dedupe_snippets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = re.sub(r"\W+", "", str(row.get("text") or "").lower())[:42]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _strip_timestamp_noise(text: str) -> str:
    return re.sub(r"\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?", "", str(text or "")).strip(" ：:-")


def _clip_sentence(text: str, limit: int) -> str:
    value = _clean_summary_text(text)
    return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip(" ，,、；;") + "…"


def _seconds(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except Exception:
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out

def _preferred_manifest_corrected_transcript(root: Path, manifest: dict[str, Any]) -> tuple[str, Path] | None:
    for key in (
        "source_arbitrated_transcript_json",
        "human_corrected_transcript_json",
        "llm_readable_transcript_json",
        "agent_readable_transcript_json",
        "readable_transcript_json",
        "llm_corrected_transcript_json",
        "corrected_transcript_json",
    ):
        value = str(manifest.get(key) or "").strip()
        if not value:
            continue
        path = _bundle_path(root, value)
        if path.exists():
            return key, path
    return None


def _transcript_source_content_gate(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        return {
            "passed": True,
            "status": "source_quality_not_applicable",
            "review_segment_count": 0,
        }
    try:
        data = read_json(path)
    except Exception as exc:
        return {
            "passed": False,
            "status": "transcript_source_unreadable",
            "review_segment_count": 0,
            "detail": str(exc),
        }
    quality = data.get("quality_summary") if isinstance(data, dict) else None
    if not isinstance(quality, dict):
        return {
            "passed": True,
            "status": "legacy_source_without_quality_summary",
            "review_segment_count": 0,
        }
    policy = (
        quality.get("summary_input_policy")
        if isinstance(quality.get("summary_input_policy"), dict)
        else {}
    )
    review_refs = quality.get("review_segment_refs")
    review_refs = review_refs if isinstance(review_refs, list) else []
    reasons = [
        str(row.get("reason") or "").casefold()
        for row in review_refs
        if isinstance(row, dict)
    ]
    content_gap_markers = (
        "low_text_density",
        "content_gap",
        "missing_segment",
        "probable_no_speech",
        "instruction_leak",
    )
    explicit_allowed = quality.get("can_use_as_summary_input")
    policy_allowed = policy.get("can_use_corrected_transcript")
    has_content_gap = any(
        marker in reason
        for reason in reasons
        for marker in content_gap_markers
    )
    if (
        explicit_allowed is False
        or policy_allowed is False
        or has_content_gap
    ):
        return {
            "passed": False,
            "status": "transcript_content_gaps",
            "review_segment_count": len(review_refs),
            "review_segment_refs": review_refs,
            "source_quality_status": str(quality.get("status") or ""),
            "detail": (
                "canonical transcript contains unresolved review/content gaps "
                "and is not eligible for Smart Summary production input"
            ),
        }
    return {
        "passed": True,
        "status": "source_quality_eligible",
        "review_segment_count": len(review_refs),
        "source_quality_status": str(quality.get("status") or ""),
    }


def _smart_summary_transcript_source_gate(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        manifest = {}
    preferred = _preferred_manifest_corrected_transcript(root, manifest)
    pack_path = root / "exports" / "smart-summary-input-pack.json"
    if pack_path.exists():
        try:
            pack = read_json(pack_path)
        except Exception as exc:
            return {"passed": False, "status": "input_pack_unreadable", "detail": str(exc)}
        if isinstance(pack, dict):
            decision = pack.get("transcript_source_decision") if isinstance(pack.get("transcript_source_decision"), dict) else {}
            uses_corrected = bool(decision.get("uses_corrected_transcript"))
            label = str(pack.get("transcript_source_label") or decision.get("selected_label") or "")
            pack_source = str(pack.get("transcript_source") or "").strip()
            if preferred:
                preferred_key, preferred_path = preferred
                try:
                    pack_path_resolved = Path(pack_source).expanduser().resolve() if pack_source else Path()
                except Exception:
                    pack_path_resolved = Path()
                if pack_path_resolved != preferred_path.resolve():
                    return {
                        "passed": False,
                        "status": "corrected_input_pack_stale",
                        "transcript_source": str(preferred_path),
                        "transcript_source_label": preferred_key,
                        "detail": f"manifest prefers {preferred_key}; smart-summary input pack still points to {label or pack_source or 'unknown'}",
                    }
            source_path = preferred[1] if preferred else None
            if source_path is None and pack_source:
                try:
                    source_path = Path(pack_source).expanduser().resolve()
                except Exception:
                    source_path = None
            if (
                uses_corrected
                and source_path is not None
                and source_path.exists()
            ):
                content_gate = _transcript_source_content_gate(source_path)
                if not content_gate["passed"]:
                    return {
                        **content_gate,
                        "transcript_source": str(source_path),
                        "transcript_source_label": label,
                    }
            return {
                "passed": uses_corrected,
                "status": "corrected" if uses_corrected else "raw_or_timeline_fallback",
                "transcript_source": pack_source,
                "transcript_source_label": label,
                "detail": "smart-summary input uses corrected transcript" if uses_corrected else f"smart-summary input is not corrected transcript: {label or 'unknown'}",
            }
    if preferred:
        key, path = preferred
        content_gate = _transcript_source_content_gate(path)
        if not content_gate["passed"]:
            return {
                **content_gate,
                "transcript_source": str(path),
                "transcript_source_label": key,
            }
        return {"passed": True, "status": "corrected_manifest", "transcript_source": str(path), "transcript_source_label": key, "detail": f"manifest has {key}"}
    return {"passed": False, "status": "missing_corrected_transcript", "detail": "smart-summary final quality requires corrected/source-arbitrated transcript input"}


def _summary_transcript_completeness_gate(root: Path) -> dict[str, Any]:
    """Require verified speech completeness before a summary is production-ready.

    Intent: stop a readable summary from masking a missing or unverified source
    transcript.
    Decision: reuse the persisted transcript-quality gate and its source
    completeness evidence; no ASR or VAD work is started here.
    Reason: one production video has proven non-silent empty chunks, while the
    other used a long single pass that has not been independently verified.
    Evidence: transcript-quality-gate.json/source_completeness.
    Effective scope: Smart Summary quality status only; existing summary text
    and transcript artifacts are never overwritten.
    """

    return transcript_completeness_status(root)


def _quality_transcript_text(source_gate: dict[str, Any]) -> str:
    value = str(source_gate.get("transcript_source") or "").strip()
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.exists():
        return ""
    try:
        cues = parse_transcript(path)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")
    return " ".join(str(cue.text or "") for cue in cues).strip()


def _quality_summary_body(text: str, *, interview_profile: bool = False) -> str:
    headings = (
        ("## 一句话概览", "## 核心主题", "## 事实时间线", "## 受访者原话与感受", "## 明确后续事项", "## 原话摘录")
        if interview_profile
        else ("## 一句话概览", "## 核心主题", "## 分段总结", "## 关键观点", "## 可执行动作清单", "## 高频话术")
    )
    return "\n".join(_section_text(text, heading) for heading in headings)


def _compact_quality_text(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value or ""))


def _quality_number_evidence(value: str) -> dict[str, set[str]]:
    without_times = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?\b", "", str(value or ""))
    without_heading_numbers = re.sub(
        r"(?m)^(\s{0,3}#{1,6}\s+)\d{1,2}(?:\.\d+)*(?:(?:[.)、:：-])\s*|\s+)",
        r"\1",
        without_times,
    )
    without_hierarchical_ordinals = re.sub(
        r"(?m)^(\s*(?:[-*+]\s+)?)\d{1,2}(?:\.\d+){1,3}"
        r"(?:(?:[.)、:：-])\s*|\s+)",
        r"\1",
        without_heading_numbers,
    )
    without_list_ordinals = re.sub(
        r"(?m)^\s*\d+(?:[.)]\s+|、\s*)",
        "",
        without_hierarchical_ordinals,
    )
    return number_evidence_map(without_list_ordinals)


def _structural_heading_numbers(value: str) -> list[str]:
    return sorted(
        {
            match.group(1)
            for match in re.finditer(
                r"(?m)^\s{0,3}#{1,6}\s+(\d{1,2})(?:(?:[.)、:：-])\s*|\s+)",
                str(value or ""),
            )
        },
        key=int,
    )


def _resolve_summary_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _normalize_markdown_newlines(value: str) -> str:
    text = str(value or "")
    if text.count("\n") <= 1 and "\\n" in text:
        return text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return text


def _number_display_mentions(mentions: set[str]) -> list[str]:
    values: set[str] = set()
    for mention in mentions:
        range_match = re.search(
            r"\d+(?:\.\d+)?\s*(?:-|–|—|~|～|到|至)\s*"
            r"\d+(?:\.\d+)?(?:场)?",
            mention,
        )
        match = range_match or re.search(r"\d+(?:\.\d+)?%?", mention)
        values.add(match.group(0) if match else mention)
    return sorted(values)


def _quality_numbers(value: str) -> list[str]:
    return sorted(
        {
            mention
            for mentions in _quality_number_evidence(value).values()
            for mention in _number_display_mentions(mentions)
        }
    )


def _unsupported_number_claims(
    value: str, unsupported_numbers: list[str]
) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    for number in unsupported_numbers:
        excerpt = next((line for line in lines if number in line), "")
        claims.append(
            {
                "number": number,
                "excerpt": excerpt[:300],
                "status": "requires_timestamped_evidence_or_human_confirmation",
            }
        )
    return claims


def _strip_quality_metadata_identifiers(value: str, root: Path) -> str:
    manifest = _read_mapping(root / "manifest.json")
    title = str(manifest.get("title") or "").strip()
    text = str(value or "")
    return text.replace(title, "") if title else text


def _trusted_visual_number_evidence(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Return numeric claims proven by successful local ebook OCR artifacts only."""
    timeline_path = root / "timeline.json"
    if not timeline_path.is_file():
        return {}
    try:
        raw_timeline = read_json(timeline_path)
    except Exception:
        return {}
    if isinstance(raw_timeline, list):
        timeline = raw_timeline
    elif isinstance(raw_timeline, dict) and isinstance(raw_timeline.get("items"), list):
        timeline = raw_timeline["items"]
    else:
        return {}
    evidence_by_number: dict[str, list[dict[str, Any]]] = {}
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        status = item.get("ebook_pipeline_status")
        if not isinstance(status, dict) or status.get("ok") is not True:
            continue
        expected_artifact = str(status.get("artifact_path") or "").strip()
        structured = item.get("structured_visual")
        rows = structured if isinstance(structured, list) else [structured]
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("source") or "") != "ebook_markdown_pipeline":
                continue
            artifact_path = str(row.get("artifact_path") or expected_artifact).strip()
            if expected_artifact and artifact_path and artifact_path != expected_artifact:
                continue
            markdown = str(row.get("markdown") or "").strip()
            if not markdown:
                continue
            evidence = {
                "source_kind": "ebook_ocr",
                "timeline_index": int(item.get("index") or position),
                "time_range": f"{format_timestamp(_seconds(item.get('start')))} - {format_timestamp(_seconds(item.get('end')))}",
                "artifact_path": artifact_path,
            }
            for key, mentions in _quality_number_evidence(markdown).items():
                evidence_by_number.setdefault(key, []).append(
                    {**evidence, "source_mentions": sorted(mentions)}
                )
    return evidence_by_number


def _summary_repetition_rate(value: str) -> float:
    paragraphs = [re.sub(r"\s+", "", part) for part in re.split(r"\n\s*\n", str(value or ""))]
    paragraphs = [part for part in paragraphs if len(part) >= 40]
    if not paragraphs:
        return 0.0
    duplicate_count = len(paragraphs) - len(set(paragraphs))
    return round(duplicate_count / len(paragraphs), 6)


def _human_key_point_recall(root: Path, summary_body: str) -> dict[str, Any]:
    candidates = [root / "quality-benchmark-key-points.json", root / "exports" / "human-key-points.json"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if not path:
        return {"evaluated": False, "recall": None, "detail": "human key points not provided; benchmark pending"}
    try:
        data = read_json(path)
    except Exception as exc:
        return {"evaluated": False, "recall": None, "detail": f"human key points unreadable: {exc}"}
    return evaluate_human_key_point_recall(
        data,
        summary_body,
        source_path=path,
        bundle_root=root,
    )


def _smart_summary_invalidation_gate(root: Path, summary_path: Path) -> dict[str, Any]:
    marker_path = root / "exports" / "smart-summary-invalidation.json"
    if not marker_path.exists():
        return {
            "passed": True,
            "status": "not_invalidated",
            "detail": "no transcript-triggered summary invalidation marker exists",
        }
    try:
        marker = read_json(marker_path)
    except Exception as exc:
        return {
            "passed": False,
            "status": "invalid_invalidation_marker",
            "detail": f"cannot validate summary freshness marker: {type(exc).__name__}: {exc}",
        }
    if not isinstance(marker, dict):
        return {
            "passed": False,
            "status": "invalid_invalidation_marker",
            "detail": "summary invalidation marker must be a JSON object",
        }
    invalidated = {
        str(row.get("sha256") or "")
        for row in marker.get("invalidated_summaries") or []
        if isinstance(row, dict) and str(row.get("sha256") or "")
    }
    if not summary_path.exists():
        return {
            "passed": False,
            "status": "summary_missing_after_invalidation",
            "detail": "summary was invalidated after transcript update and has not been regenerated",
        }
    current_hash = sha256_file(summary_path)
    if current_hash in invalidated:
        return {
            "passed": False,
            "status": "summary_invalidated_after_transcript_update",
            "summary_sha256": current_hash,
            "marker_path": str(marker_path),
            "detail": "summary content matches an invalidated pre-repair summary; regenerate from the current canonical transcript",
        }
    return {
        "passed": True,
        "status": "regenerated_after_invalidation",
        "summary_sha256": current_hash,
        "marker_path": str(marker_path),
        "detail": "summary content no longer matches any invalidated pre-repair summary",
    }

def _smart_summary_freshness_gate(summary_path: Path, transcript_source_gate: dict[str, Any]) -> dict[str, Any]:
    if not bool(transcript_source_gate.get("passed")):
        return {
            "passed": False,
            "status": "missing_corrected_transcript",
            "detail": "freshness cannot pass before corrected/source-arbitrated transcript input is available",
        }
    transcript_source = str(transcript_source_gate.get("transcript_source") or "").strip()
    if not transcript_source:
        return {"passed": True, "status": "no_source_path", "detail": "corrected transcript gate passed without a concrete source path"}
    source_path = Path(transcript_source).expanduser()
    if not source_path.exists() or not summary_path.exists():
        return {
            "passed": False,
            "status": "missing_artifact",
            "detail": f"summary_path={summary_path}; transcript_source={source_path}",
        }
    summary_mtime = summary_path.stat().st_mtime
    source_mtime = source_path.stat().st_mtime
    if summary_mtime + 0.001 < source_mtime:
        source_payload = _read_mapping(source_path)
        if (
            source_payload.get("schema")
            == "video_knowledge_pipeline.source_arbitrated_transcript.v1"
            and source_payload.get("source")
            == "transcript_semantic_correction_no_change"
        ):
            return {
                "passed": True,
                "status": "unchanged_transcript_after_review",
                "summary_path": str(summary_path),
                "transcript_source": str(source_path),
                "detail": "the newer source-arbitrated transcript records a verified no-change semantic review",
            }
        return {
            "passed": False,
            "status": "summary_stale_after_transcript_update",
            "summary_path": str(summary_path),
            "transcript_source": str(source_path),
            "detail": "smart-summary is older than corrected/source-arbitrated transcript; regenerate chapter LLM summary or rerun export",
        }
    return {
        "passed": True,
        "status": "fresh",
        "summary_path": str(summary_path),
        "transcript_source": str(source_path),
        "detail": "smart-summary is not older than corrected/source-arbitrated transcript",
    }


def write_smart_summary_dependency_snapshot(
    bundle_dir: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Record the stable evidence inputs used by an installed final summary.

    This reuses VKP's shared artifact-freshness snapshot contract. It deliberately
    excludes mutable workflow/quality reports and derived input packs so an
    ordinary preflight cannot invalidate a previously authorized summary; transcript,
    timeline, accepted section revisions, and imported courseware remain bound.
    """

    root = Path(bundle_dir).expanduser().resolve()
    inputs = _smart_summary_dependency_inputs(root)
    if not inputs:
        raise FileNotFoundError("no stable smart-summary evidence inputs are available")
    snapshot = build_dependency_snapshot(
        root,
        subject="smart-summary-final",
        inputs=inputs,
        source_run_id="smart-summary-codex",
        producer_schema=SCHEMA,
    )
    snapshot_path = root / "exports" / "smart-summary-dependency-snapshot.json"
    if write:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(snapshot_path, snapshot)
    return {
        "status": "recorded",
        "snapshot_path": str(snapshot_path),
        "input_count": len(inputs),
        "snapshot": snapshot,
    }


def _smart_summary_dependency_inputs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def add(role: str, path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if not resolved.is_file() or resolved in seen or not resolved.is_relative_to(root):
            return
        seen.add(resolved)
        rows.append({"role": role, "path": resolved})

    add("timeline", root / "timeline.json")
    add("accepted_section_revisions", root / "exports" / "smart-summary-section-llm-revisions.json")
    transcript_source = str(_smart_summary_transcript_source_gate(root).get("transcript_source") or "").strip()
    if transcript_source:
        add("canonical_transcript", Path(transcript_source))
    manifest = _read_mapping(root / "manifest.json")
    for key, role in (
        ("companion_courseware_text", "companion_courseware"),
        ("companion_courseware_text_markdown", "companion_courseware_markdown"),
    ):
        value = str(manifest.get(key) or "").strip()
        if value:
            add(role, _bundle_path(root, value))
    return rows


def _smart_summary_evidence_freshness_gate(root: Path, *, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot_path = root / "exports" / "smart-summary-dependency-snapshot.json"
    prospective = snapshot is not None
    if snapshot is None and not snapshot_path.is_file():
        return {
            "passed": True,
            "status": "legacy_not_tracked",
            "detail": "no summary evidence snapshot exists; regenerate the final summary once to bind current evidence",
        }
    try:
        snapshot = read_json(snapshot_path) if snapshot is None else snapshot
        if prospective:
            expected = write_smart_summary_dependency_snapshot(root, write=False)['snapshot']
            if snapshot.get('snapshot_sha256') != expected['snapshot_sha256']:
                raise ValueError('prospective snapshot must bind the complete current stable input set')
        validation = validate_dependency_snapshot(root, snapshot)
    except Exception as exc:
        return {
            "passed": False,
            "status": "invalid_summary_evidence_snapshot",
            "detail": f"cannot validate summary evidence snapshot: {type(exc).__name__}: {exc}",
        }
    status = str(validation.get("status") or "invalid")
    if status == "fresh":
        return {
            "passed": True,
            "status": "fresh",
            "snapshot_path": str(snapshot_path),
            "validation_basis": "prospective_installation" if prospective else "installed_snapshot",
            "detail": "smart-summary evidence inputs match the complete prospective installation snapshot" if prospective else "smart-summary evidence inputs match the installed dependency snapshot",
        }
    return {
        "passed": False,
        "status": f"summary_stale_after_evidence_{status}",
        "snapshot_path": str(snapshot_path),
        "issues": validation.get("issues") or [],
        "detail": "transcript, timeline, accepted section revision, or imported courseware changed; rerun section apply and quality review before export",
    }


def _existing_codex_summary(root: Path) -> Path | None:
    for name in CODEX_FILENAMES:
        path = root / name
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            text = path.read_text(encoding="utf-8-sig")
            if _is_codex_summary(path, text) and not _is_non_llm_scaffold(text):
                return path.resolve()
    return None


def _is_non_llm_scaffold(text: str) -> bool:
    return any(marker in text for marker in ("local_scaffold_not_llm", "codex_assisted_draft"))


def _is_codex_summary(path: Path, text: str) -> bool:
    """Recognize canonical LLM summaries without exposing program fields.

    Intent: keep the reader-facing Markdown free of internal generation metadata.
    Decision: accept the hidden final marker emitted by the deterministic reader renderer,
    while preserving the legacy visible generation-mode line.
    Reason: the renderer and selector must share one terminal-state contract.
    Evidence: smart_summary_reader_plan.render_reader_summary emits an HTML comment
    and the previous selector reported that existing candidate as missing.
    Effective scope: canonical Smart Summary selection only; quality, evidence,
    provider routing, transcript and Timeline remain unchanged.
    """

    del path
    if "VKP review-required" in text:
        return False
    final_modes = (
        "codex_final",
        "codex_llm_rewrite_final",
        "codex_llm_rewrite_substitute",
        "codex_first_llm_substitute",
        "online_llm_section_rewrite",
    )
    hidden_pattern = (
        r"<!--\s*(?:" + "|".join(re.escape(mode) for mode in final_modes) + r")\s*-->"
    )
    if re.search(hidden_pattern, text):
        return True
    pattern = r"(?m)^\s*生成方式\s*[：:]\s*\x60?(?:" + "|".join(re.escape(mode) for mode in final_modes) + r")\b"
    return bool(re.search(pattern, text))

def _transcript_max_seconds(root: Path) -> float:
    candidates = []
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        manifest = {}
    for key in ("human_corrected_transcript_json", "llm_corrected_transcript_json", "corrected_transcript_json", "source_arbitrated_transcript_json", "normalized_transcript_json", "transcript_json", "source_transcript", "transcript_path"):
        value = str(manifest.get(key) or "").strip()
        if value:
            candidates.append(_bundle_path(root, value))
    candidates.extend([root / "human-corrected-transcript.json", root / "llm-corrected-transcript.json", root / "corrected-transcript.json", root / "source-arbitrated-transcript.json", root / "normalized-transcript.json", root / "transcript.json"])
    for path in candidates:
        if path.exists():
            try:
                cues = parse_transcript(path)
            except Exception:
                continue
            if cues:
                return max(float(getattr(cue, "end", 0.0) or 0.0) for cue in cues)
    full_transcript = root / "exports" / "full-transcript.md"
    if full_transcript.exists():
        times = _timestamps(full_transcript.read_text(encoding="utf-8-sig"))
        if times:
            return max(times)
    return 0.0


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _timestamps(text: str) -> list[float]:
    values = []
    for match in re.finditer(
        r"(?<![\dT:])(?:(?P<h>\d{1,2}):)?(?P<m>[0-5]\d):(?P<s>[0-5]\d)(?:\.(?P<ms>\d{1,3}))?(?!\d)",
        text,
    ):
        h = int(match.group("h") or 0)
        m = int(match.group("m"))
        s = int(match.group("s"))
        ms = int((match.group("ms") or "0").ljust(3, "0")[:3])
        values.append(h * 3600 + m * 60 + s + ms / 1000)
    return values


def _section_text(text: str, heading: str) -> str:
    prefix, title = heading.split(" ", 1)
    pattern = re.compile(
        rf"(?m)^{re.escape(prefix)}\s+(?:\d+(?:\.\d+)*[.)、]?\s+)?{re.escape(title.strip())}(?:\s+/.*)?\s*$"
    )
    match = pattern.search(text)
    if not match:
        return ""
    next_match = re.search(r"\n##\s+", text[match.end() :])
    if not next_match:
        return text[match.end() :]
    return text[match.end() : match.end() + next_match.start()]


def _section_coverage(text: str, transcript_max: float, *, interview_profile: bool = False) -> dict[str, Any]:
    if transcript_max <= 0:
        return {"passed": False, "detail": "missing transcript duration"}
    if interview_profile:
        times = _timestamps(_section_text(text, "## 事实时间线"))
        buckets = {min(3, int((value / transcript_max) * 4)) for value in times if value >= 0}
        passed = len(buckets) >= 3 and bool(times) and max(times) >= transcript_max * 0.75
        return {"passed": passed, "detail": f"## 事实时间线:{sorted(buckets)}"}
    details = []
    passed = True
    for heading in ("## 关键观点", "## 可执行动作清单", "## 高频话术"):
        times = _timestamps(_section_text(text, heading))
        buckets = {min(3, int((value / transcript_max) * 4)) for value in times if value >= 0}
        ok = len(buckets) >= 2 or (times and max(times) >= transcript_max * 0.75)
        passed = passed and ok
        details.append(f"{heading}:{sorted(buckets)}")
    return {"passed": passed, "detail": "; ".join(details)}


def _summary_timeline_anchor_quality(
    times: list[float],
    transcript_max: float,
) -> dict[str, Any]:
    """Validate summary timestamps against the transcript's first/last quarters.

    This is the deterministic counterpart of YouTube Digest's prompt-only
    chapter coverage rule. It does not infer missing timestamps or change the
    summary; it only prevents out-of-range or tail-only timestamps from passing.
    """

    if transcript_max <= 0 or not times:
        return {
            "timestamp_range_passed": False,
            "first_last_passed": False,
            "detail": "missing transcript duration or summary timestamps",
        }
    tolerance = max(0.5, min(2.0, transcript_max * 0.002))
    ordered = sorted(float(value) for value in times)
    range_passed = all(0 <= value <= transcript_max + tolerance for value in ordered)
    first_last_passed = (
        ordered[0] <= transcript_max * 0.25 + tolerance
        and ordered[-1] >= transcript_max * 0.75 - tolerance
    )
    return {
        "timestamp_range_passed": range_passed,
        "first_last_passed": first_last_passed,
        "detail": (
            f"first={ordered[0]:.3f}; last={ordered[-1]:.3f}; "
            f"transcript_max={transcript_max:.3f}; tolerance={tolerance:.3f}"
        ),
    }


def _check(key: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"key": key, "passed": bool(passed), "detail": detail}


def _missing_quality(root: Path) -> dict[str, Any]:
    return {
        "schema": QUALITY_SCHEMA,
        "bundle_dir": str(root),
        "status": "missing_codex_summary",
        "passed": False,
        "checks": [_check("codex_final", False, "smart-summary.codex.md is missing")],
        "checked_at": now_iso(),
    }


def _next_actions(root: Path, *, status: str, quality: dict[str, Any]) -> list[str]:
    if status != "ready":
        return [
            "open exports/smart-summary-codex-prompt.md in Codex",
            "write exports/smart-summary.codex.md",
            f"run .\\scripts\\video-knowledge.ps1 generate-smart-summary-with-codex \"{root}\"",
            f"run .\\scripts\\video-knowledge.ps1 export-knowledge-note \"{root}\"",
        ]
    if not quality.get("passed"):
        return ["revise exports/smart-summary.codex.md until smart-summary-quality passes", f"rerun .\\scripts\\video-knowledge.ps1 generate-smart-summary-with-codex \"{root}\""]
    return [f"run .\\scripts\\video-knowledge.ps1 export-knowledge-note \"{root}\" to install final smart-summary.md"]


def _render_status_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Smart Summary Codex Status",
        "",
        f"- Bundle: `{result.get('bundle_dir')}`",
        f"- Status: `{result.get('status')}`",
        f"- Codex summary: `{result.get('smart_summary_codex_path')}`",
        f"- Prompt: `{result.get('prompt_path')}`",
        f"- Quality: `{(result.get('quality') or {}).get('status')}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in result.get("next_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines).rstrip() + "\n"


def _render_quality_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Smart Summary Quality",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Summary: `{result.get('summary_path')}`",
        f"- Is Codex summary: `{result.get('is_codex_summary')}`",
        f"- Automated checks passed: `{result.get('automated_checks_passed')}`",
        f"- Quality evidence complete: `{result.get('quality_evidence_complete')}`",
        f"- Production ready: `{result.get('production_ready')}`",
        f"- Is draft: `{result.get('is_draft')}`",
        f"- Transcript max: `{_fmt(float(result.get('transcript_max_seconds') or 0))}`",
        f"- Summary max: `{_fmt(float(result.get('summary_max_seconds') or 0))}`",
        "",
        "| Check | Passed | Detail |",
        "| --- | --- | --- |",
    ]
    for row in result.get("checks") or []:
        passed_label = (
            str(row.get("status") or "not_evaluated")
            if row.get("evaluated") is False
            else str(bool(row.get("passed"))).lower()
        )
        lines.append(
            f"| `{row.get('key')}` | `{passed_label}` | "
            f"{str(row.get('detail') or '').replace('|', '/')} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _fmt(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
