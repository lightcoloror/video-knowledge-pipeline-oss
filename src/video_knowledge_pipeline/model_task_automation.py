from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bilinote_mind_map_prompt_pack import build_bundle_mind_map_prompt_pack
from .model_task_gateway import model_task_api_call
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .term_arbitration_codex import (
    RESULT_SCHEMA,
    build_term_arbitration_codex_pack,
    validate_term_arbitration_codex_result,
)
from .text_llm_gateway import extract_json_document


TERM_SCHEMA = "video_knowledge_pipeline.term_arbitration_model_run.v1"
MIND_MAP_SCHEMA = "video_knowledge_pipeline.bilinote_mind_map_model_run.v1"


def run_term_arbitration_model(
    bundle_dir: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    execute: bool = False,
    max_terms: int = 60,
    min_confidence: float = 0.88,
    max_tokens: int = 5000,
    temperature: float = 0,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    pack = build_term_arbitration_codex_pack(
        root, max_terms=max_terms, min_confidence=min_confidence, write=write
    )
    candidates = pack.get("candidates") if isinstance(pack.get("candidates"), list) else []
    result: dict[str, Any] = {
        "schema": TERM_SCHEMA,
        "bundle_dir": str(root),
        "execute": bool(execute),
        "status": "planned",
        "ok": True,
        "candidate_count": len(candidates),
        "provider_call": {},
        "validation": {},
        "import": {},
        "artifacts": {
            "prompt_pack": "term-arbitration-codex-pack.json",
            "model_result": "term-arbitration-model-result.json",
            "run_report": "term-arbitration-model-run.json",
            "run_markdown": "term-arbitration-model-run.md",
        },
        "operator_boundary": {
            "preview_by_default": True,
            "execute_required_for_network_call": True,
            "provider_config_runtime_only": True,
            "model_output_must_pass_existing_validation": True,
            "raw_asr_not_modified": True,
        },
        "updated_at": now_iso(),
    }
    if not execute:
        result["next_actions"] = [
            "Rerun run-term-arbitration-model with --execute and runtime provider config.",
            "The existing Codex/manual import path remains available.",
        ]
        return _write_term_run(root, result, write=write)
    if not provider_config:
        result.update({"ok": False, "status": "missing_provider_config"})
        return _write_term_run(root, result, write=write)
    if not candidates:
        result["status"] = "no_candidates"
        return _write_term_run(root, result, write=write)

    messages = _term_messages(str(pack.get("title") or root.name), candidates)
    call = model_task_api_call(
        "term_arbitration",
        provider_config=provider_config,
        messages=messages,
        execute=True,
        temperature=temperature,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        write=False,
    )
    result["provider_call"] = _safe_call(call)
    if not call.get("ok"):
        result.update({"ok": False, "status": "provider_failed"})
        return _write_term_run(root, result, write=write)
    try:
        parsed = extract_json_document(str(call.get("content") or ""), require_object=True)
        if not isinstance(parsed, dict):
            raise ValueError("term arbitration output must be a JSON object")
    except Exception as exc:
        result.update({"ok": False, "status": "model_output_parse_failed", "parse_error": str(exc)})
        return _write_term_run(root, result, write=write)

    payload = {
        "schema": RESULT_SCHEMA,
        "source": "online_model_api",
        "decisions": parsed.get("decisions") or parsed.get("terms") or [],
    }
    model_result_path = root / "term-arbitration-model-result.json"
    if not write:
        result["model_result"] = payload
        result["status"] = "model_output_ready_not_persisted"
        result["next_actions"] = ["Rerun with write=true to validate and import the model result."]
        return _write_term_run(root, result, write=False)
    write_json(model_result_path, payload)
    validation = validate_term_arbitration_codex_result(
        root, input_json=model_result_path, min_confidence=min_confidence, write=write
    )
    result["validation"] = {
        "status": validation.get("status"),
        "accepted_decision_count": validation.get("accepted_decision_count", 0),
        "rejected_decision_count": validation.get("rejected_decision_count", 0),
    }
    if validation.get("status") != "ready_for_import":
        result.update({"ok": False, "status": "validation_failed"})
        return _write_term_run(root, result, write=write)
    imported = build_term_arbitration_codex_pack(
        root,
        input_json=model_result_path,
        max_terms=max_terms,
        min_confidence=min_confidence,
        write=write,
    )
    result["import"] = {
        "status": imported.get("status"),
        "accepted_decision_count": imported.get("accepted_decision_count", 0),
        "glossary_path": "term-arbitration-glossary.json",
    }
    result["status"] = "completed"
    result["ok"] = True
    return _write_term_run(root, result, write=write)


def run_bilinote_mind_map_model(
    bundle_dir: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    execute: bool = False,
    title: str = "",
    max_chars: int = 5000,
    limit: int = 0,
    max_tokens: int = 4000,
    temperature: float = 0,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    pack = build_bundle_mind_map_prompt_pack(root, title=title, max_chars=max_chars, write=write)
    prompts = pack.get("prompts") if isinstance(pack.get("prompts"), list) else []
    selected = prompts[: max(0, int(limit))] if limit and limit > 0 else prompts
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema": MIND_MAP_SCHEMA,
        "bundle_dir": str(root),
        "title": pack.get("title") or title or root.name,
        "execute": bool(execute),
        "status": "planned",
        "ok": True,
        "chunk_count": len(prompts),
        "selected_chunk_count": len(selected),
        "rows": rows,
        "failed_items": failed,
        "artifacts": {
            "prompt_pack": "exports/bilinote-mind-map-prompt-pack.json",
            "result_json": "exports/bilinote-mind-map-result.json",
            "result_markdown": "exports/bilinote-mind-map-result.md",
            "run_json": "exports/bilinote-mind-map-model-run.json",
        },
        "operator_boundary": {
            "preview_by_default": True,
            "execute_required_for_network_call": True,
            "provider_config_runtime_only": True,
            "generated_output_is_review_required": True,
        },
        "updated_at": now_iso(),
    }
    if not execute:
        result["next_actions"] = ["Rerun run-bilinote-mind-map-model with --execute and runtime provider config."]
        return _write_mind_map_run(root, result, write=write)
    if not provider_config:
        result.update({"ok": False, "status": "missing_provider_config"})
        return _write_mind_map_run(root, result, write=write)
    for prompt_row in selected:
        chunk_index = int(prompt_row.get("chunk_index") or len(rows) + 1)
        messages = prompt_row.get("messages") if isinstance(prompt_row.get("messages"), list) else []
        call = model_task_api_call(
            "bilinote_mind_map",
            provider_config=provider_config,
            messages=messages,
            execute=True,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            write=False,
        )
        if not call.get("ok"):
            failed.append({"id": f"chunk-{chunk_index}", "reason": "provider_failed", "detail": str(call.get("error") or call.get("status") or "failed")})
            continue
        try:
            parsed = extract_json_document(str(call.get("content") or ""), require_object=True)
            if not isinstance(parsed, dict):
                raise ValueError("mind-map output must be a JSON object")
        except Exception as exc:
            failed.append({"id": f"chunk-{chunk_index}", "reason": "model_output_parse_failed", "detail": str(exc)})
            continue
        rows.append({"chunk_index": chunk_index, "mind_map": parsed, "provider": _safe_call(call)})
    result["status"] = "completed" if rows and not failed else ("partial_failed" if rows else "failed")
    result["ok"] = bool(rows) and not failed
    return _write_mind_map_run(root, result, write=write)


def _term_messages(title: str, candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact = []
    for row in candidates:
        compact.append(
            {
                "candidate_id": row.get("id"),
                "raw_mentions": row.get("raw_mentions") or [],
                "canonical_guess": row.get("canonical_guess") or "",
                "risk_reasons": row.get("risk_reasons") or [],
                "evidence": row.get("evidence") or [],
            }
        )
    return [
        {"role": "system", "content": "你是视频逐字稿术语仲裁器。只能依据给出的证据判断，不得凭空纠正。严格返回 JSON。"},
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"视频：{title}",
                    "对每个候选返回 decisions。字段：candidate_id、canonical、aliases、confidence、action(replace/review/keep)、rationale、evidence_indexes、needs_human_review。",
                    "只有多源证据支持且 confidence 足够高时才能 action=replace；无须修改时 action=keep。",
                    json.dumps({"decisions": []}, ensure_ascii=False),
                    "候选与证据：",
                    json.dumps(compact, ensure_ascii=False),
                ]
            ),
        },
    ]


def _safe_call(call: dict[str, Any]) -> dict[str, Any]:
    plan = call.get("request_plan") if isinstance(call.get("request_plan"), dict) else {}
    return {
        "task": call.get("task"),
        "model_type": call.get("model_type"),
        "ok": bool(call.get("ok")),
        "status": call.get("status"),
        "error": call.get("error") or "",
        "provider": plan.get("provider") if isinstance(plan.get("provider"), dict) else {},
        "adapter_backend": plan.get("adapter_backend") or "",
        "content_chars": len(str(call.get("content") or "")),
    }


def _write_term_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    if write:
        write_json(root / "term-arbitration-model-run.json", result)
        (root / "term-arbitration-model-run.md").write_text(_render_term_run(result), encoding="utf-8")
    result["run_registry"] = register_bundle_run(
        root,
        run_type="term_arbitration_model",
        run_id="term-arbitration-model",
        status="completed" if result.get("status") == "completed" else ("needs_execution" if result.get("status") == "planned" else "needs_retry"),
        title="Term arbitration model",
        summary=f"Status {result.get('status')} / candidates {result.get('candidate_count', 0)}.",
        inputs={"bundle_dir": str(root), "prompt_pack": "term-arbitration-codex-pack.json"},
        parameters={"execute": bool(result.get("execute"))},
        artifacts=[{"key": "report", "path": root / "term-arbitration-model-run.md"}, {"key": "model_result", "path": root / "term-arbitration-model-result.json"}],
        failed_items=[] if result.get("ok") else [{"reason": result.get("status") or "failed", "detail": result.get("parse_error") or result.get("provider_call", {}).get("error") or "model arbitration incomplete"}],
        retry_command=f".\\scripts\\video-knowledge.ps1 run-term-arbitration-model {root}",
        next_actions=result.get("next_actions") or [],
        operator_boundary=result.get("operator_boundary") or {},
        write=write,
    )
    return result


def _write_mind_map_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    exports = root / "exports"
    if write:
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "bilinote-mind-map-model-run.json", result)
        if result.get("rows"):
            payload = {"schema": "video_knowledge_pipeline.bilinote_mind_map_result.v1", "title": result.get("title"), "chunks": [row.get("mind_map") for row in result["rows"]], "review_required": True}
            write_json(exports / "bilinote-mind-map-result.json", payload)
            (exports / "bilinote-mind-map-result.md").write_text(_render_mind_map(payload), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if isinstance(manifest, dict):
                manifest["bilinote_mind_map_result_json"] = "exports/bilinote-mind-map-result.json"
                manifest["bilinote_mind_map_result_markdown"] = "exports/bilinote-mind-map-result.md"
                write_json(manifest_path, manifest)
    result["run_registry"] = register_bundle_run(
        root,
        run_type="bilinote_mind_map_model",
        run_id="bilinote-mind-map-model",
        status="completed" if result.get("status") == "completed" else ("needs_execution" if result.get("status") == "planned" else "needs_retry"),
        title="BiliNote mind-map model",
        summary=f"Status {result.get('status')} / chunks {len(result.get('rows') or [])}/{result.get('selected_chunk_count', 0)}.",
        inputs={"bundle_dir": str(root), "prompt_pack": "exports/bilinote-mind-map-prompt-pack.json"},
        parameters={"execute": bool(result.get("execute")), "selected_chunk_count": result.get("selected_chunk_count", 0)},
        artifacts=[{"key": "result", "path": exports / "bilinote-mind-map-result.json"}, {"key": "report", "path": exports / "bilinote-mind-map-model-run.json"}],
        failed_items=result.get("failed_items") or [],
        retry_command=f".\\scripts\\video-knowledge.ps1 run-bilinote-mind-map-model {root}",
        next_actions=result.get("next_actions") or [],
        operator_boundary=result.get("operator_boundary") or {},
        write=write,
    )
    return result


def _render_term_run(result: dict[str, Any]) -> str:
    return "\n".join(["# Term Arbitration Model Run", "", f"- Status: `{result.get('status')}`", f"- Execute: `{result.get('execute')}`", f"- Candidates: `{result.get('candidate_count', 0)}`", f"- Validation: `{result.get('validation', {}).get('status', '')}`", ""])


def _render_mind_map(payload: dict[str, Any]) -> str:
    lines = [f"# {payload.get('title') or 'Video Mind Map'}", "", "> Model-generated structure; human review required.", ""]
    for chunk in payload.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        for node in chunk.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            lines.extend([f"## {node.get('title') or 'Untitled'}", "", str(node.get("summary") or ""), ""])
            for child in node.get("children") or []:
                if isinstance(child, dict):
                    lines.extend([f"### {child.get('title') or 'Detail'}", "", str(child.get("summary") or ""), ""])
    return "\n".join(lines)
