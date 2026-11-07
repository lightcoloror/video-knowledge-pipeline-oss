from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .asr_adapter import render_srt
from .model_task_gateway import model_task_api_call
from .models import TranscriptCue, now_iso
from .punctuation_model_stage import _content_only
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .text_llm_gateway import extract_json_document, openai_compatible_chat_completions_url, resolve_text_provider_config
from .transcript import format_timestamp, parse_transcript
from .transcript_speakers import speaker_display_name, speaker_label_map, speaker_payload
from .vision_api import redact_url_secrets


def call_openai_compatible_text(*, provider_config, messages, temperature=0, response_format=None, max_tokens=None):
    return model_task_api_call("transcript_readable_polish", provider_config=provider_config, messages=messages, execute=True, temperature=temperature, response_format=response_format, max_tokens=max_tokens, write=False)

SCHEMA = "video_knowledge_pipeline.readable_transcript_llm_polish.v1"
OUTPUT_SCHEMA = "video_knowledge_pipeline.llm_readable_transcript.v1"

_NUMBER_ANCHOR_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:万|亿|千|百)?")
_PERCENTAGE_ANCHOR_RE = re.compile(
    r"(?:百分之\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*[%％])"
)
_MONEY_ANCHOR_RE = re.compile(
    r"(?:[$￥¥]\s*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s*(?:人民币|美元|港元|元|块钱|万元|亿元))"
)
_DATE_ANCHOR_RE = re.compile(
    r"(?:\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}\s*(?:日|号)?|\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*(?:日|号))?)?)"
)
_ENTITY_CANDIDATE_RE = re.compile(
    r"[A-Za-z0-9一-鿿·]{1,24}(?:人寿|保险|银行|公司|集团|大学|学院|机构|医院|基金|证券|科技|平台|产品|系统|模型|工具|老师|先生|女士)"
)
_LATIN_TOKEN_PART_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_NEGATION_ANCHOR_RE = re.compile(
    r"不得|不能|不可以|不可|不会|不再|无需|毋须|没有|并非|未必|禁止|拒绝|否认|不是|不|没|未|无|非"
)
_CONDITION_ANCHOR_RE = re.compile(
    r"只要|只有|如果|假如|倘若|若是|若|除非|在[^，。！？；,.!?;]{0,20}情况下|才|方可"
)
_EXCEPTION_ANCHOR_RE = re.compile(
    r"除此之外|除[^，。！？；,.!?;]{0,16}之外|除外|例外|但不包括|不包括|仅限|除非|但是|不过|然而|但"
)


def _anchor_values(pattern: re.Pattern[str], value: str, *, surface_normalized: bool = True) -> list[str]:
    matches = [match.group(0) for match in pattern.finditer(str(value or ""))]
    if surface_normalized:
        return [_content_only(match) for match in matches]
    return matches


def _readable_semantic_guard_issues(source_text: str, candidate_text: str) -> list[dict[str, Any]]:
    """Return fail-closed issues when a readable edit changes semantic content.

    The readable stage is a surface formatter, not a correction stage.  It may
    change Unicode punctuation and whitespace (including paragraph boundaries),
    but the ordered non-punctuation character stream must remain byte-for-byte
    equivalent.  Named high-risk anchors are also compared so a rejected import
    explains *what* changed instead of reporting only a generic character lock.
    """

    source = str(source_text or "")
    candidate = str(candidate_text or "")
    checks: tuple[tuple[str, re.Pattern[str], bool], ...] = (
        ("percentage_anchor_changed", _PERCENTAGE_ANCHOR_RE, True),
        ("date_anchor_changed", _DATE_ANCHOR_RE, True),
        ("money_anchor_changed", _MONEY_ANCHOR_RE, True),
        ("number_anchor_changed", _NUMBER_ANCHOR_RE, True),
        ("named_entity_candidate_changed", _ENTITY_CANDIDATE_RE, True),
        ("negation_anchor_changed", _NEGATION_ANCHOR_RE, True),
        ("condition_anchor_changed", _CONDITION_ANCHOR_RE, True),
        ("exception_anchor_changed", _EXCEPTION_ANCHOR_RE, True),
        # Compare ordered Latin alphanumeric parts.  This permits whitespace or
        # punctuation around a token while preserving its spelling and order.
        ("latin_token_anchor_changed", _LATIN_TOKEN_PART_RE, False),
    )
    issues: list[dict[str, Any]] = []
    for kind, pattern, surface_normalized in checks:
        before = _anchor_values(pattern, source, surface_normalized=surface_normalized)
        after = _anchor_values(pattern, candidate, surface_normalized=surface_normalized)
        if before != after:
            issues.append(
                {
                    "kind": kind,
                    "source_anchors": before,
                    "candidate_anchors": after,
                    "detail": "ordered semantic anchors changed; use evidence correction instead",
                }
            )
    source_content = _content_only(source)
    candidate_content = _content_only(candidate)
    if source_content != candidate_content:
        issues.append(
            {
                "kind": "content_character_sequence_changed",
                "source_preview": source_content[:160],
                "candidate_preview": candidate_content[:160],
                "detail": "readable edits may change punctuation, whitespace, and paragraph boundaries only",
            }
        )
    return issues


def run_readable_transcript_llm_polish(
    bundle_dir: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    input_json: str | Path | None = None,
    execute: bool = False,
    agent_substitute: bool = False,
    agent_name: str = "local_agent",
    codex_substitute: bool = False,
    promote: bool = False,
    max_segments_per_batch: int = 40,
    max_prompt_chars: int = 9000,
    max_tokens: int = 4000,
    temperature: float = 0,
    write: bool = True,
) -> dict[str, Any]:
    """Polish transcript punctuation and segmentation with a text LLM.

    This layer is intentionally narrow: it may add punctuation and normalize
    whitespace or paragraph boundaries, but it must preserve the ordered content
    characters exactly. Fact and wording correction remains the job of the
    evidence-conflict/source-arbitration pipelines.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    source = _resolve_source_transcript(root, manifest)
    cues = parse_transcript(source)
    has_provider_config = bool(provider_config)
    cfg = resolve_text_provider_config(provider_config) if has_provider_config else {}
    batches = _make_batches(cues, max_segments_per_batch=max_segments_per_batch, max_prompt_chars=max_prompt_chars)
    use_agent_substitute = bool(agent_substitute or codex_substitute)
    substitute_agent_name = _normalise_agent_name(agent_name, legacy_codex=codex_substitute and not agent_substitute)
    public_provider = _public_provider_config(cfg)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "source_path": str(source),
        "execute": bool(execute),
        "agent_substitute": bool(use_agent_substitute),
        "agent_substitute_name": substitute_agent_name,
        "codex_substitute": bool(codex_substitute),
        "promote": bool(promote),
        "provider": public_provider,
        "segment_count": len(cues),
        "batch_count": len(batches),
        "parameters": {
            "max_segments_per_batch": int(max_segments_per_batch),
            "max_prompt_chars": int(max_prompt_chars),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        },
        "request_plan": {
            "url": redact_url_secrets(openai_compatible_chat_completions_url(cfg)) if cfg else "",
            "model": cfg.get("model") if cfg else "",
            "call_count": len(batches),
        },
        "artifacts": {
            "json": "llm-readable-transcript.json",
            "srt": "llm-readable-transcript.srt",
            "markdown": "llm-readable-transcript.md",
            "report_json": "readable-transcript-llm-polish.json",
            "report_markdown": "readable-transcript-llm-polish.md",
            "requests_json": "exports/readable-transcript-llm-requests.json",
        },
        "operator_boundary": {
            "preview_by_default": True,
            "execute_required_for_network_call": True,
            "provider_config_runtime_only": True,
            "secrets_redacted": True,
            "does_not_process_media": True,
            "does_not_modify_raw_asr": True,
            "fact_correction_out_of_scope": True,
            "surface_character_lock_required": True,
            "promote_requires_explicit_flag": True,
            "agent_substitute_is_local_only": True,
            "agent_substitute_supported_agents": ["codex", "workbuddy", "opencode", "hermes_agent", "openclaw", "custom_local_agent"],
            "codex_substitute_is_legacy_alias": True,
        },
        "updated_at": now_iso(),
    }
    requests = [_request_record(batch) for batch in batches]
    polished_rows: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    if input_json:
        imported = _load_import_payload(input_json)
        polished_rows, failed_items = _apply_rows(cues, imported)
        result["status"] = "imported" if not failed_items else "imported_with_rejections"
        result["input_json"] = str(Path(input_json).expanduser().resolve())
    elif use_agent_substitute:
        polished_rows = _agent_substitute_polish_rows(cues)
        result["status"] = "agent_substitute_executed"
        result["legacy_status"] = "codex_substitute_executed" if codex_substitute else ""
        result["ok"] = True
        result["source"] = "local_agent_substitute"
        result["operator_boundary"]["no_cloud_call"] = True
        result["operator_boundary"]["agent_substitute_is_final_llm_placeholder"] = True
        result["compatible_agent_runtimes"] = ["codex", "workbuddy", "opencode", "hermes_agent", "openclaw", "custom_local_agent"]
    elif not execute:
        result["status"] = "planned"
        result["ok"] = True
        result["next_actions"] = [
            "Review exports/readable-transcript-llm-requests.json, then rerun with --execute and runtime provider_config.",
            "Use --promote only after the polished transcript passes review or quality gates.",
        ]
    elif not has_provider_config:
        result["status"] = "missing_provider_config"
        result["ok"] = False
        failed_items.append({"id": "provider_config", "reason": "missing_provider_config", "detail": "execute=true requires explicit runtime provider_config"})
    else:
        for batch in batches:
            messages = _messages_for_batch(root, batch)
            response = call_openai_compatible_text(
                provider_config=cfg, messages=messages, temperature=temperature,
                response_format={"type": "json_object"}, max_tokens=max_tokens,
            )
            call_row = {
                "batch_id": batch["batch_id"],
                "ok": bool(response.get("ok")),
                "error": str(response.get("error") or ""),
                "content_chars": len(str(response.get("content") or "")),
            }
            calls.append(call_row)
            if not response.get("ok"):
                failed_items.append({"id": batch["batch_id"], "reason": "llm_call_failed", "detail": str(response.get("error") or "failed")})
                continue
            try:
                payload = extract_json_document(str(response.get("content") or ""), require_object=True)
            except Exception as exc:
                failed_items.append({"id": batch["batch_id"], "reason": "model_output_parse_failed", "detail": str(exc)})
                continue
            accepted, rejected = _apply_rows(cues, payload)
            polished_rows.extend(accepted)
            failed_items.extend({**row, "batch_id": batch["batch_id"]} for row in rejected)
        result["status"] = "executed" if polished_rows and not failed_items else ("partial_failed" if polished_rows else "failed")
        result["calls"] = calls
    if polished_rows:
        output = _build_output(root, source, cues, polished_rows)
        result["polished_segment_count"] = len(output["segments"])
        result["quality"] = _quality_summary(cues, output["segments"])
        if write:
            _write_output(root, output, manifest, promote=promote)
    result["failed_items"] = failed_items
    result["ok"] = bool(result.get("ok", bool(polished_rows) and not failed_items))
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "readable-transcript-llm-requests.json", {"schema": "video_knowledge_pipeline.readable_transcript_llm_requests.v1", "requests": requests})
        write_json(root / "readable-transcript-llm-polish.json", result)
        (root / "readable-transcript-llm-polish.md").write_text(_render_report(result), encoding="utf-8")
        manifest["readable_transcript_llm_polish_json"] = "readable-transcript-llm-polish.json"
        manifest["readable_transcript_llm_polish_markdown"] = "readable-transcript-llm-polish.md"
        manifest["readable_transcript_llm_requests_json"] = "exports/readable-transcript-llm-requests.json"
        write_json(root / "manifest.json", manifest)
    result["run_artifact"] = _register_run(root, result, write=write)
    return result


def _normalise_agent_name(agent_name: str, *, legacy_codex: bool = False) -> str:
    name = str(agent_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not name or name == "local_agent":
        return "codex" if legacy_codex else "local_agent"
    aliases = {
        "hermes": "hermes_agent",
        "hermesagent": "hermes_agent",
        "open_code": "opencode",
        "open_claw": "openclaw",
    }
    return aliases.get(name, name)


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    data = read_json(path) if path.exists() else {}
    return data if isinstance(data, dict) else {}


def _resolve_source_transcript(root: Path, manifest: dict[str, Any]) -> Path:
    keys = [
        "source_arbitrated_transcript_json",
        "human_corrected_transcript_json",
        "llm_corrected_transcript_json",
        "corrected_transcript_json",
        "readable_transcript_json",
        "postprocessed_transcript_json",
        "normalized_transcript_json",
        "transcript_json",
    ]
    for key in keys:
        value = str(manifest.get(key) or "").strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path.resolve()
    for name in ("source-arbitrated-transcript.json", "corrected-transcript.json", "readable-transcript.json", "postprocessed-transcript.json", "normalized-transcript.json"):
        path = root / name
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"no transcript sidecar found for bundle: {root}")


def _make_batches(cues: list[TranscriptCue], *, max_segments_per_batch: int, max_prompt_chars: int) -> list[dict[str, Any]]:
    limit = max(1, int(max_segments_per_batch or 40))
    char_limit = max(1200, int(max_prompt_chars or 9000))
    batches: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for index, cue in enumerate(cues):
        row = {"index": index, "start": cue.start, "end": cue.end, "text": cue.text}
        row_chars = len(str(cue.text or "")) + 48
        if current and (len(current) >= limit or current_chars + row_chars > char_limit):
            batches.append(_batch(len(batches) + 1, current))
            current = []
            current_chars = 0
        current.append(row)
        current_chars += row_chars
    if current:
        batches.append(_batch(len(batches) + 1, current))
    return batches


def _batch(number: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "batch_id": f"batch-{number:03d}",
        "start_index": rows[0]["index"],
        "end_index": rows[-1]["index"],
        "segment_count": len(rows),
        "rows": rows,
    }


def _request_record(batch: dict[str, Any]) -> dict[str, Any]:
    return {"batch_id": batch["batch_id"], "start_index": batch["start_index"], "end_index": batch["end_index"], "segment_count": batch["segment_count"], "prompt_preview": _batch_prompt(batch)}


def _messages_for_batch(root: Path, batch: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是中文课程逐字稿表面格式编辑器。只做标点、空白、换行和段落边界；不得增加、删除、替换或重排任何非标点字符。词语或事实纠错必须交给证据纠错链路。必须返回 JSON。",
        },
        {"role": "user", "content": f"Bundle: {root}\n\n{_batch_prompt(batch)}"},
    ]


def _batch_prompt(batch: dict[str, Any]) -> str:
    lines = [
        "请润色下面逐字稿片段，使其更接近可读讲稿。",
        "要求：",
        "- 保留每个 index，不要新增/删除/重排 index。",
        "- 只改 text 字段。",
        "- 允许添加中文标点、空白、换行和自然段落边界；不得删除重复口头语。",
        "- 不得增加、删除、替换或重排任何非标点字符；词语和事实纠错由证据仲裁链路处理。",
        "- 返回 JSON 对象：{\"segments\":[{\"index\":0,\"text\":\"...\"}]}",
        "",
        "segments:",
    ]
    for row in batch.get("rows") or []:
        lines.append(json.dumps({"index": row.get("index"), "start": row.get("start"), "end": row.get("end"), "text": row.get("text")}, ensure_ascii=False))
    return "\n".join(lines)


def _load_import_payload(input_json: str | Path) -> dict[str, Any]:
    path = Path(input_json).expanduser().resolve()
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("readable transcript polish import must be a JSON object")
    return data


def _apply_rows(cues: list[TranscriptCue], payload: object) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = payload.get("segments") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return [], [{"id": "segments", "reason": "missing_segments_array", "detail": "payload must contain segments list"}]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in rows:
        if not isinstance(item, dict):
            rejected.append({"id": "row", "reason": "invalid_row", "detail": str(item)})
            continue
        try:
            index = int(item.get("index"))
        except Exception:
            rejected.append({"id": str(item.get("index") or ""), "reason": "invalid_index", "detail": str(item)})
            continue
        text = str(item.get("text") or "").strip()
        if index < 0 or index >= len(cues):
            rejected.append({"id": str(index), "reason": "index_out_of_range", "detail": text[:120]})
            continue
        if index in seen:
            rejected.append({"id": str(index), "reason": "duplicate_index", "detail": text[:120]})
            continue
        seen.add(index)
        original = str(cues[index].text or "")
        if not text:
            rejected.append({"id": str(index), "reason": "empty_text", "detail": original[:120]})
            continue
        if len(text) > max(24, len(original) * 2.2) or len(text) < max(1, int(len(original) * 0.35)):
            rejected.append({"id": str(index), "reason": "length_ratio_out_of_bounds", "detail": f"original={len(original)} polished={len(text)}"})
            continue
        semantic_issues = _readable_semantic_guard_issues(original, text)
        if semantic_issues:
            rejected.append(
                {
                    "id": str(index),
                    "reason": "readable_semantic_guard_failed",
                    "detail": "semantic content changed; route the edit through transcript evidence correction",
                    "issues": semantic_issues,
                }
            )
            continue
        accepted.append({"index": index, "text": text, "original_text": original})
    return accepted, rejected


def _agent_substitute_polish_rows(cues: list[TranscriptCue]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, cue in enumerate(cues):
        polished = _agent_substitute_polish_text(str(cue.text or ""))
        if polished.strip():
            rows.append({"index": index, "text": polished, "original_text": cue.text})
    return rows


def _agent_substitute_polish_text(text: str) -> str:
    """Local placeholder for agent-assisted punctuation/segmentation.

    This is deliberately conservative. It improves readability with stable
    punctuation/spacing rules but avoids unsupported fact or terminology edits.
    The rules are intentionally phrase-boundary based instead of broad global
    replacements, so they do not create artifacts such as ``那第二点，呢``.
    """

    original = str(text or "")
    value = " ".join(original.strip().split())
    if not value:
        return ""
    value = _agent_substitute_normalize_high_confidence_text(value)
    value = _agent_substitute_insert_discourse_punctuation(value)
    value = _agent_substitute_cleanup_punctuation(value)
    if value[-1] not in "。！？!?：:":
        value += "。"
    value = _agent_substitute_cleanup_punctuation(value)
    value = value.replace("：。", "：")
    if _readable_semantic_guard_issues(original, value):
        # This branch is a last-resort invariant for future local formatting
        # rules.  Never let a formatter silently become a fact corrector.
        return original
    return value


def _agent_substitute_normalize_high_confidence_text(value: str) -> str:
    """Normalize surface whitespace only.

    Historical phrase replacements lived here (for example ``买虫`` ->
    ``买重``).  Even a plausible replacement is a semantic correction and must
    now be proposed and validated by the evidence-correction pipeline.
    """

    return " ".join(str(value or "").split())


def _agent_substitute_insert_discourse_punctuation(value: str) -> str:
    colon_patterns = [
        (r"(客户(?:这时候)?说)", r"\1："),
        (r"(顾问(?:他)?是这样说的)", r"\1："),
        (r"(顾问说)", r"\1："),
        (r"(客户是这样回的说)", r"\1："),
    ]
    for pattern, repl in colon_patterns:
        value = re.sub(pattern + r"[，,:：]?", repl, value)

    comma_after_patterns = [
        r"那第二点呢",
        r"那第三个呢",
        r"那第三点呢",
        r"那第四个呢",
        r"首先第一个",
        r"首先",
        r"第二个",
        r"第三点",
        r"第四个",
        r"接下来的话",
        r"当然了",
        r"所以",
        r"但是呢",
        r"这个时候呢",
        r"其实这个时候呢",
        r"再次沟通啊",
        r"好",
    ]
    for pattern in comma_after_patterns:
        value = re.sub(rf"({pattern})(?![，。！？；：,.!?;:])", r"\1，", value)

    boundary_before = [
        "那同时",
        "那第三",
        "那接下来",
        "大家也可以",
        "当然了",
        "因为",
        "如果大家",
        "如果您",
        "这个时候",
        "让客户",
        "你看",
    ]
    for marker in boundary_before:
        value = re.sub(rf"(?<!^)(?<=[\u4e00-\u9fff0-9])(?<![，。！？；：,.!?;:])({re.escape(marker)})", r"，\1", value)

    particle_patterns = [
        (r"(?<=[\u4e00-\u9fff])(啊|呢|呃|嗯|哦)(?=(?:那|我|您|你|他|她|它|这|那|如果|因为|所以|客户|顾问|孩子|宝妈|声音|明天|中午|晚上|保单|可以|看看|听上去))", r"\1，"),
    ]
    for pattern, repl in particle_patterns:
        value = re.sub(pattern, repl, value)
    return value


def _agent_substitute_cleanup_punctuation(value: str) -> str:
    value = value.replace(" ,", "，").replace(" .", "。")
    value = re.sub(r"[，,]{2,}", "，", value)
    value = re.sub(r"[：:]{2,}", "：", value)
    value = re.sub(r"，([。！？；：,.!?;:])", r"\1", value)
    value = re.sub(r"([：:])，", r"\1", value)
    value = re.sub(r"(说|是这样说的|是这样回的说)：$", r"\1", value)
    value = re.sub(r"\bOK(?=[\u4e00-\u9fff])", "OK，", value)
    value = value.replace("那，接下来", "那接下来")
    value = value.replace("好，的", "好的")
    value = re.sub(r"(第[一二三四五六七八九十]+点)，呢，?", r"\1呢，", value)
    value = re.sub(r"(第[一二三四五六七八九十]+个)，呢，?", r"\1呢，", value)
    value = re.sub(r"\s+([，。！？；：])", r"\1", value)
    value = re.sub(r"([，。！？；：])\s+", r"\1", value)
    return value.strip()




def _codex_substitute_polish_rows(cues: list[TranscriptCue]) -> list[dict[str, Any]]:
    return _agent_substitute_polish_rows(cues)


def _codex_substitute_polish_text(text: str) -> str:
    return _agent_substitute_polish_text(text)


def _final_readable_text_cleanup(text: str) -> str:
    value = _agent_substitute_normalize_high_confidence_text(str(text or ""))
    value = _agent_substitute_cleanup_punctuation(value)
    return value.strip()


def _build_output(root: Path, source: Path, cues: list[TranscriptCue], polished_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_index = {int(row["index"]): str(row["text"]) for row in polished_rows}
    segments = []
    speaker_labels = speaker_label_map(cues)
    for index, cue in enumerate(cues):
        text = _final_readable_text_cleanup(by_index.get(index, cue.text))
        segments.append({
            "start": cue.start,
            "end": cue.end,
            "text": text,
            **speaker_payload(cue, speaker_labels),
            "metadata": {
                **dict(cue.metadata),
                "source": "llm_readable_transcript_polish" if index in by_index else "source_transcript_passthrough",
                "source_path": str(source),
                "original_text": cue.text,
                "segment_index": index,
            },
        })
    return {
        "schema": OUTPUT_SCHEMA,
        "bundle_dir": str(root),
        "source_path": str(source),
        "created_at": now_iso(),
        "segments": segments,
    }


def _write_output(root: Path, output: dict[str, Any], manifest: dict[str, Any], *, promote: bool) -> None:
    cues = [
        TranscriptCue(
            start=float(row.get("start") or 0),
            end=float(row.get("end") or 0),
            text=str(row.get("text") or ""),
            speaker=str(row.get("speaker") or ""),
            speaker_role=str(row.get("speaker_role") or ""),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in output.get("segments") or []
    ]
    write_json(root / "llm-readable-transcript.json", output)
    (root / "llm-readable-transcript.srt").write_text(render_srt(cues), encoding="utf-8")
    (root / "llm-readable-transcript.md").write_text(_render_transcript_markdown(output), encoding="utf-8")
    manifest["llm_readable_transcript_json"] = "llm-readable-transcript.json"
    manifest["llm_readable_transcript_srt"] = "llm-readable-transcript.srt"
    manifest["llm_readable_transcript_markdown"] = "llm-readable-transcript.md"
    if promote:
        write_json(root / "corrected-transcript.json", output)
        (root / "corrected-transcript.srt").write_text(render_srt(cues), encoding="utf-8")
        manifest["corrected_transcript_json"] = "corrected-transcript.json"
        manifest["corrected_transcript_srt"] = "corrected-transcript.srt"
        manifest["transcript_json"] = "corrected-transcript.json"
        manifest["transcript_srt"] = "corrected-transcript.srt"
        manifest["corrected_transcript_source"] = "llm_readable_transcript_polish"


def _quality_summary(source_cues: list[TranscriptCue], segments: list[dict[str, Any]]) -> dict[str, Any]:
    source_text = "".join(cue.text for cue in source_cues)
    output_text = "".join(str(row.get("text") or "") for row in segments)
    return {
        "source_chars": len(source_text),
        "output_chars": len(output_text),
        "source_punctuation_density_per_1000_chars": _punct_density(source_text),
        "output_punctuation_density_per_1000_chars": _punct_density(output_text),
        "segment_count": len(segments),
    }


def _punct_density(text: str) -> float:
    chars = len(str(text or ""))
    if chars <= 0:
        return 0.0
    punct = sum(1 for char in text if char in "，。？！；：,.!?;:")
    return round(punct * 1000 / chars, 2)


def _render_transcript_markdown(output: dict[str, Any]) -> str:
    lines = ["# LLM Readable Transcript", ""]
    segments = [row for row in output.get("segments") or [] if isinstance(row, dict)]
    labels = speaker_label_map(segments)
    for row in segments:
        speaker = speaker_display_name(row, labels)
        lines.append(f"- {format_timestamp(float(row.get('start') or 0))} - {format_timestamp(float(row.get('end') or 0))}  ")
        lines.append(f"  {f'**{speaker}** ' if speaker else ''}{row.get('text', '')}")
    return "\n".join(lines).rstrip() + "\n"


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Readable Transcript LLM Polish",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Execute: `{result.get('execute')}`",
        f"- Promote: `{result.get('promote')}`",
        f"- Segments: `{result.get('segment_count')}`",
        f"- Batches: `{result.get('batch_count')}`",
    ]
    quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
    if quality:
        lines.extend([
            "",
            "## Quality",
            "",
            f"- Source punctuation density: `{quality.get('source_punctuation_density_per_1000_chars')}`",
            f"- Output punctuation density: `{quality.get('output_punctuation_density_per_1000_chars')}`",
        ])
    failed = result.get("failed_items") if isinstance(result.get("failed_items"), list) else []
    if failed:
        lines.extend(["", "## Failed Items", ""])
        for row in failed[:80]:
            lines.append(f"- `{row.get('id', '')}` / `{row.get('reason', '')}`: {row.get('detail', '')}")
    return "\n".join(lines).rstrip() + "\n"


def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    status = str(result.get("status") or "planned")
    if status == "planned":
        run_status = "needs_execution"
    elif status == "missing_provider_config":
        run_status = "needs_input"
    elif status in {"failed", "partial_failed", "imported_with_rejections"}:
        run_status = "needs_review"
    else:
        run_status = "completed"
    failed = result.get("failed_items") if isinstance(result.get("failed_items"), list) else []
    return register_bundle_run(
        root,
        run_type="readable_transcript_llm_polish",
        run_id="readable-transcript-llm-polish",
        status=run_status,
        title="Readable transcript LLM punctuation and segmentation polish",
        summary=f"status={status}; segments={int(result.get('segment_count') or 0)}; promote={bool(result.get('promote'))}",
        inputs={"bundle_dir": str(root), "source_path": str(result.get("source_path") or "")},
        parameters=result.get("parameters") if isinstance(result.get("parameters"), dict) else {},
        artifacts=[{"key": key, "path": root / value} for key, value in (result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}).items()],
        failed_items=failed,
        retry_command=f".\\scripts\\video-knowledge.ps1 readable-transcript-llm-polish '{root}'",
        next_actions=["Run with --execute after provider credentials are configured."] if status == "planned" else ["Review llm-readable-transcript.md before using it as final transcript."],
        operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {},
        write=write,
    )


def _public_provider_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": cfg.get("provider"),
        "model": cfg.get("model") if cfg else "",
        "base_url": redact_url_secrets(str(cfg.get("base_url") or "")),
        "api_key_configured": bool(cfg.get("api_key")),
        "timeout_seconds": cfg.get("timeout_seconds"),
    }
