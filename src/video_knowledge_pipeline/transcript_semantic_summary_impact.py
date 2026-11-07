from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .interval_coverage import closed_intervals_overlap as _overlaps
from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json

SCHEMA = "video_knowledge_pipeline.transcript_semantic_summary_impact.v1"


def transcript_semantic_summary_impact_report(
    bundle_dir: str | Path,
    *,
    summary_path: str | Path | None = None,
    baseline_summary_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Report whether accepted semantic corrections visibly improved smart-summary.

    This is a local evidence report. It does not call LLMs, does not run ASR or
    vision, and does not modify transcript sources. When a baseline summary is
    available, it compares before/after counts. Otherwise it verifies the current
    smart-summary against accepted semantic-correction decisions.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    ledger = _read_optional_json(root / "transcript-semantic-correction-decision-ledger.json")
    accepted_source = ledger.get("decisions") if isinstance(ledger, dict) else []
    if not accepted_source:
        validation = _read_optional_json(root / "transcript-semantic-correction-validation.json")
        accepted_source = validation.get("accepted_decisions") if isinstance(validation, dict) else []
    accepted = [row for row in (accepted_source or []) if isinstance(row, dict)]
    summary = _summary_document(root, manifest, summary_path)
    baseline = _baseline_document(baseline_summary_path)
    corrected_context = _corrected_transcript_context(root, manifest)
    rows: list[dict[str, Any]] = []
    residual_total = 0
    corrected_hit_total = 0
    context_keyword_hit_total = 0
    baseline_residual_total = 0
    baseline_corrected_hit_total = 0
    proven_rows = 0

    summary_is_final_llm = _is_final_llm_summary_text(str(summary.get("text") or ""))
    current_clean = _countable_text(summary.get("text", ""))
    for decision in (accepted if summary_is_final_llm else []):
        original = str(decision.get("original_text") or "").strip()
        corrected = str(decision.get("corrected_text") or "").strip()
        if not original or not corrected or original == corrected:
            continue
        original_count = _count_original_residual(current_clean, original, corrected)
        corrected_count = _count_corrected_text(current_clean, corrected)
        context = _decision_context(decision, corrected_context)
        context_keywords = _context_keywords(context.get("text", ""), corrected_text=corrected)
        context_hits = _keyword_hits(current_clean, context_keywords)
        context_proven = bool(original_count == 0 and (corrected_count > 0 or _context_absorbed(context_keywords, context_hits)))
        baseline_original_count = 0
        baseline_corrected_count = 0
        if baseline.get("exists"):
            baseline_clean = _countable_text(baseline.get("text", ""))
            baseline_original_count = _count_original_residual(baseline_clean, original, corrected)
            baseline_corrected_count = _count_corrected_text(baseline_clean, corrected)
        residual_total += original_count
        corrected_hit_total += corrected_count
        context_keyword_hit_total += len(context_hits)
        baseline_residual_total += baseline_original_count
        baseline_corrected_hit_total += baseline_corrected_count
        if context_proven:
            proven_rows += 1
        rows.append(
            {
                "candidate_id": decision.get("candidate_id"),
                "correction_type": decision.get("correction_type"),
                "original_text": original,
                "corrected_text": corrected,
                "confidence": decision.get("confidence"),
                "current_original_count": original_count,
                "current_corrected_count": corrected_count,
                "baseline_original_count": baseline_original_count,
                "baseline_corrected_count": baseline_corrected_count,
                "improved_vs_baseline": bool(baseline.get("exists") and original_count < baseline_original_count),
                "summary_absorption_proven": context_proven,
                "absorption_method": "exact_corrected_text" if corrected_count > 0 and original_count == 0 else ("corrected_context_keywords" if context_proven else "not_proven"),
                "source_context_text": context.get("text", ""),
                "source_context_range": context.get("time_range", ""),
                "source_context_path": context.get("path", ""),
                "context_keywords": context_keywords,
                "summary_context_keyword_hits": context_hits,
                "sample_corrected_lines": _sample_corrected_lines(current_clean, corrected),
                "sample_residual_lines": _sample_residual_lines(current_clean, original, corrected),
            }
        )

    status = _status(summary_exists=bool(summary.get("exists")), accepted_count=len(accepted), evaluable_count=len(rows), residual_total=residual_total, corrected_hit_total=corrected_hit_total, context_proven_count=proven_rows)
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": status,
        "ok": status in {"passed", "no_accepted_decisions", "no_evaluable_replacements"},
        "summary_path": summary.get("path", ""),
        "summary_exists": bool(summary.get("exists")),
        "summary_is_final_llm": summary_is_final_llm,
        "baseline_summary_path": baseline.get("path", ""),
        "baseline_exists": bool(baseline.get("exists")),
        "accepted_decision_count": len(accepted),
        "evaluable_decision_count": len(rows),
        "summary_residual_original_total": residual_total,
        "summary_corrected_hit_total": corrected_hit_total,
        "summary_context_keyword_hit_total": context_keyword_hit_total,
        "summary_absorption_proven_count": proven_rows,
        "summary_absorption_rate": round(proven_rows / len(rows), 4) if rows else 0.0,
        "baseline_residual_original_total": baseline_residual_total,
        "baseline_corrected_hit_total": baseline_corrected_hit_total,
        "baseline_residual_delta": baseline_residual_total - residual_total if baseline.get("exists") else 0,
        "corrections": rows,
        "notes": _notes(status, bool(baseline.get("exists"))),
        "artifacts": {
            "json": str(root / "transcript-semantic-summary-impact-report.json"),
            "markdown": str(root / "transcript-semantic-summary-impact-report.md"),
        },
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "does_not_run_asr": True,
            "does_not_run_vision": True,
            "does_not_modify_transcript": True,
            "does_not_modify_summary": True,
        },
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="transcript_semantic_summary_impact_report", timeout_seconds=1.0):
            write_json(root / "transcript-semantic-summary-impact-report.json", result)
            (root / "transcript-semantic-summary-impact-report.md").write_text(_render_markdown(result), encoding="utf-8")
            write_json(root / "mcp-transcript-semantic-summary-impact-report.args.json", {"bundle_dir": str(root), "write": True})
            manifest["transcript_semantic_summary_impact_report_json"] = "transcript-semantic-summary-impact-report.json"
            manifest["transcript_semantic_summary_impact_report_markdown"] = "transcript-semantic-summary-impact-report.md"
            manifest["mcp_transcript_semantic_summary_impact_report_args"] = "mcp-transcript-semantic-summary-impact-report.args.json"
            manifest["transcript_semantic_summary_impact_summary"] = {
                "status": status,
                "accepted_decision_count": len(accepted),
                "summary_residual_original_total": residual_total,
                "summary_corrected_hit_total": corrected_hit_total,
                "summary_context_keyword_hit_total": context_keyword_hit_total,
                "summary_absorption_rate": result["summary_absorption_rate"],
                "updated_at": result["updated_at"],
            }
            write_json(root / "manifest.json", manifest)
    return result


def _is_final_llm_summary_text(text: str) -> bool:
    final_modes = (
        "codex_final",
        "codex_llm_rewrite_final",
        "codex_llm_rewrite_substitute",
        "codex_first_llm_substitute",
        "online_llm_section_rewrite",
    )
    pattern = r"(?m)^\s*生成方式\s*[：:]\s*\x60?(?:" + "|".join(re.escape(mode) for mode in final_modes) + r")\b"
    return bool(re.search(pattern, text))

def _status(*, summary_exists: bool, accepted_count: int, evaluable_count: int, residual_total: int, corrected_hit_total: int, context_proven_count: int) -> str:
    if not summary_exists:
        return "missing_summary"
    if accepted_count <= 0:
        return "no_accepted_decisions"
    if evaluable_count <= 0:
        return "no_evaluable_replacements"
    if residual_total > 0:
        return "needs_fix"
    if corrected_hit_total <= 0 and context_proven_count <= 0:
        return "not_proven"
    return "passed"



def _corrected_transcript_context(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    candidates: list[Any] = []
    for key in (
        "source_arbitrated_transcript_json",
        "corrected_transcript_json",
        "llm_corrected_transcript_json",
        "human_corrected_transcript_json",
    ):
        value = manifest.get(key)
        if value:
            candidates.append(value)
    candidates.extend(["source-arbitrated-transcript.json", "corrected-transcript.json", "llm-corrected-transcript.json"])
    for value in candidates:
        path = _bundle_path(root, value)
        if not path.exists() or not path.is_file():
            continue
        data = _read_optional_json(path)
        segments = [row for row in (data.get("segments") if isinstance(data, dict) else []) or [] if isinstance(row, dict)]
        if not segments:
            continue
        by_candidate: dict[str, list[dict[str, Any]]] = {}
        for segment in segments:
            for correction in segment.get("semantic_corrections") or []:
                if not isinstance(correction, dict):
                    continue
                candidate_id = str(correction.get("candidate_id") or "").strip()
                if candidate_id:
                    by_candidate.setdefault(candidate_id, []).append(segment)
        return {"exists": True, "path": str(path), "segments": segments, "by_candidate": by_candidate}
    return {"exists": False, "path": "", "segments": [], "by_candidate": {}}


def _decision_context(decision: dict[str, Any], corrected_context: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(decision.get("candidate_id") or "").strip()
    by_candidate = corrected_context.get("by_candidate") if isinstance(corrected_context.get("by_candidate"), dict) else {}
    segments = []
    if candidate_id and candidate_id in by_candidate:
        segments = [row for row in by_candidate.get(candidate_id) or [] if isinstance(row, dict)]
    if not segments:
        all_segments = [row for row in corrected_context.get("segments") or [] if isinstance(row, dict)]
        start = _float(decision.get("start"), -1.0)
        end = _float(decision.get("end"), start)
        if end >= start >= 0:
            segments = [row for row in all_segments if _overlaps(start, end, _float(row.get("start"), 0.0), _float(row.get("end"), 0.0))]
    if segments:
        text = " ".join(str(row.get("text") or row.get("corrected_text") or "").strip() for row in segments if str(row.get("text") or row.get("corrected_text") or "").strip())
        start = min((_float(row.get("start"), 0.0) for row in segments), default=0.0)
        end = max((_float(row.get("end"), start) for row in segments), default=start)
        return {"text": _clip_context(text), "start": start, "end": end, "time_range": f"{_fmt_time(start)} - {_fmt_time(end)}", "path": corrected_context.get("path", "")}
    corrected = str(decision.get("corrected_text") or "").strip()
    start = _float(decision.get("start"), 0.0)
    end = _float(decision.get("end"), start)
    return {"text": _clip_context(corrected), "start": start, "end": end, "time_range": f"{_fmt_time(start)} - {_fmt_time(end)}", "path": corrected_context.get("path", "")}


def _context_keywords(text: str, *, corrected_text: str = "") -> list[str]:
    source = _normalise_keyword_text(" ".join([str(corrected_text or ""), str(text or "")]))
    keywords: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|\d+(?:\.\d+)?\s*(?:k|K|w|W|万|亿|元|刀|%|％)?", source):
        keywords.append(token.strip())
    chunks = re.split(r"[，。！？；：、,.!?;:\s]+|(?:首先|其次|然后|最后|因为|所以|但是|如果|那么|接下来|另外|比如|先|再|并且|以及|和|与|并)", source)
    for chunk in chunks:
        value = _clean_keyword(chunk)
        if _useful_keyword(value):
            keywords.append(value)
    return _dedupe(keywords)[:12]


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if _count_text(text, keyword) > 0]


def _context_absorbed(keywords: list[str], hits: list[str]) -> bool:
    if not keywords:
        return False
    required = 1 if len(keywords) <= 2 else 2
    return len(hits) >= required


def _normalise_keyword_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    return value.replace("`", "")


def _clean_keyword(text: str) -> str:
    value = re.sub(r"\s+", "", str(text or "").strip(" ，,、。；;：:"))
    value = re.sub(r"^(第[一二三四五六七八九十0-9]+步|第一步|第二步|第三步|第四步|第五步)", "", value)
    value = re.sub(r"^(先|再|要|需要|可以|应该|必须|不要|不能|这个|那个|这里|就是|进行|通过)", "", value)
    return value.strip(" ，,、。；;：:")


def _useful_keyword(value: str) -> bool:
    if len(value) < 2 or len(value) > 16:
        return False
    if value in {"这个", "那个", "这里", "就是", "然后", "所以", "但是", "如果", "因为", "视频", "课程", "内容", "总结"}:
        return False
    if re.fullmatch(r"[0-9]+", value):
        return False
    return True



def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    total = int(seconds)
    ms = int(round((seconds - total) * 1000))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _clip_context(text: str, limit: int = 360) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.lower() if _asciiish(value) else value
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out

def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        return {}
    try:
        data = read_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _summary_document(root: Path, manifest: dict[str, Any], summary_path: str | Path | None) -> dict[str, Any]:
    if summary_path:
        path = Path(summary_path).expanduser().resolve()
    else:
        path = _bundle_path(root, manifest.get("smart_summary") or manifest.get("knowledge_note_smart_summary_markdown") or "exports/smart-summary.md")
    return _text_document(path)


def _baseline_document(baseline_summary_path: str | Path | None) -> dict[str, Any]:
    if not baseline_summary_path:
        return {"exists": False, "path": "", "text": ""}
    return _text_document(Path(baseline_summary_path).expanduser().resolve())


def _text_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "text": ""}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return {"exists": True, "path": str(path), "text": text}


def _bundle_path(root: Path, value: Any) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return root
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _countable_text(text: str) -> str:
    text = re.sub(r"```.*?```", "\n", text or "", flags=re.S)
    cleaned: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _looks_like_artifact_line(stripped):
            continue
        cleaned.append(re.sub(r"`([^`]+)`", r"\1", stripped))
    return "\n".join(cleaned)


def _looks_like_artifact_line(line: str) -> bool:
    lowered = line.lower()
    if "\\" in line and (":" in line[:4] or "d:" in lowered or "c:" in lowered):
        return True
    return any(token in lowered for token in ("exports/", "webui-bundle", ".json", ".md", ".srt", ".mp4")) and len(line) < 260


def _count_text(text: str, needle: str) -> int:
    if not text or not needle:
        return 0
    if _asciiish(needle):
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])", re.I)
        return len(pattern.findall(text))
    return text.count(needle)


def _count_corrected_text(text: str, corrected: str) -> int:
    if not text or not corrected:
        return 0
    if _asciiish(corrected):
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(corrected) + r"(?![A-Za-z0-9])")
        return len(pattern.findall(text))
    return text.count(corrected)


def _count_original_residual(text: str, original: str, corrected: str) -> int:
    if not text or not original:
        return 0
    if _case_only_correction(original, corrected):
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(original) + r"(?![A-Za-z0-9])")
        return len(pattern.findall(text))
    return _count_text(text, original)


def _case_only_correction(original: str, corrected: str) -> bool:
    return bool(
        original
        and corrected
        and _asciiish(original)
        and _asciiish(corrected)
        and original != corrected
        and original.lower() == corrected.lower()
    )


def _asciiish(text: str) -> bool:
    chars = [ch for ch in text if not ch.isspace()]
    return bool(chars) and all(ord(ch) < 128 for ch in chars)


def _sample_lines(text: str, needle: str, *, limit: int = 3) -> list[str]:
    if not text or not needle:
        return []
    rows: list[str] = []
    for line in text.splitlines():
        if _count_text(line, needle):
            rows.append(line.strip()[:320])
            if len(rows) >= limit:
                break
    return rows


def _sample_corrected_lines(text: str, corrected: str, *, limit: int = 3) -> list[str]:
    if not text or not corrected:
        return []
    rows: list[str] = []
    for line in text.splitlines():
        if _count_corrected_text(line, corrected):
            rows.append(line.strip()[:320])
            if len(rows) >= limit:
                break
    return rows


def _sample_residual_lines(text: str, original: str, corrected: str, *, limit: int = 3) -> list[str]:
    if not text or not original:
        return []
    rows: list[str] = []
    for line in text.splitlines():
        if _count_original_residual(line, original, corrected):
            rows.append(line.strip()[:320])
            if len(rows) >= limit:
                break
    return rows


def _notes(status: str, has_baseline: bool) -> list[str]:
    notes = [
        "This report checks whether accepted semantic corrections are visible in smart-summary.md.",
        "It is local-only and does not call an LLM, ASR, vision model, or downloader.",
    ]
    if not has_baseline:
        notes.append("No baseline summary was provided, so before/after delta is inferred from accepted original/corrected text counts in the current summary.")
    if status == "no_evaluable_replacements":
        notes.append("No final LLM summary exists yet; semantic corrections are preserved in the corrected transcript and summary input pack, so summary absorption is deferred.")
    if status == "not_proven":
        notes.append("No residual original text remains, but corrected text is also absent; the summary may be too abstract, so quality improvement is not proven by text evidence alone.")
    if status == "needs_fix":
        notes.append("Accepted original text still appears in smart-summary.md; regenerate or revise the summary from source-arbitrated transcript.")
    return notes


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 智能总结语义纠错影响报告",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Accepted decisions: `{result.get('accepted_decision_count')}`",
        f"- Evaluable decisions: `{result.get('evaluable_decision_count')}`",
        f"- Summary residual original total: `{result.get('summary_residual_original_total')}`",
        f"- Summary corrected hit total: `{result.get('summary_corrected_hit_total')}`",
        f"- Summary context keyword hits: `{result.get('summary_context_keyword_hit_total')}`",
        f"- Summary absorption rate: `{result.get('summary_absorption_rate')}`",
        f"- Summary: `{result.get('summary_path')}`",
        f"- Baseline: `{result.get('baseline_summary_path') or '(none)'}`",
        "",
        "## Corrections",
        "",
    ]
    for row in result.get("corrections") or []:
        lines.append(f"### `{row.get('candidate_id')}` `{row.get('original_text')}` -> `{row.get('corrected_text')}`")
        lines.append(f"- Type: `{row.get('correction_type')}` / confidence `{row.get('confidence')}`")
        lines.append(f"- Current: original `{row.get('current_original_count')}`, corrected `{row.get('current_corrected_count')}`, method `{row.get('absorption_method')}`")
        if row.get("summary_context_keyword_hits"):
            lines.append("- Context hits: " + ", ".join(str(value) for value in row.get("summary_context_keyword_hits") or []))
        if row.get("source_context_text"):
            lines.append(f"- Source context `{row.get('source_context_range')}`: {str(row.get('source_context_text') or '')[:220]}")
        if result.get("baseline_exists"):
            lines.append(f"- Baseline: original `{row.get('baseline_original_count')}`, corrected `{row.get('baseline_corrected_count')}`, improved `{row.get('improved_vs_baseline')}`")
        samples = row.get("sample_corrected_lines") or []
        residuals = row.get("sample_residual_lines") or []
        if samples:
            lines.append(f"- Corrected sample: {samples[0]}")
        if residuals:
            lines.append(f"- Residual sample: {residuals[0]}")
        lines.append("")
    lines.extend(["## Notes", ""])
    for note in result.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"
