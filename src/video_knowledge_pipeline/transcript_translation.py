"""Evidence-preserving transcript translation sidecar.

Change rationale
----------------
Intent: produce natural Mandarin subtitles from Cantonese ASR while retaining
the exact segment/timestamp/speaker lineage.
Decision: reuse VKP's transcript parser, SRT renderer and unified local model
runtime.  Translation is a derived sidecar and never replaces the canonical
or raw transcript.
Reason: forcing Mandarin ASR loses Cantonese content, while translating after
Cantonese recognition separates recognition evidence from language rendering.
Evidence: the 2026-08-10 language gate showed SenseVoice ``yue`` preserved more
spoken content than forced ``zh`` on the same 45-second clip.
Effective scope: local ``text_llm`` execution and importable translation JSON;
no remote route, provider SDK, audio upload, or source transcript mutation.
Rollback: remove this module and CLI registration; existing transcript files
and bundle manifests remain valid because the new fields are additive.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .asr_adapter import render_srt
from .file_hash import sha256_file
from .model_runtime_client import model_runtime_request
from .models import TranscriptCue, now_iso
from .storage import read_json, write_json
from .text_llm_gateway import call_openai_compatible_text
from .transcript import parse_transcript
from .transcript_speakers import cue_speaker, cue_speaker_role


SCHEMA = "video_knowledge_pipeline.translated_transcript.v1"
PLAN_SCHEMA = "video_knowledge_pipeline.transcript_translation_plan.v1"
RuntimeCall = Callable[..., dict[str, Any]]


def translate_transcript_to_mandarin(
    bundle_dir: str | Path,
    *,
    source_path: str | Path | None = None,
    input_json: str | Path | None = None,
    batch_size: int = 24,
    route_id: str = "",
    route_revision: str = "",
    direct_lmstudio: bool = False,
    execute: bool = False,
    write: bool = True,
    runtime_call: RuntimeCall = model_runtime_request,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    source = _resolve_source(root, source_path)
    cues = parse_transcript(source)
    rows = [_source_row(cue, index) for index, cue in enumerate(cues)]
    size = max(1, min(50, int(batch_size or 24)))
    batches = [rows[index : index + size] for index in range(0, len(rows), size)]
    plan = {
        "schema": PLAN_SCHEMA,
        "bundle_dir": str(root),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_language": "yue",
        "target_language": "zh-CN",
        "segment_count": len(rows),
        "batch_size": size,
        "batch_count": len(batches),
        "execution_location": "local",
        "route_id": str(route_id or ""),
        "route_revision": str(route_revision or ""),
        "operator_boundary": {
            "derived_sidecar_only": True,
            "does_not_modify_raw_or_canonical_transcript": True,
            "remote_execution_allowed": False,
        },
        "batches": [
            {
                "batch_index": index,
                "segment_indexes": [row["index"] for row in batch],
                "segment_ids": [row["segment_id"] for row in batch],
            }
            for index, batch in enumerate(batches)
        ],
        "created_at": now_iso(),
    }
    plan_path = root / "mandarin-subtitle-translation-plan.json"
    if write:
        write_json(plan_path, plan)

    # Intent: keep reviewed import files inside the same evidence chain as
    # model-produced translations. Decision: require both the source SHA-256
    # and every segment_id before accepting any imported row. Reason: index-only
    # imports can silently attach text to a revised transcript. Evidence: the
    # synthetic lineage-drift fixture is rejected. Scope: input_json only;
    # locally generated checkpoints already enforce the same binding.
    imported = (
        _load_imported_translations(
            input_json,
            source_sha256=plan["source_sha256"],
            expected_rows=rows,
        )
        if input_json
        else None
    )
    checkpoint_path = root / "mandarin-subtitle-translation-checkpoint.json"
    translated: dict[int, str] = _load_checkpoint(
        checkpoint_path,
        source_sha256=plan["source_sha256"],
        expected_rows=rows,
    )
    translated.update(imported or {})
    batch_results: list[dict[str, Any]] = []
    if execute:
        for batch_index, batch in enumerate(batches):
            missing = [row for row in batch if row["index"] not in translated]
            if not missing:
                batch_results.append(
                    {
                        "batch_index": batch_index,
                        "status": "reused",
                        "segment_count": len(batch),
                    }
                )
                continue
            if direct_lmstudio:
                # Intent: execute the reviewed local-production builtin route
                # without misrepresenting it as a LiteLLM Proxy deployment.
                # Decision: reuse VKP's existing OpenAI-compatible text adapter
                # against the fixed loopback LM Studio endpoint and model id.
                # Reason: local-production-v1 deliberately labels this route
                # builtin/legacy, so the Proxy renderer correctly skips it.
                # Evidence: gateway render returned explicit_legacy_route for
                # local-lmstudio-qwen3-5-9b during the 2026-08-10 interview run.
                # Effective scope: only the operator-selected
                # --direct-lmstudio branch; no fallback or arbitrary URL input.
                result = call_openai_compatible_text(
                    provider_config={
                        "provider": "local_vlm",
                        "base_url": "http://127.0.0.1:1234/v1",
                        "model": "qwen/qwen3.5-9b",
                        "timeout_seconds": 300,
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    },
                    messages=_translation_messages(missing),
                    temperature=0,
                    # Local execution is not cost-metered by output tokens.
                    # Keep a production-sized ceiling so long Cantonese cues
                    # are not truncated after thinking has been disabled.
                    max_tokens=4096,
                )
                result = {
                    **result,
                    "status": "completed" if result.get("ok") else "local_model_failed",
                    "deployment": {
                        "provider": "local_vlm",
                        "model": "qwen/qwen3.5-9b",
                        "base_url": "http://127.0.0.1:1234/v1",
                        "thinking_mode": "disabled",
                    },
                }
            else:
                result = runtime_call(
                    "text_llm",
                    execution_location="local",
                    route_id=route_id,
                    route_revision=route_revision,
                    text=json.dumps(missing, ensure_ascii=False, separators=(",", ":")),
                    messages=_translation_messages(missing),
                    temperature=0,
                    execute=True,
                )
            if not bool(result.get("ok")):
                batch_results.append(
                    {
                        "batch_index": batch_index,
                        "status": str(result.get("status") or "failed"),
                        "error": str(
                            result.get("error") or "local model translation failed"
                        ),
                    }
                )
                continue
            try:
                parsed = _parse_translation_content(result.get("content"), missing)
                translated.update(parsed)
                if write:
                    # Intent: make long local translation resumable per batch.
                    # Decision: atomically persist only validated index/text
                    # pairs bound to the exact source transcript hash.
                    # Reason: a 20-batch interview must not restart after a GPU,
                    # app, or machine interruption.
                    # Evidence: VKP's chunked ASR already uses the same
                    # checkpoint-and-resume pattern for long media stability.
                    # Effective scope: derived translation state only; a
                    # complete SRT is still withheld until every index exists.
                    write_json(
                        checkpoint_path,
                        {
                            "schema": "video_knowledge_pipeline.transcript_translation_checkpoint.v1",
                            "source_sha256": plan["source_sha256"],
                            "translations": [
                                {
                                    "index": index,
                                    "segment_id": rows[index]["segment_id"],
                                    "text": text,
                                }
                                for index, text in sorted(translated.items())
                            ],
                            "updated_at": now_iso(),
                        },
                    )
                batch_results.append(
                    {
                        "batch_index": batch_index,
                        "status": "completed",
                        "segment_count": len(parsed),
                        "latency_ms": result.get("latency_ms"),
                        "deployment": result.get("deployment") or {},
                    }
                )
            except ValueError as exc:
                batch_results.append(
                    {
                        "batch_index": batch_index,
                        "status": "invalid_output",
                        "error": str(exc),
                    }
                )

    complete = len(rows) > 0 and len(translated) == len(rows)
    # Intent: expose attempted-but-invalid execution to callers. Decision:
    # return degraded whenever an executed batch failed or was rejected, even
    # when no translation row survived. Reason: planned would hide a real
    # attempt as if it never ran. Evidence: the wrong-segment model fixture.
    # Scope: incomplete executed runs only; dry plans remain planned.
    status = (
        "completed"
        if complete
        else ("degraded" if translated or (execute and batch_results) else "planned")
    )
    output_json = root / "mandarin-translated-transcript.json"
    output_srt = root / "mandarin-translated-subtitles.srt"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_language": "yue",
        "target_language": "zh-CN",
        "segment_count": len(rows),
        "translated_segment_count": len(translated),
        "missing_segment_indexes": [
            row["index"] for row in rows if row["index"] not in translated
        ],
        "segments": _translated_segments(cues, translated),
        "batch_results": batch_results,
        "operator_boundary": plan["operator_boundary"],
        "created_at": now_iso(),
    }
    if write:
        write_json(output_json, payload)
        if complete:
            translated_cues = _translated_cues(cues, translated)
            output_srt.write_text(render_srt(translated_cues), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if not isinstance(manifest, dict):
                manifest = {}
            manifest["mandarin_translated_transcript_json"] = output_json.name
            manifest["mandarin_translated_subtitles_srt"] = output_srt.name
            write_json(manifest_path, manifest)
    return {
        "schema": SCHEMA,
        "ok": complete,
        "status": status,
        "source_path": str(source),
        "source_sha256": plan["source_sha256"],
        "segment_count": len(rows),
        "translated_segment_count": len(translated),
        "missing_segment_count": len(rows) - len(translated),
        "batch_results": batch_results,
        "artifacts": {
            "plan_json": str(plan_path),
            "translated_json": str(output_json),
            "translated_srt": str(output_srt) if complete else "",
            "checkpoint_json": str(checkpoint_path),
        },
        "operator_boundary": plan["operator_boundary"],
    }


def _resolve_source(root: Path, source_path: str | Path | None) -> Path:
    if source_path:
        candidate = Path(source_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"transcript source not found: {candidate}")
        return candidate
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    for key in (
        "corrected_transcript_json",
        "readable_transcript_json",
        "normalized_transcript_json",
        "transcript_json",
    ):
        value = (
            str((manifest or {}).get(key) or "").strip()
            if isinstance(manifest, dict)
            else ""
        )
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.is_file():
                return candidate.resolve()
    for name in (
        "corrected-transcript.json",
        "readable-transcript.json",
        "normalized-transcript.json",
    ):
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"no transcript source found: {root}")


def _source_row(cue: TranscriptCue, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "segment_id": str(cue.segment_id or f"segment-{index + 1:06d}"),
        "text": str(cue.text or ""),
    }


def _translation_messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    instruction = (
        "把粤语口语逐段翻译成自然、忠实的简体中文普通话字幕。"
        "只转换表达方式，不核查或改写事实，不补充原文没有的信息。"
        "专名、数字和不确定词按听到的原文保留；不要合并、拆分或重排段落。"
        '只输出 JSON 对象：{"translations":[{"index":0,"segment_id":"...","text":"..."}]}。'
        "index 和 segment_id 必须逐项原样复制。"
    )
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _parse_translation_content(
    content: Any, expected: list[dict[str, Any]]
) -> dict[int, str]:
    if isinstance(content, dict):
        data = content
    else:
        text = str(content or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("translation response is not a JSON object")
        data = json.loads(text[start : end + 1])
    items = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("translation response must contain translations[]")
    expected_by_index = {int(row["index"]): str(row["segment_id"]) for row in expected}
    parsed: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("translation item must be an object")
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("translation index must be an integer") from exc
        segment_id = str(item.get("segment_id") or "")
        text = str(item.get("text") or "").strip()
        if index not in expected_by_index or expected_by_index[index] != segment_id:
            raise ValueError(f"translation lineage mismatch at index {index}")
        if not text:
            raise ValueError(f"translation text is empty at index {index}")
        if index in parsed:
            raise ValueError(f"duplicate translation index {index}")
        parsed[index] = text
    if set(parsed) != set(expected_by_index):
        raise ValueError(
            "translation response does not cover the exact requested indexes"
        )
    return parsed


def _load_imported_translations(
    path_value: str | Path,
    *,
    source_sha256: str,
    expected_rows: list[dict[str, Any]],
) -> dict[int, str]:
    path = Path(path_value).expanduser().resolve()
    data = read_json(path)
    if (
        not isinstance(data, dict)
        or str(data.get("source_sha256") or "") != source_sha256
    ):
        raise ValueError(
            "imported translation source hash does not match the selected transcript"
        )
    items = data.get("translations")
    if not isinstance(items, list):
        raise ValueError("imported translation JSON must contain translations[]")
    expected = {int(row["index"]): str(row["segment_id"]) for row in expected_rows}
    result: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("imported translation item must be an object")
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("imported translation index must be an integer") from exc
        segment_id = str(item.get("segment_id") or "")
        text = str(item.get("text") or "").strip()
        if index not in expected or segment_id != expected[index]:
            raise ValueError(f"imported translation lineage mismatch at index {index}")
        if not text:
            raise ValueError(f"imported translation text is empty at index {index}")
        if index in result:
            raise ValueError(f"duplicate imported translation index {index}")
        result[index] = text
    return result


def _load_checkpoint(
    path: Path,
    *,
    source_sha256: str,
    expected_rows: list[dict[str, Any]],
) -> dict[int, str]:
    if not path.is_file():
        return {}
    data = read_json(path)
    if (
        not isinstance(data, dict)
        or str(data.get("source_sha256") or "") != source_sha256
    ):
        return {}
    items = data.get("translations")
    if not isinstance(items, list):
        return {}
    expected = {int(row["index"]): str(row["segment_id"]) for row in expected_rows}
    result: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            return {}
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            return {}
        segment_id = str(item.get("segment_id") or "")
        text = str(item.get("text") or "").strip()
        if (
            index not in expected
            or segment_id != expected[index]
            or not text
            or index in result
        ):
            return {}
        result[index] = text
    return result


def _translated_cues(
    cues: list[TranscriptCue], translated: dict[int, str]
) -> list[TranscriptCue]:
    return [
        TranscriptCue(
            start=cue.start,
            end=cue.end,
            text=translated[index],
            segment_id=cue.segment_id,
            source_segment_ids=list(cue.source_segment_ids),
            transformations=list(cue.transformations)
            + [{"type": "yue_to_mandarin_translation", "boundary_changed": False}],
            speaker=cue_speaker(cue),
            speaker_role=cue_speaker_role(cue),
            metadata={
                **dict(cue.metadata),
                "source_text": cue.text,
                "derived_translation": True,
            },
        )
        for index, cue in enumerate(cues)
    ]


def _translated_segments(
    cues: list[TranscriptCue], translated: dict[int, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, cue in enumerate(cues):
        if index not in translated:
            continue
        rows.append(
            {
                "index": index,
                "segment_id": cue.segment_id,
                "source_segment_ids": list(cue.source_segment_ids),
                "start": cue.start,
                "end": cue.end,
                "speaker": cue_speaker(cue),
                "speaker_role": cue_speaker_role(cue),
                "source_text": cue.text,
                "text": translated[index],
            }
        )
    return rows
