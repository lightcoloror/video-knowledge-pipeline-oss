from __future__ import annotations

import json
from typing import Any

from .models import TranscriptCue
from .text_llm_gateway import extract_json_document
from .transcript import format_timestamp

CORRECTION_SYSTEM_PROMPT = (
    "你是一个中文视频转录校对助手。只修正语音转写中的错别字、同音字、明显断句和专有名词错误；"
    "不要总结、不要扩写、不要改变原意。严格输出 JSON。"
)


def transcript_segments_to_text(segments: list[TranscriptCue]) -> str:
    return "\n".join(f"[{format_timestamp(segment.start)}] {segment.text}" for segment in segments if segment.text).rstrip()


# Adapted from PrideWood/bilinote server/summarizer.ts splitTranscriptForMindMap.
def split_transcript_for_mind_map(transcript: str, *, max_chars: int = 5000) -> list[str]:
    limit = max(1, int(max_chars or 5000))
    lines = [line for line in str(transcript or "").splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for line in lines:
        if current and current_length + len(line) + 1 > limit:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
        current.append(line)
        current_length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# Adapted from PrideWood/bilinote server/summarizer.ts correctTranscriptSegments.
def build_transcript_correction_messages(*, title: str, segments: list[dict[str, Any]]) -> list[dict[str, str]]:
    payload = [
        {
            "index": int(segment.get("index") or position),
            "timestamp": str(segment.get("timestamp") or ""),
            "text": str(segment.get("text") or ""),
        }
        for position, segment in enumerate(segments)
        if str(segment.get("text") or "").strip()
    ]
    return [
        {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"视频文件名：{title}",
                    "下面是带时间戳的 transcript 片段。请结合上下文校正 text 字段。",
                    "必须保持 segments 数组长度和 index 不变，只返回 {\"segments\":[{\"index\":number,\"text\":string}]}。",
                    json.dumps({"segments": payload}, ensure_ascii=False),
                ]
            ),
        },
    ]


def parse_transcript_correction_json(raw_text: str) -> dict[str, Any]:
    parsed = extract_json_document(raw_text, require_object=True)
    if not isinstance(parsed, dict):
        raise ValueError("transcript correction response must be a JSON object")
    segments = parsed.get("segments")
    if not isinstance(segments, list):
        raise ValueError("transcript correction response must contain segments array")
    return {"segments": [row for row in segments if isinstance(row, dict)]}


def apply_transcript_corrections(original_segments: list[dict[str, Any]], correction_payload: dict[str, Any]) -> list[dict[str, Any]]:
    corrections: dict[int, str] = {}
    for row in correction_payload.get("segments") or []:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index"))
        except Exception:
            continue
        text = str(row.get("text") or "").strip()
        if text:
            corrections[index] = text
    corrected: list[dict[str, Any]] = []
    for position, segment in enumerate(original_segments):
        index = int(segment.get("index") or position)
        raw = str(segment.get("text") or "").strip()
        new_text = corrections.get(index, raw)
        row = {**segment, "raw_text": raw, "text": new_text, "corrected_text": new_text}
        row["changed"] = bool(raw and new_text and raw != new_text)
        if row["changed"]:
            row["correction_source"] = "bilinote_style_llm_correction"
        corrected.append(row)
    return corrected


def correction_stats(original_segments: list[dict[str, Any]], corrected_segments: list[dict[str, Any]]) -> dict[str, Any]:
    changed = sum(1 for row in corrected_segments if row.get("changed"))
    return {
        "segments": len(original_segments),
        "corrected_segments": changed,
        "unchanged_segments": max(0, len(original_segments) - changed),
        "indexes_preserved": [int(row.get("index") or idx) for idx, row in enumerate(corrected_segments)]
        == [int(row.get("index") or idx) for idx, row in enumerate(original_segments)],
    }

MIND_MAP_SYSTEM_PROMPT = (
    "你是一个中文视频学习笔记结构化助手。请基于完整转录生成可用于思维导图的层级 JSON；"
    "不要编造视频中没有的信息，重要节点必须保留时间戳。"
)


def build_mind_map_prompt_messages(
    *,
    title: str,
    transcript_chunk: str,
    chunk_index: int = 1,
    chunk_count: int = 1,
    language: str = "zh-CN",
) -> list[dict[str, str]]:
    schema = {
        "title": "string",
        "language": language,
        "chunk_index": "number",
        "nodes": [
            {
                "title": "string",
                "time_range": "HH:MM:SS.mmm-HH:MM:SS.mmm",
                "summary": "string",
                "children": [
                    {
                        "title": "string",
                        "time_range": "HH:MM:SS.mmm-HH:MM:SS.mmm",
                        "summary": "string",
                        "evidence_quote": "string",
                    }
                ],
            }
        ],
        "uncertain_terms": ["string"],
    }
    return [
        {"role": "system", "content": MIND_MAP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"视频标题：{title}",
                    f"分块：{chunk_index}/{chunk_count}",
                    "请把下面的转录改写成思维导图节点 JSON。",
                    "要求：",
                    "1. 覆盖本分块里的主要知识点，不要只总结开头。",
                    "2. 每个一级节点尽量对应一个时间范围。",
                    "3. 工具名、人名、品牌名、课程术语不确定时放入 uncertain_terms。",
                    "4. 严格只返回 JSON，不要 Markdown。",
                    "目标 JSON schema 示例：",
                    json.dumps(schema, ensure_ascii=False),
                    "转录：",
                    transcript_chunk,
                ]
            ),
        },
    ]


def build_mind_map_prompt_pack(*, title: str, transcript: str, max_chars: int = 5000) -> dict[str, Any]:
    chunks = split_transcript_for_mind_map(transcript, max_chars=max_chars)
    return {
        "schema": "video_knowledge_pipeline.bilinote_mind_map_prompt_pack.v1",
        "title": title,
        "source_reuse": "PrideWood/bilinote mind-map transcript chunking and prompt structure",
        "chunk_count": len(chunks),
        "prompts": [
            {
                "chunk_index": index,
                "chunk_text": chunk,
                "messages": build_mind_map_prompt_messages(
                    title=title,
                    transcript_chunk=chunk,
                    chunk_index=index,
                    chunk_count=len(chunks),
                ),
            }
            for index, chunk in enumerate(chunks, start=1)
        ],
        "operator_boundary": {
            "local_prompt_only": True,
            "no_llm_call": True,
            "can_be_sent_to_codex_or_configured_text_llm_after_review": True,
        },
    }
