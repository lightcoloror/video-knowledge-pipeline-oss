from __future__ import annotations

from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .bilinote_summary_tools import build_mind_map_prompt_pack, transcript_segments_to_text
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .transcript_correction_pack import _load_best_transcript

SCHEMA = "video_knowledge_pipeline.bilinote_mind_map_bundle_prompt_pack.v1"


def build_bundle_mind_map_prompt_pack(
    bundle_dir: str | Path,
    *,
    title: str = "",
    max_chars: int = 5000,
    write: bool = True,
) -> dict[str, Any]:
    """Build a BiliNote-style mind-map prompt pack from the best bundle transcript.

    This is a local prompt-pack producer. It does not call an LLM and does not
    claim that a mind map has been generated.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    cues = _load_best_transcript(root, manifest)
    note_title = title or str(manifest.get("title") or root.name)
    transcript_text = transcript_segments_to_text(cues)
    prompt_pack = build_mind_map_prompt_pack(title=note_title, transcript=transcript_text, max_chars=max_chars)
    exports = root / "exports"
    json_path = exports / "bilinote-mind-map-prompt-pack.json"
    md_path = exports / "bilinote-mind-map-prompt-pack.md"
    mcp_args_path = root / "mcp-bilinote-mind-map-prompt-pack.args.json"
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": note_title,
        "created_at": now_iso(),
        "transcript_segment_count": len(cues),
        "chunk_count": int(prompt_pack.get("chunk_count") or 0),
        "max_chars": int(max_chars or 0),
        "source_reuse": "PrideWood/BiliNote mind-map transcript chunking and prompt structure",
        "prompts": prompt_pack.get("prompts") or [],
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
            "mcp_args": str(mcp_args_path),
        },
        "operator_boundary": {
            "local_prompt_only": True,
            "no_llm_call": True,
            "not_a_generated_mind_map": True,
            "can_be_sent_to_codex_or_configured_text_llm_after_review": True,
        },
        "write": bool(write),
    }
    if write:
        exports.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        md_path.write_text(_render_markdown(result), encoding="utf-8")
        write_json(mcp_args_path, {"bundle_dir": str(root), "title": note_title, "max_chars": max_chars, "write": True})
        manifest["bilinote_mind_map_prompt_pack_json"] = "exports/bilinote-mind-map-prompt-pack.json"
        manifest["bilinote_mind_map_prompt_pack_markdown"] = "exports/bilinote-mind-map-prompt-pack.md"
        manifest["mcp_bilinote_mind_map_prompt_pack_args"] = "mcp-bilinote-mind-map-prompt-pack.args.json"
        write_json(manifest_path, manifest)
    result["run_registry"] = _register_run(root, result, write=write)
    return result


def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    transcript_count = int(result.get("transcript_segment_count") or 0)
    chunk_count = int(result.get("chunk_count") or 0)
    failed_items: list[dict[str, Any]] = []
    if transcript_count <= 0 or chunk_count <= 0:
        failed_items.append({"id": "transcript", "reason": "transcript_missing", "detail": "No transcript chunks were available for BiliNote mind-map prompt generation."})
    if not write:
        status = "needs_execution"
    elif failed_items:
        status = "needs_input"
    else:
        status = "completed"
    return register_bundle_run(
        root,
        run_type="bilinote_mind_map_prompt_pack",
        run_id="bilinote-mind-map-prompt-pack",
        status=status,
        title="BiliNote mind-map prompt pack",
        summary=f"Prepared {chunk_count} BiliNote-style mind-map prompt chunk(s) from {transcript_count} transcript segment(s).",
        inputs={"bundle_dir": str(root), "transcript_source": "best available transcript sidecar"},
        parameters={"write": bool(write), "max_chars": int(result.get("max_chars") or 0), "chunk_count": chunk_count, "transcript_segment_count": transcript_count},
        artifacts=[
            {"key": "prompt_pack_json", "path": root / "exports" / "bilinote-mind-map-prompt-pack.json"},
            {"key": "prompt_pack_markdown", "path": root / "exports" / "bilinote-mind-map-prompt-pack.md"},
            {"key": "mcp_args", "path": root / "mcp-bilinote-mind-map-prompt-pack.args.json"},
        ],
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 bilinote-mind-map-prompt-pack --bundle-dir {_ps_quote(str(root))}",
        next_actions=_run_next_actions(status),
        operator_boundary={
            "local_prompt_only": True,
            "no_llm_call": True,
            "not_a_generated_mind_map": True,
            "human_review_required_before_model_use": True,
            "source_reuse": "PrideWood/BiliNote mind-map prompt chunking workflow.",
        },
        write=write,
    )


def _run_next_actions(status: str) -> list[str]:
    if status == "needs_input":
        return ["Add or generate a transcript sidecar, then rerun bilinote-mind-map-prompt-pack for the bundle."]
    if status == "needs_execution":
        return ["Rerun bilinote-mind-map-prompt-pack with write=true to persist prompt pack artifacts."]
    return ["Review exports/bilinote-mind-map-prompt-pack.md before sending prompt chunks to Codex or a configured text LLM."]


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# BiliNote Mind Map Prompt Pack: {result.get('title') or ''}",
        "",
        f"- Created: `{result.get('created_at') or ''}`",
        f"- Transcript segments: `{result.get('transcript_segment_count') or 0}`",
        f"- Prompt chunks: `{result.get('chunk_count') or 0}`",
        f"- Max chars per chunk: `{result.get('max_chars') or 0}`",
        "- LLM called: `false`",
        "- Generated mind map: `false`",
        "",
        "> 这个文件只是 BiliNote-style 思维导图 prompt 包。发送给 Codex 或在线文本模型前应先人工确认 transcript 和术语。",
        "",
    ]
    prompts = result.get("prompts") if isinstance(result.get("prompts"), list) else []
    if not prompts:
        lines.append("（没有可用 prompt chunk。）")
        return "\n".join(lines).rstrip() + "\n"
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        chunk_index = prompt.get("chunk_index") or ""
        chunk_text = str(prompt.get("chunk_text") or "")
        messages = prompt.get("messages") if isinstance(prompt.get("messages"), list) else []
        user_message = ""
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                user_message = str(message.get("content") or "")
                break
        lines.extend(
            [
                f"## Chunk {chunk_index}",
                "",
                "### Transcript Preview",
                "",
                _clip(chunk_text, 900),
                "",
                "### User Prompt Preview",
                "",
                _clip(user_message, 1200),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _clip(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."
