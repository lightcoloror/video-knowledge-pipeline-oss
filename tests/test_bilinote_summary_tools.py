from pathlib import Path

from video_knowledge_pipeline.bilinote_mind_map_prompt_pack import build_bundle_mind_map_prompt_pack
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.bilinote_summary_tools import (
    apply_transcript_corrections,
    build_transcript_correction_messages,
    correction_stats,
    parse_transcript_correction_json,
    split_transcript_for_mind_map,
    build_mind_map_prompt_pack,
)


def test_split_transcript_for_mind_map_preserves_order_and_chunks_by_lines():
    transcript = "\n".join(f"[{idx:02d}:00] 第{idx}段内容" for idx in range(8))

    chunks = split_transcript_for_mind_map(transcript, max_chars=40)

    assert len(chunks) > 1
    assert chunks[0].startswith("[00:00]")
    assert chunks[-1].endswith("第7段内容")


def test_build_transcript_correction_messages_uses_fixed_index_contract():
    messages = build_transcript_correction_messages(
        title="浏览器自动化",
        segments=[{"index": 3, "timestamp": "00:00:10.000", "text": "brother mc p"}],
    )

    assert messages[0]["role"] == "system"
    assert "只修正" in messages[0]["content"]
    assert "保持 segments 数组长度和 index 不变" in messages[1]["content"]
    assert '"index": 3' in messages[1]["content"]


def test_parse_and_apply_transcript_correction_json_from_fenced_output():
    original = [
        {"index": 0, "start": 0, "end": 1, "timestamp": "00:00:00.000", "text": "brother mc p"},
        {"index": 1, "start": 1, "end": 2, "timestamp": "00:00:01.000", "text": "不用改"},
    ]
    payload = parse_transcript_correction_json('```json\n{"segments":[{"index":0,"text":"Browser MCP"}]}\n```')
    corrected = apply_transcript_corrections(original, payload)

    assert corrected[0]["text"] == "Browser MCP"
    assert corrected[0]["changed"] is True
    assert corrected[1]["text"] == "不用改"
    assert correction_stats(original, corrected)["corrected_segments"] == 1


def test_build_mind_map_prompt_pack_uses_json_node_contract():
    transcript = "[00:00:01.000] 第一部分讲客户特点\n[00:03:00.000] 第二部分讲成交动作"

    pack = build_mind_map_prompt_pack(title="销售课", transcript=transcript, max_chars=80)

    assert pack["schema"] == "video_knowledge_pipeline.bilinote_mind_map_prompt_pack.v1"
    assert pack["chunk_count"] >= 1
    content = pack["prompts"][0]["messages"][1]["content"]
    assert "思维导图节点 JSON" in content
    assert "uncertain_terms" in content
    assert "不要只总结开头" in content
    assert pack["operator_boundary"]["no_llm_call"] is True

def test_bundle_mind_map_prompt_pack_writes_artifacts_and_run(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    transcript_path = bundle / "normalized-transcript.json"
    write_json(
        bundle / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "title": "销售训练课",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(
        transcript_path,
        {
            "segments": [
                {"index": 0, "start": 0, "end": 3, "text": "第一部分讲客户特点。"},
                {"index": 1, "start": 3, "end": 8, "text": "第二部分讲成交动作和获取信任。"},
            ]
        },
    )

    result = build_bundle_mind_map_prompt_pack(bundle, max_chars=80)

    assert result["schema"] == "video_knowledge_pipeline.bilinote_mind_map_bundle_prompt_pack.v1"
    assert result["transcript_segment_count"] == 2
    assert result["chunk_count"] >= 1
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()
    manifest = read_json(bundle / "manifest.json")
    assert manifest["bilinote_mind_map_prompt_pack_markdown"] == "exports/bilinote-mind-map-prompt-pack.md"
    registry = read_json(bundle / "run-artifact-registry.json")
    run_types = {run["run_type"]: run["status"] for run in registry["runs"]}
    assert run_types["bilinote_mind_map_prompt_pack"] == "completed"
    run_md = bundle / "runs" / "bilinote-mind-map-prompt-pack" / "run.md"
    assert run_md.exists()
    assert "BiliNote mind-map prompt pack" in run_md.read_text(encoding="utf-8")
