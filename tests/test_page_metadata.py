from __future__ import annotations

import hashlib
import json
from pathlib import Path

from video_knowledge_pipeline.adaptive_asr_route import build_adaptive_asr_route
from video_knowledge_pipeline.cli import build_parser, run_mcp_call
from video_knowledge_pipeline.entity_lexicon import build_entity_lexicon
from video_knowledge_pipeline.page_metadata import import_page_metadata
from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
from video_knowledge_pipeline.transcript_semantic_correction import _metadata_evidence
from video_knowledge_pipeline.vdo_handoff import ingest_vdo_handoff


def _bundle(tmp_path: Path, *, title: str = "人工标题") -> Path:
    bundle = tmp_path / "webui-bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "title": title}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [{"index": 1, "start": 0, "end": 4, "transcript": "今天介绍客户管理方案。"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return bundle


def _handoff(tmp_path: Path) -> tuple[Path, Path]:
    subtitle = tmp_path / "lesson.zh.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n明亚APP\n", encoding="utf-8")
    description = tmp_path / "lesson.description"
    description.write_text("介绍明亚APP与客户经营。", encoding="utf-8")
    info = tmp_path / "lesson.info.json"
    info.write_text(
        json.dumps(
            {
                "title": "<b>明亚APP 客户经营课</b>",
                "description": "<p>介绍明亚APP、Excel方案和客户经营。</p>",
                "uploader": "课程中心",
                "tags": ["明亚APP", "Excel方案", "客户经营"],
                "chapters": [{"start_time": "00:01:30", "end_time": 120, "title": "方案制作"}],
                "webpage_url": "https://user:pass@example.com/watch?v=abc&token=secret#chapter",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    handoff = tmp_path / "vdo-handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "contract": "vdo_vkp_handoff_v1",
                "service": "video-download-orchestrator",
                "source": {
                    "source_url": "https://example.com/watch?v=abc&api_key=hidden",
                    "platform": "example",
                    "page_title": "探测标题",
                },
                "sidecars": {
                    "info_json_path": str(info),
                    "description_path": str(description),
                    "subtitle_paths": [str(subtitle)],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return handoff, subtitle


def test_import_page_metadata_normalizes_sidecars_and_preserves_manifest_title(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    handoff, subtitle = _handoff(tmp_path)

    result = import_page_metadata(bundle, handoff)

    assert result["status"] == "imported"
    metadata = json.loads((bundle / "source" / "page-metadata.json").read_text(encoding="utf-8"))
    assert metadata["title"] == "明亚APP 客户经营课"
    assert metadata["description"] == "介绍明亚APP、Excel方案和客户经营。"
    assert metadata["source_url"] == "https://example.com/watch?v=abc"
    assert "secret" not in json.dumps(metadata, ensure_ascii=False)
    assert metadata["chapters"][0]["start_seconds"] == 90.0
    assert metadata["subtitle_artifacts"][0]["sha256"] == hashlib.sha256(subtitle.read_bytes()).hexdigest()
    assert metadata["operator_boundary"]["cannot_override_transcript"] is True

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["title"] == "人工标题"
    assert manifest["page_metadata"]["title"] == "明亚APP 客户经营课"
    assert manifest["page_metadata_json"] == "source/page-metadata.json"
    assert manifest["page_metadata_summary"]["weak_context_only"] is True

    source_index = json.loads((bundle / "source-artifacts.json").read_text(encoding="utf-8"))
    keys = {row["key"] for row in source_index["artifacts"] if row["available"]}
    assert {"page_metadata_json", "page_metadata_markdown"} <= keys


def test_page_metadata_flows_to_hotwords_asr_summary_and_semantic_evidence(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    handoff, _ = _handoff(tmp_path)
    import_page_metadata(bundle, handoff)

    lexicon = build_entity_lexicon(bundle, phase="pre_asr", write=False)
    assert any("明亚" in term for term in lexicon["hotwords"])

    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"video")
    route = build_adaptive_asr_route(bundle, media, write=False)
    prompt = route["context"]["cloud_prompt"]
    assert "网页来源元数据（不可信弱上下文" in prompt
    assert "明亚APP 客户经营课" in prompt
    assert "Excel方案" in prompt

    pack = build_smart_summary_input_pack(bundle, write=False)
    assert pack["source_context"]["available"] is True
    assert pack["source_context"]["trust"] == "untrusted_weak_context"
    assert pack["source_context"]["artifact_sha256"]
    assert pack["source_context"]["chapters"][0]["title"] == "方案制作"

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    evidence = _metadata_evidence(bundle, manifest)
    assert evidence[0]["source_type"] == "page_metadata"
    assert "明亚APP" in evidence[0]["text"]


def test_page_metadata_cli_and_generic_mcp_entrypoints(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    handoff, _ = _handoff(tmp_path)
    args = build_parser().parse_args(["import-page-metadata", str(bundle), str(handoff), "--no-write"])
    assert args.command == "import-page-metadata"
    assert args.no_write is True

    mcp_args = tmp_path / "mcp-args.json"
    mcp_args.write_text(
        json.dumps({"bundle_dir": str(bundle), "metadata_json": str(handoff), "write": False}),
        encoding="utf-8",
    )
    result = run_mcp_call("import_page_metadata", mcp_args)
    assert result["status"] == "preview"
    assert not (bundle / "source" / "page-metadata.json").exists()


def test_vdo_ingest_installs_page_metadata_after_local_bundle_creation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"video")
    handoff, _ = _handoff(tmp_path)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload.update(
        {
            "ok": True,
            "status": "ready_for_ingest",
            "media_path": str(media),
            "title": "明亚APP 客户经营课",
            "ingestion": {"workspace": str(tmp_path / "run"), "title": "明亚APP 客户经营课"},
        }
    )
    handoff.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def prepare_runner(*args, **kwargs):
        return {"initial_bundle": {"status": "ok", "bundle_dir": str(bundle)}}

    result = ingest_vdo_handoff(handoff_path=handoff, execute=True, prepare_runner=prepare_runner)

    assert result["status"] == "ingested"
    assert result["page_metadata_import"]["status"] == "imported"
    assert (bundle / "source" / "page-metadata.json").is_file()
