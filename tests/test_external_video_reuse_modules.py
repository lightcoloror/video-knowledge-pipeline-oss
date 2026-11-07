from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import audit_bundle_mcp_args, main, run_mcp_call
from video_knowledge_pipeline.external_capability_pack import build_external_capability_pack
from video_knowledge_pipeline.video_rag_pack import build_video_rag_pack
from video_knowledge_pipeline.video_rag_search import search_video_rag
from video_knowledge_pipeline.video_rag_http import video_rag_http_response, video_rag_service_plan
from video_knowledge_pipeline.long_video_memory_pack import build_long_video_memory_pack
from video_knowledge_pipeline.run_artifact_registry import build_run_artifact_registry
from video_knowledge_pipeline.local_vlm_server_adapter import local_vlm_serving_smoke
from video_knowledge_pipeline.video_moment_index import build_video_moment_index


def _write_bundle(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "frames").mkdir()
    (root / "frames" / "001.jpg").write_bytes(b"fake")
    exports = root / "exports"
    exports.mkdir()
    for name in (
        "knowledge-note.md",
        "smart-summary.md",
        "smart-summary-codex-prompt.md",
        "full-transcript.md",
        "extraction-audit.md",
        "key-segments.md",
        "short-video-script-drafts.md",
        "highlight-post-drafts.md",
        "content-material-card.md",
    ):
        (exports / name).write_text(f"# {name}\n", encoding="utf-8")
    term_correction = {
        "status": "ready",
        "term_validation_status": "ready_for_import",
        "accepted_validation_decisions": 2,
        "rejected_validation_decisions": 1,
        "accepted_term_count": 2,
        "source_arbitrated_transcript_exists": True,
        "final_export_alias_total": 0,
    }
    material_card = {
        "material_id": "demo-course",
        "source_path": str(root),
        "source_type": "video",
        "source_fact_status": "ai_extracted_needs_review",
        "evidence_tier": "derived_from_bundle",
        "privacy_level": "local_private",
        "desensitized": False,
        "compliance_risk": ["needs_review"],
        "fact_check_status": "needs_review",
        "target_layer": ["content_assets"],
        "publish_surface": ["draft_only_after_review"],
        "content_stage": "evidence",
        "cta_type": "none",
        "crm_followup_needed": False,
        "owner_thread": "video-knowledge-pipeline",
        "next_action": "human_review_then_route_to_content_assets",
        "blocked_reason": "",
        "review_required": True,
        "publication_allowed": False,
        "allowed_as_inspiration": True,
        "allowed_as_fact": False,
        "circle_of_friends_status": "needs_review_inspiration",
        "term_correction": term_correction,
    }
    (exports / "content-material-card.json").write_text(json.dumps(material_card, ensure_ascii=False), encoding="utf-8")
    content_assets = {
        "review_required": True,
        "publication_allowed": False,
        "summary_path": str(exports / "knowledge-note.md"),
        "smart_summary_path": str(exports / "smart-summary.md"),
        "smart_summary_prompt_path": str(exports / "smart-summary-codex-prompt.md"),
        "timeline_path": str(exports / "full-transcript.md"),
        "audit_path": str(exports / "extraction-audit.md"),
        "key_segments_path": str(exports / "key-segments.md"),
        "short_video_script_drafts_path": str(exports / "short-video-script-drafts.md"),
        "highlight_post_drafts_path": str(exports / "highlight-post-drafts.md"),
        "content_material_card_path": str(exports / "content-material-card.json"),
        "content_material_card_markdown_path": str(exports / "content-material-card.md"),
    }
    content_candidate_pack = {
        "schema": "video_knowledge_pipeline.content_candidate_pack.v1",
        "term_correction": term_correction,
        "candidates": [
            {
                "id": "candidate-001",
                "timeline_index": 1,
                "start": 0,
                "end": 60,
                "candidate_types": ["method", "tool", "quote"],
                "viewpoint": "Browserbase 通过 CDP 连接浏览器，可以减少中间层。",
                "case_or_example": "示例说明：用 CDP 连接已登录浏览器，比厚手套式自动化更稳定。",
                "reusable_quote": "手套越薄，效果越好。",
                "fact_status": "needs_review",
                "review_required": True,
                "publication_allowed": False,
                "allowed_as_fact": False,
                "evidence_paths": ["frames/001.jpg"],
                "summary_chapter_refs": [{"chapter_index": 1, "title": "浏览器自动化工具比较"}],
                "evidence_citations": [
                    {
                        "source_type": "transcript",
                        "time_range": "00:00:00.000-00:01:00.000",
                        "text": "Browserbase 通过 CDP 连接浏览器，可以减少中间层。",
                        "evidence_path": "frames/001.jpg"
                    }
                ],
            }
        ],
    }
    (exports / "content-candidate-pack.json").write_text(json.dumps(content_candidate_pack, ensure_ascii=False), encoding="utf-8")
    (exports / "content-candidate-pack.md").write_text("# 内容素材候选\n", encoding="utf-8")
    content_assets["content_candidate_pack_path"] = str(exports / "content-candidate-pack.json")
    content_assets["content_candidate_pack_markdown_path"] = str(exports / "content-candidate-pack.md")
    (root / "manifest.json").write_text(json.dumps({"title": "Demo Course", "content_assets": content_assets, "content_candidate_pack_json": "exports/content-candidate-pack.json", "content_candidate_pack_markdown": "exports/content-candidate-pack.md"}, ensure_ascii=False), encoding="utf-8")
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 60,
            "corrected_transcript": "第一部分介绍 Browserbase 和 Playwright 的浏览器自动化差异。",
            "visual_route": "document_visual",
            "visual_text": "Browserbase Playwright CDP",
            "human_corrected_visual_text": "人工核对屏幕文字：Browserbase 通过 CDP 连接浏览器。",
            "frame_paths": ["frames/001.jpg"],
            "quality_issues": ["tile_result_needs_review"],
            "tile_review_targets": [{"tile_id": "tile-001", "confidence": 0.41, "reasons": ["tile_result_low_confidence"], "evidence_path": "frames/001.jpg"}],
            "tags": ["工具名", "结论"],
        },
        {
            "index": 2,
            "start": 60,
            "end": 120,
            "corrected_transcript": "第二部分演示如何通过 CDP 连接已登录浏览器，并降低 token 成本。",
            "visual_route": "temporal_sequence",
            "temporal_visual_understanding": {"operation_steps": ["打开浏览器", "连接 CDP", "执行搜索"]},
            "temporal_frame_paths": ["frames/001.jpg"],
            "tags": ["操作演示"],
        },
        {
            "index": 3,
            "start": 120,
            "end": 190,
            "corrected_transcript": "最后总结手套越薄效果越好，减少中间层可以提升稳定性。",
            "visual_route": "semantic_frame",
        },
    ]
    (root / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")


def test_video_moment_index_builds_queryable_evidence_chunks(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    result = build_video_moment_index(bundle, query="Browserbase CDP", target_window_seconds=90, write=True)

    assert result["schema"] == "video_knowledge_pipeline.video_moment_index.v1"
    assert result["summary"]["chunks"] == 2
    assert result["summary"]["chunks_with_visual_evidence"] >= 1
    assert result["query_hits"]
    assert result["query_hits"][0]["timeline_indexes"]
    assert (bundle / "exports" / "video-moment-index.json").exists()
    assert (bundle / "exports" / "video-moment-index.md").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_moment_index"] == "exports/video-moment-index.json"


def test_long_video_memory_pack_groups_short_memories(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    result = build_long_video_memory_pack(bundle, target_window_seconds=60, long_group_size=2, write=True)

    assert result["schema"] == "video_knowledge_pipeline.long_video_memory_pack.v1"
    assert result["summary"]["short_memories"] >= 2
    assert result["summary"]["long_memories"] >= 1
    assert result["final_memory_map"]["coverage"]["end_time"] == "00:03:10.000"
    assert "MovieChat" in " ".join(result["inspired_by"])
    assert (bundle / "exports" / "long-video-memory-pack.json").exists()
    assert (bundle / "exports" / "long-video-memory-pack.md").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["long_video_memory_pack"] == "exports/long-video-memory-pack.json"


def test_video_rag_pack_writes_jsonl_retrieval_units(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    result = build_video_rag_pack(bundle, query="CDP token", target_window_seconds=90, write=True)

    assert result["schema"] == "video_knowledge_pipeline.video_rag_pack.v1"
    assert result["summary"]["moment_chunks"] == 2
    assert result["summary"]["chunks"] > result["summary"]["moment_chunks"]
    assert result["summary"]["visual_evidence_chunks"] >= 2
    assert result["summary"]["review_gap_chunks"] >= 1
    assert result["summary"]["content_asset_chunks"] >= 1
    assert result["summary"]["content_candidate_chunks"] == 1
    assert result["summary"]["short_memory_chunks"] >= 2
    assert result["summary"]["chapter_memory_chunks"] >= 1
    assert result["summary"]["theme_memory_chunks"] >= 1
    assert result["summary"]["chunks_by_kind"]["moment"] == 2
    assert result["operator_boundary"]["multi_granularity_jsonl"] is True
    assert result["retrieval_units"]
    assert result["retrieved_moments"]
    assert result["retrieval_units"][0]["metadata"]["timeline_indexes"]
    assert result["operator_boundary"]["no_cloud_model_call"] is True
    candidate_search = search_video_rag(bundle, query="手套越薄", top_k=3, write=False)
    candidate_hit = next(hit for hit in candidate_search["hits"] if hit["chunk_kind"] == "content_candidate" and hit["candidate_id"] == "candidate-001")
    assert candidate_hit["term_validation_status"] == "ready_for_import"
    assert candidate_hit["accepted_validation_decisions"] == 2
    jsonl = bundle / "exports" / "video-rag-chunks.jsonl"
    assert jsonl.exists()
    lines = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = {row["metadata"]["chunk_kind"] for row in lines}
    assert {"moment", "visual_evidence", "review_gap", "content_asset", "content_candidate", "short_memory", "chapter_memory", "theme_memory", "memory_boundary"} <= kinds
    chapter_memory = next(row for row in lines if row["metadata"]["chunk_kind"] == "chapter_memory")
    assert chapter_memory["metadata"]["memory_level"] == "chapter"
    assert chapter_memory["metadata"]["child_memory_ids"]
    assert chapter_memory["metadata"]["child_moment_indexes"] == [1, 2]
    assert lines[0]["id"].endswith(":moment:0001")
    assert any("tile_result_low_confidence" in row["text"] for row in lines if row["metadata"]["chunk_kind"] == "review_gap")
    content_candidate = next(row for row in lines if row["metadata"]["chunk_kind"] == "content_candidate")
    assert content_candidate["metadata"]["candidate_id"] == "candidate-001"
    assert content_candidate["metadata"]["summary_chapter_ref_count"] == 1
    assert "Browserbase 通过 CDP" in content_candidate["text"]
    assert "Term validation: ready_for_import" in content_candidate["text"]
    assert content_candidate["metadata"]["term_validation_status"] == "ready_for_import"
    assert content_candidate["metadata"]["accepted_validation_decisions"] == 2
    content_asset = next(row for row in lines if row["metadata"]["chunk_kind"] == "content_asset" and row["metadata"].get("content_asset_key") == "content_material_card_path")
    assert content_asset["metadata"]["term_validation_status"] == "ready_for_import"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_rag_pack"] == "exports/video-rag-pack.json"
    assert manifest["mcp_video_rag_pack_args"] == "mcp-video-rag-pack.args.json"


def test_external_capability_pack_groups_reusable_local_capabilities(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    result = build_external_capability_pack(bundle, query="Browserbase", write=True)

    assert result["schema"] == "video_knowledge_pipeline.external_capability_pack.v1"
    keys = {row["key"] for row in result["capabilities"]}
    assert {
        "long_video_layered_summary",
        "time_localization",
        "video_rag",
        "local_vlm_adapter",
        "content_material_generation",
    } <= keys
    assert result["operator_boundary"]["no_cloud_call"] is True
    assert (bundle / "exports" / "external-capability-pack.json").exists()
    assert (bundle / "exports" / "video-rag-pack.json").exists()
    assert (bundle / "exports" / "local-vlm-adapter-plan.json").exists()
    assert (bundle / "exports" / "local-vlm-adapter-plan.md").exists()
    by_key = {row["key"]: row for row in result["capabilities"]}
    assert any(path.endswith("local-vlm-adapter-plan.md") for path in by_key["local_vlm_adapter"]["artifacts"])
    content_artifacts = by_key["content_material_generation"]["artifacts"]
    assert any(path.endswith("key-segments.md") for path in content_artifacts)
    assert any(path.endswith("short-video-script-drafts.md") for path in content_artifacts)
    assert any(path.endswith("highlight-post-drafts.md") for path in content_artifacts)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["external_capability_pack"] == "exports/external-capability-pack.json"
    assert manifest["mcp_external_capability_pack_args"] == "mcp-external-capability-pack.args.json"
    registry = build_run_artifact_registry(bundle, write=False)
    run_types = {row["run_type"] for row in registry["runs"]}
    assert {"video_moment_index", "long_video_memory_pack", "video_rag_pack", "external_capability_pack"} <= run_types
    runs_by_type = {row["run_type"]: row for row in registry["runs"]}
    assert runs_by_type["external_capability_pack"]["status"] == "needs_input"
    audit = audit_bundle_mcp_args(bundle)
    assert audit["status"] == "ok"
    assert any(row["key"] == "mcp_video_rag_pack_args" for row in audit["rows"])
    assert any(row["key"] == "mcp_external_capability_pack_args" for row in audit["rows"])
    mcp_result = run_mcp_call("external_capability_pack", bundle / "mcp-external-capability-pack.args.json")
    assert mcp_result["schema"] == "video_knowledge_pipeline.external_capability_pack.v1"


def test_external_capability_commands_work_through_cli_main(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    assert main(["video-rag-pack", str(bundle), "--query", "CDP"]) == 0
    rag_out = json.loads(capsys.readouterr().out)
    assert rag_out["schema"] == "video_knowledge_pipeline.video_rag_pack.v1"
    assert (bundle / "exports" / "video-rag-chunks.jsonl").exists()

    assert main(["external-capability-pack", str(bundle), "--query", "Browserbase"]) == 0
    pack_out = json.loads(capsys.readouterr().out)
    assert pack_out["schema"] == "video_knowledge_pipeline.external_capability_pack.v1"
    assert {row["key"] for row in pack_out["capabilities"]} >= {"video_rag", "local_vlm_adapter", "content_material_generation"}

def test_video_rag_search_reads_jsonl_and_writes_search_artifacts(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    build_video_rag_pack(bundle, query="CDP", target_window_seconds=90, write=True)

    result = search_video_rag(bundle, query="CDP token", top_k=3, write=True)

    assert result["schema"] == "video_knowledge_pipeline.video_rag_search.v1"
    assert result["summary"]["chunks_loaded"] >= 1
    assert result["hits"]
    assert result["hits"][0]["timeline_indexes"]
    assert result["operator_boundary"]["no_cloud_model_call"] is True
    candidate_search = search_video_rag(bundle, query="手套越薄", top_k=3, write=False)
    candidate_hit = next(hit for hit in candidate_search["hits"] if hit["chunk_kind"] == "content_candidate" and hit["candidate_id"] == "candidate-001")
    assert candidate_hit["term_validation_status"] == "ready_for_import"
    assert (bundle / "exports" / "video-rag-search.json").exists()
    assert (bundle / "exports" / "video-rag-search.md").exists()
    assert "Codex term validation" in (bundle / "exports" / "video-rag-search.md").read_text(encoding="utf-8")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mcp_video_rag_search_args"] == "mcp-video-rag-search.args.json"

    sqlite_result = search_video_rag(bundle, query="CDP token", top_k=3, retrieval_backend="sqlite", write=True)
    assert sqlite_result["retrieval_backend"] == "sqlite"
    assert sqlite_result["summary"]["sqlite_index_exists"] is True
    assert sqlite_result["summary"]["sqlite_index_built"] is True
    assert sqlite_result["hits"]
    assert (bundle / "exports" / "video-rag-index.sqlite").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_rag_search_backend"] == "sqlite"
    assert manifest["video_rag_sqlite_index"] == "exports/video-rag-index.sqlite"
    mcp_result = run_mcp_call("video_rag_search", bundle / "mcp-video-rag-search.args.json")
    assert mcp_result["retrieval_backend"] == "sqlite"
    assert mcp_result["operator_boundary"]["no_vector_backend_started"] is True


def test_local_vlm_serving_smoke_defaults_to_plan_only(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    extra = bundle / "frames" / "002.jpg"
    extra.write_bytes(b"fake2")
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    timeline[1]["temporal_frame_paths"] = ["frames/001.jpg", "frames/002.jpg"]
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")

    result = local_vlm_serving_smoke(provider="internvl", bundle_dir=str(bundle), execute=False, write=True, frame_group_count=8)

    assert result["schema"] == "video_knowledge_pipeline.local_vlm_serving_smoke.v1"
    assert result["execute"] is False
    assert result["provider"] == "local_vlm"
    assert result["operator_boundary"]["does_not_start_model_server"] is True
    assert result["operator_boundary"]["does_not_modify_timeline"] is True
    assert result["input_spec"]["short_frame_group_found"] is True
    assert result["input_spec"]["short_frame_group_image_count"] == 2
    matrix = {row["key"]: row for row in result["capability_matrix"]}
    assert matrix["text_json"]["status"] == "planned"
    assert matrix["short_frame_group_json"]["status"] == "planned"
    assert (bundle / "local-vlm-serving-smoke.json").exists()
    assert (bundle / "local-vlm-serving-smoke.md").exists()
    assert (bundle / "mcp-local-vlm-serving-smoke.args.json").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["local_vlm_serving_smoke"] == "local-vlm-serving-smoke.md"
    assert manifest["mcp_local_vlm_serving_smoke_args"] == "mcp-local-vlm-serving-smoke.args.json"
    run = json.loads((bundle / "runs" / "local-vlm-serving-smoke" / "run.json").read_text(encoding="utf-8"))
    assert run["run_type"] == "local_vlm_serving_smoke"
    assert run["status"] == "needs_execution"
    assert run["operator_boundary"]["does_not_start_model_server"] is True
    assert run["artifacts"][0]["path"].endswith("local-vlm-serving-smoke.json")
    mcp_result = run_mcp_call("local_vlm_serving_smoke", bundle / "mcp-local-vlm-serving-smoke.args.json")
    assert mcp_result["schema"] == "video_knowledge_pipeline.local_vlm_serving_smoke.v1"
    assert mcp_result["execute"] is False

def test_video_rag_http_response_serves_health_and_search(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    build_video_rag_pack(bundle, query="CDP", target_window_seconds=90, write=True)

    status, headers, body = video_rag_http_response(bundle, "/health")
    health = json.loads(body.decode("utf-8"))
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert health["ok"] is True
    assert health["operator_boundary"]["no_cloud_model_call"] is True

    status, _, body = video_rag_http_response(bundle, "/search?q=CDP&top_k=2")
    result = json.loads(body.decode("utf-8"))
    assert status == 200
    assert result["schema"] == "video_knowledge_pipeline.video_rag_search.v1"
    assert result["hits"]
    assert result["operator_boundary"]["no_vector_backend_started"] is True

    plan = video_rag_service_plan(bundle, port=8799, write=True)
    assert plan["schema"] == "video_knowledge_pipeline.video_rag_service_plan.v1"
    assert plan["endpoints"]["search"].startswith("http://127.0.0.1:8799/search")
    assert (bundle / "exports" / "video-rag-service-plan.md").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mcp_video_rag_service_plan_args"] == "mcp-video-rag-service-plan.args.json"
