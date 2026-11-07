from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from video_knowledge_pipeline.cli import audit_bundle_mcp_args, build_parser
from video_knowledge_pipeline.ocr_backfill import run_ocr_backfill
from video_knowledge_pipeline.screen_text_recovery import run_screen_text_recovery


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (120, 80), color=(255, 255, 255))
    image.save(path)



def test_ebook_pipeline_environment_sets_rapidocr_gpu_and_restores(monkeypatch) -> None:
    monkeypatch.setenv("EBOOK_CONVERTER_RAPIDOCR_DEVICE", "cpu")
    monkeypatch.delenv("EBOOK_CONVERTER_RAPIDOCR_CUDA_DEVICE_ID", raising=False)

    with visual_structure._ebook_pipeline_environment({"rapidocr_device": "cuda", "rapidocr_cuda_device_id": 2}):
        assert os.environ["EBOOK_CONVERTER_RAPIDOCR_DEVICE"] == "cuda"
        assert os.environ["EBOOK_CONVERTER_RAPIDOCR_CUDA_DEVICE_ID"] == "2"

    assert os.environ["EBOOK_CONVERTER_RAPIDOCR_DEVICE"] == "cpu"
    assert "EBOOK_CONVERTER_RAPIDOCR_CUDA_DEVICE_ID" not in os.environ
def test_screen_text_recovery_generates_crops_and_preserves_source_frame(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    frame = bundle / "assets" / "ui.jpg"
    _write_image(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 2,
                    "visual_route": "semantic_frame",
                    "material_types": ["ui", "text"],
                    "signals": ["browser"],
                    "visual_text": f"# {frame.stem}\n\n<!-- source: {frame} -->",
                    "frame_paths": [str(frame)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    original_bytes = frame.read_bytes()

    preview = run_screen_text_recovery(bundle, write=True)
    assert preview["crop_summary"]["written"] == 0
    assert preview["run_registry"]["run_type"] == "screen_text_recovery"
    assert preview["run_registry"]["status"] == "needs_execution"
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    assert any(row["run_type"] == "screen_text_recovery" for row in registry["runs"])
    assert frame.read_bytes() == original_bytes

    result = run_screen_text_recovery(bundle, execute_crops=True, write=True)

    assert result["crop_summary"]["written"] >= 1
    assert result["run_registry"]["run_type"] == "screen_text_recovery"
    assert result["run_registry"]["status"] == "needs_execution"
    run = json.loads((bundle / "runs" / "screen-text-recovery" / "run.json").read_text(encoding="utf-8"))
    assert run["retry_command"].startswith(".\\scripts\\video-knowledge.ps1 run-screen-text-recovery")
    assert frame.read_bytes() == original_bytes
    crop_path = Path(result["items"][0]["crops"][0]["output_path"])
    assert crop_path.exists()
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert str(crop_path) in timeline[0]["screen_text_recovery"]["crop_paths"]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mcp_screen_text_recovery_args"] == "mcp-run-screen-text-recovery.args.json"


def test_wrapper_only_ocr_import_does_not_clear_screen_text_blocker(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    frame = bundle / "assets" / "ui.jpg"
    _write_image(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 2,
                    "visual_route": "semantic_frame",
                    "material_types": ["ui", "text"],
                    "signals": ["browser"],
                    "visual_text": f"# {frame.stem}\n\n<!-- source: {frame} -->",
                    "quality_issues": ["missing_visual_text", "screen_text_low_confidence"],
                    "frame_paths": [str(frame)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    imported = tmp_path / "ocr-import.json"
    imported.write_text(
        json.dumps({"items": [{"index": 1, "text": f"# {frame.stem}\n\n<!-- likely-infographic: non_page_aspect_ratio -->\n![source image]({frame})", "source": str(frame)}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = run_ocr_backfill(bundle, input_json=imported)

    assert result["summary"]["succeeded"] == 0
    assert result["backfill"]["updated"] == 0
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[0]["quality_issues"] == ["missing_visual_text", "screen_text_low_confidence"]


def test_crop_ocr_import_requires_human_review_before_clearing_blocker(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    frame = bundle / "assets" / "ui.jpg"
    crop = bundle / "ocr-crops" / "timeline-0001" / "central_content.jpg"
    _write_image(frame)
    _write_image(crop)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 2,
                    "visual_route": "semantic_frame",
                    "material_types": ["ui", "text"],
                    "signals": ["browser"],
                    "quality_issues": ["missing_visual_text", "screen_text_low_confidence"],
                    "frame_paths": [str(frame)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    imported = tmp_path / "ocr-import.json"
    imported.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "index": 1,
                        "text": "知识星球页面，包含 CursorFAQ 和教学视频入口",
                        "source": str(crop),
                        "notes": "Imported from screen_text_recovery crop OCR.",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_ocr_backfill(bundle, input_json=imported)

    assert result["summary"]["succeeded"] == 0
    assert result["backfill"]["updated"] == 0
    assert result["items"][0]["stderr"] == "crop OCR requires human review before clearing screen text gap"
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert "visual_text" not in timeline[0]
    assert timeline[0]["quality_issues"] == ["missing_visual_text", "screen_text_low_confidence"]


def test_screen_text_recovery_indexes_select_exact_candidates(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frames = []
    for index in range(1, 5):
        frame = assets / f"ui-{index}.jpg"
        _write_image(frame)
        frames.append(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": index,
                    "start": index,
                    "end": index + 1,
                    "visual_route": "semantic_frame",
                    "material_types": ["ui", "text"],
                    "signals": ["browser"],
                    "visual_text": f"# {frames[index - 1].stem}\n\n<!-- source: {frames[index - 1]} -->",
                    "frame_paths": [str(frames[index - 1])],
                }
                for index in range(1, 5)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_screen_text_recovery(bundle, execute_crops=True, indexes=[2, 4], limit=1, write=True)

    assert result["available_candidates"] == 4
    assert result["requested_indexes"] == [2, 4]
    assert result["selected_indexes"] == [2]
    assert [item["index"] for item in result["items"]] == [2]
    assert result["crop_summary"]["written"] >= 1
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert "screen_text_recovery" not in timeline[0]
    assert timeline[1]["screen_text_recovery"]["crop_paths"]
    assert "screen_text_recovery" not in timeline[3]
def test_screen_text_recovery_cli_and_mcp_audit_contract(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "mcp_screen_text_recovery_args": "mcp-run-screen-text-recovery.args.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    (bundle / "mcp-run-screen-text-recovery.args.json").write_text(
        json.dumps({"bundle_dir": str(bundle), "execute_crops": False, "execute_ocr": False, "write": True}, ensure_ascii=False),
        encoding="utf-8",
    )

    args = build_parser().parse_args(["run-screen-text-recovery", str(bundle), "--execute-crops", "--limit", "3"])
    audit = audit_bundle_mcp_args(bundle)

    assert args.command == "run-screen-text-recovery"
    assert args.execute_crops is True
    assert args.limit == 3
    assert audit["status"] == "ok"
    assert audit["rows"][0]["tool"] == "run_screen_text_recovery"

# Moved from test_video_pipeline_smoke.py during Phase 10 split.

import json
import os
from pathlib import Path

from video_knowledge_pipeline.acceptance_check import acceptance_check
from video_knowledge_pipeline.acceptance_run import run_acceptance_bundle, run_acceptance_run
from video_knowledge_pipeline.asr_adapter import normalize_asr_output
from video_knowledge_pipeline.asr_environment import asr_environment_status
from video_knowledge_pipeline.asr_execution import asr_smoke, run_asr_plan
from video_knowledge_pipeline.asr_runner import plan_asr_run
from video_knowledge_pipeline.batch_run import batch_video_knowledge_run
from video_knowledge_pipeline.bundle_next import bundle_advance, bundle_advance_log, bundle_advance_queue, bundle_next_action
from video_knowledge_pipeline.bundle_status import bundle_status_report, controlled_execution_check
from video_knowledge_pipeline.cli import audit_bundle_mcp_args, build_parser, main as cli_main, resolve_mcp_args_path, run_mcp_call
from video_knowledge_pipeline.config import config_status, resolve_vision_execution_profile, service_url, vision_execution_profile
from video_knowledge_pipeline.controlled_execution_smoke import controlled_execution_smoke
from video_knowledge_pipeline.knowledge_coverage import build_knowledge_coverage
from video_knowledge_pipeline.knowledge_note_export import export_knowledge_note
from video_knowledge_pipeline.lecture_package import render_lecture_review_html
from video_knowledge_pipeline.local_video_run import prepare_local_video_run
from video_knowledge_pipeline.local_vlm_server_adapter import local_vlm_adapter_plan
from video_knowledge_pipeline.ocr_backfill import run_ocr_backfill
from video_knowledge_pipeline.multimodal_frame_analyzer import (
    _normalise_visual_understanding,
    run_multimodal_frame_analysis,
    vision_analysis_apply_restore,
    vision_analysis_restore_plan,
    vision_analysis_run_log,
)
from video_knowledge_pipeline.peepshow_adapter import attach_peepshow_output_to_bundle
from video_knowledge_pipeline.review_session import apply_review_notes_to_bundle, prepare_review_session, validate_review_notes_for_bundle
from video_knowledge_pipeline.source_artifacts import build_source_artifact_index, summarize_manifest_source_artifacts
from video_knowledge_pipeline.storage import bundle_write_lock, write_json
from video_knowledge_pipeline.temporal_frame_groups import run_temporal_frame_groups
from video_knowledge_pipeline.temporal_visual_analyzer import _normalise_temporal_understanding, run_temporal_visual_analysis
from video_knowledge_pipeline.transcript_resegment import resegment_transcript
from video_knowledge_pipeline.vision_acceptance import vision_acceptance_plan
from video_knowledge_pipeline.video_frame_router import run_video_frame_router
from video_knowledge_pipeline.video_source import prepare_video_source
from video_knowledge_pipeline.vision_api import parse_model_json, resolve_provider_config, test_vision_provider as run_vision_provider_test
from video_knowledge_pipeline.vision_environment import vision_environment_status
from video_knowledge_pipeline.vision_preflight import vision_execution_preflight
from video_knowledge_pipeline.vision_provider_smoke import rank_vision_providers, vision_provider_matrix, vision_provider_smoke
from video_knowledge_pipeline.webui_bridge import export_webui_bundle, refresh_bundle_review_html
import video_knowledge_pipeline.visual_structure as visual_structure
from video_knowledge_pipeline.visual_structure import run_visual_structure_plan
from video_knowledge_pipeline.high_res_tile_plan import run_high_res_tile_plan



def test_ebook_artifact_reader_prefers_verified_local_utf8_text(tmp_path: Path) -> None:
    output_dir = tmp_path / "ebook"
    output_dir.mkdir()
    artifact_path = output_dir / "book.md"
    artifact_path.write_text("### 案例\n\n正确的中文内容", encoding="utf-8")

    def fake_call_tool(name: str, args: dict[str, object]) -> dict[str, object]:
        assert name == "read_artifact"
        return {"path": str(args["path"]), "artifact_type": "markdown", "text": "����"}

    result = visual_structure._read_best_ebook_artifact(
        fake_call_tool,
        {"artifacts": [{"type": "markdown", "path": str(artifact_path)}]},
        output_dir=output_dir,
    )

    assert result["text"] == "### 案例\n\n正确的中文内容"
    assert result["text_encoding_source"] == "direct_utf8_artifact"


def test_repair_ebook_artifact_text_replaces_mojibake_without_rerunning_ocr(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    artifact_dir = bundle / "visual-structure" / "timeline-0001" / "ebook_pipeline"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "book.md"
    artifact_path.write_text("### 案例\n\n因为要了解您的家庭情况", encoding="utf-8")
    frame = bundle / "assets" / "frame.jpg"
    _write_image(frame)
    source_package = bundle / "lecture-package.json"
    timeline = [
        {
            "index": 1,
            "visual_text": "### ����",
            "legacy_visual_text": ["����"],
            "frame_paths": [str(frame)],
            "structured_visual": [{"source": "ebook_markdown_pipeline", "type": "document_visual", "markdown": "### ����"}],
            "ebook_pipeline_status": {
                "ok": True,
                "artifact_path": str(artifact_path),
                "artifact_type": "markdown",
                "image_path": str(frame),
                "output_dir": str(artifact_dir),
            },
        }
    ]
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1", "source_package": str(source_package)})
    write_json(bundle / "timeline.json", timeline)
    write_json(source_package, {"timeline": timeline})

    result = visual_structure.repair_ebook_artifact_text(bundle)
    repaired = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))[0]
    package_repaired = json.loads(source_package.read_text(encoding="utf-8"))["timeline"][0]

    assert result["status"] == "completed"
    assert result["repaired_count"] == 1
    assert result["failed_count"] == 0
    assert result["source_package_updated"] is True
    assert repaired["visual_text"] == "### 案例\n因为要了解您的家庭情况"
    assert package_repaired["visual_text"] == repaired["visual_text"]
    assert repaired["legacy_visual_text"] == []
    assert repaired["structured_visual"][-1]["text_encoding_source"] == "direct_utf8_artifact"

def test_visual_structure_http_command_uses_unified_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "video-knowledge-pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.config.v1",
                "services": {
                    "review_webui": {"type": "static_file", "entrypoint": "webui-bundle/review.html"},
                    "ebook_markdown_pipeline_http": {"host": "127.0.0.1", "port": 9876, "path": "/call"},
                    "mcp": {"transport": "stdio"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))

    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    _write_image(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 1,
                    "visual_route": "document_visual",
                    "material_types": ["table"],
                    "frame_paths": [str(frame)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_visual_structure_plan(bundle)

    command = result["items"][0]["commands"]["ebook_pipeline_http"]
    assert "http://127.0.0.1:9876/call" in command
    assert "8765" not in command
    assert result["summary"]["runtime_config"]["config_path"] == str(config_path.resolve())
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "Runtime Config" in report
    assert str(config_path.resolve()) in report
    assert "http://127.0.0.1:9876/call" in report
    run = json.loads((bundle / "runs" / "visual-structure-ebook" / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "needs_execution"
    assert "--execute-ebook-pipeline" in run["retry_command"]


def test_visual_structure_execute_imports_ebook_markdown_with_evidence(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    _write_image(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 2,
                    "visual_route": "document_visual",
                    "material_types": ["table"],
                    "frame_paths": [str(frame)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, dict]] = []
    # The production import binds the service response to a real local artifact.
    # A response-only mock no longer represents the successful ebook contract.
    output_path = bundle / "visual-structure" / "timeline-0001" / "ebook_pipeline" / "output.md"
    output_text = "| 概念 | 含义 |\n|---|---|\n| 时间资产 | 可复用的知识产出 |\n\n```text\none person company\n```"

    def fake_call_tool(name: str, payload: dict) -> dict:
        calls.append((name, payload))
        if name == "process_material":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text, encoding="utf-8")
            return {"job_id": "job-1"}
        if name == "get_job_status":
            return {
                "status": "done",
                "artifacts": [
                    {
                        "type": "markdown",
                        "path": str(bundle / "visual-structure" / "timeline-0001" / "ebook_pipeline" / "output.md"),
                    }
                ],
            }
        if name == "read_artifact":
            return {
                "path": payload["path"],
                "artifact_type": payload["artifact_type"],
                "text": "| 概念 | 含义 |\n|---|---|\n| 时间资产 | 可复用的知识产出 |\n\n```text\none person company\n```",
            }
        raise AssertionError(name)

    monkeypatch.setattr(visual_structure, "_ebook_call_tool", lambda: fake_call_tool)

    result = run_visual_structure_plan(bundle, execute_ebook_pipeline=True)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))

    assert [name for name, _payload in calls] == ["process_material", "get_job_status", "read_artifact"]
    assert result["summary"]["ebook_pipeline_succeeded"] == 1
    assert result["summary"]["updated"] == 1
    assert "时间资产" in timeline[0]["visual_text"]
    structured = timeline[0]["structured_visual"][0]
    assert structured["source"] == "ebook_markdown_pipeline"
    assert structured["type"] == "table"
    assert structured["image_path"] == str(frame.resolve())
    assert structured["artifact_path"].endswith("output.md")
    assert "one person company" in structured["markdown"]
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "Timeline 1" in report
    assert "Status: `imported`" in report
    assert "process_material -> get_job_status -> read_artifact" in report
    assert "ebook_pipeline_succeeded" in report
    run = json.loads((bundle / "runs" / "visual-structure-ebook" / "run.json").read_text(encoding="utf-8"))
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["run_type"] == "visual_structure_ebook"
    assert run["failed_items"] == []
    assert registry["status_counts"] == {"completed": 1}


def test_visual_structure_classifies_umi_ocr_missing_blocker(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    _write_image(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 2,
                    "visual_route": "document_visual",
                    "material_types": ["table"],
                    "frame_paths": [str(frame)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_call_tool(name: str, payload: dict) -> dict:
        if name == "process_material":
            return {"job_id": "job-1"}
        if name == "get_job_status":
            return {"status": "failed", "error": "Umi-OCR module not found: PPOCR_api.py", "artifacts": []}
        raise AssertionError(name)

    monkeypatch.setattr(visual_structure, "_ebook_call_tool", lambda: fake_call_tool)

    result = run_visual_structure_plan(bundle, execute_ebook_pipeline=True)

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    ebook_result = manifest["visual_structure"]["ebook_pipeline_results"][0]
    assert result["summary"]["ebook_pipeline_succeeded"] == 0
    assert result["summary"]["ebook_pipeline_blockers"] == {"umi_ocr_missing": 1}
    assert ebook_result["blocker"] == "umi_ocr_missing"
    assert "PPOCR_api.py" in ebook_result["error"]
    assert "Repair ebook_markdown_pipeline" in ebook_result["next_action"]
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "ebook blocker: `umi_ocr_missing`" in report
    assert "Repair ebook_markdown_pipeline" in report
    run = json.loads((bundle / "runs" / "visual-structure-ebook" / "run.json").read_text(encoding="utf-8"))
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    assert run["status"] == "needs_retry"
    assert run["failed_items"][0]["reason"] == "umi_ocr_missing"
    assert run["failed_items"][0]["suggested_next_tool"] == "run_visual_structure_plan"
    assert "run-visual-structure" in run["failed_items"][0]["suggested_retry_command"]
    assert "--indexes '1'" in run["failed_items"][0]["suggested_retry_command"]
    assert run["failed_items"][0]["evidence_paths"][0] == str(frame)
    assert "prepare-review-session" in run["failed_items"][0]["review_command"]
    assert "--execute-ebook-pipeline" in run["retry_command"]
    assert registry["status_counts"] == {"needs_retry": 1}


def test_visual_structure_rejects_synthetic_ebook_wrapper(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "016_0000240000ms.jpg"
    _write_image(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 24,
                    "end": 26,
                    "visual_route": "document_visual",
                    "material_types": ["table"],
                    "frame_paths": [str(frame)],
                    "visual_text": f"# {frame.stem}\n\n<!-- likely-infographic: non_page_aspect_ratio -->\n![source image]({frame})",
                    "structured_visual": [
                        {
                            "source": "ebook_markdown_pipeline",
                            "type": "structured_visual",
                            "markdown": f"# {frame.stem}\n\n<!-- likely-infographic: non_page_aspect_ratio -->\n![source image]({frame})",
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_path = bundle / "visual-structure" / "timeline-0001" / "ebook_pipeline" / "output.md"

    def fake_call_tool(name: str, payload: dict) -> dict:
        if name == "process_material":
            return {"job_id": "job-1"}
        if name == "get_job_status":
            return {"status": "done", "artifacts": [{"type": "markdown", "path": str(output_path)}]}
        if name == "read_artifact":
            return {
                "path": payload["path"],
                "artifact_type": payload["artifact_type"],
                "text": f"# {frame.stem}\n\n<!-- likely-infographic: non_page_aspect_ratio -->\n![source image]({frame})",
            }
        raise AssertionError(name)

    monkeypatch.setattr(visual_structure, "_ebook_call_tool", lambda: fake_call_tool)

    result = run_visual_structure_plan(bundle, execute_ebook_pipeline=True)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    coverage = build_knowledge_coverage({}, timeline, bundle_dir=bundle)

    ebook_result = result["summary"]["ebook_pipeline_blockers"]
    assert result["summary"]["ebook_pipeline_succeeded"] == 0
    assert ebook_result == {"ocr_wrapper_only": 1}
    assert result["summary"]["updated"] == 0
    assert not timeline[0].get("visual_text")
    assert not timeline[0].get("structured_visual")
    assert timeline[0]["ebook_pipeline_status"]["blocker"] == "ocr_wrapper_only"
    assert "needs_high_res_tile_recovery" in timeline[0]["quality_issues"]
    structured_channel = next(channel for channel in coverage["channels"] if channel["key"] == "structured_visual")
    screen_channel = next(channel for channel in coverage["channels"] if channel["key"] == "screen_text")
    assert structured_channel["covered_count"] == 0
    assert structured_channel["blocker_count"] == 1
    assert screen_channel["covered_count"] == 0
    assert screen_channel["blocker_count"] == 1
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "ebook blocker: `ocr_wrapper_only`" in report
    run = json.loads((bundle / "runs" / "visual-structure-ebook" / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "needs_retry"
    assert run["failed_items"][0]["reason"] == "ocr_wrapper_only"
    assert run["failed_items"][0]["suggested_next_tool"] == "high_res_tile_plan"
    assert "high-res-tile-plan" in run["failed_items"][0]["suggested_retry_command"]
    assert "run-visual-structure" in run["failed_items"][0]["ebook_retry_command"]
    assert "--indexes '1'" in run["failed_items"][0]["ebook_retry_command"]
    assert "vision-review-triage" in run["failed_items"][0]["multimodal_triage_command"]
    assert "prepare-review-session" in run["failed_items"][0]["review_command"]
    assert str(frame) in run["failed_items"][0]["evidence_paths"]
    assert "visual-structure-report.md" in [artifact["path"] for artifact in run["artifacts"]]

    tile_plan = run_high_res_tile_plan(bundle, execute_tiles=False)
    assert tile_plan["selected_indexes"] == [1]
    assert tile_plan["items"][0]["ebook_pipeline_status"]["blocker"] == "ocr_wrapper_only"
    assert "needs_high_res_tile_recovery" in tile_plan["items"][0]["reasons"]


def test_screen_text_recovery_routes_document_mixed_and_ui_text(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    doc_frame = assets / "doc.jpg"
    mixed_frame = assets / "mixed.jpg"
    ui_frame = assets / "ui.jpg"
    for frame in (doc_frame, mixed_frame, ui_frame):
        frame.write_bytes(b"fake image")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "mcp_visual_structure_args": "mcp-run-visual-structure.args.json",
                "mcp_ocr_backfill_args": "mcp-run-ocr-backfill.args.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 2,
                    "visual_route": "document_visual",
                    "material_types": ["table"],
                    "frame_paths": [str(doc_frame)],
                },
                {
                    "index": 2,
                    "start": 2,
                    "end": 4,
                    "visual_route": "mixed",
                    "material_types": ["text", "software"],
                    "signals": ["operation"],
                    "frame_paths": [str(mixed_frame)],
                },
                {
                    "index": 3,
                    "start": 4,
                    "end": 6,
                    "visual_route": "semantic_frame",
                    "material_types": ["ui", "text"],
                    "signals": ["browser"],
                    "visual_text": f"# {ui_frame.stem}\n\n<!-- source: {ui_frame} -->",
                    "frame_paths": [str(ui_frame)],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    visual_result = run_visual_structure_plan(bundle)
    by_index = {item["index"]: item for item in visual_result["items"]}

    assert by_index[1]["routing_decision"]["primary_tool"] == "ebook_markdown_pipeline"
    assert by_index[1]["routing_decision"]["also_requires_multimodal"] is False
    assert by_index[2]["routing_decision"]["primary_tool"] == "ebook_markdown_pipeline"
    assert by_index[2]["routing_decision"]["also_requires_multimodal"] is True

    ocr_result = run_ocr_backfill(bundle)
    recovery_by_index = {item["index"]: item["screen_text_recovery"] for item in ocr_result["items"]}

    assert ocr_result["summary"]["planned"] == 3
    assert ocr_result["summary"]["succeeded"] == 0
    assert all(item["ok"] is False for item in ocr_result["items"])
    assert all(item["image_exists"] is True for item in ocr_result["items"])
    assert recovery_by_index[1]["strategy"] == "ebook_pipeline"
    assert recovery_by_index[1]["recommended_tool"] == "ebook_markdown_pipeline"
    assert recovery_by_index[2]["strategy"] == "ebook_pipeline_plus_multimodal"
    assert "multimodal" in recovery_by_index[2]["recommended_tool"]
    assert recovery_by_index[3]["strategy"] == "crop_and_ocr"
    assert len(recovery_by_index[3]["crop_candidates"]) >= 5
    assert recovery_by_index[3]["crop_candidates"][0]["coordinate_system"] == "relative_xyxy"

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ocr_backfill"]["screen_text_recovery"]["strategy_counts"]["crop_and_ocr"] == 1
    assert "screen_text_low_confidence" in ocr_result["items"][2]["screen_text_recovery"]["issues"]
    report = Path(ocr_result["report_path"]).read_text(encoding="utf-8")
    assert "Screen Text Recovery Plan" in report
    assert "crop_and_ocr" in report


def test_knowledge_coverage_marks_small_ui_text_low_confidence_and_rejects_wrapper_only() -> None:
    frame = "assets/ui.jpg"
    manifest = {"mcp_visual_structure_args": "mcp-run-visual-structure.args.json"}
    timeline = [
        {
            "index": 1,
            "visual_route": "semantic_frame",
            "material_types": ["ui", "text"],
            "signals": ["browser"],
            "visual_text": "# ui\n\n<!-- source: assets/ui.jpg -->",
            "assets": [{"path": frame}],
        }
    ]

    coverage = build_knowledge_coverage(manifest, timeline)
    screen_channel = next(channel for channel in coverage["channels"] if channel["key"] == "screen_text")

    assert coverage["screen_text_low_confidence"] == 1
    assert coverage["ocr_text_empty"] == 1
    assert coverage["samples"]["screen_text_low_confidence"] == [1]
    assert screen_channel["covered_count"] == 0
    assert screen_channel["blocker_count"] == 1
    assert screen_channel["status"] == "blocked"


def test_visual_structure_indexes_and_limit_select_exact_candidates(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frames = []
    for index in range(1, 5):
        frame = assets / f"frame-{index}.jpg"
        frame.write_bytes(b"fake image")
        frames.append(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": index,
                    "start": index,
                    "end": index + 1,
                    "visual_route": "document_visual",
                    "material_types": ["table"],
                    "frame_paths": [str(frames[index - 1])],
                }
                for index in range(1, 5)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_visual_structure_plan(bundle, indexes=[2, 4], limit=1)

    assert result["summary"]["available_candidates"] == 4
    assert result["summary"]["total_candidates"] == 1
    assert result["summary"]["requested_indexes"] == [2, 4]
    assert result["summary"]["selected_indexes"] == [2]
    assert [item["index"] for item in result["items"]] == [2]
    template = json.loads(Path(result["input_template_json"]).read_text(encoding="utf-8"))
    assert [item["index"] for item in template["items"]] == [2]


def test_visual_structure_skips_human_closed_candidates(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frames = []
    for index in range(1, 4):
        frame = assets / f"frame-{index}.jpg"
        frame.write_bytes(b"fake image")
        frames.append(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 1,
                    "visual_route": "document_visual",
                    "material_types": ["text"],
                    "frame_paths": [str(frames[0])],
                    "review_status": "keep_image",
                    "human_review": {"status": "keep_image", "keep_image": True},
                },
                {
                    "index": 2,
                    "start": 1,
                    "end": 2,
                    "visual_route": "document_visual",
                    "material_types": ["text"],
                    "frame_paths": [str(frames[1])],
                    "review_status": "accepted_known_gap",
                    "human_review": {"status": "accepted_known_gap"},
                },
                {
                    "index": 3,
                    "start": 2,
                    "end": 3,
                    "visual_route": "document_visual",
                    "material_types": ["text"],
                    "frame_paths": [str(frames[2])],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_visual_structure_plan(bundle)

    assert result["summary"]["available_candidates"] == 1
    assert result["summary"]["selected_indexes"] == [3]
    assert [item["index"] for item in result["items"]] == [3]


def test_visual_structure_skips_generic_semantic_text_without_explicit_ocr_issue(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frames = []
    for index in range(1, 4):
        frame = assets / f"frame-{index}.jpg"
        frame.write_bytes(b"fake image")
        frames.append(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "visual_route": "semantic_frame",
                    "material_types": ["ui", "text"],
                    "frame_paths": [str(frames[0])],
                    "visual_understanding": {"objects": ["settings panel"]},
                },
                {
                    "index": 2,
                    "visual_route": "semantic_frame",
                    "material_types": ["ui", "text"],
                    "quality_issues": ["screen_text_low_confidence"],
                    "frame_paths": [str(frames[1])],
                },
                {
                    "index": 3,
                    "visual_route": "temporal_sequence",
                    "material_types": ["code"],
                    "frame_paths": [str(frames[2])],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_visual_structure_plan(bundle)

    assert result["summary"]["available_candidates"] == 2
    assert [item["index"] for item in result["items"]] == [2, 3]


def test_visual_structure_imports_explicit_ebook_artifact_payloads(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    _write_image(frame)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 2,
                    "visual_route": "document_visual",
                    "material_types": ["code"],
                    "frame_paths": [str(frame)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_path = bundle / "visual-structure" / "timeline-0001" / "ebook_pipeline" / "output.md"

    def fake_call_tool(name: str, payload: dict) -> dict:
        if name == "process_material":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("```python\nprint('hello')\n```", encoding="utf-8")
            return {"job_id": "job-1"}
        if name == "get_job_status":
            return {"status": "done", "artifacts": [{"type": "markdown", "path": str(output_path)}]}
        if name == "read_artifact":
            return {
                "path": payload["path"],
                "artifact_type": payload["artifact_type"],
                "markdown": {"content": "```python\nprint('hello')\n```"},
            }
        raise AssertionError(name)

    monkeypatch.setattr(visual_structure, "_ebook_call_tool", lambda: fake_call_tool)

    result = run_visual_structure_plan(bundle, execute_ebook_pipeline=True)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))

    assert result["summary"]["ebook_pipeline_succeeded"] == 1
    assert "print('hello')" in timeline[0]["visual_text"]
    assert timeline[0]["structured_visual"][0]["type"] == "code"


def test_knowledge_coverage_counts_visual_understanding_fields() -> None:
    manifest = {
        "mcp_video_frame_router_args": "router.args.json",
        "mcp_multimodal_frame_analysis_args": "frame.args.json",
        "mcp_temporal_visual_analysis_args": "temporal.args.json",
    }
    timeline = [
        {
            "index": 1,
            "visual_route": "semantic_frame",
            "visual_understanding": {"objects": ["screen"], "evidence_frame_paths": ["frame.jpg"]},
            "assets": [{"path": "frame.jpg"}],
        },
        {
            "index": 2,
            "visual_route": "temporal_sequence",
            "temporal_visual_understanding": {"event_sequence": ["scroll"], "evidence_frame_paths": ["frame.jpg"]},
            "assets": [{"path": "frame.jpg"}],
        },
        {
            "index": 3,
            "visual_route": "semantic_frame",
            "assets": [{"path": "frame.jpg"}],
        },
    ]

    coverage = build_knowledge_coverage(manifest, timeline)

    assert coverage["items_with_visual_route"] == 3
    assert coverage["items_with_visual_understanding"] == 1
    assert coverage["items_with_temporal_understanding"] == 1
    assert coverage["semantic_frame_without_analysis"] == 1
    assert coverage["temporal_sequence_without_analysis"] == 0
    assert coverage["missing_visual_understanding"] == 1
