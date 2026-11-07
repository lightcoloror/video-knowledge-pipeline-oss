from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import build_parser, run_mcp_call
from video_knowledge_pipeline.multimodal_sample_review import multimodal_sample_review, validate_multimodal_sample_notes
from video_knowledge_pipeline.task_console import export_task_console
from video_knowledge_pipeline.transcript_sidecar import ensure_review_transcript_sidecar
from video_knowledge_pipeline.timeline_alignment_audit import timeline_alignment_audit


def _write_bundle(bundle: Path) -> None:
    bundle.mkdir(parents=True)
    (bundle / "exports").mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "sample review lesson",
                "review_html": "review.html",
                "sources": [{"path": str(bundle / "lesson.mp4"), "title": "lesson"}],
                "content_assets": {
                    "content_candidate_pack_path": "exports/content-candidate-pack.json",
                    "content_candidate_pack_markdown_path": "exports/content-candidate-pack.md",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 5,
            "transcript": "老师说看屏幕上的工具名称。",
            "visual_route": "semantic_frame",
            "visual_text": "Browser Use",
            "structured_visual": {"title": "工具对比"},
            "visual_understanding": {"objects": ["浏览器界面"], "actions": ["展示工具排名"]},
            "frame_paths": [str(bundle / "frames" / "0001.jpg")],
            "tagger_tags": ["工具名"],
        },
        {
            "index": 2,
            "start": 5,
            "end": 10,
            "midpoint": 7.5,
            "transcript": "这里演示点击按钮。",
            "visual_route": "semantic_frame",
            "visual_understanding": {"objects": ["网页按钮"], "actions": ["点击"], "evidence_frame_paths": [str(bundle / "frames" / "0002-nested.jpg")]},
            "frame_paths": [str(bundle / "frames" / "0002.jpg")],
            "tagger_time_axis": [{"time": 6.2, "source": "qinglong"}],
            "tagger_annotations": [{"time": 6.2, "tags": ["操作", "步骤"], "text": "点击按钮开始演示"}],
        },
        {
            "index": 3,
            "start": 10,
            "end": 20,
            "transcript": "连续演示流程。",
            "visual_route": "temporal_sequence",
            "temporal_visual_understanding": {"events": ["打开页面", "输入关键词", "查看结果"]},
            "temporal_frame_paths": [str(bundle / "frames" / "0003-a.jpg"), str(bundle / "frames" / "0003-b.jpg")],
        },
        {
            "index": 4,
            "start": 20,
            "end": 30,
            "transcript": "这个名字可能识别错。",
            "visual_route": "semantic_frame",
            "quality_issues": ["semantic_frame_without_analysis"],
            "frame_paths": [str(bundle / "frames" / "0004.jpg")],
        },
    ]
    timeline.append(
        {
            "index": 5,
            "start": 30,
            "end": 40,
            "transcript": "这段可以作为内容素材。",
            "visual_route": "semantic_frame",
            "visual_text": "素材候选",
            "visual_understanding": {"summary": "展示一个可复用观点。"},
            "frame_paths": [str(bundle / "frames" / "0005.jpg")],
        }
    )
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    (bundle / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 4.25, "end": 6.0, "text": "这里演示点击按钮。"},
                    {"start": 10.0, "end": 20.0, "text": "连续演示流程。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "review.html").write_text("<html>review</html>", encoding="utf-8")
    (bundle / "exports" / "multimodal-effect-comparison-report.json").write_text(
        json.dumps({"schema": "video_knowledge_pipeline.multimodal_effect_comparison.v1", "example_indexes": [2]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "exports" / "content-candidate-pack.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.content_candidate_pack.v1",
                "candidate_count": 1,
                "review_required": True,
                "publication_allowed": False,
                "allowed_as_fact": False,
                "allowed_as_inspiration": True,
                "candidates": [
                    {
                        "id": "candidate-5",
                        "timeline_index": 5,
                        "candidate_types": ["viewpoint", "short_video_script"],
                        "viewpoint": "把这段观点作为短内容开头。",
                        "case_or_example": "老师举了一个可复用表达。",
                        "reusable_quote": "这段可以作为内容素材。",
                        "fact_check_status": "needs_review",
                        "evidence_paths": [str(bundle / "frames" / "0005.jpg")],
                        "citation_digest_status": "ready",
                        "evidence_citations": [
                            {
                                "source_type": "transcript",
                                "time": "00:00:20.000 - 00:00:25.000",
                                "timeline_indexes": [5],
                                "text": "内容素材候选来自课程观点和画面证据。",
                                "evidence_paths": [str(bundle / "frames" / "0005.jpg")],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "exports" / "content-candidate-pack.md").write_text("# 内容素材候选\n", encoding="utf-8")


def test_multimodal_sample_review_writes_static_ui_and_notes_template(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    result = multimodal_sample_review(bundle, sample_size=5, write=True)

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    notes = json.loads((bundle / "multimodal-sample-review.todo.json").read_text(encoding="utf-8"))
    html = (bundle / "multimodal-sample-review.html").read_text(encoding="utf-8")
    indexes = [row["index"] for row in result["samples"]]
    sample_types = {row["sample_type"] for row in result["samples"]}

    assert result["schema"] == "video_knowledge_pipeline.multimodal_sample_review.v1"
    assert result["comparison_loaded"] is True
    assert result["run_artifact"]["run_type"] == "multimodal_sample_review"
    assert result["run_artifact"]["status"] == "needs_input"
    assert result["run_artifact"]["failed_items"][0]["reason"] == "human_sample_label_required"
    sample_run = json.loads((bundle / "runs" / "multimodal-sample-review" / "run.json").read_text(encoding="utf-8"))
    assert sample_run["status"] == "needs_input"
    assert sample_run["failed_items"][0]["suggested_next_tool"] == "validate_multimodal_sample_notes"
    assert 2 in indexes
    sample_by_index = {row["index"]: row for row in result["samples"]}
    assert any(path.endswith("0002-nested.jpg") for path in sample_by_index[2]["frame_paths"])
    assert sample_by_index[2]["start"] == 5
    assert sample_by_index[2]["review_start"] == 4.25
    assert sample_by_index[2]["review_start_source"] == "asr_segment_start"
    assert sample_by_index[2]["video_review"]["start_seconds"] == 4.25
    assert sample_by_index[2]["video_review"]["segment_start_seconds"] == 5
    assert sample_by_index[2]["video_review"]["frame_time_seconds"] == 7.5
    assert {"content_candidate", "comparison_example", "visual_with_ocr", "temporal", "missing_visual"} <= sample_types
    assert result["content_candidate_pack"]["candidate_count"] == 1
    assert result["counts"]["content_candidate_sample_count"] == 1
    assert sample_by_index[5]["sample_type"] == "content_candidate"
    assert sample_by_index[5]["content_candidate_id"] == "candidate-5"
    assert "short_video_script" in sample_by_index[5]["content_candidate_types"]
    assert sample_by_index[5]["content_candidate_viewpoint"] == "把这段观点作为短内容开头。"
    assert sample_by_index[5]["content_candidate_citation_summary"].startswith("00:00:20.000 - 00:00:25.000 transcript")
    assert sample_by_index[5]["content_candidate_evidence_citations"][0]["source_type"] == "transcript"
    assert "内容素材候选来自课程观点" in sample_by_index[5]["content_candidate_evidence_citations"][0]["text"]
    assert any(path.endswith("0005.jpg") for path in sample_by_index[5]["frame_paths"])
    assert notes["schema"] == "video_knowledge_pipeline.multimodal_sample_review_notes.v1"
    assert len(notes["reviews"]) == len(result["samples"])
    assert notes["reviews"][0]["overall_label"] == ""
    assert notes["reviews"][0]["video_checked"] == ""
    assert notes["reviews"][0]["term_accuracy"] == ""
    assert notes["reviews"][0]["visual_fact_accuracy"] == ""
    assert notes["reviews"][0]["step_completeness"] == ""
    assert notes["reviews"][0]["timestamp_accuracy"] == ""
    assert notes["reviews"][0]["keep_image_required"] == ""
    assert notes["reviews"][0]["content_candidate_usable"] == ""
    assert notes["reviews"][0]["content_candidate_evidence_sufficient"] == ""
    assert result["media"]["path"].endswith("lesson.mp4")
    assert (bundle / "potplayer-jump.ps1").exists()
    assert (bundle / "potplayer-review-playlist.m3u8").exists()
    assert (bundle / "potplayer-review-chapters.txt").exists()
    assert (bundle / "potplayer-review-timestamps.csv").exists()
    assert (bundle / "potplayer-review-timestamps.md").exists()
    playlist = (bundle / "potplayer-review-playlist.m3u8").read_text(encoding="utf-8-sig")
    chapters = (bundle / "potplayer-review-chapters.txt").read_text(encoding="utf-8-sig")
    timestamp_md = (bundle / "potplayer-review-timestamps.md").read_text(encoding="utf-8")
    assert "#EXTM3U" in playlist
    assert "#EXTVLCOPT:start-time=4.25" in playlist
    assert "CHAPTER01=" in chapters
    assert "PotPlayer 待审核时间戳清单" in timestamp_md
    assert "Citation summary" in html
    assert "内容素材候选来自课程观点" in html
    assert sample_by_index[2]["potplayer_command"].startswith("powershell -NoProfile")
    assert "-Seconds 4.25" in sample_by_index[2]["potplayer_command"]
    assert "多模态抽样标注" in html
    assert "生成标注 JSON" in html
    assert "reviewVideo" in html
    assert "选择本地视频文件" in html
    assert "review-queue" in html
    assert "queue-2" in html
    assert "待审核时间戳队列" in html
    assert "workbench" in html
    assert "currentSamplePanel" in html
    assert "当前审核条目" in html
    assert "&quot;result&quot;" not in html
    assert "按时间轴打开" in html
    assert "seekSample(2, 4.25" in html
    assert "file:///" in html
    assert "复制 PotPlayer 命令" in html
    assert "内容素材候选" in html
    assert "素材候选可继续加工" in html
    assert "candidateusable-5" in html
    assert "candidateevidence-5" in html
    assert "review.html" in html
    assert manifest["multimodal_sample_review_html"] == "multimodal-sample-review.html"
    assert manifest["potplayer_jump_script"] == "potplayer-jump.ps1"
    assert manifest["potplayer_review_playlist"] == "potplayer-review-playlist.m3u8"
    assert manifest["potplayer_review_timestamps_markdown"] == "potplayer-review-timestamps.md"
    assert manifest["mcp_multimodal_sample_review_args"] == "mcp-multimodal-sample-review.args.json"
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    assert any(row["run_type"] == "multimodal_sample_review" for row in registry["runs"])

    mcp_result = run_mcp_call("multimodal_sample_review", bundle / "mcp-multimodal-sample-review.args.json")
    assert mcp_result["outputs"]["html"].endswith("multimodal-sample-review.html")


def test_task_console_links_multimodal_sample_review(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    multimodal_sample_review(bundle, sample_size=5, write=True)
    validate_multimodal_sample_notes(bundle, min_reviewed=1, write=True)

    console = export_task_console(bundle, write=True, refresh=False)
    command_keys = {row["key"] for row in console["commands"]}
    artifact_keys = {row["key"] for row in console["artifacts"]}
    html = (bundle / "task-console.html").read_text(encoding="utf-8")

    assert "multimodal_sample_review" in command_keys
    assert "multimodal_sample_review_html" in artifact_keys
    assert "multimodal_sample_review_summary_report" in artifact_keys
    assert "human_sample_eval_report" in artifact_keys
    assert "multimodal-sample-review" in html
    assert "human-sample-eval" in html


def test_multimodal_sample_review_cli_parser() -> None:
    args = build_parser().parse_args(["multimodal-sample-review", "bundle", "--sample-size", "12", "--no-missing", "--media-path", "video.mp4", "--potplayer-path", "PotPlayerMini64.exe"])

    assert args.command == "multimodal-sample-review"
    assert args.sample_size == 12
    assert args.no_missing is True
    assert args.media_path == "video.mp4"
    assert args.potplayer_path == "PotPlayerMini64.exe"


def test_validate_multimodal_sample_notes_summarizes_human_labels(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    multimodal_sample_review(bundle, sample_size=5, write=True)
    notes = json.loads((bundle / "multimodal-sample-review.todo.json").read_text(encoding="utf-8"))
    notes["reviews"][0].update(
        {
            "asr_correct": "yes",
            "ocr_correct": "not_applicable",
            "multimodal_added_key_info": "yes",
            "multimodal_error_or_hallucination": "no",
            "final_note_sufficient": "yes",
            "overall_label": "correct",
            "term_accuracy": "yes",
            "visual_fact_accuracy": "yes",
            "step_completeness": "yes",
            "timestamp_accuracy": "yes",
            "keep_image_required": "no",
            "human_notes": "多模态补到了按钮状态。",
        }
    )
    notes["reviews"][1].update(
        {
            "asr_correct": "partial",
            "ocr_correct": "no",
            "multimodal_added_key_info": "some",
            "multimodal_error_or_hallucination": "minor",
            "final_note_sufficient": "partial",
            "overall_label": "partial",
            "term_accuracy": "partial",
            "visual_fact_accuracy": "partial",
            "step_completeness": "partial",
            "timestamp_accuracy": "no",
            "keep_image_required": "yes",
        }
    )
    candidate_review = next(row for row in notes["reviews"] if row["sample_type"] == "content_candidate")
    candidate_review.update(
        {
            "content_candidate_usable": "yes",
            "content_candidate_evidence_sufficient": "partial",
        }
    )
    notes_path = bundle / "multimodal-sample-review-notes.json"
    notes_path.write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")

    result = validate_multimodal_sample_notes(bundle, notes_json=notes_path, min_reviewed=2, write=True)

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    report = (bundle / "multimodal-sample-review-summary.md").read_text(encoding="utf-8")
    human_eval = json.loads((bundle / "human-sample-eval.json").read_text(encoding="utf-8"))
    human_eval_report = (bundle / "human-sample-eval.md").read_text(encoding="utf-8")
    assert result["schema"] == "video_knowledge_pipeline.multimodal_sample_review_summary.v1"
    assert result["status"] == "ready"
    assert result["run_artifact"]["run_type"] == "human_sample_eval"
    assert result["run_artifact"]["status"] == "completed"
    eval_run = json.loads((bundle / "runs" / "human-sample-eval" / "run.json").read_text(encoding="utf-8"))
    assert eval_run["status"] == "completed"
    assert eval_run["failed_items"] == []
    assert result["summary"]["labeled_rows"] == 2
    assert result["summary"]["rates"]["multimodal_added_key_info_rate"] == 100.0
    assert result["summary"]["rates"]["any_hallucination_rate"] == 50.0
    assert result["summary"]["rates"]["term_accuracy_accept_rate"] == 100.0
    assert result["summary"]["rates"]["timestamp_accuracy_accept_rate"] == 50.0
    assert result["summary"]["rates"]["keep_image_required_rate"] == 50.0
    assert result["summary"]["rates"]["content_candidate_usable_rate"] == 100.0
    assert result["summary"]["rates"]["content_candidate_evidence_sufficient_rate"] == 100.0
    assert human_eval["schema"] == "video_knowledge_pipeline.human_sample_eval.v1"
    assert human_eval["rates"]["human_sampled_multimodal_net_help_rate"] == 50.0
    assert human_eval["rates"]["content_candidate_usable_rate"] == 100.0
    assert human_eval["rates"]["content_candidate_evidence_sufficient_rate"] == 100.0
    assert "人工抽样质量评估" in human_eval_report
    assert "内容素材候选可用率" in human_eval_report
    assert "多模态抽样标注汇总" in report
    assert manifest["multimodal_sample_review_summary_report"] == "multimodal-sample-review-summary.md"
    assert manifest["human_sample_eval_json"] == "human-sample-eval.json"
    assert manifest["human_sample_eval_report"] == "human-sample-eval.md"
    assert manifest["mcp_validate_multimodal_sample_notes_args"] == "mcp-validate-multimodal-sample-notes.args.json"
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    by_type = {row["run_type"]: row for row in registry["runs"]}
    assert by_type["human_sample_eval"]["status"] == "completed"
    assert by_type["multimodal_sample_review"]["status"] == "needs_input"

    mcp_result = run_mcp_call("validate_multimodal_sample_notes", bundle / "mcp-validate-multimodal-sample-notes.args.json")
    assert mcp_result["status"] == "ready"


def test_validate_multimodal_sample_notes_reports_invalid_rows(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    multimodal_sample_review(bundle, sample_size=5, write=True)
    bad_notes = {
        "schema": "video_knowledge_pipeline.multimodal_sample_review_notes.v1",
        "reviews": [
            {"index": 999, "overall_label": "correct"},
            {"index": 1, "overall_label": "made_up"},
        ],
    }
    notes_path = bundle / "multimodal-sample-review-notes.json"
    notes_path.write_text(json.dumps(bad_notes, ensure_ascii=False), encoding="utf-8")

    result = validate_multimodal_sample_notes(bundle, notes_json=notes_path, min_reviewed=1, write=False)

    assert result["status"] == "invalid"
    keys = {issue["key"] for issue in result["issues"]}
    assert "invalid_review_row" in keys
    assert "missing_review_row" in keys


def test_validate_multimodal_sample_notes_cli_parser() -> None:
    args = build_parser().parse_args(["validate-multimodal-sample-notes", "bundle", "--notes-json", "notes.json", "--min-reviewed", "6"])

    assert args.command == "validate-multimodal-sample-notes"
    assert args.notes_json == "notes.json"
    assert args.min_reviewed == 6










def test_timeline_alignment_audit_flags_review_start_and_tagger_conflicts(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-alignment"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1", "normalized_transcript_json": "normalized-transcript.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 4.25, "end": 8.0, "text": "这里开始讲第一个工具。"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 4.0,
                    "end": 9.0,
                    "midpoint": 6.5,
                    "review_start": 9.5,
                    "review_start_source": "frame_time",
                    "tagger_time_axis": [{"time": 20, "source": "qinglong"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = timeline_alignment_audit(bundle, tolerance_seconds=2)

    assert result["summary"]["review_start_mismatch"] == 1
    assert result["summary"]["tagger_time_conflict"] == 1
    assert result["items"][0]["asr_first_start"] == 4.25
    assert (bundle / "timeline-alignment-audit.md").exists()
    assert (bundle / "mcp-timeline-alignment-audit.args.json").exists()
