from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.adaptive_asr_route import build_adaptive_asr_route
from video_knowledge_pipeline.cli import build_parser


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"title": "线上陌生成交转化"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 6,
                    "visual_text": "# 高频问题的处理技巧\n\n客户不回复 / 不投保",
                    "structured_visual": [
                        {
                            "source": "ebook_markdown_pipeline",
                            "markdown": "明亚保险经纪 领航计划",
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"video")
    return bundle, media


def test_adaptive_asr_cli_exposes_filtered_pre_asr_route() -> None:
    args = build_parser().parse_args(
        [
            "adaptive-asr-route",
            "bundle",
            "lecture.mp4",
            "--task-profile",
            "terminology",
            "--base-lexicon-json",
            "industry.json",
            "--include-online-plan",
        ]
    )

    assert args.command == "adaptive-asr-route"
    assert args.task_profile == "terminology"
    assert args.base_lexicon_json == "industry.json"
    assert args.include_online_plan is True


def test_route_consumes_ocr_terms_without_replacing_transcript(tmp_path: Path) -> None:
    bundle, media = _bundle(tmp_path)
    calls: list[dict] = []

    def local_plan_builder(workspace, media_path, **kwargs):
        calls.append({"workspace": str(workspace), "media": str(media_path), **kwargs})
        plan = Path(workspace) / "plan.json"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text("{}", encoding="utf-8")
        return {
            "plan_path": str(plan),
            "available": True,
            "model_ready": {"ready": True},
            "runner": "test",
        }

    result = build_adaptive_asr_route(
        bundle,
        media,
        task_profile="accuracy",
        local_plan_builder=local_plan_builder,
    )

    assert result["ocr_policy"]["default"] == "ebook_markdown_pipeline"
    assert result["ocr_policy"]["selectable_backends"] == ["ebook_markdown_pipeline", "online_ocr"]
    assert result["context"]["ocr_input_contract"]["direct_transcript_rewrite"] is False
    assert result["transcript_layers"]["promotion_requires_quality_gate"] is True
    assert {call["preset"] for call in calls} == {"contextual-paraformer", "sensevoice"}
    assert all(call["use_itn"] is True for call in calls)
    assert result["context"]["phase"] == "pre_asr"
    assert result["context"]["post_asr_terms_do_not_trigger_rerun"] is True
    assert (bundle / "entity-hotwords.pre-asr.txt").is_file()
    assert (bundle / "entity-hotword-audit.pre-asr.json").is_file()
    assert (bundle / "adaptive-asr-route.json").is_file()


def test_online_asr_branch_is_plan_only(tmp_path: Path) -> None:
    bundle, media = _bundle(tmp_path)
    cloud_kwargs: dict = {}

    def local_plan_builder(workspace, media_path, **kwargs):
        return {"plan_path": str(Path(workspace) / "plan.json"), "available": True}

    def cloud_plan_builder(workspace, media_path, **kwargs):
        cloud_kwargs.update(kwargs)
        assert "api_key" not in kwargs["provider_config"]
        return {
            "plan_path": str(Path(workspace) / "cloud-plan.json"),
            "provider_config": {"provider": "fixture"},
            "request_plan": {"interface": "openai_audio_transcriptions"},
        }

    result = build_adaptive_asr_route(
        bundle,
        media,
        include_online_plan=True,
        provider_config={"provider": "fixture"},
        local_plan_builder=local_plan_builder,
        cloud_plan_builder=cloud_plan_builder,
    )

    assert result["online_plan"]["status"] == "planned"
    assert result["online_plan"]["execute"] is False
    assert result["execution_boundary"]["media_uploaded"] is False
    assert result["routing_decision"]["depends_on_current_agent_model"] is False
    assert result["context"]["source"] == "entity_lexicon_pre_asr"
    assert result["context"]["hotword_audit_json"].endswith("entity-hotword-audit.pre-asr.json")
    assert "明亚" in cloud_kwargs["prompt"]
    assert "mistral-ocr-4-0" not in cloud_kwargs["prompt"]
