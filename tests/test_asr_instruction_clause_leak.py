from __future__ import annotations

from video_knowledge_pipeline.asr_response_quality import assess_asr_response


def test_instruction_clause_embedded_in_long_segment_is_blocking() -> None:
    quality = assess_asr_response(
        {
            "segments": [
                {
                    "id": "40",
                    "start": 608.446,
                    "end": 649.409,
                    "text": (
                        "这一段前面包含真实讲课内容，但结尾错误出现了"
                        "中间字段可用英文字幕。"
                    ),
                    "avg_logprob": -0.96,
                }
            ]
        },
        task_instructions=(
            "请逐字转写整段中文知识视频音频，保留分段时间戳；"
            "不要做语义改写。中间字段可用英文，识别正文必须为中文。"
        ),
    )

    assert quality["status"] == "degraded"
    assert quality["failed_segment_count"] == 1
    assert "task_instruction_leak" in quality["failed_chunks"][0]["reasons"]
    assert quality["retry_plan"]["windows"][0]["start"] == 606.946
    assert quality["retry_plan"]["windows"][0]["end"] == 650.909
