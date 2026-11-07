from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.transcript_postprocess import postprocess_asr_transcript
from video_knowledge_pipeline.transcript_translation import (
    translate_transcript_to_mandarin,
)


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    source = root / "normalized-transcript.json"
    source.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "s-1",
                        "start": 0.0,
                        "end": 2.0,
                        "text": "你想点样采访呀，",
                        "speaker": "0",
                    },
                    {
                        "id": "s-2",
                        "start": 2.1,
                        "end": 4.0,
                        "text": "我问你啲问题。",
                        "speaker": "1",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"normalized_transcript_json": source.name}, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def test_preserve_punctuation_does_not_append_full_stop_after_comma(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    postprocess_asr_transcript(root, punctuation_mode="preserve")
    data = json.loads((root / "corrected-transcript.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["text"] == "你想点样采访呀，"
    assert "，。" not in data["segments"][0]["text"]


def test_local_translation_preserves_lineage_and_writes_derived_srt(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)

    def fake_runtime(task: str, **kwargs):
        assert task == "text_llm"
        assert kwargs["execution_location"] == "local"
        rows = json.loads(kwargs["text"])
        translations = []
        target = {0: "你想怎样采访？", 1: "我问你一些问题。"}
        for row in rows:
            translations.append(
                {
                    "index": row["index"],
                    "segment_id": row["segment_id"],
                    "text": target[row["index"]],
                }
            )
        return {
            "ok": True,
            "status": "completed",
            "content": json.dumps({"translations": translations}, ensure_ascii=False),
            "latency_ms": 8,
        }

    result = translate_transcript_to_mandarin(
        root, execute=True, runtime_call=fake_runtime
    )
    assert result["status"] == "completed"
    payload = json.loads(
        (root / "mandarin-translated-transcript.json").read_text(encoding="utf-8")
    )
    assert [
        (row["segment_id"], row["start"], row["end"], row["speaker"])
        for row in payload["segments"]
    ] == [
        ("s-1", 0.0, 2.0, "0"),
        ("s-2", 2.1, 4.0, "1"),
    ]
    srt = (root / "mandarin-translated-subtitles.srt").read_text(encoding="utf-8")
    assert "说话人1：你想怎样采访？" in srt
    assert "说话人2：我问你一些问题。" in srt
    source = json.loads(
        (root / "normalized-transcript.json").read_text(encoding="utf-8")
    )
    assert source["segments"][0]["text"] == "你想点样采访呀，"


def test_explicit_direct_lmstudio_uses_fixed_loopback_without_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    root = _bundle(tmp_path)
    calls = []

    def fake_local_call(*, provider_config, messages, temperature, max_tokens):
        calls.append(provider_config)
        rows = json.loads(messages[-1]["content"])
        return {
            "ok": True,
            "content": json.dumps(
                {
                    "translations": [
                        {
                            "index": row["index"],
                            "segment_id": row["segment_id"],
                            "text": f"普通话{row['index']}",
                        }
                        for row in rows
                    ]
                },
                ensure_ascii=False,
            ),
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.transcript_translation.call_openai_compatible_text",
        fake_local_call,
    )
    result = translate_transcript_to_mandarin(root, execute=True, direct_lmstudio=True)
    assert result["status"] == "completed"
    assert calls == [
        {
            "provider": "local_vlm",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "qwen/qwen3.5-9b",
            "timeout_seconds": 300,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    ]


def test_translation_fails_closed_on_lineage_mismatch(tmp_path: Path) -> None:
    root = _bundle(tmp_path)

    def fake_runtime(task: str, **kwargs):
        rows = json.loads(kwargs["text"])
        return {
            "ok": True,
            "status": "completed",
            "content": json.dumps(
                {
                    "translations": [
                        {"index": row["index"], "segment_id": "wrong", "text": "测试"}
                        for row in rows
                    ]
                },
                ensure_ascii=False,
            ),
        }

    result = translate_transcript_to_mandarin(
        root, execute=True, runtime_call=fake_runtime
    )
    assert result["status"] == "degraded"
    assert result["ok"] is False
    assert result["artifacts"]["translated_srt"] == ""
    assert not (root / "mandarin-translated-subtitles.srt").exists()


def test_imported_translation_requires_source_and_segment_binding(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    source = root / "normalized-transcript.json"
    imported = root / "reviewed-translation.json"
    imported.write_text(
        json.dumps(
            {
                "source_sha256": __import__("hashlib")
                .sha256(source.read_bytes())
                .hexdigest(),
                "translations": [
                    {
                        "index": 0,
                        "segment_id": "wrong-segment",
                        "text": "你想怎样采访？",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    try:
        translate_transcript_to_mandarin(root, input_json=imported)
    except ValueError as exc:
        assert "lineage mismatch" in str(exc)
    else:
        raise AssertionError("lineage-drifted import must fail closed")


def test_translation_resumes_validated_checkpoint_without_repeat_calls(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    source_sha = (
        __import__("hashlib")
        .sha256((root / "normalized-transcript.json").read_bytes())
        .hexdigest()
    )
    (root / "mandarin-subtitle-translation-checkpoint.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_translation_checkpoint.v1",
                "source_sha256": source_sha,
                "translations": [
                    {"index": 0, "segment_id": "s-1", "text": "你想怎样采访？"}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    seen = []

    def fake_runtime(task: str, **kwargs):
        rows = json.loads(kwargs["text"])
        seen.extend(row["index"] for row in rows)
        return {
            "ok": True,
            "status": "completed",
            "content": json.dumps(
                {
                    "translations": [
                        {"index": 1, "segment_id": "s-2", "text": "我问你一些问题。"}
                    ]
                },
                ensure_ascii=False,
            ),
        }

    result = translate_transcript_to_mandarin(
        root, execute=True, runtime_call=fake_runtime
    )
    assert result["status"] == "completed"
    assert seen == [1]


def test_translation_rejects_checkpoint_with_segment_lineage_drift(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    source_sha = (
        __import__("hashlib")
        .sha256((root / "normalized-transcript.json").read_bytes())
        .hexdigest()
    )
    (root / "mandarin-subtitle-translation-checkpoint.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_translation_checkpoint.v1",
                "source_sha256": source_sha,
                "translations": [
                    {"index": 0, "segment_id": "wrong-segment", "text": "错误缓存"}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    seen = []

    def fake_runtime(task: str, **kwargs):
        rows = json.loads(kwargs["text"])
        seen.extend(row["index"] for row in rows)
        return {
            "ok": True,
            "status": "completed",
            "content": json.dumps(
                {
                    "translations": [
                        {
                            "index": row["index"],
                            "segment_id": row["segment_id"],
                            "text": f"普通话{row['index']}",
                        }
                        for row in rows
                    ]
                },
                ensure_ascii=False,
            ),
        }

    result = translate_transcript_to_mandarin(
        root, execute=True, runtime_call=fake_runtime
    )
    assert result["status"] == "completed"
    assert seen == [0, 1]
