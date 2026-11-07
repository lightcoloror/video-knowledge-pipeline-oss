from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.transcript_stability_evaluation import (
    _levenshtein,
    build_transcript_reference_binding,
    evaluate_transcript_stability,
    evaluate_transcript_files,
    extract_logseq_original_transcript,
    main as transcript_evaluation_main,
)


def test_evaluation_reports_normalized_reference_distance_not_cer() -> None:
    reference = {
        "segments": [
            {
                "start": 0,
                "end": 10,
                "text": "这是完整的课程逐字稿，共有二十个字用于稳定测试。",
            }
        ]
    }
    candidate = {
        "segments": [
            {
                "start": 0,
                "end": 10,
                "text": "这是完整的课程逐字稿，共有二十个字用于稳定测试",
            }
        ]
    }

    result = evaluate_transcript_stability(candidate, reference)

    assert result["status"] == "passed"
    assert result["metric"]["name"] == "normalized_reference_edit_distance"
    assert result["metric"]["not_character_error_rate"] is True
    assert result["reference_must_not_enter_prompt_hotwords_or_routing"] is True


def test_evaluation_proves_when_length_gap_alone_blocks_target() -> None:
    reference = {"segments": [{"text": "a" * 100}]}
    candidate = {"segments": [{"text": "a" * 94}]}

    result = evaluate_transcript_stability(candidate, reference)
    lower_bound = result["distance_lower_bound"]

    assert result["status"] == "failed"
    assert lower_bound["normalized_character_length_delta"] == 6
    assert lower_bound["normalized_character_deficit"] == 6
    assert lower_bound["relative_length_deficit"] == 0.06
    assert lower_bound["maximum_integer_distance_that_passes"] == 4
    assert lower_bound["minimum_edit_distance_reduction_required"] == 2
    assert lower_bound["minimum_length_gap_reduction_required"] == 2
    assert lower_bound["content_recovery_required"] is True
    assert result["evaluation_state"] == "asr_quality_distance_exceeded"
    assert result["diagnostic_statuses"] == ["asr_quality_distance_exceeded"]
    assert result["completion"]["possible_long_form_loss"] is False


def test_evaluation_blocks_prompt_leak_and_long_form_loss() -> None:
    reference = {
        "duration": 120,
        "segments": [{"text": "完整讲课正文" * 100}],
    }
    candidate = {
        "duration": 30,
        "segments": [{"text": "请逐字转写整段中文知识视频音频。"}],
    }

    result = evaluate_transcript_stability(
        candidate,
        reference,
        task_instructions="请逐字转写整段中文知识视频音频，保留时间戳。",
    )

    assert result["status"] == "failed"
    assert result["prompt_leak"]["passed"] is False
    assert result["completion"]["possible_long_form_loss"] is True
    assert result["gates"]["normalized_reference_edit_distance"] is False
    assert result["evaluation_state"] == "possible_long_form_loss"
    assert "possible_long_form_loss" in result["diagnostic_statuses"]
    assert "asr_quality_distance_exceeded" in result["diagnostic_statuses"]


def test_bitparallel_levenshtein_is_exact_for_known_unicode_cases() -> None:
    cases = [
        ("", "", 0),
        ("", "明亚", 2),
        ("kitten", "sitting", 3),
        ("明亚APP", "MIAAPP", 3),
        ("保险经纪", "保险经理", 1),
        ("转录稿", "逐字稿", 2),
    ]

    for left, right, expected in cases:
        assert _levenshtein(left, right) == expected


def test_extract_logseq_original_transcript_stops_before_attachments() -> None:
    markdown = """- AI 摘要
	- 这不是参考正文
- 原始转录
	- 🟢 说话人1 [00:00:00]
		- 第一段正文。
	- 🟢 说话人1 [00:00:08]
		- 第二段正文。
- 附件与证据
	- https://example.invalid/audio?token=secret-token
- 关联
	- 不应读取
"""

    result = extract_logseq_original_transcript(markdown)

    assert [segment["text"] for segment in result["segments"]] == [
        "第一段正文。",
        "第二段正文。",
    ]
    assert result["segments"][0]["end"] == 8.0
    assert result["duration_seconds"] == 8.0
    assert "secret-token" not in json.dumps(result, ensure_ascii=False)
    assert "这不是参考正文" not in json.dumps(result, ensure_ascii=False)


def test_file_evaluation_reports_hashes_without_reference_content(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.md"
    candidate_path.write_text(
        json.dumps(
            {
                "duration": 12,
                "segments": [
                    {"start": 0, "end": 8, "text": "第一段正文"},
                    {"start": 8, "end": 12, "text": "第二段正文"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reference_path.write_text(
        """- 原始转录
	- 🟢 说话人1 [00:00:00]
		- 第一段正文。
	- 🟢 说话人1 [00:00:08]
		- 第二段正文。
- 附件与证据
	- https://example.invalid/audio?token=do-not-copy
""",
        encoding="utf-8",
    )

    strict = evaluate_transcript_files(candidate_path, reference_path)
    assert strict["evaluation_state"] == "reference_binding_invalid"
    assert strict["completion"]["possible_long_form_loss"] is False

    result = evaluate_transcript_files(
        candidate_path,
        reference_path,
        require_reference_binding=False,
    )
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["status"] == "passed"
    assert result["inputs"]["reference"]["role"] == "evaluation_only_reference"
    assert (
        result["inputs"]["reference"]["must_not_enter_prompt_hotwords_or_routing"]
        is True
    )
    assert (
        result["inputs"]["reference"]["format"] == "logseq_markdown_original_transcript"
    )
    assert len(result["inputs"]["reference"]["sha256"]) == 64
    assert "第一段正文" not in serialized
    assert "do-not-copy" not in serialized
    windows = result["comparison_windows"]
    assert windows["status"] == "available"
    assert windows["content_included"] is False
    assert windows["window_count"] == 2
    assert "正文" not in json.dumps(windows, ensure_ascii=False)


def test_logseq_reference_requires_explicit_original_transcript_block() -> None:
    try:
        extract_logseq_original_transcript("- AI 摘要\n\t- 只有总结")
    except ValueError as exc:
        assert "原始转录" in str(exc)
    else:
        raise AssertionError("missing 原始转录 block must be rejected")

def test_content_vocal_filler_profile_is_explicit_and_preserves_surface_metric() -> None:
    reference = {"segments": [{"text": "啊呃嗯哎哦啊呃嗯哎哦正文"}]}
    candidate = {"segments": [{"text": "正文"}]}

    strict = evaluate_transcript_stability(candidate, reference)
    content = evaluate_transcript_stability(
        candidate,
        reference,
        normalization_profile="content_vocal_fillers_v1",
    )

    assert strict["status"] == "failed"
    assert content["status"] == "passed"
    assert content["metric"]["value"] == 0.0
    assert content["metric"]["normalization_profile"] == "content_vocal_fillers_v1"
    assert content["surface_metric"]["normalization_profile"] == "strict_v1"
    assert content["surface_metric"]["value"] > 0.0
    assert content["normalization"]["symmetric"] is True
    assert content["normalization"]["removed_vocal_fillers"] == [
        "啊",
        "呃",
        "嗯",
        "哎",
        "哦",
    ]
    assert content["reference_must_not_enter_prompt_hotwords_or_routing"] is True


def test_exact_reference_binding_accepts_matching_media_reference_and_topic(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "团财高绩.mp4"
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.md"
    binding_path = tmp_path / "reference-binding.json"
    media_path.write_bytes(b"fixed-media-fixture")
    candidate_path.write_text(
        json.dumps(
            {
                "duration_seconds": 1800,
                "segments": [
                    {
                        "start": 0,
                        "end": 1800,
                        "text": "今天分享团财业务和增员路径，先介绍团队发展，再介绍客户经营。最后总结团财业务和增员路径。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reference_path.write_text(
        """title:: 团财业务与增员思路
getnote-id:: 1914922224168076256
- 摘要
	- **时长**：约 0小时 30分钟
- 原始转录
	- 🟢 说话人1 [00:00:00]
		- 今天分享团财业务和增员路径，先介绍团队发展，再介绍客户经营。
	- 🟢 说话人1 [00:29:40]
		- 最后总结团财业务和增员路径。
""",
        encoding="utf-8",
    )

    binding = build_transcript_reference_binding(
        media_path,
        reference_path,
        candidate_path=candidate_path,
        media_duration_seconds=1800,
        topic_anchors=["团财", "增员"],
    )
    binding_path.write_text(
        json.dumps(binding, ensure_ascii=False),
        encoding="utf-8",
    )
    result = evaluate_transcript_files(
        candidate_path,
        reference_path,
        reference_binding=binding_path,
        media_path=media_path,
        media_duration_seconds=1800,
        require_reference_binding=True,
    )

    assert result["reference_binding"]["status"] == "valid"
    assert result["diagnostic_statuses"] == []
    assert result["status"] == "passed"


def test_reference_binding_prefers_timed_transcript_over_rounded_duration_label(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "课程.mp4"
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.md"
    media_path.write_bytes(b"fixed-media")
    candidate_path.write_text(
        json.dumps(
            {
                "duration_seconds": 834,
                "segments": [
                    {"start": 0, "end": 813, "text": "方案设计和客户沟通课程正文"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reference_path.write_text(
        """title:: 方案设计和客户沟通
getnote-id:: exact-rounded-duration
- 摘要
    - **时长**：约 0小时 13分钟
- 原始转录
    - 🟢 说话人1 [00:00:00]
        - 方案设计和客户沟通课程正文。
    - 🟢 说话人1 [00:13:33]
        - 课程结束。
""",
        encoding="utf-8",
    )

    binding = build_transcript_reference_binding(
        media_path,
        reference_path,
        candidate_path=candidate_path,
        media_duration_seconds=834,
        topic_anchors=["方案设计"],
    )

    assert binding["status"] == "active"
    assert binding["reference"]["duration_seconds"] == 813.0


def test_reference_duration_mismatch_is_not_reported_as_asr_loss(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "团财高绩.mp4"
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "wrong-reference.md"
    media_path.write_bytes(b"fixed-media-fixture")
    candidate_path.write_text(
        json.dumps(
            {
                "duration_seconds": 1800,
                "segments": [{"start": 0, "end": 1800, "text": "团财高绩课程正文"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reference_path.write_text(
        """title:: 线上团财展业经验分享
getnote-id:: wrong-note
- 摘要
	- **时长**：约 0小时 48分钟
- 原始转录
	- 🟢 说话人1 [00:00:00]
		- 团财展业经验课程正文。
	- 🟢 说话人1 [00:47:40]
		- 课程结束。
""",
        encoding="utf-8",
    )
    binding = build_transcript_reference_binding(
        media_path,
        reference_path,
        candidate_path=candidate_path,
        media_duration_seconds=1800,
        topic_anchors=["团财"],
        allow_invalid=True,
    )

    result = evaluate_transcript_files(
        candidate_path,
        reference_path,
        reference_binding=binding,
        media_path=media_path,
        media_duration_seconds=1800,
        require_reference_binding=True,
    )

    assert result["status"] == "failed"
    assert result["evaluation_state"] == "reference_binding_invalid"
    assert result["diagnostic_statuses"] == ["reference_binding_invalid"]
    assert result["completion"]["assessment_status"] == "not_evaluated"
    assert result["completion"]["possible_long_form_loss"] is False
    assert "reference_duration_mismatch" in result["reference_binding"]["reasons"]


def test_same_topic_decoy_reference_requires_human_selection(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "团财实战.mp4"
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "same-topic-decoy.md"
    media_path.write_bytes(b"same-duration-media")
    candidate_path.write_text(
        json.dumps(
            {
                "duration_seconds": 1800,
                "segments": [
                    {
                        "start": 0,
                        "end": 1800,
                        "text": "欢迎参加课程，今天重点讲雇主责任险报价、理赔材料和企业风险。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reference_path.write_text(
        """title:: 团财增员与团队管理
getnote-id:: same-topic-decoy
- 摘要
	- **时长**：约 0小时 30分钟
- 原始转录
	- 🟢 说话人1 [00:00:00]
		- 大家好，今天介绍新人招募、基本法收入和团队晋升路径。
	- 🟢 说话人1 [00:29:40]
		- 团队管理课程结束。
""",
        encoding="utf-8",
    )
    binding = build_transcript_reference_binding(
        media_path,
        reference_path,
        candidate_path=candidate_path,
        media_duration_seconds=1800,
        topic_anchors=["团财"],
        allow_invalid=True,
    )

    result = evaluate_transcript_files(
        candidate_path,
        reference_path,
        reference_binding=binding,
        media_path=media_path,
        media_duration_seconds=1800,
        require_reference_binding=True,
    )

    assert result["evaluation_state"] == "reference_binding_invalid"
    assert result["reference_binding"]["requires_human_selection"] is True
    assert "topic_fingerprint_mismatch" in result["reference_binding"]["reasons"]


def test_reference_binding_rejects_reference_changed_after_approval(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "课程.mp4"
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.md"
    media_path.write_bytes(b"immutable-media")
    candidate_path.write_text(
        json.dumps(
            {
                "duration_seconds": 600,
                "segments": [
                    {
                        "start": 0,
                        "end": 600,
                        "text": "这是一段关于客户信任和成交原则的课程正文。课程结束。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    original = """title:: 客户信任与成交原则
getnote-id:: exact-note-id
- 摘要
	- **时长**：约 0小时 10分钟
- 原始转录
	- 🟢 说话人1 [00:00:00]
		- 这是一段关于客户信任和成交原则的课程正文。
	- 🟢 说话人1 [00:09:50]
		- 课程结束。
"""
    reference_path.write_text(original, encoding="utf-8")
    binding = build_transcript_reference_binding(
        media_path,
        reference_path,
        candidate_path=candidate_path,
        media_duration_seconds=600,
        topic_anchors=["客户信任", "成交原则"],
    )
    reference_path.write_text(
        original.replace("exact-note-id", "different-note-id"),
        encoding="utf-8",
    )

    result = evaluate_transcript_files(
        candidate_path,
        reference_path,
        reference_binding=binding,
        media_path=media_path,
        media_duration_seconds=600,
        require_reference_binding=True,
    )

    assert result["evaluation_state"] == "reference_binding_invalid"
    assert "reference_sha256_mismatch" in result["reference_binding"]["reasons"]
    assert "getnote_id_mismatch" in result["reference_binding"]["reasons"]
    assert result["completion"]["possible_long_form_loss"] is False


def test_cli_creates_binding_and_uses_strict_mode_by_default(tmp_path: Path) -> None:
    media_path = tmp_path / "课程.mp4"
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.md"
    binding_path = tmp_path / "binding.json"
    report_path = tmp_path / "report.json"
    media_path.write_bytes(b"cli-media")
    candidate_path.write_text(
        json.dumps(
            {
                "duration_seconds": 600,
                "segments": [
                    {
                        "start": 0,
                        "end": 600,
                        "text": "今天讲客户信任和成交原则。课程结束。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reference_path.write_text(
        """title:: 客户信任和成交原则
getnote-id:: cli-note
- 摘要
	- **时长**：约 0小时 10分钟
- 原始转录
	- 🟢 说话人1 [00:00:00]
		- 今天讲客户信任和成交原则。
	- 🟢 说话人1 [00:09:50]
		- 课程结束。
""",
        encoding="utf-8",
    )

    exit_code = transcript_evaluation_main(
        [
            str(candidate_path),
            str(reference_path),
            str(report_path),
            "--media-path",
            str(media_path),
            "--media-duration-seconds",
            "600",
            "--create-reference-binding",
            str(binding_path),
            "--topic-anchor",
            "客户信任",
        ]
    )

    assert exit_code == 0
    assert json.loads(binding_path.read_text(encoding="utf-8"))["status"] == "active"
    assert json.loads(report_path.read_text(encoding="utf-8"))["reference_binding"]["status"] == "valid"

def test_reference_binding_reuses_fresh_chunk_manifest_media_identity(
    tmp_path: Path, monkeypatch
) -> None:
    media_path = tmp_path / "课程.mp4"
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.md"
    media_path.write_bytes(b"large-media-placeholder")
    candidate_path.write_text(
        json.dumps(
            {
                "duration_seconds": 120,
                "segments": [
                    {"start": 0, "end": 120, "text": "课程正文与结尾"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reference_path.write_text(
        """title:: 课程正文
getnote-id:: exact-precomputed-media
- 原始转录
    - 🟢 说话人1 [00:00:00]
        - 课程正文与结尾。
    - 🟢 说话人1 [00:02:00]
        - 结束。
""",
        encoding="utf-8",
    )
    stat = media_path.stat()
    media_identity = {
        "path": str(media_path.resolve()),
        "sha256": "a" * 64,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "duration_seconds": 120.0,
        "source": "audio_chunk_manifest.v1",
    }

    def fail_live_media_read(*args, **kwargs):
        raise AssertionError("fresh precomputed media identity must avoid a live full-file hash")

    monkeypatch.setattr(
        "video_knowledge_pipeline.transcript_stability_evaluation.probe_video",
        fail_live_media_read,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.transcript_stability_evaluation.sha256_file",
        fail_live_media_read,
    )

    binding = build_transcript_reference_binding(
        media_path,
        reference_path,
        candidate_path=candidate_path,
        media_identity=media_identity,
    )
    result = evaluate_transcript_files(
        candidate_path,
        reference_path,
        reference_binding=binding,
        media_path=media_path,
        require_reference_binding=True,
    )

    assert binding["status"] == "active"
    assert binding["video"]["sha256"] == "a" * 64
    assert binding["video"]["file_identity"]["source"] == "audio_chunk_manifest.v1"
    assert result["reference_binding"]["status"] == "valid"


def test_reference_binding_validates_saved_identity_when_removable_media_is_unavailable(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "移动课程.mp4"
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.md"
    media_path.write_bytes(b"removable-media-placeholder")
    candidate_path.write_text(
        json.dumps(
            {
                "duration_seconds": 120,
                "segments": [{"start": 0, "end": 120, "text": "课程正文与结尾"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reference_path.write_text(
        """title:: 课程正文
getnote-id:: exact-offline-media
- 原始转录
    - 🟢 说话人1 [00:00:00]
        - 课程正文与结尾。
    - 🟢 说话人1 [00:02:00]
        - 结束。
""",
        encoding="utf-8",
    )
    stat = media_path.stat()
    media_identity = {
        "path": str(media_path.absolute()),
        "sha256": "b" * 64,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "duration_seconds": 120.0,
        "source": "audio_chunk_manifest.v1",
    }
    media_path.unlink()

    binding = build_transcript_reference_binding(
        media_path,
        reference_path,
        candidate_path=candidate_path,
        media_identity=media_identity,
        allow_unavailable_media_identity=True,
    )
    result = evaluate_transcript_files(
        candidate_path,
        reference_path,
        reference_binding=binding,
        media_path=media_path,
        media_identity=media_identity,
        require_reference_binding=True,
    )

    assert binding["status"] == "active"
    assert result["reference_binding"]["status"] == "valid"
    assert result["reference_binding"]["media_identity_source"] == "audio_chunk_manifest.v1"