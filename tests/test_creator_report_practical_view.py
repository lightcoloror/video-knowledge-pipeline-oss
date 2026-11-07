from __future__ import annotations

from copy import deepcopy

import pytest

from video_knowledge_pipeline.creator_report_practical_view import PENDING, build_practical_view


def view(transcript=None, timeline=None, shots=None, manifest=None, **kwargs):
    return build_practical_view(manifest=manifest or {}, transcript=transcript or [],
                                timeline=timeline or [], shots=shots or [], **kwargs)


def tutorial_inputs():
    return {
        "manifest": {"title": "软件导出教程", "strict_status": "blocked"},
        "transcript": [
            {"start": 0, "end": 5, "text": "首先点击“设置”按钮。"},
            {"start": 5, "end": 12, "text": "分辨率设为1080p。"},
            {"start": 12, "end": 18, "text": "然后点击“导出”按钮。"},
            {"start": 18, "end": 24, "text": "界面显示“导出完成”。"},
        ],
        "timeline": [
            {"start": 0, "end": 5, "visual_text": "设置\n分辨率 1080p", "temporal_visual_understanding": {"summary": "软件界面中的设置菜单打开"}},
            {"start": 12, "end": 24, "visual_text": "导出\n导出完成", "temporal_visual_understanding": {"events": [{"description": "点击导出按钮，界面显示导出完成"}]}},
        ],
        "shots": [{"shot_id": "shot-1", "index": 1, "start": 0, "end": 72,
                   "facts": {"shot_type": "screen_recording", "screen_text": "设置\n分辨率 1080p\n导出\n导出完成"}}],
    }


def test_tutorial_uses_instructions_not_technical_shot_boundaries():
    result = build_practical_view(**tutorial_inputs())
    assert result["video_type"] == "tutorial"
    assert result["basis"]["confidence"] == "moderate"
    assert result["basis"]["boundary_strategy"] == "instruction_to_stable_result"
    assert len(result["steps"]) == 3
    assert result["steps"][0]["button"] == "设置"
    assert result["steps"][1]["parameters"] == ["分辨率设为1080p"]
    assert "导出完成" in result["steps"][-1]["result"]
    assert result["steps"][-1]["end"] == 24
    assert result["shots"][0]["end"] == 72
    assert all(row["evidence_refs"] for row in result["steps"])


def test_title_and_one_weak_operation_do_not_claim_tutorial():
    result = view(manifest={"title": "万能教程"}, transcript=[{"start": 0, "end": 8, "text": "今天谈谈为什么点击量下降。"}])
    assert result["video_type"] == "general" and result["steps"] == []
    assert result["basis"]["confidence"] == "low"
    assert any(row["kind"] == "weak_title_hint" for row in result["basis"]["signals"])


@pytest.mark.parametrize("selected", ["tutorial", "talking_head", "visual_story", "general"])
def test_explicit_type_is_a_display_choice(selected):
    result = view(video_type=selected, manifest={"video_type": "general"})
    assert result["video_type"] == selected
    assert result["basis"]["source"] == "explicit_request"
    assert result["candidate_only"] and not result["human_confirmed"]


def test_manifest_type_is_declared_not_independently_confirmed():
    result = view(manifest={"metadata": {"video_type": "教程"}})
    assert result["video_type"] == "tutorial"
    assert result["basis"]["confidence"] == "declared"
    assert result["steps"] == []
    assert any(issue["code"] == "tutorial_steps_unavailable" for issue in result["issues"])
    assert view(manifest={"video_type": "tutorial", "metadata": {"content_type": "general"}})["video_type"] == "general"


def test_invalid_video_type_fails_without_guessing():
    with pytest.raises(ValueError, match="video_type"):
        view(video_type="marketing")


def test_exact_duplicate_cues_keep_all_source_references():
    cues = [
        {"start": 0, "end": 4, "text": "点击保存", "speaker": "A"},
        {"start": 0, "end": 4, "text": "点击保存", "speaker": "A"},
        {"start": 3.5, "end": 6, "text": "导出完成", "source_cue_id": "x"},
        {"start": 4, "end": 6, "text": "导出完成", "source_cue_id": "x"},
    ]
    result = view(transcript=cues)
    assert len(result["transcript"]) == 2
    assert result["transcript"][0]["evidence_refs"] == ["transcript:0", "transcript:1"]
    assert result["transcript"][1]["source_cue_indexes"] == [2, 3]
    assert result["cleanup"]["transcript_hidden_duplicate_count"] == 2


def test_numbers_negation_and_meaningful_punctuation_are_never_fuzzy_merged():
    texts = ["点击保存", "不要点击保存", "帧率设为30fps", "帧率设为60fps", "1.0", "10", "不，不要", "不不要"]
    result = view(transcript=[{"start": 0, "end": 5, "text": text, "source_cue_id": "same"} for text in texts])
    assert [row["text"] for row in result["transcript"]] == texts
    assert not result["cleanup"]["removed"]


def test_intentional_repetition_at_different_times_and_repeated_characters_survive():
    cues = [{"start": 0, "end": 2, "text": "好好好，点击保存。"},
            {"start": 5, "end": 7, "text": "好好好，点击保存。"},
            {"start": 7, "end": 9, "text": "好好好，点击保存。"}]
    result = view(transcript=cues)
    assert len(result["transcript"]) == 3
    assert all("好好好" in row["text"] for row in result["transcript"])


def test_speaker_difference_is_not_a_duplicate():
    result = view(transcript=[{"start": 0, "end": 2, "text": "对", "speaker": "A"},
                              {"start": 0, "end": 2, "text": "对", "speaker": "B"}])
    assert len(result["transcript"]) == 2


def test_ocr_noise_cleanup_is_traceable_and_preserves_menus_and_values():
    source = {"start": 0, "end": 8, "facts": {"screen_text": "||||~~\n为你推荐\n文件 编辑 视图 工具 帮助\n分辨率 1080p\n导出\n导出\n设置 � 50%"}}
    result = view(shots=[source])
    shown = result["shots"][0]["facts"]["screen_text"]
    assert "||||" not in shown and "为你推荐" not in shown
    assert "文件 编辑 视图 工具 帮助" in shown and "1080p" in shown and "50%" in shown
    assert shown.splitlines().count("导出") == 1
    removed = result["cleanup"]["removed"]
    assert {row["reason"] for row in removed} == {"obvious_ocr_symbol_noise", "unrelated_recommendation_ui", "repeated_ocr_display"}
    assert all(row["text"] and row["evidence_refs"] for row in removed)
    assert any(issue["code"] == "ocr_character_uncertain" for issue in result["shots"][0]["issues"])


def test_an_ocr_ui_label_mentioned_in_the_instruction_is_preserved():
    result = view(transcript=[{"start": 0, "end": 3, "text": "点击“为你推荐”菜单。"}],
                  shots=[{"start": 0, "end": 3, "facts": {"screen_text": "为你推荐"}}])
    assert result["shots"][0]["facts"]["screen_text"] == "为你推荐"


def test_uncertainty_is_local_and_missing_fields_stay_pending():
    result = view(video_type="tutorial", transcript=[
        {"start": 0, "end": 4, "text": "点击“设置”按钮。", "confidence": 0.3},
        {"start": 4, "end": 8, "text": "点击“保存”按钮，音量设为50%。界面显示保存成功。"},
    ])
    assert result["steps"][0]["status"] == "review-required"
    assert result["steps"][0]["parameters"] == [PENDING]
    assert result["steps"][0]["result"] == PENDING
    assert result["transcript"][1]["status"] == "evidence-present"
    assert result["steps"][1]["status"] == "evidence-present"
    assert result["steps"][1]["field_evidence"]["result"] == ["transcript:1"]
    assert result["operator_boundary"]["strict_gate_modified"] is False


def test_long_shot_ocr_cannot_fill_every_step_with_unlocalized_parameters_or_results():
    result = view(video_type="tutorial", transcript=[{"start": 0, "end": 5, "text": "点击“设置”按钮。"},
                                                      {"start": 20, "end": 25, "text": "点击“导出”按钮。"}],
                  shots=[{"start": 0, "end": 72, "facts": {"screen_text": "帧率60fps\n导出完成"}}])
    assert all(row["parameters"] == [PENDING] and row["result"] == PENDING for row in result["steps"])


def test_multiple_actions_in_one_cue_keep_original_timing_and_mark_it_local():
    result = view(video_type="tutorial", transcript=[{"start": 0, "end": 10,
                  "text": "点击“设置”按钮，然后点击“保存”按钮。界面显示保存成功。"}])
    assert len(result["steps"]) == 2
    assert [(row["start"], row["end"]) for row in result["steps"]] == [(0, 10), (0, 10)]
    assert all(any(issue["code"] == "shared_cue_timing" for issue in row["issues"]) for row in result["steps"])


def test_inputs_are_unchanged_outputs_detached_and_result_deterministic():
    inputs = tutorial_inputs()
    before = deepcopy(inputs)
    first = build_practical_view(**inputs)
    second = build_practical_view(**inputs)
    assert inputs == before and first == second
    first["shots"][0]["facts"]["shot_type"] = "edited display"
    first["transcript"][0]["evidence_refs"].append("fake")
    assert inputs == before
    assert first["cleanup"]["preserves_canonical"] is True


def test_reference_chapter_titles_do_not_become_button_parameter_or_result_facts():
    result = view(video_type="tutorial", transcript=[{"start": 0, "end": 5, "text": "点击“设置”按钮。"}],
                  reference_summary={"chapters": [{"start": 0, "title": "帧率60fps导出完成"}]})
    assert result["steps"][0]["parameters"] == [PENDING]
    assert result["steps"][0]["result"] == PENDING


def test_existing_canonical_issues_and_empty_cues_are_preserved():
    issue = {"code": "source_suspect", "message": "这一处人名待核", "evidence_refs": ["canonical:0"]}
    cues = [{"start": 0, "end": 4, "text": "点击“设置”按钮。", "issues": [issue],
             "status": "review-required", "evidence_refs": ["canonical:0"]},
            {"start": 4, "end": 5, "text": "", "evidence_refs": ["canonical:1"]}]
    result = view(video_type="tutorial", transcript=cues)
    assert issue in result["transcript"][0]["issues"]
    assert issue in result["steps"][0]["issues"]
    assert "canonical:0" in result["transcript"][0]["evidence_refs"]
    assert len(result["transcript"]) == 2
    assert any(row["code"] == "empty_transcript_cue" for row in result["transcript"][1]["issues"])


def test_metrics_are_not_operation_cues_and_repeated_sample_ocr_keeps_locators():
    result = view(transcript=[{"start": 0, "end": 4, "text": "点击量下降。"},
                              {"start": 4, "end": 8, "text": "下载率降低。"}],
                  timeline=[{"start": 0, "end": 4, "visual_text": "设置菜单\n50%"},
                            {"start": 4, "end": 8, "visual_text": "设置菜单\n50%"}],
                  shots=[{"start": 0, "end": 8, "facts": {"screen_text": "设置菜单\n50%"}}])
    assert result["video_type"] == "general"
    assert result["shots"][0]["facts"]["screen_text"] == "设置菜单\n50%"
    line = result["shots"][0]["ocr_lines"][0]
    assert len(line["evidence_refs"]) == 3
    assert any(row["reason"] == "repeated_ocr_projection" for row in result["cleanup"]["removed"])


@pytest.mark.parametrize(("text", "button", "parameter"), [
    ("这里我们将所有的变速调到0.5。", PENDING, "变速调到0.5"),
    ("选叠画，将它调到0.2，再选择全局应用。", "叠画", "叠画，将它调到0.2"),
    ("工具栏里点比例，这里选择9比16。", "比例", "比例，这里选择9比16"),
    ("大概保留2到3秒左右。", PENDING, "大概保留2到3秒左右"),
    ("选滤镜，将强度大概调到70左右。", "滤镜", "滤镜，将强度大概调到70左右"),
])
def test_spoken_c01_controls_and_locally_bound_parameters_are_preserved(text, button, parameter):
    result = view(video_type="tutorial", transcript=[{"start": 0, "end": 8, "text": text}])
    assert result["steps"][0]["button"] == button
    assert parameter in result["steps"][0]["parameters"]
    assert result["steps"][0]["field_evidence"]["parameters"] == ["transcript:0"]
    assert "9:16" not in str(result["steps"][0]["parameters"])


def test_spoken_short_controls_are_split_into_actions_and_keep_exact_labels():
    result = view(video_type="tutorial", transcript=[{"start": 0, "end": 12,
        "text": "剪好后，我们给视频变速，点亮视频条，点变速，常规变速。这里我们将所有的变速调到0.5。"}])
    assert any(row["button"] == "视频条" for row in result["steps"])
    assert any(row["button"] == "变速" for row in result["steps"])
    assert any("变速调到0.5" in row["parameters"] for row in result["steps"])
    assert all(row["start"] == 0 and row["end"] == 12 for row in result["steps"])


def test_unbound_scalar_is_pending_even_when_a_previous_sentence_names_an_object():
    result = view(video_type="tutorial", transcript=[
        {"start": 0, "end": 4, "text": "选叠画。"},
        {"start": 4, "end": 8, "text": "将它调到0.2。"},
        {"start": 8, "end": 12, "text": "将它调到70左右。"},
    ])
    assert result["steps"][1]["parameters"] == [PENDING]
    assert result["steps"][2]["parameters"] == [PENDING]
    assert result["steps"][1]["field_evidence"]["parameters"] == []
    same_cue = view(video_type="tutorial", transcript=[{"start": 0, "end": 8, "text": "选叠画。将它调到0.2。"}])
    assert same_cue["steps"][1]["parameters"] == [PENDING]


def test_unbound_ratio_is_not_attached_to_a_remote_proportion_control():
    result = view(video_type="tutorial", transcript=[{"start": 0, "end": 3, "text": "工具栏里点比例。"},
                                                      {"start": 20, "end": 24, "text": "这里选择9比16。"}])
    assert result["steps"][1]["button"] == PENDING
    assert result["steps"][1]["parameters"] == [PENDING]


def test_parameter_substring_numbers_and_negation_remain_distinct():
    result = view(video_type="tutorial", transcript=[{"start": 0, "end": 10, "text": "点击保存，帧率设为130fps。"}],
                  timeline=[{"start": 0, "end": 10, "visual_text": "30fps\n不要选择9比16"}])
    parameters = result["steps"][0]["parameters"]
    assert "帧率设为130fps" in parameters and "30fps" in parameters
    assert "9比16" not in parameters
    assert any("不要选择9比16" in row["screen_text"] for row in result["visual_evidence"])


def test_point_and_choice_nouns_do_not_create_short_action_false_positives():
    result = view(transcript=[{"start": 0, "end": 3, "text": "重点是观点。"},
                              {"start": 3, "end": 6, "text": "这是候选人自己的观点。"}],
                  timeline=[{"start": 0, "end": 6, "visual_text": "设置菜单"}])
    assert result["video_type"] == "general" and not result["steps"]


def test_same_sentence_c01_asr_cue_splits_keep_local_ratio_and_filter_evidence():
    result = view(video_type="tutorial", transcript=[
        {"start": 111.63, "end": 113.33, "text": "工具栏里点比例，"},
        {"start": 113.45, "end": 115.61, "text": "这里选择9比16。好。"},
        {"start": 184.35, "end": 185.55, "text": "我们选滤镜，"},
        {"start": 185.73, "end": 187.47, "text": "这里我们选高清增强，"},
        {"start": 187.59, "end": 188.85, "text": "将它调到70左右。"},
    ])
    ratio = next(row for row in result["steps"] if row["step"].startswith("这里选择"))
    strength = next(row for row in result["steps"] if row["step"].startswith("将它"))
    assert ratio["parameters"] == ["比例，这里选择9比16"]
    assert ratio["field_evidence"]["parameters"] == ["transcript:0", "transcript:1"]
    assert strength["parameters"] == ["滤镜，这里我们选高清增强，将它调到70左右"]
    assert strength["field_evidence"]["parameters"] == ["transcript:2", "transcript:3", "transcript:4"]
    assert strength["button"] == PENDING


def test_intervening_sentence_end_cancels_cross_cue_pronoun_binding():
    result = view(video_type="tutorial", transcript=[{"start": 0, "end": 2, "text": "选叠画，"},
                                                      {"start": 2.1, "end": 5, "text": "这个部分先讲完。将它调到0.2。"}])
    scalar = next(row for row in result["steps"] if row["step"].startswith("将它"))
    assert scalar["parameters"] == [PENDING]


@pytest.mark.parametrize("metadata", [
    {"quality_issues": ["screen_text_low_confidence"]},
    {"quality_issues": ["low_ocr_confidence"]},
    {"quality_issues": ["ocr_text_low_information"]},
    {"quality_issues": ["ocr_wrapper_only"]},
    {"ocr_confidence": {"mean_confidence": 69, "minimum_mean_confidence": 70}},
    {"ocr_confidence": {"quality_threshold_passed": False}},
    {"ocr_confidence": {"minimum_block_score": 0.8, "low_confidence_block_count": 1, "block_count": 2}},
    {"recognition_status": "suspicious"},
])
def test_inventory_quality_marks_stay_local_on_visual_shot_and_step(metadata):
    timeline = [{"start": 0, "end": 4, "visual_text": "导出完成", **metadata},
                {"start": 4, "end": 8, "visual_text": "保存成功"}]
    before = deepcopy(timeline)
    result = view(video_type="tutorial", transcript=[{"start": 0, "end": 4, "text": "点导出，帧率设为30fps。"},
                                                      {"start": 4, "end": 8, "text": "点保存，帧率设为30fps。"}],
                  timeline=timeline, shots=[{"start": 0, "end": 4}, {"start": 4, "end": 8}])
    assert result["visual_evidence"][0]["status"] == "review-required"
    assert result["shots"][0]["status"] == "review-required"
    assert result["steps"][0]["status"] == "review-required"
    assert result["visual_evidence"][1]["status"] == "evidence-present"
    assert result["shots"][1]["status"] == "evidence-present"
    assert result["steps"][1]["status"] == "evidence-present"
    assert result["visual_evidence"][0]["quality_issues"] == metadata.get("quality_issues", [])
    assert timeline == before


@pytest.mark.parametrize("confidence", [
    {"mean_confidence": 90, "minimum_mean_confidence": 70, "quality_threshold_passed": True, "word_count": 2},
    {"minimum_block_score": 0.8, "low_confidence_block_count": 0, "block_count": 2},
])
def test_native_ocr_scores_above_engine_threshold_do_not_create_false_warning(confidence):
    result = view(timeline=[{"start": 0, "end": 4, "visual_text": "设置菜单", "ocr_confidence": confidence}])
    assert result["visual_evidence"][0]["status"] == "evidence-present"
    assert result["visual_evidence"][0]["ocr_confidence"] == confidence


def test_empty_ocr_survives_cleanup_as_local_pending_evidence():
    result = view(video_type="tutorial", transcript=[{"start": 0, "end": 4, "text": "点击“设置”按钮。"}],
                  timeline=[{"start": 0, "end": 4, "visual_text": "", "quality_issues": ["ocr_text_empty"],
                             "ocr_confidence": {"word_count": 0, "mean_confidence": None, "quality_threshold_passed": False}}],
                  shots=[{"start": 0, "end": 4}])
    assert len(result["visual_evidence"]) == 1
    assert result["visual_evidence"][0]["screen_text"] == ""
    assert result["visual_evidence"][0]["status"] == "review-required"
    assert result["shots"][0]["status"] == "review-required"
    assert any(issue["code"] == "ocr_text_empty" for issue in result["steps"][0]["issues"])


def test_irrelevant_ui_statistics_are_hidden_unless_spoken_content_uses_metrics():
    line = "03:41·使用量7.1万"
    unrelated = view(timeline=[{"start": 0, "end": 4, "visual_text": line + "\n变速0.5"}])
    assert unrelated["visual_evidence"][0]["screen_text"] == "变速0.5"
    assert any(row["text"] == line and row["reason"] == "unrelated_ui_statistics" for row in unrelated["cleanup"]["removed"])
    relevant = view(transcript=[{"start": 0, "end": 4, "text": "这个模板的使用量是7.1万。"}],
                    timeline=[{"start": 0, "end": 4, "visual_text": line}])
    assert relevant["visual_evidence"][0]["screen_text"] == line


def test_oral_view_hides_background_logos_with_source_refs_and_preserves_all_raw_inputs():
    inputs = {"manifest": {}, "transcript": [{"start": 0, "end": 4, "text": "今天聊聊家庭保障。"}],
              "timeline": [{"start": 0, "end": 4, "visual_text": "纸品道\nComfortable\n-soft-"}],
              "shots": [{"start": 0, "end": 4, "facts": {"screen_text": "纸品道\nComfortable\n-soft-"}}]}
    before = deepcopy(inputs)
    result = build_practical_view(**inputs, video_type="talking_head")
    assert result["visual_evidence"][0]["screen_text"] == ""
    assert result["shots"][0]["facts"]["screen_text"] == ""
    hidden = [row for row in result["cleanup"]["removed"] if row["reason"] == "not_referenced_in_oral_transcript"]
    assert {row["text"] for row in hidden} == {"纸品道", "Comfortable", "-soft-"}
    assert all(row["evidence_refs"] for row in hidden)
    assert inputs == before


def test_oral_view_keeps_directly_spoken_words_numbers_and_local_uncertainty_verbatim():
    result = view(video_type="talking_head",
                  transcript=[{"start": 0, "end": 4, "text": "这个字是纸品道，英文写着Comfortable。比例是9比16，数值0.5。"}],
                  timeline=[{"start": 0, "end": 4, "visual_text": "纸品道\nComfortable\n9比16\n0.5\n-soft-",
                             "quality_issues": ["screen_text_low_confidence"]}],
                  shots=[{"start": 0, "end": 4}])
    expected = "纸品道\nComfortable\n9比16\n0.5"
    assert result["visual_evidence"][0]["screen_text"] == expected
    assert result["shots"][0]["facts"]["screen_text"] == expected
    assert result["visual_evidence"][0]["status"] == "review-required"
    assert result["shots"][0]["status"] == "review-required"
    assert "9:16" not in result["visual_evidence"][0]["screen_text"]
    hidden = [row["text"] for row in result["cleanup"]["removed"] if row["reason"] == "not_referenced_in_oral_transcript"]
    assert hidden and set(hidden) == {"-soft-"}


@pytest.mark.parametrize("selected", ["auto", "tutorial", "visual_story", "general"])
def test_oral_background_filter_does_not_change_auto_or_other_type_behavior(selected):
    result = view(video_type=selected, manifest={"video_type": "talking_head"},
                  transcript=[{"start": 0, "end": 4, "text": "今天聊聊家庭保障。"}],
                  timeline=[{"start": 0, "end": 4, "visual_text": "Comfortable"}],
                  shots=[{"start": 0, "end": 4, "facts": {"screen_text": "Comfortable"}}])
    assert result["visual_evidence"][0]["screen_text"] == "Comfortable"
    assert result["shots"][0]["facts"]["screen_text"] == "Comfortable"
    assert not any(row["reason"] == "not_referenced_in_oral_transcript" for row in result["cleanup"]["removed"])
