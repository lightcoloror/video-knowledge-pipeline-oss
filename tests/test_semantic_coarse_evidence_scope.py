from types import SimpleNamespace

from video_knowledge_pipeline.transcript_semantic_correction import _candidate_rows_for_text


def candidates(text, evidence):
    return _candidate_rows_for_text(0, SimpleNamespace(start=12, end=18), text, evidence)


def test_scene_schema_and_description_are_not_transcript_replacements():
    rows = candidates('设备当时需要例行维护。', [{'source_type': 'visual_understanding',
        'evidence_id': 'vision', 'text': 'objects: 人物；interface_state: 没有字幕；non_text_information: 木柜'}])
    assert not any(r['candidate_text'] == 'objects' for r in rows)
    assert not any(r['reason'] == 'visual_text_differs_from_transcript' for r in rows)


def test_incidental_packaging_remains_visual_evidence_without_becoming_spoken_text():
    from video_knowledge_pipeline.transcript_semantic_correction import _evidence_for_cue
    cue=SimpleNamespace(start=12,end=18,text='设备刚上线。')
    evidence=_evidence_for_cue(cue,{'index':1,'visual_text':'示例包装 DemoBrand',
        'visual_text_role':'incidental_background'})
    assert any(r['source_type']=='background_ocr' and '示例包装' in r['text'] for r in evidence)
    rows=candidates(cue.text,evidence)
    assert not any('示例包装' in r.get('candidate_text','') for r in rows)


def test_identical_sentence_in_coarse_block_has_no_neighbor_conflict():
    text = '设备当时需要例行维护。'
    rows = candidates(text, [{'source_type': 'secondary_asr', 'evidence_id': 'qwen',
        'text': '二零二一年设备刚上线。设备当时需要例行维护。之后团队检查了历史日志。'}])
    assert not any(r.get('has_conflict') for r in rows)


def test_coarse_block_preserves_real_word_disagreement_for_review():
    rows = candidates('随后团队详细核对了设备的保养方案。', [{'source_type': 'secondary_asr',
        'evidence_id': 'qwen', 'text': '前面有人说话。随后团队详细核对了设备的保修方案。后面还有其他内容。'}])
    assert any(r.get('has_conflict') and '修' in r['candidate_text'] for r in rows)


def test_numeric_risk_still_requires_review_after_scoping():
    rows = candidates('笔记有200022000多字。', [{'source_type': 'secondary_asr',
        'evidence_id': 'qwen', 'text': '团队查阅了清单。笔记有两万两千多字。随后复查了记录。'}])
    assert any(r['risk_level'] == 'high' and r['needs_human_review'] for r in rows)


def test_agreeing_sidecar_does_not_hide_a_different_sidecar_conflict():
    text = '随后团队详细核对了设备的保养方案。'
    rows = candidates(text, [
        {'source_type': 'secondary_asr', 'evidence_id': 'agreeing', 'text': '前文。' + text + '后文。'},
        {'source_type': 'secondary_asr', 'evidence_id': 'conflicting',
         'text': '前文。随后团队详细核对了设备的保修方案。后文。'},
    ])
    assert any(r.get('has_conflict') and '修' in r['candidate_text'] for r in rows)
