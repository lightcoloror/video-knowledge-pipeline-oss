from __future__ import annotations

import copy

import pytest

from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.subtitle_editor import (
    build_subtitle_editor_projection, validate_subtitle_review, apply_subtitle_review,
)
from video_knowledge_pipeline.subtitle_editor_ui import prepare_subtitle_editor


def _bundle(root):
    bundle = root/'bundle'
    bundle.mkdir()
    media = bundle/'video.mp4'
    media.write_bytes(b'synthetic only')
    write_json(bundle/'manifest.json', {'title': 'fixture oral', 'media_path': str(media),
        'media_duration_seconds': 610, 'normalized_transcript_json': 'transcript.json'})
    rows = [{'segment_id': f'cue-{index}', 'source_segment_ids': [f'raw-{index}'],
        'start': index, 'end': index+1, 'text': '' if index%12 == 0 else f'source {index}',
        'empty_text_preserved': index%12 == 0}
        for index in range(610)]
    write_json(bundle/'transcript.json', {'segments': rows})
    write_json(bundle/'mandarin-translated-transcript.json', {'segments': [
        {'segment_id': row['segment_id'], 'text': f'translation {index}'} for index, row in enumerate(rows)]})
    write_json(bundle/'timeline.json', [])
    return bundle


def _notes(projection):
    return {'schema': 'video_knowledge_pipeline.subtitle_review_notes.v1',
        'projection_sha256': projection['projection_sha256'],
        'source_sha256': projection['source_sha256'], 'human_confirmed': True,
        'segments': copy.deepcopy(projection['segments'])}


def test_blank_rows_are_read_only_and_translation_indexes_do_not_shift(tmp_path):
    bundle = _bundle(tmp_path)
    before = (bundle/'transcript.json').read_bytes()
    projection = build_subtitle_editor_projection(bundle, write=False)
    assert len(projection['source_segments']) == 610
    assert len(projection['segments']) == 559 and len(projection['read_only_gaps']) == 51
    assert projection['segments'][0]['source_index'] == 1
    assert projection['segments'][0]['mandarin_text'] == 'translation 1'
    assert projection['read_only_gaps'][0]['segment_id'] == 'cue-0'
    assert projection['read_only_gaps'][1]['source_index'] == 12
    assert all(row['read_only_gap'] for row in projection['read_only_gaps'])
    result = prepare_subtitle_editor(bundle)
    assert result['status'] == 'prepared' and result['read_only_gap_count'] == 51
    assert '只读，保存不会删除' in (bundle/'subtitle-editor.html').read_text(encoding='utf-8')
    assert (bundle/'transcript.json').read_bytes() == before


def test_apply_retains_all_gap_ids_times_and_does_not_invent_text(tmp_path):
    bundle = _bundle(tmp_path)
    raw = (bundle/'transcript.json').read_bytes()
    projection = build_subtitle_editor_projection(bundle, write=False)
    notes = _notes(projection)
    notes['segments'][0]['source_text'] = 'reviewed source 1'
    result = apply_subtitle_review(bundle, review_json=notes)
    assert result['summary']['preserved_gap_count'] == 51
    assert result['summary']['source_text_changes'] == 1
    corrected = read_json(bundle/'human-corrected-transcript.json')['segments']
    assert len(corrected) == 610
    for gap in projection['read_only_gaps']:
        row = corrected[gap['source_index']]
        assert row['segment_id'] == gap['segment_id'] and row['text'] == ''
        assert row['source_segment_ids'] == gap['source_segment_ids']
        assert row['start']*1000 == gap['start_ms'] and row['end']*1000 == gap['end_ms']
        assert row['read_only_gap']
    assert corrected[13]['text'] == 'source 13'
    assert corrected[13]['source_index'] == 13
    assert (bundle/'transcript.json').read_bytes() == raw
    assert 'source 0' not in (bundle/'human-reviewed-source.srt').read_text(encoding='utf-8')
    second_projection = build_subtitle_editor_projection(bundle, write=False)
    assert len(second_projection['source_segments']) == 610
    assert len(second_projection['read_only_gaps']) == 51


def test_gap_cannot_be_edited_or_silently_merged_away(tmp_path):
    bundle = _bundle(tmp_path)
    projection = build_subtitle_editor_projection(bundle, write=False)
    notes = _notes(projection)
    gap = copy.deepcopy(projection['read_only_gaps'][0])
    gap['source_text'] = 'invented text'
    notes['segments'].insert(0, gap)
    with pytest.raises(ValueError, match='unknown source lineage'):
        validate_subtitle_review(bundle, notes)
    notes = _notes(projection)
    left = next(row for row in notes['segments'] if row['source_index'] == 11)
    right = next(row for row in notes['segments'] if row['source_index'] == 13)
    merged = {**left, 'segment_id': 'merged-over-gap',
        'source_segment_ids': left['source_segment_ids']+right['source_segment_ids'],
        'source_lineage_ids': left['source_lineage_ids']+right['source_lineage_ids'],
        'end_ms': right['end_ms']}
    index = notes['segments'].index(left)
    notes['segments'][index:index+2] = [merged]
    with pytest.raises(ValueError, match='contiguous'):
        validate_subtitle_review(bundle, notes)
    notes = _notes(projection)
    notes['segments'][0]['end_ms'] = -1
    with pytest.raises(ValueError, match='timing'):
        validate_subtitle_review(bundle, notes)
