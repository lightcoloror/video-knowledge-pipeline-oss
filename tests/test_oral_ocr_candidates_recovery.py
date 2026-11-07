from copy import deepcopy

from PIL import Image
import pytest

from video_knowledge_pipeline.creator_report_practical_view import build_practical_view
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.ocr_backfill import _backfill_ocr_results, _results_from_imported_entries
from video_knowledge_pipeline.ocr_block_candidates import (
    bind_ocr_candidates_for_display, project_ocr_block_candidates,
)
from video_knowledge_pipeline.oral_ocr_recovery import build_oral_ocr_recovery_queue
from video_knowledge_pipeline.screen_text_recovery import run_screen_text_recovery
from video_knowledge_pipeline.storage import read_json, write_json


def _case(tmp_path):
    root = tmp_path / 'bundle'
    image = root / 'assets/frame.jpg'
    image.parent.mkdir(parents=True)
    Image.new('RGB', (1000, 1000), 'white').save(image)
    row = {'index': 1, 'start': 0, 'end': 4, 'frame_paths': [str(image)],
        'visual_route': 'semantic_frame', 'visual_text': '',
        'quality_issues': ['screen_text_low_confidence'],
        'visual_understanding': {'objects': ['出镜者'], 'actions': ['面对镜头']}}
    write_json(root / 'manifest.json', {})
    write_json(root / 'timeline.json', [row])
    result = {'index': 1, 'image_path': str(image), 'text': '每天一小时\n影响力写作\nJYAOTIVART',
        'ok': False, 'recognition_status': 'suspicious', 'executed': True,
        'authoritative_observation': True,
        'ocr_blocks': [{'text': '每天一小时', 'score': .998, 'bbox': [100, 200, 800, 300]},
                       {'text': '影响力写作', 'score': .997, 'bbox': [100, 310, 800, 410]},
                       {'text': 'JYAOTIVART', 'score': .59, 'bbox': [400, 600, 600, 640]}]}
    return root, image, row, result


def test_high_score_titles_survive_noise_without_promoting_strict_frame(tmp_path):
    root, image, row, result = _case(tmp_path)
    original = deepcopy(row)
    projected = project_ocr_block_candidates(root, [row], [result])
    assert row == original and projected[0]['visual_text'] == ''
    payload = projected[0]['ocr_block_candidates']
    assert payload['source_verified'] and payload['strict_result_ok'] is False
    assert payload['strict_frame_status'] == 'suspicious'
    assert [b['reference_usable'] for b in payload['blocks']] == [True, True, False]
    assert payload['blocks'][0]['role'] == 'title'
    assert payload['source_image_sha256'] == sha256_file(image)
    view = build_practical_view(manifest={}, transcript=[{'start': 0, 'end': 4, 'text': '写作可以练习。'}],
        timeline=projected, shots=[], video_type='talking_head')
    visual = view['visual_evidence'][0]
    assert '每天一小时' in visual['screen_text'] and '影响力写作' in visual['screen_text']
    assert 'JYAOTIVART' not in visual['screen_text']
    assert visual['status'] == 'review-required'
    assert any(issue['code'] == 'ocr_block_reference_candidate' for issue in visual['issues'])


def test_frame_change_invalidates_reference_candidates_without_mutating_stored_payload(tmp_path):
    root, image, row, result = _case(tmp_path)
    projected = project_ocr_block_candidates(root, [row], [result])
    Image.new('RGB', (1000, 1000), 'black').save(image)
    rebound = bind_ocr_candidates_for_display(root, projected)
    assert projected[0]['ocr_block_candidates']['source_verified'] is True
    assert rebound[0]['ocr_block_candidates']['source_verified'] is False
    assert not any(b['reference_usable'] for b in rebound[0]['ocr_block_candidates']['blocks'])


def test_wrong_source_and_invalid_geometry_never_become_usable(tmp_path):
    root, _, row, result = _case(tmp_path)
    result['source_sha256'] = '0' * 64
    projected = project_ocr_block_candidates(root, [row], [result])
    assert not any(b['reference_usable'] for b in projected[0]['ocr_block_candidates']['blocks'])
    result.pop('source_sha256')
    result['ocr_blocks'][0]['bbox'] = [0, 0, 2000, 2000]
    result['ocr_blocks'][1]['score'] = float('nan')
    projected = project_ocr_block_candidates(root, [row], [result])
    assert not any(b['reference_usable'] for b in projected[0]['ocr_block_candidates']['blocks'])


def test_raw_suspicious_backfill_adds_candidates_but_keeps_original_text_and_flags(tmp_path):
    root, _, row, result = _case(tmp_path)
    outcome = _backfill_ocr_results(root, {}, [row], [result], candidates=[{'index': 1}],
        apply_mode='merge', input_json=None)
    saved = read_json(root / 'timeline.json')[0]
    assert outcome['updated'] == 0 and saved['visual_text'] == ''
    assert saved['quality_issues'] == ['screen_text_low_confidence']
    assert sum(b['reference_usable'] for b in saved['ocr_block_candidates']['blocks']) == 2


def test_saved_engine_receipt_is_not_reinterpreted_as_human_correction(tmp_path):
    root, image, row, result = _case(tmp_path)
    cached = _results_from_imported_entries([result], [{'index': 1, 'image_path': str(image)}])
    assert cached[0]['ok'] is False and cached[0]['executed'] is False
    assert cached[0]['recognition_status'] == 'suspicious' and cached[0]['cached_observation']
    _backfill_ocr_results(root, {}, [row], cached, candidates=[{'index': 1}], apply_mode='merge', input_json=None)
    saved = read_json(root / 'timeline.json')[0]
    assert saved['visual_text'] == ''
    assert sum(block['reference_usable'] for block in saved['ocr_block_candidates']['blocks']) == 2


def test_region_queue_is_bounded_pure_and_uses_existing_crop_contract(tmp_path):
    root, image, row, result = _case(tmp_path)
    result['ocr_blocks'].append({'text': '不清楚的图示', 'score': .4,
        'bbox': [200, 450, 700, 700], 'role': 'tablet'})
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    queue = build_oral_ocr_recovery_queue(root, results=[result], max_regions=1, write=False)
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert len(queue['selected_regions']) == 1
    region = queue['selected_regions'][0]
    assert region['role'] == 'tablet' and region['image_sha256'] == sha256_file(image)
    crop = queue['candidates'][0]['screen_text_recovery']['crop_candidates'][0]
    assert crop['coordinate_system'] == 'relative_xyxy'
    assert crop['box'] != [0, 0, 1, 1] and min(crop['box']) >= 0 and max(crop['box']) <= 1
    assert queue['operator_boundary']['model_calls_made'] == 0
    write_json(root / 'oral-ocr-recovery-state.json', {'regions': {region['region_id']: {'attempts': 2}}})
    exhausted = build_oral_ocr_recovery_queue(root, results=[result], max_attempts=2, write=False)
    assert not exhausted['selected_regions']
    assert exhausted['blocked'][0]['reason'] == 'attempt_budget_exhausted'


def test_oral_crop_execution_uses_selected_region_only_and_preserves_source(tmp_path):
    root, image, row, result = _case(tmp_path)
    result['ocr_blocks'].append({'text': '不清楚', 'score': .4, 'bbox': [200, 450, 700, 700], 'role': 'tablet'})
    saved = root / 'saved-ocr.json'
    write_json(saved, {'items': [result]})
    original = image.read_bytes()
    preview = run_screen_text_recovery(root, input_json=saved, profile='talking_head', write=False)
    assert preview['crop_summary']['planned'] == 1 and preview['oral_recovery_queue']['plan_only'] is True
    executed = run_screen_text_recovery(root, input_json=saved, profile='talking_head', execute_crops=True, write=True)
    assert executed['crop_summary']['written'] == 1 and image.read_bytes() == original
    assert read_json(root / 'timeline.json')[0]['visual_text'] == ''
    assert not (root / 'oral-ocr-recovery-state.json').exists()  # cropping is not an engine attempt


def test_oral_engine_budget_survives_restart_without_promoting_crop_text(tmp_path, monkeypatch):
    import video_knowledge_pipeline.screen_text_recovery as recovery
    root, _, _, result = _case(tmp_path)
    result['ocr_blocks'].append({'text': '不清楚', 'score': .4, 'bbox': [200, 450, 700, 700], 'role': 'tablet'})
    saved = root / 'saved-ocr.json'
    write_json(saved, {'items': [result]})
    calls = []

    def fake_engine(candidates):
        calls.append(candidates)
        return {'available': True, 'name': 'synthetic_no_engine'}, [
            {**candidate, 'text': '合成裁剪文字', 'ok': True, 'executed': True} for candidate in candidates]

    monkeypatch.setattr(recovery, '_run_rapidocr_candidates', fake_engine)
    for _ in range(3):
        outcome = run_screen_text_recovery(root, input_json=saved, profile='talking_head',
            execute_ocr=True, engine='rapidocr', max_attempts=2, write=True)
    assert len(calls) == 2
    state = read_json(root / 'oral-ocr-recovery-state.json')
    assert next(iter(state['regions'].values()))['attempts'] == 2
    assert read_json(root / 'timeline.json')[0]['visual_text'] == ''
    assert not outcome['oral_recovery_queue']['selected_regions']


@pytest.mark.parametrize('blocks', [['not-an-object'], [None], {}])
def test_malformed_block_records_reject_before_any_write(tmp_path, blocks):
    root, _, row, result = _case(tmp_path)
    result['ocr_blocks'] = blocks
    with pytest.raises(ValueError, match='OCR blocks'):
        project_ocr_block_candidates(root, [row], [result])
