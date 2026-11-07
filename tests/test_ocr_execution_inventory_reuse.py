import video_knowledge_pipeline.ocr_backfill as backfill
from video_knowledge_pipeline.ocr_inventory import build_ocr_inventory
from video_knowledge_pipeline.storage import read_json, write_json


def bundle(tmp_path, count=80):
    root = tmp_path / 'bundle'
    root.mkdir()
    rows = []
    for index in range(1, count + 1):
        frame = root / f'{index}.png'
        frame.write_bytes(f'fixture pixels {index}'.encode())
        rows.append({'index': index, 'start': index, 'end': index+1,
                     'frame_paths': [str(frame)], 'visual_route': 'document_visual',
                     'quality_issues': ['missing_visual_text']})
    write_json(root/'timeline.json', rows)
    write_json(root/'manifest.json', {})
    return root


def test_nine_of_eighty_remain_seventy_one_and_resume(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    calls = []
    def run(candidates, **kwargs):
        calls.extend(c['index'] for c in candidates)
        return {'available': True, 'name': 'fixture'}, [dict(c, executed=True, ok=True, text='课件正文', authoritative_observation=True) for c in candidates]
    monkeypatch.setattr(backfill, '_run_captiocr_candidates', run)
    r = backfill.run_ocr_backfill(root, execute=True, limit=9)
    assert r['status'] == 'batch_completed_pending'
    assert r['summary']['whole_bundle']['required'] == 80
    assert r['summary']['whole_bundle']['executed'] == 9
    assert r['summary']['whole_bundle']['remaining'] == 71
    r = backfill.run_ocr_backfill(root, execute=True)
    assert r['summary']['whole_bundle']['remaining'] == 0
    assert calls == list(range(1,81))


def test_rapidocr_resume_preserves_selected_engine(tmp_path, monkeypatch):
    root = bundle(tmp_path, 2)
    monkeypatch.setattr(backfill, '_run_rapidocr_candidates', lambda candidates: (
        {'available': True, 'name': 'ebook_rapidocr'},
        [dict(c, executed=True, ok=True, text='合成文字') for c in candidates]))
    result = backfill.run_ocr_backfill(root, execute=True, limit=1, engine='rapidocr')
    assert result['status'] == 'batch_completed_pending'
    argv = result['summary']['resume_argv']
    assert argv[argv.index('--engine') + 1] == 'rapidocr'


def test_empty_failed_and_preflight_are_distinct(tmp_path, monkeypatch):
    root = bundle(tmp_path, 3)
    def run(candidates, **kwargs):
        index = candidates[0]['index']
        return {'available': index != 3, 'name': 'fixture'}, [dict(candidates[0], executed=True, ok=False, text='', authoritative_observation=index==1, stderr='' if index==1 else 'fixture failure')]
    monkeypatch.setattr(backfill, '_run_captiocr_candidates', run)
    r = backfill.run_ocr_backfill(root, execute=True)
    s = r['summary']['whole_bundle']
    assert (s['empty'], s['failed'], s['blocked']) == (1,1,1)
    assert s['executed'] == 2
    assert s['completed'] == 1
    assert s['usable_text'] == 0


def test_checkpoint_survives_interruption_and_pixels_invalidate(tmp_path, monkeypatch):
    root = bundle(tmp_path, 3)
    def run(candidates, **kwargs):
        if candidates[0]['index'] == 2:
            raise KeyboardInterrupt()
        return {'available': True, 'name': 'fixture'}, [dict(candidates[0], executed=True, ok=True, text='正文')]
    monkeypatch.setattr(backfill, '_run_captiocr_candidates', run)
    import pytest
    with pytest.raises(KeyboardInterrupt):
        backfill.run_ocr_backfill(root, execute=True)
    inventory = build_ocr_inventory(root, read_json(root/'timeline.json'))
    assert inventory['summary']['executed'] == 1
    assert inventory['pending_indexes'] == [2,3]
    (root/'1.png').write_bytes(b'new animation text')
    inventory = build_ocr_inventory(root, read_json(root/'timeline.json'))
    assert inventory['pending_indexes'] == [1,2,3]
    assert not inventory['items'][0]['usable_text']


def test_legacy_text_does_not_claim_executor_ran(tmp_path):
    root = bundle(tmp_path, 1)
    timeline = read_json(root/'timeline.json')
    timeline[0]['visual_text'] = '已有课件文本'
    inventory = build_ocr_inventory(root, timeline)
    assert inventory['summary']['executed'] == 0
    assert inventory['summary']['existing_evidence'] == 1
    assert not inventory['full_video_page_coverage_verified']


def test_clear_recapture_is_selected_and_old_image_receipt_cannot_survive(tmp_path):
    root=bundle(tmp_path,1)
    timeline=read_json(root/'timeline.json')
    old=backfill._first_existing_image(root,timeline[0])
    from video_knowledge_pipeline.ocr_inventory import record_ocr_result
    record_ocr_result(root,timeline,{'index':1,'ok':True,'text':'旧图文字','executed':True},engine='fixture',language='chi_sim')
    clear=root/'clear.png'
    clear.write_bytes(b'clearer pixels with new visible text')
    timeline[0]['assets']=[{'path':str(old)}]
    timeline[0]['recaptured_frame_path']=str(clear)
    assert backfill._first_existing_image(root,timeline[0])==clear
    result=build_ocr_inventory(root,timeline)
    assert result['pending_indexes']==[1]
    assert result['items'][0]['usable_text'] is False


def test_new_engine_never_borrows_old_confidence(tmp_path):
    from video_knowledge_pipeline.ocr_inventory import record_ocr_result
    root=bundle(tmp_path,1)
    timeline=read_json(root/'timeline.json')
    record_ocr_result(root,timeline,{'index':1,'ok':True,'text':'旧文字','executed':True,
        'ocr_confidence':{'mean_confidence':30}},engine='tesseract_cli',language='chi_sim')
    record_ocr_result(root,timeline,{'index':1,'ok':True,'text':'新文字','executed':False},
        engine='imported_evidence',language='chi_sim')
    row=read_json(root/'ocr-execution-inventory.json')['items'][0]
    assert row['engine']=='imported_evidence'
    assert 'ocr_confidence' not in row


def test_explicit_image_bound_import_can_classify_unchanged_existing_text(tmp_path):
    from video_knowledge_pipeline.file_hash import sha256_file
    root=bundle(tmp_path,1)
    rows=read_json(root/'timeline.json')
    rows[0].update(visual_text='包装字',quality_issues=[])
    write_json(root/'timeline.json',rows)
    source=root/'observations.json'
    write_json(source,{'items':[{'index':1,'text':'包装字','text_role':'incidental_background',
        'source_sha256':sha256_file(root/'1.png')}]})
    backfill.run_ocr_backfill(root,input_json=source)
    assert read_json(root/'timeline.json')[0]['visual_text_role']=='incidental_background'


def test_human_corrected_visual_text_closes_ocr_obligation_without_claiming_ocr_ran(tmp_path):
    root = bundle(tmp_path, 1)
    timeline = read_json(root/'timeline.json')
    timeline[0].update(
        human_corrected_visual_text='画面有二维码和讲师简介。',
        review_status='corrected_visual_text',
        human_review={'status': 'corrected_visual_text', 'corrected_visual_text': '画面有二维码和讲师简介。'},
    )

    inventory = build_ocr_inventory(root, timeline)

    assert inventory['pending_indexes'] == []
    assert inventory['summary']['executed'] == 0
    assert inventory['summary']['human_confirmed_evidence'] == 1
    assert inventory['items'][0]['execution_status'] == 'human_confirmed_evidence'


def test_human_text_binding_survives_image_drift_until_a_new_review(tmp_path):
    from video_knowledge_pipeline.ocr_inventory import persist_ocr_inventory
    root = bundle(tmp_path, 1)
    timeline = read_json(root/'timeline.json')
    timeline[0].update(human_corrected_visual_text='合成画面文字',
        review_status='corrected_visual_text',
        human_review={'status': 'corrected_visual_text', 'reviewed_at': '2026-01-01'})
    assert persist_ocr_inventory(root, timeline)['summary']['human_confirmed_evidence'] == 1
    (root/'1.png').write_bytes(b'changed synthetic pixels')
    first = persist_ocr_inventory(root, timeline)
    assert first['pending_indexes'] == [1]
    assert first['items'][0]['human_confirmed_visual_text'] is False
    # Saving a new inventory must not relabel the same old human review as new.
    assert persist_ocr_inventory(root, timeline)['pending_indexes'] == [1]
    timeline[0]['human_review']['reviewed_at'] = '2026-01-02'
    assert persist_ocr_inventory(root, timeline)['summary']['human_confirmed_evidence'] == 1


def test_audio_only_intervals_do_not_create_missing_ocr_obligations(tmp_path):
    root = bundle(tmp_path, 1)
    timeline = read_json(root/'timeline.json')
    timeline.append({'index': 2, 'start': 2, 'end': 3, 'transcript': '只有录音', 'visual_route': 'audio_only'})
    timeline.append({'index': 3, 'start': 3, 'end': 4, 'quality_issues': ['missing_frame']})
    inventory = build_ocr_inventory(root, timeline)
    assert inventory['summary']['required'] == 2
    assert inventory['pending_indexes'] == [1, 3]
    assert inventory['items'][1]['execution_status'] == 'blocked'


def test_completed_empty_is_recovery_not_failure_or_no_text_proof(tmp_path, monkeypatch):
    root = bundle(tmp_path, 2)
    monkeypatch.setattr(backfill, '_run_captiocr_candidates', lambda candidates, **kw: (
        {'available': True, 'name': 'fixture'},
        [dict(c, executed=True, ok=False, text='', authoritative_observation=True,
              empty_recovery_attempted=True) for c in candidates]))
    result = backfill.run_ocr_backfill(root, execute=True)
    assert result['status'] == 'empty_results_need_recovery'
    assert not result['ok']
    assert result['summary']['failed'] == 0
    assert result['summary']['empty'] == 2
    assert result['summary']['whole_bundle']['remaining'] == 0
    again = backfill.run_ocr_backfill(root, execute=True)
    assert again['summary']['total'] == 0
    assert again['status'] == 'empty_results_need_recovery'


def test_recovery_candidate_is_not_promoted_to_screen_text(tmp_path, monkeypatch):
    root = bundle(tmp_path, 1)
    monkeypatch.setattr(backfill, '_run_captiocr_candidates', lambda candidates, **kw: (
        {'available': True, 'name': 'fixture'},
        [dict(c, executed=True, ok=False, text='纹理误识别', authoritative_observation=True,
              recognition_status='suspicious', empty_recovery_attempted=True) for c in candidates]))
    result = backfill.run_ocr_backfill(root, execute=True)
    assert result['status'] == 'suspicious_results_need_review'
    assert result['summary']['whole_bundle']['executed'] == 1
    assert result['summary']['whole_bundle']['suspicious'] == 1
    assert result['summary']['whole_bundle']['usable_text'] == 0
    assert not read_json(root/'timeline.json')[0].get('visual_text')


def test_coverage_exposes_execution_without_promoting_empty_ocr(tmp_path, monkeypatch):
    from video_knowledge_pipeline.knowledge_coverage import audit_knowledge_coverage
    root = bundle(tmp_path, 2)
    monkeypatch.setattr(backfill, '_run_captiocr_candidates', lambda candidates, **kw: (
        {'available': True, 'name': 'fixture'},
        [dict(c, executed=True, ok=False, text='', authoritative_observation=True) for c in candidates]))
    backfill.run_ocr_backfill(root, execute=True)
    report = audit_knowledge_coverage(root)['coverage']
    assert report['ocr_execution']['summary']['executed'] == 2
    assert report['ocr_execution']['summary']['usable_text'] == 0
    screen = next(c for c in report['channels'] if c['key'] == 'screen_text')
    assert screen['covered_count'] == 0
    assert screen['status'] != 'ok'
