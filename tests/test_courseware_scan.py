from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline import courseware_scan as scan
from video_knowledge_pipeline.courseware_material import read_scan, build_courseware_material
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import write_json


def planned(tmp_path, monkeypatch):
    root = tmp_path / 'bundle'
    root.mkdir()
    write_json(root / 'manifest.json', {})
    write_json(root / 'timeline.json', [])
    source = tmp_path / 'synthetic.bin'
    source.write_bytes(b'not real media; mocked probe')
    monkeypatch.setattr(scan, 'probe_video', lambda _: SimpleNamespace(duration_seconds=100, width=320, height=180, fps=5, sha256=sha256_file(source)))
    scan.prepare_courseware_scan(root, source, start=0, end=90, excluded_intervals=[{'start': 30, 'end': 40}], window_seconds=30)
    return root, source


def test_scan_covers_tail_exclusions_and_resume_without_replay(tmp_path, monkeypatch):
    root, source = planned(tmp_path, monkeypatch)
    calls = []
    def detector(_, window, threshold, timeout):
        calls.append(window['start'])
        return [window['start'], window['end'] - 0.05]
    monkeypatch.setattr(scan, '_scan_window', detector)
    first = scan.run_courseware_scan(root, max_windows=1)
    assert first['status'] == 'partial_scan'
    assert scan.run_courseware_scan(root, max_windows=16)['status'] == 'scan_complete'
    assert calls == [0, 40, 70]
    state = read_scan(root)
    assert state['candidates'][-1]['time'] > 89
    assert not any(30 <= p['time'] < 40 for p in state['candidates'])
    scan.run_courseware_scan(root)
    assert calls == [0, 40, 70]
    assert source.read_bytes() == b'not real media; mocked probe'


def test_timeout_checkpoint_requires_explicit_resume(tmp_path, monkeypatch):
    root, _ = planned(tmp_path, monkeypatch)
    calls = []
    def detector(_, window, threshold, timeout):
        calls.append(window['start'])
        if window['start'] == 40:
            raise subprocess.TimeoutExpired('synthetic ffmpeg', 1)
        return [window['start']]
    monkeypatch.setattr(scan, '_scan_window', detector)
    assert not scan.run_courseware_scan(root, max_windows=3)['ok']
    assert read_scan(root)['completed_windows'] == [0]
    with pytest.raises(ValueError, match='resume'):
        scan.run_courseware_scan(root)
    monkeypatch.setattr(scan, '_scan_window', lambda _, w, t, timeout: [w['start']])
    assert scan.run_courseware_scan(root, resume=True, max_windows=3)['status'] == 'scan_complete'


def test_low_disk_and_candidate_budget_do_not_claim_scan_complete(tmp_path, monkeypatch):
    root, _ = planned(tmp_path, monkeypatch)
    monkeypatch.setattr(scan, '_scan_window', lambda *_: pytest.fail('must not invoke decoder'))
    assert scan.run_courseware_scan(root, disk_budget_bytes=1)['status'] == 'blocked'
    assert read_scan(root)['completed_windows'] == []
    monkeypatch.setattr(scan, '_scan_window', lambda *_: [0, 1, 2])
    assert scan.run_courseware_scan(root, resume=True, max_candidates_per_window=2)['status'] == 'blocked'
    assert not read_scan(root)['candidates']


def test_source_drift_and_scope_change_are_rejected(tmp_path, monkeypatch):
    root, source = planned(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='identity differs'):
        scan.prepare_courseware_scan(root, source, start=0, end=50)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source hash'):
        scan.run_courseware_scan(root)


def test_independent_missed_animation_candidate_and_excluded_range(tmp_path, monkeypatch):
    root, _ = planned(tmp_path, monkeypatch)
    monkeypatch.setattr(scan, '_scan_window', lambda _, w, t, timeout: [w['start']])
    scan.run_courseware_scan(root, max_windows=16)
    value = {'scan_identity': read_scan(root)['identity'], 'reviewer': 'human', 'reason': 'missed animation footnote', 'time': 45.5, 'end': 45.8}
    result = scan.add_courseware_candidate(root, value)
    assert result['ok']
    assert read_scan(root)['candidates'][-1]['origin'] == 'independent_review'
    with pytest.raises(ValueError, match='outside'):
        scan.add_courseware_candidate(root, {**value, 'time': 35, 'end': 36})


def test_native_synthetic_video_scan_capture_and_revisit(tmp_path):
    from PIL import Image, ImageDraw
    from video_knowledge_pipeline.media_tools import resolve_media_tool
    ffmpeg = resolve_media_tool('ffmpeg')
    assert ffmpeg, 'Synthetic FFmpeg acceptance needs the existing local FFmpeg'
    root = tmp_path / 'bundle'
    root.mkdir()
    write_json(root / 'manifest.json', {})
    write_json(root / 'timeline.json', [])
    before = (root / 'timeline.json').read_bytes()
    frames = tmp_path / 'inputs'
    frames.mkdir()
    for n in range(4):
        im = Image.new('RGB', (320, 180), 'white')
        draw = ImageDraw.Draw(im)
        draw.text((30, 30), 'LIMIT 10; FOOTNOTE: ONLY IF ...', fill='black')
        if n in {1, 2}:
            draw.rectangle((20, 70, 100 + n * 40, 90), fill='black')
        im.save(frames / f'{n:02d}.png')
    source = tmp_path / 'synthetic.mkv'
    subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-framerate', '2', '-i', str(frames / '%02d.png'),
                    '-c:v', 'ffv1', str(source)], check=True, capture_output=True, timeout=20)
    original = sha256_file(source)
    scan.prepare_courseware_scan(root, source, start=0, end=2, window_seconds=1, threshold=0)
    assert scan.run_courseware_scan(root, max_windows=2)['status'] == 'scan_complete'
    state = read_scan(root)
    assert len(state['candidates']) == 4
    assert state['candidates'][-1]['time'] == 1.5
    assert scan.extract_courseware_candidates(root, limit=2)['captured_this_batch'] == 2
    assert scan.extract_courseware_candidates(root, limit=2)['captured_this_batch'] == 2
    assert scan.extract_courseware_candidates(root)['captured_this_batch'] == 0
    state = read_scan(root)
    for page in state['candidates']:
        with Image.open(root / page['frame']) as im:
            assert im.size == (320, 180)
    assert sha256_file(source) == original
    assert (root / 'timeline.json').read_bytes() == before
    assert not build_courseware_material(root, write=False)['complete']


def test_collect_preserves_visual_text_beyond_structured_caption(tmp_path):
    from test_courseware_material import fixture_bundle
    root = fixture_bundle(tmp_path)
    page = read_scan(root)['candidates'][0]
    write_json(root / 'courseware/ocr-bundle/timeline.json', [{
        'courseware_page_id': page['page_id'], 'courseware_frame_sha256': page['frame_sha256'],
        'frame_paths': [str(root / page['frame'])],
        'structured_visual': [{'type': 'table', 'markdown': '完整表格标题与单位。', 'rows': [[1, 20, 'percent']]}],
        'visual_text': '另有页脚限定：仅在满足条件时适用；不能忽略。',
    }])
    result = scan.collect_courseware_extractions(root, executor='synthetic', model_actual='fixture')
    blocks = result['material']['pages'][0]['blocks']
    assert len(blocks) == 2
    assert blocks[0]['structure']['rows'] == [[1, 20, 'percent']]
    assert '仅在满足条件' in blocks[1]['text']


def test_ocr_worker_failure_is_not_hidden_by_successful_capture(tmp_path, monkeypatch):
    from test_courseware_material import fixture_bundle
    from video_knowledge_pipeline import visual_structure_batch
    root = fixture_bundle(tmp_path)
    state = read_scan(root)
    state['source'].update(width=320, height=180, fps=2)
    scan.store_scan(root, state)
    monkeypatch.setattr(scan, '_check_source', lambda _: tmp_path / 'unused-synthetic-source')
    monkeypatch.setattr(visual_structure_batch, 'run_visual_structure_ebook_batches',
                        lambda *a, **k: {'ok': False, 'status': 'partial_failed', 'failed_indexes': [1]})
    original = (root / 'timeline.json').read_bytes()
    result = scan.extract_courseware_candidates(root, execute_ocr=True)
    assert not result['ok'] and result['status'] == 'ocr_partial_or_failed'
    assert result['ocr_result']['failed_indexes'] == [1]
    assert (root / 'timeline.json').read_bytes() == original
