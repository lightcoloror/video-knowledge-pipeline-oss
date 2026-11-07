"""Synthetic provider execution; no real HTTP, pixels or model inference."""
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import multimodal_frame_analyzer as semantic
from video_knowledge_pipeline import temporal_visual_analyzer as temporal
from video_knowledge_pipeline.canonical_json import canonical_json_sha256
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.visual_evidence import visual_recognition_provenance


def _run(tmp_path, monkeypatch, *, kind='semantic', response_model='actual-model', drift=False, external=False):
    module = semantic if kind == 'semantic' else temporal
    root = tmp_path / 'bundle'
    (root / 'assets').mkdir(parents=True)
    paths = []
    for i in range(1 if kind == 'semantic' else 5):
        path = root / 'assets' / f'frame-{i:02d}.jpg'
        path.write_bytes(f'synthetic-frame-{i}'.encode())
        paths.append(path)
    item = dict(start=0, end=8, transcript='spoken context', visual_text='OCR context',
                visual_route='semantic_frame' if kind == 'semantic' else 'temporal_sequence',
                frame_paths=[p.relative_to(root).as_posix() for p in paths],
                temporal_frame_paths=[p.relative_to(root).as_posix() for p in paths])
    (root / 'manifest.json').write_text(json.dumps({'schema': 'lecture_webui_bundle.v1'}))
    (root / 'timeline.json').write_text(json.dumps([item]))
    config = dict(provider='openai_compatible', model='requested-model', execution_location='remote')
    monkeypatch.setattr(module, 'resolve_vision_task_execution_route', lambda *a, **kw: {'legacy_fallback_blocked': False, 'provider_config': config})
    monkeypatch.setattr(module, 'resolve_vision_execution_profile', lambda **kw: dict(provider_config=config, multimodal_limit=1, temporal_limit=1, frame_count=5))
    monkeypatch.setattr(module, 'resolve_provider_config', lambda cfg: cfg)
    monkeypatch.setattr(module, '_execution_control', lambda *a, **kw: ({}, {}))
    calls = []
    probe = root / 'assets' / 'probe.jpg'
    if external:
        probe = tmp_path / 'outside.jpg'
    probe.write_bytes(b'synthetic resized/contact-sheet')
    monkeypatch.setattr(module, 'prepare_image_probe', lambda *a, **kw: {'image_paths': [str(probe)]})

    def call(**kwargs):
        calls.append(kwargs)
        payload = dict(objects=['slide'], actions=['display'], confidence=0.9,
                       evidence_frame_paths=[str(p) for p in paths],
                       event_sequence=['static_sequence'], state_changes=['none'],
                       model_actual='forged-body-model', executor='forged',
                       recognition_provenance={'frame_sha256s': ['f' * 64]})
        if drift:
            paths[0].write_bytes(b'changed while request in flight')
        return dict(ok=True, complete=True, content=json.dumps(payload),
                    runtime_result={'response': {'model': response_model},
                                    'deployment': {'model': 'configured-singleton', 'selection': 'singleton_route'}})

    monkeypatch.setattr(module, 'call_vision_model_with_retries', call)
    fn = semantic.run_multimodal_frame_analysis if kind == 'semantic' else temporal.run_temporal_visual_analysis
    timeline_before = (root / 'timeline.json').read_bytes()
    result = fn(root, execute=True, limit=1, indexes=[1], provider_config=config)
    result['_fixture_timeline_before'] = timeline_before
    return root, paths, probe, calls, result


@pytest.mark.parametrize('kind', ['semantic', 'temporal'])
def test_provider_inputs_and_reported_model_reach_summary_projection(tmp_path, monkeypatch, kind):
    root, paths, probe, calls, result = _run(tmp_path, monkeypatch, kind=kind)
    assert len(calls) == 1
    field = 'visual_understanding' if kind == 'semantic' else 'temporal_visual_understanding'
    item = json.loads((root / 'timeline.json').read_text())[0]
    payload = item[field]
    assert payload['model_actual'] == 'actual-model'
    assert payload['model_requested'] == 'requested-model'
    record = payload['recognition_provenance']
    assert record['frame_sha256s'] == sorted(sha256_file(p) for p in paths)
    assert record['sent_image_sha256s'] == [sha256_file(probe)]
    assert record['context_text_sha256s'] == sorted([canonical_json_sha256('spoken context'), canonical_json_sha256('OCR context')])
    assert record['prompt_sha256'] == canonical_json_sha256(calls[0]['prompt'])
    assert record['independence_is_verified'] is False
    projected = visual_recognition_provenance(item, [field], text='visible result')
    assert projected['frame_sha256s'] == record['frame_sha256s']
    assert projected['sent_image_sha256s'] == record['sent_image_sha256s']
    assert projected['model_actual'] == 'actual-model'


def test_missing_reported_model_never_falls_back_to_request_or_body(tmp_path, monkeypatch):
    root, *_ = _run(tmp_path, monkeypatch, response_model='')
    payload = json.loads((root / 'timeline.json').read_text())[0]['visual_understanding']
    assert payload['model_actual'] == ''
    audit = json.loads((root / 'vision-analysis-runs.jsonl').read_text().splitlines()[-1])
    assert audit['model_actual'] == ''


@pytest.mark.parametrize('kind', ['semantic', 'temporal'])
def test_changed_input_keeps_raw_response_but_no_formal_result(tmp_path, monkeypatch, kind):
    root, paths, probe, calls, result = _run(tmp_path, monkeypatch, kind=kind, drift=True)
    field = 'visual_understanding' if kind == 'semantic' else 'temporal_visual_understanding'
    assert field not in json.loads((root / 'timeline.json').read_text())[0]
    assert len(calls) == 1
    assert result['items'][0]['ok'] is False
    assert result['items'][0]['raw_content']
    assert 'input_identity_drift' in result['items'][0]['error']
    assert (root / 'timeline.json').read_bytes() == result['_fixture_timeline_before']


def test_probe_outside_bundle_is_rejected_before_call(tmp_path, monkeypatch):
    root, paths, probe, calls, result = _run(tmp_path, monkeypatch, external=True)
    assert calls == []
    assert result['items'][0]['ok'] is False
    assert 'visual_understanding' not in json.loads((root / 'timeline.json').read_text())[0]
    assert result['summary']['status'] == 'input_identity_blocked'
    assert (root / 'timeline.json').read_bytes() == result['_fixture_timeline_before']


@pytest.mark.parametrize('response,expected', [
    ({'raw_response': {'model': 'openai-actual'}}, 'openai-actual'),
    ({'raw_response': {'modelVersion': 'gemini-actual'}}, 'gemini-actual'),
    ({'runtime_result': {'deployment': {'model': 'singleton-config'}}}, ''),
    ({'raw_response': {'model': 'x' * 513}}, ''),
])
def test_actual_identity_only_uses_bounded_response_metadata(tmp_path, response, expected):
    from video_knowledge_pipeline.evidence_lineage import call_visual_with_provenance
    path = tmp_path / 'frame.png'
    path.write_bytes(b'synthetic')
    result = call_visual_with_provenance(root=tmp_path, candidate={}, original_image_paths=[str(path)],
        sent_image_paths=[str(path)], prompt='prompt', provider_config={'model': 'request'},
        invoke=lambda: dict(ok=True, **response))
    assert result['model_actual'] == expected


@pytest.mark.parametrize('case', ['missing', 'over_count', 'wrong_type', 'sent_drift'])
def test_bounded_file_admission_and_postcall_drift(tmp_path, case):
    from video_knowledge_pipeline.evidence_lineage import call_visual_with_provenance
    frame = tmp_path / 'frame.png'
    probe = tmp_path / ('probe.mp4' if case == 'wrong_type' else 'probe.png')
    frame.write_bytes(b'frame')
    if case != 'missing':
        probe.write_bytes(b'probe')
    called = []
    def invoke():
        called.append(True)
        if case == 'sent_drift':
            probe.write_bytes(b'drift')
        return dict(ok=True, content='raw response')
    result = call_visual_with_provenance(root=tmp_path, candidate={}, original_image_paths=[str(frame)],
        sent_image_paths=[str(probe)] * (33 if case == 'over_count' else 1),
        prompt='prompt', provider_config={}, invoke=invoke)
    assert result['ok'] is False
    assert len(called) == (1 if case == 'sent_drift' else 0)


def test_matching_prepared_image_is_correlated_even_with_distinct_declared_roots():
    from video_knowledge_pipeline.evidence_lineage import aggregate_source_votes
    rows = [dict(evidence_id=str(i), source_type='visual_understanding',
                 original_source_sha256=str(i) * 64,
                 recognition_provenance={'sent_image_sha256s': ['f' * 64]}) for i in (1, 2)]
    result = aggregate_source_votes(rows, [dict(vote='supports_candidate', source_weight=w) for w in (70, 65)])
    assert result['candidate_weight'] == 70
    assert result['independence_is_verified'] is False
