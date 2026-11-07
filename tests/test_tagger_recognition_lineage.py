"""Offline producer/import contracts; no RAM weights or image inference."""
import json
from pathlib import Path

import pytest

from test_video_structure_and_general_tagger import _bundle
from video_knowledge_pipeline import general_tagger_adapter as tagger
from video_knowledge_pipeline.storage import read_json, write_json


@pytest.fixture(autouse=True)
def isolate_model_discovery(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'synthetic.pth'
    checkpoint.write_bytes(b'synthetic checkpoint, not weights')
    monkeypatch.setattr(tagger, 'general_tagger_status', lambda **k: {'blockers':[]})
    monkeypatch.setattr(tagger, '_source_root', lambda *a, **k: None)
    monkeypatch.setattr(tagger, '_checkpoint', lambda *a, **k: checkpoint)


@pytest.fixture
def local_bundle(tmp_path):
    root, _ = _bundle(tmp_path)
    return root


def output(_path):
    return {'tags_zh':['演示'], 'tags_en':['presentation'], 'model_actual':'synthetic-ram', 'executor':'synthetic'}


def test_frame_changed_during_tagger_is_not_imported(local_bundle):
    root = local_bundle
    before = (root/'timeline.json').read_bytes()
    def drift(path):
        path.write_bytes(b'changed synthetic frame')
        return output(path)
    result = tagger.run_general_tagger(root, execute=True, _inference_backend=drift)
    assert result['status'] == 'failed'
    assert 'drift' in result['error']['message']
    assert (root/'timeline.json').read_bytes() == before
    assert 'timeline_import' not in result


def test_cached_language_and_time_binding_cannot_leak_previous_observation(local_bundle):
    root = local_bundle
    calls = []
    def backend(path):
        calls.append(path.name)
        return output(path)
    args = dict(execute=True,write=False,import_annotations=False,
                cache_dir=root/'synthetic-cache', _inference_backend=backend)
    first = tagger.run_general_tagger(root, prefer_language='zh', **args)
    timeline = read_json(root/'timeline.json')
    timeline[0]['start'], timeline[0]['end'] = 10,80
    write_json(root/'timeline.json',timeline)
    second = tagger.run_general_tagger(root,prefer_language='en',**args)
    assert first['annotations'][0]['text'] == '演示'
    assert second['annotations'][0]['text'] == 'presentation'
    assert second['annotations'][0]['start'] == 10
    assert second['annotations'][0]['end'] == 80


def test_producer_binding_reaches_timeline_and_does_not_invent_actual_model(local_bundle):
    root = local_bundle
    result = tagger.run_general_tagger(root,execute=True,run_id='synthetic-run',_inference_backend=output)
    annotation = read_json(root/'timeline.json')[0]['tagger_annotations'][0]
    assert annotation['executor'] == 'synthetic'
    assert annotation['model_actual'] == 'synthetic-ram'
    assert annotation['run_id'] == 'synthetic-run'
    record = annotation['recognition_provenance']
    assert record['frame_sha256s'] == [result['annotations'][0]['artifact_sha256']]
    assert record['independence_is_verified'] is False
    assert annotation['recognition_binding']['frame_id']
    assert annotation['candidate_only'] is True


@pytest.mark.parametrize('drift',['frame','timeline','manifest'])
def test_stale_generated_tagger_import_fails_closed(local_bundle,drift):
    from video_knowledge_pipeline.tagger_import import import_tagger_annotations
    root = local_bundle
    result = tagger.run_general_tagger(root,execute=True,import_annotations=False,_inference_backend=output)
    assert result['status'] == 'completed'
    if drift == 'frame':
        (root/'assets/frame.jpg').write_bytes(b'changed synthetic')
    else:
        path = root / (drift+'.json')
        value = read_json(path)
        if drift == 'timeline': value[0]['start'] = 2
        else: value['title'] = 'new identity'
        write_json(path,value)
    before = (root/'timeline.json').read_bytes()
    with pytest.raises(ValueError,match='tagger.*drift'):
        import_tagger_annotations(root,root/'exports/general-tagger.json')
    assert (root/'timeline.json').read_bytes() == before


def test_fixed_ram_source_interface_runs_without_loading_torch():
    import ast
    import os
    from contextlib import nullcontext
    from types import SimpleNamespace
    source_root = os.environ.get('VKP_RAM_SOURCE')
    if not source_root:
        pytest.skip('explicit fixed RAM source required; no model download')
    source = Path(source_root)/'ram/inference.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name == 'inference_ram')
    namespace = {'torch':SimpleNamespace(no_grad=nullcontext)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    model = SimpleNamespace(generate_tag=lambda image: (['presentation'],['演示']))
    assert namespace['inference_ram']('synthetic tensor',model) == ('presentation','演示')


def test_cache_origin_survives_import_and_correction(local_bundle):
    from types import SimpleNamespace
    from video_knowledge_pipeline.transcript_semantic_correction import _evidence_for_cue
    root = local_bundle
    args = dict(execute=True, cache_dir=root/'synthetic-cache', _inference_backend=output)
    first = tagger.run_general_tagger(root, run_id='origin-run', import_annotations=False, **args)
    second = tagger.run_general_tagger(root, run_id='reuse-run', **args)
    assert second['cache_hit_count'] == len(first['annotations'])
    item = read_json(root/'timeline.json')[0]
    annotation = item['tagger_annotations'][0]
    assert annotation['recognition_origin'] == first['annotations'][0]['recognition_binding']
    evidence = _evidence_for_cue(SimpleNamespace(start=0,end=90,text='合成讲解'),item)
    rows = [row for row in evidence if row['source_type'] == 'tagger' and row.get('recognition_provenance')]
    assert rows
    assert any(row['recognition_provenance']['model_actual'] == 'synthetic-ram' for row in rows)
    assert all(row['evidence_level'] == 'candidate_weak' for row in rows)


@pytest.mark.parametrize('field',['timestamp','frame_id','tags'])
def test_import_rejects_observation_binding_drift(local_bundle,field):
    from video_knowledge_pipeline.tagger_import import import_tagger_annotations
    root = local_bundle
    result = tagger.run_general_tagger(root,execute=True,import_annotations=False,_inference_backend=output)
    result['annotations'][0][field] = {'timestamp':17,'frame_id':'wrong-frame','tags':['虚构内容']}[field]
    path = root/'tampered.json'
    write_json(path,result)
    before = (root/'timeline.json').read_bytes()
    with pytest.raises(ValueError,match='tagger.*drift'):
        import_tagger_annotations(root,path)
    assert (root/'timeline.json').read_bytes() == before


@pytest.mark.parametrize('actual',['synthetic-ram',''])
def test_temporal_producer_reaches_final_summary_without_invented_model(tmp_path,actual):
    from test_temporal_tag_delta import _bundle_with_sequences
    from video_knowledge_pipeline.temporal_tag_delta import run_temporal_tag_delta
    from video_knowledge_pipeline.video_moment_index import build_video_moment_index
    from video_knowledge_pipeline.smart_summary_section_workflow import _moment_citations
    from video_knowledge_pipeline.smart_summary_global_reduce import _chapter_fact_pack
    from video_knowledge_pipeline.summary_citation_audit import build_citation_index
    root = _bundle_with_sequences(tmp_path)
    result = run_temporal_tag_delta(root,execute_tagger=True,frame_count=5,
        _inference_backend=lambda path: {'tags_zh':['课件'],'model_actual':actual,'executor':'synthetic'})
    row = result['items'][0]
    assert row['decision'] == 'coarse_weak_evidence'
    assert row['model_actual'] == actual
    assert len(row['raw_observations']) == 5
    assert all(observation.get('recognition_binding') for observation in row['raw_observations'])
    moment = build_video_moment_index(root,write=False)
    section = {'section_id':'chapter-0001','title':'合成章节','evidence':{},
               'citations':_moment_citations(moment,0,10,limit=8)}
    pack = _chapter_fact_pack(root,[section],{'sections':[section]})
    index = build_citation_index(pack)
    assert not index.errors
    refs = [ref for ref in index.refs.values() if ref.get('observation_kind') == 'continuous_tag_weak_evidence']
    assert len(refs) == 1
    record = refs[0]['recognition_provenance']
    assert record['model_actual'] == actual
    assert len(record['frame_sha256s']) == 5
    assert refs[0]['evidence_level'] == 'candidate_weak'
    assert record['independence_is_verified'] is False


def test_gap_tag_summary_cannot_be_read_as_weak_fact():
    from video_knowledge_pipeline.smart_summary_input_pack import _compact_evidence_item
    item = {'index':1,'temporal_tag_delta':{'decision':'review_gap','stable_tags':['课件'],
            'coarse_summary':'旧的可能课件描述'}}
    row = _compact_evidence_item(1,item,kind='continuous_tag_weak_evidence')
    assert row['evidence_level'] == 'unavailable'
    assert row['excerpt'] == ''


@pytest.mark.parametrize('mode',['representative','continuous'])
def test_frame_enumeration_checks_lexical_path_before_resolving(local_bundle,monkeypatch,mode):
    root = local_bundle
    # Exercise the shared reparse gate without requiring Windows symlink rights.
    guard = tagger.safe_path
    def guarded(root,value,**kwargs):
        if str(value).endswith('frame.jpg'):
            raise ValueError('synthetic reparse gate rejected')
        return guard(root,value,**kwargs)
    monkeypatch.setattr(tagger,'safe_path',guarded)
    with pytest.raises(ValueError,match='reparse'):
        tagger.bundle_frame_records(root,read_json(root/'timeline.json'),frame_mode=mode)


def test_damaged_cache_is_not_replayed_or_silently_reinferred(local_bundle):
    root = local_bundle
    calls = []
    def backend(path):
        calls.append(path)
        return output(path)
    args = dict(execute=True,write=False,import_annotations=False,
                cache_dir=root/'synthetic-cache',_inference_backend=backend)
    first = tagger.run_general_tagger(root,**args)
    frame_id = first['annotations'][0]['frame_id']
    cached = root/'synthetic-cache'/frame_id/'continuous-tag-frame/annotation.json'
    value = read_json(cached)
    value['tags'] = ['corrupted']
    write_json(cached,value)
    before = (root/'timeline.json').read_bytes()
    count = len(calls)
    second = tagger.run_general_tagger(root,**args)
    assert second['status'] == 'failed'
    assert 'drift' in second['error']['message']
    assert len(calls) == count
    assert (root/'timeline.json').read_bytes() == before
