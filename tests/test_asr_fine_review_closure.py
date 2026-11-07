from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_consensus import build_asr_consensus
from video_knowledge_pipeline.asr_diff_adjudication import (
    apply_asr_diff_adjudication, asr_diff_review_status, build_asr_diff_adjudication,
)
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json


def prepare(root):
    p,s,c=(root/name for name in ('primary.json','secondary.json','canonical.json'))
    write_json(p,{'segments':[{'start':0,'end':5,'text':'金额100元。'},
        {'start':7,'end':12,'text':'再检查。'},{'start':13,'end':13.1,'text':'？'}]})
    write_json(s,{'segments':[{'start':0,'end':12,'text':'金额一百元再检查'}]})
    write_json(c,read_json(p))
    write_json(root/'manifest.json',{'source_arbitrated_transcript_json':str(c)})
    write_json(root/'timeline.json',[])
    build_asr_consensus(root,primary_transcript=p,secondary_transcript=s,agreement_sample_size=0)
    pack=build_asr_diff_adjudication(root,cluster_token_gap=0)
    comparison=read_json(root/'asr-consensus.json')
    write_json(root/'dual-asr-review.json',{'ok':True,'status':'dual_inputs_compared','consensus':{
        'review_required_count':comparison['review_required_count']},'stages':{
            role:{'path':str(path),'sha256':sha256_file(path)} for role,path in [('primary',p),('secondary',s)]}})
    pack=build_asr_diff_adjudication(root,cluster_token_gap=0)
    evidence=root/'reference.md'
    evidence.write_text('金额100元（即一百元）；再检查。仅保留问号原始证据。',encoding='utf-8')
    path=root/'decisions.json'
    payload={'reviewer':'fixture_agent_not_human','evidence_bindings':{
        'reference':{'path':str(evidence),'sha256':sha256_file(evidence)}},
        'rows':[{'diff_id':d['diff_id'],'choice':'keep_primary','confidence':.99,
            'reason':'来源支持保留原计数或非词语标点，不删除证据。','evidence_refs':['reference']}
            for d in pack['differences']]}
    write_json(path,payload)
    return pack,path,payload


def test_fine_units_have_native_cue_times_and_retain_punctuation_window(tmp_path):
    pack,_,_=prepare(tmp_path)
    assert len(pack['review_window_ids'])==2
    assert pack['difference_count']==2
    numeric=next(d for d in pack['differences'] if d['primary_text']=='100')
    assert numeric['estimated_time']=={'start':0.,'end':5.}
    assert numeric['time_basis']=='primary_provider_segment'
    assert any(d['primary_text']=='？' for d in pack['differences'])
    assert all(c['audio_review_window']['duration_seconds']<15 for c in pack['clusters'])


def test_closed_review_is_bound_and_never_changes_canonical_or_raw(tmp_path):
    pack,path,_=prepare(tmp_path)
    before={r:Path(b['path']).read_bytes() for r,b in pack['source_bindings'].items()}
    result=apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True)
    assert result['ok'] and result['remaining_count']==0 and result['resolved_count']==2
    assert asr_diff_review_status(tmp_path)['ok']
    assert all(Path(pack['source_bindings'][r]['path']).read_bytes()==data for r,data in before.items())
    assert not (tmp_path/'asr-consensus-patched-transcript.json').exists()
    from video_knowledge_pipeline.workflow_completion import workflow_completion_status
    stage=next(r for r in workflow_completion_status(tmp_path)['stages'] if r['key']=='secondary_asr_and_differences')
    assert stage['status']=='completed'
    assert stage['evidence']['raw_aggregate_review_count']==2


def test_partial_review_progress_is_recomputed_without_closing_aggregate(tmp_path):
    _,path,payload=prepare(tmp_path)
    payload['rows'].pop()
    write_json(path,payload)
    result=apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True)
    assert result['status']=='review-required' and result['resolved_count']==1
    current=asr_diff_review_status(tmp_path)
    assert not current['ok'] and current['remaining_count']==1 and current['resolved_count']==1
    from video_knowledge_pipeline.workflow_completion import workflow_completion_status
    stage=next(r for r in workflow_completion_status(tmp_path)['stages'] if r['key']=='secondary_asr_and_differences')
    assert stage['status']=='review-required'
    assert stage['evidence']['raw_aggregate_review_count']==2
    assert stage['evidence']['fine_review_resolved_count']==1
    (tmp_path/'reference.md').write_text('changed evidence',encoding='utf-8')
    assert asr_diff_review_status(tmp_path)['status']=='stale_or_invalid'


@pytest.mark.parametrize('change',['missing','duplicate','unknown','non_object','no_evidence','low_confidence','nan','no_reason','unbound_ref','invent_number'])
def test_incomplete_or_unsupported_decision_cannot_close_even_keep_primary(tmp_path,change):
    _,path,payload=prepare(tmp_path)
    row=payload['rows'][0]
    if change=='missing':
        payload['rows'].pop()
    elif change=='duplicate':
        payload['rows'].append(dict(row))
    elif change=='unknown':
        row['diff_id']='unknown'
    elif change=='non_object':
        payload['rows'].append(7)
    elif change=='no_evidence':
        row['evidence_refs']=[]
    elif change=='low_confidence':
        row['confidence']=.2
    elif change=='nan':
        row['confidence']=float('nan')
    elif change=='no_reason':
        row['reason']=''
    elif change=='unbound_ref':
        row['evidence_refs']=['not_registered']
    elif change=='invent_number':
        row.update(choice='use_canonical',resolved_text='999')
    write_json(path,payload)
    result=apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True)
    assert not result['ok'] and result['remaining_count']>0
    assert not asr_diff_review_status(tmp_path)['ok']


@pytest.mark.parametrize('role',['primary','secondary','canonical','consensus','dual_workflow','evidence','decisions','pack'])
def test_post_review_hash_drift_invalidates_closure(tmp_path,role):
    pack,path,_=prepare(tmp_path)
    assert apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True)['ok']
    target=Path(pack['source_bindings'][role]['path']) if role in pack['source_bindings'] else {
        'evidence':tmp_path/'reference.md','decisions':path,'pack':tmp_path/'asr-consensus-adjudication-pack.json'}[role]
    target.write_bytes(target.read_bytes()+b'\n')
    assert not asr_diff_review_status(tmp_path)['ok']


def test_same_numeric_prose_cannot_hide_missing_parent_window(tmp_path):
    _,path,_=prepare(tmp_path)
    pack_path=tmp_path/'asr-consensus-adjudication-pack.json'
    pack=read_json(pack_path)
    pack['differences']=pack['differences'][:-1]
    write_json(pack_path,pack)
    assert not apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True)['ok']


def test_word_timing_requires_matching_native_word_text(tmp_path):
    primary=tmp_path/'p.json'
    secondary=tmp_path/'s.json'
    write_json(primary,{'segments':[{'start':5,'end':10,'text':'正文'}]})
    write_json(secondary,{'segments':[{'start':0,'end':10,'text':'提示正文','metadata':{'words':[
        {'word':'提示','start':0.1,'end':1.1},{'word':'正文','start':5,'end':8}]}}]})
    write_json(tmp_path/'manifest.json',{})
    build_asr_consensus(tmp_path,primary_transcript=primary,secondary_transcript=secondary,agreement_sample_size=0)
    pack=build_asr_diff_adjudication(tmp_path)
    first=pack['differences'][0]
    assert first['time_basis']=='secondary_native_word_alignment'
    assert first['estimated_time']=={'start':.1,'end':1.1}


def test_source_media_drift_invalidates_closed_text_review(tmp_path):
    _,path,_=prepare(tmp_path)
    media=tmp_path/'audio.wav'
    media.write_bytes(b'synthetic media bytes')
    comparison=read_json(tmp_path/'asr-consensus.json')
    comparison['media_path']=str(media)
    write_json(tmp_path/'asr-consensus.json',comparison)
    build_asr_diff_adjudication(tmp_path,cluster_token_gap=0)
    assert apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True)['ok']
    media.write_bytes(b'changed media bytes')
    assert not asr_diff_review_status(tmp_path)['ok']


def test_review_archives_source_bytes_without_overwriting_earlier_evidence(tmp_path):
    _,path,_=prepare(tmp_path)
    assert apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True)['ok']
    indexes=list((tmp_path/'runs/asr-adjudication-reviews').glob('*/evidence-index.json'))
    assert len(indexes)==1
    history=read_json(indexes[0])
    assert any(row['role']=='source-canonical' for row in history)
    assert any(row['role']=='evidence-reference' for row in history)
    (tmp_path/'reference.md').write_text('new evidence, old archive must survive',encoding='utf-8')
    assert not asr_diff_review_status(tmp_path)['ok']
    assert all(sha256_file(tmp_path/row['path'])==row['sha256'] for row in history)


@pytest.mark.parametrize('unsafe', ['promotion','no_evidence','lower_threshold'])
def test_review_only_cannot_bypass_gate_via_legacy_switches(tmp_path,unsafe):
    _,path,_=prepare(tmp_path)
    if unsafe=='promotion':
        with pytest.raises(ValueError,match='forbids promotion'):
            apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True,promote=True)
    elif unsafe=='no_evidence':
        with pytest.raises(ValueError,match='requires evidence'):
            apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True,require_evidence=False)
    else:
        assert not apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True,min_confidence=.1)['ok']


def test_failed_higher_threshold_never_becomes_success_via_status_reader(tmp_path):
    _,path,payload=prepare(tmp_path)
    payload['rows'][0]['confidence']=.8
    write_json(path,payload)
    result=apply_asr_diff_adjudication(tmp_path,decisions_json=path,review_only=True,min_confidence=.95)
    assert not result['ok'] and result['min_confidence']==.95
    assert not asr_diff_review_status(tmp_path)['ok']


def test_canonical_pointer_change_invalidates_closed_review_even_with_old_bytes_retained(tmp_path):
    _, path, _ = prepare(tmp_path)
    assert apply_asr_diff_adjudication(tmp_path, decisions_json=path, review_only=True)['ok']
    original = tmp_path / 'canonical.json'
    replacement = tmp_path / 'replacement-canonical.json'
    replacement.write_bytes(original.read_bytes())
    manifest = read_json(tmp_path / 'manifest.json')
    manifest['source_arbitrated_transcript_json'] = str(replacement)
    write_json(tmp_path / 'manifest.json', manifest)
    assert not asr_diff_review_status(tmp_path)['ok']
