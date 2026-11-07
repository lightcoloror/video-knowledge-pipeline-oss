import pytest

from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import write_json, read_json
from video_knowledge_pipeline.transcript_source_completeness import _bound_word_alignment_coverage
from video_knowledge_pipeline.transcript_quality_gate import run_transcript_quality_gate


def fixture(tmp_path):
    media = tmp_path/'source.wav'
    media.write_bytes(b'synthetic audio')
    transcript = tmp_path/'normalized-transcript.json'
    segments = [{'start':1,'end':3,'text':'甲乙丙丁'}]
    write_json(transcript, {'segments':segments})
    raw = {'input':str(media),'duration_seconds':4}
    path = tmp_path/'alignment.json'
    write_json(path, {'schema':'video_knowledge_pipeline.qwen3_forced_aligner_output.v1',
        'status':'completed','ok':True,'input_path':str(media),'input_sha256':sha256_file(media),
        'transcript_path':str(transcript),'transcript_sha256':sha256_file(transcript),
        'timing_resolution_seconds':.08,'words':[
            {'text':text,'start':start,'end':end} for text,start,end in
            [('甲',1,1.42),('乙',1.5,1.92),('丙',2,2.42),('丁',2.5,3)]]})
    entry = {'preset':'qwen3-forced-aligner','path':str(path),
             'sidecar_sha256':sha256_file(path),'transcript_sha256':sha256_file(transcript)}
    manifest = {'asr_alignment_sidecars':[entry]}
    vad = {'verified':True,'segments':[{'start':1,'end':3}], 'summary':{
        'vad_settings':{'speech_pad_ms':0,'min_silence_duration_ms':100,'sample_rate_hz':16000},
        'upstream':{'model':'silero_v5_bundled_onnx'}}}
    return transcript,segments,raw,manifest,vad,path


def assess(tmp_path, args):
    return _bound_word_alignment_coverage(tmp_path,*args[:5])


def replace_words(args, words):
    path=args[5]
    data=read_json(path)
    data['words']=words
    write_json(path,data)
    args[3]['asr_alignment_sidecars'][0]['sidecar_sha256']=sha256_file(path)


def test_quantized_word_boundaries_keep_raw_measurement_and_original_thresholds(tmp_path):
    args=fixture(tmp_path)
    result=assess(tmp_path,args)
    assert result['strict_coverage']['coverage_ratio'] < .95
    assert result['status']=='passed'
    assert result['coverage']['minimum_coverage_ratio']==.95
    assert result['coverage']['minimum_gap_seconds']==2
    assert result['measurement_collar_seconds']==pytest.approx(.112)
    assert result['text_correctness_verified'] is False


def test_real_long_missing_interval_still_fails(tmp_path):
    args=fixture(tmp_path)
    args[4]['segments']=[{'start':0,'end':4}]
    replace_words(args,[{'text':'甲乙','start':0,'end':.25}, {'text':'丙丁','start':3.75,'end':4}])
    result=assess(tmp_path,args)
    assert result['status']=='degraded'
    assert result['coverage']['gaps'][0]['duration_seconds'] > 3


@pytest.mark.parametrize('change', ['transcript','media','sidecar','padded_vad','incomplete_text','coarse_clock'])
def test_unbound_or_unsuitable_evidence_does_not_reconcile(tmp_path,change):
    args=fixture(tmp_path)
    if change=='transcript':
        args[0].write_text('changed',encoding='utf-8')
    elif change=='media':
        (tmp_path/'source.wav').write_bytes(b'different audio')
    elif change=='sidecar':
        args[5].write_text('{}',encoding='utf-8')
    elif change=='padded_vad':
        args[4]['summary']['vad_settings']['speech_pad_ms']=400
    else:
        data=read_json(args[5])
        if change=='incomplete_text':
            data['words'].pop()
        else:
            data['timing_resolution_seconds']=1
        write_json(args[5],data)
        args[3]['asr_alignment_sidecars'][0]['sidecar_sha256']=sha256_file(args[5])
    assert assess(tmp_path,args)['status']!='passed'


def test_zero_length_words_never_become_whole_segment_coverage(tmp_path):
    args=fixture(tmp_path)
    replace_words(args,[{'text':'甲乙','start':1,'end':1},{'text':'丙丁','start':3,'end':3}])
    result=assess(tmp_path,args)
    assert result['status']=='degraded'
    assert result['coverage']['coverage_ratio']==0
    assert result['zero_duration_word_count']==2


@pytest.mark.parametrize('verified',[True,False])
def test_silent_media_margins_use_verified_speech_not_span(tmp_path,monkeypatch,verified):
    write_json(tmp_path/'manifest.json',{'duration_seconds':30,'normalized_transcript_json':'normalized-transcript.json'})
    write_json(tmp_path/'normalized-transcript.json',{'segments':[{'start':10,'end':20,'text':'完整保留了原片中间说出的话。'}]})
    monkeypatch.setattr('video_knowledge_pipeline.transcript_quality_gate.assess_transcript_source_completeness',
        lambda *a,**k: {'speech_completeness_verified':verified,'issues':[]})
    result=run_transcript_quality_gate(tmp_path,write=False)
    assert result['timeline_span_coverage']==pytest.approx(1/3,abs=1e-6)
    assert ('audio_coverage_low' in {r['kind'] for r in result['issues']}) is (not verified)
