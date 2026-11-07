from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from video_knowledge_pipeline import qwen_failed_window_recovery as recovery
from video_knowledge_pipeline import dual_asr_workflow as workflow
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json


def _fixture(root):
    media = root / 'media.wav'
    media.write_bytes(b'synthetic media only')
    command = [sys.executable, '-m', 'video_knowledge_pipeline.qwen3_asr_python_runner',
        '--input', str(media), '--output', str(root / 'partial.json'),
        '--model', 'fixture/qwen0.6B', '--forced-aligner', 'fixture/aligner',
        '--chunk-seconds', '300', '--chunk-indexes', '1', '--timestamp-basis', 'chunk_relative']
    partial = {'provider': 'qwen3-asr', 'status': 'degraded', 'model': 'fixture/qwen0.6B',
        'forced_aligner': 'fixture/aligner', 'input_path': str(media),
        'input_sha256': sha256_file(media), 'chunk_seconds': 300,
        'timestamp_output_basis': 'media_absolute', 'window_plan': {'duration_ms': 900000},
        'successful_chunk_indexes': [0, 2],
        'results': [{'chunk_index': index, 'window_start_seconds': index*300,
                    'window_end_seconds': (index+1)*300} for index in (0, 2)],
        'segments': [{'start': index*300+1, 'end': index*300+2, 'text': f'healthy {index}'} for index in (0, 2)],
        'failed_chunks': [{'chunk_index': 1, 'start': 300., 'end': 600.,
            'retry_exhausted': True, 'attempt_count': 2, 'retry_command': {'command': command}}]}
    partial_path = root / 'partial.json'
    write_json(partial_path, partial)
    activity = root / 'activity.json'
    write_json(activity, {'source_media_sha256': sha256_file(media),
        'speech_intervals': [{'start': 0, 'end': 900}],
        'evidence_kind': 'non_silent_audio_not_proven_speech', 'detector': 'fixture_silence'})
    return media, partial_path, activity


def _child(plan, root, indexes=None):
    indexes = indexes if indexes is not None else [r['index'] for r in plan['selected_windows']]
    payload = {'provider': 'qwen3-asr', 'status': 'completed',
        'model': plan['model_identity']['model'], 'forced_aligner': plan['model_identity']['forced_aligner'],
        'input_sha256': plan['media']['sha256'], 'chunk_seconds': plan['window_seconds'],
        'timestamp_output_basis': 'media_absolute', 'successful_chunk_indexes': indexes,
        'failed_chunks': [], 'results': [{'chunk_index': index,
            'window_start_seconds': index*60, 'window_end_seconds': (index+1)*60} for index in indexes],
        'segments': [{'start': index*60+1, 'end': index*60+2, 'text': f'recovered {index}'} for index in indexes]}
    path = root / 'recovered.json'
    write_json(path, payload)
    return path


def test_compose_only_missing_windows_preserves_sources_and_absolute_times(tmp_path):
    media, partial, activity = _fixture(tmp_path)
    original = partial.read_bytes()
    plan = recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery',
        media_path=media, speech_activity_json=activity)
    assert [row['index'] for row in plan['selected_windows']] == [5, 6, 7, 8, 9]
    child = _child(plan, tmp_path)
    result = recovery.compose_qwen_failed_window_recovery(plan, recovered_json=child)
    assert result['ok'] and result['status'] == 'completed_composed_reference'
    assert [row['start'] for row in result['segments']] == [1, 301, 361, 421, 481, 541, 601]
    assert result['single_run_execution_attested'] is False
    assert result['automatic_canonical_promotion_allowed'] is False
    assert result['speech_activity_binding']['evidence_kind'] == 'non_silent_audio_not_proven_speech'
    assert partial.read_bytes() == original
    assert Path(plan['retained_partial_json']).read_bytes() == original


@pytest.mark.parametrize('change', ['media', 'model', 'timestamp', 'unexpected_window'])
def test_rejects_changed_or_unbound_recovered_evidence(tmp_path, change):
    media, partial, _ = _fixture(tmp_path)
    plan = recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery')
    child = _child(plan, tmp_path)
    raw = read_json(child)
    if change == 'media':
        raw['input_sha256'] = '0'*64
    elif change == 'model':
        raw['model'] = 'different/model'
    elif change == 'timestamp':
        raw['segments'][0]['start'] = 1
    else:
        raw['successful_chunk_indexes'].append(0)
    write_json(child, raw)
    with pytest.raises(ValueError):
        recovery.compose_qwen_failed_window_recovery(plan, recovered_json=child)
    assert not Path(plan['composed_reference_json']).exists()


def test_original_retry_must_expire_and_smaller_windows_must_not_cross_healthy_audio(tmp_path):
    _, partial, _ = _fixture(tmp_path)
    raw = read_json(partial)
    raw['failed_chunks'][0]['retry_exhausted'] = False
    write_json(partial, raw)
    with pytest.raises(ValueError, match='retry budget'):
        recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery')
    raw['failed_chunks'][0].update(retry_exhausted=True, start=305)
    write_json(partial, raw)
    with pytest.raises(ValueError, match='healthy audio'):
        recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery')
    raw['window_plan'] = {}
    write_json(partial, raw)
    with pytest.raises(ValueError, match='native media duration'):
        recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery')


def test_budget_and_child_failures_remain_partial(tmp_path):
    _, partial, _ = _fixture(tmp_path)
    plan = recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery', max_windows=3)
    child = _child(plan, tmp_path, indexes=[5, 6])
    result = recovery.compose_qwen_failed_window_recovery(plan, recovered_json=child)
    assert result['status'] == 'partial_composed_reference' and not result['ok']
    assert {row['index'] for row in result['remaining_windows']} == {7, 8, 9}


def test_preview_never_executes_and_saved_child_drift_is_rejected(tmp_path, monkeypatch):
    _, partial, _ = _fixture(tmp_path)
    plan = recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery')
    monkeypatch.setattr(recovery, 'run_asr_plan', lambda *a, **k: pytest.fail('must not execute'))
    assert recovery.run_qwen_failed_window_recovery(plan)['status'] == 'planned'
    saved = read_json(Path(plan['child_plan_json']))
    saved['command'].append('--no-timestamps')
    write_json(Path(plan['child_plan_json']), saved)
    with pytest.raises(ValueError, match='child plan drift'):
        recovery.run_qwen_failed_window_recovery(plan, execute=True)
    tampered = copy.deepcopy(plan)
    tampered['child_plan']['command'][0] = 'different-executable'
    with pytest.raises(ValueError, match='execution contract drift'):
        recovery.run_qwen_failed_window_recovery(tampered)


def test_recovery_rejects_media_or_activity_changes(tmp_path):
    media, partial, activity = _fixture(tmp_path)
    raw = read_json(activity)
    raw['source_media_sha256'] = '0'*64
    write_json(activity, raw)
    with pytest.raises(ValueError, match='matching media hash'):
        recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery', speech_activity_json=activity)
    plan = recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery')
    media.write_bytes(b'changed media')
    with pytest.raises(ValueError, match='hash drift'):
        recovery.run_qwen_failed_window_recovery(plan)


def test_planning_does_not_overwrite_a_different_recovery_contract(tmp_path):
    _, partial, _ = _fixture(tmp_path)
    plan = recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery')
    original_plan = Path(plan['child_plan_json']).read_bytes()
    with pytest.raises(ValueError, match='new output directory'):
        recovery.plan_qwen_failed_window_recovery(partial, output_dir=tmp_path/'recovery', max_windows=2)
    assert Path(plan['child_plan_json']).read_bytes() == original_plan
    Path(plan['retained_partial_json']).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='retained recovery evidence'):
        recovery.run_qwen_failed_window_recovery(plan)


def test_supplied_comparison_reuses_bound_non_silent_sidecar_without_models(tmp_path, monkeypatch):
    media, _, activity = _fixture(tmp_path)
    primary, secondary = tmp_path/'primary.json', tmp_path/'secondary.json'
    for path in (primary, secondary):
        write_json(path, {'segments': [{'start': 0, 'end': 10, 'text': 'same words'}]})
    write_json(tmp_path/'dual-asr-review.json', {'speech_activity_json': str(activity)})
    monkeypatch.setattr(workflow, 'plan_asr_run', lambda *a, **k: pytest.fail('no recognition required'))
    result = workflow.run_dual_asr_workflow(tmp_path, media_path=media,
        primary_transcript=primary, secondary_transcript=secondary, execute=False)
    assert result['status'] == 'dual_inputs_compared' and result['activity_reused']
    coverage = result['consensus']['review_audit']['coverage']
    assert coverage['status'] == 'assessed'
    assert coverage['evidence_kind'] == 'non_silent_audio_not_proven_speech'
    assert coverage['transcript_accuracy_proven'] is False
    assert not result['dual_full_transcription_complete']


def test_workflow_retries_original_contract_then_recovers_and_reuses_composition(tmp_path, monkeypatch):
    media, partial, activity = _fixture(tmp_path)
    primary = tmp_path/'primary.json'
    write_json(primary, {'segments': [{'start': 0, 'end': 900, 'text': 'primary reference words'}]})
    raw = read_json(partial)
    raw['failed_chunks'][0]['retry_exhausted'] = False
    raw['failed_chunks'][0]['attempt_count'] = 1
    write_json(partial, raw)
    plan_path = tmp_path/'native-plan.json'
    write_json(plan_path, {'command': raw['failed_chunks'][0]['retry_command']['command'], 'available': True})
    calls=[]
    monkeypatch.setattr(workflow, 'plan_asr_run', lambda *a, **k: {'plan_path': str(plan_path)})
    def original_run(*args, **kwargs):
        calls.append('original_contract')
        value = read_json(partial)
        if len(calls) == 2:
            value['failed_chunks'][0].update(retry_exhausted=True, attempt_count=2)
            write_json(partial, value)
        return {'status': 'degraded', 'raw_output_json': str(partial)}
    monkeypatch.setattr(workflow, 'run_asr_plan', original_run)
    def smaller_run(path, **kwargs):
        calls.append('new_smaller_contract')
        child = read_json(Path(path))
        command = child['command']
        assert command[command.index('--chunk-indexes')+1] == '5,6,7,8,9'
        assert command[command.index('--chunk-seconds')+1] == '60'
        plan = read_json(Path(path).parent/'recovery-plan.json')
        produced = _child(plan, Path(path).parent)
        return {'status': 'ok', 'raw_output_json': str(produced)}
    monkeypatch.setattr(recovery, 'run_asr_plan', smaller_run)
    result = workflow.run_dual_asr_workflow(tmp_path, media_path=media,
        primary_transcript=primary, speech_activity_json=activity, execute=True)
    assert calls == ['original_contract', 'original_contract', 'new_smaller_contract']
    assert result['status'] == 'dual_inputs_compared'
    assert result['stages']['secondary']['status'] == 'composed_reference'
    assert result['stages']['secondary_recovery']['status'] == 'completed_composed_reference'
    assert not result['dual_full_transcription_complete'] and not result['quality_verified']
    again = workflow.run_dual_asr_workflow(tmp_path, media_path=media,
        primary_transcript=primary, speech_activity_json=activity, execute=True)
    assert again['stages']['secondary']['reused']
    assert len(calls) == 3
