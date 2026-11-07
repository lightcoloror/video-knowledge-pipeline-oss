from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline import dual_asr_workflow as workflow
from video_knowledge_pipeline import qwen3_asr_python_runner as runner
from video_knowledge_pipeline.storage import write_json


def test_unaligned_text_keeps_audio_bounds_without_word_precision():
    rows = runner._segments([], '独立转写', fallback_start=30, fallback_end=60)
    assert (rows[0]['start'], rows[0]['end']) == (30, 60)
    assert rows[0]['timestamp_precision'] == 'audio_window_only'
    from video_knowledge_pipeline.asr_adapter import _asr_metadata
    metadata = _asr_metadata(rows[0])
    assert metadata['timestamp_precision'] == 'audio_window_only'
    assert metadata['alignment'] == ''


@pytest.mark.parametrize('mode,available,want', [
    ('auto', False, False), ('auto', True, True),
    ('required', False, True), ('window', True, False),
])
def test_alignment_capability_bound_to_plan_receipt_and_resume(tmp_path, monkeypatch, mode, available, want):
    media = tmp_path/'audio.wav'
    media.write_bytes(b'fixture')
    primary = tmp_path/'primary.json'
    write_json(primary, {'segments': [{'start': 0, 'end': 10, 'text': '内容'}]})
    monkeypatch.setattr(workflow, '_local_qwen_aligner_path', lambda: None)
    monkeypatch.setattr(workflow, '_model_ready', lambda **k: {'ready': available})
    monkeypatch.setattr(workflow, 'probe_video', lambda *a: SimpleNamespace(duration_seconds=10))
    monkeypatch.setattr(workflow, 'probe_audio_silence', lambda *a, **k: {'ok': True, 'activity_intervals': [{'start': 0, 'end': 10}]})
    plans = []
    def plan(root, media, **kwargs):
        plans.append(kwargs)
        return {'plan_path': str(root/'plan.json')}
    def run(path, **kwargs):
        dest = Path(path).parent/'transcript.json'
        write_json(dest, {'segments': [{'start': 0, 'end': 10, 'text': '内容'}]})
        return {'status': 'ok', 'normalized': {'json_path': str(dest)}}
    monkeypatch.setattr(workflow, 'plan_asr_run', plan)
    monkeypatch.setattr(workflow, 'run_asr_plan', run)
    r = workflow.run_dual_asr_workflow(tmp_path, media_path=media, primary_transcript=primary,
            secondary_timestamps=mode, execute=True)
    assert plans[0]['qwen_timestamps'] is want
    assert r['stages']['secondary']['timestamp_capability']['use_forced_aligner'] is want
    assert r['resume_argv'][r['resume_argv'].index('--secondary-timestamps')+1] == mode
    assert not r['quality_verified']
    workflow.run_dual_asr_workflow(tmp_path, media_path=media, primary_transcript=primary,
            secondary_timestamps=mode, execute=True)
    assert len(plans) == 1
    other = 'window' if want else 'required'
    changed = workflow.run_dual_asr_workflow(tmp_path, media_path=media, primary_transcript=primary,
            secondary_timestamps=other, execute=True)
    assert changed['execution_id'] != r['execution_id']
    assert len(plans) == 2
