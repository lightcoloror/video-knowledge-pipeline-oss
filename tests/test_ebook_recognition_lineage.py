"""Actual VKP checkpoint/import, fake external OCR, only synthetic files."""
import json

import pytest

from video_knowledge_pipeline import visual_structure as vs
from video_knowledge_pipeline.canonical_json import canonical_json_sha256
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.visual_evidence import visual_recognition_provenance
from test_visual_structure_checkpoint import _write_bundle


TEXT = '# 合成课程\n\n- 第一项知识及适用条件\n- 第二项重要例外说明'


def _produce(tmp_path, monkeypatch, *, drift=False):
    root = tmp_path / 'bundle'
    frame = _write_bundle(root)
    monkeypatch.setattr(vs, '_ebook_call_tool', lambda: object())
    def flow(_tool, image_path, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / 'output.md'
        artifact.write_text(TEXT, encoding='utf-8')
        if drift:
            frame.write_bytes(b'changed synthetic frame')
        return dict(ok=True, job={'status': 'done'}, model_actual='untrusted-extra-field',
                    artifact={'path': str(artifact), 'artifact_type': 'markdown', 'text': TEXT})
    monkeypatch.setattr(vs, '_run_ebook_material_flow', flow)
    results = vs._run_ebook_pipeline_candidates(root, [{'index': 1, 'start': 0, 'end': 2, 'image_path': str(frame)}], timeout_seconds=1)
    checkpoint = root / 'visual-structure/timeline-0001/ebook_pipeline/result-checkpoint.json'
    return root, frame, checkpoint, results


def test_new_ocr_checkpoint_preserves_observed_input_to_summary_and_is_idempotent(tmp_path, monkeypatch):
    root, frame, checkpoint, results = _produce(tmp_path, monkeypatch)
    binding = results[0]['recognition_binding']
    assert binding['input']['frame_sha256'] == sha256_file(frame)
    assert binding['recognized_text_sha256'] == canonical_json_sha256(TEXT)
    assert binding['projected_text_sha256'] == canonical_json_sha256(TEXT.replace('\n\n', '\n'))
    first = vs.reconcile_ebook_pipeline_checkpoints(root)
    item = json.loads((root / 'timeline.json').read_text(encoding='utf-8'))[0]
    record = visual_recognition_provenance(item, ['visual_text'], text=item['visual_text'])
    assert record['frame_sha256s'] == [sha256_file(frame)]
    assert record['executor'] == 'ebook_markdown_pipeline'
    assert record['model_actual'] == ''  # upstream job contract doesn't report it
    assert record['independence_is_verified'] is False
    from test_visual_recognition_lineage import cue
    from video_knowledge_pipeline.transcript_semantic_correction import _evidence_for_cue
    from video_knowledge_pipeline.smart_summary_input_pack import _compact_evidence_item
    correction = next(row for row in _evidence_for_cue(cue(), item) if row['source_type'] == 'ocr')
    compact = _compact_evidence_item(1, item, kind='visual_text')
    assert correction['recognition_provenance'] == compact['recognition_provenance'] == record
    assert item['structured_visual'][0]['recognition_provenance']['frame_sha256s'] == record['frame_sha256s']
    before = (root / 'timeline.json').read_bytes()
    assert first['recovered_indexes'] == [1]
    assert vs.reconcile_ebook_pipeline_checkpoints(root)['recovered_indexes'] == []
    assert (root / 'timeline.json').read_bytes() == before


@pytest.mark.parametrize('drift', ['frame', 'artifact', 'text', 'time', 'binding', 'directory_index', 'image_reference'])
def test_stale_checkpoint_cannot_fall_back_to_legacy_markdown(tmp_path, monkeypatch, drift):
    root, frame, checkpoint, results = _produce(tmp_path, monkeypatch)
    payload = json.loads(checkpoint.read_text(encoding='utf-8'))
    if drift == 'frame':
        frame.write_bytes(b'changed')
    elif drift == 'artifact':
        (checkpoint.parent / 'output.md').write_text(TEXT + '\n新增条件', encoding='utf-8')
    elif drift == 'time':
        timeline = json.loads((root / 'timeline.json').read_text())
        timeline[0]['start'] = 10
        (root / 'timeline.json').write_text(json.dumps(timeline))
    elif drift == 'text':
        payload['result']['artifact']['text'] = TEXT + '\n伪造输出'
    elif drift == 'binding':
        payload['result']['recognition_binding']['input']['frame_sha256'] = 'f' * 64
    elif drift == 'image_reference':
        extra = root / 'assets/other.jpg'
        extra.write_bytes(b'not the recognized image')
        payload['result']['image_path'] = str(extra)
    else:
        payload['result']['index'] = 2
    checkpoint.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    before = (root / 'timeline.json').read_bytes()
    result = vs.reconcile_ebook_pipeline_checkpoints(root)
    assert result['status'] == 'degraded'
    assert result['recovered_count'] == 0
    assert (root / 'timeline.json').read_bytes() == before


def test_frame_changed_during_ocr_is_not_usable(tmp_path, monkeypatch):
    root, frame, checkpoint, results = _produce(tmp_path, monkeypatch, drift=True)
    assert results[0]['ok'] is False
    assert 'identity' in results[0]['error']
    assert results[0]['artifact']['text'] == TEXT  # keep raw work, don't rerun it
    assert vs._ebook_results_to_import_rows(results) == []


def test_text_projection_does_not_borrow_stale_or_human_identity(tmp_path, monkeypatch):
    root, frame, checkpoint, results = _produce(tmp_path, monkeypatch)
    vs.reconcile_ebook_pipeline_checkpoints(root)
    item = json.loads((root / 'timeline.json').read_text(encoding='utf-8'))[0]
    item['visual_text'] = '后来替换的文本'
    record = visual_recognition_provenance(item, ['visual_text'], text=item['visual_text'])
    assert record['status'] == 'invalid'
    item['human_corrected_visual_text'] = '人工修订'
    human = visual_recognition_provenance(item, ['human_corrected_visual_text'], text='人工修订')
    assert human['executor'] == 'human_review'
    assert human['frame_sha256s'] == []


def test_legacy_checkpoint_keeps_unknown_input_identity(tmp_path):
    root = tmp_path / 'bundle'
    frame = _write_bundle(root)
    output = root / 'visual-structure/timeline-0001/ebook_pipeline'
    vs._write_ebook_result_checkpoint(output, {'index': 1, 'ok': True, 'image_path': str(frame),
        'artifact': {'text': TEXT, 'path': str(output / 'old.md')}})
    vs.reconcile_ebook_pipeline_checkpoints(root)
    item = json.loads((root / 'timeline.json').read_text(encoding='utf-8'))[0]
    record = visual_recognition_provenance(item, ['visual_text'], text=TEXT)
    assert record['status'] == 'legacy_inputs_unrecorded'
    assert record['frame_sha256s'] == []


def test_new_checkpoint_missing_binding_is_not_a_legacy_checkpoint(tmp_path, monkeypatch):
    root, frame, checkpoint, results = _produce(tmp_path, monkeypatch)
    payload = json.loads(checkpoint.read_text(encoding='utf-8'))
    payload['result'].pop('recognition_binding')
    checkpoint.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    assert vs.reconcile_ebook_pipeline_checkpoints(root)['recovered_count'] == 0


def test_corrupt_checkpoint_cannot_use_same_directory_markdown_fallback(tmp_path, monkeypatch):
    root, frame, checkpoint, results = _produce(tmp_path, monkeypatch)
    checkpoint.write_text('{broken', encoding='utf-8')
    before = (root / 'timeline.json').read_bytes()
    result = vs.reconcile_ebook_pipeline_checkpoints(root)
    assert result['invalid_count'] == 1
    assert result['recovered_count'] == 0
    assert (root / 'timeline.json').read_bytes() == before


def test_text_repair_rejects_bound_artifact_drift(tmp_path, monkeypatch):
    root, frame, checkpoint, results = _produce(tmp_path, monkeypatch)
    vs.reconcile_ebook_pipeline_checkpoints(root)
    before = (root / 'timeline.json').read_bytes()
    (checkpoint.parent / 'output.md').write_text(TEXT + '\n不属于当时识别的文字', encoding='utf-8')
    result = vs.repair_ebook_artifact_text(root)
    assert result['ok'] is False
    assert result['repaired_count'] == 0
    assert (root / 'timeline.json').read_bytes() == before


def test_verified_same_text_adds_missing_provenance_and_repair_preserves_it(tmp_path, monkeypatch):
    root, frame, checkpoint, results = _produce(tmp_path, monkeypatch)
    entries = vs._ebook_results_to_import_rows(results)
    legacy = [{k: v for k, v in entries[0].items() if k != 'recognition_provenance'}]
    timeline = json.loads((root / 'timeline.json').read_text())
    vs._apply_visual_structure_entries(timeline, legacy)
    (root / 'timeline.json').write_text(json.dumps(timeline, ensure_ascii=False), encoding='utf-8')
    vs.reconcile_ebook_pipeline_checkpoints(root)
    item = json.loads((root / 'timeline.json').read_text(encoding='utf-8'))[0]
    assert item['visual_text_recognition_provenance']['frame_sha256s'] == [sha256_file(frame)]
    assert vs.repair_ebook_artifact_text(root)['ok'] is True
    item = json.loads((root / 'timeline.json').read_text(encoding='utf-8'))[0]
    assert item['structured_visual'][0]['recognition_provenance']['frame_sha256s'] == [sha256_file(frame)]


def test_actual_upstream_job_contract_does_not_invent_model_identity():
    import importlib.util
    import os
    from pathlib import Path
    source = os.environ.get('VKP_EBOOK_SOURCE')
    if not source:
        pytest.skip('explicit existing ebook source not provided')
    path = Path(source) / 'artifact_schema.py'
    spec = importlib.util.spec_from_file_location('vkp_test_ebook_artifact_schema', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    job = module.job_payload(job_id='synthetic', kind='image', status='done', started_at='fixture',
                             input_path='synthetic.png', output_path='output', total=1)
    assert job['schema_version'] == 'ebook-job-v1'
    assert 'model_actual' not in job
    assert module.material_consumer_contract()['network_transfer_allowed'] is False
