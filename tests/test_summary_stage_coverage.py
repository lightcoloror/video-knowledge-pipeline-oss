"""Offline stage-loss regression, reusing existing human-reference semantics."""
from __future__ import annotations

import json

import pytest

from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot
from video_knowledge_pipeline.file_hash import sha256_file


TEXT = '甲产品2026版只有完成审核才能承保。'
OTHER = '乙产品2025版线上申请。'


def fixture(root):
    write_json(root / 'timeline.json', [{'text': OTHER}, {'text': TEXT}])
    snap = JsonEvidenceSnapshot.read(root, 'timeline.json')
    refs = [{'evidence_id': f'r{i}', 'source_kind': 'asr', 'fact_status': 'candidate_evidence',
             'snippet': text, 'source_locators': [snap.locator([i, 'text'])]} for i, text in enumerate((OTHER, TEXT))]
    pack = {'sections': [{'section_id': 's', 'evidence_refs': refs}]}
    write_json(root / 'quality-benchmark-key-points.json', {'key_points': [{'id': 'important', 'text': TEXT,
        'claim_scope': {'subject': '甲产品', 'version': '2026版'}, 'evidence_ids': ['r1']}],
        'source_bindings': [{'path': 'timeline.json', 'sha256': sha256_file(root / 'timeline.json')}]})
    return pack


def prompt(text):
    return '请总结。\n\n输入 JSON：\n' + json.dumps({'chapters': [{'markdown': text}]}, ensure_ascii=False)


def test_stage_matrix_distinguishes_input_loss_and_output_omission(tmp_path):
    from video_knowledge_pipeline.summary_stage_coverage import evaluate_summary_stage_coverage
    pack = fixture(tmp_path)
    lost = evaluate_summary_stage_coverage(tmp_path, pack, prompt(OTHER), output_text=TEXT)
    row = lost['points'][0]
    assert row['stages'] == {'source_evidence': True, 'material': True, 'model_input': False, 'output': True}
    assert row['first_loss'] == 'model_input'
    assert lost['input_gate_passed'] is False  # output cannot launder input loss
    omitted = evaluate_summary_stage_coverage(tmp_path, pack, prompt(TEXT), output_text=OTHER)
    assert omitted['points'][0]['first_loss'] == 'output'
    assert omitted['input_gate_passed'] is True
    assert omitted['output_recall'] == 0
    assert omitted['material_utilization'] == 0


def test_instruction_echo_and_metadata_do_not_count_as_evidence_input(tmp_path):
    from video_knowledge_pipeline.summary_stage_coverage import evaluate_summary_stage_coverage
    pack = fixture(tmp_path)
    value = TEXT + '\n\n输入 JSON：\n' + json.dumps({'title': TEXT, 'chapters': [{'markdown': OTHER}]}, ensure_ascii=False)
    report = evaluate_summary_stage_coverage(tmp_path, pack, value)
    assert report['points'][0]['stages']['model_input'] is False
    assert report['points'][0]['stages']['output'] is None


def test_required_prompt_material_is_existing_source_not_reference_answer(tmp_path):
    from video_knowledge_pipeline.summary_stage_coverage import required_prompt_evidence
    pack = fixture(tmp_path)
    selected = required_prompt_evidence(tmp_path, pack)
    assert selected == [{'evidence_id': 'r1', 'snippet': TEXT, 'fact_status': 'candidate_evidence', 'source_kind': 'asr'}]
    pack['sections'][0]['evidence_refs'][1]['fact_status'] = 'review_gap_not_fact'
    assert required_prompt_evidence(tmp_path, pack) == []


@pytest.mark.parametrize('kind', ['missing_reference', 'stale_reference', 'bad_locator'])
def test_unknown_or_stale_cannot_become_complete(tmp_path, kind):
    from video_knowledge_pipeline.summary_stage_coverage import evaluate_summary_stage_coverage
    pack = fixture(tmp_path)
    if kind == 'missing_reference':
        (tmp_path / 'quality-benchmark-key-points.json').unlink()
    elif kind == 'stale_reference':
        write_json(tmp_path / 'timeline.json', [])
    else:
        pack['sections'][0]['evidence_refs'][1]['source_locators'][0]['artifact_sha256'] = 'a' * 64
    report = evaluate_summary_stage_coverage(tmp_path, pack, prompt(TEXT), output_text=TEXT)
    assert report['passed'] is not True
    assert report['status'] in {'not_assessed', 'blocked'}


def test_reduce_prompt_preserves_required_source_beyond_default_quote_sample(tmp_path):
    from video_knowledge_pipeline.smart_summary_global_reduce import _reduce_prompt_plan
    pack = fixture(tmp_path)
    # Important source is outside the old first-two-snippets selection.
    refs = pack['sections'][0]['evidence_refs']
    refs.insert(1, {**refs[0], 'evidence_id': 'unrelated'})
    rows = [{'section_id': 's', 'time_range': '00:00:00 - 00:10:00', 'final_markdown': OTHER}]
    result = _reduce_prompt_plan(tmp_path, rows, {}, fact_pack=pack, max_input_chars=20000)
    body = json.loads(result['prompt'].split('输入 JSON：\n', 1)[1])
    assert body['required_key_point_evidence'][0]['snippet'] == TEXT
    assert result['stage_coverage']['input_gate_passed'] is True
    assert result['stage_coverage']['input_basis'] == 'prepared_payload_not_proof_of_model_observation'


def save_projection(root, pack, value):
    from video_knowledge_pipeline.summary_stage_coverage import evaluate_summary_stage_coverage
    report = evaluate_summary_stage_coverage(root, pack, value)
    write_json(root / 'exports/smart-summary-chapter-fact-pack.json', pack)
    write_json(root / 'exports/smart-summary-global-reduce.json', {'stage_coverage': report})
    (root / 'exports/smart-summary-global-reduce-prompt.md').write_text(value, encoding='utf-8')


def test_saved_projection_rechecks_actual_output_and_preserves_unknown_observation(tmp_path):
    from video_knowledge_pipeline.summary_stage_coverage import stored_summary_stage_coverage
    pack = fixture(tmp_path)
    save_projection(tmp_path, pack, prompt(TEXT))
    report = stored_summary_stage_coverage(tmp_path, output_text=TEXT)
    assert report['passed'] is True
    assert report['generation_input_provenance'] == 'not_assessed'
    omitted = stored_summary_stage_coverage(tmp_path, output_text=OTHER)
    assert omitted['passed'] is False
    assert omitted['points'][0]['first_loss'] == 'output'


@pytest.mark.parametrize('drift', ['prompt', 'pack', 'hash_removed', 'source'])
def test_saved_projection_drift_is_fail_closed(tmp_path, drift):
    from video_knowledge_pipeline.summary_stage_coverage import stored_summary_stage_coverage
    pack = fixture(tmp_path)
    save_projection(tmp_path, pack, prompt(TEXT))
    if drift == 'prompt':
        (tmp_path / 'exports/smart-summary-global-reduce-prompt.md').write_text(prompt(OTHER), encoding='utf-8')
    elif drift == 'pack':
        write_json(tmp_path / 'exports/smart-summary-chapter-fact-pack.json', {})
    elif drift == 'hash_removed':
        path = tmp_path / 'exports/smart-summary-global-reduce.json'
        saved = json.loads(path.read_text(encoding='utf-8'))
        del saved['stage_coverage']['prompt_utf8_sha256']
        write_json(path, saved)
    else:
        write_json(tmp_path / 'timeline.json', [])
    report = stored_summary_stage_coverage(tmp_path, output_text=TEXT)
    assert report['status'] == 'blocked'
    assert report['passed'] is False


def test_quality_does_not_count_review_only_output_as_retained(tmp_path):
    from video_knowledge_pipeline.smart_summary_codex import smart_summary_quality_check
    pack = fixture(tmp_path)
    write_json(tmp_path / 'manifest.json', {})
    save_projection(tmp_path, pack, prompt(TEXT))
    path = tmp_path / 'exports/smart-summary.md'
    path.write_text('## 核心主题\n' + OTHER + '\n\n## 待复核点\n' + TEXT, encoding='utf-8')
    result = smart_summary_quality_check(tmp_path, write=False)
    stage = result['summary_stage_coverage']
    assert stage['points'][0]['first_loss'] == 'output'
    gate = next(row for row in result['checks'] if row['key'] == 'declared_important_point_stage_retention')
    assert gate['evaluated'] and gate['passed'] is False


def test_composite_index_does_not_duplicate_source_or_supply_group_text(tmp_path):
    from video_knowledge_pipeline.summary_stage_coverage import evaluate_summary_stage_coverage
    pack = fixture(tmp_path)
    refs = pack['sections'][0]['evidence_refs']
    refs.append({'evidence_id': 'g', 'source_kind': 'evidence_group', 'member_evidence_ids': ['r0', 'r1'],
                 'fact_status': 'candidate_evidence', 'snippet': ''})
    report = evaluate_summary_stage_coverage(tmp_path, pack, prompt(TEXT), output_text=TEXT)
    assert report['passed'] is True
    refs[-1]['snippet'] = TEXT
    assert evaluate_summary_stage_coverage(tmp_path, pack, prompt(TEXT), output_text=TEXT)['status'] == 'blocked'


@pytest.mark.parametrize('bad', [{'chapters': [1]}, {'chapters': 'invalid'},
                                {'chapters': [{'facts': 'invalid'}]}, {'required_key_point_evidence': [1]}])
def test_malformed_input_is_receipted_not_uncaught(tmp_path, bad):
    from video_knowledge_pipeline.summary_stage_coverage import evaluate_summary_stage_coverage
    pack = fixture(tmp_path)
    report = evaluate_summary_stage_coverage(tmp_path, pack, '输入 JSON：\n' + json.dumps(bad))
    assert report['status'] == 'blocked'


def reduce_fixture(root, monkeypatch):
    from video_knowledge_pipeline import smart_summary_global_reduce as module
    pack = fixture(root)
    write_json(root / 'manifest.json', {})
    write_json(root / 'exports/smart-summary-section-workflow.json', {'sections': [{'section_id': 's'}]})
    write_json(root / 'exports/smart-summary-section-llm-revisions.json', {'rows': [
        {'section_id': 's', 'time_range': '00:00:00 - 00:10:00', 'final_markdown': OTHER}]})
    monkeypatch.setattr(module, '_chapter_fact_pack', lambda *a, **k: pack)
    monkeypatch.setattr(module, 'evaluate_production_artifact_gate', lambda *a, **k: {'formal_generation_allowed': True})
    return module, pack


def test_live_reduce_stale_important_point_binding_never_calls_provider(tmp_path, monkeypatch):
    module, _ = reduce_fixture(tmp_path, monkeypatch)
    write_json(tmp_path / 'timeline.json', [{'text': OTHER}, {'text': TEXT}, {'text': 'changed'}])
    # Update source locators only; the human review binding deliberately remains old.
    snap = JsonEvidenceSnapshot.read(tmp_path, 'timeline.json')
    pack = module._chapter_fact_pack()
    for i, ref in enumerate(pack['sections'][0]['evidence_refs']):
        ref['source_locators'] = [snap.locator([i, 'text'])]
    monkeypatch.setattr(module, 'model_task_api_call', lambda *a, **k: pytest.fail('must not call Provider'))
    result = module.run_smart_summary_global_reduce(tmp_path, execute=True, write=False)
    assert result['status'] == 'blocked_important_point_input'
    assert result['provider_call_performed'] is False


def test_live_reduce_omitted_important_point_cannot_install(tmp_path, monkeypatch):
    module, _ = reduce_fixture(tmp_path, monkeypatch)
    captured = []
    def call(*a, **kwargs):
        captured.append(kwargs['messages'][-1]['content'])
        return {'ok': True, 'content': '{}'}
    monkeypatch.setattr(module, 'model_task_api_call', call)
    monkeypatch.setattr(module, '_validated_reader_plan_content', lambda *a, **k: ({'source_section_ids': ['s']}, {'passed': True}, '## 核心主题\n' + OTHER))
    monkeypatch.setattr(module, '_shape_quality', lambda *a, **k: {'passed': True, 'checks': []})
    monkeypatch.setattr(module, 'generate_smart_summary_with_codex', lambda *a, **k: pytest.fail('must not install'))
    result = module.run_smart_summary_global_reduce(tmp_path, execute=True, write=False,
                provider_config={'provider': 'local', 'base_url': 'http://127.0.0.1:1/v1', 'model': 'synthetic'})
    assert len(captured) == 1
    assert TEXT in captured[0]
    assert result['status'] == 'reduce_quality_failed'
    assert result['stage_coverage']['points'][0]['first_loss'] == 'output'


def test_coverage_and_existing_workbenches_share_read_only_stage_projection(tmp_path):
    from video_knowledge_pipeline.knowledge_coverage import audit_knowledge_coverage
    from video_knowledge_pipeline.task_console import export_task_console, _render_task_console_html
    from video_knowledge_pipeline.video_workbench import export_video_workbench, _render_workbench_html
    pack = fixture(tmp_path)
    write_json(tmp_path / 'manifest.json', {})
    save_projection(tmp_path, pack, prompt(TEXT))
    (tmp_path / 'exports/smart-summary.md').write_text('## 核心主题\n' + OTHER, encoding='utf-8')
    before = {path.relative_to(tmp_path).as_posix(): sha256_file(path) for path in tmp_path.rglob('*') if path.is_file()}
    coverage = audit_knowledge_coverage(tmp_path, write=False)['coverage']['summary_stage_coverage']
    for export, render in ((export_task_console, _render_task_console_html),
                           (export_video_workbench, lambda value: _render_workbench_html(value, []))):
        result = export(tmp_path, write=False)
        stage = result['summary_stage_coverage']
        assert stage == coverage
        assert stage['points'][0]['first_loss'] == 'output'
        html = render(result)
        assert 'id="summary-stage-coverage"' in html
        assert '准备输入不代表模型已读' in html
    assert before == {path.relative_to(tmp_path).as_posix(): sha256_file(path) for path in tmp_path.rglob('*') if path.is_file()}


def test_actual_clipping_is_reported_even_with_all_section_ids(tmp_path):
    from video_knowledge_pipeline.smart_summary_global_reduce import _reduce_prompt_plan
    result = _reduce_prompt_plan(tmp_path, [{'section_id': 's', 'final_markdown': OTHER * 250}], {},
                                 fact_pack={'sections': []}, max_input_chars=60000)
    assert result['all_sections_included'] is True
    assert result['source_markdown_clipped_section_ids'] == ['s']


def test_stage_panel_escapes_external_metadata():
    from video_knowledge_pipeline.task_console import _summary_stage_panel_html
    value = '<script>alert(1)</script>'
    result = _summary_stage_panel_html({'required': True, 'status': value,
        'points': [{'id': value, 'first_loss': value}], 'errors': [value]})
    assert '<script>' not in result
    assert '&lt;script&gt;' in result


def test_declared_evidence_id_cannot_borrow_matching_text_from_another_source(tmp_path):
    from video_knowledge_pipeline.summary_stage_coverage import evaluate_summary_stage_coverage, required_prompt_evidence
    pack = fixture(tmp_path)
    # Text still occurs globally, but not in the bound human reference source.
    path = tmp_path / 'quality-benchmark-key-points.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['key_points'][0]['evidence_ids'] = ['r0']
    write_json(path, payload)
    report = evaluate_summary_stage_coverage(tmp_path, pack, prompt(TEXT), output_text=TEXT)
    assert report['passed'] is False
    assert report['points'][0]['first_loss'] == 'source_evidence'
    assert report['stage_recall']['material'] == 0
    assert required_prompt_evidence(tmp_path, pack) == []


def test_duplicate_text_prefers_the_declared_source_not_first_global_match(tmp_path):
    from video_knowledge_pipeline.summary_stage_coverage import required_prompt_evidence
    pack = fixture(tmp_path)
    pack['sections'][0]['evidence_refs'][0]['snippet'] = TEXT
    assert required_prompt_evidence(tmp_path, pack)[0]['evidence_id'] == 'r1'


@pytest.mark.parametrize('missing', ['receipt', 'contract'])
def test_registered_stage_contract_cannot_silently_revert_to_legacy(tmp_path, missing):
    from video_knowledge_pipeline.summary_stage_coverage import SCHEMA, stored_summary_stage_coverage
    write_json(tmp_path / 'manifest.json', {'smart_summary_stage_coverage_schema': SCHEMA})
    if missing == 'contract':
        write_json(tmp_path / 'exports/smart-summary-global-reduce.json', {})
    report = stored_summary_stage_coverage(tmp_path, output_text=TEXT)
    assert report['required'] and report['status'] == 'blocked'
