"""Synthetic request/response identity tests; never call a real Provider."""
import json

import pytest

from video_knowledge_pipeline.storage import write_json


def capture(root):
    from video_knowledge_pipeline.summary_model_input import capture_request, capture_response, bind_candidate
    pack = {'sections': []}
    plan = {'schema': 'fixture', 'source_section_ids': ['s']}
    request = capture_request(root, messages=[{'role': 'system', 'content': 'contract'},
        {'role': 'user', 'content': '输入 JSON：\n{"chapters":[]}'}], fact_pack=pack,
        max_tokens=500, temperature=0, write=True)
    response = capture_response(root, request, raw_response='{"answer":"original"}',
                                gateway_ok=True, executor='model_task_gateway', write=True)
    binding = bind_candidate(root, response, plan=plan, fact_pack=pack, write=True)
    write_json(root / 'manifest.json', {'smart_summary_generation_binding': binding})
    return pack, plan, request, response, binding


def test_input_capture_is_immutable_and_original_prompt_survives_previews(tmp_path):
    from video_knowledge_pipeline.summary_model_input import generation_input_status
    pack, plan, request, response, binding = capture(tmp_path)
    (tmp_path / 'exports/smart-summary-global-reduce-prompt.md').write_text('new preview', encoding='utf-8')
    report = generation_input_status(tmp_path, fact_pack=pack, plan=plan)
    assert report['status'] == 'bound_gateway_input'
    assert report['prompt'].startswith('输入 JSON：')
    assert report['model_observation'] == 'not_assessed'
    assert report['request'] == request and report['response'] == response
    assert capture(tmp_path)[-1] == binding


@pytest.mark.parametrize('target', ['request', 'response', 'binding', 'plan', 'fact_pack'])
def test_every_binding_drift_is_fail_closed(tmp_path, target):
    from video_knowledge_pipeline.summary_model_input import generation_input_status
    pack, plan, request, response, binding = capture(tmp_path)
    if target == 'plan':
        plan['changed'] = True
    elif target == 'fact_pack':
        pack['changed'] = True
    else:
        ref = {'request': request, 'response': response, 'binding': binding}[target]
        write_json(tmp_path / ref['path'], {'changed': True})
    assert generation_input_status(tmp_path, fact_pack=pack, plan=plan)['status'] == 'blocked'


def test_old_candidate_without_capture_stays_unknown(tmp_path):
    from video_knowledge_pipeline.summary_model_input import generation_input_status
    write_json(tmp_path / 'manifest.json', {})
    result = generation_input_status(tmp_path, fact_pack={}, plan={})
    assert result['status'] == 'not_assessed'
    assert result['passed'] is None


def test_recovery_reads_original_response_without_gateway(tmp_path):
    from video_knowledge_pipeline.summary_model_input import recover_response
    pack, plan, request, response, binding = capture(tmp_path)
    recovered = recover_response(tmp_path, tmp_path / response['path'])
    assert recovered['content'] == '{"answer":"original"}'
    assert recovered['input_capture'] == response
    assert recovered['network_requests_made'] is False


def test_no_write_capture_does_not_claim_persistence(tmp_path):
    from video_knowledge_pipeline.summary_model_input import capture_request
    ref = capture_request(tmp_path, messages=[{'role': 'user', 'content': 'fixture'}], fact_pack={},
                          max_tokens=1, temperature=0, write=False)
    assert ref is None
    assert list(tmp_path.iterdir()) == []


def live_fixture(tmp_path, monkeypatch):
    from test_smart_summary_global_reduce_business_authorization import _bundle, _patch_reduce_quality
    from video_knowledge_pipeline import smart_summary_global_reduce as module
    root, _ = _bundle(tmp_path)
    _patch_reduce_quality(monkeypatch)
    monkeypatch.setattr(module, 'resolve_model_api_provider_config', lambda *a, **k: {'provider': 'local', 'model': 'synthetic', 'base_url': 'http://127.0.0.1:1/v1'})
    monkeypatch.setattr(module, 'resolve_text_provider_config', lambda cfg: cfg)
    calls = []
    def gateway(*a, **kwargs):
        calls.append(kwargs['messages'])
        return {'ok': True, 'content': '{}'}
    monkeypatch.setattr(module, 'model_task_api_call', gateway)
    return root, module, calls


def test_live_reduce_reuse_keeps_original_input_not_new_preview(tmp_path, monkeypatch):
    from video_knowledge_pipeline.summary_model_input import generation_input_status
    root, module, calls = live_fixture(tmp_path, monkeypatch)
    first = module.run_smart_summary_global_reduce(root, execute=True, install=False)
    assert first['generation_input']['status'] == 'bound_gateway_input'
    original = calls[0][-1]['content']
    render = module._render_reduce_prompt
    monkeypatch.setattr(module, '_render_reduce_prompt', lambda *a, **k: 'new instructions\n' + render(*a, **k))
    module.run_smart_summary_global_reduce(root, execute=False, install=False)
    reused = module.run_smart_summary_global_reduce(root, reuse_candidate=True, install=False)
    assert len(calls) == 1
    assert reused['generation_input']['status'] == 'bound_gateway_input'
    assert reused['generation_stage_coverage']['input_basis'].startswith('captured_gateway_input')
    plan = json.loads((root / 'exports/smart-summary-reader-plan.json').read_text(encoding='utf-8'))
    pack = json.loads((root / 'exports/smart-summary-chapter-fact-pack.json').read_text(encoding='utf-8'))
    assert generation_input_status(root, fact_pack=pack, plan=plan)['prompt'] == original


def test_crash_after_response_before_parse_recovers_without_second_call(tmp_path, monkeypatch):
    from video_knowledge_pipeline.summary_model_input import RESPONSE_SCHEMA
    root, module, calls = live_fixture(tmp_path, monkeypatch)
    original = module._validated_reader_plan_content
    def crash(*a, **k):
        raise RuntimeError('synthetic crash after gateway')
    monkeypatch.setattr(module, '_validated_reader_plan_content', crash)
    with pytest.raises(RuntimeError, match='synthetic crash'):
        module.run_smart_summary_global_reduce(root, execute=True, install=False)
    responses = [path for path in (root / 'exports/summary-input-records').glob('*.json')
                 if json.loads(path.read_text(encoding='utf-8')).get('schema') == RESPONSE_SCHEMA]
    assert len(responses) == 1
    monkeypatch.setattr(module, '_validated_reader_plan_content', original)
    recovered = module.run_smart_summary_global_reduce(root, recovery_execution_report=responses[0], install=False)
    assert len(calls) == 1
    assert recovered['generation_input']['status'] == 'bound_gateway_input'
    assert recovered['recovery']['network_requests_made'] is False


def test_actual_input_missing_point_cannot_be_laundered_by_rebuilt_prompt(tmp_path):
    from test_summary_stage_coverage import fixture, prompt, save_projection, TEXT, OTHER
    from video_knowledge_pipeline.summary_model_input import capture_request, capture_response, bind_candidate
    from video_knowledge_pipeline.summary_stage_coverage import stored_summary_stage_coverage
    pack = fixture(tmp_path)
    plan = {'schema': 'fixture'}
    request = capture_request(tmp_path, messages=[{'role': 'user', 'content': prompt(OTHER)}], fact_pack=pack,
                              max_tokens=5, temperature=0, write=True)
    response = capture_response(tmp_path, request, raw_response='{}', gateway_ok=True,
                                executor='model_task_gateway', write=True)
    binding = bind_candidate(tmp_path, response, plan=plan, fact_pack=pack, write=True)
    write_json(tmp_path / 'manifest.json', {'smart_summary_generation_binding': binding})
    write_json(tmp_path / 'exports/smart-summary-reader-plan.json', plan)
    save_projection(tmp_path, pack, prompt(TEXT))  # new prompt includes the missing point
    result = stored_summary_stage_coverage(tmp_path, output_text=TEXT)
    assert result['input_basis'].startswith('captured_gateway_input')
    assert result['points'][0]['first_loss'] == 'model_input'
    assert result['passed'] is False


def test_capture_failure_prevents_gateway_call(tmp_path, monkeypatch):
    root, module, calls = live_fixture(tmp_path, monkeypatch)
    def broken(*a, **k):
        raise ValueError('synthetic capture drift')
    monkeypatch.setattr(module, 'capture_request', broken)
    result = module.run_smart_summary_global_reduce(root, execute=True, install=False)
    assert result['status'] == 'blocked_request_capture'
    assert result['provider_call_performed'] is False and calls == []


def test_response_capture_failure_keeps_raw_response_and_does_not_install(tmp_path, monkeypatch):
    root, module, calls = live_fixture(tmp_path, monkeypatch)
    def broken(*a, **k):
        raise ValueError('synthetic post-call drift')
    monkeypatch.setattr(module, 'capture_response', broken)
    result = module.run_smart_summary_global_reduce(root, execute=True, install=False)
    assert result['status'] == 'blocked_input_capture' and len(calls) == 1
    assert (root / 'exports/smart-summary-global-reduce-raw-response.txt').read_text(encoding='utf-8') == '{}'
    assert not (root / 'exports/smart-summary-reader-plan.json').exists()


def test_delegate_request_drift_and_outside_bundle_refused(tmp_path):
    from video_knowledge_pipeline.summary_model_input import capture_request, capture_response
    root = tmp_path / 'bundle'
    root.mkdir()
    messages = [{'role': 'user', 'content': 'fixture'}]
    path = root / 'request.json'
    value = {'task': 'smart_summary_global_reduce', 'messages': messages,
             'generation_parameters': {'max_tokens': 5, 'temperature': 0, 'response_format': 'json_object'}}
    write_json(path, value)
    ref = capture_request(root, messages=messages, fact_pack={}, max_tokens=5, temperature=0,
                          delegate_request=path, write=True)
    write_json(path, {**value, 'changed': True})
    with pytest.raises(ValueError, match='changed during execution'):
        capture_response(root, ref, raw_response='{}', gateway_ok=True, executor='consented_model_connector', write=True)
    outside = tmp_path / 'outside.json'
    write_json(outside, value)
    with pytest.raises(ValueError):
        capture_request(root, messages=messages, fact_pack={}, max_tokens=5, temperature=0,
                        delegate_request=outside, write=True)


def test_valid_capture_links_are_projected_without_prompt_content(tmp_path):
    from test_summary_stage_coverage import save_projection
    from video_knowledge_pipeline.summary_stage_coverage import stored_summary_stage_coverage
    pack, plan, request, response, binding = capture(tmp_path)
    write_json(tmp_path / 'exports/smart-summary-reader-plan.json', plan)
    save_projection(tmp_path, pack, '输入 JSON：\n{"chapters":[]}')
    status = stored_summary_stage_coverage(tmp_path, output_text='')
    assert status['artifacts']['generation_request'] == request['path']
    assert status['artifacts']['generation_response'] == response['path']
    assert 'prompt' not in status['generation_input']
