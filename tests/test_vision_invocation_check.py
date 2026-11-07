import json
from pathlib import Path

import pytest

import video_knowledge_pipeline.vision_execution_route as routing
import video_knowledge_pipeline.multimodal_frame_analyzer as semantic
import video_knowledge_pipeline.temporal_visual_analyzer as temporal


@pytest.fixture
def implicit_local(monkeypatch):
    monkeypatch.setattr(routing, 'probe_lm_studio_endpoint', lambda: {'listening': True, 'vision_model_verified': False})
    monkeypatch.setattr(routing, 'resolve_route_based_vision_gateway_profile', lambda task: {
        'status': 'gateway_unavailable', 'route_configured': True,
        'provider_config': {'provider': 'openai_compatible', 'model': 'remote-fixture', 'base_url': 'https://example.invalid/v1'}})


def test_local_presence_requires_explicit_choice_before_execute(implicit_local, capsys):
    route = routing.resolve_vision_task_execution_route('semantic_frame')
    with pytest.raises(ValueError, match='vision_route_choice_required'):
        routing.check_vision_invocation(execute=True, route=route)
    assert '1234' in capsys.readouterr().err


def test_preview_is_visible_and_does_not_select_local_automatically(implicit_local, capsys):
    route = routing.resolve_vision_task_execution_route('semantic_frame')
    result = routing.check_vision_invocation(execute=False, route=route)
    assert result['mode'] == 'preview_only'
    assert result['route_choice_required'] is True
    assert result['automatic_route_switch'] is False
    assert route['provider_config']['model'] == 'remote-fixture'
    assert 'PREVIEW ONLY' in capsys.readouterr().err


@pytest.mark.parametrize('provider', ['local_vlm', 'openai_compatible'])
def test_explicit_choice_does_not_probe_or_switch(monkeypatch, provider):
    def forbidden(*args, **kwargs): raise AssertionError('Unexpected implicit route/probe')
    monkeypatch.setattr(routing, 'probe_lm_studio_endpoint', forbidden)
    monkeypatch.setattr(routing, 'resolve_route_based_vision_gateway_profile', forbidden)
    config={'provider':provider,'base_url':'http://127.0.0.1:1234/v1','model':'actual-fixture','api_key':''}
    route=routing.resolve_vision_task_execution_route('semantic_frame',provider_config=config)
    assert routing.check_vision_invocation(execute=True,route=route)['route_choice_required'] is False
    assert route['provider_config'] == config


@pytest.mark.parametrize('config', [{}, {'provider':'local_vlm'}, {'provider':'local_vlm','model':'named'}])
def test_incomplete_explicit_config_rejected(config):
    route=routing.resolve_vision_task_execution_route('semantic_frame',provider_config=config)
    with pytest.raises(ValueError,match='config_incomplete'):
        routing.check_vision_invocation(execute=True,route=route)


def test_closed_port_does_not_block_configured_route(monkeypatch, implicit_local, capsys):
    monkeypatch.setattr(routing,'probe_lm_studio_endpoint',lambda:{'listening':False})
    route=routing.resolve_vision_task_execution_route('semantic_frame')
    assert routing.check_vision_invocation(execute=True,route=route)['route_choice_required'] is False
    assert 'No explicit --provider-config' in capsys.readouterr().err


@pytest.mark.parametrize('module,func', [(semantic,'run_multimodal_frame_analysis'),(temporal,'run_temporal_visual_analysis')])
def test_both_entrypoints_block_before_writes_or_calls(tmp_path,implicit_local,module,func,monkeypatch):
    (tmp_path/'manifest.json').write_text('{}',encoding='utf-8')
    def forbidden(*args,**kwargs): raise AssertionError('Vision call must not happen')
    monkeypatch.setattr(module,'call_vision_model',forbidden)
    with pytest.raises(ValueError,match='vision_route_choice_required'):
        getattr(module,func)(tmp_path,execute=True)
    assert sorted(p.name for p in tmp_path.iterdir())==['manifest.json']


@pytest.mark.parametrize('module,func', [(semantic,'run_multimodal_frame_analysis'),(temporal,'run_temporal_visual_analysis')])
def test_preview_report_and_registry_say_needs_execution(tmp_path,module,func,monkeypatch):
    (tmp_path/'manifest.json').write_text('{}',encoding='utf-8')
    (tmp_path/'timeline.json').write_text('[]',encoding='utf-8')
    monkeypatch.setattr(module,'_candidates',lambda *a,**k: [])
    result=getattr(module,func)(tmp_path,execute=False,provider_config={'provider':'fixture','model':'fixture'})
    assert result['summary']['status']=='preview_only'
    assert result['summary']['invocation_check']['mode']=='preview_only'
    report=Path(result['report_path']).read_text(encoding='utf-8')
    assert 'PREVIEW ONLY' in report and 'preview_only' in report
    assert semantic._vision_run_registry_status({'status':'preview_only','execute':False},[],1)=='needs_execution'
    assert semantic._vision_run_failed_items({'status':'preview_only'},[],execute=False)==[]


def test_probe_is_loopback_only_and_bounded(monkeypatch):
    seen=[]
    def unavailable(address,timeout):
        seen.append((address,timeout));raise OSError('closed')
    monkeypatch.setattr(routing.socket,'create_connection',unavailable)
    result=routing.probe_lm_studio_endpoint()
    assert seen==[(('127.0.0.1',1234),0.25)]
    assert result['listening'] is False and result['vision_model_verified'] is False


@pytest.mark.parametrize('provider',['local_vlm','local_qwen_vl'])
def test_empty_local_key_does_not_inherit_remote_credentials(monkeypatch,provider):
    from video_knowledge_pipeline.vision_api import resolve_provider_config
    monkeypatch.setenv('LECTURE_VISION_API_KEY','synthetic-cloud-key')
    monkeypatch.setenv('LOCAL_VLM_API_KEY','synthetic-local-key')
    cfg=resolve_provider_config({'provider':provider,'api_key':''})
    assert not cfg['api_key']
    cfg=resolve_provider_config({'provider':provider,'api_key':'explicit-local'})
    assert cfg['api_key']=='explicit-local'


def test_local_route_cannot_point_at_remote_host():
    route=routing.resolve_vision_task_execution_route('semantic_frame',provider_config={
        'provider':'local_vlm','model':'fixture','base_url':'https://example.invalid/v1'})
    with pytest.raises(ValueError,match='loopback'):
        routing.check_vision_invocation(execute=True,route=route)
