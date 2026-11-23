import json
from pathlib import Path
from unittest.mock import patch
import pytest
from test_targeted_review import request_pack,fake_probe
from video_knowledge_pipeline.review_export_context import main,normalize_project_identity,resolve_review_export_context,configured_identity,identity_for_preparation,CONTEXT_FILENAME
from video_knowledge_pipeline.targeted_review import prepare_targeted_review_pack,load_targeted_review_bundle

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
def identity(topic='水质采样',video='实验记录',ident='water-2026'):
    return {'topic_title':topic,'video_title':video,'video_id':ident,'time_zone':'Asia/Shanghai'}
def prepare(source,out):
    with patch('video_knowledge_pipeline.targeted_review.probe_video',side_effect=fake_probe):return prepare_targeted_review_pack(source,out)

def test_new_exact_project_generates_bound_context_from_request_identity(tmp_path):
    source=request_pack(tmp_path);raw=json.loads(source.read_text(encoding='utf-8'));raw['project_identity']=identity();write(source,raw)
    out=tmp_path/'review';receipt=prepare(source,out);_,pack=load_targeted_review_bundle(out)
    context=json.loads((out/CONTEXT_FILENAME).read_text(encoding='utf-8'))
    assert receipt['export_naming_status']=='configured'
    assert pack['project_identity']==identity() and context['video_id']=='water-2026'
    assert context['pack_payload_sha256']==pack['pack_payload_sha256']
    assert context['sources'][0]['file']=='candidate.mp4'
    assert not list(out.glob('*.mp4'))
    assert '水质采样' in (out/'targeted-review.html').read_text(encoding='utf-8')

def test_second_project_configures_once_and_preparation_consumes_it(tmp_path,capsys):
    source=request_pack(tmp_path)
    assert main(['configure',str(tmp_path),'--topic-title','城市散步','--video-title','秋日记录','--video-id','walk-2026'])==0
    out=tmp_path/'review';prepare(source,out);ctx=json.loads((out/CONTEXT_FILENAME).read_text(encoding='utf-8'))
    assert ctx['topic_title']=='城市散步' and ctx['video_id']=='walk-2026'
    assert 'RWE' not in json.dumps(ctx,ensure_ascii=False)
    assert main(['status',str(tmp_path)])==0
    assert 'configured' in capsys.readouterr().out

def test_missing_identity_keeps_review_available_but_no_guessed_export(tmp_path):
    source=request_pack(tmp_path);out=tmp_path/'review';receipt=prepare(source,out)
    assert receipt['export_naming_status']=='missing_project_identity'
    assert not (out/CONTEXT_FILENAME).exists()
    html=(out/'targeted-review.html').read_text(encoding='utf-8')
    assert '导出前请在项目配置中补充选题' in html
    assert 'videoCount > 1' not in html and '多视频-${' not in html

def test_manifest_identity_is_available_to_all_generic_producers(tmp_path):
    write(tmp_path/'manifest.json',{'title':'camera title is not project identity','project_identity':identity()})
    for schema in ['lecture_knowledge_package.v1','video_knowledge_pipeline.multimodal_sample_review.v1','video_knowledge_pipeline.smart_summary_section_editor.v1','video_knowledge_pipeline.task_console.v1']:
        ctx=resolve_review_export_context({'schema':schema,'bundle_dir':str(tmp_path),'media':{'path':'CAM_001.mp4'}})
        assert ctx['topic_title']=='水质采样' and ctx['video_id']=='water-2026'

def test_explicit_incomplete_identity_is_rejected_at_configuration():
    with pytest.raises(ValueError,match='缺少'):normalize_project_identity({'topic_title':'only-topic'})

def test_legacy_bound_context_stays_supported_and_wrong_binding_is_rejected(tmp_path):
    write(tmp_path/'targeted-transcript-export-context.json',{**identity(),'pack_payload_sha256':'old-pack'})
    assert resolve_review_export_context({'pack_payload_sha256':'old-pack'},root=tmp_path)['video_id']=='water-2026'
    with pytest.raises(ValueError,match='review_export_context_pack_changed'):resolve_review_export_context({'pack_payload_sha256':'other-pack'},root=tmp_path)

def test_webui_export_and_refresh_preserve_configured_project_identity(tmp_path):
    from video_knowledge_pipeline.storage import ensure_project_dirs
    from video_knowledge_pipeline.webui_bridge import export_webui_bundle, refresh_bundle_review_html

    paths = ensure_project_dirs(tmp_path/'data')
    paths['lecture_packages'].mkdir(parents=True, exist_ok=True)
    write(paths['root']/CONTEXT_FILENAME, identity())
    write(paths['lecture_packages']/'lecture-package.json', {
        'title':'CAM_001', 'sources':[], 'timeline':[], 'coverage':{},
    })
    bundle = tmp_path/'webui-bundle'
    export_webui_bundle(paths['root'], bundle)
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['project_identity'] == identity()
    assert '水质采样' in (bundle/'review.html').read_text(encoding='utf-8')
    assert '水质采样' in (bundle/'task-console.html').read_text(encoding='utf-8')
    # An explicit output-project config must also reach the refresh path.
    write(bundle/CONTEXT_FILENAME, identity('城市散步', '秋日记录', 'walk-2026'))
    refresh_bundle_review_html(bundle)
    refreshed = (bundle/'review.html').read_text(encoding='utf-8')
    assert 'walk-2026' in refreshed
    assert '"video_id": "water-2026"' not in refreshed


def test_empty_inline_identity_claims_are_preserved_not_dropped(tmp_path):
    for key in ('project_identity', 'review_export_context'):
        resolved = resolve_review_export_context({key: {}}, root=tmp_path)
        assert resolved is not None
        assert resolved['topic_title'] == '' and resolved['video_id'] == ''
    explicit_empty = resolve_review_export_context({}, explicit={}, root=tmp_path)
    assert explicit_empty is not None and explicit_empty['video_id'] == ''


def test_empty_manifest_identity_does_not_fall_through_to_second_key(tmp_path):
    write(tmp_path / 'manifest.json', {'project_identity': {}, 'review_export_context': identity()})
    assert configured_identity(tmp_path) == {}
    resolved = resolve_review_export_context({}, root=tmp_path)
    assert resolved is not None and resolved['video_id'] == ''


def test_malformed_manifest_identity_fails_hard(tmp_path):
    write(tmp_path / 'manifest.json', {'project_identity': 'not-a-dict'})
    with pytest.raises(ValueError, match='JSON对象'):
        resolve_review_export_context({}, root=tmp_path)


def test_preparation_non_none_claim_stops_fallthrough_to_root_config(tmp_path):
    input_root = tmp_path / 'in'; output_root = tmp_path / 'out'
    input_root.mkdir(); output_root.mkdir()
    write(input_root / CONTEXT_FILENAME, identity('输入选题', '输入视频', 'input-id'))
    write(output_root / CONTEXT_FILENAME, identity('输出选题', '输出视频', 'output-id'))
    with pytest.raises(ValueError, match='缺少'):
        identity_for_preparation({'project_identity': {}}, input_root, output_root)
    with pytest.raises(ValueError, match='缺少'):
        identity_for_preparation({}, input_root, output_root, explicit={})
    assert identity_for_preparation({'project_identity': identity()}, input_root, output_root)['video_id'] == 'water-2026'
    assert identity_for_preparation({}, input_root, output_root)['video_id'] == 'output-id'
    assert identity_for_preparation({'project_identity': identity()}, input_root, output_root, explicit=identity('显式选题', '显式视频', 'explicit-id'))['video_id'] == 'explicit-id'


_UNSET = object()


def _webui_bundle(tmp_path, *, root_identity=None, output_identity=_UNSET):
    from video_knowledge_pipeline.storage import ensure_project_dirs
    from video_knowledge_pipeline.webui_bridge import export_webui_bundle

    paths = ensure_project_dirs(tmp_path / 'data')
    paths['lecture_packages'].mkdir(parents=True, exist_ok=True)
    if root_identity is not None:
        write(paths['root'] / CONTEXT_FILENAME, root_identity)
    write(paths['lecture_packages'] / 'lecture-package.json', {
        'title': 'CAM_001', 'sources': [], 'timeline': [], 'coverage': {},
    })
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    if output_identity is not _UNSET:
        write(bundle / CONTEXT_FILENAME, output_identity)
    export_webui_bundle(paths['root'], bundle)
    return paths, bundle


def test_webui_export_preserves_empty_output_identity_claim(tmp_path):
    _, bundle = _webui_bundle(
        tmp_path,
        root_identity=identity('根选题', '根视频', 'root-id'),
        output_identity={},
    )
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['project_identity']['video_id'] == ''
    assert 'root-id' not in json.dumps(manifest, ensure_ascii=False)
    html = (bundle / 'review.html').read_text(encoding='utf-8')
    assert '根选题' not in html
    assert 'root-id' not in html


def test_webui_refresh_preserves_empty_output_identity_claim_over_old_manifest(tmp_path):
    from video_knowledge_pipeline.webui_bridge import refresh_bundle_review_html

    _, bundle = _webui_bundle(tmp_path, root_identity=identity('根选题', '根视频', 'root-id'))
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['project_identity']['video_id'] == 'root-id'
    write(bundle / CONTEXT_FILENAME, {})
    refresh_bundle_review_html(bundle)
    refreshed_manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    assert refreshed_manifest['project_identity']['video_id'] == ''
    refreshed = (bundle / 'review.html').read_text(encoding='utf-8')
    assert 'root-id' not in refreshed and '"video_id": "root-id"' not in refreshed
    assert '根选题' not in refreshed


def test_webui_export_prefers_nonempty_output_identity_over_root(tmp_path):
    paths, bundle = _webui_bundle(
        tmp_path,
        root_identity=identity('根选题', '根视频', 'root-id'),
        output_identity=identity('输出选题', '输出视频', 'output-id'),
    )
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['project_identity'] == identity('输出选题', '输出视频', 'output-id')
    html = (bundle / 'review.html').read_text(encoding='utf-8')
    assert '输出选题' in html and '根选题' not in html
    assert 'root-id' not in json.dumps(manifest, ensure_ascii=False)


def test_webui_export_without_any_config_keeps_review_available(tmp_path):
    _, bundle = _webui_bundle(tmp_path)
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    assert 'project_identity' not in manifest
    assert (bundle / 'review.html').is_file()
