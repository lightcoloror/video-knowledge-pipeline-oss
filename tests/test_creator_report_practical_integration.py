"""Practical output is a separate reading projection, never strict admission."""
import copy
import json

import pytest

from test_creator_report import bundle
from video_knowledge_pipeline import creator_report as adapter
from video_knowledge_pipeline.cli import build_parser, main
from video_knowledge_pipeline.creator_report_renderers import html_document
from video_knowledge_pipeline.smart_summary_handoff import STATE, register_review_candidate
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.video_workbench import export_video_workbench


def _registered_bundle(tmp_path):
    root = bundle(tmp_path)
    manifest = read_json(root / 'manifest.json')
    manifest.update(video_type='tutorial', corrected_transcript_json='corrected-transcript.json')
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'corrected-transcript.json', {'segments': [
        {'start': 0, 'end': 2, 'text': '点击“导入”按钮。'},
        {'start': 2, 'end': 4, 'text': '选择“分辨率”，设置为1080p。'},
        {'start': 4, 'end': 4.1, 'text': ''},
        {'start': 4.1, 'end': 10, 'text': '点击“导出”按钮，界面显示“导出完成”。'},
    ]})
    write_json(root / 'transcript-quality-gate.json', {'status': 'failed', 'ok': False})
    source = root / 'reference.md'
    source.write_text('# 参考总结\n\n## 基本信息\n内部运行元数据\n\n## 一句话概览\n导入并导出视频。\n\n'
                      '## 分段总结\n先导入，设置分辨率，再导出。\n\n## 待复核点\n严格记录仍保留。', encoding='utf-8')
    register_review_candidate(root, source, gate={'blocking_reasons': ['synthetic_unresolved']})
    return root


def _snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_practical_build_is_read_only_and_keeps_strict_projection(tmp_path):
    root = _registered_bundle(tmp_path)
    before = _snapshot(root)
    strict = adapter.build_creator_report(root)
    report = adapter.build_creator_report(root, view='practical', video_type='tutorial')
    assert report['summary'] == strict['summary'] == ''
    assert report['transcript'] == strict['transcript']
    assert report['stages'] == strict['stages']
    assert report['practical']['reference_summary']['status'] == 'reference'
    assert '导入并导出视频' in report['practical']['reference_summary']['text']
    assert report['practical']['video_type'] == 'tutorial'
    assert report['candidate_only'] and not report['publication_allowed']
    assert report['operator_boundary']['quality_gate_bypassed'] is False
    assert _snapshot(root) == before
    assert adapter.build_creator_report(root, view='practical', video_type='tutorial')['input_fingerprint'] == report['input_fingerprint']


def test_practical_render_is_reader_oriented_and_marks_local_gaps(tmp_path):
    report = adapter.build_creator_report(_registered_bundle(tmp_path), view='practical', video_type='tutorial')
    text = adapter.render_creator_markdown(report)
    assert '导入并导出视频' in text
    assert '步骤—按钮—参数—结果' in text
    assert '1080p' in text and '待核' in text
    assert '内部运行元数据' not in text
    assert '## 复刻准备' not in text and 'ScriptBreak' not in text
    assert '景别**：未知' not in text
    assert '严格验收记录（保留）' in text
    assert str(report['sources'][0]['path']) not in text


def test_practical_export_uses_own_pointer_and_preserves_strict_files(tmp_path):
    root = _registered_bundle(tmp_path)
    adapter.export_creator_report(root)
    before = _snapshot(root)
    result = adapter.export_creator_report(root, view='practical')
    assert result['status'] == 'exported_candidate'
    assert result['view'] == 'practical'
    assert all(a['path'].startswith(adapter.PRACTICAL_EXPORT_ROOT + '/') for a in result['artifacts'])
    for name, raw in before.items():
        assert (root / name).read_bytes() == raw
    assert adapter.creator_report_status(root, view='practical')['status'] == 'fresh'
    assert read_json(root / STATE)['status'] == 'review-required'
    assert read_json(root / 'transcript-quality-gate.json')['status'] == 'failed'
    assert not (root / 'exports/smart-summary.codex.md').exists()
    cards = export_video_workbench(root, write=False)['artifacts']
    assert next(c for c in cards if c['key'] == 'practical_report_index')['status'] == 'fresh'


def test_changed_reference_candidate_invalidates_practical_export(tmp_path):
    root = _registered_bundle(tmp_path)
    adapter.export_creator_report(root, view='practical')
    state = read_json(root / STATE)
    (root / state['review_candidate_path']).write_text('changed candidate', encoding='utf-8')
    assert adapter.creator_report_status(root, view='practical')['status'] == 'stale'
    report = adapter.build_creator_report(root, view='practical')
    assert not report['practical']['reference_summary']['text']


def test_mid_export_reference_drift_cannot_advance_pointer(tmp_path, monkeypatch):
    root = _registered_bundle(tmp_path)
    previous = adapter.export_creator_report(root, view='practical')
    real = adapter.export_format

    def render(*args, **kwargs):
        real(*args, **kwargs)
        state = read_json(root / STATE)
        (root / state['review_candidate_path']).write_text('changed during export', encoding='utf-8')

    monkeypatch.setattr(adapter, 'export_format', render)
    result = adapter.export_creator_report(root, view='practical')
    assert result['status'] == 'stale_during_export'
    assert read_json(root / adapter.PRACTICAL_EXPORT_ROOT / 'latest.json')['input_fingerprint'] == previous['input_fingerprint']


@pytest.mark.parametrize('view,kind', [('other', 'auto'), ('practical', 'unknown')])
def test_invalid_practical_parameters_fail_before_writes(tmp_path, view, kind):
    root = bundle(tmp_path)
    before = _snapshot(root)
    with pytest.raises(ValueError):
        adapter.export_creator_report(root, view=view, video_type=kind)
    assert _snapshot(root) == before


def test_practical_report_cannot_trigger_creative_templates(tmp_path):
    root = _registered_bundle(tmp_path)
    with pytest.raises(ValueError):
        adapter.export_creator_report(root, view='practical', scriptbreak_script='anything.mjs')
    with pytest.raises(ValueError):
        adapter.export_creator_report(root, view='practical', brief={'product': 'x'})


def test_practical_view_cli_and_mcp_share_existing_entrypoint(tmp_path, capsys, monkeypatch):
    import sys
    import types
    from video_knowledge_pipeline import mcp_server

    root = _registered_bundle(tmp_path)
    args = build_parser().parse_args(['export-creator-report', str(root), '--view', 'practical', '--video-type', 'tutorial', '--no-write'])
    assert args.view == 'practical' and args.video_type == 'tutorial'
    assert main(['export-creator-report', str(root), '--view', 'practical', '--no-write', '--verbose']) == 0
    assert json.loads(capsys.readouterr().out)['report']['practical']['reference_summary']['status'] == 'reference'
    tools = {}

    class Server:
        def __init__(self, *a, **kw): pass
        def tool(self, *a, **kw):
            def add(fn):
                tools[fn.__name__] = fn
                return fn
            return add
        def run(self): pass

    monkeypatch.setitem(sys.modules, 'mcp.server.fastmcp', types.SimpleNamespace(FastMCP=Server))
    mcp_server.main()
    result = tools['export_creator_report'](str(root), view='practical', video_type='tutorial', write=False)
    assert result['report']['practical']['video_type'] == 'tutorial'
    assert tools['creator_report_status'](str(root), view='practical')['status'] == 'missing'


def test_practical_safe_renderer_does_not_embed_untrusted_links(tmp_path):
    report = adapter.build_creator_report(_registered_bundle(tmp_path), view='practical', video_type='tutorial')
    malicious = '<script>alert(1)</script> ![remote](https://example.com/private.png)'
    report = copy.deepcopy(report)
    report['practical']['reference_summary']['text'] = malicious
    report['practical']['steps'][0]['result'] = malicious
    html = html_document(adapter.render_creator_markdown(report), report['title'])
    assert '<script>' not in html and 'src="https://' not in html


def test_practical_defaults_to_known_talking_head_scope_without_auto_classifying(tmp_path, monkeypatch):
    from video_knowledge_pipeline import creator_report_practical_view as practical
    root = _registered_bundle(tmp_path)
    # The fixture's declaration is tutorial; this phase uses the caller's oral
    # scope, rather than interpreting content to select a different workflow.
    real = practical._select_type
    requested = []

    def selected(*args):
        requested.append(args[-1])
        return real(*args)

    monkeypatch.setattr(practical, '_select_type', selected)
    report = adapter.build_creator_report(root, view='practical')
    assert report['practical']['video_type'] == 'talking_head'
    assert requested == ['talking_head']
    text = adapter.render_creator_markdown(report)
    assert '口播与画面分段' in text
    assert '步骤—按钮—参数—结果' not in text
    assert '导入并导出视频' in text
    args = build_parser().parse_args(['export-creator-report', str(root), '--view', 'practical'])
    assert args.video_type == 'talking_head'


@pytest.mark.parametrize('start,end', [('unknown', 2), (float('nan'), 2), (0, float('inf')), (-1, 2)])
def test_malformed_source_issue_clock_is_a_local_note_not_export_failure(tmp_path, start, end):
    report = adapter.build_creator_report(_registered_bundle(tmp_path), view='practical')
    report['practical']['issues'].append({'code': 'synthetic_uncertainty',
        'message': '此处时间待核', 'start': start, 'end': end})
    text = adapter.render_creator_markdown(report)
    assert '此处时间待核' in text and '时间待核：' in text


def test_missing_fine_review_does_not_hide_unresolved_strict_asr_stage(tmp_path):
    root = _registered_bundle(tmp_path)
    report = adapter.build_creator_report(root, view='practical')
    assert any(i['code'] == 'aggregate_asr_review_pending' for i in report['practical']['issues'])
    assert '双路转写核对尚未完成' in adapter.render_creator_markdown(report)
    assert not (root / 'asr-consensus-adjudication-review.json').exists()
