"""Bounded semantic-view/export checks; no real media or model execution."""
import copy
import csv
import hashlib
import io
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from test_creator_report import bundle
from video_knowledge_pipeline import creator_report as adapter
from video_knowledge_pipeline.creator_report_interactive import _SCRIPT, interactive_html_document
from video_knowledge_pipeline.creator_report_oral_chapters import _reference_chapters
from video_knowledge_pipeline.creator_report_renderers import html_document
from video_knowledge_pipeline.creator_report_timed_exports import render_report_srt
from video_knowledge_pipeline.smart_summary_handoff import register_review_candidate
from video_knowledge_pipeline.storage import read_json, write_json


def oral_bundle(tmp_path):
    root = bundle(tmp_path)
    Image.new('RGB', (160, 90), 'blue').save(root / 'frames/b.jpg')
    manifest = read_json(root / 'manifest.json')
    manifest['corrected_transcript_json'] = 'canonical.json'
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'canonical.json', {'segments': [
        {'segment_id': 'source-a', 'source_segment_ids': ['raw-a'], 'start': 0, 'end': 4, 'text': '第一部分。'},
        {'segment_id': 'source-gap', 'start': 4, 'end': 5, 'text': '', 'empty_text_preserved': True},
        {'segment_id': 'source-b', 'start': 5, 'end': 10, 'text': '第二部分。'}]})
    candidate = root / 'candidate.md'
    candidate.write_text('## 一句话概览\n两部分口播。\n\n## 分段总结\n'
                         '### 00:00:00–00:00:05｜开头\n第一部分。\n'
                         '### 00:00:05–00:00:10｜结尾\n第二部分。', encoding='utf-8')
    register_review_candidate(root, candidate, gate={'blocking_reasons': ['unreviewed_fixture']})
    return root


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_oral_chapters_have_images_without_technical_shots_and_keep_sources(tmp_path):
    root = oral_bundle(tmp_path)
    (root / 'exports/scene-detection.json').unlink()
    before = snapshot(root)
    report = adapter.build_creator_report(root, view='practical')
    assert not report['shots'] and len(report['chapter_previews']) == 2
    assert [r['title'] for r in report['practical']['chapters']] == ['开头', '结尾']
    assert all(r['boundary_kind'] == 'semantic_chapter' for r in report['practical']['chapters'])
    text = adapter.render_creator_markdown(report)
    assert text.count('data:image/jpeg;base64,') == 2
    assert '章节 1' in text and '章节 2' in text and '技术分镜详情' in text
    assert report['summary'] == '' and report['practical']['reference_summary']['status'] == 'reference'
    assert snapshot(root) == before


def test_oral_repeated_frames_are_deduplicated_across_chapters(tmp_path):
    root = oral_bundle(tmp_path)
    (root / 'frames/b.jpg').write_bytes((root / 'frames/a.jpg').read_bytes())
    report = adapter.build_creator_report(root, view='practical')
    assert len(report['chapter_previews']) == 1
    assert report['practical']['oral_visual_summary']['unique_sample_hashes'] == 1
    assert adapter.render_creator_markdown(report).count('data:image/jpeg;base64,') == 1


def test_reference_chapter_navigation_covers_rounded_tail_but_rejects_partial_map():
    text = '## 分段总结\n### 00:00:00–00:00:05｜开头\n### 00:00:06–00:00:10｜结尾'
    rows = _reference_chapters(text, 10.9)
    assert rows[0]['end'] == 6 and rows[-1]['end'] == 10.9
    assert rows[-1]['declared_end'] == 10
    assert not _reference_chapters('## 分段总结\n### 00:00:05–00:00:10｜只有后半段', 10)


def test_large_technical_detail_does_not_block_oral_reading(tmp_path, monkeypatch):
    root = oral_bundle(tmp_path)
    original = adapter.build_shot_breakdown(root, write=False)
    large = copy.deepcopy(original)
    large['shots'] = [{**copy.deepcopy(original['shots'][0]), 'shot_id': f'candidate-{i}', 'index': i+1}
                      for i in range(501)]
    monkeypatch.setattr(adapter, 'build_shot_breakdown', lambda *a, **k: copy.deepcopy(large))
    report = adapter.build_creator_report(root, view='practical')
    assert len(report['shots']) == 501 and len(report['chapter_previews']) == 2
    assert adapter.render_creator_markdown(report).count('### 分段 ') == 0
    with pytest.raises(ValueError, match='at most 500'):
        adapter.build_creator_report(root)


def test_oral_hides_only_shot_specific_issue_details_without_closing_review(tmp_path):
    report = adapter.build_creator_report(oral_bundle(tmp_path), view='practical')
    report['practical']['issues'] += [
        {'code': 'shot_missing', 'message': '纯技术分镜缺项详情', 'evidence_refs': ['shots:0']},
        {'code': 'asr_pending', 'message': '真实语音差异仍待核', 'evidence_refs': ['transcript:0']},
        {'code': 'ocr_pending', 'message': '真实OCR疑点仍待核', 'evidence_refs': ['shots:0', 'timeline:0']},
    ]
    before = copy.deepcopy(report)
    text = adapter.render_creator_markdown(report)
    assert '纯技术分镜缺项详情' not in text
    assert '真实语音差异仍待核' in text and '真实OCR疑点仍待核' in text
    assert '局部标记' in text and report == before


def test_external_export_is_source_read_only_and_all_formats_share_binding(tmp_path):
    root = oral_bundle(tmp_path)
    before = snapshot(root)
    target = tmp_path / 'delivery'
    result = adapter.export_creator_report(root, view='practical', formats=['md', 'html', 'srt', 'csv'],
                                           interactive_html=True, output_dir=target)
    assert result['status'] == 'exported_candidate' and snapshot(root) == before
    assert result['publication_allowed'] is False and result['candidate_only'] is True
    files = {row['format']: Path(row['path']) for row in result['artifacts']}
    assert {'md', 'html', 'srt', 'csv', 'technical-shots-csv', 'subtitle-source-map', 'projection'} == set(files)
    source_map = read_json(files['subtitle-source-map'])
    assert source_map['input_fingerprint'] == result['input_fingerprint']
    assert [r['srt_index'] for r in source_map['rows']] == [1, None, 2]
    assert source_map['rows'][0]['source_cue_id'] == 'source-a'
    assert source_map['rows'][0]['source_segment_ids'] == ['raw-a']
    srt = files['srt'].read_text(encoding='utf-8')
    assert re.findall(r'(?m)^\d+$', srt) == ['1', '2']
    assert '00:00:05,000 --> 00:00:10,000' in srt
    rows = list(csv.DictReader(io.StringIO(files['csv'].read_text(encoding='utf-8'))))
    assert len(rows) == 2 and rows[-1]['end_time'] == '00:00:10.000'
    assert all(r['boundary_kind'] == 'semantic_chapter' and r['candidate_only'] == 'True' for r in rows)
    assert adapter.creator_report_status(root, view='practical', output_dir=target)['status'] == 'fresh'
    files['srt'].write_text('tampered', encoding='utf-8')
    status = adapter.creator_report_status(root, view='practical', output_dir=target)
    assert status['status'] == 'stale' and 'artifact_changed:srt' in status['issues']


@pytest.mark.parametrize('location', ['same', 'child', 'ancestor'])
def test_external_export_rejects_source_tree_overlap_before_writes(tmp_path, location):
    root = oral_bundle(tmp_path)
    before = snapshot(root)
    destination = {'same': root, 'child': root / 'new-delivery', 'ancestor': root.parent}[location]
    with pytest.raises(ValueError, match='must not overlap'):
        adapter.export_creator_report(root, view='practical', output_dir=destination)
    assert snapshot(root) == before
    if location == 'child':
        assert not destination.exists()


def test_external_output_resolved_alias_into_source_is_rejected(tmp_path, monkeypatch):
    root = oral_bundle(tmp_path)
    alias = tmp_path / 'junction-alias'
    original_resolve = Path.resolve
    monkeypatch.setattr(Path, 'resolve', lambda self, *a, **kw: root if self == alias else original_resolve(self, *a, **kw))
    with pytest.raises(ValueError, match='must not overlap'):
        adapter.export_creator_report(root, view='practical', output_dir=alias)
    assert not alias.exists()


def test_external_export_preserves_unowned_or_user_changed_index(tmp_path):
    root = oral_bundle(tmp_path)
    unowned = tmp_path / 'unowned'
    unowned.mkdir()
    (unowned / 'index.html').write_text('unrelated existing index', encoding='utf-8')
    before = snapshot(unowned)
    with pytest.raises(ValueError, match='unowned index'):
        adapter.export_creator_report(root, view='practical', output_dir=unowned)
    assert snapshot(unowned) == before
    owned = tmp_path / 'owned'
    first = adapter.export_creator_report(root, view='practical', output_dir=owned)
    second = adapter.export_creator_report(root, view='practical', output_dir=owned)
    assert first['input_fingerprint'] == second['input_fingerprint']
    (owned / 'index.html').write_text('user changed index', encoding='utf-8')
    before = snapshot(owned)
    with pytest.raises(ValueError, match='ownership/content changed'):
        adapter.export_creator_report(root, view='practical', output_dir=owned)
    assert snapshot(owned) == before
    assert 'index_changed' in adapter.creator_report_status(root, view='practical', output_dir=owned)['issues']


def test_srt_rejects_nonempty_zero_duration_but_preserves_zero_duration_gap():
    report = {'input_fingerprint': 'f' * 64, 'transcript_scope': 'canonical', 'transcript': [
        {'start': 1, 'end': 1, 'text': 'nonempty', 'source_cue_id': 'original'}]}
    before = copy.deepcopy(report)
    with pytest.raises(ValueError, match='positive duration'):
        render_report_srt(report)
    assert report == before
    report['transcript'] = [{'start': 0, 'end': 0, 'text': '', 'source_cue_id': 'gap'},
                            {'start': 1, 'end': 2, 'text': 'valid', 'source_cue_id': 'voice'}]
    subtitle, mapping = render_report_srt(report)
    assert [r['srt_index'] for r in mapping['rows']] == [None, 1]
    assert mapping['rows'][0]['source_cue_id'] == 'gap' and '00:00:01,000 --> 00:00:02,000' in subtitle


def test_interactive_html_is_explicit_safe_and_has_time_and_frame_seek(tmp_path):
    report = adapter.build_creator_report(oral_bundle(tmp_path), view='practical')
    markdown = adapter.render_creator_markdown(report)
    plain = html_document(markdown, report['title'])
    interactive = interactive_html_document(markdown, report['title'], report)
    assert '<script>' not in plain and '<video' not in plain
    assert 'data-seek="5.000"' in interactive and 'class="seek-frame"' in interactive
    assert 'media-src blob:' in interactive and "connect-src &#x27;none&#x27;" in interactive
    assert "script-src 'unsafe-inline'" not in interactive
    import base64
    expected = base64.b64encode(hashlib.sha256(_SCRIPT.encode()).digest()).decode()
    assert 'sha256-' + expected in interactive
    assert str(tmp_path) not in interactive and 'src="file:' not in interactive


def test_player_actual_javascript_events_without_browser(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.fail('Node is required for the offline player behavior acceptance')
    script = tmp_path / 'player.js'
    script.write_text(_SCRIPT, encoding='utf-8')
    harness = Path(__file__).with_name('creator_report_player_harness.cjs')
    result = subprocess.run([node, str(harness), str(script)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'player behavior assertions passed' in result.stdout
