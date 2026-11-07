from __future__ import annotations

import copy
import json

import pytest

from video_knowledge_pipeline.storage import write_json


def citations(count=2):
    return [{'source_type': 'visual_understanding_inferred', 'evidence_level': 'inferred',
             'inference_basis': '合成 OCR 与画面一致但仍为推断', 'time': '00:00:00 - 00:00:02',
             'timeline_indexes': [1], 'text': '相同的前缀。' * 30 + f'尾部限制 {n}，续期除外。',
             'evidence_paths': [f'frames/{n}.png'], 'evidence_id': f'cell-{n}',
             'source_context': {'cell_index': n, 'footnote_applicability': 'not_assessed'},
             'source_locators': []} for n in range(count)]


def test_navigation_roundtrip_preserves_levels_tail_conditions_and_identities():
    from video_knowledge_pipeline.smart_summary_codex import _chapter_citation_rows, _citation_lines
    from video_knowledge_pipeline.knowledge_note_export import _compact_content_candidate_citation, _candidate_citation_lines
    source = citations(30)
    source[1]['evidence_level'] = 'candidate_weak'
    source[1]['source_type'] = 'continuous_tag_weak_evidence'
    before = copy.deepcopy(source)
    rows = _chapter_citation_rows([{'index': 1, 'citation_digest': source}])
    assert len(rows) == 30
    assert rows[0]['evidence_level'] == 'inferred' and rows[1]['evidence_level'] == 'candidate_weak'
    assert rows[-1]['text'] == source[-1]['text']
    assert rows[0]['source_context'] == source[0]['source_context']
    compact = _compact_content_candidate_citation(rows[0])
    assert compact['evidence_level'] == 'inferred' and compact['text'] == rows[0]['text']
    text = '\n'.join(_citation_lines(rows))
    assert '24/30' in text and 'smart-summary-chapters.json' in text
    assert 'evidence_level=inferred' in text and 'evidence_level=candidate_weak' in text
    assert '8/30' in '\n'.join(_candidate_citation_lines(rows))
    assert source == before


def test_unclassified_legacy_citation_is_not_promoted_to_confirmed():
    from video_knowledge_pipeline.smart_summary_codex import _chapter_citation_rows, _citation_lines
    rows = _chapter_citation_rows([{'index': 1, 'citation_digest': [{'source_type': 'ocr_or_ebook', 'text': '老识别结果'}]}])
    assert rows[0]['evidence_level'] == 'not_assessed'
    assert 'evidence_level=confirmed' not in '\n'.join(_citation_lines(rows))


def test_chapter_digest_retains_full_input_trace_not_first_four_or_sixteen():
    from video_knowledge_pipeline.smart_summary_chapters import _chapter_evidence_trace, _chapter_citation_digest
    bucket = [{'segment_id': f's{n}', 'timeline_index': 1, 'start': 0, 'end': 2,
               'raw_text': '原文语句，不能丢失最后限定条件。' * 15 + str(n)} for n in range(6)]
    trace = {'ocr_items': [{'timeline_index': 1, 'excerpt': '观察值 ' + str(n),
                          'evidence_level': 'candidate_evidence'} for n in range(121)]}
    scoped = _chapter_evidence_trace(trace, bucket, 0, 2)
    assert len(scoped['ocr_items']) == 121
    digest = _chapter_citation_digest(bucket, scoped)
    assert len(digest) == 127
    assert digest[5]['text'] == bucket[5]['raw_text']
    assert digest[-1]['evidence_level'] == 'candidate_evidence'


def native_chapters(tmp_path, monkeypatch):
    from test_courseware_material import fixture_bundle
    from test_courseware_structure import native_page
    from video_knowledge_pipeline.courseware_material import import_courseware_page
    from video_knowledge_pipeline import smart_summary_chapters as chapters
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, native_page(root))
    monkeypatch.setattr(chapters, 'load_bundle_term_replacements', lambda *_: [('方案甲', '不得改写原始识别')])
    pack = chapters.build_smart_summary_chapter_pack(root, write=True, target_chapters=1, chapter_mode='fixed')
    return root, pack


def test_native_courseware_is_in_real_chapter_digest_without_raw_replacement(tmp_path, monkeypatch):
    root, pack = native_chapters(tmp_path, monkeypatch)
    rows = pack['chapters'][0]['citation_digest']
    native = [r for r in rows if r.get('evidence_id', '').startswith('courseware:')]
    assert len(native) == 10
    row = next(r for r in native if r['evidence_id'].endswith(':cell:4'))
    assert row['evidence_level'] == 'candidate_evidence' and row['timeline_indexes'] == [1]
    assert '方案甲' in row['text'] and '不得改写' not in row['text']
    assert row['source_locators'] and row['source_context']['footnote_applicability'] == 'not_assessed'
    from video_knowledge_pipeline.citation_digest import audit_citation_digest
    report = audit_citation_digest(root)
    assert report['passed'] and report['status'] == 'verified_navigation'
    assert report['semantic_support'] == 'not_assessed'
    assert report['source_completeness'] == 'not_assessed'


@pytest.mark.parametrize('drift', ['row', 'timeline', 'page', 'frame', 'binding'])
def test_navigation_drift_blocks_existing_summary_quality_gate(tmp_path, monkeypatch, drift):
    from video_knowledge_pipeline.citation_digest import audit_citation_digest
    root, pack = native_chapters(tmp_path, monkeypatch)
    if drift == 'row':
        pack['chapters'][0]['citation_digest'][-1]['evidence_level'] = 'confirmed'
        write_json(root / 'exports/smart-summary-chapters.json', pack)
    elif drift == 'binding':
        pack.pop('citation_digest_binding')
        write_json(root / 'exports/smart-summary-chapters.json', pack)
    else:
        path = {'timeline': 'timeline.json', 'page': 'courseware/pages/page-0.json',
                'frame': 'courseware/frames/0.png'}[drift]
        target = root / path
        target.write_bytes(target.read_bytes() + b'\n')
    before = (root / 'timeline.json').read_bytes()
    result = audit_citation_digest(root)
    assert result['evaluated'] and not result['passed'] and result['status'] == 'blocked'
    assert (root / 'timeline.json').read_bytes() == before
    from video_knowledge_pipeline.smart_summary_codex import smart_summary_quality_check
    (root / 'exports/smart-summary.codex.md').write_text('# 合成总结\n\n' + '需要复核。' * 40, encoding='utf-8')
    quality = smart_summary_quality_check(root, write=False)
    assert any(r['key'] == 'citation_digest_integrity' and r['passed'] is False for r in quality['checks'])


def test_old_navigation_without_binding_is_unknown_not_a_completeness_pass(tmp_path):
    from video_knowledge_pipeline.citation_digest import audit_citation_digest
    write_json(tmp_path / 'exports/smart-summary-chapters.json', {'chapters': [{'citation_digest': citations()}]})
    result = audit_citation_digest(tmp_path)
    assert result['status'] == 'not_assessed' and result['passed'] is None


def test_bound_transcript_artifact_drift_is_checked_even_if_timeline_unchanged(tmp_path):
    from video_knowledge_pipeline.citation_digest import SCHEMA, digest_records, bind_citation_digest, audit_citation_digest
    from video_knowledge_pipeline.file_hash import sha256_file
    write_json(tmp_path / 'timeline.json', [])
    source = tmp_path / 'transcript.json'
    write_json(source, {'segments': [{'text': '首次适用，续期除外。'}]})
    rows = digest_records([{'source_type': 'transcript', 'text': '首次适用，续期除外。',
        'source_path': str(source), 'source_sha256': sha256_file(source)}])
    pack = {'citation_digest_schema': SCHEMA, 'chapters': [{'index': 1, 'start': 0, 'end': 1, 'citation_digest': rows}]}
    pack['citation_digest_binding'] = bind_citation_digest(tmp_path, pack)
    write_json(tmp_path / 'exports/smart-summary-chapters.json', pack)
    assert audit_citation_digest(tmp_path)['passed']
    write_json(source, {'segments': [{'text': '不再适用。'}]})
    report = audit_citation_digest(tmp_path)
    assert report['status'] == 'blocked' and 'transcript_source_hash_drift' in report['detail']


def test_digest_records_have_explicit_bounds_and_no_prefix_dedup():
    from video_knowledge_pipeline.citation_digest import digest_records, MAX_RECORDS
    assert len(digest_records(citations(3))) == 3
    with pytest.raises(ValueError, match='budget'):
        digest_records([{'text': 'x'}] * (MAX_RECORDS + 1))
    with pytest.raises(ValueError, match='promoted'):
        digest_records([{'source_type': 'continuous_tag_weak_evidence', 'evidence_level': 'confirmed', 'text': '标签'}])
    conflict = citations()
    conflict[1]['evidence_id'] = conflict[0]['evidence_id']
    with pytest.raises(ValueError, match='conflicting_evidence_id'):
        digest_records(conflict)
    repeated = [{**citations(1)[0], 'chapter_index': n} for n in (1, 2)]
    assert len(digest_records(repeated)) == 2


def test_root_trace_keeps_late_ocr_and_all_original_text():
    from video_knowledge_pipeline.smart_summary_input_pack import _evidence_trace
    original = '这个值并非适用于所有情况；' * 60 + '最后脚注：除外责任。'
    timeline = [{'index': n + 1, 'start': n, 'end': n + 1, 'ocr_text': original} for n in range(125)]
    trace = _evidence_trace(timeline, [], {}, transcript_path=None, manifest={})
    assert len(trace['ocr_items']) == 125
    assert trace['ocr_items'][-1]['excerpt'] == original
    assert trace['ocr_items'][-1]['evidence_level'] == 'candidate_evidence'


@pytest.mark.parametrize('missing', ['file', 'all_markers'])
def test_registered_navigation_cannot_silently_become_legacy(tmp_path, monkeypatch, missing):
    from video_knowledge_pipeline.citation_digest import audit_citation_digest
    root, pack = native_chapters(tmp_path, monkeypatch)
    path = root / 'exports/smart-summary-chapters.json'
    if missing == 'file':
        path.rename(path.with_suffix('.retained.json'))  # synthetic file only
    else:
        write_json(path, {'chapters': []})
    report = audit_citation_digest(root)
    assert report['status'] == 'blocked' and report['passed'] is False


def test_workbenches_and_coverage_share_readonly_navigation_audit(tmp_path, monkeypatch):
    from video_knowledge_pipeline.file_hash import sha256_file
    from video_knowledge_pipeline.knowledge_coverage import audit_knowledge_coverage
    from video_knowledge_pipeline.task_console import export_task_console, _render_task_console_html, _citation_digest_panel_html
    from video_knowledge_pipeline.video_workbench import export_video_workbench, _render_workbench_html
    root, _ = native_chapters(tmp_path, monkeypatch)
    before = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob('*') if p.is_file()}
    report = audit_knowledge_coverage(root, write=False)['coverage']['citation_digest_integrity']
    for export, render in ((export_task_console, _render_task_console_html),
                           (export_video_workbench, lambda v: _render_workbench_html(v, []))):
        result = export(root, write=False)
        assert result['citation_digest_integrity'] == report
        rendered = render(result)
        assert 'id="citation-digest-integrity"' in rendered and '完整引用 / 等级 / 精确定位' in rendered
    assert before == {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob('*') if p.is_file()}
    html = _citation_digest_panel_html({'evaluated': True, 'status': '<script>', 'detail': '<script>',
                                     'evidence_levels': {'<script>': 1}})
    assert '<script>' not in html and '&lt;script&gt;' in html
