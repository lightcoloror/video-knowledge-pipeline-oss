"""Offline native Docling contract adaptation, not a second table parser."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import re

import pytest

from test_courseware_material import fixture_bundle, page_result
from video_knowledge_pipeline.courseware_material import (
    build_courseware_material, import_courseware_page, render_courseware_model_input,
)
from video_knowledge_pipeline.evidence_locator import verify_evidence_locators


def cell(row, col, text, **flags):
    return dict(text=text, row_span=1, col_span=1, start_row_offset_idx=row,
                end_row_offset_idx=row + 1, start_col_offset_idx=col,
                end_col_offset_idx=col + 1, column_header=False, row_header=False,
                row_section=False) | flags


def native_page(root):
    page = page_result(root, text='合成产品对照表：首期与续期分别列示，数值不可跨行拼接。')
    page['blocks'][0].update(kind='table', docling_ref='#/tables/0')
    page['docling_document'] = {
        'schema_name': 'DoclingDocument', 'version': '1.10.0',
        'texts': [{'self_ref': '#/texts/0', 'label': 'footnote',
                   'text': '仅限指定范围，是否约束某个单元格仍需复核。', 'prov': []}],
        'tables': [{'self_ref': '#/tables/0', 'label': 'table', 'prov': [],
                    'footnotes': [{'$ref': '#/texts/0'}],
                    'data': {'num_rows': 3, 'num_cols': 3, 'table_cells': [
                        cell(0, 0, '产品', column_header=True),
                        cell(0, 1, '首期比例', column_header=True),
                        cell(0, 2, '续期比例', column_header=True),
                        cell(1, 0, '方案甲', row_header=True), cell(1, 1, '80%'), cell(1, 2, '60%'),
                        cell(2, 0, '方案乙', row_header=True), cell(2, 1, '90%'), cell(2, 2, '70%'),
                    ]}}],
    }
    return page


def projected(root):
    material = build_courseware_material(root, write=False)
    from video_knowledge_pipeline.smart_summary_global_reduce import _chapter_fact_section
    section = _chapter_fact_section({'section_id': 's'}, {'evidence': {'courseware_material': material}})
    return material, section


def test_native_cells_keep_row_column_notes_and_exact_source_identity(tmp_path):
    root = fixture_bundle(tmp_path)
    page = native_page(root)
    import_courseware_page(root, page)
    source = root / 'courseware/pages/page-0.json'
    before = source.read_bytes()
    material, section = projected(root)
    refs = [r for r in section['evidence_refs'] if ':cell:' in r['evidence_id']]
    assert len(refs) == 9  # Header-only cells are also retained, not silently omitted.
    assert sum(r['source_context']['cell_role'] == 'value' for r in refs) == 4
    first = next(r for r in refs if r['evidence_id'].endswith(':cell:4'))
    assert '方案甲' in first['snippet'] and '首期比例' in first['snippet'] and '80%' in first['snippet']
    assert '方案乙' not in first['snippet'] and '90%' not in first['snippet']
    assert first['source_context']['footnote_applicability'] == 'not_assessed'
    assert first['fact_status'] == 'candidate_evidence'
    assert len({r['source_context']['context_id'] for r in refs}) == 9
    selectors = [x['selector'] for x in first['source_locators']]
    assert ['docling_document', 'tables', 0, 'data', 'table_cells', 4] in selectors
    assert ['docling_document', 'texts', 0] in selectors
    assert verify_evidence_locators(root, first['source_locators'])['status'] == 'verified'
    prompt = render_courseware_model_input(material)
    assert '80%' in prompt and '首期比例' in prompt and 'footnote_applicability' in prompt
    assert 'source_locators' not in prompt
    assert source.read_bytes() == before and json.loads(before) == page
    source.write_bytes(before + b'\n')
    with pytest.raises(ValueError, match='drift'):
        verify_evidence_locators(root, first['source_locators'])


@pytest.mark.parametrize('case', ['zero_span', 'overlap', 'outside', 'bool_offset', 'span_mismatch',
    'external_ref', 'missing_ref', 'cross_page', 'wrong_note_kind', 'missing_table',
    'orphan_table', 'spoof_projection', 'nonfinite_bbox', 'huge_grid', 'missing_version', 'rich_cell'])
def test_invalid_native_structure_refuses_whole_import(tmp_path, case):
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, page_result(root))
    source = root / 'courseware/pages/page-0.json'
    before = source.read_bytes()
    timeline = (root / 'timeline.json').read_bytes()
    page = native_page(root)
    from video_knowledge_pipeline.file_hash import sha256_file
    page['supersedes_sha256'] = sha256_file(source)
    table = page['docling_document']['tables'][0]
    cells = table['data']['table_cells']
    if case == 'zero_span': cells[4]['row_span'] = 0
    elif case == 'overlap': cells.append(copy.deepcopy(cells[4]))
    elif case == 'outside': cells[4]['end_col_offset_idx'] = 4
    elif case == 'bool_offset': cells[4]['start_row_offset_idx'] = True
    elif case == 'span_mismatch': cells[4]['row_span'] = 2
    elif case == 'external_ref': table['footnotes'][0]['$ref'] = '../other.json#/texts/0'
    elif case == 'missing_ref': table['footnotes'][0]['$ref'] = '#/texts/99'
    elif case == 'cross_page':
        page['docling_page_no'] = 1
        table['prov'] = [{'page_no': 2}]
    elif case == 'wrong_note_kind': page['docling_document']['texts'][0]['label'] = 'title'
    elif case == 'missing_table': page['blocks'][0]['docling_ref'] = '#/tables/1'
    elif case == 'orphan_table': page['docling_document']['tables'].append(copy.deepcopy(table))
    elif case == 'spoof_projection': page['blocks'][0]['structural_evidence'] = [{'fact_status': 'confirmed'}]
    elif case == 'nonfinite_bbox': cells[4]['bbox'] = {'l': 0, 't': 0, 'r': float('inf'), 'b': 1, 'coord_origin': 'TOPLEFT'}
    elif case == 'huge_grid': table['data']['num_rows'] = 1000000
    elif case == 'missing_version': del page['docling_document']['version']
    elif case == 'rich_cell': cells[4]['ref'] = {'$ref': '#/groups/0'}
    with pytest.raises(ValueError, match='structure'):
        import_courseware_page(root, page)
    assert source.read_bytes() == before and (root / 'timeline.json').read_bytes() == timeline
    receipts = list((root / 'courseware/raw').glob('*.validation.json'))
    assert any(json.loads(p.read_text())['changed'] is False for p in receipts)


def test_merged_header_native_span_and_missing_flags_are_not_guessed(tmp_path):
    root = fixture_bundle(tmp_path)
    page = native_page(root)
    cells = page['docling_document']['tables'][0]['data']['table_cells']
    cells[1].update(col_span=2, end_col_offset_idx=3, text='比例（%）')
    del cells[2]
    cells[2]['row_header'] = False  # "方案甲" alone is not a declared row header.
    import_courseware_page(root, page)
    _, section = projected(root)
    refs = [r for r in section['evidence_refs'] if ':cell:' in r['evidence_id'] and '80%' in r['snippet']]
    assert len(refs) == 1 and '比例（%）' in refs[0]['snippet']
    assert '方案甲' not in refs[0]['snippet']
    assert 'row_header_not_declared' in refs[0]['source_context']['limitations']


def test_legacy_pages_do_not_gain_structure_or_change_bytes(tmp_path):
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, page_result(root))
    before = (root / 'courseware/pages/page-0.json').read_bytes()
    material, section = projected(root)
    assert not any(':cell:' in r['evidence_id'] for r in section['evidence_refs'])
    assert 'structural_evidence' not in material['pages'][0]['blocks'][0]
    assert (root / 'courseware/pages/page-0.json').read_bytes() == before


def test_native_bbox_preserves_coordinate_origin_without_guessing_pixel_geometry(tmp_path):
    root = fixture_bundle(tmp_path)
    page = native_page(root)
    page['docling_page_no'] = 1
    table = page['docling_document']['tables'][0]
    box = {'l': 10, 'r': 20, 't': 90, 'b': 80, 'coord_origin': 'BOTTOMLEFT'}
    table['prov'] = [{'page_no': 1, 'bbox': box}]
    table['data']['table_cells'][4]['bbox'] = box
    import_courseware_page(root, page)
    material, section = projected(root)
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'].endswith(':cell:4'))
    assert ref['source_context']['bbox'] == box
    assert material['pages'][0]['blocks'][0]['native_structure']['native_document_version'] == '1.10.0'


@pytest.mark.parametrize('resolution', [None, 'unresolved', 'not_applicable', 'applies'])
def test_actual_review_prepare_keeps_cell_context_and_stays_bounded(tmp_path, resolution):
    from video_knowledge_pipeline.canonical_json import canonical_json_bytes
    from video_knowledge_pipeline.storage import write_json
    from video_knowledge_pipeline.summary_claim_review import prepare_summary_claim_review
    from video_knowledge_pipeline.summary_citation_audit import build_citation_index
    from video_knowledge_pipeline.summary_claim_review import _scope_sources
    root = fixture_bundle(tmp_path)
    page = native_page(root)
    if resolution == 'not_applicable':
        page['docling_document']['texts'][0]['text'] = '仅方案乙受本脚注限制，方案甲不适用。'
    elif resolution == 'applies':
        page['docling_document']['texts'][0]['text'] = '方案甲仅限首次参与。'
    import_courseware_page(root, page)
    _, section = projected(root)
    pack = {'bundle_dir': str(root), 'sections': [section]}
    cited = 'courseware:page-0:0:cell:4'
    plan = {'overview': {'text': '方案甲首期比例为80%，脚注适用范围需复核。', 'evidence_ids': [cited]}}
    if resolution == 'applies':
        plan['overview']['text'] = '方案甲首次参与的首期比例为80%。'
    elif resolution == 'not_applicable':
        plan['overview']['text'] = '方案甲首期比例为80%，本脚注仅适用于方案乙。'
    write_json(root / 'exports/smart-summary-reader-plan.json', plan)
    write_json(root / 'exports/smart-summary-chapter-fact-pack.json', pack)
    index = build_citation_index(pack)
    assert not index.errors
    sources = _scope_sources(root, index, pack, [cited])
    row = next(s for s in sources if s['evidence_id'] == cited)
    assert row['source_context']['row_header_indexes'] == [3]
    assert row['context_id'] == cited
    assert any(o.get('text') == '80%' for o in row['original_objects'])
    from video_knowledge_pipeline.summary_claim_review import _task_pack, _view
    from video_knowledge_pipeline.file_hash import sha256_file
    expected = {key: sha256_file(root / key) for key in (
        'exports/smart-summary-reader-plan.json', 'exports/smart-summary-chapter-fact-pack.json')}
    expected['summary-claim-review.json'] = '0' * 64
    sizes = {s['evidence_id']: len(canonical_json_bytes(_task_pack(
        _view(plan, pack)[0], 'overview.text', plan['overview']['text'], [s], [cited], expected, {},
        {'index': 4095, 'count': 4096, 'partition_sha256': '0' * 64}))) for s in sources}
    assert max(sizes.values()) + 512 <= 8192, sizes
    task = prepare_summary_claim_review(root)
    assert task['status'] == 'waiting_result', task
    assert len(canonical_json_bytes(task)) <= 8192
    assert prepare_summary_claim_review(root) == task
    from video_knowledge_pipeline.summary_claim_review import import_summary_claim_review, summary_claim_review_report
    seen, last_review, receipt = [], None, None
    slots = dict(subject='方案甲', property='首期比例', value='80', unit='%',
                 negated=False, conditions=['首次参与'] if resolution == 'applies' else [], time_scope='', version='')
    for _ in range(40):
        task = prepare_summary_claim_review(root)
        if 'task_id' not in task:
            break
        assert len(canonical_json_bytes(task)) <= 8192
        result = {key: task[key] for key in ('task_id', 'task_pack_sha256')}
        result.update(executor='synthetic', model_actual='fixture',
                      executor_run_id=task['phase'],
                      observed_source_ids=[s['evidence_id'] for s in task['sources']])
        if task['phase'] == 'annotate':
            seen.extend(result['observed_source_ids'])
            result.update(claims=[{'claim_id': 'c', 'evidence_ids': [cited], 'slots': slots}], assertions=[])
            for s in task['sources']:
                if s['evidence_id'] == cited:
                    result['assertions'].append({'assertion_id': 'a', 'evidence_id': cited,
                        'start': 0, 'end': len(s['snippet']), 'slots': slots})
                    if resolution:
                        result['qualification_assessments'] = [dict(assertion_id='a', qualifier_id='#/texts/0',
                            status=resolution, reason='Synthetic note scope checked against the named product.',
                            condition_indexes=[0] if resolution == 'applies' else [])]
        else:
            last_review = task
            result.update(source_interpretation_correct=True, output_interpretation_correct=True,
                          decomposition_complete=True, context_association_correct=True,
                          reason='Synthetic cell, row, column and declared note decision checked; unresolved remains a gap.')
        if task['phase'] == 'annotate' and resolution == 'applies' and result.get('qualification_assessments'):
            state_path = root / 'summary-claim-review.json'
            before = state_path.read_bytes() if state_path.exists() else None
            invalid = copy.deepcopy(result)
            invalid['qualification_assessments'][0]['condition_indexes'] = [12]
            write_json(root / 'invalid-note-answer.json', invalid)
            rejected = import_summary_claim_review(root, root / 'invalid-note-answer.json')
            assert not rejected['ok'] and not rejected['changed']
            assert (state_path.read_bytes() if state_path.exists() else None) == before
            assert prepare_summary_claim_review(root) == task
        write_json(root / 'table-answer.json', result)
        receipt = import_summary_claim_review(root, root / 'table-answer.json')
        assert receipt['ok'], receipt
    else:
        pytest.fail('Bounded table review did not terminate')
    assert set(seen) == {s['evidence_id'] for s in sources} and len(seen) == len(set(seen))
    report = summary_claim_review_report(root, plan, pack)
    assert report['passed'] is (resolution in {'applies', 'not_applicable'})
    from video_knowledge_pipeline.transcript_semantic_transaction import apply_transcript_semantic_restore
    tx = receipt['transaction']
    assert apply_transcript_semantic_restore(root, tx['restore_plan'], execute=True,
        confirm_transaction_id=tx['transaction_id'])['status'] == 'restored'
    assert prepare_summary_claim_review(root) == last_review


def test_optional_fixed_docling_source_slice_and_official_csv_fixture(tmp_path):
    source = os.environ.get('VKP_DOCLING_SOURCE')
    if not source:
        pytest.skip('Explicit local Docling source required; no downloads')
    path = Path(source)
    tree = ast.parse((path / 'docling/backend/html_backend.py').read_text(encoding='utf-8'))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_get_cell_spans']
    assert len(nodes) == 1
    node = copy.deepcopy(nodes[0])
    node.decorator_list = []
    namespace = {'Tag': dict, 're': re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), 'docling-source-slice', 'exec'), namespace)
    assert namespace['_get_cell_spans']({'colspan': '2', 'rowspan': '1'}) == (2, 1)
    assert namespace['_get_cell_spans']({'colspan': '0'}) == (0, 1)  # Our safety boundary must reject this.
    document = json.loads((path / 'tests/data/groundtruth/docling_v2/csv-comma-in-cell.csv.json').read_text(encoding='utf-8'))
    root = fixture_bundle(tmp_path)
    page = page_result(root, text='Public synthetic CSV fixture table with quoted commas.')
    page['blocks'][0].update(kind='table', docling_ref='#/tables/0')
    page['docling_document'] = document
    import_courseware_page(root, page)
    _, section = projected(root)
    assert any(':cell:' in r['evidence_id'] for r in section['evidence_refs'])
