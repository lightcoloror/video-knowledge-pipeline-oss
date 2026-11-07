"""Native table context stays explicit; position is not semantic applicability."""
from copy import deepcopy

import pytest

from test_courseware_material import fixture_bundle, page_result
from test_courseware_structure import cell, projected
from test_qualification_scope import case, check
from video_knowledge_pipeline.canonical_json import canonical_json_bytes, canonical_json_sha256 as digest
from video_knowledge_pipeline.courseware_material import import_courseware_page
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.summary_citation_audit import build_citation_index
from video_knowledge_pipeline import summary_claim_review as review


def native_context_page(root, kind='row_section'):
    page = page_result(root, text='合成分组表，两个版本的数值相同。')
    page['blocks'][0].update(kind='table', docling_ref='#/tables/0')
    page['docling_document'] = dict(schema_name='DoclingDocument', version='1.10.0', texts=[],
        tables=[dict(self_ref='#/tables/0', label='table', prov=[], data=dict(num_rows=5, num_cols=1,
            table_cells=[cell(0, 0, '比例', column_header=True),
                         cell(1, 0, '方案甲2026版', row_section=True), cell(2, 0, '80%'),
                         cell(3, 0, '方案乙2025版', row_section=True), cell(4, 0, '80%')]))])
    if kind == 'captions':
        document = page['docling_document']
        for value in document['tables'][0]['data']['table_cells']:
            value['row_section'] = False
        document['texts'] = [dict(self_ref=f'#/texts/{i}', label='caption', text=text, prov=[])
                             for i, text in enumerate(('方案甲2026版', '方案乙2025版'))]
        document['tables'][0]['captions'] = [{'$ref': f'#/texts/{i}'} for i in range(2)]
    return page


def context_bundle(tmp_path, kind='row_section'):
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, native_context_page(root, kind))
    _, section = projected(root)
    cited = 'courseware:page-0:0:cell:2'
    # One explicit section scope with one value reference: all native context
    # still must be supplied inside that reference, not guessed from neighbors.
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'] == cited)
    pack = {'sections': [{'section_id': 's', 'evidence_refs': [ref]}]}
    plan = {'overview': {'text': '方案甲2026版比例80%。', 'evidence_ids': [cited]}}
    write_json(root / review.PLAN, plan)
    write_json(root / review.FACTS, pack)
    return root, plan, pack, ref


def test_native_row_sections_remain_candidates_with_original_locators(tmp_path):
    root, plan, pack, ref = context_bundle(tmp_path)
    context = ref['source_context']
    expected = {'#/tables/0/data/table_cells/1': 'row_section', '#/tables/0/data/table_cells/3': 'row_section'}
    assert context['qualification_roles'] == expected
    assert context['qualification_refs'] == sorted(expected)
    task = review.prepare_summary_claim_review(root)
    assert len(canonical_json_bytes(task)) <= 8192
    objects = task['sources'][0]['original_objects']
    assert {'方案甲2026版', '方案乙2025版'} <= {o['text'] for o in objects}
    assert task['sources'][0]['source_context']['qualification_roles'] == expected


@pytest.mark.parametrize('tamper', ['catalog', 'roles', 'locator'])
def test_context_omission_cannot_bypass_original_native_catalog(tmp_path, tamper):
    root, _, pack, ref = context_bundle(tmp_path)
    if tamper == 'catalog':
        ref['source_context']['qualification_refs'] = []
    elif tamper == 'roles':
        ref['source_context']['qualification_roles'] = {}
    else:
        ref['source_locators'] = [loc for loc in ref['source_locators']
                                 if loc['selector'][-1] != 3]
    write_json(root / review.FACTS, pack)
    with pytest.raises(ValueError, match='qualification_catalog'):
        review.prepare_summary_claim_review(root)


def scoped_case(status='applies'):
    index, scope, claims, texts = case(status)
    ref = deepcopy(index.refs['cell'])
    ref['source_context']['qualification_roles'] = {'#/texts/0': 'captions'}
    index = build_citation_index({'sections': [{'evidence_refs': [ref]}]})
    scope['reference_snapshot_sha256'] = index.snapshot_sha256
    scope['assertions'][0]['reference_canonical_sha256'] = digest(ref)
    scope['assertions'][0]['slots']['version'] = '2026版'
    claims[0]['slots']['version'] = '2026版'
    decision = scope['qualification_assessments'][0]
    decision.update(condition_indexes=[], slot_fields=['version'] if status == 'applies' else [])
    return index, scope, claims, texts


def test_context_applies_to_declared_version_not_invented_condition():
    result = check(scoped_case())
    assert result['valid'], result['errors']
    assert result['claims'][0]['status'] == 'supported'
    wrong = scoped_case()
    wrong[2][0]['slots']['version'] = '2025版'
    assert check(wrong)['claims'][0]['status'] == 'insufficient_evidence'


@pytest.mark.parametrize('bad', ['empty_fields', 'empty_slot', 'blank_slot', 'unknown_field', 'unresolved_mapping', 'wrong_role'])
def test_invalid_context_slot_mapping_is_atomic(bad):
    data = scoped_case()
    decision = data[1]['qualification_assessments'][0]
    if bad == 'empty_fields': decision['slot_fields'] = []
    elif bad == 'empty_slot': data[1]['assertions'][0]['slots']['version'] = ''
    elif bad == 'blank_slot': data[1]['assertions'][0]['slots']['version'] = '   '
    elif bad == 'unknown_field': decision['slot_fields'] = ['made_up']
    elif bad == 'unresolved_mapping': decision['status'] = 'unresolved'
    else:
        # No role declaration means the legacy footnote contract; it cannot
        # bypass condition preservation by supplying a version instead.
        data[0].refs['cell']['source_context'].pop('qualification_roles')
        data[1]['assertions'][0]['reference_canonical_sha256'] = digest(data[0].refs['cell'])
    result = check(data)
    assert not result['valid'] and result['claims'] == []


@pytest.mark.parametrize('role', [[], {}, 1, 'unknown'])
def test_malformed_role_returns_invalid_audit_not_uncaught_error(role):
    data = scoped_case()
    data[0].refs['cell']['source_context']['qualification_roles']['#/texts/0'] = role
    data[1]['assertions'][0]['reference_canonical_sha256'] = digest(data[0].refs['cell'])
    result = check(data)
    assert not result['valid'] and result['claims'] == []


def test_previously_reviewed_missing_context_is_not_grandfathered(tmp_path, monkeypatch):
    root, plan, pack, ref = context_bundle(tmp_path)
    # Reconstruct a legacy projection which did not carry section titles, with
    # a matching historical annotation, not an accidental hash-drift test.
    ref['source_context']['qualification_refs'] = []
    ref['source_context'].pop('qualification_roles', None)
    ref['source_locators'] = [loc for loc in ref['source_locators'] if loc['selector'][-1] not in (1, 3)]
    slots = dict(subject='方案甲', property='比例', value='80', unit='%', negated=False,
                 conditions=[], time_scope='', version='2026版')
    binding = review._view(plan, pack)[0]
    annotation = dict(executor_run_id='old-annotator', claims=[dict(claim_id='c', output_location='overview.text',
        output_canonical_sha256=digest(plan['overview']['text']), evidence_ids=[ref['evidence_id']], slots=slots)],
        assertions=[dict(assertion_id='a', evidence_id=ref['evidence_id'], context_id=ref['source_context']['context_id'],
            reference_canonical_sha256=digest(ref), span=dict(start=0, end=len(ref['snippet']),
            canonical_sha256=digest(ref['snippet'])), slots=deepcopy(slots))])
    assessed = dict(executor_run_id='old-reviewer', source_interpretation_correct=True,
        output_interpretation_correct=True, decomposition_complete=True, context_association_correct=True)
    monkeypatch.setattr(review, '_state', lambda _: {'sessions': {binding: {'overview.text': {
        'annotation': annotation, 'review': assessed}}}})
    before = (root / 'courseware/pages/page-0.json').read_bytes()
    with pytest.raises(ValueError, match='qualification_catalog'):
        review.summary_claim_review_report(root, plan, pack)
    assert (root / 'courseware/pages/page-0.json').read_bytes() == before


@pytest.mark.parametrize('kind', ['row_section', 'captions'])
@pytest.mark.parametrize('resolution', ['missing', 'unresolved', 'resolved', 'wrong_version'])
def test_native_context_full_review_and_restore(tmp_path, resolution, kind):
    from video_knowledge_pipeline.transcript_semantic_transaction import apply_transcript_semantic_restore
    root, plan, pack, ref = context_bundle(tmp_path, kind)
    source_before = (root / 'courseware/pages/page-0.json').read_bytes()
    task = review.prepare_summary_claim_review(root)
    slots = dict(subject='方案甲', property='比例', value='80', unit='%', negated=False,
                 conditions=[], time_scope='', version='2026版')
    result = {key: task[key] for key in ('task_id', 'task_pack_sha256')}
    result.update(executor='synthetic', executor_run_id='annotator', model_actual='fixture',
        observed_source_ids=[ref['evidence_id']],
        claims=[dict(claim_id='c', evidence_ids=[ref['evidence_id']], slots=slots)],
        assertions=[dict(assertion_id='a', evidence_id=ref['evidence_id'], start=0,
                         end=len(ref['snippet']), slots=deepcopy(slots))])
    if resolution == 'wrong_version':
        result['claims'][0]['slots'] = {**slots, 'version': '2025版'}
    if resolution != 'missing':
        keys = ref['source_context']['qualification_refs']
        result['qualification_assessments'] = [dict(assertion_id='a', qualifier_id=key,
            status=('applies' if key == keys[0] else 'not_applicable') if resolution in {'resolved', 'wrong_version'} else 'unresolved',
            reason='Synthetic named section and version reviewed.', condition_indexes=[],
            slot_fields=['subject', 'version'] if resolution in {'resolved', 'wrong_version'} and key == keys[0] else [])
            for key in keys]
    write_json(root / 'result.json', result)
    receipt = review.import_summary_claim_review(root, root / 'result.json')
    assert receipt['ok'], receipt
    task = review.prepare_summary_claim_review(root)
    assert task['phase'] == 'review' and len(canonical_json_bytes(task)) <= 8192
    result = {key: task[key] for key in ('task_id', 'task_pack_sha256')}
    result.update(executor='synthetic', executor_run_id='reviewer', model_actual='fixture',
        observed_source_ids=[ref['evidence_id']], source_interpretation_correct=True,
        output_interpretation_correct=True, decomposition_complete=True,
        context_association_correct=True, reason='Synthetic original context reviewed; unknown remains unknown.')
    write_json(root / 'result.json', result)
    receipt = review.import_summary_claim_review(root, root / 'result.json')
    assert receipt['ok'], receipt
    report = review.summary_claim_review_report(root, plan, pack)
    assert report['passed'] is (resolution == 'resolved')
    tx = receipt['transaction']
    assert apply_transcript_semantic_restore(root, tx['restore_plan'], execute=True,
        confirm_transaction_id=tx['transaction_id'])['status'] == 'restored'
    assert review.prepare_summary_claim_review(root) == task
    assert (root / 'courseware/pages/page-0.json').read_bytes() == source_before
