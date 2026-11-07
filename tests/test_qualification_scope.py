"""Declared table-note applicability, never inferred by matching numbers."""
from copy import deepcopy

import pytest

from video_knowledge_pipeline.canonical_json import canonical_json_sha256 as digest
from video_knowledge_pipeline.summary_citation_audit import build_citation_index
from video_knowledge_pipeline.structured_claim_verification import audit_structured_claims


def case(status=None, *, conditions=None):
    conditions = [] if conditions is None else conditions
    ref = dict(evidence_id='cell',source_kind='visual',fact_status='candidate_evidence',
               snippet='方案甲80%。脚注：仅限首次参与。', source_context={
                   'scope':'declared_native_table_associations_not_verified_semantics',
                   'qualification_refs':['#/texts/0'], 'footnote_applicability':'not_assessed'})
    index = build_citation_index({'sections':[{'evidence_refs':[ref]}]})
    slots = dict(subject='方案甲',property='比例',value='80',unit='%',negated=False,
                 conditions=conditions,time_scope='',version='')
    assertion = dict(assertion_id='a',evidence_id='cell',context_id='cell',slots=slots,
                     reference_canonical_sha256=digest(ref),
                     span=dict(start=0,end=len(ref['snippet']),canonical_sha256=digest(ref['snippet'])))
    scope = dict(schema='video_knowledge_pipeline.claim_verification_scope.v1',
                 reference_snapshot_sha256=index.snapshot_sha256,required_locations=['overview.text'],
                 assertions=[assertion])
    if status:
        scope['qualification_assessments'] = [dict(assertion_id='a',qualifier_id='#/texts/0',
            status=status,reason='Synthetic applicability explanation checked against the note.',
            condition_indexes=[0] if status == 'applies' else [])]
    text = '方案甲比例80%。' + '；'.join(conditions)
    claims = [dict(claim_id='c',output_location='overview.text',output_canonical_sha256=digest(text),
                   evidence_ids=['cell'],slots=deepcopy(slots))]
    return index, scope, claims, {'overview.text':text}


def check(data):
    index,scope,claims,texts = data
    return audit_structured_claims(claims,scope=scope,citation_index=index,
        output_texts=texts,output_citations={'overview.text':['cell']})


@pytest.mark.parametrize('status',[None,'unresolved'])
def test_unknown_qualification_cannot_support_unconditional_value(status):
    result = check(case(status))
    assert result['valid'], result['errors']
    assert result['claims'][0]['status'] == 'insufficient_evidence'
    assert result['qualification_audit']['unresolved']


@pytest.mark.parametrize('status',['applies','not_applicable'])
def test_explicit_scoped_resolution_allows_existing_comparator(status):
    result = check(case(status,conditions=['首次参与'] if status == 'applies' else []))
    assert result['valid'],result['errors']
    assert result['claims'][0]['status'] == 'supported'
    assert result['source_interpretation_accuracy'] == 'not_assessed'


@pytest.mark.parametrize('bad',['extra','duplicate','missing_reason','bad_condition','empty_applies'])
def test_bad_qualification_resolution_is_atomic_protocol_failure(bad):
    data = case('applies',conditions=['首次参与'])
    rows = data[1]['qualification_assessments']
    if bad == 'extra': rows[0]['qualifier_id'] = '#/texts/999'
    elif bad == 'duplicate': rows.append(deepcopy(rows[0]))
    elif bad == 'missing_reason': rows[0]['reason'] = ''
    elif bad == 'bad_condition': rows[0]['condition_indexes'] = [4]
    else: rows[0]['condition_indexes'] = []
    result = check(data)
    assert not result['valid']
    assert result['claims'] == []


def test_applies_condition_cannot_be_omitted_from_output_claim():
    data = case('applies',conditions=['首次参与'])
    data[2][0]['slots']['conditions'] = []
    assert check(data)['claims'][0]['status'] == 'insufficient_evidence'


def test_native_projection_supplies_exact_qualifier_refs(tmp_path):
    from test_courseware_structure import native_page,projected
    from test_courseware_material import fixture_bundle
    from video_knowledge_pipeline.courseware_material import import_courseware_page
    root = fixture_bundle(tmp_path)
    import_courseware_page(root,native_page(root))
    _,section = projected(root)
    cell = next(ref for ref in section['evidence_refs'] if ref['evidence_id'].endswith(':cell:4'))
    assert cell['source_context']['qualification_refs'] == ['#/texts/0']


@pytest.mark.parametrize('catalog',[[], ['#/texts/999'], None, 'locator_removed'])
def test_review_catalog_cannot_omit_or_replace_original_footnote(tmp_path,catalog):
    from test_courseware_structure import native_page,projected
    from test_courseware_material import fixture_bundle
    from video_knowledge_pipeline.courseware_material import import_courseware_page
    from video_knowledge_pipeline.summary_claim_review import _scope_sources
    root = fixture_bundle(tmp_path)
    import_courseware_page(root,native_page(root))
    _,section = projected(root)
    cited = 'courseware:page-0:0:cell:4'
    cell = next(ref for ref in section['evidence_refs'] if ref['evidence_id'] == cited)
    if catalog == 'locator_removed':
        cell['source_context']['qualification_refs'] = []
        cell['source_locators'] = [loc for loc in cell['source_locators']
                                  if loc['selector'][:2] != ['docling_document', 'texts']]
    elif catalog is None:
        cell['source_context'].pop('qualification_refs')
    else:
        cell['source_context']['qualification_refs'] = catalog
    pack = {'sections':[section]}
    index = build_citation_index(pack)
    with pytest.raises(ValueError,match='qualification_catalog'):
        _scope_sources(root,index,pack,[cited])


def test_unresolved_note_is_not_proof_of_opposite_value():
    data = case('unresolved')
    data[2][0]['slots']['value'] = '90'
    assert check(data)['claims'][0]['status'] == 'insufficient_evidence'
