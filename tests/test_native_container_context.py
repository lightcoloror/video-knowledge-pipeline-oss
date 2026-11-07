"""Every supported native table selector retains the same qualification gate."""
import pytest
from copy import deepcopy

from test_native_block_context import block_bundle
from video_knowledge_pipeline.evidence_locator import read_evidence_snapshot
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline import summary_claim_review as review
from video_knowledge_pipeline.canonical_json import canonical_json_bytes
from video_knowledge_pipeline.courseware_structure import native_qualification_catalog
from video_knowledge_pipeline.summary_citation_audit import build_citation_index


SELECTORS = [
    ['docling_document', 'tables', 0],
    ['docling_document', 'tables', 0, 'data'],
    ['docling_document', 'tables', 0, 'data', 'table_cells'],
    ['docling_document', 'tables'],
    ['docling_document'],
    ['blocks'],
]


def container_bundle(tmp_path, selector, *, strip=False, kind='row_section'):
    root, plan, pack, ref = block_bundle(tmp_path, kind)
    loc = ref['source_locators'][0]
    snap = read_evidence_snapshot(root, loc['artifact_path'], loc['schema'])
    ref['source_locators'] = [snap.locator(selector), *ref['source_locators'][1:]]
    if strip:
        ref['source_locators'] = ref['source_locators'][:1]
        ref['source_context'] = {'scope': 'observed_block_not_verified_table_relationship'}
    write_json(root / review.FACTS, pack)
    return root, plan, pack, ref


@pytest.mark.parametrize('selector', SELECTORS)
@pytest.mark.parametrize('consumer', ['prepare', 'report'])
def test_native_container_cannot_launder_removed_context(tmp_path, selector, consumer):
    root, plan, pack, _ = container_bundle(tmp_path, selector, strip=True)
    before = (root / 'courseware/pages/page-0.json').read_bytes()
    with pytest.raises(ValueError, match='qualification_catalog'):
        if consumer == 'prepare':
            review.prepare_summary_claim_review(root)
        else:
            review.summary_claim_review_report(root, plan, pack)
    assert (root / 'courseware/pages/page-0.json').read_bytes() == before


@pytest.mark.parametrize('selector', SELECTORS)
def test_bounded_container_with_original_qualifiers_is_not_discarded(tmp_path, selector):
    root, plan, pack, ref = container_bundle(tmp_path, selector)
    task = review.prepare_summary_claim_review(root)
    assert len(canonical_json_bytes(task)) <= 8192
    assert task['sources'][0]['source_context']['qualification_refs'] == ref['source_context']['qualification_refs']
    report = review.summary_claim_review_report(root, plan, pack)
    assert report['status'] == 'review_required'  # Preparation is not approval.


@pytest.mark.parametrize('selector', [
    ['docling_document', 'texts'], ['docling_document', 'texts', 0],
    ['docling_document', 'texts', 0, 'text']])
@pytest.mark.parametrize('consumer', ['prepare', 'report'])
def test_native_related_text_cannot_drop_declared_parent_tables(tmp_path, selector, consumer):
    root, plan, pack, _ = container_bundle(tmp_path, selector, strip=True, kind='captions')
    with pytest.raises(ValueError, match='qualification_catalog'):
        if consumer == 'prepare':
            review.prepare_summary_claim_review(root)
        else:
            review.summary_claim_review_report(root, plan, pack)


@pytest.mark.parametrize('consumer', ['prepare', 'report'])
def test_colliding_native_qualifier_ids_from_distinct_files_require_split(tmp_path, consumer):
    root, plan, pack, ref = block_bundle(tmp_path)
    loc = ref['source_locators'][0]
    snap = read_evidence_snapshot(root, loc['artifact_path'], loc['schema'])
    other = 'courseware/pages/other-synthetic.json'
    write_json(root / other, snap.data)
    second = read_evidence_snapshot(root, other, loc['schema'])
    ref['source_locators'].extend(second.locator(loc['selector']) for loc in list(ref['source_locators']))
    write_json(root / review.FACTS, pack)
    with pytest.raises(ValueError, match='qualification_catalog_ambiguous'):
        if consumer == 'prepare':
            review.prepare_summary_claim_review(root)
        else:
            review.summary_claim_review_report(root, plan, pack)


@pytest.mark.parametrize('kind', ['captions', 'footnotes'])
@pytest.mark.parametrize('selector', [
    ['docling_document', 'texts'], ['docling_document', 'texts', 0],
    ['docling_document', 'texts', 0, 'text']])
def test_related_text_with_complete_context_remains_reviewable(tmp_path, kind, selector):
    root, _, _, ref = container_bundle(tmp_path, selector, kind=kind)
    task = review.prepare_summary_claim_review(root)
    assert len(canonical_json_bytes(task)) <= 8192
    assert task['sources'][0]['source_context']['qualification_refs'] == ref['source_context']['qualification_refs']


def test_splitting_cross_file_sources_restores_unambiguous_preparation(tmp_path):
    root, plan, pack, ref = block_bundle(tmp_path)
    loc = ref['source_locators'][0]
    snap = read_evidence_snapshot(root, loc['artifact_path'], loc['schema'])
    other = 'courseware/pages/other-synthetic.json'
    write_json(root / other, snap.data)
    second = read_evidence_snapshot(root, other, loc['schema'])
    other_ref = deepcopy(ref)
    other_ref.update(evidence_id='other-native', source_locators=[second.locator(v['selector']) for v in ref['source_locators']])
    other_ref['source_context']['context_id'] = 'other-native'
    pack['sections'][0]['evidence_refs'].append(other_ref)
    write_json(root / review.FACTS, pack)
    sources = review._scope_sources(root, build_citation_index(pack), pack, [ref['evidence_id']])
    assert len(sources) == 2
    assert {v['source_locators'][0]['artifact_path'] for v in sources} == {loc['artifact_path'], other}
    assert len(canonical_json_bytes(review.prepare_summary_claim_review(root))) <= 8192


@pytest.mark.parametrize('drop_other_parent', [False, True])
def test_shared_caption_keeps_all_declared_parent_table_context(tmp_path, drop_other_parent):
    root, plan, pack, ref = block_bundle(tmp_path)
    loc = ref['source_locators'][0]
    snap = read_evidence_snapshot(root, loc['artifact_path'], loc['schema'])
    page = deepcopy(snap.data)
    doc = page['docling_document']
    doc['texts'] = [dict(self_ref='#/texts/0', label='caption', text='共享表题', prov=[])]
    doc['tables'][0]['captions'] = [{'$ref': '#/texts/0'}]
    second = deepcopy(doc['tables'][0])
    second['self_ref'] = '#/tables/1'
    second['data']['table_cells'][1]['text'] = '方案丙2027版'
    doc['tables'].append(second)
    page['blocks'].append({**page['blocks'][0], 'docling_ref': '#/tables/1'})
    write_json(root / loc['artifact_path'], page)
    snap = read_evidence_snapshot(root, loc['artifact_path'], loc['schema'])
    catalog = {**native_qualification_catalog(doc, 0), **native_qualification_catalog(doc, 1)}
    if drop_other_parent:
        catalog = native_qualification_catalog(doc, 0)
    ref['source_context'].update(qualification_refs=sorted(catalog),
        qualification_roles={key: row['role'] for key, row in catalog.items()})
    ref['source_locators'] = [snap.locator(['docling_document', 'texts', 0, 'text'])] + [
        snap.locator(row['selector']) for row in catalog.values()]
    write_json(root / review.FACTS, pack)
    if drop_other_parent:
        with pytest.raises(ValueError, match='qualification_catalog'):
            review.prepare_summary_claim_review(root)
    else:
        task = review.prepare_summary_claim_review(root)
        assert len(canonical_json_bytes(task)) <= 8192
        assert '#/tables/1/data/table_cells/1' in task['sources'][0]['source_context']['qualification_refs']
        assert '方案丙2027版' in str(task['sources'][0]['original_objects'])
