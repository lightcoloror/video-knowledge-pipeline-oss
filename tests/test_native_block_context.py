"""The original OCR block must not bypass its native table's context review."""
import pytest

import test_table_context_admission as context_tests
from test_courseware_material import fixture_bundle, page_result
from test_courseware_structure import projected, native_page
from video_knowledge_pipeline.courseware_material import import_courseware_page
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.evidence_locator import read_evidence_snapshot
from video_knowledge_pipeline import summary_claim_review as review


def block_bundle(tmp_path, kind='row_section'):
    root = fixture_bundle(tmp_path)
    page = (native_page(root) if kind == 'footnotes' else context_tests.native_context_page(root, kind))
    page['blocks'][0]['text'] = '方案甲2026版比例80%。'
    import_courseware_page(root, page)
    _, section = projected(root)  # actual _chapter_fact_section producer
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'] == 'courseware:page-0:0')
    plan = {'overview': {'text': page['blocks'][0]['text'], 'evidence_ids': [ref['evidence_id']]}}
    pack = {'sections': [{'section_id': 's', 'evidence_refs': [ref]}]}
    write_json(root / review.PLAN, plan)
    write_json(root / review.FACTS, pack)
    return root, plan, pack, ref


@pytest.mark.parametrize('kind', ['row_section', 'captions', 'footnotes'])
def test_raw_block_retained_with_same_native_qualification_catalog(tmp_path, kind):
    root, _, _, ref = block_bundle(tmp_path, kind)
    assert '方案甲2026版比例80%' in ref['snippet']
    assert ref['fact_status'] == 'candidate_evidence'
    assert ref['source_context']['scope'] == 'declared_native_table_associations_not_verified_semantics'
    assert ref['source_context']['qualification_refs']
    task = review.prepare_summary_claim_review(root)
    objects = task['sources'][0]['original_objects']
    assert any(o.get('docling_ref') == '#/tables/0' for o in objects)
    assert len(objects) == 1 + len(ref['source_context']['qualification_refs'])
    # No whole table copy, no substitution of a model interpretation for raw OCR.
    assert not any(o.get('label') == 'table' for o in objects)


@pytest.mark.parametrize('kind', ['row_section', 'captions', 'footnotes'])
@pytest.mark.parametrize('consumer', ['prepare', 'report'])
def test_old_block_projection_cannot_strip_context_and_stay_eligible(tmp_path, kind, consumer):
    root, plan, pack, ref = block_bundle(tmp_path, kind)
    ref['source_context'].pop('qualification_refs', None)
    ref['source_context'].pop('qualification_roles', None)
    ref['source_context']['scope'] = 'observed_block_not_verified_table_relationship'
    ref['source_locators'] = ref['source_locators'][:1]
    write_json(root / review.FACTS, pack)
    before = (root / 'courseware/pages/page-0.json').read_bytes()
    with pytest.raises(ValueError, match='qualification_catalog'):
        if consumer == 'prepare':
            review.prepare_summary_claim_review(root)
        else:
            review.summary_claim_review_report(root, plan, pack)
    assert (root / 'courseware/pages/page-0.json').read_bytes() == before


@pytest.mark.parametrize('kind', ['row_section', 'captions'])
@pytest.mark.parametrize('resolution', ['missing', 'unresolved', 'resolved', 'wrong_version'])
def test_block_path_uses_existing_full_review_and_restore(tmp_path, monkeypatch, resolution, kind):
    monkeypatch.setattr(context_tests, 'context_bundle', block_bundle)
    context_tests.test_native_context_full_review_and_restore(tmp_path, resolution, kind)


def test_non_native_ocr_block_keeps_existing_contract(tmp_path):
    root = fixture_bundle(tmp_path)
    page = page_result(root, text='普通 OCR 文本，不含原生表格关联。')
    import_courseware_page(root, page)
    _, section = projected(root)
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'] == 'courseware:page-0:0')
    assert 'qualification_refs' not in ref['source_context']
    plan = {'overview': {'text': '普通 OCR 文本。', 'evidence_ids': [ref['evidence_id']]}}
    pack = {'sections': [{'section_id': 's', 'evidence_refs': [ref]}]}
    write_json(root / review.PLAN, plan)
    write_json(root / review.FACTS, pack)
    assert review.prepare_summary_claim_review(root)['phase'] == 'annotate'


@pytest.mark.parametrize('target', ['block_text', 'cell_text'])
@pytest.mark.parametrize('consumer', ['prepare', 'report'])
def test_exact_nested_text_locator_cannot_strip_native_parent_context(tmp_path, target, consumer):
    root, plan, pack, ref = block_bundle(tmp_path)
    loc = ref['source_locators'][0]
    snapshot = read_evidence_snapshot(root, loc['artifact_path'], loc['schema'])
    selector = (['blocks', 0, 'text'] if target == 'block_text'
                else ['docling_document', 'tables', 0, 'data', 'table_cells', 2, 'text'])
    ref['source_locators'] = [snapshot.locator(selector)]
    ref['source_context'] = {'scope': 'observed_block_not_verified_table_relationship'}
    write_json(root / review.FACTS, pack)
    with pytest.raises(ValueError, match='qualification_catalog'):
        if consumer == 'prepare':
            review.prepare_summary_claim_review(root)
        else:
            review.summary_claim_review_report(root, plan, pack)
