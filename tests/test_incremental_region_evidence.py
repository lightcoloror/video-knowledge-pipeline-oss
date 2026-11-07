"""Synthetic recognition/normalization conservation, no actual OCR model."""
from types import SimpleNamespace

import pytest

from test_courseware_dedupe import image_bundle
from test_courseware_incremental_ocr import Engine, picture, normalize, run
from video_knowledge_pipeline.courseware_incremental_ocr import run_incremental_courseware_ocr
from video_knowledge_pipeline.storage import read_json


@pytest.mark.parametrize('mutation', ['drop', 'replace', 'duplicate'])
def test_normalizer_cannot_lose_or_invent_recognized_courseware(tmp_path, mutation):
    root = image_bundle(tmp_path, [picture()])
    before = (root / 'timeline.json').read_bytes()
    def changed(raw):
        blocks = normalize(raw)
        if mutation == 'drop':
            return blocks[:1]
        if mutation == 'replace':
            return [dict(blocks[0], text='invented condition'), blocks[1]]
        return [*blocks, blocks[0]]
    with pytest.raises(ValueError, match='region.*projection'):
        run_incremental_courseware_ocr(root, indexes=[1], engine=Engine(),
            runtime_identity={'model_actual': 'synthetic-v1'}, output_factory=SimpleNamespace,
            normalize_blocks=changed)
    assert not (root / 'courseware/pages/page-0.json').exists()
    assert (root / 'timeline.json').read_bytes() == before
    failures = list((root / 'courseware/incremental-raw').glob('*.validation.json'))
    assert len(failures) == 1
    report = read_json(failures[0])
    assert report['changed'] is False and report['ok'] is False
    raw = read_json(root / report['raw_path'])
    assert len(raw['region_events']) == 2
    assert raw['final_recognition']['txts']


def test_engine_final_output_cannot_invent_recognition(tmp_path):
    root = image_bundle(tmp_path, [picture()])
    class WrongFinal(Engine):
        def __call__(self, path):
            raw = super().__call__(path)
            raw.txts = ('invented final text', *raw.txts[1:])
            return raw
    with pytest.raises(ValueError, match='region.*projection'):
        run(root, WrongFinal(), [1])
    assert not (root / 'courseware/pages/page-0.json').exists()


def test_identical_crops_have_one_actual_recognition_and_explicit_reuse(tmp_path):
    root = image_bundle(tmp_path, [picture(10, 10), picture(10, 10)])
    engine = Engine()
    receipt = run(root, engine, [1, 2])
    assert receipt['recognized_regions'] == 1 and engine.regions == 1
    page = read_json(root / 'courseware/pages/page-0.json')
    raw = read_json(root / page['incremental_ocr']['raw_path'])
    events = raw['region_events']
    assert [e['recognized_now'] for e in events] == [True, False]
    assert events[0]['recognition_id'] == events[1]['recognition_id']
    assert page['region_evidence']['normalized_block_count'] == 2
    second = read_json(root / 'courseware/pages/page-1.json')
    second_raw = read_json(root / second['incremental_ocr']['raw_path'])
    assert not any(e['recognized_now'] for e in second_raw['region_events'])
    assert {e['recognition_id'] for e in second_raw['region_events']} == {events[0]['recognition_id']}


def test_filtered_recognized_region_is_explicit_gap(tmp_path):
    root = image_bundle(tmp_path, [picture()])
    class Filtered(Engine):
        def __call__(self, path):
            raw = super().__call__(path)
            raw.txts, raw.scores = raw.txts[:1], raw.scores[:1]
            return raw
    run(root, Filtered(), [1])
    page = read_json(root / 'courseware/pages/page-0.json')
    assert page['region_evidence']['unrepresented_region_count'] == 1
    assert any('recognition output' in gap for gap in page['gaps'])


def test_failed_normalization_resumes_cached_recognition_without_replaying_model(tmp_path):
    root = image_bundle(tmp_path, [picture()])
    engine = Engine()
    args = dict(indexes=[1], engine=engine, runtime_identity={'model_actual':'synthetic-v1'},
                output_factory=SimpleNamespace)
    with pytest.raises(ValueError):
        run_incremental_courseware_ocr(root, normalize_blocks=lambda raw: normalize(raw)[:1], **args)
    assert engine.regions == 2
    result = run_incremental_courseware_ocr(root, normalize_blocks=normalize, resume=True, **args)
    assert engine.regions == 2 and result['recognized_regions'] == 0
    formal = (root / 'courseware/pages/page-0.json').read_bytes()
    run_incremental_courseware_ocr(root, normalize_blocks=normalize, resume=True, **args)
    assert (root / 'courseware/pages/page-0.json').read_bytes() == formal
    assert engine.regions == 2


def test_raw_resealed_outer_text_cannot_disagree_with_recognition(tmp_path):
    from video_knowledge_pipeline.courseware_material import import_courseware_page
    from video_knowledge_pipeline.file_hash import sha256_file
    from video_knowledge_pipeline.storage import write_json
    root = image_bundle(tmp_path, [picture()])
    run(root, Engine(), [1])
    path = root / 'courseware/pages/page-0.json'
    before = path.read_bytes()
    page = read_json(path)
    raw_path = root / page['incremental_ocr']['raw_path']
    raw = read_json(raw_path)
    page['blocks'][0]['text'] = raw['result']['blocks'][0]['text'] = 'invented exception'
    write_json(raw_path, raw)
    page['incremental_ocr']['raw_sha256'] = sha256_file(raw_path)
    with pytest.raises(ValueError, match='region.*projection'):
        import_courseware_page(root, page)
    assert path.read_bytes() == before


def test_reuse_sources_reach_material_and_summary_input(tmp_path):
    from video_knowledge_pipeline.courseware_material import build_courseware_material, render_courseware_model_input
    from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
    root = image_bundle(tmp_path, [picture(), picture(b=21)])
    run(root, Engine(), [1,2])
    material = build_courseware_material(root, write=False)
    first, second = material['pages'][0]['blocks'][0], material['pages'][1]['blocks'][0]
    assert first['incremental_region_context']['recognition_ids'] == second['incremental_region_context']['recognition_ids']
    assert second['incremental_region_context']['origins'] == ['cache_reuse']
    assert second['recognition_provenance']['independence_is_verified'] is False
    assert len(second['recognition_provenance']['frame_sha256s']) == 2
    assert any('/incremental-raw/' in loc['artifact_path'] for loc in second['source_locators'])
    assert 'cache_reuse' in render_courseware_model_input(material)
    pack = build_smart_summary_input_pack(root, write=False)
    assert pack['courseware_material']['pages'][1]['blocks'][0]['incremental_region_context'] == second['incremental_region_context']
    from video_knowledge_pipeline.citation_digest import courseware_records
    from video_knowledge_pipeline.smart_summary_global_reduce import _chapter_fact_section
    from video_knowledge_pipeline.summary_citation_audit import build_citation_index
    from video_knowledge_pipeline.summary_claim_review import _scope_sources
    section = _chapter_fact_section({}, {'evidence': {'courseware_material': material}})
    facts = {'sections':[section]}
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'] == second['evidence_id'])
    assert ref['recognition_provenance'] == second['recognition_provenance']
    digest = courseware_records(material, [])
    assert next(r for r in digest if r['evidence_id'] == second['evidence_id'])['recognition_provenance'] == ref['recognition_provenance']
    sources = _scope_sources(root, build_citation_index(facts), facts, [second['evidence_id']])
    selected = next(s for s in sources if s['evidence_id'] == second['evidence_id'])
    assert selected['recognition_provenance'] == ref['recognition_provenance']
    assert any(o.get('origin') == 'cache_reuse' for o in selected['original_objects'])


def test_multi_artifact_task_factoring_is_lossless_and_idempotent():
    from video_knowledge_pipeline.summary_claim_review import _task_pack
    locators = [dict(schema='locator',artifact_path=path,artifact_sha256=digest,
                     selector=['text',i],selected_sha256='c'*64)
                for i,(path,digest) in enumerate([('one.json','a'*64),('one.json','a'*64),('two.json','b'*64)])]
    source = dict(evidence_id='r',source_context={'frame_region':{'pixel_observation':'not_assessed'}},
                  source_locators=locators,original_objects=['原始条件不得省略'])
    pack = _task_pack('bound','overview.text','内容',[source],['r'],{}, {})
    compact = pack['sources'][0]
    assert len(compact['locator_artifacts']) == 2
    restored = [{**{k:v for k,v in loc.items() if k != 'artifact_index'},
                 **compact['locator_artifacts'][loc['artifact_index']]}
                for loc in compact['source_locators']]
    assert restored == locators
    assert compact['original_objects'] == source['original_objects']
    again = _task_pack('bound','overview.text','内容',pack['sources'],['r'],{}, {})
    assert again == pack


def test_factored_schema_keeps_original_strict_slot_contract():
    import copy
    from jsonschema import Draft202012Validator
    from video_knowledge_pipeline.summary_claim_review import _result_schema
    from video_knowledge_pipeline.structured_claim_verification import SLOTS_SCHEMA, PARTIAL_SLOTS_SCHEMA
    from test_summary_claim_review import answer
    modern = _result_schema('annotate')
    legacy = copy.deepcopy(modern)
    legacy['$defs'].update(slots=SLOTS_SCHEMA, source_slots=PARTIAL_SLOTS_SCHEMA)
    base = answer({'phase':'annotate','task_id':'id','task_pack_sha256':'a'*64,
                   'sources':[{'snippet':'产品甲2026版首次投保等待期90天。'}]})
    variants = [base]
    for group in ('claims','assertions'):
        for field in SLOTS_SCHEMA['properties']:
            absent = copy.deepcopy(base)
            absent[group][0]['slots'].pop(field)
            variants.append(absent)
            wrong = copy.deepcopy(base)
            wrong[group][0]['slots'][field] = {'invalid':'type'}
            variants.append(wrong)
        extra = copy.deepcopy(base)
        extra[group][0]['slots']['extra'] = 'not permitted'
        variants.append(extra)
    validators = Draft202012Validator(modern), Draft202012Validator(legacy)
    assert validators[0].is_valid(base)
    for value in variants:
        assert validators[0].is_valid(value) == validators[1].is_valid(value)
