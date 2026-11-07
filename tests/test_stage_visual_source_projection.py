"""Typed local observations count as evidence, never as original-media truth."""
import pytest

from test_summary_stage_coverage import TEXT, OTHER, prompt
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot
from video_knowledge_pipeline.video_moment_index import _moment_observations
from video_knowledge_pipeline.summary_stage_coverage import evaluate_summary_stage_coverage


def case(root, kind, *, metadata_only=False):
    content = OTHER if metadata_only else TEXT
    field = {'visual_understanding': 'objects', 'temporal_visual_understanding': 'event_sequence'}
    value = {'evidence_level': 'confirmed', 'evidence_frame_paths': ['assets/synthetic.png'],
             'model_actual': TEXT, 'run_id': TEXT, 'source': TEXT}
    if kind == 'continuous_tag_weak_evidence':
        value.update(decision='coarse_weak_evidence', stable_tags=['课件'],
                     coarse_summary=content, evidence_level='candidate_weak')
        item = {'temporal_tag_delta': value}
    else:
        value[field[kind]] = [content]
        item = {kind: value}
    write_json(root / 'timeline.json', [item])
    snapshot = JsonEvidenceSnapshot.read(root, 'timeline.json')
    refs = _moment_observations(item, 1, snapshot=snapshot, source_position=0)
    ref = next(r for r in refs if r['observation_kind'] == kind)
    pack = {'sections': [{'section_id': 's', 'evidence_refs': [ref]}]}
    write_json(root / 'quality-benchmark-key-points.json', {'key_points': [
        {'id': 'important', 'text': TEXT, 'evidence_ids': [ref['evidence_id']]}],
        'source_bindings': [{'path': 'timeline.json', 'sha256': sha256_file(root / 'timeline.json')}]})
    return pack


@pytest.mark.parametrize('kind', ['visual_understanding', 'temporal_visual_understanding', 'continuous_tag_weak_evidence'])
@pytest.mark.parametrize('metadata_only', [False, True])
def test_bound_typed_source_is_projected_without_model_or_path_words(tmp_path, kind, metadata_only):
    pack = case(tmp_path, kind, metadata_only=metadata_only)
    report = evaluate_summary_stage_coverage(tmp_path, pack, prompt(TEXT), output_text=TEXT)
    assert report['source_unknown_count'] == 0, report
    assert report['points'][0]['stages']['source_evidence'] is (not metadata_only)
    assert report['passed'] is (not metadata_only)
    assert report['whole_media_completeness'] == 'not_assessed'
    assert report['factual_truth'] == 'not_assessed'
    if kind == 'continuous_tag_weak_evidence':
        assert pack['sections'][0]['evidence_refs'][0]['evidence_level'] == 'candidate_weak'


@pytest.mark.parametrize('mutation', ['wrong_field', 'wrong_field_string', 'superseded', 'unavailable', 'unavailable_text'])
def test_old_or_arbitrary_object_cannot_claim_typed_source(tmp_path, mutation):
    pack = case(tmp_path, 'visual_understanding')
    from video_knowledge_pipeline.storage import read_json
    timeline = read_json(tmp_path / 'timeline.json')
    if mutation.startswith('wrong_field'):
        timeline[0]['untrusted_metadata'] = TEXT if mutation.endswith('string') else timeline[0]['visual_understanding']
    elif mutation == 'superseded':
        timeline[0]['human_corrected_visual_understanding'] = {
            **timeline[0]['visual_understanding'], 'objects': [OTHER]}
    else:
        timeline[0]['visual_understanding']['evidence_level'] = 'unavailable'
        if mutation == 'unavailable_text':
            timeline[0]['visual_understanding']['text'] = TEXT
    write_json(tmp_path / 'timeline.json', timeline)
    snapshot = JsonEvidenceSnapshot.read(tmp_path, 'timeline.json')
    ref = pack['sections'][0]['evidence_refs'][0]
    ref['source_locators'] = [snapshot.locator([0, 'untrusted_metadata' if mutation.startswith('wrong_field') else 'visual_understanding'])]
    reference = read_json(tmp_path / 'quality-benchmark-key-points.json')
    reference['source_bindings'][0]['sha256'] = sha256_file(tmp_path / 'timeline.json')
    write_json(tmp_path / 'quality-benchmark-key-points.json', reference)
    report = evaluate_summary_stage_coverage(tmp_path, pack, prompt(TEXT), output_text=TEXT)
    assert report['passed'] is False
    assert report['source_unknown_count'] >= 1
