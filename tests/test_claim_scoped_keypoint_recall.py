"""Synthetic ablations of lexical recall versus claim-scoped support."""
import pytest

from video_knowledge_pipeline.smart_summary_keypoint_eval import evaluate_human_key_point_recall


@pytest.mark.parametrize('reference,summary', [
    ('家庭预算上限为2万元', '家庭预算上限为5万元。旅行费用为2万元。'),
    ('允许线上投保', '本产品不允许线上投保。'),
    ('孩子不带身故责任', '孩子带身故责任。'),
    ('只有完成审核才能承保', '完成审核即可承保。'),
    ('本方案仅限成年人购买', '本方案所有人都可以购买。'),
])
def test_high_similarity_does_not_wash_out_claim_constraints(reference, summary):
    result = evaluate_human_key_point_recall({'key_points': [reference]}, summary)
    assert result['decisions'][0]['matched'] is False
    assert result['recall'] == 0


def test_subject_and_version_scope_must_match_the_same_fragment():
    payload = {'key_points': [{
        'text': '甲产品2026版等待期为30日',
        'claim_scope': {'subject': '甲产品', 'version': '2026版', 'property': '等待期'},
    }]}
    result = evaluate_human_key_point_recall(
        payload, '乙产品2026版等待期为30日。甲产品2025版等待期为30日。')
    assert result['recall'] == 0
    assert result['decisions'][0]['claim_scope'] == payload['key_points'][0]['claim_scope']


def test_explicit_alias_is_checked_in_its_own_scope():
    payload = {'key_points': [{'text': '只有审核通过才能承保', 'aliases': ['承保以通过审核为前提']}]}
    result = evaluate_human_key_point_recall(payload, '承保以通过审核为前提。')
    assert result['recall'] == 1
    assert result['decisions'][0]['method'] == 'explicit_alias'
    assert result['assessment_scope'] == 'reference_recall_not_factual_truth'


def test_missing_goldset_is_not_a_completeness_pass():
    result = evaluate_human_key_point_recall({}, '完整总结')
    assert result['evaluated'] is False
    assert result['recall'] is None


@pytest.mark.parametrize('scope', [{'subject': 42}, {'unknown_key': 'anything'}, {'subject': ''}])
def test_malformed_claim_scope_is_not_silently_ignored(scope):
    result = evaluate_human_key_point_recall({'key_points': [{'text': '甲产品30日', 'claim_scope': scope}]}, '甲产品30日')
    assert result['evaluated'] is False
    assert result['recall'] is None


def test_correct_fragment_is_not_hidden_by_a_similar_wrong_fragment():
    result = evaluate_human_key_point_recall(
        {'key_points': ['家庭预算上限为2万元']},
        '家庭预算上限为5万元。家庭预算上限为2万元。')
    assert result['recall'] == 1
    assert '2万元' in result['decisions'][0]['matched_excerpt']


def test_required_number_is_not_satisfied_by_a_different_unit_elsewhere():
    result = evaluate_human_key_point_recall(
        {'key_points': ['等待时间为2小时']}, '等待时间为2分钟。培训时长为2小时。')
    assert result['recall'] == 0


@pytest.mark.parametrize('kind', ['outside', 'media', 'duplicate', 'missing', 'invalid_hash'])
def test_reference_binding_fail_closed_without_reading_outside_text_evidence(tmp_path, monkeypatch, kind):
    import video_knowledge_pipeline.smart_summary_keypoint_eval as module

    root = tmp_path / 'bundle'
    root.mkdir()
    text = root / 'timeline.json'
    text.write_text('[]', encoding='utf-8')
    outside = tmp_path / 'external.json'
    outside.write_text('private fixture', encoding='utf-8')
    media = root / 'frame.png'
    media.write_bytes(b'synthetic bytes')
    hashes_read = []
    def fake_hash(path):
        assert path == text
        hashes_read.append(path)
        return 'a'*64
    monkeypatch.setattr(module, 'sha256_file', fake_hash)
    row = {'path': str(text), 'sha256': 'a'*64}
    bindings = [row]
    if kind == 'outside':
        row['path'] = str(outside)
    elif kind == 'media':
        row['path'] = str(media)
    elif kind == 'missing':
        row['path'] = 'missing.json'
    elif kind == 'invalid_hash':
        row['sha256'] = 'bad'
    else:
        bindings.append(dict(row))
    result = module.evaluate_human_key_point_recall(
        {'key_points': ['test'], 'source_bindings': bindings}, 'test', bundle_root=root)
    assert not result['evaluated']
    assert result['recall'] is None
    assert result['reference_freshness']['status'] == 'invalid'
    assert len(hashes_read) <= 1
