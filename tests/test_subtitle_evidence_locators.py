from __future__ import annotations

import copy
import json

import pytest

from video_knowledge_pipeline.storage import write_json


def subtitle():
    return ('WEBVTT\r\n\r\nrepeat\r\n00:00.000 --> 00:02.000 align:start\r\n'
            '<c.yellow>相同文字&nbsp;🙂</c>\r\n第二行\r\n\r\n'
            'repeat\r\n00:10.000 --> 00:12.000\r\n'
            '<c.yellow>相同文字&nbsp;🙂</c>\r\n第二行\r\n')


def parsed(root):
    from video_knowledge_pipeline.evidence_locator import TextEvidenceSnapshot
    from video_knowledge_pipeline.transcript import parse_transcript
    path = root / 'source.vtt'
    path.write_bytes(b'\xef\xbb\xbf' + subtitle().encode('utf-8'))
    snap = TextEvidenceSnapshot.read(root, 'source.vtt')
    return snap, parse_transcript(path, source_snapshot=snap)


def test_vtt_spans_retain_original_crlf_tags_unicode_and_repeated_cues(tmp_path):
    from video_knowledge_pipeline.evidence_locator import verify_evidence_locators
    from video_knowledge_pipeline.transcript import parse_transcript
    snap, cues = parsed(tmp_path)
    legacy = parse_transcript(tmp_path / 'source.vtt')
    assert [(c.start, c.end, c.text, c.segment_id) for c in cues] == [(c.start, c.end, c.text, c.segment_id) for c in legacy]
    locators = [c.metadata['source_locators'][0] for c in cues]
    assert locators[0] != locators[1] and len(cues) == 2
    originals = [snap.selected(loc['selector']) for loc in locators]
    assert '00:00.000 -->' in originals[0] and '00:10.000 -->' in originals[1]
    assert all('<c.yellow>' in text and '\r\n第二行' in text for text in originals)
    assert cues[0].metadata['source_observation_text'] == '相同文字 🙂 第二行'
    assert verify_evidence_locators(tmp_path, locators)['status'] == 'verified'


@pytest.mark.parametrize('mutation', ['file', 'range', 'hash', 'outside', 'schema', 'boolean'])
def test_text_locator_drift_and_invalid_ranges_fail_closed(tmp_path, mutation):
    from video_knowledge_pipeline.evidence_locator import verify_evidence_locators
    snap, cues = parsed(tmp_path)
    loc = copy.deepcopy(cues[0].metadata['source_locators'][0])
    if mutation == 'file': (tmp_path / 'source.vtt').write_bytes(b'changed')
    elif mutation == 'range': loc['selector'] = [0, len(snap.data) + 1]
    elif mutation == 'hash': loc['selected_sha256'] = '0' * 64
    elif mutation == 'outside': loc['artifact_path'] = '../source.vtt'
    elif mutation == 'schema': loc['schema'] = 'video_knowledge_pipeline.evidence_locator.v1'
    else: loc['selector'] = [False, 1]
    with pytest.raises((ValueError, OSError)):
        verify_evidence_locators(tmp_path, [loc])


def test_text_and_json_locators_share_existing_audit(tmp_path):
    from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot, audit_fact_pack_locators
    _, cues = parsed(tmp_path)
    write_json(tmp_path / 'timeline.json', [{'text': '合成文字'}])
    locs = [JsonEvidenceSnapshot.read(tmp_path, 'timeline.json').locator([0, 'text']),
            cues[0].metadata['source_locators'][0]]
    report = audit_fact_pack_locators(tmp_path, {'sections': [{'evidence_refs': [{'evidence_id': 'mixed', 'source_locators': locs}]}]})
    assert report['status'] == 'verified' and report['artifact_count'] == 2


def test_public_subtitle_pack_and_digest_have_exact_text_locators(tmp_path):
    from video_knowledge_pipeline.smart_summary_chapters import build_smart_summary_chapter_pack
    from video_knowledge_pipeline.citation_digest import audit_citation_digest
    parsed(tmp_path)
    write_json(tmp_path / 'manifest.json', {'title': 'synthetic', 'source_transcript': 'source.vtt'})
    write_json(tmp_path / 'timeline.json', [{'index': 1, 'start': 0, 'end': 12, 'transcript': '已有时间线'}])
    original = (tmp_path / 'source.vtt').read_bytes()
    pack = build_smart_summary_chapter_pack(tmp_path, target_chapters=1, chapter_mode='fixed', write=True)
    rows = [r for c in pack['chapters'] for r in c['citation_digest'] if r['source_type'] == 'transcript']
    assert len(rows) == 2 and all(r['source_locators'] for r in rows)
    assert audit_citation_digest(tmp_path)['passed']
    assert (tmp_path / 'source.vtt').read_bytes() == original
    (tmp_path / 'source.vtt').write_bytes(original + b'\n')
    assert audit_citation_digest(tmp_path)['status'] == 'blocked'


def test_existing_claim_review_and_stage_coverage_read_raw_subtitle_block(tmp_path):
    from test_summary_claim_review import fixture
    from video_knowledge_pipeline.summary_claim_review import prepare_summary_claim_review
    from video_knowledge_pipeline.summary_citation_audit import build_citation_index
    from video_knowledge_pipeline.summary_stage_coverage import _source_units
    snap, cues = parsed(tmp_path)
    plan, facts = fixture(tmp_path)
    loc = cues[0].metadata['source_locators'][0]
    ref = facts['sections'][0]['evidence_refs'][0]
    ref['source_locators'] = [loc]
    ref['snippet'] = cues[0].text
    plan['overview']['text'] = cues[0].text
    write_json(tmp_path / 'exports/smart-summary-reader-plan.json', plan)
    write_json(tmp_path / 'exports/smart-summary-chapter-fact-pack.json', facts)
    task = prepare_summary_claim_review(tmp_path)
    assert task['sources'][0]['original_objects'] == [snap.selected(loc['selector'])]
    values, unknown = _source_units(tmp_path, build_citation_index(facts), ['r'])
    assert values == [snap.selected(loc['selector'])] and unknown == 0


def test_text_snapshot_budget_and_format_restrictions(tmp_path):
    from video_knowledge_pipeline.evidence_locator import TextEvidenceSnapshot
    (tmp_path / 'source.srt').write_text('合成文本', encoding='utf-8')
    with pytest.raises(ValueError, match='budget'):
        TextEvidenceSnapshot.read(tmp_path, 'source.srt', max_bytes=1)
    (tmp_path / 'video.mp4').write_bytes(b'synthetic not video')
    with pytest.raises(ValueError):
        TextEvidenceSnapshot.read(tmp_path, 'video.mp4')


@pytest.mark.parametrize('newline', ['\n', '\r', '\r\n'])
def test_srt_ranges_and_shared_locator_dedup_keep_existing_semantics(tmp_path, newline):
    from video_knowledge_pipeline.evidence_locator import TextEvidenceSnapshot
    from video_knowledge_pipeline.transcript import parse_transcript
    from video_knowledge_pipeline.transcript_projection import _is_proven_duplicate
    content = newline.join(['1', '00:00:00,000 --> 00:00:02,000', '说话人1：同一段话', '',
                            '1', '00:00:10,000 --> 00:00:12,000', '说话人1：同一段话', ''])
    path = tmp_path / 'source.srt'
    path.write_bytes(content.encode('utf-8'))
    snap = TextEvidenceSnapshot.read(tmp_path, 'source.srt')
    cues = parse_transcript(path, source_snapshot=snap)
    assert len(cues) == 2
    assert not _is_proven_duplicate(cues[0], cues[1])
    copied = copy.deepcopy(cues[0])
    copied.start, copied.end = 20, 22
    assert _is_proven_duplicate(cues[0], copied)
    assert '00:00:10,000' in snap.selected(cues[1].metadata['source_locators'][0]['selector'])


def test_optional_fixed_bilinote_source_method_matches_standard_srt(tmp_path):
    import ast
    import logging
    import os
    from pathlib import Path
    from types import SimpleNamespace
    from typing import Optional
    from video_knowledge_pipeline.transcript import parse_transcript
    from video_knowledge_pipeline.evidence_locator import TextEvidenceSnapshot
    source_root = os.environ.get('VKP_BILINOTE_SOURCE')
    if not source_root:
        pytest.skip('fixed local BiliNote source not supplied')
    source = Path(source_root) / 'backend/app/downloaders/bilibili_downloader.py'
    nodes = [n for n in ast.walk(ast.parse(source.read_text(encoding='utf-8'))) if isinstance(n, ast.FunctionDef) and n.name == '_parse_srt_content']
    assert len(nodes) == 1
    ns = {'Optional': Optional, 'TranscriptResult': SimpleNamespace, 'TranscriptSegment': SimpleNamespace,
          'logger': logging.getLogger('offline-bilinote-source')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    text = '1\n00:00:00,000 --> 00:00:02,000\n合成原话\n\n2\n00:00:03,000 --> 00:00:04,000\n限定条件。\n'
    upstream = ns['_parse_srt_content'](None, text, 'zh')
    (tmp_path / 'source.srt').write_bytes(text.encode())
    local = parse_transcript(tmp_path / 'source.srt', source_snapshot=TextEvidenceSnapshot.read(tmp_path, 'source.srt'))
    assert [(c.start, c.end, c.text) for c in local] == [(c.start, c.end, c.text) for c in upstream.segments]


def test_subtitle_review_import_and_restore_reuse_existing_transaction(tmp_path):
    from test_summary_claim_review import fixture, submit
    from video_knowledge_pipeline.evidence_locator import TextEvidenceSnapshot
    from video_knowledge_pipeline.transcript import parse_transcript
    from video_knowledge_pipeline.summary_claim_review import prepare_summary_claim_review, summary_claim_review_report
    from video_knowledge_pipeline.transcript_semantic_transaction import apply_transcript_semantic_restore
    plan, facts = fixture(tmp_path)
    text = '1\n00:00:00,000 --> 00:00:02,000\n产品甲2026版首次投保等待期90天。\n'
    path = tmp_path / 'source.srt'
    path.write_bytes(text.encode())
    cue = parse_transcript(path, source_snapshot=TextEvidenceSnapshot.read(tmp_path, 'source.srt'))[0]
    facts['sections'][0]['evidence_refs'][0]['source_locators'] = cue.metadata['source_locators']
    write_json(tmp_path / 'exports/smart-summary-chapter-fact-pack.json', facts)
    before = path.read_bytes()
    first = prepare_summary_claim_review(tmp_path)
    assert submit(tmp_path, first)['ok']
    review = prepare_summary_claim_review(tmp_path)
    final = submit(tmp_path, review, run='separate-synthetic-review')
    assert final['ok'] and summary_claim_review_report(tmp_path, plan, facts)['passed']
    tx = final['transaction']
    restored = apply_transcript_semantic_restore(tmp_path, tx['restore_plan'], execute=True,
        confirm_transaction_id=tx['transaction_id'])
    assert restored['status'] == 'restored'
    assert not summary_claim_review_report(tmp_path, plan, facts)['passed']
    assert path.read_bytes() == before
