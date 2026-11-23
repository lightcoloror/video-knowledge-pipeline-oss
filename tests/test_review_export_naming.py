import json
import re
import shutil
import subprocess

import pytest

from video_knowledge_pipeline.review_export_naming import review_export_script


def _node_names(script, payload, count):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required to execute the browser naming function')
    body = re.sub(r'^<script[^>]*>\n', '', script).removesuffix('\n</script>')
    js = 'global.window=global;\n' + body
    js += '\nconst payload=' + json.dumps(payload) + ';const before=JSON.stringify(payload);const out=[];for(let i=0;i<' + str(count) + ';i++) out.push(vkpReviewExportName("综合审核",payload,new Date("2026-09-18T08:30:25.123Z")));if(JSON.stringify(payload)!==before)throw Error("payload mutated");console.log(JSON.stringify(out));'
    result = subprocess.run([node, '-'], input=js, encoding='utf-8', capture_output=True, check=True)
    return json.loads(result.stdout)


def names_legacy(context, payload, count=1):
    """Old one-argument callers: public fallback when no identity claim exists."""
    return _node_names(review_export_script(context), payload, count)


def names_bound(context, payload, count=1, project_context=None):
    """Current two-argument callers: explicit, strict identity dispatch."""
    return _node_names(review_export_script(context, project_context), payload, count)


def project(context):
    return {**context, 'project_identity': {'topic_title': '通用测试选题', 'video_title': '介绍视频', 'video_id': 'project-test', 'time_zone': 'Asia/Shanghai'}}


# ---------------------------------------------------------------------------
# Public baseline contract: legacy one-argument producers without identity keep
# the historical offline filename behavior.
# ---------------------------------------------------------------------------

def test_public_selected_clip_uses_source_filename_range_beijing_time_and_collision_suffix():
    context = {'title': 'generic', 'sources': [{'source_id': 'v', 'path': 'D:/media/第三节.mp4'}],
               'requests': [{'review_id': 'a', 'review_start_seconds': 750, 'review_end_seconds': 920},
                            {'review_id': 'b', 'review_start_seconds': 1000, 'review_end_seconds': 1001}]}
    first, second = names_legacy(context, {'decisions': [{'review_id': 'a'}]}, 2)
    assert first == '第三节__综合审核__00h12m30s-00h15m20s__20260918-163025123.json'
    assert second.endswith('163025123-1.json')


def test_public_disjoint_and_unknown_ranges_are_not_claimed_as_full_video():
    ctx = {'title': '视频:问题?'}
    assert '__多片段-2项__' in names_legacy(ctx, {'reviews': [{'start': 0, 'end': 2}, {'start': 9, 'end': 10}]})[0]
    assert '__范围见JSON__' in names_legacy(ctx, {'reviews': [{'candidate_id': 'missing'}]})[0]
    assert names_legacy(ctx, {})[0].startswith('视频-问题-__')
    assert '__全视频__' in names_legacy(ctx, {'review_scope': 'full_video'})[0]


def test_public_multiple_sources_and_unsafe_script_text():
    ctx = {'title': '</script><script>alert(1)</script>', 'sources': [
        {'source_id': 'a', 'path': 'one.mp4'}, {'source_id': 'b', 'path': 'two.mp4'}]}
    script = review_export_script({'title': ctx['title']})
    assert script.count('</script>') == 1
    assert names_legacy(ctx, {'decisions': []})[0].startswith('多视频-2个-one__')


def test_public_cross_bundle_review_and_timeline_index_lookup():
    assert names_legacy({'title': '课程'}, {'reviews': [{'bundle_dir': 'a'}, {'bundle_dir': 'b'}]})[0].startswith('多视频-2个-')
    result = names_legacy({'title': '课程', 'timeline': [{'index': 7, 'start': 0.5, 'end': 2}]}, {'reviews': [{'timeline_index': 7}]})[0]
    assert '__00h00m00s500-00h00m02s__' in result


def test_public_schema_body_is_not_modified_by_naming():
    payload = {'schema': 'lecture_review_notes.v1', 'reviews': [{'start': 1, 'end': 2}]}
    before = json.dumps(payload)
    names_legacy({'title': 'title'}, payload)
    assert json.dumps(payload) == before


def test_legacy_one_argument_source_only_without_title_still_falls_back():
    """Old source-only callers without a title keep the public fallback filename."""
    context = {'sources': [{'path': 'VID_001.mp4'}]}
    name = names_legacy(context, {'reviews': [{'start': 0, 'end': 1}]})[0]
    assert name.startswith('VID_001__综合审核__00h00m00s-00h00m01s__')


# ---------------------------------------------------------------------------
# Current contract: explicitly dispatched two-argument calls are strict and
# bound identity drives the explicit Beijing filename and JSON identity.
# ---------------------------------------------------------------------------

def test_bound_selected_clip_uses_source_filename_range_beijing_time_and_collision_suffix():
    context = {'title': 'generic', 'sources': [{'source_id': 'v', 'path': 'D:/media/第三节.mp4'}],
               'requests': [{'review_id': 'a', 'review_start_seconds': 750, 'review_end_seconds': 920},
                            {'review_id': 'b', 'review_start_seconds': 1000, 'review_end_seconds': 1001}]}
    first, second = names_bound(project(context), {'decisions': [{'review_id': 'a'}]}, 2)
    assert first == '通用测试选题_介绍视频_project-test_综合审核_00h12m30s-00h15m20s_导出20260918_163025_123_UTC+0800.json'
    assert second.endswith('UTC+0800_1.json')


def test_bound_disjoint_and_unknown_ranges_are_not_claimed_as_full_video():
    ctx = project({'title': '视频:问题?'})
    assert '_多片段-2项_' in names_bound(ctx, {'reviews': [{'start': 0, 'end': 2}, {'start': 9, 'end': 10}]})[0]
    assert '_1项_' in names_bound(ctx, {'reviews': [{'candidate_id': 'missing'}]})[0]
    assert '_全视频_' in names_bound(ctx, {'review_scope': 'full_video'})[0]


def test_bound_multiple_sources_and_unsafe_script_text():
    ctx = project({'sources': [
        {'source_id': 'a', 'path': 'one.mp4'}, {'source_id': 'b', 'path': 'two.mp4'}]}
    )
    ctx['project_identity']['topic_title'] = '</script><script>alert(1)</script>'
    script = review_export_script(ctx, None)
    assert script.count('</script>') == 1
    assert 'one' not in names_bound(ctx, {'decisions': []})[0]


def test_bound_cross_bundle_review_and_timeline_index_lookup():
    with pytest.raises(subprocess.CalledProcessError):
        names_bound(project({}), {'reviews': [{'bundle_dir': 'a'}, {'bundle_dir': 'b'}]})
    result = names_bound(project({'timeline': [{'index': 7, 'start': 0.5, 'end': 2}]}), {'reviews': [{'timeline_index': 7}]})[0]
    assert '_00h00m00s500-00h00m02s_' in result


def test_bound_schema_body_is_not_modified_by_naming():
    payload = {'schema': 'lecture_review_notes.v1', 'reviews': [{'start': 1, 'end': 2}]}
    before = json.dumps(payload)
    names_bound(project({}), payload)
    assert json.dumps(payload) == before


def test_bound_exact_project_context_takes_precedence_over_multiple_camera_filenames():
    context = {'pack_payload_sha256': 'pack', 'sources': [{'source_id': 'source-001', 'label': 'VID_A.mp4'}, {'source_id': 'source-002', 'label': 'VID_B.mp4'}],
               'transcript_edit': {'chapters': [{'number': '01', 'file': 'VID_A.mp4', 'source_text_hash': 'a'}, {'number': '02', 'file': 'VID_B.mp4', 'source_text_hash': 'b'}]}, 'requests': [{'review_id': 'a'}, {'review_id': 'b'}]}
    identity = {'topic_title': '合成多来源选题', 'video_title': '合成示例视频', 'video_id': 'synthetic-video-2026', 'time_zone': 'Asia/Shanghai', 'pack_payload_sha256': 'pack', 'source_hashes': {'01': 'a', '02': 'b'}, 'source_files': {'01': 'VID_A.mp4', '02': 'VID_B.mp4'}}
    payload = {'schema_version': 'vkp-targeted-review-decisions.draft.v1', 'pack_payload_sha256': 'pack', 'decisions': [{'review_id': 'a', 'decision': 'pass', 'note': '获准'}]}
    result = names_bound(context, payload, project_context=identity)[0]
    assert result == '合成多来源选题_合成示例视频_synthetic-video-2026_2段口播_精确审核结果_1项_导出20260918_163025_123_UTC+0800.json'
    assert '多视频' not in result and 'VID_A' not in result


def test_bound_exact_project_context_cannot_be_reused_for_a_different_pack():
    with pytest.raises(ValueError, match='review_export_context_pack_changed'):
        review_export_script({'pack_payload_sha256': 'a'}, {'pack_payload_sha256': 'b'})


def test_missing_identity_blocks_instead_of_guessing_camera_filename():
    # Explicit two-argument call, no identity claim: the page may render, but
    # the export function must fail rather than guess a camera filename.
    with pytest.raises(subprocess.CalledProcessError):
        names_bound({'sources': [{'path': 'VID_001.mp4'}]}, {'reviews': [{'start': 0, 'end': 1}]})


def test_explicit_none_with_title_does_not_enter_legacy_guessing():
    context = {'title': 'legacy title', 'sources': [{'source_id': 'v', 'path': 'VID_001.mp4'}]}
    with pytest.raises(subprocess.CalledProcessError):
        names_bound(context, {'reviews': [{'start': 0, 'end': 1}]}, project_context=None)


def test_incomplete_claimed_identity_with_title_does_not_enter_legacy_guessing():
    context = {'title': 'legacy title', 'sources': [{'source_id': 'v', 'path': 'VID_001.mp4'}]}
    incomplete = {'topic_title': 'only-topic', 'pack_payload_sha256': 'pack'}
    with pytest.raises(subprocess.CalledProcessError):
        names_bound(context, {'reviews': [{'start': 0, 'end': 1}]}, project_context=incomplete)


def test_one_argument_empty_inline_project_identity_is_a_claim_not_legacy():
    context = {'title': 'legacy title', 'sources': [{'path': 'VID_001.mp4'}], 'project_identity': {}}
    with pytest.raises(subprocess.CalledProcessError):
        names_legacy(context, {'reviews': [{'start': 0, 'end': 1}]})


def test_one_argument_empty_inline_review_export_context_is_a_claim_not_legacy():
    context = {'title': 'legacy title', 'sources': [{'path': 'VID_001.mp4'}], 'review_export_context': {}}
    with pytest.raises(subprocess.CalledProcessError):
        names_legacy(context, {'reviews': [{'start': 0, 'end': 1}]})


def test_explicit_empty_identity_does_not_fall_back():
    context = {'title': 'legacy title', 'sources': [{'path': 'VID_001.mp4'}]}
    with pytest.raises(subprocess.CalledProcessError):
        names_bound(context, {'reviews': [{'start': 0, 'end': 1}]}, project_context={})
