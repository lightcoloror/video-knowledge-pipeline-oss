import json
import shutil
import subprocess

import pytest

from video_knowledge_pipeline.review_export_naming import review_export_script


def names(context, payload, count=1):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required to execute the browser naming function')
    script = review_export_script(context).removeprefix('<script>\n').removesuffix('\n</script>')
    js = 'global.window=global;\n' + script
    js += '\nconst payload=' + json.dumps(payload) + ';const before=JSON.stringify(payload);const out=[];for(let i=0;i<' + str(count) + ';i++) out.push(vkpReviewExportName("综合审核",payload,new Date("2026-09-18T08:30:25.123Z")));if(JSON.stringify(payload)!==before)throw Error("payload mutated");console.log(JSON.stringify(out));'
    result = subprocess.run([node, '-'], input=js, encoding='utf-8', capture_output=True, check=True)
    return json.loads(result.stdout)


def test_selected_clip_uses_source_filename_range_beijing_time_and_collision_suffix():
    context = {'title': 'generic', 'sources': [{'source_id': 'v', 'path': 'D:/media/第三节.mp4'}],
               'requests': [{'review_id': 'a', 'review_start_seconds': 750, 'review_end_seconds': 920},
                            {'review_id': 'b', 'review_start_seconds': 1000, 'review_end_seconds': 1001}]}
    first, second = names(context, {'decisions': [{'review_id': 'a'}]}, 2)
    assert first == '第三节__综合审核__00h12m30s-00h15m20s__20260918-163025123.json'
    assert second.endswith('163025123-1.json')


def test_disjoint_and_unknown_ranges_are_not_claimed_as_full_video():
    ctx = {'title': '视频:问题?'}
    assert '__多片段-2项__' in names(ctx, {'reviews': [{'start': 0, 'end': 2}, {'start': 9, 'end': 10}]})[0]
    assert '__范围见JSON__' in names(ctx, {'reviews': [{'candidate_id': 'missing'}]})[0]
    assert names(ctx, {})[0].startswith('视频-问题-__')
    assert '__全视频__' in names(ctx, {'review_scope': 'full_video'})[0]


def test_multiple_sources_and_unsafe_script_text():
    ctx = {'title': '</script><script>alert(1)</script>', 'sources': [
        {'source_id': 'a', 'path': 'one.mp4'}, {'source_id': 'b', 'path': 'two.mp4'}]}
    script = review_export_script({'title': ctx['title']})
    assert script.count('</script>') == 1
    assert names(ctx, {'decisions': []})[0].startswith('多视频-2个-one__')


def test_cross_bundle_review_and_timeline_index_lookup():
    assert names({'title': '课程'}, {'reviews': [{'bundle_dir': 'a'}, {'bundle_dir': 'b'}]})[0].startswith('多视频-2个-')
    result = names({'title': '课程', 'timeline': [{'index': 7, 'start': 0.5, 'end': 2}]}, {'reviews': [{'timeline_index': 7}]})[0]
    assert '__00h00m00s500-00h00m02s__' in result


def test_schema_body_is_not_modified_by_naming():
    payload = {'schema': 'lecture_review_notes.v1', 'reviews': [{'start': 1, 'end': 2}]}
    before = json.dumps(payload)
    names({'title': 'title'}, payload)
    assert json.dumps(payload) == before
