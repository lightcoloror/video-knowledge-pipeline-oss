from __future__ import annotations
import json
import os
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from test_targeted_review import request_pack, fake_probe
from video_knowledge_pipeline.targeted_review import prepare_targeted_review_pack, TargetedReviewError, PACK_FILENAME
from video_knowledge_pipeline.targeted_review_ui import HTML_TEMPLATE

class LayeredReviewTests(unittest.TestCase):
    def setUp(self):
        self.parent=Path(os.environ.get('VKP_MANAGED_TEST_ROOT',str(Path(__file__).resolve().parents[1]/'.test-runtime')))
        self.root=self.parent/('review-layers-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.path=request_pack(self.root)
        self.payload=json.loads(self.path.read_text(encoding='utf-8'))

    def tearDown(self):
        assert self.root.resolve().parent==self.parent.resolve()
        shutil.rmtree(self.root)

    def prepare(self, cues):
        self.payload['requests'][0]['subtitle_cues']=cues
        self.path.write_text(json.dumps(self.payload),encoding='utf-8')
        with patch('video_knowledge_pipeline.targeted_review.probe_video',side_effect=fake_probe):
            return prepare_targeted_review_pack(self.path,self.root/'review')

    def test_concurrent_age_and_dialogue_survive_pack_without_media(self):
        cues=[dict(start_seconds=2,end_seconds=8,text='1岁3个月',layer='age',font_size_1080=48),
              dict(start_seconds=2,end_seconds=4,text='爸比~！',layer='dialogue',speaker='child',motion='bounce',font_size_1080=96,timing_status='candidate'),
              dict(start_seconds=4,end_seconds=5,text='唉',layer='dialogue',speaker='father',motion='none',font_size_1080=76)]
        result=self.prepare(cues)
        self.assertEqual(result['media_copies_created'],0)
        pack=json.loads((self.root/'review'/PACK_FILENAME).read_text(encoding='utf-8'))
        for actual, expected in zip(pack['requests'][0]['subtitle_cues'],cues):
            for key,value in expected.items(): self.assertEqual(actual[key],value)
        self.assertEqual({p.suffix for p in (self.root/'review').iterdir()},{'.html','.json'})

    def test_invalid_presentation_fails_closed(self):
        for key,value in [('layer','html'),('speaker','admin'),('motion','javascript'),('timing_status','approved_by_ai'),('font_size_1080',999)]:
            with self.subTest(key=key):
                with self.assertRaises(TargetedReviewError):
                    self.prepare([dict(start_seconds=2,end_seconds=4,text='test',**{key:value})])

    def test_legacy_subtitles_remain_plain(self):
        self.prepare([dict(start_seconds=2,end_seconds=4,text='legacy')])
        pack=json.loads((self.root/'review'/PACK_FILENAME).read_text(encoding='utf-8'))
        self.assertNotIn('motion',pack['requests'][0]['subtitle_cues'][0])

    @unittest.skipUnless(shutil.which('node'), 'Node is needed to exercise review UI functions')
    def test_time_marks_validate_bounds_and_preserve_notes_without_approval(self):
        capture = HTML_TEMPLATE.split('    function captureDialogueTime',1)[1].split('    function refreshTimeDisplayWhilePlaying',1)[0]
        js = '''const assert=require('node:assert/strict');
let dialogueMarks={start:null,end:null}; let persisted=0;
const row={review_start_seconds:0,review_end_seconds:10};
const elements={player:{currentTime:0,pause(){}},timingMarks:{},timingFeedback:{},timingSpeaker:{value:'father'},timingText:{value:'hello'},note:{value:'existing note'}};
const document={getElementById(id){return elements[id];}};
function currentRow(){return row;}
function preciseTimecode(x){return String(x);}
function persistCurrentNote(){persisted++;}
'''+ 'function captureDialogueTime'+capture+'''
captureDialogueTime('start'); elements.player.currentTime=1.394; captureDialogueTime('end');
appendDialogueTiming(); assert.equal(persisted,1);
assert.ok(elements.note.value.startsWith('existing note\\n[对白定位] '));
const data=JSON.parse(elements.note.value.split('[对白定位] ')[1]);
assert.deepEqual(data,{speaker:'father',start_seconds:0,end_seconds:1.394,text:'hello'});
assert.equal(dialogueMarks.start,null); assert.equal(dialogueMarks.end,null);
elements.timingText.value='bad'; dialogueMarks={start:9,end:11};
appendDialogueTiming(); assert.equal(persisted,1);
dialogueMarks={start:5,end:4}; appendDialogueTiming(); assert.equal(persisted,1);
'''
        script=self.root/'timing-check.cjs'
        script.write_text(js,encoding='utf-8')
        result=subprocess.run(['node',str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__': unittest.main()
