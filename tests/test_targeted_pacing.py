import hashlib,json,tempfile,unittest
from pathlib import Path
from video_knowledge_pipeline.targeted_pacing import load_pacing_candidates,validate_pacing_review,PACING_FILENAME

class TargetedPacingTests(unittest.TestCase):
 def packet(self):
  source={'number':'02','source_text_hash':'text','source_sha256':'media','duration':10,'segments':[{'timings':[{'start':1,'end':2},{'start':6.2,'end':7}]}]}
  row={'id':'pacing:example','chapter':'02','source_hash':'text','source_sha256':'media','protect_start':2,'pause_start':2.1,'start':2.375,'end':5.925,'pause_end':6.2,'protect_end':6.2}
  return {'pack_payload_sha256':'pack','transcript_edit':{'chapters':[source]}},{'schema':'vkp.targeted-pacing-candidates.v1','pack_payload_sha256':'pack','candidates':[row]}
 def seal(self,value):
  value['sha256']=hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!='sha256'},ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest();return value
 def test_source_bound_packet_loads_and_absence_is_backwards_compatible(self):
  pack,data=self.packet()
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.assertIsNone(load_pacing_candidates(root,pack));(root/PACING_FILENAME).write_text(json.dumps(self.seal(data)),encoding='utf-8');self.assertEqual(load_pacing_candidates(root,pack),data)
 def test_changed_hash_source_and_speech_overlap_are_rejected(self):
  for kind in ('hash','source','word','nan'):
   pack,data=self.packet();row=data['candidates'][0]
   if kind=='source':row['source_sha256']='wrong'
   if kind=='word':row['start']=1.5;row['pause_start']=1.4;row['protect_start']=1
   if kind=='nan':row['start']=float('nan')
   self.seal(data)
   if kind=='hash':data['sha256']='wrong'
   with tempfile.TemporaryDirectory() as d:
    root=Path(d);(root/PACING_FILENAME).write_text(json.dumps(data),encoding='utf-8')
    with self.assertRaises(ValueError):load_pacing_candidates(root,pack)
 def test_pending_audition_is_not_confirmed_compression(self):
  _,data=self.packet();self.seal(data);state={'packet_sha256':data['sha256'],'enabled':True,'decisions':{'pacing:example':{'action':'keep_pending_review'}}};validate_pacing_review(state,data)
  state['decisions']['pacing:example']['action']='compress'
  with self.assertRaisesRegex(ValueError,'audio_context_review_required'):validate_pacing_review(state,data)
  state['decisions']['pacing:example'].update(audio_checked=True,context_checked=True,semantic_role='ordinary',evidence='human review');validate_pacing_review(state,data)
 def test_wrong_packet_unknown_decision_and_non_boolean_toggle_fail(self):
  _,data=self.packet();self.seal(data)
  for state in ({'packet_sha256':'wrong','enabled':True,'decisions':{}},{'packet_sha256':data['sha256'],'enabled':'yes','decisions':{}},{'packet_sha256':data['sha256'],'enabled':True,'decisions':{'missing':{'action':'keep'}}}):
   with self.assertRaises(ValueError):validate_pacing_review(state,data)
