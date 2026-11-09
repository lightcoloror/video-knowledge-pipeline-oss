import {afterEach,describe,it,expect} from 'vitest';
import {createReviewEditor,initialDocument,ensureMetadata,reviewPayload,trackedDelete,type Chapter} from '../model';
import {buildPreviewPlan} from '../preview';
import {initialPacingReview,validatePacingReview,pacingAuditionCandidates,residualPauseWarnings,type PauseCandidate} from '../pacing-review';

// All data here is invented for this test. No real review pack, manuscript or media is read.
const editors:ReturnType<typeof createReviewEditor>[]=[];
const chapter:Chapter={number:'02',file:'synthetic.mp4',duration:10,source_text_hash:'fixture',script:'',segments:[
 {id:'s1',start:1,end:2,text:'结论？',preselection:'keep',preselection_reason:'',script_match:'',timings:[{start_char:0,end_char:2,text:'结论',start:1,end:2}]},
 {id:'s2',start:2.24,end:9,text:'可以继续。',preselection:'keep',preselection_reason:'',script_match:'',timings:[{start_char:0,end_char:2,text:'可以',start:6.2,end:7},{start_char:2,end_char:4,text:'继续',start:8,end:9}]}]};
const pause:PauseCandidate={id:'pacing:fixture',chapter:'02',file:'synthetic.mp4',source_hash:'fixture',kind:'pause',where:'between_sentences',
 protect_start:2,pause_start:2.1,start:2.375,end:5.925,pause_end:6.2,protect_end:6.2,preserve_s:.55,
 left:{segment_id:'s1',start_char:0,end_char:2},right:{segment_id:'s2',start_char:0,end_char:2},
 intent:'synthetic',decision:'pending',reason:'synthetic',evidence:'synthetic',effective_scope:'synthetic'};
const packet={schema:'vkp.targeted-pacing-candidates.v1',sha256:'synthetic-packet-sha256',policy:{name:'synthetic-policy',word_tail_guard_s:.15,between_sentences_compress_from_s:.9,within_sentence_compress_from_s:.5},audition_enabled_by_user_request:true,candidates:[pause]};
function make(ch=chapter,doc=initialDocument(ch)){const e=createReviewEditor({element:document.createElement('div'),content:doc});editors.push(e);return e;}
afterEach(()=>editors.splice(0).forEach(e=>e.destroy()));

describe('v3 pacing candidate audition without invented audio approval',()=>{
 it('flags the retained gap around a quiet island without treating noise as authorization to cut',()=>{
  const p={...pause,protect_start:68.32,protect_end:79.32,start:72.259,end:79.045,where:'between_sentences'};
  const policy={between_sentences_compress_from_s:.9,within_sentence_compress_from_s:.5};
  expect(residualPauseWarnings([p],[{start:66.49,end:72.259},{start:79.045,end:85.121}],policy)[0]).toContain('4.214');
  expect(residualPauseWarnings([p],[{start:66.49,end:68.595},{start:79.045,end:85.121}],policy)).toEqual([]);
  expect(residualPauseWarnings([p],[{start:66.49,end:72.259}],undefined)).toEqual([]);
 });
 it('initializes all suggestions as pending, keeps audition separate from confirmation',()=>{
  const state=initialPacingReview(packet)!;
  expect(state.enabled).toBe(true);
  expect(state.decisions).toEqual({});
  expect(packet.candidates.every((c:any)=>!c.audio_checked&&c.decision==='pending')).toBe(true);
 });
 it('compresses only the safe middle and restores both native tail/onset handles',()=>{
  const e=make(),meta={};ensureMetadata(e,chapter,meta);
  const plan=buildPreviewPlan(e,chapter,meta,10,[],[pause]);
  expect(plan.sections).toEqual([{start:1,end:2.375},{start:5.925,end:9}]);
  expect(plan.sections[0].end-pause.pause_start+pause.pause_end-plan.sections[1].start).toBeCloseTo(.55);
  expect(plan.pacing_audition_ids).toEqual([pause.id]);
  expect(plan.subtitles.s2).toBe('可以继续。');
 });
 it('keeps original lexical and manual cuts in control',()=>{
  const e=make(),meta={};ensureMetadata(e,chapter,meta);
  const withManual=buildPreviewPlan(e,chapter,meta,10,[{start:6.5,end:6.8}],[pause]);
  expect(withManual.cuts.some(c=>c.start===6.5&&c.end===6.8)).toBe(true);
  let p=1;e.state.doc.descendants((n,pos)=>{if(n.isText&&n.text!.includes('可以'))p=pos;});
  e.commands.setTextSelection({from:p,to:p+2});trackedDelete(e);ensureMetadata(e,chapter,meta);
  expect(buildPreviewPlan(e,chapter,meta,10,[],[pause]).pacing_audition_ids).toEqual([]);
 });
 it('can keep one original gap or disable all auditions without accepting a cut',()=>{
  const state=initialPacingReview(packet)!;
  state.decisions[pause.id]={action:'keep'};
  expect(pacingAuditionCandidates(packet,state,pause.chapter).some(x=>x.id===pause.id)).toBe(false);
  state.enabled=false;
  expect(pacingAuditionCandidates(packet,state,pause.chapter)).toEqual([]);
 });
 it('rejects stale packets and fabricated completed review',()=>{
  const state=initialPacingReview(packet)!;
  state.packet_sha256='stale';
  expect(()=>validatePacingReview(state,packet)).toThrow();
  state.packet_sha256=packet.sha256;
  state.decisions[pause.id]={action:'compress'};
  expect(()=>validatePacingReview(state,packet)).toThrow();
  state.decisions[pause.id]={action:'compress',audio_checked:true,context_checked:true,semantic_role:'ordinary',evidence:'synthetic explicit review'};
  expect(()=>validatePacingReview(state,packet)).not.toThrow();
 });
 it('audits synthetic chapters: retained words keep coverage and old revisions stay intact',()=>{
  const auditChapter:Chapter={number:'01',file:'synthetic-audit.mp4',duration:20,source_text_hash:'audit-fixture',script:'',segments:[
   {id:'a1',start:2,end:8,text:'正常错词，继续正常。',preselection:'keep',preselection_reason:'',script_match:'',timings:[
    {text:'正常',start_char:0,end_char:2,start:2,end:2.8},{text:'错词',start_char:2,end_char:4,start:3,end:4},{text:'继续正常',start_char:5,end_char:9,start:5,end:8}]},
   {id:'a2',start:10,end:15,text:'另一句正常。',preselection:'keep',preselection_reason:'',script_match:'',timings:[{text:'另一句正常',start_char:0,end_char:5,start:10,end:15}]}]};
  const auditPause:PauseCandidate={...pause,id:'pacing:audit',chapter:'01',file:'synthetic-audit.mp4',source_hash:'audit-fixture',protect_start:8,pause_start:8.2,start:8.4,end:9.4,pause_end:9.6,protect_end:9.8,left:{segment_id:'a1',start_char:8,end_char:9},right:{segment_id:'a2',start_char:0,end_char:1}};
  const e=make(auditChapter),meta={};ensureMetadata(e,auditChapter,meta);
  const beforePayload=reviewPayload(auditChapter,e,meta);
  const before=buildPreviewPlan(e,auditChapter,meta,20);
  const after=buildPreviewPlan(e,auditChapter,meta,20,[],[auditPause]);
  const covered=(t:{start:number;end:number},sections:{start:number;end:number}[])=>sections.reduce((n,s)=>n+Math.max(0,Math.min(t.end,s.end)-Math.max(t.start,s.start)),0);
  for(const s of auditChapter.segments)for(const t of s.timings!)expect(covered(t,after.sections)+1e-6).toBeGreaterThanOrEqual(covered(t,before.sections));
  expect(reviewPayload(auditChapter,e,meta)).toEqual(beforePayload);
  expect(after.editedDuration).toBeLessThanOrEqual(before.editedDuration);
 });
});
