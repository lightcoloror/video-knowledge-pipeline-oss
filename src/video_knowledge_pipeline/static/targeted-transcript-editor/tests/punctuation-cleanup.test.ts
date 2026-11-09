import {afterEach,expect,it} from 'vitest';
import {createReviewEditor,initialDocument,trackedReplace,ensureMetadata,reviewPayload,type Chapter} from '../model';
import {applyPunctuationPlan,visibleTextIndex} from '../punctuation-cleanup';
import {buildPreviewPlan} from '../preview';
const ch:Chapter={number:'01',file:'synthetic.mp4',duration:10,source_text_hash:'fixture',script:'',segments:[
  {id:'s1',start:1,end:3,text:'正常？',preselection:'keep',preselection_reason:'',script_match:'',timings:[{text:'正常',start_char:0,end_char:2,start:1,end:3}]},
  {id:'s2',start:4,end:8,text:'？。继续，。',preselection:'keep',preselection_reason:'',script_match:'',timings:[{text:'继续',start_char:2,end_char:4,start:5,end:7}]},
]};
const editors:ReturnType<typeof createReviewEditor>[]=[];
function make(c=ch){const e=createReviewEditor({element:document.createElement('div'),content:initialDocument(c)});editors.push(e);return e;}
function plan(e:ReturnType<typeof make>){return{chapter:'01',source_hash:'fixture',original_flat:visibleTextIndex(e).text,protected_user_characters_skipped:0,removals:[{start_char:3,end_char:4,text:'？'},{start_char:4,end_char:5,text:'。'},{start_char:7,end_char:8,text:'，'}]};}
afterEach(()=>editors.splice(0).forEach(e=>e.destroy()));
it('adds tracked transcript-only corrections while retaining the immutable original',()=>{
 const e=make(),meta:any={};const before=reviewPayload(ch,e,meta);const ids=applyPunctuationPlan(e,ch,meta,plan(e));const after=reviewPayload(ch,e,meta);
 expect(after.original_text).toBe(before.original_text);expect(after.proposed_text.replace(/\n/g,'')).toBe('正常？继续。');expect(ids.length).toBeGreaterThan(0);
 expect(after.revisions.every(r=>r.effect==='transcript_only'&&r.origin==='punctuation_cleanup')).toBe(true);
 for(const r of after.revisions)for(const key of ['intent','decision','reason','evidence','effective_scope'])expect((r as any)[key]).toBeTruthy();
});
it('preserves the exact media sections and cuts',()=>{const e=make(),meta:any={};ensureMetadata(e,ch,meta);const before=buildPreviewPlan(e,ch,meta,10);applyPunctuationPlan(e,ch,meta,plan(e));const after=buildPreviewPlan(e,ch,meta,10);expect(after.sections).toEqual(before.sections);expect(after.cuts).toEqual(before.cuts);});
it('does not merge into or invalidate an existing confirmed human correction',()=>{
 const e=make(),meta:any={};e.commands.setTextSelection({from:1,to:3});trackedReplace(e,'健康');const[g]=ensureMetadata(e,ch,meta);meta[g.id].status='confirmed';meta[g.id].confirmed_at='2026-10-26T00:00:00Z';
 const before=reviewPayload(ch,e,meta).revisions.find(r=>r.change_id===g.id);applyPunctuationPlan(e,ch,meta,plan(e));expect(reviewPayload(ch,e,meta).revisions.find(r=>r.change_id===g.id)).toEqual(before);
});
it('retains sound when the correction empties an entire punctuation-only cue',()=>{
 const c=structuredClone(ch);c.segments=[c.segments[0],{id:'punct',start:3.2,end:4,text:'？',preselection:'keep',preselection_reason:'',script_match:''},c.segments[1]];
 const e=make(c),meta:any={};ensureMetadata(e,c,meta);const before=buildPreviewPlan(e,c,meta,10);
 const p={chapter:'01',source_hash:'fixture',original_flat:visibleTextIndex(e).text,protected_user_characters_skipped:0,removals:[{start_char:3,end_char:4,text:'？'}]};applyPunctuationPlan(e,c,meta,p);const after=buildPreviewPlan(e,c,meta,10);
 expect(after.subtitles.punct).toBe('');expect(after.sections).toEqual(before.sections);
});
it('fails closed on stale text and lexical edits',()=>{const e=make(),meta:any={};expect(()=>applyPunctuationPlan(e,ch,meta,{...plan(e),original_flat:'stale'})).toThrow();expect(()=>applyPunctuationPlan(e,ch,meta,{...plan(e),removals:[{start_char:0,end_char:1,text:'正'}]})).toThrow();});
it('never edits an existing insertion',()=>{
 const e=make(),meta:any={};e.commands.setTextSelection({from:1,to:3});trackedReplace(e,'？？');const p={...plan(e),removals:[{start_char:0,end_char:1,text:'？'}]};expect(()=>applyPunctuationPlan(e,ch,meta,p)).toThrow(/人工修订/);
});
it('can restore each automatic punctuation correction via the upstream reject command',()=>{const e=make(),meta:any={};const original=visibleTextIndex(e).text;const ids=applyPunctuationPlan(e,ch,meta,plan(e));ids.forEach(id=>e.commands.rejectChange(id));expect(visibleTextIndex(e).text).toBe(original);});
