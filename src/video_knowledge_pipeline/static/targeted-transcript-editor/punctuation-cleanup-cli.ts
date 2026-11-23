import { readFileSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createReviewEditor, reviewPayload, ensureMetadata, type RevisionMeta } from './model';
import { buildPreviewPlan } from './preview';
import { applyPunctuationPlan, visibleTextIndex } from './punctuation-cleanup';

const dom = new JSDOM('<!doctype html><body></body>', { url: 'http://127.0.0.1' });
for (const key of ['window','document','navigator','Node','HTMLElement','MutationObserver','getComputedStyle','KeyboardEvent','Range']) Object.defineProperty(globalThis,key,{value:(dom.window as any)[key],configurable:true});
if (!Range.prototype.getClientRects) Range.prototype.getClientRects=()=>[] as any;
if (!Range.prototype.getBoundingClientRect) Range.prototype.getBoundingClientRect=()=>({top:0,left:0,bottom:0,right:0,width:0,height:0}) as any;
const input=JSON.parse(readFileSync(process.argv[2],'utf8'));
const review=structuredClone(input.review), evidence:any[]=[];
const preservedKeys=['change_id','kind','original','corrected','intent','decision','reason','evidence','effective_scope','effect','status','origin','confirmed_at','content_fingerprint','source_anchors'];
for (const chapter of input.chapters) {
  const saved=review.chapters.find((c:any)=>c.chapter===chapter.number);
  const editor=createReviewEditor({element:document.createElement('div'),content:saved.doc});
  const meta:Record<string,RevisionMeta>=Object.fromEntries(saved.revisions.map((r:any)=>[r.change_id,{...r}]));
  ensureMetadata(editor,chapter,meta);
  const baseline=reviewPayload(chapter,editor,meta), before=buildPreviewPlan(editor,chapter,meta,chapter.duration,review.manual_time_ranges?.[chapter.number]||[]);
  const plan=input.plan.chapters.find((c:any)=>c.chapter===chapter.number);
  const added=applyPunctuationPlan(editor,chapter,meta,plan);
  const after=reviewPayload(chapter,editor,meta), preview=buildPreviewPlan(editor,chapter,meta,chapter.duration,review.manual_time_ranges?.[chapter.number]||[]);
  for(const old of baseline.revisions){const current=after.revisions.find((r:any)=>r.change_id===old.change_id);assert.ok(current,'existing revision disappeared');for(const key of preservedKeys)assert.deepEqual((current as any)[key],(old as any)[key],`existing revision ${old.change_id}: ${key}`);}
  assert.deepEqual(preview.sections,before.sections,'audio keep ranges changed');assert.deepEqual(preview.cuts,before.cuts,'audio cuts changed');assert.equal(preview.editedDuration,before.editedDuration);assert.equal(after.original_text,baseline.original_text);
  assert.ok(after.revisions.filter((r:any)=>added.includes(r.change_id)).every((r:any)=>r.effect==='transcript_only'));
  Object.assign(saved,after,{preview});
  evidence.push({chapter:chapter.number,added_revisions:added.length,removed_punctuation:plan.removals.length,protected_user_characters_skipped:plan.protected_user_characters_skipped,existing_revisions_preserved:true,media_ranges_unchanged:true,before_flat:plan.original_flat,after_flat:visibleTextIndex(editor).text});editor.destroy();
}
review.exported_at=new Date().toISOString();review.punctuation_cleanup={policy:input.plan.policy,effect:'transcript_only',media_modified:false};
writeFileSync(process.argv[3],JSON.stringify({review,evidence},null,2));
console.log(JSON.stringify({chapters:evidence.length,added_revisions:evidence.reduce((n,c)=>n+c.added_revisions,0),removed_punctuation:evidence.reduce((n,c)=>n+c.removed_punctuation,0),media_modified:false}));
