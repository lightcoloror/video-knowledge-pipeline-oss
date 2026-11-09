// Reuses source-bound MVP spoken_pacing.pause_cut_window results. No second
// pause estimator, automatic approval or text-edit engine is implemented here.
export type PauseCandidate = { id: string; chapter: string; file: string; source_hash: string; kind: 'pause'; where: string; start: number; end: number; pause_start: number; pause_end: number; protect_start: number; protect_end: number; preserve_s: number; left: {segment_id:string;start_char:number;end_char:number}; right: {segment_id:string;start_char:number;end_char:number}; intent: string; decision: string; reason: string; evidence: string; effective_scope: string };
export type PauseDecision = { action: 'keep_pending_review' | 'keep' | 'compress'; audio_checked?: boolean; context_checked?: boolean; semantic_role?: string; evidence?: string; confirmed_at?: string };
export type PacingReview = { packet_sha256: string; enabled: boolean; decisions: Record<string, PauseDecision> };
export type PacingPacket = { sha256: string; policy: { name: string; word_tail_guard_s?: number; between_sentences_compress_from_s: number; between_sentences_preserve_s: number; within_sentence_compress_from_s: number; within_sentence_preserve_s: number }; audition_enabled_by_user_request: boolean; candidates: PauseCandidate[] };
export function initialPacingReview(packet?: PacingPacket): PacingReview | undefined { return packet ? {packet_sha256:packet.sha256,enabled:packet.audition_enabled_by_user_request,decisions:{}} : undefined; }
export function validatePacingReview(state: PacingReview | undefined, packet?: PacingPacket) {
  if (!state) return;
  if (!packet || state.packet_sha256 !== packet.sha256 || typeof state.enabled !== 'boolean' || !state.decisions || typeof state.decisions !== 'object') throw Error('气口候选版本不匹配，已停止导入。');
  for (const [id, decision] of Object.entries(state.decisions)) {
    if (!packet.candidates.some(c=>c.id===id) || !['keep_pending_review','keep','compress'].includes(decision.action)) throw Error('气口决策无效。');
    if (decision.action === 'compress' && !(decision.audio_checked === true && decision.context_checked === true && decision.semantic_role === 'ordinary' && decision.evidence?.trim())) throw Error('确认压缩前须回听、核对普通空隙并填写依据。');
  }
}
export function pacingAuditionCandidates(packet: PacingPacket | undefined, state: PacingReview | undefined, chapter: string): PauseCandidate[] {
  if (!packet || !state?.enabled || packet.sha256 !== state.packet_sha256) return [];
  return packet.candidates.filter(c=>c.chapter===chapter && state.decisions[c.id]?.action !== 'keep');
}

// A conservative quiet island may leave a long word-to-word gap around clicks
// or breaths. Report actual retained coverage; never silently extend the cut.
export function residualPauseWarnings(pauses: PauseCandidate[], sections: {start:number;end:number}[], policy?: {between_sentences_compress_from_s?:number;within_sentence_compress_from_s?:number}) {
  if (!policy) return [] as string[];
  return pauses.flatMap(p => {
    const threshold = p.where === 'between_sentences' ? policy.between_sentences_compress_from_s : policy.within_sentence_compress_from_s;
    const kept = sections.reduce((n,s)=>n+Math.max(0,Math.min(s.end,p.protect_end)-Math.max(s.start,p.protect_start)),0);
    return Number.isFinite(threshold) && kept > threshold! + .001
      ? [`气口 ${p.id} 处理后仍保留 ${kept.toFixed(3)} 秒（原片 ${p.protect_start.toFixed(3)}–${p.protect_end.toFixed(3)} 秒）；可能被杂音分隔，需回听复核。`] : [];
  });
}
