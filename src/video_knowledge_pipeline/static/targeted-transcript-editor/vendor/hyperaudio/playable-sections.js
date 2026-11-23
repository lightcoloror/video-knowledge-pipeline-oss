/*
 * Public-source curation marker, 2026-11-10.
 * Existing local VKP adaptation is retained below; this marker does not modify,
 * replace or re-date that adaptation.
 * Upstream base: hyperaudio/hyperaudio-lite-editor, editor-audio-cut.js,
 * commit b8bd4728f6d6914a37350d119cf52d4c29eb9968.
 * Original upstream copyright and license texts (vendor/hyperaudio/LICENSE,
 * vendor/hyperaudio/LICENSE-MIT, vendor/hyperaudio/LICENSES) remain retained.
 * No new authorship is claimed. The date above is the public-source curation
 * date only; it is not an inferred historical modification date.
 */

// Adapted from Hyperaudio Project, editor-audio-cut.js, b8bd4728f6d6914a37350d119cf52d4c29eb9968.
// Default upstream license: AGPL-3.0; original LICENSE and LICENSES retained.
// Only adaptation: explicit player/settings dependencies instead of page globals.
function isStruck(el) { return el.style.textDecoration.includes("line-through"); }
export function computePlayableSections(transcript, options = {}) {
    const wordEls = transcript.querySelectorAll('[data-m]');
    const player = options.player;
    const removeGapsEnabled = !!options.removeGapsEnabled, gapThreshold = options.gapThreshold ?? 0.5, gapBuffer = options.gapBuffer ?? 0.1;
    const duration = (player && !isNaN(player.duration) && player.duration > 0) ? player.duration : Infinity;

    if (wordEls.length === 0) {
      return [{ start: 0, end: duration }];
    }

    const words = Array.from(wordEls).map(el => {
      const start = parseInt(el.getAttribute('data-m')) / 1000;
      return { start, end: start + (parseInt(el.getAttribute('data-d')) || 0) / 1000,
               struck: isStruck(el), text: el.textContent };
    });

    // Two passes. First, the regions between consecutive KEPT (unstruck) words:
    // everything in such a region is non-speech once its struck words are cut,
    // so with gap skipping on we treat the region as one combined pause — if
    // the pause left after removing the struck words exceeds the threshold,
    // cut the whole region (keeping a small buffer at each edge so word edges
    // aren't clipped). Otherwise cut just the struck spans. The old one-pass
    // version only gap-checked between two unstruck neighbours, so the pauses
    // AROUND a struck filler word were never skipped.
    const cuts = [];
    let prevKeptEnd = null;
    let prevKeptWord = null;
    let pendingStruck = [];

    // A run of struck words in one region is always ONE cut spanning from the
    // first struck word's start to the last one's end, so the silences BETWEEN
    // struck words go with them (cutting only the word spans left those pauses
    // playing back-to-back — audible fragments of the struck region).
    const struckRunCut = (struckSpans) => {
      if (struckSpans.length === 0) return null;
      let end = struckSpans[0].end;
      struckSpans.forEach(s => { if (s.end > end) end = s.end; }); // tolerate overlapping word timings
      return { start: struckSpans[0].start, end };
    };

    // #703 — a word whose timestamps claim far less time than its syllables need has
    // almost certainly been cut short by the recogniser, and the "silence"
    // after it is the rest of the word rather than a pause. Cutting there
    // removes speech, and in an export that loss is permanent.
    //
    // Counting vowel groups is crude and English-shaped. That is acceptable
    // precisely because this only ever DECLINES to remove audio: a false
    // positive keeps a pause someone wanted gone, a false negative is
    // today's behaviour. It never invents or extends a timing — the
    // timestamps stay exactly as the recogniser reported them, because
    // everything else in the editor trusts them.
    //
    // The bar is set from the TRANSCRIPT'S OWN median, not a constant.
    // Recordings differ enormously in how long a syllable is held: across a
    // sample of sixteen, medians ran from 0.080 to 0.240 s/syllable — a
    // factor of three. A fixed bar calibrated to the slow end flagged 16% of
    // words in the fastest file and 0.5% in the slowest, which is a
    // threshold measuring the recording rather than the defect. Relative to
    // the median it stays between 0% and 2% across all of them.
    const TRUNCATION_RATIO = 0.4;
    // Vowel groups, accents included: stripping to [a-z] deletes the vowel in
    // "perché" or "città" and undercounts the word. Normalising first keeps
    // every Latin-script language countable by the same crude rule.
    //
    // Still English-shaped, and knowingly so. The silent terminal 'e' below
    // is an English convention — Italian "cognome" is co-gno-me, three, and
    // this returns two. That error makes a word look LONGER per syllable and
    // therefore flags less, which is the safe direction: it fails towards
    // today's behaviour rather than towards switching gap skipping off.
    const syllableCount = (text) => {
      const w = String(text || '')
        .toLowerCase()
        .normalize('NFD').replace(/[\u0300-\u036f]/g, '') // é -> e, à -> a
        .replace(/[^a-z]/g, '');
      if (w === '') return 0;
      const groups = w.match(/[aeiouy]+/g);
      let n = groups ? groups.length : 1;
      if (w.endsWith('e') && n > 1) n -= 1; // silent terminal e (English)
      return Math.max(1, n);
    };
    // Speaker labels are spans too, carrying data-d="0" because nobody speaks
    // them. Left in, every one reads as an infinitely truncated word and
    // spares the gap after it — which is most paragraph openings.
    const spokenRate = (word) => {
      if (!word) return null;
      const syllables = syllableCount(word.text);
      const seconds = word.end - word.start;
      if (syllables < 1 || seconds <= 0) return null;
      return seconds / syllables;
    };
    const rates = words.map(spokenRate).filter((r) => r !== null).sort((a, b) => a - b);
    // Too few words to characterise the recording: judge nothing.
    const medianRate = rates.length >= 20 ? rates[Math.floor(rates.length / 2)] : null;
    const looksTruncated = (word) => {
      if (medianRate === null) return false;
      const syllables = syllableCount(word ? word.text : '');
      // One-syllable words are legitimately very short — 80 ms for "the" is
      // ordinary. Only a multi-syllable word carries enough signal to call
      // its duration implausible.
      if (syllables < 2) return false;
      const rate = spokenRate(word);
      return rate !== null && rate < medianRate * TRUNCATION_RATIO;
    };

    const flushRegion = (regionStart, regionEnd, struckSpans, bufferLeft, bufferRight, precedingWord) => {
      const run = struckRunCut(struckSpans);
      // Struck words are still cut when the gap is spared: the user asked for
      // those to go, and only the inferred silence is in doubt.
      if (removeGapsEnabled && regionStart !== null && !looksTruncated(precedingWord)) {
        const struckTotal = struckSpans.reduce((sum, s) => sum + (s.end - s.start), 0);
        const effectivePause = (regionEnd - regionStart) - struckTotal;
        if (effectivePause > gapThreshold) {
          // cut the whole region; the edge buffers protect the neighbouring
          // kept words' tails, but never at the price of keeping struck audio
          const cutStart = Math.min(regionStart + (bufferLeft ? gapBuffer : 0), run !== null ? run.start : Infinity);
          const cutEnd = Math.max(regionEnd - (bufferRight ? gapBuffer : 0), run !== null ? run.end : -Infinity);
          if (cutEnd > cutStart) {
            cuts.push({ start: cutStart, end: cutEnd });
            return;
          }
        }
      }
      // gaps kept (or gap skipping off): cut the struck run whole
      if (run !== null) cuts.push(run);
    };

    // The LEADING region (#577): media start to the first kept word. Handled
    // separately because the loop below only knows regions BETWEEN kept words
    // — prevKeptEnd is null until one exists — so the setting-up before anyone
    // speaks was never a candidate, however long. Struck words at the head
    // belong to this region too, so their audio goes with it.
    const firstKeptIndex = words.findIndex((w) => !w.struck);
    if (firstKeptIndex !== -1 && words[firstKeptIndex].start > 0) {
      const leadingStruck = words.slice(0, firstKeptIndex).filter((w) => w.struck);
      // bufferLeft false: nothing precedes media start to protect
      flushRegion(0, words[firstKeptIndex].start, leadingStruck, false, true);
      // those struck words are now accounted for; the loop must not cut them again
      words.slice(0, firstKeptIndex).forEach((w) => { w.struck = false; w.consumed = true; });
    }

    words.forEach(word => {
      if (word.consumed) return; // already covered by the leading region
      if (word.struck) {
        pendingStruck.push(word);
        return;
      }
      if (pendingStruck.length > 0 || prevKeptEnd !== null) {
        const regionStart = prevKeptEnd !== null ? prevKeptEnd : (pendingStruck.length ? pendingStruck[0].start : null);
        if (regionStart !== null && word.start > regionStart) {
          flushRegion(regionStart, word.start, pendingStruck, prevKeptEnd !== null, true, prevKeptWord);
        } else {
          const run = struckRunCut(pendingStruck);
          if (run !== null) cuts.push(run);
        }
      }
      pendingStruck = [];
      prevKeptEnd = word.end;
      prevKeptWord = word;
    });

    // The TRAILING region: the last kept word to the end of the media (#577).
    // Previously this only existed when struck words followed the last kept
    // word, so plain dead air after the final word always played out. The
    // media's real end is only usable when metadata has loaded — with an
    // unknown duration the tail is left alone rather than guessed.
    const regionStart = prevKeptEnd !== null
      ? prevKeptEnd
      : (pendingStruck.length > 0 ? pendingStruck[0].start : null);
    if (regionStart !== null) {
      const struckEnd = pendingStruck.length > 0
        ? pendingStruck[pendingStruck.length - 1].end : regionStart;
      const regionEnd = Number.isFinite(duration) ? Math.max(duration, struckEnd) : struckEnd;
      if (regionEnd > regionStart) {
        // bufferRight false: nothing follows the media end to protect
        flushRegion(regionStart, regionEnd, pendingStruck, prevKeptEnd !== null, false, prevKeptWord);
      }
    }

    // Second pass: subtract the merged cuts from [0, duration].
    cuts.sort((a, b) => a.start - b.start);
    const sections = [];
    let cursor = 0;
    cuts.forEach(cut => {
      const start = Math.max(cut.start, cursor);
      if (start >= cut.end) { cursor = Math.max(cursor, cut.end); return; }
      if (start > cursor) sections.push({ start: cursor, end: start });
      cursor = cut.end;
    });
    if (cursor < duration) sections.push({ start: cursor, end: duration });
    if (sections.length === 0) sections.push({ start: duration, end: duration });

    return sections;
  }
