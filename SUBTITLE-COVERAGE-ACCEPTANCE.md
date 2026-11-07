# Subtitle coverage public increment

2026-11-07, Codex; exact runtime model not independently verified.

The main video-delivery composition exposed a missing first-party module:
31 tests failed because subtitle_coverage.py was absent from the public source.
The original implementation and its focused tests were frozen from untracked
working files, not attributed to the observed source HEAD. Original selected
bytes and tracked state were preserved. Existing AGPL-3.0-only and notices apply;
no media, models, fonts, third-party implementation or private paths are added.

An independent static review led to one reproduced implementation fix: a
meaningful words/timestamps length mismatch without usable sentence fallback
must raise instead of silently discarding that record. A valid fallback remains
supported. Evidence refs are caller labels; explicit declared evidence-file
hashes are checked separately by check_subtitles. Pure audit does not verify
the truth of caller ASR-quality or human-listening assertions. New boundary
fields state what local checks actually ran. Publication remains unauthorized.

Focused Windows synthetic checks: 50 passed, no skips. They cover source byte
drift, duration mismatch, declared evidence drift, ASR source/output hash binding,
changed ASR input path, malformed records/fallback, preserved returned lyric
text and unresolved human conflicts. Direct file-binding unit tests use real
temporary bytes/hashes with an explicitly mocked duration probe; they do not
claim those bytes were valid media or that actual ASR ran.

Main composition against actual public Owners plus this explicit candidate:
95 passed, 58 subtests passed; one browser regression was not run because it
requires explicitly supplied Node and Playwright paths. The original 31 missing
module failures and two subsequent TEMP-root harness failures remain in the
private host receipt. Correcting process-local TEMP/TMP/TMPDIR in the owned test
workspace required no ACL, proxy, security or system-setting change.

The composition includes synthetic local FFmpeg operations, not real-media,
customer, personal voice, provider, author or platform verification. A complete
audit and its omission/evidence context must accompany human review; a prompt
or hash match cannot establish that someone listened. Source publication and
passing fixtures grant no per-use execution or publishing authority.
