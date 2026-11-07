# Native context adapter — bounded acceptance

2026-11-07. Codex host review; exact runtime model not independently verified.

Focused host execution: Python 3.12.11, eight passed, zero failed, zero skipped.
The tests call the actual public Main handoff preparation, native first-step
review plan, purpose review, candidate-ready predecessor verification and
context validation. Candidate-ready is explicitly not workflow completion.
Cases cover full rerun, same-hash reuse, changed audio/window policy, legitimate
separately bound audio, unbound audio, missing predecessor/purpose, context
hash drift and unchanged-audio changed-window rejection without a new output.

Three earlier failed fixture runs were retained (2 pass/6 fail, 1 pass/7 fail,
7 pass/1 fail). Corrections concern schema role versus recipe source role,
unique recipe roles and native exception-code assertions. No main schema,
purpose, predecessor or hash guard was relaxed. The actual adapter additionally
requires the expected output contract and bound audio, and rejects invalid
changed-window reuse before the shared writer runs.

Scope: synthetic metadata decisions; no ASR model, actual audio decoding,
voice cloning, media rendering, human review, publishing or platform outcome.
Wheel contents and installed entry points are independently checked by the
release host; tests do not substitute for those checks or production use.
Task-level process deviations are retained privately in the host handoff
receipt; accepting tested code does not establish a clean delegation run.
