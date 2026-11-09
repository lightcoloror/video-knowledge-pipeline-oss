# Media Equivalence Audit PowerShell Entry Wrapper

`docs/media-equivalence-entry-wrapper.md` documents the optional, source-only PowerShell entry wrapper in `scripts/audit-media-equivalence.ps1`, licensed under AGPL-3.0-only as part of the Video Knowledge Pipeline (VKP). It delegates directly to the backend Python module `video_knowledge_pipeline.media_equivalence_audit`.

## Invocation and Environment

The wrapper captures remaining CLI arguments via `ValueFromRemainingArguments`, resolves repository paths using `-LiteralPath` to safely handle directory paths with literal brackets, sets `$env:PYTHONPATH` to `$root/src`, configures `PYTHONIOENCODING=utf-8`, and forwards `@RemainingArgs` to the operator's `python` on `PATH`. It requires existing local Python, VKP dependencies, and installed FFmpeg/ffprobe tools.

Run the wrapper in a separate PowerShell process to inspect CLI options:

```powershell
powershell.exe -NoProfile -File .\scripts\audit-media-equivalence.ps1 --help
```

## CLI Parameters and Reporting

The underlying CLI accepts positional `candidate_media` and `retained_media`, requires `--output-json`, and supports optional `--output-markdown`, transcripts, ASR plans, and visual evidence. The existing backend owns the named storage-quality policies (`--policy space_saving` or `archive_lossless`) and its separate coverage/alignment inputs. A completed invocation writes the required JSON report and, if requested, Markdown; this is not a no-write command. There are no `--dry-run` or `--plan-only` flags.

## Deletion and Verification Boundaries

The tool generates audit reports only; neither the module nor the wrapper deletes media or grants deletion authority. A report declaring `safe_to_delete: true` and terminating with exit code 0 represents programmatic criterion satisfaction, not human approval or an operational cleanup command. Owner non-executing ParseFile checks under PowerShell 5.1.26100.4652 Desktop reported no syntax errors in the original or candidate. The parameter block and all 4 other top-level statements remain unchanged; the only source change is LiteralPath root resolution. Neither wrapper nor Python backend, CLI help, FFmpeg/ffprobe, ASR or media audit was executed. Runtime forwarding/exit behavior, PATH binding, evidence quality, installed-wheel packaging, other platforms and the full workflow remain unverified.
