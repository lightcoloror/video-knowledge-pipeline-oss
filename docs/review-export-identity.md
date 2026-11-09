# Review-export project identity (standalone module)

Status: source-only delta for the existing first-party identity module. This guide
covers purpose, usage and limits. Attempted synthetic verification was blocked by a filesystem permission error (see *Verification status*); no passing-suite claim is made.

## What it is

`video_knowledge_pipeline.review_export_context` is the explicit project-identity
module shared by local review producers. It records a topic title, video title,
video identifier and the Beijing-time contract in
`vkp.review-export-context.v1`. These fields are explicit caller-supplied
metadata claims recorded for export naming rather than inferred from camera
filenames, but they do not prove project ownership or media authenticity.

The module reuses the public package's `storage.read_json` / `storage.write_json`
helpers. `storage.py` is its only relative import. It adds no framework, router,
database, platform layer or packaging dependency.

## Files in this delta

- `src/video_knowledge_pipeline/review_export_context.py` — unchanged candidate
  source, SHA256 `e0a1073b04ad952f0dbad57a3f14a2416f6fbde36504e803a5fd8f9072b8c9b7`.
- `tests/test_review_export_identity_standalone.py` — focused synthetic checks.
- `docs/review-export-identity.md` — this guide.
- `README.md` — usage summary and link to this guide.

## Requirements

- Python 3.11+ (the public package already requires 3.11) and the real public
  `video_knowledge_pipeline` package tree (`__init__.py`, `storage.py`, and the
  rest). No install, network, model, media or browser step is needed for the
  identity contract itself.

## Command line

`configure` writes `review-export-context.json` in an existing operator-selected
directory. It writes the configuration atomically using the package's storage
writer (staging through a temporary file before replacement):

```sh
python -m video_knowledge_pipeline.review_export_context configure "<existing-root>" \
  --topic-title "<topic>" --video-title "<video title>" --video-id "<video id>"
```

`status` only reads. It prints the configured identity or a missing-identity
report as JSON:

```sh
python -m video_knowledge_pipeline.review_export_context status "<existing-root>"
```

`configure` always requires all three identity fields and full validation. All
three flags are required by the parser, and empty/whitespace-only values are
rejected. `time_zone` is fixed to `Asia/Shanghai`; there is no timezone flag.
`status` can raise on a malformed or explicitly incomplete claimed configuration
instead of reporting it as missing.

Synthetic configure output (paths and values are examples):

```json
{
  "status": "configured",
  "path": "<root>/review-export-context.json",
  "project_identity": {
    "topic_title": "<topic>",
    "video_title": "<video title>",
    "video_id": "<video id>",
    "time_zone": "Asia/Shanghai"
  },
  "review_decisions_modified": false
}
```

## Identity contract

- Required fields: `topic_title`, `video_title`, `video_id`.
- `time_zone` must be `Asia/Shanghai`; any other value raises.
- A claimed identity is any non-`None` value. An empty dict is a claim (an
  incomplete one) and must not fall through to another identity.
- Malformed claimed shapes (for example a JSON array or string where an object is
  required) raise; they are not silently dropped or converted to `None`.
- Missing identity (no claim at any layer) resolves to `None`; an explicitly empty
  claim resolves to an identity whose fields are empty strings.
- `allow_incomplete=True` only exists for display resolution. It never relaxes the
  timezone check, and the producer/preparation path still requires full identity.

### Lookup order

`configured_identity(root)` searches, in order:

1. `review-export-context.json`
2. `targeted-transcript-export-context.json` (legacy name)
3. `manifest.json` — `project_identity` when non-`None`, otherwise
   `review_export_context` when non-`None`

`resolve_review_export_context(context, explicit, root)` selects:
`explicit` → `context.review_export_context` → `context.project_identity` →
configured identity at `root` or `context.bundle_dir` / `artifacts.bundle_dir`.

`identity_for_preparation(raw, input_root, output_root, explicit)` selects:
`explicit` → `raw.project_identity` → `raw.review_export_context` → output-root
config → input-root config. Note that this producer order differs from
`resolve_review_export_context`.

## Python API

- `normalize_project_identity(value, allow_incomplete=False)` — validate and strip
  identity fields; raise on missing/invalid fields or a non-Beijing timezone.
- `configured_identity(root)` — read the configured claim from an operator root.
- `project_root(context)` — resolve `bundle_dir` or `artifacts.bundle_dir`.
- `source_catalog(context)` — normalize `sources` (or the `media` fallback) into
  deterministic source rows; it asserts no approval.
- `resolve_review_export_context(context, explicit=None, root=None)` — build the
  `vkp.review-export-context.v1` result, including pack hash, sources and chapter
  binding checks.
- `identity_for_preparation(raw, input_root, output_root, explicit=None)` — return
  the fully validated producer identity, or `None`.
- `write_bound_review_export_context(root, context)` — resolve and, when a value
  exists, write it to `root/review-export-context.json`; returns the value.
- `main(argv=None)` — the `configure` / `status` CLI.

## Bound data checks

- When both context and candidate provide a non-empty `pack_payload_sha256`, a mismatch
  raises `review_export_context_pack_changed`.
- When the context has `transcript_edit.chapters` and the candidate provides a non-None
  `source_hashes` or `source_files`, a mismatch with the current chapter binding raises.
- The resolved result records `bundle_dir` when a root is resolvable, so it
  contains a real runtime path.

## Error behavior

- Missing or empty required identity: `ValueError` with the existing Chinese
  missing-identity message.
- Non-Beijing `time_zone`: `ValueError` with the existing Beijing-time message.
- Non-object claimed configuration: `ValueError` (“项目导出配置必须是JSON对象。”).
- Stale pack or chapter-source binding: `ValueError` with the existing messages.
- Nonexistent/non-directory operator root in the CLI: `ValueError`; invalid JSON
  in an existing configuration file propagates the reader's JSON error.

## Verification status

The standalone test file is designed to run against the public package checkout
where `storage.py` and `models.py` are present.

During verification in the authoring environment, the synthetic run was stopped
by a filesystem `PermissionError` (WinError 5) when pytest accessed its temporary
directory. The run remained incomplete and unverified; no passing-suite claim is
made. The test command below is informational for operators in a separately authorized, suitable
environment and does not imply prior verification:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_review_export_identity_standalone.py
```

Synthetic-only boundaries: no real approval, clinical, customer or business
outcome is encoded. Passing identity checks are not browser, media, end-to-end,
consumer-bridge or editor integration proof, and they are not a real human review.
This delta does not distribute a full targeted-review UI, npm/editor/vendor or
compiled assets; those remain separate and are not claimed done.
The module records runtime paths; results are not sanitized public runtime output
and provide no OS sandbox guarantee. Consumers need explicit integration and
regeneration; current public pages do not adopt this context automatically.

## Provenance and license

Owner-authorized first-party generic source snapshot adapted from the current
feature, exact file SHA256
`e0a1073b04ad952f0dbad57a3f14a2416f6fbde36504e803a5fd8f9072b8c9b7`, and not
attributed to public baseline commit
`ab1bd2b9ac639a06ab1fbf22467b83ed4b1a69f5`. No source module was copied from a
third party. The root AGPL-3.0-only license governs distribution; this delta makes
no extra vendor-rights claim.
