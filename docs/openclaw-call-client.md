# OpenClaw Video Knowledge Pipeline Call Client

`docs/openclaw-call-client.md` documents the optional, source-only client script in `examples/openclaw/openclaw_video_knowledge_call.py`, licensed under AGPL-3.0-only. The script uses the Python 3 standard library exclusively, requiring no external packages.

## Configuration and Usage

The client formats and dispatches structured requests for the Video Knowledge Pipeline `/call` endpoint across 11 supported commands (`status`, `contract`, `live-smoke`, `plan`, `ingest`, `link`, `handoff`, `ingest-handoff`, `content-status`, `batch-content-status`, and `content-handoff-pack`).

Operators configure connection endpoints and mount mappings via CLI arguments or environment variables:
- **API Base**: `--api-base` (defaults to `VKP_API_BASE` or `http://host.docker.internal:8931`)
- **Mount Roots**: `--mount-container-root` (`VKP_CONTAINER_ROOT`) and `--mount-host-root` (`VKP_HOST_ROOT`)

Preview constructed JSON payloads locally without issuing network requests:

```bash
python examples/openclaw/openclaw_video_knowledge_call.py plan "Synthetic outline" --print-payload
```

## Security and Verification Boundaries

- **Lexical Mapping**: `translate_container_path` performs basic string prefix substitution between container and host roots. It does not inspect path traversal (`..`), enforce jail containment, or validate filesystem permissions.
- **Payload Declarations**: Flags such as `--execute`, `--allow-download`, and `--confirm-sensitive` declare payload parameters only; they do not authenticate user identity or grant execution authorization.
- **Operational Impact**: Without `--print-payload`, the client executes HTTP `POST` requests that may trigger server-side media processing or file writes; it is not read-only.

Owner acceptance directly invoked all 11 unchanged plain functions in `tests/test_openclaw_client.py`, with all original assertions; all passed under existing CPython 3.13.5. This was not pytest discovery or a whole-suite run. The missing-source baseline failed with FileNotFoundError as expected. A separate `main(... --print-payload)` check emitted the expected JSON without calling a mocked `post_call`; omitting that flag reached the blocked mock control. Network/subprocess audit guards were active, and no real HTTP request, media operation or download was performed. Live HTTP/error handling, server-side task execution, Docker mounts, native consent, installed-wheel packaging, other platforms and the complete workflow remain unverified. Unknown paths are returned unchanged by the lexical mapper; server-side policy remains responsible for authorization and filesystem constraints.
