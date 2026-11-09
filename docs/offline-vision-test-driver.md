# Offline Vision Test Driver

`docs/offline-vision-test-driver.md` documents the first-party offline test driver script under AGPL-3.0-only in the Video Knowledge Pipeline (VKP). It executes targeted vision-executor and vision-session regressions with declared in-process model, subprocess and socket-event guards. Their runtime enforcement was not exercised in this batch.

## Invocation and Positional Arguments

The example below is not an executed test result. Run from the source checkout using an already prepared Python environment and the project's existing test dependencies; no dependency, service or model is installed or started by this publication.

The driver requires an uncreated output directory as its first argument (`sys.argv[1]`). It does not use `argparse` or provide a `--help` flag; omitting the argument raises `IndexError`, and passing an existing directory fails with `FileExistsError`. Additional arguments are forwarded directly to `pytest`:

```bash
python -B scripts/run-codex-vision-offline-tests.py "NEW_OUTPUT_DIRECTORY" -k "test_preview_is_offline"
```

The script targets `tests/test_codex_vision_executor.py` and `tests/test_codex_vision_session.py` by default. Forwarded arguments can change selection, options and plugin loading, not merely narrow the default files. They must be trusted operator input; this is not an immutable test allowlist or a security boundary.

## In-Process Guards and Process Isolation

Before invoking `pytest.main`, the driver configures process-level restrictions:

- **Environment & Host Stubs**: Sets `HF_HUB_OFFLINE='1'`, `TRANSFORMERS_OFFLINE='1'`, disables bytecode caching, and patches `platform.node` to return `'offline-test-host'` within the process.
- **Model Import Blocking**: A custom `MetaPathFinder` (`NoModels`) intercepts and rejects imports of `torch`, `transformers`, `funasr`, `onnxruntime`, and `faster_whisper`. This finder does not cover already-loaded modules or all model libraries/native code; `pytest` itself is imported before these driver hooks. Offline environment flags alone do not prove complete offline isolation.
- **Subprocess Denial**: An audit hook blocks `subprocess.Popen`, `os.system`, and `os.posix_spawn`.
- **Network Interception**: Loads `evidence_offline_guard`, declaring a socket audit hook for the listed CPython network audit events while allowing only the exact Windows stdlib `_fallback_socketpair` loopback pair for internal asyncio IPC.

## Operational Boundaries

The driver creates the output directory, writes `--junitxml` test reports, and allocates a scratch directory (`--basetemp`) for synthetic test files; it is not read-only. In-process audit hooks and import finders do not constitute operating-system sandboxing, container isolation, or kernel network namespaces. Syntax was verified via AST parsing without test execution; live model availability and full test suite passes remain unverified.
