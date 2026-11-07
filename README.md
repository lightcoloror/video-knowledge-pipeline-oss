# Video Knowledge Pipeline

Local-first video transcription, timestamped evidence, material manifests and
review tooling. The public source edition retains AGPL-3.0-only and the
component notices. It includes code and synthetic tests. Model weights,
third-party runtimes, private captures and machine configuration are excluded.

## Install and run the synthetic contract

```sh
git clone https://github.com/lightcoloror/video-knowledge-pipeline-oss.git
cd video-knowledge-pipeline-oss
python -m venv .venv
# Activate this environment using your shell's activation command.
python -m pip install ".[test]"
video-knowledge --help
video-knowledge-portable --project-root . doctor
video-knowledge-portable --project-root . smoke --output-dir .local/synthetic-smoke
video-knowledge-portable --project-root . validate-smoke --run-root .local/synthetic-smoke
python -m pytest -q tests/test_portable.py tests/test_quick_health.py tests/test_path_defaults.py
```

The explicit project root supplies the clone's discovery metadata, lock and
synthetic fixture to the installed Python code. The synthetic smoke checks
evidence and material contracts. It does not process real video or establish
ASR, OCR, speaker identification, model quality or publication readiness.

Optional extras configure local, VAD, online, evaluation and shared Gateway
integrations. The Gateway's uv source is pinned to the reviewed public
repository. External executables, providers and model licenses require their
own configuration. The source test collection includes optional integration
and deployment tests beyond the focused portable regression command above.

See [AGENT_DISCOVERY.md](AGENT_DISCOVERY.md), [LICENSE](LICENSE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES](LICENSES).
