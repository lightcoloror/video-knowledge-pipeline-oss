# Video Knowledge Pipeline discovery

Install the package in an isolated environment and provide the public clone
as the portable command's explicit `--project-root`.

```sh
video-knowledge-portable --project-root . doctor
video-knowledge-portable --project-root . smoke --output-dir .local/synthetic-smoke
video-knowledge-portable --project-root . validate-smoke --run-root .local/synthetic-smoke
```

These entries use bundled synthetic records. Model runtimes, real media,
credentials, uploads and publication have separate configuration and consent.
The discovery manifest grants no execution authority. Shared external contract
bundles are optional for metadata discovery and must match their recorded hash
when explicitly supplied. Optional runtime support has independent readiness.
