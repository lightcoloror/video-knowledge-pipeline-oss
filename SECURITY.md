# Security policy

## Supported version

Security fixes are applied to the current `master` branch. Historical commits, local private-history branches, generated Bundles, model caches, and operator-managed services are not supported release artifacts.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting page when available:

https://github.com/lightcoloror/video-knowledge-pipeline/security/advisories/new

Do not include API keys, customer recordings, transcripts, health/insurance records, cookies, credentials, or exploit data in a public issue. If private reporting is unavailable, open a minimal public issue asking the maintainer to establish a private channel; do not disclose the vulnerability details there.

A useful report includes the affected commit, component, reproducible steps using synthetic data, security impact, and a proposed mitigation. Remove secrets and personal data before attaching logs.

## Security boundaries

- Provider credentials are read from environment/DPAPI-backed local configuration and must never enter commits, reports, MCP arguments, or logs.
- Remote model execution requires an explicit route, destination allowlist, exact artifact hashes, consent/business authorization, and call/cost limits. A Secure MCP Tunnel does not enlarge those permissions.
- VKP does not silently fall back between local and remote execution locations.
- Model weights, media, transcripts, review notes, runtime output, and `.local` state are local/operator data and are excluded from the public repository.
- Sample reports must use synthetic or de-identified content. Do not submit customer, health, family, insurance, or organizational recordings as fixtures.

## Dependency and model responsibility

Third-party packages, independently installed tools, model weights, datasets, and hosted APIs keep their own security and license lifecycle. Review `THIRD_PARTY_NOTICES.md`, pin exact versions where supported, verify artifact hashes, and apply upstream security updates deliberately rather than enabling an automatic cross-provider fallback.
