# Upstream provenance

- Project: `Moyf/moys-asr-workflow`
- Version: `v1.3.1`
- Commit: `949bc84058cdae1d9c021c50203e6d2742f9392c`
- License: `AGPL-3.0-only`
- Imported scope: the eight root files from upstream `web/` only.
- Excluded scope: launcher, transcription providers, API-key UI, desktop shell,
  server implementation, FFmpeg orchestration, and model code.

VKP keeps the upstream editor assets intact except for two explicit template
slots (`__VKP_ADAPTER_CSS__`, `__VKP_ADAPTER_JS__`). VKP behavior is layered in
`vkp-adapter.css` and `vkp-adapter.js` so the upstream algorithms remain easy to
audit and refresh.
