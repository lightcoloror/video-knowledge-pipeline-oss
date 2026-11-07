# Panorama synthetic fixtures

Panorama-v1 tests generate all media at runtime. This directory intentionally contains no real video, camera export, audio, face data, GPS data or model output.

The generated equirectangular fixture must contain:

- six stable, color-coded and text-labelled directions: front, right, back, left, up and down;
- one marker crossing the equirectangular left/right seam;
- one moving marker for temporal direction selection;
- one OCR label;
- optional synthetic tone/speech-like audio generated without network access.

Fixture requirements:

- deterministic bytes for a fixed generator version;
- source and derived SHA-256 values recorded by tests;
- no source-media overwrite;
- no online Provider or model call;
- generated files stay inside the test's explicit temporary root;
- projection tests use the existing FFmpeg outlet;
- py360convert may be used only as an optional independent oracle when its fixed source is explicitly available.

## Change record

- **Intent:** Test spherical geometry, seams and temporal selection without exposing real media.
- **Decision:** Generate small labelled fixtures during tests rather than committing binary video.
- **Reason:** Text-labelled directions make wrong projection or orientation visible and keep the repository small.
- **Evidence:** FFmpeg v360, py360convert and equilib synthetic smokes passed during planning.
- **Effective scope:** Panorama-only tests.
- **Rollback:** Remove this fixture contract and the generated test directories; no production or source media is affected.
