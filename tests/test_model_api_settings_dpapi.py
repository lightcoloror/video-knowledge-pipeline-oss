from __future__ import annotations

import os

import pytest

from video_knowledge_pipeline.model_api_settings import _protect_secret, _unprotect_secret


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI is Windows-only")
def test_windows_dpapi_round_trip_does_not_embed_plaintext() -> None:
    secret = "vkp-dpapi-round-trip-secret"
    ciphertext = _protect_secret(secret)

    assert ciphertext
    assert secret not in ciphertext
    assert _unprotect_secret(ciphertext) == secret
