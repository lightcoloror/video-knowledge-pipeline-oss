from __future__ import annotations


GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
GEMINI_THROUGHPUT_MODEL = "gemini-3.5-flash-lite"


_GEMINI_FIXED_SAMPLING_MODELS = frozenset(
    {
        GEMINI_DEFAULT_MODEL,
        GEMINI_THROUGHPUT_MODEL,
    }
)


def gemini_omits_legacy_sampling_parameters(model: object) -> bool:
    """Return whether Google requires deprecated sampling fields to be omitted."""

    model_id = str(model or "").strip().lower().removeprefix("gemini/")
    return model_id in _GEMINI_FIXED_SAMPLING_MODELS
