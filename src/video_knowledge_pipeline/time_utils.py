from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso_seconds() -> str:
    """Return the established aware UTC ISO timestamp at whole-second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def parse_utc_datetime_or_none(value: object) -> datetime | None:
    """Parse the established consent timestamp contract and normalize it to UTC."""
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
