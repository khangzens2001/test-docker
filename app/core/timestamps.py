from datetime import datetime, timezone


def to_rfc3339(dt: datetime | None) -> str | None:
    """Serialize a datetime to RFC 3339 with a `Z` UTC suffix.

    Naive datetimes are assumed to be UTC (SQLite stores TIMESTAMPs without tz).
    Never double-appends `Z` when isoformat() already emits an offset.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    s = dt.astimezone(timezone.utc).isoformat()
    if s.endswith("+00:00"):
        s = s[: -len("+00:00")] + "Z"
    return s
