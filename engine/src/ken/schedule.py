"""Minimal cron matching for OnSchedule triggers — no dependency. (AGENTS.md)

Standard 5-field cron: minute hour day-of-month month day-of-week. Each field is
'*', a number, a comma list, or '*/step'. Day-of-week is 0-6 with 0=Sunday (7 also
means Sunday). Minute resolution — the engine checks at most once a minute.
"""
from __future__ import annotations

from datetime import datetime


def _field(spec: str, value: int, *, floor: int = 0) -> bool:
    """`floor` is the field's minimum (0 for minute/hour/dow, 1 for day/month), so
    '*/step' counts from the field's start as standard cron does — '*/15' on
    day-of-month is 1,16,31, not 15,30."""
    for part in spec.split(","):
        if part == "*":
            return True
        if part.startswith("*/"):
            step = int(part[2:])
            if step and (value - floor) % step == 0:
                return True
        elif part.isdigit() and int(part) == value:
            return True
    return False


def cron_matches(cron: str, dt: datetime) -> bool:
    """Does `dt` (minute resolution) fall on an occurrence of `cron`?"""
    fields = cron.split()
    if len(fields) != 5:
        raise ValueError(f"cron needs 5 fields, got {cron!r}")
    minute, hour, dom, month, dow = fields
    sun0 = (dt.weekday() + 1) % 7          # Python Mon=0..Sun=6 -> cron Sun=0..Sat=6
    dow_ok = _field(dow, sun0) or (sun0 == 0 and _field(dow, 7))   # 7 == Sunday too
    return (_field(minute, dt.minute) and _field(hour, dt.hour)
            and _field(dom, dt.day, floor=1) and _field(month, dt.month, floor=1)
            and dow_ok)
