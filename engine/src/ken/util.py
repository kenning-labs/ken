"""Tiny shared helpers."""
from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta


def when_label(ms: int, now: int | None = None) -> str:
    """A compact human time on the box's local clock — 'Sat 3pm' / 'Sat 3:30pm'. Adds a
    date ('Sun Aug 24 3pm') when `now` is given and the target is more than ~a week away,
    so this Sunday and next Sunday don't collide."""
    dt = datetime.fromtimestamp(ms / 1000)  # noqa: DTZ006 — local wall clock, by design
    h = dt.hour % 12 or 12
    mins = f":{dt.minute:02d}" if dt.minute else ""
    time = f"{h}{mins}{'am' if dt.hour < 12 else 'pm'}"
    if now is not None and abs(ms - now) >= 6 * 86_400_000:
        return f"{dt:%a %b} {dt.day} {time}"
    return f"{dt:%a} {time}"


def norm(text: str) -> str:
    """A comparison key robust to trivial re-phrasings: collapse whitespace and
    unicode-lower (Python's lower folds Cyrillic/CJK; SQLite's is ASCII-only)."""
    return " ".join(text.split()).lower()


GLUE = frozenset({"for", "and", "the", "with", "from"})   # title words that name nothing


def title_of(kind: str) -> str:
    """'projects.recipes_in_app' -> 'recipes in app': the one name a record shows everywhere."""
    return kind.split(".", 1)[-1].replace("_", " ")


def owner_label(owner: str | None) -> str | None:
    """A task's owner as shown: None = the user; a person record by its title; else the name as given."""
    if not owner:
        return None
    return title_of(owner) if owner.startswith("people.") else owner


def family_words(titles: list[str]) -> set[str]:
    """Words that appear in three or more titles ('crumbly' in every Crumbly project)
    name a family, not a record: alone they never count as a match."""
    df = Counter(w for t in titles for w in {w for w in t.split() if len(w) > 2})
    return {w for w, n in df.items() if n >= 3}


def slug(text: str, *, fallback: str | None = None) -> str:
    """A stable key from free text: unicode word characters are kept (so a non-Latin
    topic — Cyrillic, CJK — does not collapse to nothing and collide), lowercased,
    runs of other characters joined by '_'. When the text has no word characters at
    all, returns `fallback` if given, else a short content hash so distinct wordless
    inputs still get distinct keys."""
    s = re.sub(r"\W+", "_", text.strip().lower()).strip("_")
    if s:
        return s
    return fallback or hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def local_ms(value: str) -> int:
    """A local ISO 8601 date or datetime ('2026-03-11', '2026-03-11T09:30') -> ms epoch.
    Naive = the box's local clock. Raises ValueError on anything else."""
    return int(datetime.fromisoformat(str(value).strip()).timestamp() * 1000)


def real_ms() -> int:
    """The wall clock, never frozen: liveness heartbeats and poll throttles only."""
    return int(time.time() * 1000)


def now_ms() -> int:
    """The engine's clock. `KEN_NOW` (a local ISO date/datetime) freezes it, read on every
    call so a harness can move it between steps: for tests and benchmarks only."""
    frozen = os.environ.get("KEN_NOW")
    return local_ms(frozen) if frozen else real_ms()


def new_id() -> str:
    return uuid.uuid4().hex


def stable_id(*parts: object) -> str:
    """A deterministic id from its parts. Re-delivering the same input yields the
    same id, so INSERT OR IGNORE can deduplicate."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:16]


def day_from_ms(ms: int) -> str:
    """UTC ISO date for a ms-epoch timestamp — a safe fallback when a connector
    does not stamp its own day."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).date().isoformat()


def ms_from_day(day: str) -> int:
    """Ms epoch for UTC midnight of an ISO date — the inverse of day_from_ms, so a
    connector that reports a `day` can stamp a stable event time for the cursor."""
    dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


# ── a deadline phrase in a task's text ──────────────────────────────────────
# ponytail: English phrases only, and only a clear "by/due/before + a date" (month-day, day-month,
# ISO, "the 14th", a bare weekday, today/tomorrow). "next Friday", "end of month", "in 2 weeks"
# and other languages stay in the text; add a pattern when a real task shows one.
_MONTHS = {m: i + 1 for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"))}
_MON = r"(?P<mon{n}>" + "|".join(sorted({m for full in _MONTHS for m in (full, full[:3], "sept")},
                                         key=len, reverse=True)) + r")\.?"
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_ORD = r"(?:st|nd|rd|th)?"
_DATE = (r"(?:" + _MON.format(n=1) + r"\s+(?P<d1>\d{1,2})" + _ORD + r"(?:,?\s+(?P<y1>\d{4}))?"
         r"|(?:the\s+)?(?P<d2>\d{1,2})" + _ORD + r"\s+(?:of\s+)?" + _MON.format(n=2) + r"(?:,?\s+(?P<y2>\d{4}))?"
         r"|(?P<iso>\d{4}-\d{2}-\d{2})"
         r"|the\s+(?P<d3>\d{1,2})(?:st|nd|rd|th)"
         r"|(?P<wd>" + "|".join(_WEEKDAYS) + r")"
         r"|(?P<rel>today|tomorrow))")
_TIME = r"(?:\s+at\s+(?P<h>\d{1,2})(?::(?P<mi>\d{2}))?\s*(?P<ap>am|pm)?)?"
_PHRASE = re.compile(r"(?:(?P<paren>\()\s*(?:due|by)|,?\s*\b(?:due(?:\s+(?:on|by))?|by|before))\s*:?\s+"
                     + _DATE + _TIME + r"\b(?(paren)\s*\)|(?:\.(?=\s*$))?)", re.IGNORECASE)


def _resolve(m: re.Match, base: datetime) -> str | None:
    """The match -> 'YYYY-MM-DD' (or with 'THH:MM'), resolved forward from `base`; None if not a real date."""
    g = m.groupdict()
    today = base.date()
    try:
        if g["iso"]:
            day = datetime.fromisoformat(g["iso"]).date()
        elif g["d3"]:
            day = today.replace(day=int(g["d3"]))
            if day < today:                                      # passed: the next month's
                day = (today.replace(day=1) + timedelta(days=32)).replace(day=int(g["d3"]))
        elif g["wd"]:
            day = today + timedelta(days=(_WEEKDAYS.index(g["wd"].lower()) - today.weekday()) % 7)
        elif g["rel"]:
            day = today + timedelta(days=g["rel"].lower() == "tomorrow")
        else:
            mon = (g["mon1"] or g["mon2"]).lower()
            month = _MONTHS.get(mon) or next(v for k, v in _MONTHS.items() if k.startswith(mon[:3]))
            year = g["y1"] or g["y2"]
            day = today.replace(year=int(year) if year else today.year, month=month, day=int(g["d1"] or g["d2"]))
            if not year and day < today:                         # passed this year: next year's
                day = day.replace(year=today.year + 1)
    except ValueError:                                           # Feb 30, the 31st of a short month
        return None
    if g["h"] is None:
        return day.isoformat()
    if not (g["ap"] or g["mi"]):                                 # "at 3": am or pm? leave it alone
        return None
    h = int(g["h"]) % 12 + (12 if (g["ap"] or "").lower() == "pm" else 0) if g["ap"] else int(g["h"])
    if h > 23 or int(g["mi"] or 0) > 59:
        return None
    return f"{day.isoformat()}T{h:02d}:{int(g['mi'] or 0):02d}"


def split_deadline(text: str, base: datetime) -> tuple[str, str | None]:
    """A task text -> (the text without its deadline phrase, the phrase's date as local ISO).
    'Order tiles by March 20, 2026' -> ('Order tiles', '2026-03-20'). No clear phrase, a date
    that does not exist, or nothing left after stripping -> (text unchanged, None)."""
    m = _PHRASE.search(text or "")
    date = _resolve(m, base) if m else None
    if date is None:
        return text, None
    clean = " ".join((text[:m.start()] + " " + text[m.end():]).split()).strip(" ,;:-")
    clean = re.sub(r"\s+([,.;:])", r"\1", clean)
    return (clean, date) if clean else (text, None)


def gone(at: int, now: int) -> bool:
    """A due time already past. A DAY (stored at local midnight, or 09:00 for the morning
    nudge) is past only when the whole day is: "by today" at 3pm still holds."""
    local = datetime.fromtimestamp(at / 1000)   # noqa: DTZ006 — local, by design
    if (local.hour, local.minute, local.second) in ((0, 0, 0), (9, 0, 0)):
        return int(local.replace(hour=23, minute=59, second=59).timestamp() * 1000) < now
    return at < now
