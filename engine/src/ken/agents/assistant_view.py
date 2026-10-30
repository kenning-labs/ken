"""The Assistant's Today panel — the bridge: a deterministic standing view of the day.

Pure function over the open tasks, recent events, now, and (optionally) a rank score
per task. No LLM: the panel is instant and free. Emits the generic block spec
(note / timeline / lanes / agenda / bars) the engine stores opaquely and any interface
draws — nothing here is known to the engine or the CLI. The Assistant refreshes it
every run.

Lanes: overdue (timed, in the past) · today (timed today, or ranked "today") · this
week · someday. With no ranker every untimed task is "someday" — an honest floor, not
a guess.
"""
from __future__ import annotations

from datetime import datetime

from ..util import when_label

RUBRIC = ["someday — no rush, parked", "this week", "today", "now — urgent, do it next"]
RUBRIC_ASK = ("how soon does this task need doing? Judge from its own words (a date, "
              "'urgent', a person waiting) — not from how it was phrased.")
_MAX_CARDS = 2
_MAX_NAMES = 2
_TEXT = 60                        # a card/name never wraps: one line, cut with an ellipsis


def _short(text: str, n: int = _TEXT) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _names(tasks) -> str:
    names = [_short(t.text, 24) for t in tasks[:_MAX_NAMES]]
    more = f" · +{len(tasks) - _MAX_NAMES}" if len(tasks) > _MAX_NAMES else ""
    return " · ".join(names) + more


def _minute_of(ms: int) -> int:
    d = datetime.fromtimestamp(ms / 1000)  # noqa: DTZ006 — local wall clock, by design
    return d.hour * 60 + d.minute


def _due_label(t, now: int, *, overdue: bool) -> str:
    """The card's right-hand label: '⏰ Sat 6pm · 12d overdue', 'due Fri' for a date-only
    task, '' for an untimed one placed by rank."""
    if not t.deadline:
        return ""
    if _date_only(t.deadline):
        d = datetime.fromtimestamp(t.deadline / 1000)  # noqa: DTZ006 — local, by design
        label = f"due {d:%a}" if now - t.deadline < 6 * 86_400_000 else f"due {d:%a %b} {d.day}"
        return f"{label} · {_age(t.deadline + 86_400_000, now)}" if overdue else label
    label = f"⏰ {when_label(t.deadline, now)}"
    return f"{label} · {_age(t.deadline, now)}" if overdue else label


def _age(ms: int, now: int) -> str:
    d = (now - ms) // 86_400_000
    if d >= 1:
        return f"{d}d overdue"
    h = (now - ms) // 3_600_000
    return f"{h}h overdue" if h >= 1 else "just now"


def _date_only(ms: int) -> bool:
    """A task scheduled at local midnight carries a DAY, not a time (the date-only
    convention): it is due that day, not overdue from 00:00, and gets no clock mark."""
    d = datetime.fromtimestamp(ms / 1000)  # noqa: DTZ006 — local wall clock, by design
    return d.hour == 0 and d.minute == 0


def lanes_for(tasks, now: int, scores: list[float] | None = None) -> dict[str, list]:
    """Bucket the open tasks. A timed task is placed by its time (past → overdue, today →
    today, later → this week / someday by distance; a date-only task is due for its whole
    day); an untimed one by its rank score — `scores` aligned with the UNTIMED tasks in
    order — rounded to the nearest rubric step (2 today, 1 this week, else someday), or
    someday when there is no score."""
    dt = datetime.fromtimestamp(now / 1000)  # noqa: DTZ006 — local day, by design
    day_end = int(dt.replace(hour=23, minute=59, second=59).timestamp() * 1000)
    out = {"overdue": [], "today": [], "this week": [], "someday": []}
    it = iter(scores or [])
    for t in tasks:
        if t.deadline:
            due = t.deadline
            if _date_only(due):
                due = min(due + 86_400_000 - 1, int(datetime.fromtimestamp(due / 1000)  # noqa: DTZ006
                                                   .replace(hour=23, minute=59, second=59).timestamp() * 1000))
            if due < now:
                out["overdue"].append(t)
            elif due <= day_end:
                out["today"].append(t)
            elif due - now <= 7 * 86_400_000:
                out["this week"].append(t)
            else:
                out["someday"].append(t)
            continue
        step = round(next(it, 0.0) or 0.0)
        out["today" if step >= 2 else "this week" if step == 1 else "someday"].append(t)
    return out


def _headline(lanes: dict, n_open: int) -> str:
    od, td = len(lanes["overdue"]), len(lanes["today"])
    if od and td:
        return f"{od} overdue, {td} for today."
    if od:
        return f"{od} {'thing is' if od == 1 else 'things are'} overdue. Nothing for today."
    if td:
        return f"{td} {'thing' if td == 1 else 'things'} for today."
    if n_open:
        return "Nothing for today."
    return "Nothing open. Clear."


def build_view(tasks, now: int, *, events=(), done=(), scores: list[float] | None = None,
               ranked_at: int | None = None) -> dict:
    """The bridge: a headline sentence, the day strip, four lanes, cards for what needs
    doing first, a bar for what happened this week, one nudge."""
    dt = datetime.fromtimestamp(now / 1000)  # noqa: DTZ006 — the box's local day, by design
    lanes = lanes_for(tasks, now, scores)
    week_ago = now - 7 * 86_400_000
    happened = [e for e in events if getattr(e, "at", 0) >= week_ago]
    done = [t for t in done if (getattr(t, "completed_at", None) or 0) >= week_ago]
    tail = f"{len(tasks)} open"
    if done:
        tail += f" · {len(done)} done this week"
    if happened:
        tail += f" · {len(happened)} happened"
    blocks: list[dict] = [
        {"type": "note", "text": _headline(lanes, len(tasks)), "strong": True, "tail": tail},
        {"type": "timeline", "label": "day",
         "marks": [_minute_of(t.deadline) for t in lanes["today"]
                   if t.deadline and not _date_only(t.deadline)]},
        # `ids` are not drawn: they let another reader (mcp_server.today) reuse the ranking
        # kip published instead of re-bucketing without the scores.
        {"type": "lanes", "items": [
            {"label": label, "level": level, "count": len(lanes[label]), "names": _names(lanes[label]),
             "ids": [t.id for t in lanes[label]]}
            for label, level in (("overdue", "bad"), ("today", "high"),
                                 ("this week", "mid"), ("someday", "low"))]},
    ]
    first = lanes["overdue"] + lanes["today"]
    if first:
        blocks.append({"type": "agenda", "items": [
            {"edge": "bad" if t in lanes["overdue"] else "high", "text": _short(t.text),
             "when": _due_label(t, now, overdue=t in lanes["overdue"]),
             "timed": bool(t.deadline)}
            for t in first[:_MAX_CARDS]]})
    bars = []
    if done:
        bars.append({"label": "done ·7d", "level": "good", "count": len(done),
                     "tail": " · ".join(_short(t.text, 28) for t in done[:3])})
    if happened:
        bars.append({"label": "happened", "level": "mid", "count": len(happened),
                     "tail": " · ".join(_short(e.summary, 28) for e in happened[:3])})
    if bars:
        blocks.append({"type": "bars", "items": bars})
    od = len(lanes["overdue"])
    if od:
        nudge = f"ask kip to reschedule the {od} overdue, or tap done"
    elif lanes["today"]:
        nudge = f"next up: {_short(lanes['today'][0].text, 40)}"
    elif lanes["this week"] or lanes["someday"]:
        pick = (lanes["this week"] or lanes["someday"])[0]
        nudge = f"pick one for today: {_short(pick.text, 40)}?"
    else:
        nudge = ""
    if nudge:
        blocks.append({"type": "note", "text": nudge, "accent": True})
    sub = f"{dt:%a %b} {dt.day} · {dt:%H:%M}"
    if ranked_at:
        sub += f" · ranked {datetime.fromtimestamp(ranked_at / 1000):%H:%M}"  # noqa: DTZ006
    return {"title": "today", "subtitle": sub, "blocks": blocks}
