"""The maintainer — what the engine does while nobody is talking to it.

Two passes, each a card, never a silent change (everything goes through the gate):

  close the loop   a task that was REPORTED on (an event cites it) but is still open,
                   overdue or untimed, LOOP_DAYS after the last report → "done?" — a Yes
                   marks it done (act:mark_done, via kip), a No leaves it open
  stale            a task overdue by STALE_DAYS+ with no report → "drop it?" — a Yes
                   drops it, not counted as done (act:discard, via kip); a No keeps it

A No is remembered: the stamp per (task, due time | last report) means the same question
is never asked twice; a Yes moves the task or closes it, so the stamp stops matching.
Expiry needs no pass: an event's `until` hides it from reads once passed (never deleted).
Each pass writes ONE audit row (`maintain`) with what it did. Runs once a day from the
engine tick (after MAINT_HOUR local) and on demand with `ken maintain`.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from . import db as _db
from .agents.base import Proposal
from .consent import gate
from .contracts import Evidence, Provenance
from .conversation import bus
from .memory import store
from .util import now_ms

STALE_DAYS = 7
LOOP_DAYS = 2            # a report this old on a still-open task: ask "done?"
MAINT_HOUR = 9           # the daily pass runs on the first tick after 09:00 local
_DAY = 86_400_000


def tomorrow_at(now: int, hour: int = 9) -> int:
    """Tomorrow at `hour`:00 local, in ms."""
    d = datetime.fromtimestamp(now / 1000) + timedelta(days=1)  # noqa: DTZ006 — local, by design
    return int(d.replace(hour=hour, minute=0, second=0, microsecond=0).timestamp() * 1000)


async def _open_question(task_id: str):
    """The maintainer's open card on a task, if any (its ask id and kind). One question per
    task at a time: the card's meta carries the task id, which is how the client finds it too."""
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT ask_id, kind, meta FROM pending WHERE author='kip' AND ask_type='act' AND meta LIKE ?",
            (f'%"task_id": "{task_id}"%',))
        return await cur.fetchone()


async def _ask(t, *, intent: str, content: str, context: str, stamp: str, value: str,
               conversation_id: str, extra: dict | None = None) -> bool:
    """Raise one maintainer card on a task through the gate, and stamp it so the same
    question is never asked twice (the stamp holds while the card is open and after a No;
    a Yes changes the task, so the value stops matching). Returns True when a card went up."""
    if await store.get_meta(stamp) == value:
        return False
    out = await gate.apply(Proposal(
        intent=intent, content=content, provenance=Provenance.AGENT_INFERRED,
        evidence=[Evidence(ref=t.id, source="maintainer")],
        meta={"task_id": t.id, "via": "kip", "to": t.text, "context": context, **(extra or {})}),
        conversation_id=conversation_id, author="kip")
    if out != "pending":
        return False
    await store.set_meta(stamp, value)
    return True


async def close_the_loop(tasks, now: int, *, conversation_id: str = bus.MAIN) -> int:
    """One "done?" card per open task whose LAST report (by insert order) is LOOP_DAYS+ old
    while the task is overdue or untimed (a report on a task with a future due date is
    progress, not closure). Asked once per (task, report); a newer question supersedes an
    open "still real?" card on the same task, so a task never carries two."""
    raised = 0
    for t in tasks:
        if t.deadline and t.deadline > now:
            continue
        rep = await store.latest_report(t.id)
        if rep is None or now - rep.at < LOOP_DAYS * _DAY:
            continue
        stamp, value = f"maint:loop:{t.id}", rep.id
        if await store.get_meta(stamp) == value:
            continue
        open_q = await _open_question(t.id)
        stale_q = open_q is not None and '"origin": "maintainer"' in (open_q["meta"] or "")
        if open_q is not None and open_q["kind"].startswith("act:discard:") and not stale_q:
            continue                                   # a question to close this task is already open
        if stale_q or (open_q is not None and open_q["kind"].startswith("act:reschedule:")):
            # it was reported on: "done?" is the better question than "drop it?"
            await gate.withdraw(open_q["ask_id"], reason="superseded by a done? card")
        days = (now - rep.at) // _DAY
        raised += await _ask(
            t, intent="act:mark_done", content=f"{t.text} — done?",
            context=f"You reported on it {days} days ago: \u201c{rep.summary}\u201d. Yes marks it done, No leaves it open.",
            stamp=stamp, value=value, conversation_id=conversation_id)
    return raised


async def stale_cards(tasks, now: int, *, conversation_id: str = bus.MAIN) -> int:
    """One "still real?" card per task overdue by STALE_DAYS+ that carries no open question
    already. Asked once per (task, due time)."""
    raised = 0
    for t in tasks:
        if not t.deadline or now - t.deadline < STALE_DAYS * _DAY:
            continue
        if await _open_question(t.id) is not None:
            continue
        # a week overdue with no news is most often a task that stopped mattering: ask to drop
        # it (one tap clears the list), never to push it to tomorrow, which keeps the clutter
        days = (now - t.deadline) // _DAY
        raised += await _ask(
            t, intent="act:discard", content=f"{t.text} — {days} days overdue, drop it?",
            context=f"{days} days overdue, no news on it. Yes drops it (not counted as done), No keeps it on your list.",
            stamp=f"maint:asked:{t.id}", value=str(t.deadline), conversation_id=conversation_id,
            extra={"reason": f"stale: {days} days overdue", "origin": "maintainer"})
    return raised


async def run_once(now: int | None = None) -> dict:
    """Every maintenance pass, on demand (`ken maintain`): loop cards first, then stale
    cards for the rest; ONE audit row with the counts, and the expired-events gauge."""
    now = now or now_ms()
    tasks = await store.open_tasks()          # ponytail: one LIKE scan per overdue/untimed task; index if it hurts
    loop = await close_the_loop(tasks, now)
    stale = await stale_cards(tasks, now)
    done = {"loop_cards": loop, "stale_cards": stale, "expired_events": await store.expired_events(now)}
    await _db.audit("maintain", "pass", json.dumps(done))
    return done


async def run_scheduled(now: int | None = None) -> int:
    """The daily pass: once per local day, on the first tick at or after MAINT_HOUR.
    Idempotent across ticks and restarts (the day stamp lives in the store)."""
    now = now or now_ms()
    local = datetime.fromtimestamp(now / 1000)  # noqa: DTZ006 — local, by design
    if local.hour < MAINT_HOUR:
        return 0
    stamp = local.strftime("%Y-%m-%d")
    if await store.get_meta("maint:day") == stamp:
        return 0
    done = await run_once(now)              # stamp only after a pass that completed:
    await store.set_meta("maint:day", stamp)  # a failure retries on the next tick
    return done["loop_cards"] + done["stale_cards"]
