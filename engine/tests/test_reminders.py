"""Reminders — a due scheduled task nudges the user once, via the OnSchedule tick."""
from ken import db
from ken.contracts import Provenance, RenderableKind
from ken.conversation import bus
from ken.engine import _REMINDER_STALE_MS, Engine
from ken.memory import store


async def _eng(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    return Engine()


async def _nudges(text="call mom"):
    # a reminder is an ADVICE whose body flags it (type=reminder) with NO glyph — the
    # engine never formats the text; the interface draws the clock.
    return [r for _, r in await bus.read_outbox(bus.MAIN, 0)
            if r.kind is RenderableKind.ADVICE and r.body.get("type") == "reminder"
            and r.body.get("text") == text]


async def test_a_due_task_fires_a_reminder_exactly_once(tmp_path):
    eng = await _eng(tmp_path)
    await store.add_task("call mom", Provenance.AGENT_INFERRED, deadline=1000)
    await eng._run_reminders(2000)          # its time has passed -> fire
    await eng._run_reminders(3000)          # checked again -> must NOT re-fire
    assert len(await _nudges()) == 1


async def test_a_future_task_does_not_fire(tmp_path):
    eng = await _eng(tmp_path)
    await store.add_task("later", Provenance.AGENT_INFERRED, deadline=10_000)
    await eng._run_reminders(5_000)         # before its time
    assert await bus.read_outbox(bus.MAIN, 0) == []


async def test_a_stale_reminder_is_retired_not_fired(tmp_path):
    eng = await _eng(tmp_path)
    await store.add_task("old", Provenance.AGENT_INFERRED, deadline=1000)
    await eng._run_reminders(1000 + _REMINDER_STALE_MS + 60_000)   # hours late
    assert await bus.read_outbox(bus.MAIN, 0) == []                # no burst of old alarms
    assert await store.due_reminders(9_999_999_999) == []          # but marked, won't linger


async def test_a_stale_reminder_leaves_an_audit_row_not_a_silent_drop(tmp_path):
    eng = await _eng(tmp_path)
    await store.add_task("old", Provenance.AGENT_INFERRED, deadline=1000)
    await eng._run_reminders(1000 + _REMINDER_STALE_MS + 60_000)   # hours late
    async with db.get_db() as c:
        cur = await c.execute("SELECT ref FROM audit WHERE kind='reminder_stale'")
        rows = await cur.fetchall()
    assert len(rows) == 1                                          # the miss left a trail


async def test_a_fresh_reminder_body_carries_no_glyph(tmp_path):
    eng = await _eng(tmp_path)
    await store.add_task("call mom", Provenance.AGENT_INFERRED, deadline=1000)
    await eng._run_reminders(2000)
    (nudge,) = await _nudges()
    assert nudge.body["text"] == "call mom"        # the engine never formats the text
    assert "⏰" not in nudge.body["text"]           # no pre-formatted glyph
    assert nudge.body["at"] == 1000                # the scheduled time, for the interface


async def test_a_completed_task_is_never_reminded(tmp_path):
    eng = await _eng(tmp_path)
    tid, _ = await store.add_task("done one", Provenance.AGENT_INFERRED, deadline=1000)
    await store.complete_task(tid)
    await eng._run_reminders(2000)
    assert await bus.read_outbox(bus.MAIN, 0) == []


async def test_an_untimed_task_is_never_reminded(tmp_path):
    eng = await _eng(tmp_path)
    await store.add_task("no time", Provenance.AGENT_INFERRED)     # deadline None
    await eng._run_reminders(9_999_999_999)
    assert await bus.read_outbox(bus.MAIN, 0) == []


async def test_due_reminders_respects_the_per_tick_cap(tmp_path):
    await _eng(tmp_path)
    for i in range(3):
        await store.add_task(f"t{i}", Provenance.AGENT_INFERRED, deadline=1000)
    assert len(await store.due_reminders(2000, limit=2)) == 2      # capped; overflow next tick


async def test_migration_backfills_past_due_tasks_but_not_future_ones(tmp_path):
    from ken.util import now_ms
    db.configure(str(tmp_path / "ken.db"))
    now = now_ms()
    async with db.get_db() as c:                                   # simulate a pre-column DB
        await c.execute(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, text TEXT NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'open', deadline INTEGER, "
            "provenance TEXT NOT NULL, created_at INTEGER NOT NULL)")
        await c.execute("INSERT INTO tasks VALUES ('p','past','open',?, 'world', ?)", (now - 1000, now))
        await c.execute("INSERT INTO tasks VALUES ('f','future','open',?, 'world', ?)", (now + 10_000_000, now))
        await c.commit()
    await db.init_db()                                             # migration + one-time backfill
    async with db.get_db() as c:
        marked = {r["id"]: r["reminded_at"] for r in
                  await (await c.execute("SELECT id, reminded_at FROM tasks")).fetchall()}
    assert marked["p"] is not None      # already past-due at upgrade -> stamped, no surprise nudge
    assert marked["f"] is None          # future -> still fires at its time
