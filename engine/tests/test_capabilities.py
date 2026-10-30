"""The injected capabilities — the engine's slice-scoped seam onto memory.

ctx.memory is READ-ONLY, always LIVE, and slice-scoped (out-of-scope RAISES, never a
silent []). ctx.propose is the WRITE path through the gate (complete/reschedule are
fast-track but AUDITED). ctx.view publishes the standing panel. No network — a real
temp DB, no LLM.
"""
import json

import pytest

from ken import db
from ken.agents.base import Slice
from ken.capabilities import MemoryReader, Proposer, ScopeError, ViewPublisher
from ken.contracts import Provenance
from ken.memory import store


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


# ── ctx.memory: live, slice-scoped, loud out-of-scope ────────────────────────

async def test_memory_open_tasks_is_live(tmp_path):
    await _setup(tmp_path)
    mem = MemoryReader(Slice(tasks="open"))
    assert await mem.open_tasks() == []                 # live: nothing yet
    await store.add_task("call mom", Provenance.AGENT_INFERRED)
    assert [t.text for t in await mem.open_tasks()] == ["call mom"]   # sees the new task NOW


async def test_memory_structured_reads_a_granted_namespace(tmp_path):
    await _setup(tmp_path)
    await store.upsert_structured(
        store.StructuredEntry(id="preferences.coffee", kind="preferences.coffee",
                              subject="you", content="black", provenance=Provenance.USER_INSTRUCTED),
        change_note="seed")
    mem = MemoryReader(Slice(structured=["preferences"]))
    got = await mem.structured("preferences")
    assert [e.content for e in got] == ["black"]


async def test_memory_out_of_scope_structured_raises(tmp_path):
    await _setup(tmp_path)
    mem = MemoryReader(Slice(structured=["preferences"]))   # facts NOT granted
    with pytest.raises(ScopeError):
        await mem.structured("facts")                       # loud, never a silent []


async def test_memory_tasks_without_the_scope_raises(tmp_path):
    await _setup(tmp_path)
    mem = MemoryReader(Slice(structured=["preferences"]))    # no tasks grant
    with pytest.raises(ScopeError):
        await mem.open_tasks()
    with pytest.raises(ScopeError):
        await mem.find_tasks("x")


async def test_memory_find_tasks_reaches_completed(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("book the dentist", Provenance.AGENT_INFERRED)
    await store.complete_task(tid)                          # now done — gone from open_tasks
    mem = MemoryReader(Slice(tasks="open"))
    assert await mem.open_tasks() == []                     # the open view cannot see it
    hits = await mem.find_tasks("dentist", include_done=True)
    assert [t.text for t in hits] == ["book the dentist"] and hits[0].status == "done"


async def _log_event(summary, at=1000):
    from ken.agents import Proposal
    from ken.consent import gate
    await gate.apply(Proposal(intent="event", content=summary, at=at),
                     conversation_id="main", author="Assistant")


async def test_memory_recent_events_is_live(tmp_path):
    await _setup(tmp_path)
    mem = MemoryReader(Slice(events="recent"))
    assert await mem.recent_events() == []                   # live: nothing yet
    await _log_event("ran 10k")
    assert [e.summary for e in await mem.recent_events()] == ["ran 10k"]   # sees it NOW


async def test_memory_find_events_searches_the_log(tmp_path):
    await _setup(tmp_path)
    await _log_event("had lunch with Sarah", at=1)
    await _log_event("finished the report", at=2)
    mem = MemoryReader(Slice(events="recent"))
    hits = await mem.find_events("lunch")
    assert [e.summary for e in hits] == ["had lunch with Sarah"]   # substring match
    assert await mem.find_events("nothing here") == []             # honest empty


async def test_memory_events_without_the_scope_raises(tmp_path):
    await _setup(tmp_path)
    mem = MemoryReader(Slice(tasks="open"))                  # no events grant
    with pytest.raises(ScopeError):
        await mem.recent_events()
    with pytest.raises(ScopeError):
        await mem.find_events("x")


# ── ctx.propose: the write path through the gate, audited ────────────────────

async def _agent_log(action_prefix):
    async with db.get_db() as c:
        rows = await (await c.execute(
            "SELECT agent, action, target FROM agent_log ORDER BY at")).fetchall()
    return [(r["agent"], r["action"], r["target"]) for r in rows
            if r["action"].startswith(action_prefix)]


async def test_propose_complete_goes_through_the_gate_and_is_audited(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("water plants", Provenance.AGENT_INFERRED)
    prop = Proposer(conversation_id="main", author="Assistant")
    assert await prop.complete(tid) == "completed"
    assert await store.open_tasks() == []                   # applied by the gate
    assert await _agent_log("complete") == [("Assistant", "completed", tid)]   # audited


async def test_propose_complete_missing_task_is_task_absent(tmp_path):
    await _setup(tmp_path)
    prop = Proposer(conversation_id="main", author="Assistant")
    assert await prop.complete("no-such-id") == "task_absent"


_FUTURE = 9_999_999_999_999                                # ~year 2286 — safely ahead of now


async def test_propose_reschedule_goes_through_the_gate_and_is_audited(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("ring the plumber", Provenance.AGENT_INFERRED)
    await store.mark_reminded(tid, 1)                       # an old reminder is set
    prop = Proposer(conversation_id="main", author="Assistant")
    assert await prop.reschedule(tid, _FUTURE) == "rescheduled"
    task = (await store.open_tasks())[0]
    assert task.deadline == _FUTURE                    # re-timed by the gate
    async with db.get_db() as c:
        row = await (await c.execute(
            "SELECT reminded_at FROM tasks WHERE id=?", (tid,))).fetchone()
    assert row["reminded_at"] is None                       # cleared -> a fresh nudge can fire
    assert await _agent_log("reschedule") == [("Assistant", "rescheduled", tid)]   # audited


async def test_gate_reschedule_refuses_a_past_time(tmp_path):
    # defense-in-depth: the past-time floor lives in the gate, not only the Assistant
    # closure — so ANY proposer is refused, and the dead schedule is never applied.
    await _setup(tmp_path)
    tid, _ = await store.add_task("ring the plumber", Provenance.AGENT_INFERRED,
                                  deadline=9_000)
    prop = Proposer(conversation_id="main", author="Assistant")
    assert await prop.reschedule(tid, 1) == "reschedule_invalid"     # 1 ms epoch is long past
    assert (await store.open_tasks())[0].deadline == 9_000      # unchanged, not nulled


async def test_gate_reschedule_refuses_a_missing_time(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("ring the plumber", Provenance.AGENT_INFERRED,
                                  deadline=9_000)
    prop = Proposer(conversation_id="main", author="Assistant")
    assert await prop.reschedule(tid, None) == "reschedule_invalid"  # None would null the time
    assert (await store.open_tasks())[0].deadline == 9_000      # schedule survives


# ── ctx.view: publish the standing panel to a session ────────────────────────

async def test_view_publish_writes_the_panel(tmp_path):
    await _setup(tmp_path)
    view = ViewPublisher("main")
    await view.publish({"title": "Today", "blocks": [{"type": "divider"}]})
    async with db.get_db() as c:
        row = await (await c.execute(
            "SELECT body FROM views WHERE conversation_id='main'")).fetchone()
    assert json.loads(row["body"]) == {"title": "Today", "blocks": [{"type": "divider"}]}
