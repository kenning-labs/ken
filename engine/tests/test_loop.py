"""The walking skeleton: schedule -> agent -> advice/consent -> memory.

Stub agents on an every-minute cron exercise the whole loop end-to-end, in memory,
on a temp database. (An agent fetches its own data through tools; the loop only
wakes it.)
"""
import pytest

from ken import db
from ken.agents import Advice, Agent, OnMessage, OnSchedule, Proposal, Slice
from ken.capabilities import ScopeError
from ken.contracts import Evidence, Provenance
from ken.conversation import bus
from ken.engine import Engine
from ken.memory import store

EVERY_MINUTE = "* * * * *"      # fires on the first tick — the loop's "wake" for tests


class AdviceAgent(Agent):
    triggers = [OnSchedule(EVERY_MINUTE)]
    slice = Slice()
    readiness = 40                  # what its (imaginary) tool fetched

    async def run(self, ctx):
        if self.readiness < 50:
            return Advice(text=f"readiness {self.readiness} — ease off today")
        return None


class MaintainerAgent(Agent):
    triggers = [OnSchedule(EVERY_MINUTE)]
    slice = Slice(structured=["patterns"])

    async def run(self, ctx):
        return Proposal(intent="structured:patterns.recovery",
                        content="readiness dips on high-load days",
                        evidence=[Evidence(ref="readiness-2026-08-10", source="tracker")])


class GoalAgent(Agent):
    triggers = [OnSchedule(EVERY_MINUTE)]
    slice = Slice()

    async def run(self, ctx):
        return Proposal(intent="structured:goals.race",
                        content="switch focus to recovery for now?")


class ProbeAgent(Agent):
    """Captures the Context the engine hands it — so a test can assert the engine
    actually injected the slice-scoped capabilities (memory/propose/view) itself,
    not just that a hand-built Context works."""
    triggers = [OnMessage()]
    slice = Slice(structured=["preferences"], tasks="open")
    session = "probe-tab"

    def __init__(self) -> None:
        self.seen = None

    async def run(self, ctx):
        self.seen = ctx


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


async def test_engine_injects_slice_scoped_capabilities(tmp_path):
    # the wiring itself — a regression (wrong slice, cid/session swap, a dropped
    # capability) would leave every hand-built-Context test green, so assert against
    # the REAL _assemble/run_agent. Expected slice/session/author are DERIVED from the
    # agent, never hardcoded.
    await _setup(tmp_path)
    eng = Engine()
    probe = ProbeAgent()
    eng.register_agent(probe)
    await store.add_task("live task", Provenance.AGENT_INFERRED)
    # answer in a DIFFERENT conversation than the agent owns, to prove propose uses the
    # reply cid while view uses the agent's OWN session (the cid-vs-session distinction).
    await eng.run_agent(probe, message="hi", conversation_id="reply-tab")
    ctx = probe.seen

    assert ctx.memory is not None and ctx.propose is not None and ctx.view is not None
    # memory is LIVE and scoped to the probe's declared slice
    assert [t.text for t in await ctx.memory.open_tasks()] == ["live task"]
    await ctx.memory.structured("preferences")             # granted -> no raise
    with pytest.raises(ScopeError):
        await ctx.memory.structured("facts")               # NOT in the slice -> loud
    # the proposer carries the reply conversation + the agent's own name
    assert ctx.propose._cid == "reply-tab"
    assert ctx.propose._author == type(probe).__name__
    # the view publisher is scoped to the agent's OWN session, never the reply tab
    assert ctx.view._session == probe.session
    await ctx.view.publish({"title": "t", "blocks": []})
    async with db.get_db() as c:
        rows = await (await c.execute("SELECT conversation_id FROM views")).fetchall()
    assert [r["conversation_id"] for r in rows] == [probe.session]   # own tab, not reply-tab


async def test_assemble_fills_context_events_from_the_slice(tmp_path):
    # a slice that grants events -> the engine preloads Context.events; the default
    # (no grant) leaves it empty. Assert against the REAL _assemble via run_agent.
    await _setup(tmp_path)

    class EventProbe(Agent):
        triggers = [OnMessage()]
        slice = Slice(events="recent")
        session = "ev-tab"

        def __init__(self) -> None:
            self.seen = None

        async def run(self, ctx):
            self.seen = ctx

    eng = Engine()
    probe = EventProbe()
    eng.register_agent(probe)
    await gate_apply_event("saw the dentist")
    await eng.run_agent(probe, message="hi", conversation_id="ev-tab")
    assert [e.summary for e in probe.seen.events] == ["saw the dentist"]   # preloaded


async def gate_apply_event(summary):
    from ken.consent import gate
    await gate.apply(Proposal(intent="event", content=summary),
                     conversation_id=bus.MAIN, author="X")


async def test_loop_schedule_to_advice(tmp_path):
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(AdviceAgent())

    await eng.tick()

    out = await bus.read_outbox(bus.MAIN, 0)
    assert len(out) == 1
    _, r = out[0]
    assert r.kind.value == "advice" and "ease off" in r.body["text"]


class WatcherAgent(Agent):
    # a proactive drop-in with its OWN session — its advice must still surface in the
    # one visible stream (bus.MAIN), not vanish into a tab the v1 CLI never shows.
    name = "watcher"
    session = "watcher"
    triggers = [OnSchedule(EVERY_MINUTE)]
    slice = Slice()

    async def run(self, ctx):
        return Advice(text="watcher saw readiness 40")


async def test_proactive_agent_with_own_session_surfaces_in_main(tmp_path):
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(WatcherAgent())

    await eng.tick()

    # the wake landed in MAIN (visible), NOT the agent's own "watcher" session
    main_out = await bus.read_outbox(bus.MAIN, 0)
    assert any(r.author == "watcher" and "watcher saw readiness 40" in r.body["text"]
               for _, r in main_out)
    assert await bus.read_outbox("watcher", 0) == []      # nothing hidden in its own tab


async def test_maintenance_proposal_becomes_structured(tmp_path):
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(MaintainerAgent())

    await eng.tick()

    entries = await store.get_structured("patterns.")
    assert len(entries) == 1
    e = entries[0]
    assert e.kind == "patterns.recovery"
    assert e.evidence and e.evidence[0].source == "tracker"


async def test_high_stakes_proposal_asks_instead_of_writing(tmp_path):
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(GoalAgent())

    await eng.tick()

    assert await store.get_structured("goals.") == []          # not applied
    out = await bus.read_outbox(bus.MAIN, 0)
    assert any(r.kind.value == "ask" for _, r in out)          # asked instead
