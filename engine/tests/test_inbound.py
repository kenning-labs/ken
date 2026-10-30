"""The inbound half — the pump reads the inbox and closes the loop.

  message          -> the author who spoke (anchor), or the assistant (bare);
  action_response  -> approve/reject a consent ask, or a validated internal tool;
  a bad action     -> handled and audited, never left to jam the inbox.

One test walks the full circle: a schedule wakes an agent, it asks, the user taps
yes, and the next tick applies the write.
"""
import json

from ken import db
from ken.agents import Advice, Agent, OnMessage, OnSchedule, Proposal, Slice
from ken.consent import gate
from ken.contracts import (
    Action,
    ActionKind,
    Evidence,
    Provenance,
    RenderableKind,
    StructuredEntry,
)
from ken.conversation import bus
from ken.engine import Engine
from ken.memory import store
from ken.util import new_id, now_ms


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


# ── stub agents ──────────────────────────────────────────────────────────────

class Other(Agent):
    triggers = [OnMessage()]
    slice = Slice()
    session = "other"          # its own tab
    last = None

    async def run(self, ctx):
        type(self).last = ctx.message
        return Advice(text=f"other heard: {ctx.message}")


class Assistant(Agent):
    triggers = [OnMessage()]
    slice = Slice()
    last = None

    async def run(self, ctx):
        type(self).last = ctx.message
        return Advice(text=f"assistant heard: {ctx.message}")


class Goal(Agent):
    """A proposing agent — records if a reply ever reaches it."""
    triggers = [OnMessage()]
    slice = Slice()
    last = None

    async def run(self, ctx):
        type(self).last = ctx.message


def _msg(text, *, anchor=None, session=bus.MAIN):
    return Action(id=new_id(), conversation_id=session, kind=ActionKind.MESSAGE,
                  channel="cli:local", text=text, anchor=anchor, at=now_ms())


def _resp(token):
    return Action(id=new_id(), conversation_id=bus.MAIN,
                  kind=ActionKind.ACTION_RESPONSE, channel="cli:local",
                  token=token, at=now_ms())


async def _asks_for(target):
    """Every ASK card shown for a target, oldest first (each carries its own id
    and Yes/No tokens)."""
    out = await bus.read_outbox(bus.MAIN, 0)
    return [r for _, r in out
            if r.kind is RenderableKind.ASK and r.body.get("target") == target]


async def _latest_ask(target):
    return (await _asks_for(target))[-1]


def _engine_with_agents():
    eng = Engine()
    eng.register_agent(Other())
    eng.register_agent(Assistant(), assistant=True)
    return eng


# ── message routing (address by session / tab) ───────────────────────────────

async def test_message_routes_to_its_session_owner(tmp_path):
    await _setup(tmp_path)
    Other.last = Assistant.last = None
    eng = _engine_with_agents()
    await bus.append_action(_msg("move my run to evening", session="other"))
    await eng.pump_inbox()

    assert Other.last == "move my run to evening"           # the other tab -> Other
    assert Assistant.last is None


async def test_bare_message_routes_to_the_assistant(tmp_path):
    await _setup(tmp_path)
    Other.last = Assistant.last = None
    eng = _engine_with_agents()
    await bus.append_action(_msg("what should I do today?"))
    await eng.pump_inbox()

    assert Assistant.last == "what should I do today?"       # the front door
    assert Other.last is None


async def test_message_in_an_unowned_session_falls_back_to_assistant(tmp_path):
    await _setup(tmp_path)
    Other.last = Assistant.last = None
    eng = _engine_with_agents()
    await bus.append_action(_msg("hello?", session="ghost-session"))
    await eng.pump_inbox()
    assert Assistant.last == "hello?"       # no agent owns it -> the front door
    assert Other.last is None


async def test_bare_message_without_assistant_is_audited(tmp_path):
    await _setup(tmp_path)
    eng = Engine()                                            # no assistant
    await bus.append_action(_msg("anyone there?"))
    await eng.pump_inbox()
    async with db.get_db() as c:
        cur = await c.execute("SELECT COUNT(*) FROM audit WHERE kind='no_route'")
        (n,) = await cur.fetchone()
    assert n == 1


async def test_seed_sessions_lists_each_agents_tab(tmp_path):
    await _setup(tmp_path)
    eng = _engine_with_agents()
    await eng.seed_sessions()
    async with db.get_db() as c:
        cur = await c.execute("SELECT id FROM conversations ORDER BY id")
        ids = {r["id"] for r in await cur.fetchall()}
    assert {"main", "other"} <= ids       # a tab exists for each agent, pre-conversation


async def test_misroute_replies_in_the_originating_session_and_audits(tmp_path):
    await _setup(tmp_path)
    eng = _engine_with_agents()
    await bus.append_action(_msg("hello?", session="ghost-session"))
    await eng.pump_inbox()

    # the fallback reply lands where the user spoke, not silently in "main"
    out = await bus.read_outbox("ghost-session", 0)
    assert any(r.kind is RenderableKind.ADVICE and "hello?" in r.body["text"]
               for _, r in out)
    assert await bus.read_outbox(bus.MAIN, 0) == []
    async with db.get_db() as c:
        cur = await c.execute("SELECT COUNT(*) FROM audit WHERE kind='misroute'")
        (n,) = await cur.fetchone()
    assert n == 1


async def test_two_agents_on_one_session_is_refused(tmp_path):
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(), assistant=True)      # owns "main"
    try:
        eng.register_agent(Goal())                       # also defaults to "main"
    except ValueError as e:
        assert "main" in str(e)
    else:
        raise AssertionError("expected a session collision to be refused")


async def test_seed_backfills_a_null_title(tmp_path):
    await _setup(tmp_path)
    await bus.ensure_conversation("main")                # a plain writer, title null
    eng = _engine_with_agents()
    await eng.seed_sessions()
    async with db.get_db() as c:
        cur = await c.execute("SELECT title FROM conversations WHERE id='main'")
        (title,) = await cur.fetchone()
    assert title == "Assistant"                          # seed filled the label


async def test_seed_orders_tabs_by_registration(tmp_path):
    await _setup(tmp_path)
    eng = _engine_with_agents()                          # Other first, Assistant second
    await eng.seed_sessions()
    async with db.get_db() as c:
        cur = await c.execute(
            "SELECT id FROM conversations WHERE parent_id IS NULL ORDER BY created_at")
        order = [r["id"] for r in await cur.fetchall()]
    assert order == ["other", "main"]                   # deterministic, no same-ms tie


# ── token dispatch (internal tools only) ─────────────────────────────────────

async def test_approve_applies_the_pending_write(tmp_path):
    await _setup(tmp_path)
    p = Proposal(intent="structured:goals.race",
                 content="switch focus to recovery",
                 provenance=Provenance.AGENT_INFERRED)
    assert await gate.apply(p, conversation_id=bus.MAIN, author="Goal") == "pending"
    assert await store.get_structured("goals.") == []        # not yet written
    card = await _latest_ask("goals.race")

    eng = Engine()
    await bus.append_action(_resp(card.actions[0].token))    # tap "Yes"
    await eng.pump_inbox()

    (entry,) = await store.get_structured("goals.")
    assert entry.content == "switch focus to recovery"
    assert entry.provenance is Provenance.USER_INSTRUCTED    # you authored the write


async def test_reject_drops_the_pending_write(tmp_path):
    await _setup(tmp_path)
    p = Proposal(intent="structured:goals.race", content="switch focus",
                 provenance=Provenance.AGENT_INFERRED)
    await gate.apply(p, conversation_id=bus.MAIN, author="Goal")
    card = await _latest_ask("goals.race")

    eng = Engine()
    await bus.append_action(_resp(card.actions[1].token))    # tap "No"
    await eng.pump_inbox()

    assert await store.get_structured("goals.") == []        # nothing written
    # the pending row is gone, so a second tap on the same ask finds nothing
    assert await gate.resolve(card.id, approved=True,
                              conversation_id=bus.MAIN) == "no_pending"


async def test_superseded_card_tap_is_a_safe_noop(tmp_path):
    await _setup(tmp_path)
    a = Proposal(intent="structured:goals.race", content="run a marathon",
                 provenance=Provenance.AGENT_INFERRED)
    await gate.apply(a, conversation_id=bus.MAIN, author="Goal")
    card_a = await _latest_ask("goals.race")

    b = Proposal(intent="structured:goals.race", content="run a 10k",
                 provenance=Provenance.AGENT_INFERRED)
    await gate.apply(b, conversation_id=bus.MAIN, author="Goal")
    card_b = await _latest_ask("goals.race")
    assert card_a.id != card_b.id                            # a fresh ask, a fresh token

    eng = Engine()
    # tapping the OLD card resolves a deleted ask -> nothing is written
    await bus.append_action(_resp(card_a.actions[0].token))
    await eng.pump_inbox()
    assert await store.get_structured("goals.") == []

    # tapping the CURRENT card writes the belief it actually showed
    await bus.append_action(_resp(card_b.actions[0].token))
    await eng.pump_inbox()
    (entry,) = await store.get_structured("goals.")
    assert entry.content == "run a 10k"


async def test_identical_reproposal_emits_one_card(tmp_path):
    await _setup(tmp_path)
    p = Proposal(intent="structured:goals.race", content="run a 10k",
                 provenance=Provenance.AGENT_INFERRED)
    await gate.apply(p, conversation_id=bus.MAIN, author="Goal")
    await gate.apply(p, conversation_id=bus.MAIN, author="Goal")   # same ask again
    assert len(await _asks_for("goals.race")) == 1           # not spammed


async def test_typed_reply_to_an_ask_reprompts_not_routes(tmp_path):
    await _setup(tmp_path)
    Goal.last = None
    p = Proposal(intent="structured:goals.race", content="run a 10k",
                 provenance=Provenance.AGENT_INFERRED)
    await gate.apply(p, conversation_id=bus.MAIN, author="Goal")
    card = await _latest_ask("goals.race")

    eng = Engine()
    eng.register_agent(Goal())                               # would receive a routed reply
    await bus.append_action(_msg("yes do it", anchor=card.id))
    await eng.pump_inbox()

    assert Goal.last is None                                 # not routed to the agent
    assert await store.get_structured("goals.") == []        # consent not silently applied
    reshown = await _asks_for("goals.race")
    assert len(reshown) == 2                                 # original + re-prompt
    assert reshown[-1].body.get("note") == "Please tap Yes or No."


async def test_typed_yes_resolves_the_ask_end_to_end(tmp_path):
    # the whole loop: an ask is open -> the user types "yes" (the CLI turns a bare
    # yes/number into this action_response, anchored to the card) -> the fact is
    # stored, a confirmation is posted so the spinner clears, and the pending row is gone
    await _setup(tmp_path)
    p = Proposal(intent="structured:facts.son", content="your son is Leo",
                 provenance=Provenance.AGENT_INFERRED)
    await gate.apply(p, conversation_id=bus.MAIN, author="Assistant")
    card = await _latest_ask("facts.son")

    eng = Engine()
    # the CLI's interpret("yes", card) yields exactly this: the approve token + anchor
    approve = next(a.token for a in card.actions if a.token.startswith("approve:"))
    await bus.append_action(Action(id=new_id(), conversation_id=bus.MAIN,
                                   kind=ActionKind.ACTION_RESPONSE, channel="cli:local",
                                   token=approve, anchor=card.id, at=now_ms()))
    await eng.pump_inbox()

    (entry,) = await store.get_structured("facts.")           # the fact is stored
    assert entry.content == "your son is Leo"
    assert entry.provenance is Provenance.USER_INSTRUCTED
    confirmations = [r.body.get("text") for _, r in await bus.read_outbox(bus.MAIN, 0)
                     if r.kind is RenderableKind.ADVICE]
    assert "Saved." in confirmations                          # the spinner can clear
    async with db.get_db() as c:
        (n,) = await (await c.execute("SELECT COUNT(*) FROM pending")).fetchone()
    assert n == 0                                              # the pending row is gone


async def test_replayed_approve_is_idempotent(tmp_path):
    await _setup(tmp_path)
    p = Proposal(intent="structured:goals.race", content="run a 10k",
                 provenance=Provenance.AGENT_INFERRED)
    await gate.apply(p, conversation_id=bus.MAIN, author="Goal")
    card = await _latest_ask("goals.race")
    ask_id = card.id

    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "applied"
    # a crash-replay (acted, not yet acked) re-runs the same approve
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "no_pending"

    async with db.get_db() as c:
        cur = await c.execute("SELECT changelog FROM structured WHERE id='goals.race'")
        row = await cur.fetchone()
    assert len(json.loads(row["changelog"])) == 1           # applied once, not twice


async def test_done_token_completes_a_task(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("call mom", Provenance.USER_STATED)
    eng = Engine()
    await bus.append_action(_resp(f"done:{tid}"))
    await eng.pump_inbox()
    assert await store.open_tasks() == []


async def test_forged_done_token_changes_nothing(tmp_path):
    await _setup(tmp_path)
    eng = Engine()
    await bus.append_action(_resp("done:does-not-exist"))
    await eng.pump_inbox()
    async with db.get_db() as c:
        cur = await c.execute("SELECT COUNT(*) FROM audit WHERE kind='token_noop'")
        (n,) = await cur.fetchone()
    assert n == 1                                             # audited, no invented task


async def test_unknown_token_does_nothing_but_audit(tmp_path):
    await _setup(tmp_path)
    eng = Engine()
    await bus.append_action(_resp("delete_everything:now"))   # not an internal tool
    await eng.pump_inbox()
    async with db.get_db() as c:
        cur = await c.execute("SELECT COUNT(*) FROM audit WHERE kind='unknown_token'")
        (n,) = await cur.fetchone()
    assert n == 1


# ── resilience: one bad action must not jam the inbox ────────────────────────

class BoomAssistant(Agent):
    triggers = [OnMessage()]
    slice = Slice()

    async def run(self, ctx):
        raise RuntimeError("boom")


async def test_bad_action_is_handled_not_jammed(tmp_path):
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(BoomAssistant(), assistant=True)
    await bus.append_action(_msg("this explodes"))
    await eng.pump_inbox()

    assert await bus.take_unhandled_actions() == []          # drained despite the error
    async with db.get_db() as c:
        cur = await c.execute("SELECT COUNT(*) FROM audit WHERE kind='action_error'")
        (n,) = await cur.fetchone()
    assert n == 1


# ── the full circle: schedule -> ask -> user approves -> applied ─────────────

class GoalAgent(Agent):
    triggers = [OnSchedule("* * * * *")]     # fires on the first tick (any minute)
    slice = Slice()

    async def run(self, ctx):
        return Proposal(intent="structured:goals.race",
                        content="switch focus to recovery?")


async def test_round_trip_schedule_ask_then_user_approves(tmp_path):
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(GoalAgent())

    await eng.tick()                                          # cron -> proposal -> ask
    out = await bus.read_outbox(bus.MAIN, 0)
    ask = next(r for _, r in out if r.kind is RenderableKind.ASK)
    assert await store.get_structured("goals.") == []        # nothing written yet

    await bus.append_action(_resp(ask.actions[0].token))     # you tap "Yes"
    await eng.tick()                                          # inbox pumped -> applied

    (entry,) = await store.get_structured("goals.")
    assert entry.provenance is Provenance.USER_INSTRUCTED


# ── retire (the maintainer prunes a fast-track belief; user-owned is protected) ─

async def test_retire_deletes_a_fast_track_belief(tmp_path):
    await _setup(tmp_path)
    await store.upsert_structured(StructuredEntry(
        id="patterns.focus.x", kind="patterns.focus.x", subject="you",
        content="c", provenance=Provenance.AGENT_INFERRED), change_note="t")
    out = await gate.apply(Proposal(intent="retire:patterns.focus.x", content="",
                                    provenance=Provenance.AGENT_INFERRED),
                           conversation_id=bus.MAIN, author="PatternAgent")
    assert out == "retired" and await store.get_structured("patterns.") == []


async def test_retire_of_a_user_belief_asks_not_denies(tmp_path):
    # a user-owned belief is no longer silently DENIED — retiring an existing one raises a
    # confirmation card and deletes only on approve (an absent one is still 'absent', below).
    await _setup(tmp_path)
    await store.upsert_structured(
        StructuredEntry(id="goals.race", kind="goals.race", subject="you", content="run a 10k"),
        change_note="seed")
    out = await gate.apply(Proposal(intent="retire:goals.race", content="",
                                    provenance=Provenance.AGENT_INFERRED),
                           conversation_id=bus.MAIN, author="PatternAgent")
    assert out == "pending"                      # confirm first; nothing silently dropped
    assert [e.kind for e in await store.get_structured(prefix="goals.race")] == ["goals.race"]


async def test_retire_of_a_missing_belief_is_absent(tmp_path):
    await _setup(tmp_path)
    out = await gate.apply(Proposal(intent="retire:patterns.focus.ghost", content="",
                                    provenance=Provenance.AGENT_INFERRED),
                           conversation_id=bus.MAIN, author="PatternAgent")
    assert out == "absent"


# ── an approved act is performed by its agent, in its own code ───────────────

class Sender(Agent):
    """Declares one action. `act` records what it was asked to do."""
    name = "sender"; session = "sender"; triggers = [OnMessage()]; slice = Slice()
    def __init__(self): self.done = []
    async def run(self, ctx): return None
    async def act(self, ctx, proposal):
        self.done.append((proposal.intent, proposal.content, dict(proposal.meta)))
        return f"Sent to #{proposal.meta['channel']}."


class Mute(Agent):
    name = "mute"; session = "mute"; triggers = [OnMessage()]; slice = Slice()
    async def run(self, ctx): return None          # no act(): the base raises


async def _tap(eng, ask_id, approve=True, cid=bus.MAIN):
    await bus.append_action(Action(id=new_id(), conversation_id=cid, kind=ActionKind.ACTION_RESPONSE,
                                   channel="cli:local", token=f"{'approve' if approve else 'reject'}:{ask_id}",
                                   anchor=ask_id, at=now_ms()))
    await eng.pump_inbox()


async def _ask_id():
    async with db.get_db() as c:
        return (await (await c.execute("SELECT ask_id FROM pending")).fetchone())["ask_id"]


async def _advice_texts(cid=bus.MAIN):
    return [r.body.get("text") for _, r in await bus.read_outbox(cid, 0) if r.kind is RenderableKind.ADVICE]


async def test_approved_act_is_dispatched_to_the_proposing_agent(tmp_path):
    db.configure(str(tmp_path / "ken.db")); await db.init_db()
    eng = Engine(); s = Sender(); eng.register_agent(s)
    await gate.apply(Proposal(intent="act:reply", content="ok, Friday", meta={"channel": "eng", "ts": "1"},
                              provenance=Provenance.USER_INSTRUCTED), conversation_id=bus.MAIN, author="sender")
    await _tap(eng, await _ask_id())
    assert s.done == [("act:reply", "ok, Friday", {"channel": "eng", "ts": "1"})]
    assert "Sent to #eng." in await _advice_texts()
    async with db.get_db() as c:
        kinds = [r[0] for r in await (await c.execute("SELECT kind FROM audit ORDER BY at")).fetchall()]
    assert "act_approved" in kinds and "act_done" in kinds


async def test_act_delegates_to_the_agent_named_in_meta_via(tmp_path):
    db.configure(str(tmp_path / "ken.db")); await db.init_db()
    eng = Engine(); s = Sender(); m = Mute(); eng.register_agent(s); eng.register_agent(m)
    # kip-style: proposed by one agent, performed by another (meta.via)
    await gate.apply(Proposal(intent="act:reply", content="on it", meta={"via": "sender", "channel": "ops", "ts": "2"},
                              provenance=Provenance.USER_INSTRUCTED), conversation_id=bus.MAIN, author="mute")
    await _tap(eng, await _ask_id())
    assert s.done and s.done[0][2]["channel"] == "ops"


async def test_act_failure_is_shown_and_audited_never_a_crash(tmp_path):
    db.configure(str(tmp_path / "ken.db")); await db.init_db()
    eng = Engine(); eng.register_agent(Mute())
    await gate.apply(Proposal(intent="act:reply", content="x", meta={"channel": "c"},
                              provenance=Provenance.USER_INSTRUCTED), conversation_id=bus.MAIN, author="mute")
    await _tap(eng, await _ask_id())                 # Mute has no act() → NotImplementedError inside
    errs = [r.body for _, r in await bus.read_outbox(bus.MAIN, 0)
            if r.kind is RenderableKind.ADVICE and r.body.get("type") == "error"]
    assert errs and "Couldn't do that" in errs[0]["text"] and "tap Yes to retry" in errs[0]["text"]
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM audit WHERE kind='act_failed'")).fetchone())[0] == 1
        # the ask is still open: a second tap retries instead of resolving to no_pending
        assert (await (await c.execute("SELECT count(*) FROM pending")).fetchone())[0] == 1
    assert await bus.take_unhandled_actions() == []  # the inbox was acked; nothing jammed


async def test_a_successful_act_closes_the_ask_and_a_second_tap_is_a_noop(tmp_path):
    db.configure(str(tmp_path / "ken.db")); await db.init_db()
    eng = Engine(); s = Sender(); eng.register_agent(s)
    await gate.apply(Proposal(intent="act:reply", content="ok", meta={"channel": "eng", "ts": "1"},
                              provenance=Provenance.USER_INSTRUCTED), conversation_id=bus.MAIN, author="sender")
    ask_id = await _ask_id()
    await _tap(eng, ask_id)
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM pending")).fetchone())[0] == 0
    await _tap(eng, ask_id)                                        # replayed / double tap
    assert len(s.done) == 1                                        # sent exactly once


# ── delegation: one agent asks another ───────────────────────────────────────


# ── a card carries the id of the action that raised it ────────────────────────

class Proposer_(Agent):
    """An agent whose reply is a people proposal: the gate cards it."""
    triggers = [OnMessage()]
    slice = Slice()

    async def run(self, ctx):
        return Proposal(intent="structured:people.jack", content="Jack is my manager",
                        evidence=[Evidence(ref=ctx.message, source="user")])


async def test_a_card_carries_the_action_that_raised_it(tmp_path):
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Proposer_(), assistant=True)
    a = _msg("Jack is my manager")
    await bus.append_action(a)
    await eng.pump_inbox()
    assert await gate.cards_for(a.id) == [(await _asks_for("people.jack"))[-1].id]
    assert await gate.cards_for("some-other-action") == []
    # a Yes closes it: nothing left for that action
    (card,) = await _asks_for("people.jack")
    await bus.append_action(_resp(f"approve:{card.id}"))
    await eng.pump_inbox()
    assert await gate.cards_for(a.id) == []


# ── a typed write from an interface: no model turn, judged, answered with its cards ───

async def test_a_typed_tool_action_runs_the_capture_and_answers_with_outcome(tmp_path):
    from ken.agents.assistant import Assistant
    from ken.memory import store
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)
    a = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:user",
               text=json.dumps({"tool": "add_task", "args": {"text": "buy milk"}}),
               meta={"provenance": "user", "source": "claude-app"}, at=now_ms())
    await bus.append_action(a)
    await eng.pump_inbox()
    (t,) = await store.open_tasks()
    assert t.text == "buy milk" and t.provenance is Provenance.USER_INSTRUCTED   # user token → user
    reply = [r for _, r in await bus.read_outbox(bus.MAIN, 0)
             if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == a.id]
    assert reply[0].body["outcome"] == "task" and reply[0].body["cards"] == []
    assert reply[0].body["result"] == [{"tool": "add_task", "outcome": "task", "text": "added to your tasks"}]
    # an AGENT token asking for user provenance is clamped to WORLD: the task CARDS instead
    b = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:agent:codex",
               text=json.dumps({"tool": "add_task", "args": {"text": "review PR 12"}}),
               meta={"provenance": "user", "source": "codex"}, at=now_ms())
    await bus.append_action(b)
    await eng.pump_inbox()
    reply = [r for _, r in await bus.read_outbox(bus.MAIN, 0)
             if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == b.id]
    assert reply[0].body["outcome"] == "pending" and len(reply[0].body["cards"]) == 1
    assert [t.text for t in await store.open_tasks()] == ["buy milk"]              # not written
    # the user's own token, writing AS an agent (a report): AGENT_INFERRED — an event logs
    d = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:user",
               text=json.dumps({"tool": "log_event", "args": {"text": "shipped the milk run"}}),
               meta={"provenance": "agent", "source": "claude-code", "task_id": t.id}, at=now_ms())
    await bus.append_action(d)
    await eng.pump_inbox()
    (ev,) = await store.recent_events()
    assert ev.summary == "shipped the milk run" and ev.provenance is Provenance.AGENT_INFERRED and ev.until is None
    # `until`: when it stops mattering; an unreadable one is dropped and said
    f = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:user",
               text=json.dumps({"tool": "log_event", "args": {"text": "offer stands", "until": "2099-01-01T09:00"}}),
               meta={"provenance": "agent", "source": "claude-code"}, at=now_ms())
    g = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:user",
               text=json.dumps({"tool": "log_event", "args": {"text": "sale on", "until": "next week-ish"}}),
               meta={"provenance": "agent", "source": "claude-code"}, at=now_ms())
    await bus.append_action(f); await bus.append_action(g)
    await eng.pump_inbox(); await eng.pump_inbox()
    by = {e.summary: e for e in await store.recent_events()}
    assert by["offer stands"].until is not None and by["sale on"].until is None
    reply = [r for _, r in await bus.read_outbox(bus.MAIN, 0)
             if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == g.id]
    assert "never expires" in reply[0].body["text"]
    # an AGENT token's event is WORLD: it cards (never denied, never silently history)
    e = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:agent:codex",
               text=json.dumps({"tool": "log_event", "args": {"text": "you agreed to pay"}}),
               meta={"provenance": "agent", "source": "codex", "task_id": t.id}, at=now_ms())
    await bus.append_action(e)
    await eng.pump_inbox()
    reply = [r for _, r in await bus.read_outbox(bus.MAIN, 0)
             if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == e.id]
    assert reply[0].body["outcome"] == "pending" and len(reply[0].body["cards"]) == 1
    from ken import mcp_server
    (card,) = [c for c in await mcp_server.pending_cards() if c["card_id"] == reply[0].body["cards"][0]]
    assert card["type"] == "event" and card["task_id"] == t.id                     # rides on the card
    assert len(await store.recent_events()) == 3                                  # the WORLD one did not land
    # an unknown tool is refused, loudly, with nothing written
    c = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:user",
               text=json.dumps({"tool": "drop_table", "args": {}}), at=now_ms())
    await bus.append_action(c)
    await eng.pump_inbox()
    reply = [r for _, r in await bus.read_outbox(bus.MAIN, 0)
             if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == c.id]
    assert reply[0].body["outcome"] == "invalid"


async def test_hydrate_skips_the_judge_on_empty_namespaces_and_judges_once_filled(tmp_path):
    from ken.agents.assistant import Assistant
    from ken.consent import judge as _judge
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)
    calls = []

    async def counting_judge(proposal, *, message, existing):
        calls.append(proposal.intent)
        return _judge.keep()
    eng.judge = counting_judge

    async def hydrate(items):
        a = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:user",
                   text=json.dumps({"tool": "hydrate", "args": {"items": items}}),
                   meta={"provenance": "agent", "source": "hydrate"}, at=now_ms())
        await bus.append_action(a)
        await eng.pump_inbox()
        return next(r for _, r in await bus.read_outbox(bus.MAIN, 0)
                    if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == a.id).body

    # fifty people over five calls: zero judge calls, five cards
    for n in range(5):
        body = await hydrate([{"kind": "person", "name": f"Person {n}{i}", "note": f"note {n}{i}"} for i in range(10)])
        assert body["outcome"] == "pending" and len(body["cards"]) == 1
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM pending WHERE ask_type='batch'")).fetchone())[0] == 5
    assert calls == []
    # eleven at once is refused before anything is proposed
    body = await hydrate([{"kind": "person", "name": f"P{i}", "note": "x"} for i in range(11)])
    assert body["outcome"] == "invalid" and "at most 10" in body["text"]
    # fill the people namespace (approve one batch); the next people batch is judged one by one
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending WHERE ask_type='batch' LIMIT 1")).fetchone()
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "applied"
    assert len(await store.get_structured("people.")) == 10
    body = await hydrate([{"kind": "person", "name": "Zed", "note": "new"},
                          {"kind": "project", "name": "Reader", "note": "the app"},     # projects still empty
                          {"kind": "robot", "name": "R2", "note": "no such kind"}])
    assert calls == ["structured:people.zed"]                     # judged: people is filled; projects batched
    assert body["outcome"] == "pending" and len(body["cards"]) == 2   # Zed's own card + the projects batch
    assert "R2: skipped" in body["text"]
    # nothing landed → never "applied": every item skipped is "failed"
    body = await hydrate([{"kind": "robot", "name": "R2", "note": "x"}])
    assert body["outcome"] == "failed"
    # a duplicate inside one batch is named, and the first row wins
    body = await hydrate([{"kind": "project", "name": "Ken", "note": "the engine"},
                          {"kind": "project", "name": "ken", "note": "a typo of the same"}])
    assert "ken: same as 'Ken' in this batch" in body["text"]
    (card_id,) = body["cards"]
    from ken import mcp_server
    (card,) = [c for c in await mcp_server.pending_cards() if c["card_id"] == card_id]
    assert [i["note"] for i in card["items"]] == ["the engine"]
    # the same batch again (a retry) is the SAME card — and the retry gets its id back
    again = await hydrate([{"kind": "project", "name": "Ken", "note": "the engine"}])
    assert again["outcome"] == "pending" and again["cards"] == [card_id]


async def test_a_preference_rewrite_says_replaced_and_a_told_report_cites_the_action_and_its_source(tmp_path):
    from ken.agents.assistant import Assistant
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)

    async def typed(tool, args):
        a = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:user",
                   text=json.dumps({"tool": tool, "args": args}), meta={"provenance": "agent", "source": "coffee chat"}, at=now_ms())
        await bus.append_action(a)
        await eng.pump_inbox()
        return next(r for _, r in await bus.read_outbox(bus.MAIN, 0)
                    if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == a.id).body
    first = await typed("remember", {"kind": "preference", "name": "coffee beans", "note": "likes light roasts"})
    assert first["text"] == "saved"
    again = await typed("remember", {"kind": "preference", "name": "coffee beans", "note": "likes light roasts; owns a humidor"})
    assert again["outcome"] == "applied" and "replaced your earlier note on 'coffee beans'" in again["text"]
    same = await typed("remember", {"kind": "preference", "name": "coffee beans", "note": "likes light roasts; owns a humidor"})
    assert same["text"] == "saved"                                                     # a no-op rewrite is not "replaced"
    # a told report (a MESSAGE through kip) carries the action id and the tell source as evidence
    tid, _ = await store.add_task("decide on coffee beans", Provenance.USER_STATED)
    m = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.MESSAGE, channel="mcp:user",
               text="(via web research) found 3 shops", meta={"provenance": "agent", "source": "web research", "task_id": tid}, at=now_ms())
    await bus.append_action(m)
    await eng.pump_inbox()
    ctx_seen = {}

    class _Spy(Assistant):
        async def run(self, ctx):
            ctx_seen.update(action_id=ctx.action_id, source=ctx.source, task_id=ctx.task_id)
            return await super().run(ctx)
    eng2 = Engine(); eng2.register_agent(_Spy(llm=None), assistant=True)
    m2 = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.MESSAGE, channel="mcp:user",
                text="(via web research) found 3 shops", meta={"provenance": "agent", "source": "web research", "task_id": tid}, at=now_ms())
    await bus.append_action(m2)
    await eng2.pump_inbox()
    assert ctx_seen == {"action_id": m2.id, "source": "web research", "task_id": tid}


async def test_a_date_alone_or_midnight_means_nine_in_the_morning(tmp_path):
    from datetime import datetime

    from ken.agents.assistant import Assistant
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)

    async def typed(tool, args):
        a = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:user",
                   text=json.dumps({"tool": tool, "args": args}), meta={"provenance": "user"}, at=now_ms())
        await bus.append_action(a)
        await eng.pump_inbox()
        return next(r for _, r in await bus.read_outbox(bus.MAIN, 0)
                    if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == a.id).body
    assert (await typed("add_task", {"text": "recipes sync", "when": "2099-09-29"}))["outcome"] == "task"
    assert (await typed("add_task", {"text": "write Q4 priorities", "when": "2099-09-30T00:00"}))["outcome"] == "task"
    assert (await typed("add_task", {"text": "dentist", "when": "2099-09-30T15:30"}))["outcome"] == "task"
    when = {t.text: datetime.fromtimestamp(t.deadline / 1000) for t in await store.open_tasks()}
    assert (when["recipes sync"].hour, when["recipes sync"].minute) == (9, 0)
    assert (when["write Q4 priorities"].day, when["write Q4 priorities"].hour) == (30, 9)
    assert (when["dentist"].hour, when["dentist"].minute) == (15, 30)                 # a real time is kept
    assert (await typed("reschedule_task", {"task": "dentist", "when": "2099-10-01"}))["outcome"] == "rescheduled"
    t = next(t for t in await store.open_tasks() if t.text == "dentist")
    assert datetime.fromtimestamp(t.deadline / 1000).hour == 9


async def test_remember_with_links_and_url_through_the_inbox(tmp_path):
    from ken.agents.assistant import Assistant
    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)
    await store.upsert_structured(StructuredEntry(id="projects.recipes", kind="projects.recipes", subject="Recipes", content="recipes in-app",
                                                  evidence=[Evidence(ref="m0", source="user")],
                                                  provenance=Provenance.USER_STATED, updated_at=1), change_note="t")

    async def typed(args):
        a = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:user",
                   text=json.dumps({"tool": "remember", "args": args}), meta={"provenance": "user"}, at=now_ms())
        await bus.append_action(a)
        await eng.pump_inbox()
        return next(r for _, r in await bus.read_outbox(bus.MAIN, 0)
                    if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == a.id).body
    body = await typed({"kind": "asset", "name": "Ledger", "note": "the finance sheet", "url": "https://sheets/1",
                        "links": [{"kind": "project", "name": "Recipes", "rel": "part_of"},
                                  {"kind": "person", "name": "Nobody", "rel": "member"},
                                  {"kind": "project", "name": "Recipes", "rel": "buddy"}]})
    # a new record in the user's own words: saved at once, no card
    assert body["outcome"] == "applied" and body["cards"] == []
    assert "link to 'Nobody' dropped: no such person yet" in body["text"] and "link to 'Recipes' dropped: needs" in body["text"]
    assert [(lk["src"], lk["dst"], lk["rel"]) for lk in await store.links_of(["assets.ledger"])] == [("assets.ledger", "projects.recipes", "part_of")]
    (asset,) = await store.get_structured("assets.")
    assert asset.url == "https://sheets/1" and asset.subject == "you"


async def test_hydrate_links_resolve_to_card_mates_only_and_show_the_resolved_title(tmp_path):
    from ken.agents.assistant import Assistant
    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)
    await store.upsert_structured(StructuredEntry(id="people.robert_smith", kind="people.robert_smith", subject="Robert Smith", content="designer",
                                                  evidence=[Evidence(ref="m0", source="user")],
                                                  provenance=Provenance.USER_STATED, updated_at=1), change_note="t")
    a = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:user",
               text=json.dumps({"tool": "hydrate", "args": {"items": [
                   {"kind": "project", "name": "Recipes", "note": "recipes in-app",
                    "links": [{"kind": "person", "name": "Robert", "rel": "member"},          # a twin: resolves to Robert Smith
                              {"kind": "person", "name": "Ann", "rel": "member"},             # skipped item below: not a card-mate
                              {"kind": "asset", "name": "Figma", "rel": "asset"}]},          # a card-mate
                   {"kind": "person", "name": "Ann", "note": ""},
                   {"kind": "asset", "name": "Figma", "note": "the design file", "url": "https://figma/x"}]}}),
               meta={"provenance": "user"}, at=now_ms())
    await bus.append_action(a)
    await eng.pump_inbox()
    body = next(r for _, r in await bus.read_outbox(bus.MAIN, 0)
                if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == a.id).body
    assert body["outcome"] == "pending" and "Recipes: link to 'Ann' dropped" in body["text"]
    from ken import mcp_server
    (card,) = await mcp_server.pending_cards()
    recipes = next(i for i in card["items"] if i["name"] == "Recipes")
    assert [(lk["rel"], lk["name"]) for lk in recipes["links"]] == [("member", "Robert Smith"), ("asset", "Figma")]
    assert await gate.resolve(card["card_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    got = sorted((lk["src"], lk["dst"], lk["rel"]) for lk in await store.links_of(["projects.recipes"]))
    assert got == [("assets.figma", "projects.recipes", "part_of"), ("people.robert_smith", "projects.recipes", "member")]


async def test_an_agent_token_cannot_edit_a_task_silently_and_a_ref_becomes_the_evidence(tmp_path):
    from datetime import datetime, timedelta

    from ken.agents.assistant import Assistant
    from ken.consent import gate
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)

    async def typed(tool, args, channel="mcp:agent:commitments", ref=""):
        meta = {"provenance": "agent", "source": "commitments"}
        if ref:
            meta["ref"] = ref
        a = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel=channel,
                   text=json.dumps({"tool": tool, "args": args}), meta=meta, at=now_ms())
        await bus.append_action(a)
        await eng.pump_inbox()
        return next(r for _, r in await bus.read_outbox(bus.MAIN, 0)
                    if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == a.id).body
    tid, _ = await store.add_task("send the deck to Jack", Provenance.USER_STATED)
    # an agent token reporting "done": a card, the task stays open
    body = await typed("complete_task", {"task": "send the deck"}, ref="https://slack.com/archives/C1/p1")
    assert body["outcome"] == "pending" and "done? card is up" in body["text"] and len(body["cards"]) == 1
    assert tid in {t.id for t in await store.open_tasks()}
    card = next(r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.kind is RenderableKind.ASK and r.id == body["cards"][0])
    assert card.body["title"] == "Mark done?" and "commitments reports it done" in card.body["note"]
    # a Yes: kip performs it
    await eng._dispatch_token(type("A", (), {"token": f"approve:{card.id}", "text": "", "channel": "tui",
                                            "conversation_id": bus.MAIN, "id": "a"})())
    assert tid not in {t.id for t in await store.open_tasks()}
    # an agent token moving a task: a Reschedule? card; the user's own token still moves it at once
    tid2, _ = await store.add_task("review PR 12", Provenance.USER_STATED)
    when = (datetime.now() + timedelta(days=3)).replace(hour=15, minute=0, second=0, microsecond=0).isoformat()
    body = await typed("reschedule_task", {"task": "review PR 12", "when": when})
    assert body["outcome"] == "pending" and "reschedule card is up" in body["text"]
    assert next(t for t in await store.open_tasks() if t.id == tid2).deadline is None
    body = await typed("reschedule_task", {"task": "review PR 12", "when": when}, channel="mcp:user")
    assert body["outcome"] == "rescheduled"
    # a permalink ref is the evidence on a typed write
    body = await typed("add_task", {"text": "waiting on Priya: review of #12"}, ref="https://slack.com/archives/C1/p2")
    assert body["outcome"] == "pending"
    await gate.resolve(body["cards"][0], approved=True, conversation_id=bus.MAIN)
    t = next(t for t in await store.open_tasks() if t.text.startswith("waiting on Priya"))
    assert [(e.ref, e.source) for e in t.evidence] == [("https://slack.com/archives/C1/p2", "commitments")]
    # and on a told report: the context carries the ref for the reconciler's captures
    seen = {}

    class _Spy(Assistant):
        async def run(self, ctx):
            seen.update(ref=ctx.ref, source=ctx.source)
            return await super().run(ctx)
    eng2 = Engine(); eng2.register_agent(_Spy(llm=None), assistant=True)
    m = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.MESSAGE, channel="mcp:agent:commitments",
               text="(via commitments) Jack says the deck is sent", meta={"provenance": "agent", "source": "commitments",
                                                                          "ref": "https://slack.com/archives/C1/p3"}, at=now_ms())
    await bus.append_action(m)
    await eng2.pump_inbox()
    assert seen == {"ref": "https://slack.com/archives/C1/p3", "source": "commitments"}


async def test_world_task_edits_one_card_per_task_show_the_time_and_never_roll_forward(tmp_path):
    from datetime import datetime, timedelta

    from ken.agents.assistant import Assistant
    from ken.engine import _provenance_of
    await _setup(tmp_path)
    # an agent channel is WORLD even with no provenance key at all
    assert _provenance_of(type("A", (), {"channel": "mcp:agent:x", "meta": {}})()) is Provenance.WORLD
    assert _provenance_of(type("A", (), {"channel": "mcp:user", "meta": {}})()) is None
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)

    async def typed(tool, args, source):
        a = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel=f"mcp:agent:{source}",
                   text=json.dumps({"tool": tool, "args": args}), meta={"source": source}, at=now_ms())
        await bus.append_action(a)
        await eng.pump_inbox()
        return next(r for _, r in await bus.read_outbox(bus.MAIN, 0)
                    if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == a.id).body
    await store.add_task("send the deck to Jack", Provenance.USER_STATED)
    # two agents report the same task done: ONE card, and the second reply carries its id
    b1 = await typed("complete_task", {"task": "send the deck"}, "commitments")
    b2 = await typed("complete_task", {"task": "send the deck"}, "github")
    assert b1["outcome"] == "pending" and b2["outcome"] == "pending" and b1["cards"] == b2["cards"]
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM pending")).fetchone())[0] == 1
    card = next(r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.kind is RenderableKind.ASK)
    assert "commitments reports it done" in card.body["note"]           # the source, never the channel label
    # a reschedule card shows the time it would move to
    tid2, _ = await store.add_task("review PR 12", Provenance.USER_STATED)
    when = (datetime.now() + timedelta(days=2)).replace(hour=15, minute=0, second=0, microsecond=0)
    b3 = await typed("reschedule_task", {"task": "review PR 12", "when": when.isoformat()}, "commitments")
    card2 = next(r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.kind is RenderableKind.ASK and r.id == b3["cards"][0])
    assert card2.body["title"] == "Reschedule?"
    assert "3pm" in card2.body["question"].lower() or "15:00" in card2.body["question"]     # the time is on the card
    assert "reports a new time" in card2.body["note"]
    # a late Yes on a REPORT card never rolls forward to tomorrow 09:00: it refuses honestly
    async with db.get_db() as c:
        await c.execute("UPDATE pending SET meta=json_set(meta, '$.at', ?) WHERE ask_id=?", (now_ms() - 60_000, card2.id))
        await c.commit()
    await eng._dispatch_token(type("A", (), {"token": f"approve:{card2.id}", "text": "", "channel": "tui",
                                            "conversation_id": bus.MAIN, "id": "a"})())
    assert next(t for t in await store.open_tasks() if t.id == tid2).deadline is None
    line = [r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.kind is RenderableKind.ADVICE and "already passed" in (r.body.get("text") or "")]
    assert line, "the late Yes must say the time passed"


# ── task time: a slot is planning (applied at once), a deadline keeps its old path ───

async def _tool(eng, tool: str, args: dict, channel: str = "mcp:agent:daily") -> dict:
    a = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel=channel,
               text=json.dumps({"tool": tool, "args": args}), meta={"provenance": "user" if channel == "mcp:user" else "agent", "source": "daily-agent"},
               at=now_ms())
    await bus.append_action(a)
    await eng.pump_inbox()
    return next(r.body for _, r in await bus.read_outbox(bus.MAIN, 0)
                if r.kind is RenderableKind.ADVICE and r.body.get("reply_to") == a.id)


async def test_schedule_task_sets_and_clears_a_slot_without_a_card(tmp_path):
    from datetime import datetime, timedelta

    from ken.agents.assistant import Assistant
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)
    tid, _ = await store.add_task("write the brief", Provenance.USER_STATED)
    start = (datetime.now() + timedelta(days=1)).replace(hour=14, minute=0, second=0, microsecond=0)
    # an AGENT token slots it: applied at once, no card (a slot is planning, not world knowledge)
    body = await _tool(eng, "schedule_task", {"task": tid, "start": start.isoformat(), "minutes": 90})
    assert body["outcome"] == "slotted" and body["cards"] == []
    t = await store.task_by_id(tid)
    assert (t.slot_start, t.slot_minutes, t.deadline) == (int(start.timestamp() * 1000), 90, None)
    # no default duration: missing or zero minutes is refused with the reason, the slot stays
    for bad in ({"task": tid, "start": start.isoformat()}, {"task": tid, "start": start.isoformat(), "minutes": 0}):
        body = await _tool(eng, "schedule_task", bad)
        assert body["outcome"] == "slot_invalid" and "minutes is required" in body["text"]
    body = await _tool(eng, "schedule_task", {"task": tid, "start": "2026-10-21", "minutes": 30})
    assert body["outcome"] == "slot_invalid"                       # a date alone is not a start
    assert (await store.task_by_id(tid)).slot_minutes == 90
    # an empty start clears it
    body = await _tool(eng, "schedule_task", {"task": tid, "start": ""})
    assert body["outcome"] == "slot_cleared"
    t = await store.task_by_id(tid)
    assert (t.slot_start, t.slot_minutes) == (None, None)


async def test_a_slot_the_user_set_by_hand_is_held_from_agents(tmp_path):
    from datetime import datetime, timedelta

    from ken.agents.assistant import Assistant
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)
    tid, _ = await store.add_task("review the mockups", Provenance.USER_STATED)
    start = (datetime.now() + timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)
    other = (start + timedelta(hours=3)).isoformat()
    # the agent's slot: by agent, and the agent may move it again
    assert (await _tool(eng, "schedule_task", {"task": tid, "start": start.isoformat(), "minutes": 30}))["outcome"] == "slotted"
    assert (await store.task_by_id(tid)).slot_by == "agent"
    assert (await _tool(eng, "schedule_task", {"task": tid, "start": other, "minutes": 30}))["outcome"] == "slotted"
    # the user moves it by hand: now it is held, an agent can neither move nor clear it
    body = await _tool(eng, "schedule_task", {"task": tid, "start": start.isoformat(), "minutes": 45}, channel="mcp:user")
    assert body["outcome"] == "slotted" and (await store.task_by_id(tid)).slot_by == "user"
    for args in ({"task": tid, "start": other, "minutes": 30}, {"task": tid, "start": ""}):
        assert (await _tool(eng, "schedule_task", args))["outcome"] == "slot_held"
    t = await store.task_by_id(tid)
    assert (t.slot_start, t.slot_minutes, t.slot_by) == (int(start.timestamp() * 1000), 45, "user")
    # the user still clears it; then an agent may slot it again
    assert (await _tool(eng, "schedule_task", {"task": tid, "start": ""}, channel="mcp:user"))["outcome"] == "slot_cleared"
    assert (await store.task_by_id(tid)).slot_by is None
    assert (await _tool(eng, "schedule_task", {"task": tid, "start": other, "minutes": 30}))["outcome"] == "slotted"


async def test_the_old_when_name_still_sets_the_deadline(tmp_path):
    from datetime import datetime, timedelta

    from ken.agents.assistant import Assistant
    await _setup(tmp_path)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)
    day = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    body = await _tool(eng, "add_task", {"text": "file the taxes", "when": day}, channel="mcp:user")
    assert body["outcome"] == "task"
    t = next(t for t in await store.open_tasks() if t.text == "file the taxes")
    assert datetime.fromtimestamp(t.deadline / 1000).strftime("%Y-%m-%d") == day and t.slot_start is None
    later = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%dT10:00")
    body = await _tool(eng, "reschedule_task", {"task": t.id, "when": later}, channel="mcp:user")
    assert body["outcome"] == "rescheduled"
    assert datetime.fromtimestamp((await store.task_by_id(t.id)).deadline / 1000).strftime("%Y-%m-%dT%H:%M") == later
