"""The consent gate — the single applier. (ARCHITECTURE.md)

`apply` turns a judged proposal into a task, an event, a fast-tracked belief, or
a pending ask. `resolve` closes a pending ask deterministically. These are the
gate's units; the pump's use of them is exercised in test_inbound.
"""
import json

from ken import db
from ken.agents import Proposal
from ken.consent import gate
from ken.contracts import Action, ActionKind, Evidence, Provenance, RenderableKind
from ken.conversation import bus
from ken.memory import store
from ken.util import new_id, now_ms


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


async def _feedback(conversation_id=bus.MAIN):
    """The confirmation lines the gate posted back (advice renderables) for a resolve."""
    return [r.body.get("text") for _, r in await bus.read_outbox(conversation_id, 0)
            if r.kind is RenderableKind.ADVICE]


async def _open_ask(conversation_id, target):
    await gate.apply(Proposal(intent=f"structured:{target}", content="run a 10k",
                              provenance=Provenance.AGENT_INFERRED),
                     conversation_id=conversation_id, author="Goal")
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute(
            "SELECT ask_id FROM pending WHERE kind=?", (target,))).fetchone()
    return ask_id


async def test_apply_task_intent_writes_a_task(tmp_path):
    await _setup(tmp_path)
    p = Proposal(intent="task", content="book the dentist")
    assert await gate.apply(p, conversation_id=bus.MAIN, author="Daily") == "task"
    tasks = await store.open_tasks()
    assert [t.text for t in tasks] == ["book the dentist"]
    assert tasks[0].provenance is Provenance.AGENT_INFERRED


async def test_world_task_cards_instead_of_applying(tmp_path):
    # a task inferred from WORLD-class content (e.g. a Slack message) is untrusted — it must
    # ASK, not auto-add, even on a judge KEEP. Its scheduled time rides onto the pending ask.
    from ken.consent import judge
    await _setup(tmp_path)
    out = await gate.apply(
        Proposal(intent="task", content="review PR 58114", at=1893456000000, provenance=Provenance.WORLD),
        conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()), message="review PR")
    assert out == "pending"                                  # carded, not applied
    assert await store.open_tasks() == []                    # nothing landed yet
    async with db.get_db() as c:
        row = await (await c.execute("SELECT kind, at FROM pending")).fetchone()
    assert row["kind"].startswith("task:") and row["at"] == 1893456000000


async def test_approving_a_world_task_ask_creates_the_task_with_its_time(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    await gate.apply(
        Proposal(intent="task", content="wire the deposit", at=1893456000000, provenance=Provenance.WORLD),
        conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()))
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "applied"
    tasks = await store.open_tasks()
    assert [t.text for t in tasks] == ["wire the deposit"]
    assert tasks[0].deadline == 1893456000000           # the time carried through the card
    assert "Saved." in await _feedback()


async def test_rejecting_a_world_task_ask_creates_nothing(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    await gate.apply(Proposal(intent="task", content="pay the invoice", provenance=Provenance.WORLD),
                     conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()))
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=False, conversation_id=bus.MAIN) == "rejected"
    assert await store.open_tasks() == []


async def test_world_event_cards_never_writes_history_unapproved(tmp_path):
    # untrusted world content must not write immutable history un-approved (memory
    # poisoning): it cards, a No drops it, a Yes writes it as the user's own instruction.
    from ken.consent import judge
    await _setup(tmp_path)
    out = await gate.apply(Proposal(intent="event", content="you agreed to pay $5000", provenance=Provenance.WORLD),
                           conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()))
    assert out == "pending"
    assert await store.recent_events() == []
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=False, conversation_id=bus.MAIN) == "rejected"
    assert await store.recent_events() == []
    await gate.apply(Proposal(intent="event", content="you agreed to pay $5000", provenance=Provenance.WORLD),
                     conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()))
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "applied"
    (ev,) = await store.recent_events()
    assert ev.summary == "you agreed to pay $5000" and ev.provenance is Provenance.USER_INSTRUCTED


async def test_non_world_task_still_fast_tracks(tmp_path):
    # kip's tasks (default AGENT_INFERRED provenance, user-driven) must keep applying without a card.
    from ken.consent import judge
    await _setup(tmp_path)
    out = await gate.apply(Proposal(intent="task", content="book the dentist"),
                           conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.keep()))
    assert out == "task"
    assert [t.text for t in await store.open_tasks()] == ["book the dentist"]
    async with db.get_db() as c:
        (n,) = await (await c.execute("SELECT COUNT(*) FROM pending")).fetchone()
    assert n == 0                                            # no card


async def test_world_fast_track_belief_cards_instead_of_writing(tmp_path):
    # the WORLD choke covers task/event; a WORLD belief in a FAST-TRACK namespace
    # (preferences/patterns) must not slip past it and write silently on a judge KEEP.
    from ken.consent import judge
    await _setup(tmp_path)
    out = await gate.apply(
        Proposal(intent="structured:preferences.tone", content="prefers terse replies",
                 provenance=Provenance.WORLD),
        conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()))
    assert out == "pending"                                          # carded, not written
    assert await store.get_structured(prefix="preferences.") == []   # nothing landed
    async with db.get_db() as c:
        (n,) = await (await c.execute("SELECT COUNT(*) FROM pending")).fetchone()
    assert n == 1                                                    # a card is up


async def test_non_world_fast_track_belief_still_writes(tmp_path):
    # kip's own inferred preference (AGENT_INFERRED) must keep fast-tracking with no card.
    from ken.consent import judge
    await _setup(tmp_path)
    out = await gate.apply(
        Proposal(intent="structured:preferences.tone", content="prefers terse replies"),
        conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.keep()))
    assert out == "applied"
    assert [e.content for e in await store.get_structured(prefix="preferences.")] == ["prefers terse replies"]
    async with db.get_db() as c:
        (n,) = await (await c.execute("SELECT COUNT(*) FROM pending")).fetchone()
    assert n == 0                                                    # no card


async def test_approving_a_stale_world_task_drops_the_past_time(tmp_path):
    # a card that waited until its scheduled time passed must not queue a task in the past
    # (which would fire an immediate stale nudge) — the time is dropped to None on approval.
    from ken.consent import judge
    await _setup(tmp_path)
    await gate.apply(
        Proposal(intent="task", content="wire the deposit", at=now_ms() - 60_000,
                 provenance=Provenance.WORLD),
        conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()))
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "applied"
    tasks = await store.open_tasks()
    assert [t.text for t in tasks] == ["wire the deposit"]
    assert tasks[0].deadline is None            # the past time was dropped, no stale nudge


async def test_approving_a_duplicate_world_task_reports_it_already_existed(tmp_path):
    # approving a card for a task the user already has is a no-op add — the gate must not
    # claim "Saved." for a task that already existed.
    from ken.consent import judge
    await _setup(tmp_path)
    await store.add_task("wire the deposit", Provenance.USER_INSTRUCTED)   # already on the list
    await gate.apply(
        Proposal(intent="task", content="wire the deposit", provenance=Provenance.WORLD),
        conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()))
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "applied"
    assert [t.text for t in await store.open_tasks()] == ["wire the deposit"]   # still one
    assert "Already on your list." in await _feedback()


async def _seed_person(kind="people.dana", content="My manager"):
    from ken.contracts import StructuredEntry
    await store.upsert_structured(
        StructuredEntry(id=kind, kind=kind, subject="you", content=content), change_note="seed")


async def test_forget_a_saved_person_cards_then_deletes_on_approve(tmp_path):
    # a user-owned belief is never silently dropped: a retire ASKS, and only the approve deletes.
    await _setup(tmp_path)
    await _seed_person()
    out = await gate.apply(Proposal(intent="retire:people.dana", content="forget Dana"),
                           conversation_id=bus.MAIN, author="slack")
    assert out == "pending"                                            # a confirmation card, no delete yet
    assert [e.kind for e in await store.get_structured(prefix="people.dana")] == ["people.dana"]
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "applied"
    assert await store.get_structured(prefix="people.dana") == []     # gone
    assert "Forgotten." in await _feedback()


async def test_rejecting_a_forget_keeps_the_belief(tmp_path):
    await _setup(tmp_path)
    await _seed_person()
    await gate.apply(Proposal(intent="retire:people.dana", content="forget Dana"),
                     conversation_id=bus.MAIN, author="slack")
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=False, conversation_id=bus.MAIN) == "rejected"
    assert [e.kind for e in await store.get_structured(prefix="people.dana")] == ["people.dana"]  # kept
    assert "Kept." in await _feedback()


async def test_forgetting_an_absent_person_is_a_noop(tmp_path):
    await _setup(tmp_path)
    out = await gate.apply(Proposal(intent="retire:people.nobody", content="forget nobody"),
                           conversation_id=bus.MAIN, author="slack")
    assert out == "absent"                                             # nothing saved → nothing to forget
    async with db.get_db() as c:
        (n,) = await (await c.execute("SELECT COUNT(*) FROM pending")).fetchone()
    assert n == 0                                                      # no card


async def test_world_forget_of_a_fast_track_belief_asks_not_deletes(tmp_path):
    # a WORLD-provenance retire (untrusted, injectable) of a fast-track namespace must never
    # silently delete — it asks, like any belief. A trusted retire of it deletes straight.
    await _setup(tmp_path)
    from ken.contracts import StructuredEntry
    await store.upsert_structured(
        StructuredEntry(id="preferences.tone", kind="preferences.tone", subject="you",
                        content="terse"), change_note="seed")
    out = await gate.apply(Proposal(intent="retire:preferences.tone", content="",
                                    provenance=Provenance.WORLD),
                           conversation_id=bus.MAIN, author="slack")
    assert out == "pending"                                              # carded, not deleted
    assert [e.kind for e in await store.get_structured(prefix="preferences.tone")] == ["preferences.tone"]
    out = await gate.apply(Proposal(intent="retire:preferences.tone", content="",
                                    provenance=Provenance.USER_INSTRUCTED),
                           conversation_id=bus.MAIN, author="you")
    assert out == "retired"                                              # trusted → straight delete
    assert await store.get_structured(prefix="preferences.tone") == []


async def test_a_forget_and_an_update_card_never_both_open(tmp_path):
    # a retire ask and an update ask target the SAME belief — only the newer stays open, so
    # approving can't first forget then resurrect.
    await _setup(tmp_path)
    await _seed_person()
    await gate._raise_ask(Proposal(intent="retire:people.dana", content="forget Dana"),
                          kind="retire:people.dana", ask_type="retire",
                          conversation_id=bus.MAIN, author="slack")
    await gate._raise_ask(Proposal(intent="structured:people.dana", content="My skip-level"),
                          kind="people.dana", ask_type="belief",
                          conversation_id=bus.MAIN, author="slack")
    async with db.get_db() as c:
        rows = await (await c.execute("SELECT kind FROM pending")).fetchall()
    assert [r["kind"] for r in rows] == ["people.dana"]                 # update superseded the retire


def test_card_label_names_the_action_and_subject():
    # a card must say WHAT it asks: the action, and (for a named entity) the name.
    from ken.consent.gate import _card_label
    assert _card_label("task:confirm the rollout", None) == ("New task", None)
    assert _card_label("people.omar_haddad", None) == ("Save person", "Omar Haddad")
    assert _card_label("people.omar_haddad", "old note") == ("Update person", "Omar Haddad")
    assert _card_label("retire:people.dana", None) == ("Forget person", None)
    assert _card_label("projects.crumbly_ios", None) == ("Save project", "Crumbly Ios")
    assert _card_label("preferences.tone", None) == ("Save preference", None)   # a topic, no name heading


async def test_a_raised_person_card_carries_its_title_and_name(tmp_path):
    # the card the interface renders must carry the action label + the subject name, not just a
    # bare description — so "Save person / Omar Haddad" instead of an anonymous note.
    from ken.consent import judge
    await _setup(tmp_path)
    await gate.apply(
        Proposal(intent="structured:people.omar_haddad",
                 content="Mobile/Crumbly engineer.", provenance=Provenance.WORLD),
        conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()))
    cards = [r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.kind is RenderableKind.ASK]
    assert cards and cards[-1].body["title"] == "Save person"
    assert cards[-1].body["subject"] == "Omar Haddad"


async def test_editing_a_card_saves_the_users_value_as_trusted(tmp_path):
    # confirm-AND-correct: an edited approve saves the USER's text (not the proposal), and because
    # the user authored it, the belief is USER_STATED (trusted), not the untrusted WORLD source.
    from ken.consent import judge
    await _setup(tmp_path)
    await gate.apply(
        Proposal(intent="structured:people.alex_morgan",
                 content="Mobile/Crumbly engineer focused on attribution.",
                 provenance=Provenance.WORLD),
        conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()))
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    out = await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN,
                             edited="Mobile lead; owns the Larder storefront.")
    assert out == "applied"
    saved = await store.get_structured(prefix="people.alex_morgan")
    assert [e.content for e in saved] == ["Mobile lead; owns the Larder storefront."]   # the user's text
    assert saved[0].provenance is Provenance.USER_STATED                            # trusted, not WORLD
    assert saved[0].confidence == 1.0                                               # user-typed → full confidence
    assert saved[0].evidence == []                                                  # WORLD evidence dropped — they rewrote it
    assert "Saved your version." in await _feedback()


async def test_editing_a_task_card_saves_the_edited_text(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    await gate.apply(
        Proposal(intent="task", content="Confirm Larder rollout", provenance=Provenance.WORLD),
        conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.keep()))
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN,
                       edited="Ping Jack re: Larder rollout")
    assert [t.text for t in await store.open_tasks()] == ["Ping Jack re: Larder rollout"]   # the edit


async def test_collapse_never_drops_a_new_belief_onto_a_different_kind(tmp_path):
    # the silent-write-loss remember_person activated: a new person whose note matches ANOTHER
    # person's must never collapse onto them. A cross-kind COLLAPSE writes the new belief as its own.
    from ken.consent import judge
    await _setup(tmp_path)
    await _seed_person(kind="people.dana", content="my manager")
    out = await gate.apply(
        Proposal(intent="structured:people.priya", content="my manager",
                 provenance=Provenance.AGENT_INFERRED),
        conversation_id=bus.MAIN, author="kip",
        judge=_FixedJudge(judge.collapse(target_id="people.dana")))
    assert out == "conflict"                                         # an "Update Dana?" card, nothing dropped
    async with db.get_db() as c:
        (row,) = await (await c.execute("SELECT ask_id, kind, origin FROM pending")).fetchall()
    assert (row["kind"], row["origin"]) == ("identity:people.dana:people.priya", "identity")
    assert await gate.resolve(row["ask_id"], approved=False, conversation_id=bus.MAIN) == "reasked"   # "not the same person"
    async with db.get_db() as c:
        kinds = [r["kind"] for r in await (await c.execute("SELECT kind FROM pending")).fetchall()]
    assert kinds == ["people.priya"]                                # then a card for Priya, not "already saved"
    assert [e.content for e in await store.get_structured(prefix="people.dana")] == ["my manager"]  # Dana untouched


async def test_apply_event_intent_writes_an_event(tmp_path):
    await _setup(tmp_path)
    p = Proposal(intent="event", content="ran 10k")
    assert await gate.apply(p, conversation_id=bus.MAIN, author="Coach") == "event"
    async with db.get_db() as c:
        cur = await c.execute("SELECT summary FROM events")
        rows = await cur.fetchall()
    assert [r["summary"] for r in rows] == ["ran 10k"]


# ── evidence rides through onto the task and the event (Step 1 plumbing) ──────

async def test_task_stores_its_evidence(tmp_path):
    await _setup(tmp_path)
    ev = [Evidence(ref="cap-1", source="assistant")]
    await gate.apply(Proposal(intent="task", content="book the dentist", evidence=ev),
                     conversation_id=bus.MAIN, author="Daily")
    async with db.get_db() as c:
        (raw,) = await (await c.execute(
            "SELECT evidence FROM tasks WHERE text='book the dentist'")).fetchone()
    assert store.evidence_load(raw) == ev


async def test_event_stores_its_evidence(tmp_path):
    await _setup(tmp_path)
    ev = [Evidence(ref="metric-3", source="tracker")]
    await gate.apply(Proposal(intent="event", content="ran 10k", at=42, evidence=ev),
                     conversation_id=bus.MAIN, author="Coach")
    async with db.get_db() as c:
        (raw,) = await (await c.execute(
            "SELECT evidence FROM events WHERE summary='ran 10k'")).fetchone()
    assert store.evidence_load(raw) == ev


async def test_event_uses_proposal_at_when_present(tmp_path):
    await _setup(tmp_path)
    # an event gets the time it HAPPENED, not ingest time.
    await gate.apply(Proposal(intent="event", content="ran 10k", at=1234),
                     conversation_id=bus.MAIN, author="Coach")
    async with db.get_db() as c:
        (at,) = await (await c.execute("SELECT at FROM events WHERE summary='ran 10k'")).fetchone()
    assert at == 1234


async def test_event_falls_back_to_now_without_at(tmp_path):
    await _setup(tmp_path)
    before = now_ms()
    await gate.apply(Proposal(intent="event", content="woke up"),   # at unset
                     conversation_id=bus.MAIN, author="Coach")
    async with db.get_db() as c:
        (at,) = await (await c.execute("SELECT at FROM events WHERE summary='woke up'")).fetchone()
    assert at >= before        # a real ingest timestamp, never None


class _FixedJudge:
    """A deterministic judge that returns the same decision every call."""
    def __init__(self, decision):
        self.decision = decision

    async def __call__(self, proposal, *, message, existing):
        return self.decision


async def test_judge_conflict_on_an_event_degrades_to_keep_never_asks(tmp_path):
    # events are append-only history — there is no current value to contradict, so even a
    # (mistaken) CONFLICT verdict must NOT raise a card; the gate degrades it to a plain
    # keep and writes the event.
    from ken.consent import judge
    await _setup(tmp_path)
    p = Proposal(intent="event", content="ran 10k")
    out = await gate.apply(p, conversation_id=bus.MAIN, author="Coach",
                           judge=_FixedJudge(judge.conflict("whatever")), message="ran 10k")
    assert out == "event"                                    # kept, not "conflict"
    async with db.get_db() as c:
        (n,) = await (await c.execute("SELECT COUNT(*) FROM pending")).fetchone()
        (summary,) = await (await c.execute("SELECT summary FROM events")).fetchone()
    assert n == 0 and summary == "ran 10k"                   # no card raised; the event landed


async def test_judge_collapse_dedups_a_duplicate_event(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    # seed an existing event, then propose an identical one the judge collapses onto it
    await gate.apply(Proposal(intent="event", content="ran 10k"),
                     conversation_id=bus.MAIN, author="Coach")
    existing = (await store.recent_events())[0]
    out = await gate.apply(Proposal(intent="event", content="ran 10k"),
                           conversation_id=bus.MAIN, author="Coach",
                           judge=_FixedJudge(judge.collapse(existing.id)), message="ran 10k")
    assert out == "collapsed"                                # a duplicate — nothing new stored
    assert len(await store.recent_events()) == 1             # still one event, not two


async def test_apply_unknown_intent_raises(tmp_path):
    await _setup(tmp_path)
    p = Proposal(intent="delete_all", content="oops")
    try:
        await gate.apply(p, conversation_id=bus.MAIN, author="X")
        raised = False
    except ValueError:
        raised = True
    assert raised


async def test_resolve_without_pending_is_a_noop(tmp_path):
    await _setup(tmp_path)
    out = await gate.resolve("no-such-ask", approved=True, conversation_id=bus.MAIN)
    assert out == "no_pending"
    assert await store.get_structured("goals.") == []


# ── a resolve always posts feedback, so an interface never hangs (FIX 1b) ─────

async def test_approve_posts_a_saved_confirmation(tmp_path):
    await _setup(tmp_path)
    ask_id = await _open_ask(bus.MAIN, "goals.race")
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "applied"
    assert "Saved." in await _feedback()            # the interface can clear its spinner


async def test_reject_posts_a_not_saved_confirmation(tmp_path):
    await _setup(tmp_path)
    ask_id = await _open_ask(bus.MAIN, "goals.race")
    assert await gate.resolve(ask_id, approved=False, conversation_id=bus.MAIN) == "rejected"
    assert "Not saved." in await _feedback()


async def test_a_gone_card_posts_nothing(tmp_path):
    await _setup(tmp_path)
    # a superseded / already-answered card is SILENT here — a confirmation would
    # contradict the real outcome already shown. The interface clears its spinner by
    # noticing the ask left `pending`, not by a message from the gate.
    out = await gate.resolve("no-such-ask", approved=True, conversation_id=bus.MAIN)
    assert out == "no_pending"
    assert await bus.read_outbox(bus.MAIN, 0) == []


async def test_a_replayed_approve_posts_no_new_renderable(tmp_path):
    await _setup(tmp_path)
    ask_id = await _open_ask(bus.MAIN, "goals.race")
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "applied"
    after_first = len(await bus.read_outbox(bus.MAIN, 0))    # includes the one 'Saved.'
    # a crash-replay (acted, not yet acked) re-runs the same approve -> no_pending
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "no_pending"
    assert len(await bus.read_outbox(bus.MAIN, 0)) == after_first   # NOTHING new, no contradiction


async def test_resolve_without_a_conversation_posts_nothing(tmp_path):
    await _setup(tmp_path)
    # a direct call with no conversation to answer in must not post an orphan renderable
    await gate.resolve("no-such-ask", approved=True)
    assert await bus.read_outbox(bus.MAIN, 0) == []


async def test_reprompt_on_a_resolved_ask_is_silent(tmp_path):
    await _setup(tmp_path)
    # the ask is already gone (tapped, or never existed): re-prompting adds nothing
    await gate.reprompt("no-such-ask")
    assert await bus.read_outbox(bus.MAIN, 0) == []


async def test_resolve_from_a_foreign_conversation_is_refused(tmp_path):
    await _setup(tmp_path)
    p = Proposal(intent="structured:goals.race", content="run a 10k")
    await gate.apply(p, conversation_id="conv-A", author="Goal")
    async with db.get_db() as c:
        cur = await c.execute("SELECT ask_id FROM pending WHERE kind='goals.race'")
        (ask_id,) = await cur.fetchone()

    # a decision arriving from another conversation must not resolve this ask
    assert await gate.resolve(ask_id, approved=True,
                              conversation_id="conv-B") == "no_pending"
    assert await store.get_structured("goals.") == []        # untouched
    # the rightful conversation still can
    assert await gate.resolve(ask_id, approved=True,
                              conversation_id="conv-A") == "applied"


async def _tap(ask_id, *, approve: bool, edited: str = "", at=None):
    """What an interface writes when the user taps a card, then what the engine does."""
    await bus.append_action(Action(
        id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.ACTION_RESPONSE,
        channel="cli:local", token=f"{'approve' if approve else 'reject'}:{ask_id}",
        anchor=ask_id, text=edited, at=at or now_ms()))
    return await gate.resolve(ask_id, approved=approve, conversation_id=bus.MAIN,
                              edited=edited or None)


async def test_a_card_tap_is_a_user_turn_in_every_agents_history(tmp_path):
    """The user's decision on a card is visible to the model — as THEIR turn, for every
    agent — so a refused proposal is not silently re-proposed next turn."""
    await _setup(tmp_path)
    ask_id = await _open_ask(bus.MAIN, "goals.race")
    assert await _tap(ask_id, approve=False) == "rejected"
    hist = await bus.history(bus.MAIN, author="Goal")
    line = hist[-2]                                    # the tap; then the gate's "Not saved."
    assert line["role"] == "user"
    assert "tapped No" in line["content"] and "run a 10k" in line["content"]
    assert line in await bus.history(bus.MAIN, author="kip")   # same user turn, not kip's words

    # approve with an edit: history shows the value that was actually SAVED
    ask_id = await _open_ask(bus.MAIN, "goals.race")
    assert await _tap(ask_id, approve=True, edited="run a marathon") == "applied"
    line = (await bus.history(bus.MAIN, author="Goal"))[-2]
    assert line["role"] == "user" and "tapped Yes" in line["content"]
    assert "run a marathon" in line["content"] and "run a 10k" not in line["content"]


async def test_a_tap_sorts_before_a_message_typed_right_after_it(tmp_path):
    """Tap at t1, type at t2, both drained in one tick: the decision line is stamped at
    the TAP's time, so the typed message stays the last user turn (the assistant folds
    memory into history[-1] — a decision there would be sliced off and the message doubled)."""
    await _setup(tmp_path)
    ask_id = await _open_ask(bus.MAIN, "goals.race")
    t1 = now_ms()
    # the message is typed a minute after the tap: the gate's "Not saved." is stamped at real
    # time when the tap resolves, so the gap must be wider than the slowest test run, not 5 ms
    await bus.append_action(Action(
        id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.MESSAGE,
        channel="cli:local", text="no, make it a half", at=t1 + 60_000))
    assert await _tap(ask_id, approve=False, at=t1) == "rejected"      # resolves AFTER the message
    hist = await bus.history(bus.MAIN, author="Goal")
    assert hist[-1] == {"role": "user", "content": "no, make it a half"}
    assert "tapped No" in hist[-3]["content"]                           # then the gate's "Not saved."


# ── act: an approved card hands the proposal back; nothing is written ────────

async def _open_act(cid=bus.MAIN, text="Sounds good, shipping Friday.", **meta):
    out = await gate.apply(Proposal(intent="act:reply", content=text,
                                    provenance=Provenance.USER_INSTRUCTED,
                                    meta=meta or {"channel": "eng", "ts": "1.2"}),
                           conversation_id=cid, author="slack")
    assert out == "pending"
    async with db.get_db() as c:
        row = await (await c.execute("SELECT * FROM pending WHERE ask_type='act'")).fetchone()
    return row["ask_id"], row


async def test_act_always_cards_and_stores_meta(tmp_path):
    await _setup(tmp_path)
    ask_id, row = await _open_act()
    assert row["kind"].startswith("act:reply:sounds good, shipping friday.:") and row["author"] == "slack"
    assert json.loads(row["meta"]) == {"channel": "eng", "ts": "1.2"}
    card = next(r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.kind is RenderableKind.ASK)
    assert card.body["title"] == "Reply" and card.body["question"] == "Sounds good, shipping Friday."
    assert [a.token for a in card.actions] == [f"approve:{ask_id}", f"reject:{ask_id}"]


async def test_act_approve_writes_nothing_and_hands_the_proposal_back(tmp_path):
    await _setup(tmp_path)
    ask_id, _ = await _open_act()
    details = {}
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN, details=details) == "act"
    p = details["proposal"]
    assert p.intent == "act:reply" and p.content == "Sounds good, shipping Friday."
    assert p.meta == {"channel": "eng", "ts": "1.2"} and p.provenance is Provenance.USER_INSTRUCTED
    assert details["author"] == "slack" and details["conversation_id"] == bus.MAIN
    assert details["ask_id"] == ask_id
    # the ask stays OPEN until the engine reports the act done (close_act) — so a failed
    # send can be retried with a second tap; nothing is written meanwhile
    assert await gate.open_ask(ask_id) is not None
    assert await store.open_tasks() == [] and await store.recent_events() == []   # nothing written
    assert await _feedback() == []                                 # no "Saved." — the agent's line is the confirmation
    await gate.close_act(ask_id)
    assert await gate.open_ask(ask_id) is None
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "no_pending"


async def test_act_approve_with_an_edit_sends_the_users_words(tmp_path):
    await _setup(tmp_path)
    ask_id, _ = await _open_act()
    details = {}
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN,
                              edited="No, push it to Friday.", details=details) == "act"
    p = details["proposal"]
    assert p.content == "No, push it to Friday." and p.provenance is Provenance.USER_STATED
    assert p.meta == {"channel": "eng", "ts": "1.2"}                # the target is untouched


async def test_same_draft_to_two_threads_is_two_cards(tmp_path):
    await _setup(tmp_path)
    await _open_act(channel="eng", ts="1.2")
    await _open_act(channel="ops", ts="7.7")                        # same text, another thread
    async with db.get_db() as c:
        rows = await (await c.execute("SELECT meta FROM pending WHERE ask_type='act' ORDER BY created_at")).fetchall()
    assert [json.loads(r["meta"])["channel"] for r in rows] == ["eng", "ops"]
    asks = [r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.kind is RenderableKind.ASK]
    assert len(asks) == 2
    await _open_act(channel="eng", ts="1.2")                        # identical → still one card for eng
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM pending WHERE ask_type='act'")).fetchone())[0] == 2


async def test_act_reject_says_not_sent(tmp_path):
    await _setup(tmp_path)
    ask_id, _ = await _open_act()
    assert await gate.resolve(ask_id, approved=False, conversation_id=bus.MAIN) == "rejected"
    assert await _feedback() == ["Not sent."]


async def test_every_resolve_leaves_a_card_outcome_row_and_eval_reads_it(tmp_path):
    from ken.__main__ import eval_rows
    await _setup(tmp_path)
    a1 = await _open_ask(bus.MAIN, "goals.race")
    assert await gate.resolve(a1, approved=True, conversation_id=bus.MAIN) == "applied"
    a2, _ = await _open_act()
    assert await gate.resolve(a2, approved=False, conversation_id=bus.MAIN) == "rejected"
    async with db.get_db() as c:
        reasons = [r[0] for r in await (await c.execute(
            "SELECT reason FROM audit WHERE kind='card_outcome' ORDER BY at")).fetchall()]
    r0, r1 = json.loads(reasons[0]), json.loads(reasons[1])
    assert (r0["author"], r0["ask_type"], r0["outcome"]) == ("Goal", "belief", "approved") and r0["age_ms"] >= 0
    assert (r1["author"], r1["ask_type"], r1["outcome"]) == ("slack", "act", "rejected")
    # an old, still-open card counts as ignored
    async with db.get_db() as c:
        await c.execute("INSERT INTO pending (ask_id, conversation_id, kind, content, author, ask_type, created_at) "
                        "VALUES ('old', ?, 'task:x', 'x', 'slack', 'task', ?)", (bus.MAIN, now_ms() - 2 * 86_400_000))
        await c.commit()
    rows = await eval_rows()
    assert rows[("Goal", "belief")] == {"cards": 1, "approved": 1, "rejected": 0, "ignored": 0}
    assert rows[("slack", "act")] == {"cards": 1, "approved": 0, "rejected": 1, "ignored": 0}
    assert rows[("slack", "task")]["ignored"] == 1


async def test_an_agent_name_with_a_space_still_scores(tmp_path):
    from ken.__main__ import eval_rows
    await _setup(tmp_path)
    await gate.apply(Proposal(intent="act:reply", content="ok", provenance=Provenance.USER_INSTRUCTED,
                              meta={"channel": "c"}), conversation_id=bus.MAIN, author="Slack Triage")
    async with db.get_db() as c:
        ask_id = (await (await c.execute("SELECT ask_id FROM pending")).fetchone())["ask_id"]
    assert await gate.resolve(ask_id, approved=False, conversation_id=bus.MAIN) == "rejected"
    assert (await eval_rows())[("Slack Triage", "act")] == {"cards": 1, "approved": 0, "rejected": 1, "ignored": 0}


async def test_an_act_card_names_where_it_lands_and_quotes_what_it_answers(tmp_path):
    """A Reply card must say WHERE (meta.to → subject) and WHAT it answers (meta.context →
    note), on the card and again on a reprompt — a Yes is never a surprise."""
    await _setup(tmp_path)
    p = Proposal(intent="act:reply", content="On it — will confirm by 3pm.",
                 provenance=Provenance.USER_INSTRUCTED,
                 meta={"channel": "eng", "ts": "17.5", "to": "#eng · Jack",
                       "context": "“can you confirm the copy before the build cut?”"})
    assert await gate.apply(p, conversation_id=bus.MAIN, author="slack") == "pending"
    (card,) = [r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.kind is RenderableKind.ASK]
    assert card.body["title"] == "Reply" and card.body["subject"] == "#eng · Jack"
    assert card.body["note"].startswith("“can you confirm")
    assert card.body["question"] == "On it — will confirm by 3pm."
    await gate.reprompt(card.id)
    again = [r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.kind is RenderableKind.ASK][-1]
    assert again.body["subject"] == "#eng · Jack" and "can you confirm" in again.body["note"]


class _BombJudge:
    """A judge that must NOT be called."""
    async def __call__(self, proposal, *, message, existing):
        raise AssertionError("the judge ran on a hydrate batch")


async def test_hydrate_batch_never_judges_and_applies_keep_edit_drop_in_one_go(tmp_path):
    from ken.contracts import Evidence
    await _setup(tmp_path)
    items = [{"kind": "people", "name": "Ruth", "note": "PM on Crumbly"},
             {"kind": "people", "name": "Bob", "note": "a recruiter"},
             {"kind": "projects", "name": "Crumbly", "note": "the bakery app"}]
    p = Proposal(intent="batch:hydrate", content="Save 3 new records from slack?", meta={"items": items},
                 evidence=[Evidence(ref="sweep-1", source="slack")])
    out = await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_BombJudge())
    assert out == "pending" and await store.get_structured() == []
    async with db.get_db() as c:
        rows = await (await c.execute("SELECT ask_id, ask_type FROM pending")).fetchall()
    assert len(rows) == 1 and rows[0]["ask_type"] == "batch"
    ask_id = rows[0]["ask_id"]
    # the same batch again is the same card, not a second one
    assert await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_BombJudge()) == "pending"
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM pending")).fetchone())[0] == 1
    # a No saves nothing
    assert await gate.resolve(ask_id, approved=False, conversation_id=bus.MAIN) == "rejected"
    assert await store.get_structured() == []
    # raise it again; a Yes with per-item edits applies every kept record in one transaction
    await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_BombJudge())
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN, edited="[1, 2]") == "edit_invalid"
    assert await gate.open_ask(ask_id) is not None
    # a name that is not on the card (a typo) is refused, never silently kept as proposed
    typo = '[{"name": "Bobb", "drop": true}]'
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN, edited=typo) == "edit_invalid"
    assert await gate.open_ask(ask_id) is not None and await store.get_structured() == []
    # a bad shape is refused too: a note must be text, a drop a bool
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN,
                              edited='{"Bob": {"note": 5}}') == "edit_invalid"
    edited = '{"Ruth": "PM on Crumbly; owns the toolbar", "Bob": null}'      # the map form: null = drop
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN, edited=edited) == "applied"
    got = {e.kind: e for e in await store.get_structured()}
    assert set(got) == {"people.ruth", "projects.crumbly"}
    assert got["people.ruth"].content == "PM on Crumbly; owns the toolbar"
    assert got["people.ruth"].provenance is Provenance.USER_STATED and got["people.ruth"].evidence == []
    assert got["projects.crumbly"].provenance is Provenance.USER_INSTRUCTED
    assert got["projects.crumbly"].evidence[0].ref == "sweep-1" and got["projects.crumbly"].subject == "Crumbly"
    assert await gate.open_ask(ask_id) is None


def test_batch_edits_one_grammar():
    from ken.consent.gate import _batch_edits
    assert _batch_edits("") == {}
    assert _batch_edits('{"Bob": null}') == {"bob": {"note": "", "drop": True}}
    assert _batch_edits('{"Bob": ""}') == {"bob": {"note": "", "drop": False}}          # "" = keep as proposed
    assert _batch_edits('{"Bob": {"drop": true}}') == {"bob": {"note": "", "drop": True}}
    assert _batch_edits('[{"name": "Bob", "note": " x "}]') == {"bob": {"note": "x", "drop": False}}
    import pytest
    for bad in ('[{"note": "x"}]', '{"Bob": 5}', '[{"name": "Bob", "drop": "yes"}]', '"x"'):
        with pytest.raises(ValueError):
            _batch_edits(bad)


async def test_batch_never_overwrites_a_record_that_exists_at_resolve_time(tmp_path):
    from ken.contracts import Evidence
    await _setup(tmp_path)
    a = Proposal(intent="batch:hydrate", content="Save 2 from slack?", evidence=[Evidence(ref="a", source="slack")],
                 meta={"items": [{"kind": "people", "name": "Ruth", "note": "PM on Crumbly"},
                                 {"kind": "people", "name": "Ann", "note": "designer"}]})
    b = Proposal(intent="batch:hydrate", content="Save 2 from mail?", evidence=[Evidence(ref="b", source="mail")],
                 meta={"items": [{"kind": "people", "name": "Ruth", "note": "a recruiter"},
                                 {"kind": "people", "name": "Zed", "note": "new hire"}]})
    for p in (a, b):
        assert await gate.apply(p, conversation_id=bus.MAIN, author="kip") == "pending"
    async with db.get_db() as c:
        ids = [r["ask_id"] for r in await (await c.execute("SELECT ask_id FROM pending ORDER BY created_at")).fetchall()]
    assert len(ids) == 2
    assert await gate.resolve(ids[0], approved=True, conversation_id=bus.MAIN) == "applied"
    assert await gate.resolve(ids[1], approved=True, conversation_id=bus.MAIN) == "applied"
    got = {e.kind: e.content for e in await store.get_structured()}
    assert got == {"people.ruth": "PM on Crumbly", "people.ann": "designer", "people.zed": "new hire"}   # held, not replaced
    # …unless the user rewrote it on the second card: their own word wins
    await gate.apply(b, conversation_id=bus.MAIN, author="kip")
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN,
                              edited='{"Ruth": "recruiter at Acme, ignore"}') == "applied"
    assert {e.kind: e.content for e in await store.get_structured()}["people.ruth"] == "recruiter at Acme, ignore"


async def test_batch_with_malformed_items_is_refused_not_a_crash(tmp_path):
    await _setup(tmp_path)
    bad = Proposal(intent="batch:import", content="x", meta={"items": [{"title": "x"}]})
    assert await gate.apply(bad, conversation_id=bus.MAIN, author="drop-in") == "invalid"
    assert await gate.apply(Proposal(intent="batch:import", content="x"), conversation_id=bus.MAIN, author="drop-in") == "invalid"
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM pending")).fetchone())[0] == 0


async def test_reprompt_shows_a_batch_cards_items(tmp_path):
    await _setup(tmp_path)
    p = Proposal(intent="batch:hydrate", content="Save 1 from slack?",
                 meta={"items": [{"kind": "people", "name": "Ruth", "note": "PM on Crumbly"}]})
    await gate.apply(p, conversation_id=bus.MAIN, author="kip")
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    await gate.reprompt(ask_id)
    cards = [r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.id == ask_id]
    assert len(cards) == 2 and all("- Ruth (person): PM on Crumbly" in c.body["note"] for c in cards)


# ── identity: the same person under two ids ───────────────────────────────────

async def _person(kind, subject, content):
    from ken.contracts import Evidence, StructuredEntry
    await store.upsert_structured(StructuredEntry(id=kind, kind=kind, subject=subject, content=content,
                                                  evidence=[Evidence(ref="m0", source="user")],
                                                  provenance=Provenance.USER_STATED, updated_at=1), change_note="t")


async def _pending_rows():
    async with db.get_db() as c:
        return await (await c.execute("SELECT ask_id, kind, origin, ask_type, meta FROM pending ORDER BY created_at")).fetchall()


async def _one_pending():
    rows = await _pending_rows()
    assert len(rows) == 1, [dict(r) for r in rows]
    return rows[0]


async def _card(ask_id):
    return next(r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.id == ask_id)


async def test_a_cross_id_collapse_on_a_person_becomes_an_update_card_and_a_yes_merges(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    await _person("people.lucas", "Lucas", "Tallyhouse external partner contact; supports integrations")
    p = Proposal(intent="structured:people.lucas_tallyhouse", content="Tallyhouse vendor contact on the attribution model",
                 evidence=[__import__("ken.contracts", fromlist=["Evidence"]).Evidence(ref="s9", source="slack")])
    out = await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.lucas")))
    assert out == "conflict"
    row = await _one_pending()
    assert row["kind"] == "identity:people.lucas:people.lucas_tallyhouse" and row["ask_type"] == "identity"
    card = await _card(row["ask_id"])
    assert card.body["title"] == "Update person" and card.body["subject"] == "Lucas"
    assert card.body["existing"].startswith("Tallyhouse external") and "the judge" in card.body["note"]
    assert await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    (e,) = await store.get_structured("people.")
    assert e.kind == "people.lucas" and e.subject == "Lucas"                       # merged, not duplicated
    assert e.content == "Tallyhouse external partner contact; supports integrations. Tallyhouse vendor contact on the attribution model"
    assert {x.ref for x in e.evidence} == {"m0", "s9"} and e.provenance is Provenance.USER_STATED   # kept: user-authored
    # an EDIT rewrites the note in the user's words
    p2 = Proposal(intent="structured:people.lucas_at_tallyhouse", content="whatever")
    await gate.apply(p2, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.lucas")))
    row = await _one_pending()
    assert await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN, edited="Lucas at Tallyhouse, our attribution vendor") == "applied"
    (e,) = await store.get_structured("people.")
    assert e.content == "Lucas at Tallyhouse, our attribution vendor" and e.provenance is Provenance.USER_STATED and e.subject == "Lucas"


async def test_a_name_twin_asks_even_when_the_judge_keeps_but_a_shared_first_name_does_not(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    await _person("people.lucas_of_tallyhouse", "Lucas (Tallyhouse)", "Tallyhouse contact")     # punctuation in the subject
    await _person("people.michael_novak", "Michael Novak", "Crumbly iOS")
    p = Proposal(intent="structured:people.lucas_tallyhouse", content="Tallyhouse vendor contact")
    assert await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.keep())) == "conflict"
    row = await _one_pending()
    assert row["kind"] == "identity:people.lucas_of_tallyhouse:people.lucas_tallyhouse" and "the same name" in (await _card(row["ask_id"])).body["note"]
    await gate.resolve(row["ask_id"], approved=False, conversation_id=bus.MAIN)
    row = await _one_pending(); await gate.resolve(row["ask_id"], approved=False, conversation_id=bus.MAIN)
    p = Proposal(intent="structured:people.michael_davis", content="Support-side contact")
    assert await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.keep())) == "pending"
    row = await _one_pending()
    assert row["kind"] == "people.michael_davis"                                       # a plain "Save person" card


async def test_an_identity_card_never_evicts_a_forget_card_and_survives_the_target_moving(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    await _person("people.lucas", "Lucas", "Tallyhouse contact")
    await gate.apply(Proposal(intent="retire:people.lucas", content="forget Lucas"), conversation_id=bus.MAIN, author="kip")
    p = Proposal(intent="structured:people.lucas_tallyhouse", content="Tallyhouse vendor contact")
    await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.lucas")))
    rows = await _pending_rows()
    assert [r["ask_type"] for r in rows] == ["retire", "identity"]                       # both open
    # the same pair again is the same card, not a second one
    await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.lucas")))
    assert len(await _pending_rows()) == 2
    # the target moves under the card: a Yes re-asks against the current value, writes nothing
    await _person("people.lucas", "Lucas", "Tallyhouse contact, now at Acme")
    ident = rows[1]["ask_id"]
    assert await gate.resolve(ident, approved=True, conversation_id=bus.MAIN) == "restaled"
    fresh = [r for r in await _pending_rows() if r["ask_type"] == "identity"]
    assert len(fresh) == 1 and fresh[0]["ask_id"] != ident
    assert (await _card(fresh[0]["ask_id"])).body["existing"] == "Tallyhouse contact, now at Acme"
    assert {e.content for e in await store.get_structured("people.")} == {"Tallyhouse contact, now at Acme"}
    # the target is forgotten under the card: a Yes cannot resurrect it — it asks to save as new
    await gate.resolve(rows[0]["ask_id"], approved=True, conversation_id=bus.MAIN)      # the forget
    assert await gate.resolve(fresh[0]["ask_id"], approved=True, conversation_id=bus.MAIN) == "restaled"
    (row,) = await _pending_rows()
    assert row["kind"] == "people.lucas_tallyhouse" and row["ask_type"] == "belief"
    assert await store.get_structured("people.") == []


async def test_reprompt_keeps_an_identity_cards_versions(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    await _person("people.lucas", "Lucas", "Tallyhouse contact")
    await gate.apply(Proposal(intent="structured:people.lucas_tallyhouse", content="vendor"),
                     conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.lucas")))
    row = await _one_pending()
    await gate.reprompt(row["ask_id"])
    cards = [r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.id == row["ask_id"]]
    assert len(cards) == 2 and all(c.body["title"] == "Update person" and c.body["existing"] == "Tallyhouse contact"
                                   and "Looks like the same person" in c.body["note"] for c in cards)


async def test_a_same_kind_conflict_and_a_task_collapse_are_unchanged_by_identity(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    await _person("people.lucas", "Lucas", "Tallyhouse contact")
    p = Proposal(intent="structured:people.lucas", content="Tallyhouse contact, now at Acme")
    assert await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.conflict("people.lucas"))) == "conflict"
    assert (await _one_pending())["origin"] == "judge-conflict"


async def test_an_identity_ask_and_a_plain_ask_for_the_same_record_are_one_question(tmp_path):
    from ken.consent import judge
    from ken.contracts import Evidence
    await _setup(tmp_path)
    await _person("people.lucas", "Lucas", "Tallyhouse contact")
    p = Proposal(intent="structured:people.lucas_tallyhouse", content="Tallyhouse vendor contact",
                 evidence=[Evidence(ref="s1", source="slack")], meta={"task_id": "T1"})
    await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.lucas")))
    row = await _one_pending()
    # No → the save-as-new card keeps the task id (a client ties it to the task)
    assert await gate.resolve(row["ask_id"], approved=False, conversation_id=bus.MAIN) == "reasked"
    row = await _one_pending()
    assert row["kind"] == "people.lucas_tallyhouse" and json.loads(row["meta"]) == {"task_id": "T1"}
    # the agent retries the same proposal: the open save-as-new card IS the question — no second card
    await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.lucas")))
    row = await _one_pending()
    assert row["kind"] == "people.lucas_tallyhouse"
    assert await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN, edited="Lucas from Tallyhouse") == "applied"
    assert {e.kind: e.content for e in await store.get_structured("people.")}["people.lucas_tallyhouse"] == "Lucas from Tallyhouse"
    # a Yes with an edit while the target vanished: the edit travels to the save-as-new card
    await _person("people.ann", "Ann", "designer")
    await gate.apply(Proposal(intent="structured:people.ann_lee", content="Ann Lee, designer on Crumbly"),
                     conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.keep()))
    row = await _one_pending()
    assert row["kind"] == "identity:people.ann:people.ann_lee"
    await store.delete_structured("people.ann")
    assert await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN, edited="Ann Lee, our Crumbly designer") == "restaled"
    row = await _one_pending()
    (card,) = [r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.id == row["ask_id"]]
    assert row["kind"] == "people.ann_lee" and card.body["question"] == "Ann Lee, our Crumbly designer"


async def test_a_revision_keeps_the_previous_content_in_the_changelog(tmp_path):
    await _setup(tmp_path)
    await _person("people.lucas", "Lucas", "Tallyhouse contact")
    await _person("people.lucas", "Lucas", "Tallyhouse contact, now at Acme")
    (e,) = await store.get_structured("people.")
    assert e.content == "Tallyhouse contact, now at Acme"
    async with db.get_db() as c:
        (raw,) = await (await c.execute("SELECT changelog FROM structured WHERE id='people.lucas'")).fetchone()
    log = json.loads(raw)
    assert len(log) == 2 and log[1]["prev"] == "Tallyhouse contact" and "prev" not in log[0]


# ── links and assets ──────────────────────────────────────────────────────────

async def _links(*kinds):
    return sorted((lk["src"], lk["dst"], lk["rel"]) for lk in await store.links_of(list(kinds)))


async def test_links_ride_on_the_card_are_written_on_yes_never_dangle_and_die_with_the_record(tmp_path):
    from ken.contracts import Evidence
    await _setup(tmp_path)
    await _person("people.ruth", "Ruth", "the only Crumbly designer")
    p = Proposal(intent="structured:projects.recipes", content="recipes in-app; Q4 P0",
                 evidence=[Evidence(ref="m1", source="user")],
                 meta={"links": [{"dst": "people.ruth", "rel": "member", "name": "Ruth"},
                                 {"dst": "people.ghost", "rel": "member", "name": "Ghost"},
                                 {"dst": "people.ruth", "rel": "friend", "name": "Ruth"}]})   # bad rel
    assert await gate.apply(p, conversation_id=bus.MAIN, author="kip") == "pending"
    row = await _one_pending()
    card = await _card(row["ask_id"])
    assert "link: member \u2192 Ruth" in card.body["note"] and "Ghost" in card.body["note"]
    assert await _links("projects.recipes") == []                                     # nothing before the Yes
    assert await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    assert await _links("projects.recipes") == [("people.ruth", "projects.recipes", "member")]   # stored in the rel's direction; no dangling, no bad rel
    # an asset carries its url; a later rewrite without a url keeps it
    a = Proposal(intent="structured:assets.recipes_figma", content="the Recipes design file",
                 meta={"url": "https://figma.com/file/abc", "links": [{"dst": "projects.recipes", "rel": "part_of", "name": "Recipes"}]})
    await gate.apply(a, conversation_id=bus.MAIN, author="kip")
    row = await _one_pending()
    assert (await _card(row["ask_id"])).body["title"] == "Save asset" and "url: https://figma.com/file/abc" in (await _card(row["ask_id"])).body["note"]
    await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN)
    (asset,) = await store.get_structured("assets.")
    assert asset.url == "https://figma.com/file/abc"
    await gate.apply(Proposal(intent="structured:assets.recipes_figma", content="the Recipes design file, v2"),
                     conversation_id=bus.MAIN, author="kip")
    row = await _one_pending(); await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN)
    (asset,) = await store.get_structured("assets.")
    assert asset.content.endswith("v2") and asset.url == "https://figma.com/file/abc"
    assert await _links("projects.recipes") == [("assets.recipes_figma", "projects.recipes", "part_of"),
                                               ("people.ruth", "projects.recipes", "member")]
    # retiring a record takes its links with it, both directions
    await store.delete_structured("people.ruth")
    assert await _links("projects.recipes") == [("assets.recipes_figma", "projects.recipes", "part_of")]


async def test_a_batch_links_its_own_items_once_they_exist(tmp_path):
    await _setup(tmp_path)
    items = [{"kind": "projects", "name": "Recipes", "note": "recipes in-app",
              "links": [{"dst": "people.ruth", "rel": "member", "name": "Ruth"},
                        {"dst": "people.ann", "rel": "member", "name": "Ann"}]},
             {"kind": "people", "name": "Ruth", "note": "designer"},
             {"kind": "people", "name": "Ann", "note": "PM"},
             {"kind": "assets", "name": "Ledger", "note": "the finance sheet", "url": "https://sheets/1",
              "links": [{"dst": "projects.recipes", "rel": "part_of", "name": "Recipes"}]}]
    await gate.apply(Proposal(intent="batch:hydrate", content="Save 4?", meta={"items": items}),
                     conversation_id=bus.MAIN, author="kip")
    row = await _one_pending()
    note = (await _card(row["ask_id"])).body["note"]
    assert "    link: member \u2192 Ruth" in note and "    url: https://sheets/1" in note
    assert await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN, edited='{"Ann": null}') == "applied"
    assert await _links("projects.recipes") == [("assets.ledger", "projects.recipes", "part_of"),
                                               ("people.ruth", "projects.recipes", "member")]   # Ann was dropped: no link
    assert {e.kind: e.url for e in await store.get_structured("assets.")} == {"assets.ledger": "https://sheets/1"}


async def test_links_land_on_a_duplicate_note_show_on_conflict_and_identity_cards_and_respect_direction(tmp_path):
    from ken.consent import judge
    from ken.contracts import orient
    await _setup(tmp_path)
    await _person("people.ruth", "Ruth", "the only Crumbly designer")
    await _person("projects.recipes", "Recipes", "recipes in-app")
    # the same note again, with a link: in a consented namespace the link ASKS (a change to
    # the record), and a Yes writes the unchanged note plus the link
    p = Proposal(intent="structured:people.ruth", content="the only Crumbly designer",
                 meta={"links": [{"dst": "projects.recipes", "rel": "member", "name": "Recipes"}]})
    assert await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.ruth"))) == "pending"
    assert await _links("people.ruth") == []
    row = await _one_pending()
    card = await _card(row["ask_id"])
    assert card.body["title"] == "Update person" and card.body["note"] == "Add links:\nlink: member \u2192 Recipes"
    assert await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    assert await _links("people.ruth") == [("people.ruth", "projects.recipes", "member")]
    assert (await store.get_structured("people.ruth"))[0].content == "the only Crumbly designer"
    # a fast-track namespace (preferences) lands them without a card
    await _person("preferences.tooling", "you", "minimal tools")
    p = Proposal(intent="structured:preferences.tooling", content="minimal tools",
                 meta={"links": [{"dst": "projects.recipes", "rel": "part_of", "name": "Recipes"}]})
    assert await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("preferences.tooling"))) == "collapsed"
    # a conflict card shows the link a Yes will write
    p = Proposal(intent="structured:people.ruth", content="designer; now also on iOS",
                 meta={"links": [{"dst": "projects.recipes", "rel": "owner", "name": "Recipes"}]})
    assert await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.conflict("people.ruth"))) == "conflict"
    row = await _one_pending()
    assert "link: owner \u2192 Recipes" in (await _card(row["ask_id"])).body["note"]
    await gate.resolve(row["ask_id"], approved=False, conversation_id=bus.MAIN)
    # an identity card shows the link too, and a Yes writes it onto the EXISTING record
    p = Proposal(intent="structured:people.ruth_van_dalen", content="Crumbly designer",
                 meta={"links": [{"dst": "projects.recipes", "rel": "owner", "name": "Recipes"}]})
    assert await gate.apply(p, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.ruth"))) == "conflict"
    row = await _one_pending()
    assert row["ask_type"] == "identity" and "link: owner \u2192 Recipes" in (await _card(row["ask_id"])).body["note"]
    assert await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    assert await _links("people.ruth") == [("people.ruth", "projects.recipes", "member"), ("people.ruth", "projects.recipes", "owner")]
    # direction is a property of the rel, whichever way it was named; a pair that does not fit is dropped
    assert orient("projects.recipes", "people.ruth", "member") == ("people.ruth", "projects.recipes", "member")
    assert orient("projects.recipes", "assets.figma", "asset") == ("assets.figma", "projects.recipes", "part_of")
    assert orient("people.ruth", "people.ann", "member") is None
    assert await store.set_links("people.ruth", [{"dst": "people.ann", "rel": "member"}]) == 0


async def test_a_batch_writes_links_for_a_held_item_too(tmp_path):
    await _setup(tmp_path)
    await _person("people.ruth", "Ruth", "designer")
    items = [{"kind": "projects", "name": "Recipes", "note": "recipes in-app",
              "links": [{"dst": "people.ruth", "rel": "member", "name": "Ruth"}]}]
    await gate.apply(Proposal(intent="batch:hydrate", content="Save 1?", meta={"items": items}),
                     conversation_id=bus.MAIN, author="kip")
    await _person("projects.recipes", "Recipes", "recipes in-app; Q4 P0")          # appears under the card
    row = await _one_pending()
    assert await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    (e,) = await store.get_structured("projects.")
    assert e.content == "recipes in-app; Q4 P0"                                    # held, not overwritten
    assert await _links("projects.recipes") == [("people.ruth", "projects.recipes", "member")]   # the promised link landed


async def test_a_links_card_writes_links_only_asks_once_and_merges_a_second_proposal(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    await _person("people.ruth", "Ruth", "the only Crumbly designer")
    await _person("projects.recipes", "Recipes", "recipes in-app")
    await _person("projects.crumbly", "Crumbly", "the app")
    before = (await store.get_structured("people.ruth"))[0]
    p1 = Proposal(intent="structured:people.ruth", content="the only Crumbly designer",
                  meta={"links": [{"dst": "projects.recipes", "rel": "member", "name": "Recipes"}]})
    assert await gate.apply(p1, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.ruth"))) == "pending"
    row = await _one_pending()
    assert row["kind"] == "links:people.ruth" and row["ask_type"] == "links"
    # a second proposal with MORE links while the card is open merges into it: still one card
    p2 = Proposal(intent="structured:people.ruth", content="the only Crumbly designer",
                  meta={"links": [{"dst": "projects.recipes", "rel": "member", "name": "Recipes"},
                                  {"dst": "projects.crumbly", "rel": "owner", "name": "Crumbly"}]})
    assert await gate.apply(p2, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.ruth"))) == "pending"
    row = await _one_pending()
    assert [lk["dst"] for lk in json.loads(row["meta"])["links"]] == ["projects.recipes", "projects.crumbly"]
    cards = [r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.id == row["ask_id"]]
    assert "link: owner \u2192 Crumbly" in cards[-1].body["note"]
    # a Yes writes the links ONLY: subject, evidence, provenance and note are untouched
    assert await gate.resolve(row["ask_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    after = (await store.get_structured("people.ruth"))[0]
    assert (after.subject, after.provenance, [e.ref for e in after.evidence], after.content) == \
           (before.subject, before.provenance, [e.ref for e in before.evidence], before.content)
    assert await _links("people.ruth") == [("people.ruth", "projects.crumbly", "owner"), ("people.ruth", "projects.recipes", "member")]
    # the same mention again, links already stored: a plain collapse, no card
    assert await gate.apply(p2, conversation_id=bus.MAIN, author="kip", judge=_FixedJudge(judge.collapse("people.ruth"))) == "collapsed"
    assert await _pending_rows() == []
    # WORLD content never lands links silently: it asks on its own links card too
    w = Proposal(intent="structured:people.ruth", content="the only Crumbly designer", provenance=Provenance.WORLD,
                 meta={"links": [{"dst": "projects.crumbly", "rel": "client", "name": "Crumbly"}]})
    assert await gate.apply(w, conversation_id=bus.MAIN, author="slack", judge=_FixedJudge(judge.collapse("people.ruth"))) == "pending"
    assert (await _one_pending())["kind"] == "links:people.ruth"
    # a No links nothing
    row = await _one_pending()
    assert await gate.resolve(row["ask_id"], approved=False, conversation_id=bus.MAIN) == "rejected"
    assert await _links("people.ruth") == [("people.ruth", "projects.crumbly", "owner"), ("people.ruth", "projects.recipes", "member")]
