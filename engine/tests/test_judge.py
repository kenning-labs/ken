"""The judged gate — the machine skeptic before the consent gate. (ARCHITECTURE.md)

Every route is exercised through a DETERMINISTIC fake judge injected into the gate,
so no test touches the network. One case parses the real LlmJudge against a fake LLM
to prove it reads a verdict and never smuggles content. Another feeds a synthetic
non-Assistant proposal to prove the judge is domain-agnostic.
"""
import json
from types import SimpleNamespace

import pytest

from ken import db
from ken.agents import Proposal
from ken.consent import gate, judge
from ken.consent.judge import Decision, DecisionsJudge, FallbackJudge, JudgeError, LlmJudge, Verdict
from ken.contracts import Evidence, Provenance, RenderableKind, StructuredEntry
from ken.conversation import bus
from ken.memory import store


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


class FakeJudge:
    """Returns a canned decision and records what it saw — the deterministic stand-in
    for the LLM judge, so a test drives any route with no network."""
    def __init__(self, decision: Decision):
        self.decision = decision
        self.seen = None

    async def __call__(self, proposal, *, message, existing):
        self.seen = {"proposal": proposal, "message": message, "existing": existing}
        return self.decision


class BoomJudge:
    async def __call__(self, proposal, *, message, existing):
        raise JudgeError("model down")


async def _outbox_texts(cid=bus.MAIN):
    return [r.body.get("text") for _, r in await bus.read_outbox(cid, 0)
            if r.kind is RenderableKind.ADVICE]


async def _has_ask(cid=bus.MAIN):
    return any(r.kind is RenderableKind.ASK for _, r in await bus.read_outbox(cid, 0))


async def _last_ask_body(cid=bus.MAIN):
    """The body of the MOST RECENT ask card posted — after a re-raise the outbox holds
    the old card too, so we read the newest one."""
    asks = [r.body for _, r in await bus.read_outbox(cid, 0) if r.kind is RenderableKind.ASK]
    return asks[-1] if asks else None


# ── keep: routes to today's write ────────────────────────────────────────────

async def test_keep_writes_the_task(tmp_path):
    await _setup(tmp_path)
    j = FakeJudge(judge.keep())
    out = await gate.apply(Proposal(intent="task", content="book the dentist"),
                           conversation_id=bus.MAIN, author="Assistant", judge=j, message="book it")
    assert out == "task"
    assert [t.text for t in await store.open_tasks()] == ["book the dentist"]
    assert j.seen["message"] == "book it"           # the message reached the judge


async def test_keep_task_never_shows_a_card(tmp_path):
    await _setup(tmp_path)
    await gate.apply(Proposal(intent="task", content="call mum"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.keep()), message="call mum")
    assert not await _has_ask()                      # task keep is implicit fast-track


async def test_keep_event_never_shows_a_card(tmp_path):
    await _setup(tmp_path)
    out = await gate.apply(Proposal(intent="event", content="ran 10k", at=42),
                           conversation_id=bus.MAIN, author="Assistant",
                           judge=FakeJudge(judge.keep()), message="I ran 10k")
    assert out == "event"
    assert not await _has_ask()


async def test_keep_fast_track_structured_applies_silently(tmp_path):
    await _setup(tmp_path)
    out = await gate.apply(Proposal(intent="structured:preferences.tone", content="be brief"),
                           conversation_id=bus.MAIN, author="Assistant",
                           judge=FakeJudge(judge.keep()), message="keep it brief")
    assert out == "applied"
    assert not await _has_ask()
    assert [e.content for e in await store.get_structured("preferences.")] == ["be brief"]


# ── collapse: no duplicate; a new time is adopted onto the existing task ──────

async def test_collapse_does_not_duplicate_and_adopts_a_time(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("call the plumber", Provenance.AGENT_INFERRED)
    j = FakeJudge(judge.collapse(tid))
    out = await gate.apply(Proposal(intent="task", content="ring the plumber", at=5000),
                           conversation_id=bus.MAIN, author="Assistant", judge=j,
                           message="ring the plumber at 5")
    assert out == "collapsed"
    tasks = await store.open_tasks()
    assert len(tasks) == 1                            # no second copy
    assert tasks[0].deadline == 5000            # the new time was adopted


async def test_collapse_keeps_an_existing_time(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("call the plumber", Provenance.AGENT_INFERRED,
                                  deadline=1000)
    await gate.apply(Proposal(intent="task", content="ring the plumber", at=9999),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.collapse(tid)), message="ring plumber")
    tasks = await store.open_tasks()
    assert tasks[0].deadline == 1000            # never silently re-scheduled


# ── collapse safety: a mis-set target can NEVER silently drop a write ─────────

async def test_collapse_with_a_missing_task_target_falls_back_to_keep(tmp_path):
    await _setup(tmp_path)
    # the judge names a duplicate task that does not exist (a garbled/hallucinated or
    # since-completed id). The write must NOT be dropped with a false "collapsed" — it
    # falls back to keep and lands.
    j = FakeJudge(judge.collapse("ghost-task"))
    out = await gate.apply(Proposal(intent="task", content="book the dentist"),
                           conversation_id=bus.MAIN, author="Assistant", judge=j,
                           message="add book the dentist")
    assert out == "task"                              # kept, not collapsed
    assert [t.text for t in await store.open_tasks()] == ["book the dentist"]


async def test_collapse_with_a_falsy_target_falls_back_to_keep(tmp_path):
    await _setup(tmp_path)
    # a collapse verdict with no target_id at all — still never a silent drop.
    j = FakeJudge(Decision(Verdict.COLLAPSE, target_id=None))
    out = await gate.apply(Proposal(intent="task", content="water the plants"),
                           conversation_id=bus.MAIN, author="Assistant", judge=j,
                           message="add water the plants")
    assert out == "task"
    assert [t.text for t in await store.open_tasks()] == ["water the plants"]


async def test_collapse_onto_a_real_task_still_collapses(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("call the plumber", Provenance.AGENT_INFERRED)
    j = FakeJudge(judge.collapse(tid))
    out = await gate.apply(Proposal(intent="task", content="ring the plumber", at=5000),
                           conversation_id=bus.MAIN, author="Assistant", judge=j,
                           message="ring the plumber at 5")
    assert out == "collapsed"                         # a confirmed duplicate still collapses
    tasks = await store.open_tasks()
    assert len(tasks) == 1 and tasks[0].deadline == 5000


async def test_collapse_with_a_missing_structured_target_falls_back_to_keep(tmp_path):
    await _setup(tmp_path)
    # a structured collapse whose claimed duplicate belief is absent — write it, do not
    # drop it with a false "collapsed".
    j = FakeJudge(judge.collapse("preferences.ghost"))
    out = await gate.apply(
        Proposal(intent="structured:preferences.tone", content="be brief"),
        conversation_id=bus.MAIN, author="Assistant", judge=j, message="keep it brief")
    assert out == "applied"                           # written through the keep path
    assert [e.content for e in await store.get_structured("preferences.")] == ["be brief"]


async def test_collapse_with_a_missing_event_target_falls_back_to_keep(tmp_path):
    await _setup(tmp_path)
    j = FakeJudge(judge.collapse("evt-ghost"))
    out = await gate.apply(Proposal(intent="event", content="ran 10k", at=42),
                           conversation_id=bus.MAIN, author="Assistant", judge=j,
                           message="I ran 10k")
    assert out == "event"                             # written through the keep path
    async with db.get_db() as c:
        rows = await (await c.execute("SELECT summary FROM events")).fetchall()
    assert [r["summary"] for r in rows] == ["ran 10k"]


# ── no filter verdict exists: the judge never drops a write ───────────────────

async def test_filter_verdict_is_gone_from_the_enum():
    # the judge has exactly three verdicts now — no 'filter'. Proves the drop route was
    # removed cleanly (an old 'filter' string no longer maps to any verdict).
    assert {v.value for v in Verdict} == {"keep", "collapse", "conflict"}
    assert not hasattr(judge, "drop")


async def test_llm_judge_rejects_a_filter_verdict():
    # a model that still answers 'filter' is now an unknown verdict — a loud JudgeError,
    # never a silent drop.
    with pytest.raises(JudgeError):
        await LlmJudge(ScriptedLLM('{"verdict":"filter","reason":"noise"}'))(
            Proposal(intent="task", content="x"), message="m", existing=[])


# ── conflict: asks, tags the pending row, snapshots the belief ───────────────

async def test_conflict_raises_an_ask_with_a_snapshot(tmp_path):
    await _setup(tmp_path)
    # an existing fast-track belief the new proposal contradicts.
    await gate.apply(Proposal(intent="structured:preferences.coffee", content="black, no sugar"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.keep()), message="I take my coffee black")
    j = FakeJudge(judge.conflict("preferences.coffee"))
    out = await gate.apply(
        Proposal(intent="structured:preferences.coffee", content="lots of milk and sugar"),
        conversation_id=bus.MAIN, author="Assistant", judge=j, message="I love milky sweet coffee")
    assert out == "conflict"
    assert await _has_ask()                          # a Yes/No card is up
    async with db.get_db() as c:
        row = await (await c.execute(
            "SELECT origin, existing_snapshot FROM pending WHERE kind='preferences.coffee'"
        )).fetchone()
    assert row["origin"] == "judge-conflict"
    snap = json.loads(row["existing_snapshot"])
    assert snap["content"] == "black, no sugar"      # the belief being challenged
    # the belief itself is untouched until the user answers
    assert [e.content for e in await store.get_structured("preferences.coffee")] == ["black, no sugar"]


async def test_conflict_card_carries_existing_and_proposed(tmp_path):
    await _setup(tmp_path)
    await gate.apply(Proposal(intent="structured:preferences.coffee", content="black, no sugar"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.keep()), message="black")
    await gate.apply(
        Proposal(intent="structured:preferences.coffee", content="milk and sugar"),
        conversation_id=bus.MAIN, author="Assistant",
        judge=FakeJudge(judge.conflict("preferences.coffee")), message="milky")
    body = await _last_ask_body()
    assert body["existing"] == "black, no sugar"     # the current belief the card shows
    assert body["proposed"] == "milk and sugar"      # the new value, in a generic field


async def test_plain_consent_ask_carries_no_existing_or_proposed(tmp_path):
    await _setup(tmp_path)
    # a non-fast-track structured proposal (no judge) raises a PLAIN consent ask — it must
    # NOT carry existing/proposed, so the interface renders it exactly as before.
    await gate.apply(Proposal(intent="structured:goals.race", content="run a 10k"),
                     conversation_id=bus.MAIN, author="Goal")
    body = await _last_ask_body()
    assert body is not None and "existing" not in body and "proposed" not in body


async def test_conflict_snapshot_is_the_real_current_belief(tmp_path):
    await _setup(tmp_path)
    # write, then OVERWRITE the belief; the conflict snapshot must be the LATEST value,
    # proving the id=kind singleton resolves the real current belief, not a stale one.
    for value, msg in (("black", "black"), ("oat milk", "oat now")):
        await gate.apply(Proposal(intent="structured:preferences.coffee", content=value),
                         conversation_id=bus.MAIN, author="Assistant",
                         judge=FakeJudge(judge.keep()), message=msg)
    await gate.apply(Proposal(intent="structured:preferences.coffee", content="espresso"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.conflict("preferences.coffee")), message="espresso")
    async with db.get_db() as c:
        (snap,) = await (await c.execute(
            "SELECT existing_snapshot FROM pending WHERE kind='preferences.coffee'")).fetchone()
    assert json.loads(snap)["content"] == "oat milk"   # the CURRENT belief, not the first


# ── a differing structured value is a conflict, not a silent collapse (Finding 5) ──

async def test_collapse_of_a_differing_structured_value_becomes_a_conflict(tmp_path):
    await _setup(tmp_path)
    await gate.apply(Proposal(intent="structured:preferences.coffee", content="black, no sugar"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.keep()), message="black")
    # the judge says collapse, but the value DIFFERS — the gate must ASK, never no-op.
    out = await gate.apply(
        Proposal(intent="structured:preferences.coffee", content="milk and two sugars"),
        conversation_id=bus.MAIN, author="Assistant",
        judge=FakeJudge(judge.collapse("preferences.coffee")), message="sweet")
    assert out == "conflict"
    assert await _has_ask()
    # the stored belief is untouched — the differing value was NOT silently dropped
    assert [e.content for e in await store.get_structured("preferences.coffee")] == ["black, no sugar"]
    body = await _last_ask_body()
    assert body["existing"] == "black, no sugar" and body["proposed"] == "milk and two sugars"


async def test_collapse_of_an_identical_structured_value_stays_a_collapse(tmp_path):
    await _setup(tmp_path)
    await gate.apply(Proposal(intent="structured:preferences.coffee", content="black, no sugar"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.keep()), message="black")
    # same value (case/space only) -> a genuine duplicate: collapse, no card.
    out = await gate.apply(
        Proposal(intent="structured:preferences.coffee", content="Black,  No Sugar"),
        conversation_id=bus.MAIN, author="Assistant",
        judge=FakeJudge(judge.collapse("preferences.coffee")), message="black")
    assert out == "collapsed"
    assert not await _has_ask()
    assert [e.content for e in await store.get_structured("preferences.coffee")] == ["black, no sugar"]


# ── resolve re-validates a conflict Yes (the stale-approval guard) ────────────

async def _open_conflict(cid=bus.MAIN, *, existing="black", proposed="milk"):
    await gate.apply(Proposal(intent="structured:preferences.coffee", content=existing),
                     conversation_id=cid, author="Assistant",
                     judge=FakeJudge(judge.keep()), message=existing)
    await gate.apply(Proposal(intent="structured:preferences.coffee", content=proposed),
                     conversation_id=cid, author="Assistant",
                     judge=FakeJudge(judge.conflict("preferences.coffee")), message=proposed)
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute(
            "SELECT ask_id FROM pending WHERE kind='preferences.coffee'")).fetchone()
    return ask_id


async def test_conflict_yes_applies_when_the_belief_is_unchanged(tmp_path):
    await _setup(tmp_path)
    ask_id = await _open_conflict()
    out = await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN)
    assert out == "applied"
    assert [e.content for e in await store.get_structured("preferences.coffee")] == ["milk"]


async def test_conflict_yes_reraises_when_the_belief_changed(tmp_path):
    await _setup(tmp_path)
    ask_id = await _open_conflict()               # existing 'black', proposed 'milk'
    # the belief MOVES under the open ask — a third value is written silently.
    await gate.apply(Proposal(intent="structured:preferences.coffee", content="oat milk"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.keep()), message="oat now")
    out = await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN)
    assert out == "restaled"
    # the STALE proposed 'milk' was NOT applied — the current 'oat milk' stands
    assert [e.content for e in await store.get_structured("preferences.coffee")] == ["oat milk"]
    # the user was told it changed, and a FRESH card is up against the current value
    assert any("changed since" in t.lower() for t in await _outbox_texts())
    async with db.get_db() as c:
        row = await (await c.execute(
            "SELECT origin, content, existing_snapshot FROM pending WHERE kind='preferences.coffee'"
        )).fetchone()
    assert row["origin"] == "judge-conflict"
    assert row["content"] == "milk"               # the same proposed value, re-asked
    assert json.loads(row["existing_snapshot"])["content"] == "oat milk"   # vs the CURRENT value
    body = await _last_ask_body()
    assert body["existing"] == "oat milk" and body["proposed"] == "milk"


async def test_conflict_yes_reraise_then_yes_applies(tmp_path):
    # a stale Yes re-raises; a SECOND Yes (now against the current value) applies.
    await _setup(tmp_path)
    ask_id = await _open_conflict()
    await gate.apply(Proposal(intent="structured:preferences.coffee", content="oat milk"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.keep()), message="oat now")
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "restaled"
    async with db.get_db() as c:
        (fresh_id,) = await (await c.execute(
            "SELECT ask_id FROM pending WHERE kind='preferences.coffee'")).fetchone()
    assert await gate.resolve(fresh_id, approved=True, conversation_id=bus.MAIN) == "applied"
    assert [e.content for e in await store.get_structured("preferences.coffee")] == ["milk"]


async def _pending_rows(kind="preferences.coffee"):
    async with db.get_db() as c:
        return await (await c.execute(
            "SELECT ask_id, content, existing_snapshot FROM pending WHERE kind=?", (kind,))).fetchall()


async def test_reraise_leaves_exactly_one_open_pending_never_zero(tmp_path):
    # FIX 1: the re-raise deletes the stale ask and saves the fresh one ATOMICALLY, so
    # for a (conversation, kind) there is always EXACTLY one open pending ask — the fresh
    # one — and never zero (no silent-loss window between delete and save).
    await _setup(tmp_path)
    stale_id = await _open_conflict()                 # existing 'black', proposed 'milk'
    await gate.apply(Proposal(intent="structured:preferences.coffee", content="oat milk"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.keep()), message="oat now")
    assert await gate.resolve(stale_id, approved=True, conversation_id=bus.MAIN) == "restaled"
    rows = await _pending_rows()
    assert len(rows) == 1                              # exactly one — never zero, never two
    assert rows[0]["ask_id"] != stale_id              # the FRESH ask, not the stale one
    assert rows[0]["content"] == "milk"               # same proposed value, re-asked
    assert json.loads(rows[0]["existing_snapshot"])["content"] == "oat milk"   # vs current


# ── FIX 2: the belief is RETIRED under an open conflict (the riskiest branch) ──

async def test_conflict_yes_after_belief_retired_reraises_then_applies(tmp_path):
    await _setup(tmp_path)
    ask_id = await _open_conflict()                   # existing 'black', proposed 'milk'
    # the belief is RETIRED before the user taps Yes — the current belief is now absent.
    assert await store.delete_structured("preferences.coffee") is True
    # resolve re-validates against absence: the stored proposed value was chosen against a
    # value that no longer exists, so it re-raises (never applies blindly).
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "restaled"
    assert await store.get_structured("preferences.coffee") == []   # still absent
    rows = await _pending_rows()
    assert len(rows) == 1 and rows[0]["content"] == "milk"
    assert rows[0]["existing_snapshot"] is None       # re-raised against an ABSENT belief
    # a subsequent Yes now applies the proposed value as a FRESH belief.
    assert await gate.resolve(rows[0]["ask_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    assert [e.content for e in await store.get_structured("preferences.coffee")] == ["milk"]


# ── FIX 3: norm() stale-check + no-op when already satisfied ──────────────────

async def test_cosmetic_only_change_does_not_reraise(tmp_path):
    # the belief moves 'black' -> 'Black' (case only) under the open ask. norm() sees no
    # real move, so the conflict Yes applies the proposed value — no needless re-raise.
    await _setup(tmp_path)
    ask_id = await _open_conflict()                   # existing 'black', proposed 'milk'
    await gate.apply(Proposal(intent="structured:preferences.coffee", content="Black"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.keep()), message="Black")
    out = await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN)
    assert out == "applied"
    assert [e.content for e in await store.get_structured("preferences.coffee")] == ["milk"]
    assert not any("changed since" in t.lower() for t in await _outbox_texts())


async def test_conflict_yes_is_a_noop_when_already_satisfied(tmp_path):
    # the belief moves to the PROPOSED value (norm-equal) under the open ask. A Yes must
    # not raise a pointless "you have milk — save milk?" card: it is already satisfied.
    await _setup(tmp_path)
    ask_id = await _open_conflict()                   # existing 'black', proposed 'milk'
    await gate.apply(Proposal(intent="structured:preferences.coffee", content="Milk"),
                     conversation_id=bus.MAIN, author="Assistant",
                     judge=FakeJudge(judge.keep()), message="milk now")
    out = await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN)
    assert out == "applied"
    # the belief already holds the wanted value — untouched, no fresh card, honest outcome
    assert [e.content for e in await store.get_structured("preferences.coffee")] == ["Milk"]
    assert await _pending_rows() == []                # no open ask left, none re-raised
    assert any("already what you have" in t.lower() for t in await _outbox_texts())


# ── judge error: loud, no write, honest outcome ──────────────────────────────

async def test_judge_error_writes_nothing_and_is_loud(tmp_path):
    await _setup(tmp_path)
    out = await gate.apply(Proposal(intent="task", content="book flights"),
                           conversation_id=bus.MAIN, author="Assistant",
                           judge=BoomJudge(), message="book flights")
    assert out == "judge_error"
    assert await store.open_tasks() == []            # NOTHING written
    texts = await _outbox_texts()
    assert texts and "did not save" in texts[0].lower()   # a visible, loud error
    async with db.get_db() as c:
        rows = await (await c.execute(
            "SELECT 1 FROM audit WHERE kind='judge_error'")).fetchall()
    assert rows                                      # the failure is audited too


# ── agnostic: a synthetic non-Assistant proposal judges unchanged ────────────

async def test_agnostic_proposal_judges_fine(tmp_path):
    await _setup(tmp_path)
    # a Finance-style structured belief from a non-Assistant author — no assistant-
    # or domain-specific logic anywhere on the path.
    j = FakeJudge(judge.keep())
    out = await gate.apply(
        Proposal(intent="structured:preferences.budget", content="cap dining at $200/mo",
                 evidence=[Evidence(ref="txn-9", source="finance")]),
        conversation_id=bus.MAIN, author="FinanceAgent", judge=j, message="")
    assert out == "applied"
    assert j.seen["proposal"].content == "cap dining at $200/mo"


# ── no judge injected: the gate applies directly, as before ──────────────────

async def test_no_judge_applies_directly(tmp_path):
    await _setup(tmp_path)
    out = await gate.apply(Proposal(intent="task", content="water the plants"),
                           conversation_id=bus.MAIN, author="Daily")
    assert out == "task"
    assert [t.text for t in await store.open_tasks()] == ["water the plants"]


# ── the real LlmJudge: parses a verdict, never returns content ───────────────

class ScriptedLLM:
    def __init__(self, reply):
        self.reply = reply
        self.seen = None

    async def complete(self, system, messages, *, tools=None):
        self.seen = (system, messages)
        return self.reply


async def test_llm_judge_parses_a_verdict():
    llm = ScriptedLLM('{"verdict":"collapse","target_id":"t-7","reason":"dup"}')
    d = await LlmJudge(llm)(Proposal(intent="task", content="x"), message="m", existing=[])
    assert d.verdict is Verdict.COLLAPSE and d.target_id == "t-7"
    assert d.reason == "dup"
    assert not hasattr(d, "user_asked")             # the field is gone entirely


async def test_llm_judge_tolerates_a_fenced_block():
    llm = ScriptedLLM('```json\n{"verdict":"conflict","target_id":"b-1","reason":"clash"}\n```')
    d = await LlmJudge(llm)(Proposal(intent="task", content="x"), message="m", existing=[])
    assert d.verdict is Verdict.CONFLICT and d.target_id == "b-1"


async def test_llm_judge_raises_on_garbage():
    with pytest.raises(JudgeError):
        await LlmJudge(ScriptedLLM("I think keep it, maybe?"))(
            Proposal(intent="task", content="x"), message="m", existing=[])


async def test_llm_judge_raises_on_unknown_verdict():
    with pytest.raises(JudgeError):
        await LlmJudge(ScriptedLLM('{"verdict":"delete"}'))(
            Proposal(intent="task", content="x"), message="m", existing=[])


async def test_llm_judge_never_returns_content():
    # even if the model tries to smuggle a content field, the Decision carries none.
    llm = ScriptedLLM('{"verdict":"keep","content":"malicious rewrite"}')
    d = await LlmJudge(llm)(Proposal(intent="task", content="x"), message="m", existing=[])
    assert not hasattr(d, "content")


async def test_llm_judge_error_bubbles_as_judge_error():
    class BoomLLM:
        async def complete(self, system, messages, *, tools=None):
            raise RuntimeError("network")
    with pytest.raises(JudgeError):
        await LlmJudge(BoomLLM())(Proposal(intent="task", content="x"), message="m", existing=[])


# ── the Decisions judge + its LLM fallback ────────────────────────────────────

def _client(handler):
    """A real Decisions client over a mocked httpx transport: `handler(body)` returns the
    reply (a dict = 200 JSON, an int = that HTTP status). Records each request body."""
    import httpx

    from ken.decisions import Decisions
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url == "https://api.openai.com/v1/decisions"
        assert request.headers["authorization"] == "Bearer sk-test"
        out = handler(body)
        return httpx.Response(out) if isinstance(out, int) else httpx.Response(200, json=out)
    return Decisions("sk-test", transport=httpx.MockTransport(respond)), seen


def _verdict(choice, confidence=0.9, *, grounded=0.97):
    def handler(body):
        answers = [{"type": "choice", "name": "verdict", "choice": choice,
                    "probabilities": [], "confidence": confidence}]
        if any(q["name"] == "grounded" for q in body["questions"]):   # answered only when asked
            answers.append({"type": "predicate", "name": "grounded", "probability": grounded})
        return {"answers": answers}
    return _client(handler)


_EXISTING = [SimpleNamespace(id="t1", kind="", text="book the dentist"),
             SimpleNamespace(id="facts.dentist", kind="facts.dentist", content="Dr Smith")]


async def test_decisions_judge_keep():
    c, seen = _verdict("keep")
    d = await DecisionsJudge(c)(Proposal(intent="task", content="call mom"), message="call mom",
                                existing=_EXISTING)
    assert d.verdict is Verdict.KEEP and d.target_id is None
    assert d.reason == "decisions 0.90 grounded 0.97"     # the grounding predicate rides along
    body = seen[0]
    assert body["model"] == "gpt-6-luna"
    q = {q["name"]: q for q in body["questions"]}
    assert q["grounded"]["type"] == "predicate"
    assert q["verdict"]["type"] == "choice" and {c["value"] for c in q["verdict"]["choices"]} == {
        "keep", "collapse:t1", "conflict:t1", "collapse:facts.dentist", "conflict:facts.dentist"}
    evidence = json.loads(body["input"])
    assert evidence["proposed"] == {"intent": "task", "content": "call mom"}
    assert evidence["existing"][0] == {"id": "t1", "kind": "", "text": "book the dentist"}


async def test_decisions_judge_collapse_and_conflict_carry_the_target():
    d = await DecisionsJudge(_verdict("collapse:t1")[0])(
        Proposal(intent="task", content="dentist"), message="", existing=_EXISTING)
    assert d.verdict is Verdict.COLLAPSE and d.target_id == "t1"
    d = await DecisionsJudge(_verdict("conflict:facts.dentist", 0.6)[0])(
        Proposal(intent="structured:facts.dentist", content="Dr Jones"), message="", existing=_EXISTING)
    assert d.verdict is Verdict.CONFLICT and d.target_id == "facts.dentist"
    assert d.reason == "decisions 0.60"


async def test_decisions_judge_offers_no_conflict_for_an_event():
    c, seen = _verdict("keep")
    d = await DecisionsJudge(c)(Proposal(intent="event", content="saw the dentist"), message="",
                                existing=[SimpleNamespace(id="e1", summary="saw the dentist")])
    (q,) = seen[0]["questions"]       # no user message (a scheduled sweep) → no grounding asked
    assert {c["value"] for c in q["choices"]} == {"keep", "collapse:e1"}
    assert d.reason == "decisions 0.90"


async def test_decisions_judge_failure_refusal_and_unknown_value_are_loud():
    for c in (_client(lambda body: 500)[0],
              _client(lambda body: {"answers": [{"type": "refusal", "name": "verdict"}]})[0],
              _verdict("collapse:nope")[0]):
        with pytest.raises(JudgeError):
            await DecisionsJudge(c)(Proposal(intent="task", content="x"), message="", existing=_EXISTING)


async def test_decisions_verdict_wins_and_the_llm_judge_answers_only_when_it_fails():
    p = Proposal(intent="task", content="x")
    llm = FakeJudge(judge.collapse("t1", reason="llm"))
    d = await FallbackJudge(DecisionsJudge(_verdict("keep")[0]), llm)(p, message="", existing=_EXISTING)
    assert d.verdict is Verdict.KEEP and llm.seen is None            # Decisions decided alone
    refusal = {"answers": [{"type": "refusal", "name": "verdict"}]}
    for c in (_client(lambda b: 503)[0], _client(lambda b: refusal)[0]):
        d = await FallbackJudge(DecisionsJudge(c), llm)(p, message="", existing=_EXISTING)
        assert d.verdict is Verdict.COLLAPSE and d.reason == "llm"  # the LLM judge stepped in
    with pytest.raises(JudgeError):                                  # both down: still loud
        await FallbackJudge(DecisionsJudge(_client(lambda b: 503)[0]), BoomJudge())(
            p, message="", existing=_EXISTING)


def test_the_engine_wires_decisions_first_only_when_enabled(tmp_path):
    from ken import app, config, db
    db.configure(str(tmp_path / "ken.db"))
    llm = "llm:\n  provider: openai\n  api_key: sk-test\n"
    config.config_path().write_text(llm + "decisions:\n  enabled: false\n")
    eng = app.build_engine()
    assert isinstance(eng.judge, LlmJudge) and eng.assistant._rank is None   # turned off: no Decisions
    config.config_path().write_text(llm)
    eng = app.build_engine()
    assert isinstance(eng.judge, FallbackJudge) and eng.assistant._rank is not None   # on by default


async def test_decisions_ranker_scores_align_with_items_and_never_raise():
    from ken.rank import DecisionsRanker
    c, seen = _client(lambda body: {"answers": [
        {"type": "score", "name": q["name"], "score": float(i), "probabilities": [], "confidence": 1}
        for i, q in enumerate(body["questions"])]})
    out = await DecisionsRanker(c)(["a", "b", "c"], instructions="how soon?", rubric=["0", "1", "2"])
    assert out == [0.0, 1.0, 2.0]
    assert [it["text"] for it in json.loads(seen[0]["input"])["items"]] == ["a", "b", "c"]
    q2 = seen[0]["questions"][2]
    assert q2["name"] == "s2" and q2["type"] == "score"
    assert q2["instructions"].startswith("For item i=2 only:")
    assert [lv["label"] for lv in q2["levels"]] == ["0", "1", "2"]
    import logging
    from unittest import mock
    refused = _client(lambda body: {"answers": [{"type": "refusal", "name": "s0"}]})[0]
    for bad in (_client(lambda body: 500)[0], refused):
        with mock.patch.object(logging.getLogger("ken.rank"), "warning") as warn:
            assert await DecisionsRanker(bad)(["a"], instructions="", rubric=["0", "1"]) is None
        assert warn.called                                      # a failure is logged, never silent
    assert await DecisionsRanker(c)([], instructions="", rubric=["0"]) == []


def test_decisions_is_on_by_default_and_openai_only():
    import logging
    from unittest import mock

    from ken.decisions import build_decisions
    key = {"provider": "openai", "api_key": "sk-x"}
    assert build_decisions({}, key) is not None                          # on by default
    assert build_decisions({"enabled": False}, key) is None              # turned off
    with mock.patch.object(logging.getLogger("ken.decisions"), "warning") as warn:
        assert build_decisions({}, {"provider": "claude-code"}) is None  # not usable: quiet
    assert not warn.called
    on = {"enabled": True}
    d = build_decisions({**on, "model": "gpt-6-luna"}, key)
    assert d is not None and d.model == "gpt-6-luna"
    assert build_decisions(on, {**key, "base_url": "https://api.openai.com/v1/"}) is not None
    assert build_decisions(on, {"provider": "openai"}) is None           # no key
    assert build_decisions(on, {**key, "api_key": "YOUR_KEY"}) is None
    assert build_decisions(on, {**key, "base_url": "https://openrouter.ai/api/v1"}) is None
    assert build_decisions(on, {"provider": "claude-code"}) is None
    assert build_decisions(on, {**key, "provider": "claude_code"}) is None


async def test_a_restaled_yes_writes_no_card_outcome_row(tmp_path):
    """The stale-approval guard re-raises a fresh card and applies nothing — that is not an
    outcome the scoreboard may count (it would double-count one decision)."""
    await _setup(tmp_path)
    kind = "facts.dentist"
    await store.upsert_structured(StructuredEntry(id=kind, kind=kind, subject="you", content="Dr Smith"),
                                  change_note="seed")
    # a judge-conflict card against "Dr Smith"
    await gate.apply(Proposal(intent=f"structured:{kind}", content="Dr Jones"),
                     conversation_id=bus.MAIN, author="kip",
                     judge=FakeJudge(judge.conflict(kind)), message="my dentist is Dr Jones")
    async with db.get_db() as c:
        ask_id = (await (await c.execute("SELECT ask_id FROM pending")).fetchone())["ask_id"]
    # the belief moves under the open card
    await store.upsert_structured(StructuredEntry(id=kind, kind=kind, subject="you", content="Dr Brown"),
                                  change_note="moved")
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "restaled"
    async with db.get_db() as c:
        n = (await (await c.execute("SELECT count(*) FROM audit WHERE kind='card_outcome'")).fetchone())[0]
    assert n == 0


async def test_decisions_judge_hands_a_slice_too_big_to_see_to_the_fallback():
    from types import SimpleNamespace

    from ken.consent.judge import _DECIDE_MAX_EXISTING, DecisionsJudge, FallbackJudge, JudgeError
    c, seen = _client(lambda body: {"answers": [{"type": "choice", "name": "verdict", "choice": "keep",
                                                 "probabilities": [], "confidence": 1}]})
    big = [SimpleNamespace(id=f"t{i}", kind="task", text=f"task {i}") for i in range(_DECIDE_MAX_EXISTING + 1)]
    prop = SimpleNamespace(intent="task", content="new task")
    import pytest
    with pytest.raises(JudgeError):
        await DecisionsJudge(c)(prop, message="", existing=big)
    assert seen == []                                   # never asked blind

    class Fallback:
        async def __call__(self, proposal, *, message, existing):
            return "fallback"
    assert await FallbackJudge(DecisionsJudge(c), Fallback())(prop, message="", existing=big) == "fallback"


async def test_any_primary_failure_falls_back():
    from types import SimpleNamespace

    from ken.consent.judge import FallbackJudge

    class Broken:
        async def __call__(self, proposal, *, message, existing):
            raise ValueError("confidence: 'high'")

    class Fallback:
        async def __call__(self, proposal, *, message, existing):
            return "fallback"
    prop = SimpleNamespace(intent="task", content="x")
    assert await FallbackJudge(Broken(), Fallback())(prop, message="", existing=[]) == "fallback"


def test_decisions_off_written_as_text_is_off():
    from ken.decisions import build_decisions
    key = {"provider": "openai", "api_key": "sk-x"}
    for off in ("false", "off", "No", "0"):
        assert build_decisions({"enabled": off}, key) is None
    assert build_decisions({"enabled": "true"}, key) is not None


def _valid_decisions_request(body):
    """The request rules OpenAI enforces with a 400 (seen live 2026-10-24): 1-400 questions,
    non-empty instructions, a choice has at least 2 choices with unique values."""
    qs = body["questions"]
    assert 1 <= len(qs) <= 400
    for q in qs:
        assert q["instructions"].strip()
        if q["type"] == "choice":
            values = [c["value"] for c in q["choices"]]
            assert len(values) >= 2, f"choices array too short: {values}"
            assert len(set(values)) == len(values)


async def test_decisions_judge_first_write_of_a_kind_sends_a_valid_request():
    # an empty slice left `keep` as the only choice → 400 "array too short" on every first
    # task / event / person: the 13 Decisions failures of the long-history benchmark
    def handler(body):
        _valid_decisions_request(body)
        return {"answers": [{"type": "predicate", "name": "grounded", "probability": 0.8}]}
    c, seen = _client(handler)
    d = await DecisionsJudge(c)(Proposal(intent="task", content="call mom"), message="call mom", existing=[])
    assert d.verdict is Verdict.KEEP and d.reason == "decisions nothing to compare grounded 0.80"
    assert [q["name"] for q in seen[0]["questions"]] == ["grounded"]
    d = await DecisionsJudge(c)(Proposal(intent="task", content="x"), message="", existing=[])
    assert d.verdict is Verdict.KEEP and len(seen) == 1          # nothing to ask: no call at all


async def test_decisions_http_error_keeps_the_response_body():
    import httpx

    from ken.decisions import Decisions
    err = {"error": {"message": "Invalid 'questions[0].choices': array too short.", "code": "array_below_min_length"}}
    c = Decisions("sk-test", transport=httpx.MockTransport(lambda r: httpx.Response(400, json=err)))
    assert await c.decide([{"type": "predicate", "name": "p", "instructions": "x"}], "in") is None
    assert "400" in c.last_error and "array too short" in c.last_error and "sk-test" not in c.last_error
