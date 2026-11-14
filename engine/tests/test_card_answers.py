"""events(answers=true): an always-on agent learns what the user decided on its cards since its last run."""
import json

from ken import db, mcp_server
from ken.agents import Proposal
from ken.consent import gate
from ken.contracts import Provenance
from ken.conversation import bus
from ken.memory import store
from ken.util import now_ms


async def _raise(target: str, action_id: str, channel: str) -> str:
    """A card raised by one inbound action from `channel` (the inbox row is what the gate reads back)."""
    async with db.get_db() as c:
        await c.execute("INSERT INTO inbox (id, conversation_id, kind, channel, at) VALUES (?,?,?,?,?)",
                        (action_id, bus.MAIN, "message", channel, now_ms()))
        await c.commit()
    await gate.apply(Proposal(intent=f"structured:{target}", content="a new note", provenance=Provenance.AGENT_INFERRED),
                     conversation_id=bus.MAIN, author="kip", action_id=action_id)
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending WHERE kind=?", (target,))).fetchone()
    return ask_id


async def _events_as(caller: str, **kw) -> dict:
    tok = mcp_server._CALLER.set(caller)
    try:
        return json.loads(await mcp_server.events(**kw))
    finally:
        mcp_server._CALLER.reset(tok)


async def test_an_agent_sees_the_answers_to_its_own_cards_only(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    mine = await _raise("people.nina", "act-agent", "mcp:agent:commitments")
    other = await _raise("projects.launch", "act-user", "mcp:user")
    assert await gate.resolve(mine, approved=True, conversation_id=bus.MAIN) == "applied"
    assert await gate.resolve(other, approved=False, conversation_id=bus.MAIN) == "rejected"

    agent = await _events_as("mcp:agent:commitments", answers=True)
    assert [(a["card_id"], a["about"], a["answer"]) for a in agent["answers"]] == [(mine, "people.nina", "yes")]
    assert "raised_by" not in agent["answers"][0]                       # an agent learns nothing about other callers

    assert (await _events_as("mcp:agent:someone-else", answers=True))["answers"] == []

    user = await _events_as("mcp:user", answers=True)
    by_card = {a["card_id"]: a for a in user["answers"]}
    assert by_card[mine]["raised_by"] == "mcp:agent:commitments" and by_card[other]["answer"] == "no"

    assert "answers" not in await _events_as("mcp:agent:commitments")   # opt-in: plain events stay as they were


async def test_since_limits_the_answers_to_after_the_last_run(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    card = await _raise("people.sam", "act-1", "mcp:agent:commitments")
    assert await gate.resolve(card, approved=True, conversation_id=bus.MAIN) == "applied"
    before = await _events_as("mcp:agent:commitments", since="2000-01-01", answers=True)   # a last run before the answer
    assert [a["card_id"] for a in before["answers"]] == [card]
    after = await _events_as("mcp:agent:commitments", since="2999-01-01", until="2999-01-02", answers=True)
    assert after["answers"] == []                                                         # a last run after it


# ── the review's cases: who raised a card is set once, and every answer says what it did ──

async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


async def _inbox(action_id: str, channel: str) -> None:
    async with db.get_db() as c:
        await c.execute("INSERT INTO inbox (id, conversation_id, kind, channel, at) VALUES (?,?,?,?,?)",
                        (action_id, bus.MAIN, "message", channel, now_ms()))
        await c.commit()


async def _only_ask() -> str:
    async with db.get_db() as c:
        rows = await (await c.execute("SELECT ask_id FROM pending")).fetchall()
    assert len(rows) == 1
    return rows[0]["ask_id"]


async def test_a_second_caller_repeating_a_card_never_takes_it_over(tmp_path):
    await _setup(tmp_path)
    card = await _raise("people.nina", "act-a", "mcp:agent:a")
    await _inbox("act-b", "mcp:agent:b")
    await gate.apply(Proposal(intent="structured:people.nina", content="a new note", provenance=Provenance.AGENT_INFERRED),
                     conversation_id=bus.MAIN, author="kip", action_id="act-b")     # the identical ask, from B
    assert await gate.cards_for("act-b") == [card]                                   # B's reply still gets the card id
    assert await gate.resolve(card, approved=True, conversation_id=bus.MAIN) == "applied"
    assert (await _events_as("mcp:agent:b", answers=True))["answers"] == []          # B learns nothing of A's card
    assert [a["card_id"] for a in (await _events_as("mcp:agent:a", answers=True))["answers"]] == [card]


async def test_a_reasked_card_stays_the_agents_and_no_means_not_same(tmp_path):
    from ken.contracts import StructuredEntry
    await _setup(tmp_path)
    await store.upsert_structured(StructuredEntry(id="people.nina", kind="people.nina", subject="Nina", content="Designer"),
                                  change_note="test")
    await _inbox("act-a", "mcp:agent:a")
    entry = (await store.get_structured(prefix="people.nina"))[0]
    tok = gate._ACTION.set("act-a")
    try:
        await gate._identity_ask(Proposal(intent="structured:people.nina_k", content="Nina from Kyiv",
                                          provenance=Provenance.AGENT_INFERRED), entry, "the same name",
                                 conversation_id=bus.MAIN, author="kip")
    finally:
        gate._ACTION.reset(tok)
    first = await _only_ask()
    assert await gate.resolve(first, approved=False, conversation_id=bus.MAIN) == "reasked"   # the user's tap
    second = await _only_ask()
    async with db.get_db() as c:
        (raiser,) = await (await c.execute("SELECT raised_by FROM pending WHERE ask_id=?", (second,))).fetchone()
    assert raiser == "mcp:agent:a"                                                   # the new card is still the agent's
    assert await gate.resolve(second, approved=True, conversation_id=bus.MAIN) == "applied"
    got = {a["card_id"]: a["answer"] for a in (await _events_as("mcp:agent:a", answers=True))["answers"]}
    assert got == {first: "not_same", second: "yes"}


async def test_a_batch_answer_says_what_the_yes_kept(tmp_path):
    await _setup(tmp_path)
    await _inbox("act-h", "mcp:agent:hydrate")
    items = [{"kind": "people", "name": "Ruth", "note": "PM"}, {"kind": "people", "name": "Bob", "note": "recruiter"}]
    await gate.apply(Proposal(intent="batch:hydrate", content="Save 2?", meta={"items": items}),
                     conversation_id=bus.MAIN, author="kip", action_id="act-h")
    card = await _only_ask()
    drop_all = json.dumps([{"name": "Ruth", "drop": True}, {"name": "Bob", "drop": True}])
    assert await gate.resolve(card, approved=True, conversation_id=bus.MAIN, edited=drop_all) == "applied"
    (a,) = (await _events_as("mcp:agent:hydrate", answers=True))["answers"]
    assert (a["answer"], a["kept"], a["dropped"], a["held"]) == ("yes", 0, 2, 0)


async def test_an_act_answers_once_when_it_is_done(tmp_path):
    await _setup(tmp_path)
    await _inbox("act-s", "mcp:agent:slack")
    await gate.apply(Proposal(intent="act:reply", content="On it.", provenance=Provenance.USER_INSTRUCTED, meta={"channel": "c"}),
                     conversation_id=bus.MAIN, author="slack", action_id="act-s")
    card = await _only_ask()
    assert await gate.resolve(card, approved=True, conversation_id=bus.MAIN) == "act"   # the send failed: no answer yet
    assert await gate.resolve(card, approved=True, conversation_id=bus.MAIN) == "act"   # retried
    assert (await _events_as("mcp:agent:slack", answers=True))["answers"] == []
    await gate.close_act(card)                                                           # done
    assert [a["answer"] for a in (await _events_as("mcp:agent:slack", answers=True))["answers"]] == ["yes"]


async def test_one_answer_per_card_and_more_when_cut(tmp_path):
    await _setup(tmp_path)
    async with db.get_db() as c:
        for i, (ref, outcome) in enumerate([("c1", "rejected"), ("c1", "approved"), ("c2", "approved"), ("c3", "approved")]):
            await c.execute("INSERT INTO audit (id, kind, ref, reason, at) VALUES (?,?,?,?,?)",
                            (f"a{i}", "card_outcome", ref,
                             json.dumps({"outcome": outcome, "channel": "mcp:agent:x", "kind": "k"}), 1000 + i))
        await c.commit()
    rows, more = await store.card_answers(0, 10_000, channel="mcp:agent:x")
    assert [(r["card_id"], r["outcome"]) for r in rows] == [("c3", "approved"), ("c2", "approved"), ("c1", "approved")]
    assert not more
    rows, more = await store.card_answers(0, 10_000, channel="mcp:agent:x", limit=2)
    assert [r["card_id"] for r in rows] == ["c3", "c2"] and more


async def test_an_old_non_json_row_is_skipped_never_an_error(tmp_path):
    await _setup(tmp_path)
    async with db.get_db() as c:
        await c.execute("INSERT INTO audit (id, kind, ref, reason, at) VALUES ('old', 'card_outcome', 'c0', 'approved', ?)",
                        (now_ms(),))
        await c.commit()
    assert (await _events_as("mcp:agent:x", answers=True))["answers"] == []


async def test_an_act_nobody_can_perform_is_answered_failed_not_yes(tmp_path):
    await _setup(tmp_path)
    await _inbox("act-s", "mcp:agent:slack")
    await gate.apply(Proposal(intent="act:reply", content="On it.", provenance=Provenance.USER_INSTRUCTED, meta={"channel": "c"}),
                     conversation_id=bus.MAIN, author="slack", action_id="act-s")
    card = await _only_ask()
    assert await gate.resolve(card, approved=True, conversation_id=bus.MAIN) == "act"
    await gate.close_act(card, done=False)
    assert [a["answer"] for a in (await _events_as("mcp:agent:slack", answers=True))["answers"]] == ["failed"]


async def test_a_card_open_across_the_upgrade_gets_its_raiser(tmp_path):
    await _setup(tmp_path)
    await _inbox("act-old", "mcp:agent:a")
    async with db.get_db() as c:                       # a card saved by the old code: no raised_by
        await c.execute("INSERT INTO pending (ask_id, conversation_id, kind, content, author, ask_type, action_id, created_at) "
                        "VALUES ('k1', ?, 'people.zoe', 'x', 'kip', 'belief', 'act-old', ?)", (bus.MAIN, now_ms()))
        await c.commit()
    await db.init_db()                                 # the next start runs the migration
    async with db.get_db() as c:
        (raiser,) = await (await c.execute("SELECT raised_by FROM pending WHERE ask_id='k1'")).fetchone()
    assert raiser == "mcp:agent:a"
