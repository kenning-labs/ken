"""events(answers=true): an always-on agent learns what the user decided on its cards since its last run."""
import json

from ken import db, mcp_server
from ken.agents import Proposal
from ken.consent import gate
from ken.contracts import Provenance
from ken.conversation import bus
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
