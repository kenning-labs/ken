"""pending is scoped like events(answers=true): an agent token sees only the cards its own writes raised.
And the Inbox's readable view never doubles a task name the model wrote after its id."""
import json

from ken import db, dream, mcp_server

from .test_card_answers import _raise
from .test_dream_readable import _one_ask


async def test_pending_shows_an_agent_only_the_cards_it_raised(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    mine = await _raise("people.nina", "act-agent", "mcp:agent:commitments")
    other = await _raise("projects.launch", "act-user", "mcp:user")
    srv = mcp_server.build_server()

    async def pending_as(caller):
        tok = mcp_server._CALLER.set(caller)
        try:
            return json.dumps(await srv.call_tool("pending", {}), default=str)
        finally:
            mcp_server._CALLER.reset(tok)
    agent = await pending_as("mcp:agent:commitments")
    assert mine in agent and other not in agent
    assert mine not in await pending_as("mcp:agent:someone-else")
    user = await pending_as("mcp:user")                                 # the user sees every card
    assert mine in user and other in user


async def test_an_id_right_before_its_name_is_dropped_not_doubled(tmp_path):
    tid, item = await _one_ask(tmp_path)
    # a prefix with a letter in it: an all-digit token is never an id, and ~2% of ids start with 8 digits
    short = next(tid[:n] for n in range(8, len(tid) + 1) if any(c in "abcdef" for c in tid[:n]))
    item["evidence"] = [f'Task {tid} "book the consulate appointment": status open',
                        f"{short}: “book the consulate appointment” is open"]
    out = await dream.readable(item)
    assert out["evidence"] == ['"book the consulate appointment": status open', "“book the consulate appointment” is open"]
