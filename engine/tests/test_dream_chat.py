"""inbox_chat: every Inbox item is a talk with the Dreamer, which carries out what the user asks."""
import json

from ken import dream, mcp_server
from ken.agents.assistant import Assistant
from ken.conversation import bus
from ken.llm import _run
from ken.memory import store
from ken.util import now_ms

from .test_dream import FakeLLM, _ask, _pumped, _setup


class ChatLLM:
    """A scripted Dreamer for inbox_chat: runs the given tool calls (with the real tools), then replies."""

    def __init__(self, calls, reply):
        self.calls, self.reply, self.seen = calls, reply, []

    async def complete(self, system, messages, *, tools=None, max_hops=8):
        self.seen.append((system, messages))
        by = {t.name: t for t in tools}
        for name, args in self.calls:
            await _run(by[name].fn, args)
        return self.reply


async def test_a_quick_answer_is_carried_out_and_closes_the_item(tmp_path, monkeypatch):
    task, _ = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM({"asks": [_ask("Close the dead PR task?", f"task {task.id}")]}))
    (item,) = await dream.items()
    llm = ChatLLM([("discard_task", {"task": task.id, "reason": "the placement is closed"}),
                   ("close", {"outcome": "Dropped the PR task: the placement is closed."})], "Done. I dropped it.")
    monkeypatch.setattr(dream, "_llm", lambda: llm)
    r = await dream.chat(item["id"], "Close it as dropped")
    assert (r["outcome"], r["closed"], r["reply"]) == ("closed", True, "Done. I dropped it.")
    assert [a["tool"] for a in r["actions"]] == ["discard_task"]
    assert (await store.task_by_id(task.id)).status == "dropped"       # the user's instruction: applied at once
    (closed,) = await dream.items("closed")
    assert closed["outcome"] == "Dropped the PR task: the placement is closed."
    assert [m["role"] for m in closed["thread"]] == ["user", "dreamer"] and closed["thread"][1]["actions"]
    assert "Close the dead PR task?" in llm.seen[0][0]                   # the model sees the item it talks about


async def test_a_question_keeps_the_thread_open_and_the_next_turn_sees_it(tmp_path, monkeypatch):
    task, _ = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM({"asks": [_ask("Re-plan GTM?", f"task {task.id}")]}))
    (item,) = await dream.items()
    monkeypatch.setattr(dream, "_llm", lambda: ChatLLM([], "Which tasks: all 13, or only the dated ones?"))
    r = await dream.chat(item["id"], "move them")
    assert (r["outcome"], r["closed"]) == ("answered", False)
    llm = ChatLLM([("close", {"outcome": "Left as is."})], "Ok, I left them.")
    monkeypatch.setattr(dream, "_llm", lambda: llm)
    await dream.chat(item["id"], "actually leave them")
    assert [m["role"] for m in llm.seen[0][1]] == ["user", "assistant", "user"]   # the whole talk goes back
    assert (await dream.items("closed"))[0]["outcome"] == "Left as is."
    assert (await dream.chat(item["id"], "again"))["outcome"] == "no_item"       # a closed item takes no turn


async def test_a_tidy_fix_is_rewritten_in_the_users_words_not_applied(tmp_path, monkeypatch):
    _, when = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM({"tidy": [{"kind": "project", "name": "emigration",
                                               "add": "The consulate requires a round-trip ticket.", "source": {"event": when}}]}))
    (fix,) = await dream.items()
    monkeypatch.setattr(dream, "_llm", lambda: ChatLLM([("revise_fix", {"add": "A return ticket is required. Booked by Dec."})],
                                                       "Changed. Press Apply to save it."))
    r = await dream.chat(fix["id"], "say a return ticket, and that I book it by December")
    assert r["closed"] is False
    after = await dream.get(fix["id"])
    assert after["add"] == "A return ticket is required. Booked by Dec."
    assert after["adds"] == ["A return ticket is required.", "Booked by Dec."]
    rec = next(e for e in await store.get_structured("projects.") if e.kind == "projects.emigration")
    assert rec.content == "Moving to Portugal on a D8 visa."           # still only proposed


async def test_inbox_chat_runs_in_the_engine_and_refuses_agents(tmp_path, monkeypatch):
    from ken.contracts import Action, ActionKind
    from ken.engine import Engine
    from ken.util import new_id
    task, _ = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM({"asks": [_ask("x", f"task {task.id}")]}))
    (item,) = await dream.items()
    monkeypatch.setattr(dream, "_llm", lambda: ChatLLM([], "Which one?"))
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    out = await _pumped(eng, mcp_server.inbox_chat(item["id"], "hm", timeout=5))
    assert (out["reply"], out["outcome"]) == ("Which one?", "answered")
    assert [m["role"] for m in out["item"]["thread"]] == ["user", "dreamer"]
    tok = mcp_server._CALLER.set("mcp:agent:codex")
    try:
        assert json.loads(await mcp_server.inbox_chat(item["id"], "close it"))["outcome"] == "forbidden"
    finally:
        mcp_server._CALLER.reset(tok)
    # an agent that posts the action itself is refused by the engine too
    await bus.append_action(Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.TOOL, channel="mcp:agent:codex",
                                   text=json.dumps({"tool": "inbox_chat", "args": {"item_id": item["id"], "text": "close"}}),
                                   at=now_ms()))
    await eng.pump_inbox()
    assert len((await dream.get(item["id"]))["thread"]) == 2           # no third message
