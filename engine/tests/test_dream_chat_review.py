"""inbox_chat review fixes: failures still record what was done, tidy talks never close a fix by mistake,
one turn per item runs beside the engine loop, and a reply carries only its own actions."""
import asyncio
import json

from ken import dream, mcp_server
from ken.agents.assistant import Assistant
from ken.llm import _run
from ken.memory import store

from .test_dream import FakeLLM, _ask, _pumped, _setup
from .test_dream_chat import ChatLLM


class FailingLLM:
    """Runs the given tool calls, then the model fails (a 5xx, a rate limit)."""

    def __init__(self, calls):
        self.calls = calls

    async def complete(self, system, messages, *, tools=None, max_hops=8):
        by = {t.name: t for t in tools}
        for name, args in self.calls:
            await _run(by[name].fn, args)
        raise RuntimeError("upstream 503")


async def _one(tmp_path, night):
    task, when = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM(night(task, when)))
    (item,) = await dream.items()
    return task, item


async def test_a_model_failure_still_records_what_was_done(tmp_path, monkeypatch):
    task, item = await _one(tmp_path, lambda t, w: {"asks": [_ask("x", f"task {t.id}")]})
    monkeypatch.setattr(dream, "_llm", lambda: FailingLLM([("discard_task", {"task": task.id, "reason": "closed"})]))
    r = await dream.chat(item["id"], "drop it")
    assert r["outcome"] == "error" and not r["closed"] and "Already done" in r["reply"]
    assert (await store.task_by_id(task.id)).status == "dropped"
    last = (await dream.get(item["id"]))["thread"][-1]
    assert last["role"] == "dreamer" and last["actions"][0]["tool"] == "discard_task"


async def test_a_tidy_talk_cannot_close_only_skip_and_only_a_skip_is_refused_later(tmp_path, monkeypatch):
    def night(t, w):
        return {"tidy": [{"kind": "project", "name": "emigration", "add": "The consulate requires a round-trip ticket.",
                          "source": {"event": w}}]}
    _, fix = await _one(tmp_path, night)
    seen = []

    class Look(ChatLLM):
        async def complete(self, system, messages, *, tools=None, max_hops=8):
            seen.append({t.name for t in tools}); seen.append(system)
            return "Press Apply under the fix to save it."
    monkeypatch.setattr(dream, "_llm", lambda: Look([], ""))
    r = await dream.chat(fix["id"], "yes, apply it")
    assert "close" not in seen[0] and {"skip", "revise_fix"} <= seen[0] and "press Apply" in seen[1]
    assert r["closed"] is False and (await dream.get(fix["id"]))["status"] == "open"
    monkeypatch.setattr(dream, "_llm", lambda: ChatLLM([("skip", {"reason": "not needed"})], "Skipped."))
    assert (await dream.chat(fix["id"], "skip it"))["closed"] is True
    assert (await dream.items("closed"))[0]["outcome"] == "skipped: not needed"
    again = await dream.run_once(llm=FakeLLM(night(None, (await dream.get(fix["id"]))["sources"][0]["event"])))
    assert "a fact you skipped before was left out" in again["tidy"][0]


async def test_a_closed_ask_does_not_count_as_a_skipped_fact(tmp_path):
    _, item = await _one(tmp_path, lambda t, w: {"asks": [_ask("x", f"task {t.id}")]})
    await dream.close(item["id"], "Left as is.")
    assert await dream._refused(dream.now_ms()) == {}


async def test_a_revision_needs_words_and_keeps_question_and_exclamation_marks(tmp_path, monkeypatch):
    def night(t, w):
        return {"tidy": [{"kind": "project", "name": "emigration", "add": "The consulate requires a round-trip ticket.",
                          "source": {"event": w}}]}
    _, fix = await _one(tmp_path, night)
    monkeypatch.setattr(dream, "_llm", lambda: ChatLLM([("revise_fix", {"add": ""})], "A fix needs a sentence."))
    await dream.chat(fix["id"], "drop the sentence")
    assert (await dream.get(fix["id"]))["add"] == "The consulate requires a round-trip ticket."   # unchanged
    monkeypatch.setattr(dream, "_llm", lambda: ChatLLM([("revise_fix", {"add": "Is a return ticket needed? Yes! Booked."})], "ok"))
    await dream.chat(fix["id"], "rewrite")
    assert (await dream.get(fix["id"]))["adds"] == ["Is a return ticket needed?", "Yes!", "Booked."]


async def test_earlier_actions_go_back_to_the_model(tmp_path, monkeypatch):
    task, item = await _one(tmp_path, lambda t, w: {"asks": [_ask("x", f"task {t.id}")]})
    monkeypatch.setattr(dream, "_llm", lambda: ChatLLM([("reschedule_task", {"task": task.id, "deadline": "2030-01-07"})],
                                                       "Moved it. The others too?"))
    await dream.chat(item["id"], "move it to January")
    llm = ChatLLM([], "ok")
    monkeypatch.setattr(dream, "_llm", lambda: llm)
    await dream.chat(item["id"], "yes, all")
    assert "[done: reschedule_task:" in llm.seen[0][1][1]["content"]


async def test_a_turn_runs_beside_the_loop_one_per_item_and_answers_with_its_own_actions(tmp_path, monkeypatch):
    from ken.engine import Engine
    task, item = await _one(tmp_path, lambda t, w: {"asks": [_ask("x", f"task {t.id}")]})
    gate_open = asyncio.Event()

    class Slow(ChatLLM):
        async def complete(self, system, messages, *, tools=None, max_hops=8):
            await gate_open.wait()
            return await super().complete(system, messages, tools=tools, max_hops=max_hops)
    monkeypatch.setattr(dream, "_llm", lambda: Slow([("complete_task", {"task": task.id})], "Done."))
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    first = asyncio.create_task(_pumped(eng, mcp_server.inbox_chat(item["id"], "mark it done", timeout=5)))
    await asyncio.sleep(0.3)
    busy = await _pumped(eng, mcp_server.inbox_chat(item["id"], "and again", timeout=5))   # the loop still answers
    assert busy["outcome"] == "busy"
    gate_open.set()
    out = await first
    assert out["outcome"] == "answered" and [a["tool"] for a in out["actions"]] == ["complete_task"]


async def test_a_fix_closed_elsewhere_during_the_turn_is_not_rewritten(tmp_path, monkeypatch):
    def night(t, w):
        return {"tidy": [{"kind": "project", "name": "emigration", "add": "The consulate requires a round-trip ticket.",
                          "source": {"event": w}}]}
    _, fix = await _one(tmp_path, night)

    class SkipMeanwhile(ChatLLM):
        async def complete(self, system, messages, *, tools=None, max_hops=8):
            await dream.close(fix["id"], "skipped")              # the user taps Skip in another client
            return await super().complete(system, messages, tools=tools, max_hops=max_hops)
    monkeypatch.setattr(dream, "_llm", lambda: SkipMeanwhile([("revise_fix", {"add": "Something else."})], "Changed."))
    r = await dream.chat(fix["id"], "change it")
    after = await dream.get(fix["id"])
    assert r["closed"] is False and after["add"] == "The consulate requires a round-trip ticket."
    assert after["status"] == "closed" and json.dumps(after["thread"])        # the talk is still kept
