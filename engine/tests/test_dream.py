"""The Dreamer: the engine checks what the model says before anything reaches the user."""
import asyncio
import json
from datetime import datetime

from ken import db, dream, mcp_server
from ken.agents.assistant import Assistant, _parse_when
from ken.agents.base import Context
from ken.capabilities import MemoryReader, Proposer
from ken.captures import Captures
from ken.contracts import Provenance, StructuredEntry
from ken.conversation import bus
from ken.llm import _run
from ken.memory import store
from ken.util import now_ms

DAY = 86_400_000


class FakeLLM:
    """Calls each tool once (so the reads are wired), then answers with the night it was given."""

    def __init__(self, night: dict):
        self.night, self.prompts, self.tool_out = night, [], {}

    async def complete(self, system, messages, *, tools=None, max_hops=8):
        self.prompts.append(messages[0]["content"])
        for t in tools or []:
            self.tool_out[t.name] = await _run(t.fn, {})      # as the real loop runs a tool
        return "Here it is:\n" + json.dumps(self.night)


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    ctx = Context(conversation_id=bus.MAIN, now=now_ms(), message="")
    ctx.memory = MemoryReader(Assistant.slice)
    ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
    cap = Captures(ctx, parse_when=_parse_when, provenance=Provenance.USER_INSTRUCTED)
    await store.upsert_structured(StructuredEntry(
        id="projects.emigration", kind="projects.emigration", subject="emigration",
        content="Moving to Portugal on a D8 visa.", provenance=Provenance.USER_STATED), change_note="test")
    await cap.add_task("book the consulate appointment")
    when = datetime.now().replace(hour=17, minute=45, second=0, microsecond=0)
    await cap.log_event("The consulate answered: a round-trip ticket is required", when=when.strftime("%Y-%m-%dT%H:%M"))
    (task,) = await store.open_tasks()
    return task, when.strftime("%Y-%m-%dT%H:%M")


def _ask(title, *evidence):
    return {"title": title, "preview": "p", "body": ["b"], "evidence": list(evidence), "quick": ["Do it", "Leave it"],
            "why_now": "w"}


async def test_a_night_keeps_what_its_evidence_holds_and_drops_the_rest(tmp_path):
    task, _ = await _setup(tmp_path)
    night = {"asks": [_ask("real", f"task {task.id[:8]}"),
                      _ask("garbled id", "task 83c4b6b4"),          # an id Ken does not hold
                      _ask("no evidence"),
                      _ask("slack ts is not an id", f"task {task.id}", "thread_ts=1793138851.805299")],
             "looks": [_ask(f"look {i}", f"task {task.id[:8]}") for i in range(6)],
             "tidy": [], "skipped": ["a quiet thing"]}
    llm = FakeLLM(night)
    done = await dream.run_once(llm=llm)
    assert (done["asks"], done["looks"]) == (2, 4)                  # the cap is 6 across both
    assert any("unknown ids ['83c4b6b4']" in d for d in done["dropped"])
    assert any("no evidence" in d for d in done["dropped"])
    assert sum("over the cap" in d for d in done["dropped"]) == 2
    # every read tool answered, and the prompt carries the real clock
    assert set(llm.tool_out) == {"world", "tasks", "events", "who_is", "find", "get_context", "chat", "pending"}
    assert datetime.now().strftime("%d %B %Y") in llm.prompts[0]
    items = await dream.items()
    assert {i["title"] for i in items} >= {"real", "slack ts is not an id"} and all(i["status"] == "open" for i in items)


async def test_tidy_is_an_inbox_fix_never_a_card(tmp_path):
    _, when = await _setup(tmp_path)
    night = {"asks": [], "looks": [], "tidy": [
        {"kind": "project", "name": "emigration", "add": "The consulate requires a round-trip ticket.",
         "why": "the 17:45 event", "source": {"event": when}},
        {"kind": "project", "name": "emigration", "add": "The appointment call was made.",
         "why": "the task is closed", "source": {"event": "2001-01-01T09:00"}},   # no such event
        {"kind": "project", "name": "emigration", "add": "Also true.", "why": "x"},  # no source at all
        {"kind": "project", "name": "Emigration", "add": "The consulate answered on the 13th.",
         "why": "same record, a second fact", "source": {"event": when}},
        {"kind": "person", "name": "nobody", "add": "x", "source": {"event": when}}]}
    done = await dream.run_once(llm=FakeLLM(night))
    # one fix per record, the unproven facts left out, and no card in the shared queue
    assert done["tidy"][0] == "emigration: proposed (a fact with no event or chat message behind it was left out; " \
                              "a fact with no event or chat message behind it was left out)"
    assert "no person 'nobody'" in done["tidy"][1]
    assert await mcp_server.pending_cards() == []
    (fix,) = await dream.items()
    assert (fix["kind"], fix["title"]) == ("tidy", "Add to emigration")
    assert fix["add"] == "The consulate requires a round-trip ticket. The consulate answered on the 13th."
    rec = next(e for e in await store.get_structured("projects.") if e.kind == "projects.emigration")
    assert rec.content == "Moving to Portugal on a D8 visa."         # untouched until the user applies it
    again = await dream.run_once(llm=FakeLLM(night))                 # the next night: one fix per record waits
    assert again["tidy"][0].startswith("emigration: a fix for it already waits in the Inbox")


async def _pumped(eng, coro):
    """Run an MCP call while the engine drains its inbox, as the live engine would."""
    async def pump():
        for _ in range(60):
            await eng.pump_inbox(); await asyncio.sleep(0.05)
    t = asyncio.create_task(pump())
    try:
        return json.loads(await coro)
    finally:
        t.cancel()


async def test_apply_saves_the_fix_on_the_note_as_it_is_now(tmp_path):
    from ken.engine import Engine
    _, when = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM({"tidy": [{"kind": "project", "name": "emigration",
                                               "add": "The consulate requires a round-trip ticket.", "source": {"event": when}}]}))
    (fix,) = await dream.items()
    # the user edits the note after the night: Apply keeps the edit and adds the fix
    await store.upsert_structured(StructuredEntry(id="projects.emigration", kind="projects.emigration", subject="emigration",
                                                  content="Moving to Portugal on a D8 visa in June.",
                                                  provenance=Provenance.USER_STATED), change_note="test")
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    out = await _pumped(eng, mcp_server.inbox_apply(fix["id"]))
    assert out["outcome"] == "applied", out
    rec = next(e for e in await store.get_structured("projects.") if e.kind == "projects.emigration")
    assert rec.content == "Moving to Portugal on a D8 visa in June. The consulate requires a round-trip ticket."
    assert await mcp_server.pending_cards() == []                   # the tap was the consent: no card left
    assert (await dream.items("closed"))[0]["outcome"] == "applied"


async def test_apply_waits_while_an_outside_card_holds_the_record(tmp_path):
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence
    _, when = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM({"tidy": [{"kind": "project", "name": "emigration",
                                               "add": "The consulate requires a round-trip ticket.", "source": {"event": when}}]}))
    await gate.apply(Proposal(intent="structured:projects.emigration", content="Moving to Spain instead.",
                              evidence=[Evidence(ref="x", source="codex")]), conversation_id=bus.MAIN, author="codex")
    (fix,) = await dream.items()
    out = json.loads(await mcp_server.inbox_apply(fix["id"]))
    assert out["outcome"] == "blocked" and "answer it first" in out["text"]
    (card,) = await mcp_server.pending_cards()
    assert card["raised_by"] == "codex"                              # the agent's card is untouched
    assert (await dream.items())[0]["status"] == "open"


async def test_old_dreamer_cards_go_back_and_agents_cannot_apply(tmp_path):
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence
    _, when = await _setup(tmp_path)
    await gate.apply(Proposal(intent="structured:projects.emigration", content="Moving to Portugal. Old style.",
                              evidence=[Evidence(ref="dreamer:x", source="Dreamer")]), conversation_id=bus.MAIN, author="Dreamer")
    assert len(await mcp_server.pending_cards()) == 1
    await dream.run_once(llm=FakeLLM({"tidy": [{"kind": "project", "name": "emigration",
                                               "add": "The consulate requires a round-trip ticket.", "source": {"event": when}}]}))
    assert await mcp_server.pending_cards() == []                   # withdrawn: the Dreamer keeps no cards
    (fix,) = await dream.items()
    tok = mcp_server._CALLER.set("mcp:agent:codex")
    try:
        assert json.loads(await mcp_server.inbox_apply(fix["id"]))["outcome"] == "forbidden"
    finally:
        mcp_server._CALLER.reset(tok)


async def test_closed_items_feed_the_next_night_and_old_ones_expire(tmp_path):
    task, _ = await _setup(tmp_path)
    now = now_ms()
    await dream.run_once(now - 9 * DAY, llm=FakeLLM({"asks": [_ask("old one", f"task {task.id}")]}))
    await dream.run_once(now - DAY, llm=FakeLLM({"asks": [_ask("re-plan GTM?", f"task {task.id}")]}))
    (item,) = await dream.items()
    assert item["title"] == "re-plan GTM?"                           # "old one" is past a week: expired
    assert await dream.close(item["id"], "Move all to next week") and not await dream.close(item["id"], "again")
    llm = FakeLLM({"asks": []})
    await dream.run_once(now, llm=llm)
    assert "Move all to next week" in llm.prompts[0] and "expired" in llm.prompts[0]
    assert {i["outcome"] for i in await dream.items("closed")} == {"Move all to next week", "expired"}


async def test_the_night_runs_once_a_day_after_its_hour_and_only_when_on(tmp_path, monkeypatch):
    await _setup(tmp_path)
    started = []

    async def fake_run(now=None, llm=None):
        started.append(now)
        return {}
    monkeypatch.setattr(dream, "run_once", fake_run)
    at_2 = int(datetime.now().replace(hour=2, minute=0).timestamp() * 1000)
    at_4 = int(datetime.now().replace(hour=4, minute=0).timestamp() * 1000)
    monkeypatch.setattr(dream, "_cfg", dict)
    assert not await dream.run_scheduled(at_4)                       # off by default
    monkeypatch.setattr(dream, "_cfg", lambda: {"enabled": True})
    assert not await dream.run_scheduled(at_2)                       # before 03:00
    assert await dream.run_scheduled(at_4)
    await dream._running
    assert not await dream.run_scheduled(at_4 + 3_600_000)           # once a day
    assert started == [at_4]


async def test_an_agent_token_cannot_close_the_inbox(tmp_path):
    task, _ = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM({"asks": [_ask("x", f"task {task.id}")]}))
    (item,) = await dream.items()
    srv = mcp_server.build_server()
    tok = mcp_server._CALLER.set("mcp:agent:codex")
    try:
        out = await srv.call_tool("inbox_close", {"item_id": item["id"], "outcome": "no"})
    finally:
        mcp_server._CALLER.reset(tok)
    assert "forbidden" in json.dumps(out, default=str)
    assert (await dream.items())[0]["status"] == "open"
    out = await srv.call_tool("inbox_close", {"item_id": item["id"], "outcome": "Leave it"})   # the user's token
    assert '\\"outcome\\": \\"closed\\"' in json.dumps(out, default=str)
    assert (await dream.items("closed"))[0]["outcome"] == "Leave it"


async def _person(name, note="A colleague."):
    await store.upsert_structured(StructuredEntry(id=f"people.{name}", kind=f"people.{name}", subject=name, content=note,
                                                  provenance=Provenance.USER_STATED), change_note="test")


async def test_unproven_facts_and_links_to_nobody_are_refused_a_real_link_applies(tmp_path):
    from ken.engine import Engine
    _, when = await _setup(tmp_path)
    await _person("maria")
    night = {"tidy": [
        {"kind": "project", "name": "emigration", "add": "", "link": {"kind": "person", "name": "x", "rel": "member"}},
        {"kind": "project", "name": "emigration", "add": "Sashko sold the car in October.",   # the 17:45 event is about tickets
         "source": {"event": when}},
        {"kind": "person", "name": "maria", "add": "", "link": {"kind": "project", "name": "emigration", "rel": "member"}}]}
    done = await dream.run_once(llm=FakeLLM(night))
    assert done["tidy"][0].startswith("emigration: nothing to add (a link to 'x', which Ken does not hold")
    assert done["tidy"][1] == "maria: proposed"
    (fix,) = await dream.items()
    assert (fix["title"], fix["add"]) == ("Link maria", "")
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    out = await _pumped(eng, mcp_server.inbox_apply(fix["id"]))
    assert out["outcome"] == "applied", out
    assert {(lk["src"], lk["dst"]) for lk in await store.links_of(["people.maria"])} == {("people.maria", "projects.emigration")}


async def test_apply_finds_the_record_by_its_key_and_adds_only_missing_sentences(tmp_path):
    from ken.engine import Engine
    _, when = await _setup(tmp_path)
    # the record's display subject differs from its key: Apply must still write projects.emigration
    await store.upsert_structured(StructuredEntry(id="projects.emigration", kind="projects.emigration", subject="Emigration to Portugal",
                                                  content="Moving on a D8 visa.", provenance=Provenance.USER_STATED), change_note="test")
    await dream.run_once(llm=FakeLLM({"tidy": [
        {"kind": "project", "name": "emigration", "add": "The consulate requires a round-trip ticket.", "source": {"event": when}},
        {"kind": "project", "name": "emigration", "add": "The consulate answered on the 13th.", "source": {"event": when}}]}))
    (fix,) = await dream.items()
    await store.upsert_structured(StructuredEntry(id="projects.emigration", kind="projects.emigration", subject="Emigration to Portugal",
                                                  content="Moving on a D8 visa. The consulate answered on the 13th.",
                                                  provenance=Provenance.USER_STATED), change_note="the user typed one fact")
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    assert (await _pumped(eng, mcp_server.inbox_apply(fix["id"])))["outcome"] == "applied"
    kinds = [e.kind for e in await store.get_structured("projects.")]
    assert kinds == ["projects.emigration"]                          # no copy under another key
    rec = (await store.get_structured("projects."))[0]
    assert rec.content == "Moving on a D8 visa. The consulate answered on the 13th. The consulate requires a round-trip ticket."
    assert rec.evidence and rec.evidence[-1].ref.startswith("dreamer:")   # the fix keeps where it came from


async def test_a_skipped_fact_is_not_proposed_again(tmp_path):
    _, when = await _setup(tmp_path)
    night = {"tidy": [{"kind": "project", "name": "emigration", "add": "The consulate requires a round-trip ticket.",
                       "source": {"event": when}}]}
    await dream.run_once(llm=FakeLLM(night))
    (fix,) = await dream.items()
    await dream.close(fix["id"], "skipped")
    llm = FakeLLM(night)
    again = await dream.run_once(llm=llm)
    assert again["tidy"] == ["emigration: nothing to add (a fact you skipped before was left out)"]
    assert "The consulate requires a round-trip ticket." in llm.prompts[0]   # the model sees which fact was refused


async def test_a_card_an_earlier_apply_left_is_approved_not_a_block(tmp_path):
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence
    from ken.engine import Engine
    _, when = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM({"tidy": [{"kind": "project", "name": "emigration",
                                               "add": "The consulate requires a round-trip ticket.", "source": {"event": when}}]}))
    (fix,) = await dream.items()
    # Ken was busy: the first Apply's write landed later as a card, with the Dreamer as its source
    await gate.apply(Proposal(intent="structured:projects.emigration",
                              content="Moving to Portugal on a D8 visa. The consulate requires a round-trip ticket.",
                              evidence=[Evidence(ref="dreamer:x", source="Dreamer")]), conversation_id=bus.MAIN, author="kip")
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    assert (await _pumped(eng, mcp_server.inbox_apply(fix["id"])))["outcome"] == "applied"
    assert await mcp_server.pending_cards() == []


async def test_answered_cards_are_known_ids_and_a_repeat_title_is_dropped(tmp_path):
    task, _ = await _setup(tmp_path)
    await db.audit("card_outcome", "3fa9c1d2b7e04a5f9c1d2b7e04a5f9c1", json.dumps({"outcome": "rejected"}))
    first = await dream.run_once(llm=FakeLLM({"asks": [_ask("Re-plan GTM?", "you said no on card 3fa9c1d2")]}))
    assert first["asks"] == 1 and first["dropped"] == []
    again = await dream.run_once(llm=FakeLLM({"asks": [_ask("re-plan gtm?", f"task {task.id}")]}))
    assert again["asks"] == 0 and again["dropped"] == ["re-plan gtm?: raised before"]


async def test_no_model_is_audited_and_dreamer_llm_stands_alone(tmp_path, monkeypatch):
    await _setup(tmp_path)
    monkeypatch.setattr(dream.config, "section", lambda name: {"provider": "claude-code"} if name == "llm" else {})
    monkeypatch.setattr(dream, "_cfg", lambda: {"llm": {"api_key": "YOUR_KEY", "api_style": "chat"}})
    assert await dream.run_once() == {"error": "no model"}           # the placeholder key: no model, no claude -p
    async with db.get_db() as c:
        (reason,) = await (await c.execute("SELECT reason FROM audit WHERE kind='dream_error'")).fetchone()
    assert reason.startswith("no model")
    monkeypatch.setattr(dream, "_cfg", lambda: {"llm": {"api_key": "sk-x", "api_style": "chat", "model": "deepseek-flash",
                                                         "base_url": "https://api.deepseek.com", "reasoning_effort": "max"}})
    llm = dream._llm()
    assert (type(llm).__name__, llm._model, llm._effort, llm._api_style) == ("OpenAILLM", "deepseek-flash", "max", "chat")


def test_a_chat_style_agent_never_gets_the_shared_effort(monkeypatch):
    from ken import app
    cfg = {"api_key": "sk-x", "api_style": "chat", "reasoning_effort": "high"}
    monkeypatch.setattr(app.config, "section", lambda name: cfg if name == "llm" else {})
    eng = app.build_engine()
    assert eng.assistant._llm._effort is None                         # was sent before: a 400 on many hosts
    cfg["api_style"] = "responses"
    assert app.build_engine().assistant._llm._effort == "high"
    cfg.update(api_style="chat", base_url="https://api.deepseek.com", reasoning_effort="max")   # DeepSeek takes it
    assert app.build_engine().assistant._llm._effort == "max"


async def test_apply_in_the_users_own_words(tmp_path):
    """Ken Cloud's item talk: the user changed the wording, the client applies their version."""
    from ken.engine import Engine
    _, when = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM({"tidy": [{"kind": "project", "name": "emigration",
                                               "add": "The consulate requires a round-trip ticket.", "source": {"event": when}}]}))
    (fix,) = await dream.items()
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    out = await _pumped(eng, mcp_server.inbox_apply(fix["id"], "  A one-way ticket is enough,   per the 13 Nov email. "))
    assert out["outcome"] == "applied", out
    rec = next(e for e in await store.get_structured("projects.") if e.kind == "projects.emigration")
    assert rec.content.endswith("A one-way ticket is enough, per the 13 Nov email.")
    assert "round-trip" not in rec.content                           # the proposal is not written as well
    (done,) = await dream.items("closed")
    assert done["add"] == "A one-way ticket is enough, per the 13 Nov email." and done["outcome"] == "applied"


async def test_an_edit_survives_a_block_and_replaces_an_earlier_applys_waiting_card(tmp_path):
    """Review of #24: an edit blocked by an outside card is kept on the item for the retry; a card an
    earlier Apply left (the OLD words) is declined, and the NEW words are written. Empty or huge: refused."""
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence
    from ken.engine import Engine
    _, when = await _setup(tmp_path)
    await dream.run_once(llm=FakeLLM({"tidy": [{"kind": "project", "name": "emigration",
                                               "add": "The consulate requires a round-trip ticket.", "source": {"event": when}}]}))
    (fix,) = await dream.items()
    for bad in ("   ", "x" * 1001):
        assert json.loads(await mcp_server.inbox_apply(fix["id"], bad))["outcome"] == "error"
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    # an outside agent's card holds the record: blocked, and the user's words are kept for the retry
    await gate.apply(Proposal(intent="structured:projects.emigration", content="Moving to Spain instead.",
                              evidence=[Evidence(ref="x", source="codex")]), conversation_id=bus.MAIN, author="codex")
    out = json.loads(await mcp_server.inbox_apply(fix["id"], "A one-way ticket is enough."))
    assert out["outcome"] == "blocked" and "kept" in out["text"]
    assert (await dream.get(fix["id"]))["add"] == "A one-way ticket is enough."
    (theirs,) = await mcp_server.pending_cards()
    await _pumped(eng, mcp_server.decide(theirs["card_id"], False))
    # an earlier Apply (Ken was busy) left a Dreamer card with the OLD words
    await gate.apply(Proposal(intent="structured:projects.emigration", content="Old words: a round-trip ticket.",
                              evidence=[Evidence(ref="dreamer:x", source="Dreamer")]), conversation_id=bus.MAIN, author="Dreamer")
    out = await _pumped(eng, mcp_server.inbox_apply(fix["id"], "A one-way ticket is enough."))
    assert out["outcome"] == "applied", out
    rec = next(e for e in await store.get_structured("projects.") if e.kind == "projects.emigration")
    assert "A one-way ticket is enough." in rec.content and "round-trip" not in rec.content
    assert await mcp_server.pending_cards() == []
