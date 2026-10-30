"""Four gaps in Ken's model of a person's world: a task's owner, task-to-task dependencies,
one read for closed tasks and events, and event time (`at`) on direct writes so a late,
old update never wins. Deterministic: captures + the gate + the store, no model."""
import asyncio
import json
from datetime import datetime, timedelta

from ken import db, mcp_server
from ken.agents.assistant import Assistant, _parse_when
from ken.agents.base import Context
from ken.capabilities import MemoryReader, Proposer, ViewPublisher
from ken.captures import Captures
from ken.contracts import Provenance, StructuredEntry
from ken.conversation import bus
from ken.memory import store
from ken.util import local_ms, now_ms


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


def _cap(prov: Provenance = Provenance.USER_INSTRUCTED) -> Captures:
    ctx = Context(conversation_id=bus.MAIN, now=now_ms(), message="")
    ctx.memory = MemoryReader(Assistant.slice)
    ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
    return Captures(ctx, parse_when=_parse_when, provenance=prov)


def _iso(days: float = 0, hour: int = 10) -> str:
    return (datetime.now() + timedelta(days=days)).replace(hour=hour, minute=0, second=0,
                                                           microsecond=0).strftime("%Y-%m-%dT%H:%M")


async def _person(name: str):
    kind = f"people.{name.lower()}"
    await store.upsert_structured(StructuredEntry(id=kind, kind=kind, subject=name, content=f"{name} is a colleague",
                                                  provenance=Provenance.USER_STATED), change_note="test")


def _row(rows, text):
    return next(r for r in rows if r["text"] == text)


# ── 1. task owner ────────────────────────────────────────────────────────────

async def test_add_with_an_owner_and_reassign(tmp_path):
    await _setup(tmp_path)
    await _person("Robert")
    await _person("Ruth")
    r = await _cap().add_task("send the venue contract", owner="Robert")
    assert r.outcome == "task"
    (t,) = await store.open_tasks()
    assert t.owner == "people.robert"                       # a known person: the record's kind
    await _cap().add_task("book the flights")
    assert _row(json.loads(await mcp_server.tasks())["tasks"], "book the flights")["owner"] is None   # the user
    r = await _cap().assign_task(t.id, owner="Ruth")
    assert r.outcome == "assigned"
    assert (await store.task_by_id(t.id)).owner == "people.ruth"
    row = _row(json.loads(await mcp_server.tasks())["tasks"], "send the venue contract")
    assert row["owner"] == "ruth"                           # the new owner shown, the old one gone
    r = await _cap().assign_task("venue contract", owner="me")
    assert r.outcome == "assigned" and (await store.task_by_id(t.id)).owner is None    # back to the user


async def test_owner_appears_in_every_row_shape(tmp_path):
    await _setup(tmp_path)
    await _person("Robert")
    await _cap().add_task("ask Robert for the budget", owner="Robert")
    (t,) = await store.open_tasks()
    assert _row(json.loads(await mcp_server.tasks())["tasks"], t.text)["owner"] == "robert"
    ctx = json.loads(await mcp_server.get_context(t.id))
    assert ctx["task"]["owner"] == "robert"
    assert json.loads(await mcp_server.find("budget"))["tasks"][0]["owner"] == "robert"
    person = next(p for p in json.loads(await mcp_server.world())["people"] if p["name"] == "robert")
    assert person["tasks"][0]["owner"] == "robert"


async def test_an_unknown_owner_is_stored_as_the_name_given_and_said(tmp_path):
    await _setup(tmp_path)
    r = await _cap().add_task("sign the lease", owner="Dana Kim")
    assert r.outcome == "task" and "Dana Kim" in r.text     # the reply says Ken has no such person
    (t,) = await store.open_tasks()
    assert t.owner == "Dana Kim"
    assert _row(json.loads(await mcp_server.tasks())["tasks"], "sign the lease")["owner"] == "Dana Kim"


async def test_assign_from_an_agent_token_asks_first(tmp_path):
    await _setup(tmp_path)
    await _person("Ruth")
    tid, _ = await store.add_task("draft the brief", Provenance.USER_STATED)
    r = await _cap(Provenance.WORLD).assign_task(tid, owner="Ruth")
    assert r.outcome == "pending"
    (card,) = await mcp_server.pending_cards()
    assert card["title"] == "Assign?"
    assert (await store.task_by_id(tid)).owner is None      # nothing changed before the Yes
    from ken.consent import gate
    details: dict = {}
    assert await gate.resolve(card["card_id"], approved=True, details=details) == "act"
    ctx = Context(conversation_id=bus.MAIN, now=now_ms(), message="")
    ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
    assert await Assistant().act(ctx, details["proposal"]) == "Reassigned."    # kip performs it on the Yes
    assert (await store.task_by_id(tid)).owner == "people.ruth"


# ── 2. task dependencies ─────────────────────────────────────────────────────

async def test_link_blocks_until_the_dependency_is_done_then_unlink(tmp_path):
    await _setup(tmp_path)
    a, _ = await store.add_task("send the invitations", Provenance.USER_STATED)
    b, _ = await store.add_task("confirm the venue", Provenance.USER_STATED)
    r = await _cap().link_tasks(a, depends_on=b)
    assert r.outcome == "linked"
    row = _row(json.loads(await mcp_server.tasks())["tasks"], "send the invitations")
    assert row["blocked"] is True
    assert [d["id"] for d in row["depends_on"]] == [b] and row["depends_on"][0]["text"] == "confirm the venue"
    ctx = json.loads(await mcp_server.get_context(a))
    assert ctx["task"]["blocked"] is True and ctx["task"]["depends_on"][0]["id"] == b
    assert (await _cap().complete_task(b)).outcome == "completed"
    row = _row(json.loads(await mcp_server.tasks())["tasks"], "send the invitations")
    assert row["blocked"] is False and row["depends_on"][0]["status"] == "done"
    r = await _cap().link_tasks("invitations", depends_on="confirm the venue", remove=True)
    assert r.outcome == "unlinked"
    assert "depends_on" not in _row(json.loads(await mcp_server.tasks())["tasks"], "send the invitations")


async def test_world_task_rows_carry_the_dependency(tmp_path):
    await _setup(tmp_path)
    await _person("Robert")
    a, _ = await store.add_task("ask Robert to sign off", Provenance.USER_STATED)
    b, _ = await store.add_task("finish the draft", Provenance.USER_STATED)
    assert (await _cap().link_tasks(a, depends_on=b)).outcome == "linked"
    person = next(p for p in json.loads(await mcp_server.world())["people"] if p["name"] == "robert")
    assert person["tasks"][0]["depends_on"][0]["id"] == b and person["tasks"][0]["blocked"] is True


async def test_a_dependency_cycle_is_refused(tmp_path):
    await _setup(tmp_path)
    a, _ = await store.add_task("task a", Provenance.USER_STATED)
    b, _ = await store.add_task("task b", Provenance.USER_STATED)
    c, _ = await store.add_task("task c", Provenance.USER_STATED)
    assert (await _cap().link_tasks(a, depends_on=b)).outcome == "linked"
    assert (await _cap().link_tasks(b, depends_on=c)).outcome == "linked"
    r = await _cap().link_tasks(c, depends_on=a)            # c → a → b → c
    assert r.outcome == "cycle" and "cycle" in r.text
    assert (await _cap().link_tasks(a, depends_on=a)).outcome == "cycle"     # itself
    assert "depends_on" not in _row(json.loads(await mcp_server.tasks())["tasks"], "task c")


async def test_assign_and_link_are_typed_tools_through_the_engine(tmp_path):
    from ken.engine import Engine
    await _setup(tmp_path)
    await _person("Ruth")
    a, _ = await store.add_task("ship it", Provenance.USER_STATED)
    b, _ = await store.add_task("test it", Provenance.USER_STATED)
    eng = Engine()
    eng.register_agent(Assistant(llm=None), assistant=True)

    async def pump():
        for _ in range(40):
            await eng.pump_inbox()
            await asyncio.sleep(0.05)
    t = asyncio.create_task(pump())
    out = json.loads(await mcp_server.typed("assign_task", {"task": a, "owner": "Ruth"}, provenance="user", timeout=5))
    out2 = json.loads(await mcp_server.typed("link_tasks", {"task": a, "depends_on": b}, provenance="user", timeout=5))
    await t
    assert out["outcome"] == "assigned" and out2["outcome"] == "linked"
    assert {"assign_task", "link_tasks", "events"} <= set(mcp_server.TOOL_NAMES)


async def test_kip_has_the_owner_and_assign_tools(tmp_path):
    await _setup(tmp_path)
    class _LLM:
        async def complete(self, system, messages, *, tools=None):
            self.tools = {t.name: t for t in tools or []}
            self.system = system
            return "ok"
    llm = _LLM()
    ctx = Context(conversation_id=bus.MAIN, now=now_ms(), message="hi", history=[])
    ctx.memory = MemoryReader(Assistant.slice)
    ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
    ctx.view = ViewPublisher("main")
    await Assistant(llm=llm)._chat(ctx, "hi")
    assert "owner" in llm.tools["add_task"].parameters["properties"]
    assert set(llm.tools["assign_task"].parameters["properties"]) >= {"task", "owner"}
    assert "assign_task" in llm.system


# ── 3. one read for closed tasks and events ──────────────────────────────────

async def test_tasks_status_reads_done_and_dropped_default_unchanged(tmp_path):
    await _setup(tmp_path)
    await store.add_task("open one", Provenance.USER_STATED)
    d, _ = await store.add_task("done one", Provenance.USER_STATED)
    x, _ = await store.add_task("dropped one", Provenance.USER_STATED)
    await store.complete_task(d)
    await store.discard_task(x, "no longer needed")
    default = json.loads(await mcp_server.tasks())
    assert [r["text"] for r in default["tasks"]] == ["open one"]
    assert json.loads(await mcp_server.tasks(status="open")) == default
    done = json.loads(await mcp_server.tasks(status="done"))["tasks"]
    assert [(r["text"], r["status"]) for r in done] == [("done one", "done")]
    dropped = json.loads(await mcp_server.tasks(status="dropped"))["tasks"]
    assert [(r["text"], r["status"], r["drop_reason"]) for r in dropped] == [("dropped one", "dropped", "no longer needed")]
    every = {r["text"]: r["status"] for r in json.loads(await mcp_server.tasks(status="all"))["tasks"]}
    assert every == {"open one": "open", "done one": "done", "dropped one": "dropped"}
    assert "error" in json.loads(await mcp_server.tasks(status="later"))


async def test_events_in_a_date_window(tmp_path):
    await _setup(tmp_path)
    cap = _cap()
    await cap.log_event("signed the lease", when=_iso(-3))
    await cap.log_event("met the architect", when=_iso(-10))
    await cap.log_event("old offsite", when=_iso(-60))
    recent = json.loads(await mcp_server.events())["events"]                  # default: the last 30 days
    assert [e["what"] for e in recent] == ["signed the lease", "met the architect"]
    assert recent[0]["when"] == _iso(-3)
    since, until = _iso(-12)[:10], _iso(-5)[:10]
    win = json.loads(await mcp_server.events(since=since, until=until))["events"]
    assert [e["what"] for e in win] == ["met the architect"]
    assert "error" in json.loads(await mcp_server.events(since="last week"))


# ── 4. event time on direct writes: a late, old update never wins ────────────

async def test_a_late_older_reschedule_is_stale(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("renew the passport", Provenance.USER_STATED)
    t2_deadline, t1_deadline = _iso(20), _iso(10)
    r = await _cap().reschedule_task(tid, deadline=t2_deadline, at=_iso(-1))
    assert r.outcome == "rescheduled"
    r = await _cap().reschedule_task(tid, deadline=t1_deadline, at=_iso(-2))   # said earlier, arrived later
    assert r.outcome == "stale"
    assert t2_deadline[:10] in r.text                                         # the current value
    assert (await store.task_by_id(tid)).deadline == _parse_when(t2_deadline)


async def test_a_late_still_open_update_does_not_reopen_a_done_task(tmp_path):
    await _setup(tmp_path)
    await _cap().add_task("pay the deposit", at=_iso(-5))
    assert (await _cap().complete_task("pay the deposit", at=_iso(-1))).outcome == "completed"
    r = await _cap().add_task("pay the deposit", at=_iso(-3))
    assert r.outcome == "stale"
    r = await _cap().reschedule_task("pay the deposit", deadline=_iso(7), at=_iso(-3))
    assert r.outcome == "stale"
    (t,) = await store.find_tasks("pay the deposit")
    assert t.status == "done" and await store.open_tasks() == []


async def test_stale_applies_to_every_task_write_and_to_remember(tmp_path):
    await _setup(tmp_path)
    await _person("Ruth")
    a, _ = await store.add_task("write the spec", Provenance.USER_STATED)
    b, _ = await store.add_task("review the spec", Provenance.USER_STATED)
    assert (await _cap().assign_task(a, owner="Ruth", at=_iso(-1))).outcome == "assigned"
    old = _iso(-4)
    assert (await _cap().assign_task(a, owner="me", at=old)).outcome == "stale"
    assert (await _cap().link_tasks(a, depends_on=b, at=old)).outcome == "stale"
    assert (await _cap().schedule_task(a, start=_iso(1), minutes=30, at=old)).outcome == "stale"
    assert (await _cap().discard_task(a, reason="x", at=old)).outcome == "stale"
    assert (await _cap().complete_task(a, at=old)).outcome == "stale"
    t = await store.task_by_id(a)
    assert (t.owner, t.status, t.slot_start) == ("people.ruth", "open", None)
    # a structured record: an older remember never replaces a newer note
    assert (await _cap().remember("preference", "coffee", "oat flat white", at=_iso(-1))).outcome == "applied"
    r = await _cap().remember("preference", "coffee", "black, no sugar", at=_iso(-3))
    assert r.outcome == "stale" and "oat flat white" in r.text
    assert (await store.get_structured("preferences.coffee"))[0].content == "oat flat white"
    assert (await _cap().forget("preference", "coffee", at=_iso(-3))).outcome == "stale"


async def test_no_at_behaves_as_today_and_a_bad_at_changes_nothing(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("call the bank", Provenance.USER_STATED)
    assert (await _cap().reschedule_task(tid, deadline=_iso(5), at=_iso(-1))).outcome == "rescheduled"
    assert (await _cap().reschedule_task(tid, deadline=_iso(9))).outcome == "rescheduled"   # no at: applies
    assert (await store.task_by_id(tid)).deadline == _parse_when(_iso(9))
    r = await _cap().reschedule_task(tid, deadline=_iso(12), at="yesterday-ish")
    assert r.outcome == "invalid"
    assert (await store.task_by_id(tid)).deadline == _parse_when(_iso(9))


async def test_log_event_at_is_when_reported_and_when_stays_the_event_time(tmp_path):
    await _setup(tmp_path)
    await _cap().log_event("the roof leaked", at=_iso(-2))                       # no `when`: it happened when said
    await _cap().log_event("the plumber came", when=_iso(-6), at=_iso(-2))
    ev = {e["what"]: e["when"] for e in json.loads(await mcp_server.events())["events"]}
    assert ev == {"the roof leaked": _iso(-2), "the plumber came": _iso(-6)}


async def test_kip_records_a_dependency_from_the_users_words(tmp_path):
    """'I can only send the invitations after the venue is confirmed': kip calls link_tasks."""
    await _setup(tmp_path)
    await store.add_task("send the invitations", Provenance.USER_STATED)
    b, _ = await store.add_task("confirm the venue", Provenance.USER_STATED)

    class _LLM:
        def __init__(self, calls):
            self.calls, self.results = calls, []

        async def complete(self, system, messages, *, tools=None):
            self.system = system
            specs = {t.name: t for t in tools or []}
            for name, args in self.calls:
                self.results.append(await specs[name].fn(**args))
            return "noted"

    def _ctx(msg):
        ctx = Context(conversation_id=bus.MAIN, now=now_ms(), message=msg, history=[])
        ctx.memory = MemoryReader(Assistant.slice)
        ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
        ctx.view = ViewPublisher("main")
        return ctx

    llm = _LLM([("link_tasks", {"task": "send the invitations", "depends_on": "confirm the venue"}),
                ("link_tasks", {"task": "confirm the venue", "depends_on": "send the invitations"})])
    await Assistant(llm=llm).run(_ctx("I can only send the invitations after the venue is confirmed"))
    assert "link_tasks" in llm.system and "waits on" in llm.system
    assert "cycle" in llm.results[1]                                  # the reverse link is refused
    row = _row(json.loads(await mcp_server.tasks())["tasks"], "send the invitations")
    assert row["blocked"] is True and [d["id"] for d in row["depends_on"]] == [b]
    assert "depends_on" not in _row(json.loads(await mcp_server.tasks())["tasks"], "confirm the venue")
    llm = _LLM([("link_tasks", {"task": "invitations", "depends_on": "venue", "remove": True})])
    await Assistant(llm=llm).run(_ctx("the invitations no longer wait on the venue"))
    assert "depends_on" not in _row(json.loads(await mcp_server.tasks())["tasks"], "send the invitations")


# ── review fixes ─────────────────────────────────────────────────────────────


async def test_a_second_assign_report_with_another_owner_replaces_the_card(tmp_path):
    await _setup(tmp_path)
    await _person("Robert")
    await _person("Anna")
    tid, _ = await store.add_task("send the contract", Provenance.USER_STATED)
    assert (await _cap(Provenance.WORLD).assign_task(tid, owner="Robert")).outcome == "pending"
    assert (await _cap(Provenance.WORLD).assign_task(tid, owner="Anna")).outcome == "pending"
    (card,) = await mcp_server.pending_cards()
    assert "anna" in json.dumps(card).lower() and "robert" not in json.dumps(card).lower()


async def test_a_second_link_report_to_another_task_replaces_the_card(tmp_path):
    await _setup(tmp_path)
    a, _ = await store.add_task("ship the app", Provenance.USER_STATED)
    b, _ = await store.add_task("fix the login", Provenance.USER_STATED)
    c, _ = await store.add_task("write the docs", Provenance.USER_STATED)
    assert (await _cap(Provenance.WORLD).link_tasks(a, depends_on=b)).outcome == "pending"
    assert (await _cap(Provenance.WORLD).link_tasks(a, depends_on=c)).outcome == "pending"
    (card,) = await mcp_server.pending_cards()
    assert "write the docs" in json.dumps(card) and "fix the login" not in json.dumps(card)


async def test_planning_edits_do_not_make_a_late_completion_stale(tmp_path):
    await _setup(tmp_path)
    a, _ = await store.add_task("call the bank", Provenance.USER_STATED)
    b, _ = await store.add_task("find the account number", Provenance.USER_STATED)
    assert (await _cap().schedule_task(a, start=_iso(1), minutes=30)).outcome == "slotted"
    assert (await _cap().link_tasks(a, depends_on=b)).outcome == "linked"
    assert (await _cap().complete_task(a, at=_iso(-1))).outcome == "completed"


async def test_a_date_only_at_means_the_end_of_that_day(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("renew the passport", Provenance.USER_STATED)
    assert (await _cap().reschedule_task(tid, deadline=_iso(10))).outcome == "rescheduled"   # changed now
    today = datetime.now().strftime("%Y-%m-%d")
    assert (await _cap().complete_task(tid, at=today)).outcome == "completed"


async def test_a_future_at_is_clamped_to_now(tmp_path):
    await _setup(tmp_path)
    await _person("Ruth")
    tid, _ = await store.add_task("draft the brief", Provenance.USER_STATED)
    assert (await _cap().assign_task(tid, owner="Ruth", at=_iso(365))).outcome == "assigned"
    assert (await store.task_by_id(tid)).changed_at <= now_ms()
    assert (await _cap().complete_task(tid, at=datetime.now().strftime("%Y-%m-%d"))).outcome == "completed"


async def test_completed_at_is_when_the_work_was_done(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("pay the rent", Provenance.USER_STATED)
    done = _iso(-3)
    assert (await _cap().complete_task(tid, at=done)).outcome == "completed"
    assert (await store.task_by_id(tid)).completed_at == local_ms(done)


async def test_ambiguity_wins_over_stale_and_no_stale_from_an_unrelated_closed_task(tmp_path):
    await _setup(tmp_path)
    a, _ = await store.add_task("write the sales report", Provenance.USER_STATED)
    await store.add_task("write the board report", Provenance.USER_STATED)
    assert (await _cap().reschedule_task(a, deadline=_iso(5))).outcome == "rescheduled"
    assert (await _cap().complete_task("report", at=_iso(-2))).outcome == "ambiguous"
    await _cap().add_task("pay the deposit")
    assert (await _cap().complete_task("pay the deposit")).outcome == "completed"
    assert (await _cap().complete_task("deposit for the hall", at=_iso(-2))).outcome == "task_absent"


async def test_add_task_says_when_the_owner_was_not_applied(tmp_path):
    await _setup(tmp_path)
    await _person("Robert")
    await _person("Anna")
    await _cap().add_task("send the contract", owner="Robert")
    r = await _cap().add_task("send the contract", owner="Anna")
    assert "keeps the owner" in r.text
    assert (await store.open_tasks())[0].owner == "people.robert"


async def test_forgetting_a_person_leaves_their_tasks_a_plain_name_owner(tmp_path):
    await _setup(tmp_path)
    await _person("Robert")
    await _cap().add_task("send the contract", owner="Robert")
    assert await store.delete_structured("people.robert")
    (t,) = await store.open_tasks()
    assert t.owner == "Robert"
