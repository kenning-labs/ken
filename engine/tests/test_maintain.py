"""The maintainer: a stale task becomes one "drop it?" card; kip performs the drop on a Yes."""
from datetime import datetime

from ken import db, maintain
from ken.agents.assistant import Assistant
from ken.contracts import Provenance, RenderableKind
from ken.conversation import bus
from ken.engine import Engine
from ken.memory import store
from ken.util import now_ms

DAY = 86_400_000


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


async def _asks():
    return [r for _, r in await bus.read_outbox(bus.MAIN, 0) if r.kind is RenderableKind.ASK]


async def test_stale_task_gets_one_drop_card_and_kip_drops_it(tmp_path):
    await _setup(tmp_path)
    now = now_ms()
    old_id, _ = await store.add_task("start the birdhouse build", Provenance.USER_STATED, deadline=now - 14 * DAY)
    await store.add_task("call dentist", Provenance.USER_STATED, deadline=now - 2 * DAY)   # not stale
    await store.add_task("read that paper", Provenance.USER_STATED)                              # untimed
    assert await maintain.run_once(now) == {"loop_cards": 0, "stale_cards": 1, "expired_events": 0}
    assert await maintain.stale_cards(await store.open_tasks(), now) == 0                       # one open card per task
    (card,) = await _asks()
    assert card.body["title"] == "Discard?" and card.author == "kip"
    assert card.body["subject"] == "start the birdhouse build" and "14 days overdue" in card.body["note"]
    # a Yes: kip drops it — closed as dropped, not counted as done
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    await eng._dispatch_token(type("A", (), {"token": f"approve:{card.id}", "text": "", "channel": "tui",
                                            "conversation_id": bus.MAIN, "id": "a"})())
    assert old_id not in {t.id for t in await store.open_tasks()}
    assert (await store.task_by_id(old_id)).status == "dropped"
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM pending")).fetchone())[0] == 0
        assert (await (await c.execute("SELECT count(*) FROM audit WHERE kind='maintain'")).fetchone())[0] == 1


async def test_daily_pass_runs_once_per_day_after_nine(tmp_path):
    await _setup(tmp_path)
    await store.add_task("stale", Provenance.USER_STATED, deadline=now_ms() - 20 * DAY)
    at_8 = int(datetime.now().replace(hour=8, minute=0).timestamp() * 1000)
    at_10 = int(datetime.now().replace(hour=10, minute=0).timestamp() * 1000)
    assert await maintain.run_scheduled(at_8) == 0                    # too early
    assert await maintain.run_scheduled(at_10) == 1                   # the day's pass
    assert await maintain.run_scheduled(at_10 + 3_600_000) == 0       # already ran today
    assert await maintain.run_scheduled(at_10 + DAY) == 0             # tomorrow: asked already → no duplicate


async def test_a_no_keeps_the_task_and_is_remembered_until_it_moves(tmp_path):
    from ken.consent import gate
    await _setup(tmp_path)
    now = now_ms()
    tid, _ = await store.add_task("stale thing", Provenance.USER_STATED, deadline=now - 10 * DAY)
    assert await maintain.stale_cards(await store.open_tasks(), now) == 1
    (card,) = await _asks()
    await gate.resolve(card.id, approved=False)                       # No: keep it
    assert (await store.task_by_id(tid)).status == "open"
    assert await maintain.stale_cards(await store.open_tasks(), now + DAY) == 0                  # not asked again…
    await store.reschedule(tid, now - 9 * DAY)                         # …until the task moves
    assert await maintain.stale_cards(await store.open_tasks(), now + DAY) == 1


async def test_engine_tick_runs_the_maintainer_and_survives_its_failure(tmp_path, monkeypatch):
    await _setup(tmp_path)
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    calls = []

    async def fake(now):
        calls.append(now); raise RuntimeError("boom")
    monkeypatch.setattr(maintain, "run_scheduled", fake)
    await eng.tick()
    assert calls
    async with db.get_db() as c:
        row = await (await c.execute("SELECT reason FROM audit WHERE kind='maintain_error'")).fetchone()
    assert "boom" in row["reason"]


def _tap(eng, card_id):
    return eng._dispatch_token(type("A", (), {"token": f"approve:{card_id}", "text": "", "channel": "tui",
                                             "conversation_id": bus.MAIN, "id": "a"})())


async def _report(task_id, text, days_ago, now):
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence
    await gate.apply(Proposal(intent="event", content=text, at=now - days_ago * DAY,
                              evidence=[Evidence(ref=task_id, source="report")]),
                     conversation_id=bus.MAIN, author="kip")


async def test_reported_open_task_gets_one_done_card_and_a_yes_closes_it(tmp_path):
    await _setup(tmp_path)
    now = now_ms()
    tid, _ = await store.add_task("order the briefcase", Provenance.USER_STATED, deadline=now - 10 * DAY)
    await _report(tid, "ordered it from Bellroy, arrives Friday", 3, now)
    fresh, _ = await store.add_task("write the memo", Provenance.USER_STATED)             # untimed, fresh report
    await _report(fresh, "drafted half", 0, now)
    future, _ = await store.add_task("book flights", Provenance.USER_STATED, deadline=now + 5 * DAY)
    await _report(future, "compared prices", 4, now)                                       # progress, not closure
    done = await maintain.run_once(now)
    assert done == {"loop_cards": 1, "stale_cards": 0, "expired_events": 0}   # loop first; no stale card on the same task
    (card,) = await _asks()
    assert card.body["title"] == "Mark done?" and card.body["subject"] == "order the briefcase"
    assert "3 days ago" in card.body["note"] and "Bellroy" in card.body["note"]
    assert await maintain.run_once(now) == {"loop_cards": 0, "stale_cards": 0, "expired_events": 0}  # asked once
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)
    await _tap(eng, card.id)
    assert tid not in {t.id for t in await store.open_tasks()}
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM pending")).fetchone())[0] == 0
        assert (await (await c.execute("SELECT count(*) FROM audit WHERE kind='maintain'")).fetchone())[0] == 2


async def test_a_no_on_a_done_card_is_remembered_until_a_new_report(tmp_path):
    from ken.consent import gate
    await _setup(tmp_path)
    now = now_ms()
    tid, _ = await store.add_task("call the bank", Provenance.USER_STATED)
    await _report(tid, "left a voicemail", 3, now)
    assert (await maintain.run_once(now))["loop_cards"] == 1
    (card,) = await _asks()
    assert await gate.resolve(card.id, approved=False, conversation_id=bus.MAIN) == "rejected"
    assert (await maintain.run_once(now + DAY))["loop_cards"] == 0            # the No holds
    await _report(tid, "they called back, sorted", 0, now + DAY)                # a NEW report: ask again later
    assert (await maintain.run_once(now + DAY))["loop_cards"] == 0            # too fresh
    assert (await maintain.run_once(now + 4 * DAY))["loop_cards"] == 1


async def test_an_expired_event_is_hidden_never_deleted(tmp_path):
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence
    await _setup(tmp_path)
    now = now_ms()
    tid, _ = await store.add_task("decide on the offer", Provenance.USER_STATED)
    await gate.apply(Proposal(intent="event", content="Acme offer stands until Friday", at=now - 5 * DAY,
                              evidence=[Evidence(ref=tid, source="report")], meta={"until": now - DAY}),
                     conversation_id=bus.MAIN, author="kip")
    await gate.apply(Proposal(intent="event", content="signed the lease", at=now - 5 * DAY),
                     conversation_id=bus.MAIN, author="kip")
    assert [e.summary for e in await store.recent_events(now=now)] == ["signed the lease"]
    assert await store.find_events("offer", now=now) == []
    assert [e.summary for e in await store.recent_events(now=now - 3 * DAY)] == ["Acme offer stands until Friday", "signed the lease"]
    (rep,) = await store.events_citing(tid)                                     # a report stays reachable by ref
    assert rep.summary.startswith("Acme") and rep.until == now - DAY
    assert await store.expired_events(now) == 1
    import json

    from ken import mcp_server
    ctx = json.loads(await mcp_server.get_context(tid, now=now))
    assert ctx["events"] == [] and ctx["last_report"]["what"].startswith("Acme")


async def test_a_done_card_supersedes_an_open_still_real_card_and_a_no_does_not_hide_the_task(tmp_path):
    from ken.consent import gate
    await _setup(tmp_path)
    now = now_ms()
    tid, _ = await store.add_task("file the taxes", Provenance.USER_STATED, deadline=now - 10 * DAY)
    assert (await maintain.run_once(now))["stale_cards"] == 1                  # day 10: still real?
    await _report(tid, "sent the forms to the accountant", 3, now + 3 * DAY)
    done = await maintain.run_once(now + 3 * DAY)
    assert done["loop_cards"] == 1 and done["stale_cards"] == 0
    asks = await _asks()
    open_ids = {r["ask_id"] for r in await _pending()}
    assert len(asks) == 2 and len(open_ids) == 1                                # the old card was withdrawn
    (card,) = [a for a in asks if a.id in open_ids]
    assert card.body["title"] == "Mark done?"
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM audit WHERE kind='card_withdrawn'")).fetchone())[0] == 1
    # a No on done? is remembered for THAT report, but the task is not hidden from the stale pass
    assert await gate.resolve(card.id, approved=False, conversation_id=bus.MAIN) == "rejected"
    done = await maintain.run_once(now + 4 * DAY)
    assert done == {"loop_cards": 0, "stale_cards": 0, "expired_events": 0}   # stale stamp (task, due) still holds
    await store.reschedule(tid, now + 5 * DAY)                                   # moved… and missed again
    done = await maintain.run_once(now + 13 * DAY)
    assert done["loop_cards"] == 0 and done["stale_cards"] == 1                # asked afresh for the new due


async def _pending():
    async with db.get_db() as c:
        return await (await c.execute("SELECT ask_id, kind FROM pending")).fetchall()


async def test_the_loop_stamp_follows_the_newest_report_by_insert_order(tmp_path):
    from ken.consent import gate
    await _setup(tmp_path)
    now = now_ms()
    tid, _ = await store.add_task("paint the fence", Provenance.USER_STATED)
    await _report(tid, "bought the paint", 3, now)
    assert (await maintain.run_once(now))["loop_cards"] == 1
    (card,) = await _asks()
    await gate.resolve(card.id, approved=False, conversation_id=bus.MAIN)
    await _report(tid, "did the first half two weeks ago, forgot to say", 14, now)   # newer word, older `at`
    assert (await maintain.run_once(now))["loop_cards"] == 1                    # a new report re-asks
    assert "14 days ago" in (await _asks())[-1].body["note"]


async def test_until_floors_past_and_date_only_and_non_integers(tmp_path):
    from ken.agents.assistant import Assistant, _parse_when
    from ken.agents.base import Context, Proposal
    from ken.capabilities import MemoryReader, Proposer
    from ken.captures import Captures
    from ken.consent import gate
    await _setup(tmp_path)
    now = now_ms()
    a = Assistant(llm=None)
    ctx = Context(conversation_id=bus.MAIN, now=now, message="")
    ctx.memory = MemoryReader(a.slice); ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
    cap = Captures(ctx, parse_when=_parse_when)
    r = await cap.log_event("offer expired", until="2020-01-01T09:00")
    assert r.outcome == "event" and "already passed" in r.text
    r = await cap.log_event("sale on", when="tuesday-ish", until="soon-ish")
    assert "set it to just now" in r.text and "never expires" in r.text
    r = await cap.log_event("offer stands", until="2099-03-05")
    assert r.text == "logged that"
    by = {e.summary: e for e in await store.recent_events(now=now)}
    assert by["offer expired"].until is None and by["sale on"].until is None
    assert datetime.fromtimestamp(by["offer stands"].until / 1000).strftime("%Y-%m-%d %H:%M:%S") == "2099-03-05 23:59:59"
    # a caller that skips the parser: a text `until` is dropped at the write site, audited
    await gate.apply(Proposal(intent="event", content="from a drop-in", meta={"until": "2026-10-01T09:00"}),
                     conversation_id=bus.MAIN, author="dropin")
    assert {e.summary: e.until for e in await store.recent_events(now=now)}["from a drop-in"] is None
    async with db.get_db() as c:
        assert (await (await c.execute("SELECT count(*) FROM audit WHERE kind='until_dropped'")).fetchone())[0] == 1


async def test_a_task_id_names_exactly_one_task_where_the_text_is_ambiguous(tmp_path):
    from ken.agents.assistant import Assistant, _parse_when
    from ken.agents.base import Context
    from ken.capabilities import MemoryReader, Proposer
    from ken.captures import Captures
    await _setup(tmp_path)
    a = Assistant(llm=None)
    ctx = Context(conversation_id=bus.MAIN, now=now_ms(), message="")
    ctx.memory = MemoryReader(a.slice); ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
    cap = Captures(ctx, parse_when=_parse_when)
    one, _ = await store.add_task("investigate pre order skips lines", Provenance.USER_INSTRUCTED)
    two, _ = await store.add_task("investigate pre order loses place", Provenance.USER_INSTRUCTED)
    assert (await cap.complete_task("investigate pre order")).outcome == "ambiguous"
    assert (await cap.complete_task(two)).outcome == "completed"
    assert [t.id for t in await store.open_tasks()] == [one]
    assert (await cap.complete_task("")).outcome == "task_absent"          # an empty call matches nothing
