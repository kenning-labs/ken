"""Ken as an MCP server — the tools on the inbox/outbox seam, plus the reads."""
import asyncio
import json
import time

from ken import db, mcp_server
from ken.agents import assistant_view
from ken.contracts import Provenance, Renderable, RenderableKind
from ken.conversation import bus
from ken.memory import store
from ken.util import new_id, now_ms


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


async def _fake_engine(reply: str | None, *, card: bool = False):
    """kip answers the next message: an ADVICE echoing the action id, or a consent card."""
    for _ in range(100):
        pending = await bus.take_unhandled_actions()
        if pending:
            seq, action = pending[0]
            if card:                                        # a real card, stamped with the action
                from ken.agents.base import Proposal
                from ken.consent import gate
                from ken.contracts import Evidence
                await gate.apply(Proposal(intent="structured:people.sam", content="Sam is my manager",
                                          evidence=[Evidence(ref=action.id, source="user")]),
                                 conversation_id=bus.MAIN, author="kip", action_id=action.id)
            elif reply:
                await bus.append_renderable(Renderable(
                    id=new_id(), conversation_id=bus.MAIN, kind=RenderableKind.ADVICE, author="kip",
                    body={"text": reply, "reply_to": action.id, "outcome": "advice", "cards": [],
                          "result": []}, at=now_ms()))
            await bus.mark_handled(seq)
            return action
        await asyncio.sleep(0.05)


async def _nudge(text: str, kind: str = "reminder"):
    await bus.append_renderable(Renderable(
        id=new_id(), conversation_id=bus.MAIN, kind=RenderableKind.ADVICE, author="kip",
        body={"type": kind, "text": text}, at=now_ms()))


async def test_tell_posts_a_tagged_message_and_returns_kips_reply(tmp_path):
    await _setup(tmp_path)
    task = asyncio.create_task(_fake_engine("Marked done."))
    out = json.loads(await mcp_server.tell("  finished the   LICENSE task ", "claude-code", timeout=5))
    assert out["text"] == "Marked done." and out["cards"] == []
    action = await task
    assert action.channel == "mcp:user"                                # stdio / in-process: the user ceiling
    assert action.text == "(via claude-code) finished the LICENSE task"
    assert action.meta == {"source": "claude-code", "provenance": "agent"}


async def test_tell_ignores_a_nudge_that_is_not_the_reply(tmp_path):
    """A reminder / the brief posted meanwhile carries no reply_to — never handed back."""
    await _setup(tmp_path)

    async def engine():
        await _nudge("ship at 5pm")                    # fires between the post and the answer
        return await _fake_engine("Noted.")
    task = asyncio.create_task(engine())
    assert json.loads(await mcp_server.tell("finished it", "x", timeout=5))["text"] == "Noted."
    await task


async def test_tell_reports_a_consent_card_instead_of_timing_out(tmp_path):
    await _setup(tmp_path)
    task = asyncio.create_task(_fake_engine(None, card=True))
    out = json.loads(await mcp_server.tell("Sam is now my manager", "x", timeout=2))
    assert out["text"] == mcp_server.CARD_RAISED and out["outcome"] == "pending"
    assert out["cards"] == [c["card_id"] for c in await mcp_server.pending_cards()]
    await task


def test_message_of_parses_json_or_raw_and_caps():
    assert mcp_server.message_of('{"text": "  hi   there "}') == "hi there"
    assert mcp_server.message_of('{"text": 5}') == "5"
    assert mcp_server.message_of("{not json") == "{not json"
    assert len(mcp_server.message_of("x" * 5000)) == 2000
    assert mcp_server.message_of("") == ""


async def test_tell_without_an_engine_says_so_and_ignores_empty(tmp_path):
    await _setup(tmp_path)
    assert json.loads(await mcp_server.tell("", "x"))["outcome"] == "empty"
    out = json.loads(await mcp_server.tell("hello", "  a very long\n source " + "x" * 100, timeout=0.5))
    assert out["outcome"] == "queued" and "when the engine starts" in out["text"]   # never read as a failure
    (_, a), = await bus.take_unhandled_actions()               # the message IS queued for later
    assert a.text.startswith("(via a very long source xxx") and len(a.text) < 80


async def test_today_is_the_lanes(tmp_path):
    await _setup(tmp_path)
    now = now_ms()
    await store.add_task("pay the fine", Provenance.USER_STATED, deadline=now - 86_400_000)
    await store.add_task("read that paper", Provenance.USER_STATED)
    assert (await mcp_server.today(now)).splitlines() == ["overdue:", "  - pay the fine",
                                                          "someday:", "  - read that paper"]


async def test_today_uses_the_lanes_kip_published(tmp_path):
    """An untimed task kip ranked 'today' on the panel is 'today' here too, not someday."""
    await _setup(tmp_path)
    now = now_ms()
    ship, _ = await store.add_task("ship the release notes", Provenance.USER_STATED)
    await store.add_task("read that paper", Provenance.USER_STATED)
    await store.set_view(bus.MAIN, assistant_view.build_view(await store.open_tasks(), now, scores=[2, 0]))
    assert (await mcp_server.today(now)).splitlines() == ["today:", "  - ship the release notes",
                                                          "someday:", "  - read that paper"]
    assert ship in (await store.get_view(bus.MAIN))["blocks"][2]["items"][1]["ids"]


def test_server_builds_with_every_tool():
    srv = mcp_server.build_server()
    names = {t.name for t in asyncio.run(srv.list_tools())}
    assert names == set(mcp_server.TOOL_NAMES)
    assert "schedule_task" in names and not names & {"set_plan", "get_plan"}   # the plan record is gone


def test_http_serves_mcp_and_the_plain_routes_behind_one_token(tmp_path):
    """`ken-mcp --http`: a real uvicorn on a free port. A wrong token is refused on every
    route; POST / answers with kip's plain reply; GET /today the day; /mcp lists both tools
    and tell_ken round-trips through the same fake engine."""
    import socket
    import threading

    import httpx
    import uvicorn
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    asyncio.run(_setup(tmp_path))
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(mcp_server.http_app("s3cret", "127.0.0.1", {"codex": "ag3nt"}),
                                           host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    auth = {"Authorization": "Bearer s3cret"}
    try:
        assert httpx.get(f"{url}/today").status_code == 401
        assert httpx.post(f"{url}/mcp", json={}).status_code == 401
        assert httpx.get(f"{url}/today", headers=auth).text == "Nothing open. Clear."
        assert httpx.get(f"{url}/pending").status_code == 401
        assert httpx.get(f"{url}/pending", headers=auth).text == "No open questions."
        assert httpx.get(f"{url}/t/s3cret/today").text == "Nothing open. Clear."   # token in the path
        assert httpx.get(f"{url}/t/wrong/today").status_code == 401
        # a tunnel rewrites Host: with a token set the Bearer is the gate, not the Host check
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
        r = httpx.post(f"{url}/mcp", json=init, headers={**auth, "Host": "abc.ngrok-free.dev",
                                                        "Accept": "application/json, text/event-stream"})
        assert r.status_code == 200, r.text

        async def engine_and_calls():
            eng = asyncio.create_task(_fake_engine("**Marked** done."))
            async with httpx.AsyncClient(timeout=10) as hc:
                r = await hc.post(url, content="finished it", headers=auth)
            assert r.text == "Marked done."
            await eng
            eng = asyncio.create_task(_fake_engine("Noted."))
            async with create_mcp_http_client(headers=auth) as hc, \
                    streamable_http_client(f"{url}/mcp", http_client=hc) as (read, write, *_), \
                    ClientSession(read, write) as session:
                await session.initialize()
                assert {t.name for t in (await session.list_tools()).tools} == set(mcp_server.TOOL_NAMES)
                out = await session.call_tool("tell", {"text": "finished it", "source": "codex"})
                assert json.loads(out.content[0].text)["text"] == "Noted."
            a = await eng
            assert a.text == "(via codex) finished it" and a.channel == "mcp:user"   # the user token
            # an AGENT token is accepted and stamps the agent ceiling on the action
            eng = asyncio.create_task(_fake_engine("Ok."))
            agent_auth = {"Authorization": "Bearer ag3nt"}
            async with create_mcp_http_client(headers=agent_auth) as hc, \
                    streamable_http_client(f"{url}/mcp", http_client=hc) as (read, write, *_), \
                    ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("tell", {"text": "did a thing", "source": "codex"})
            assert (await eng).channel == "mcp:agent:codex"
        asyncio.run(engine_and_calls())
    finally:
        server.should_exit = True


# ── the reads: straight from the store ─────────────────────────────────────────

async def test_tasks_lists_lanes_ids_and_open_questions(tmp_path):
    await _setup(tmp_path)
    now = now_ms()
    from ken.contracts import Evidence
    link = "https://x.slack.com/archives/C1/p1"
    fine, _ = await store.add_task("pay the fine", Provenance.USER_STATED, deadline=now - 86_400_000,
                                   evidence=[Evidence(ref=link, source="#fines · Jack · Sep 30")])
    await store.add_task("read that paper", Provenance.USER_STATED)
    out = json.loads(await mcp_server.tasks(now=now))
    assert [(t["text"], t["lane"]) for t in out["tasks"]] == [("pay the fine", "overdue"), ("read that paper", "someday")]
    assert out["tasks"][0]["id"] == fine and out["open_questions"] == 0
    # the origin rides along as stored, so a surface can show "from #fines · Jack" with the link
    assert out["tasks"][0]["evidence"] == [{"ref": link, "source": "#fines · Jack · Sep 30"}]
    assert out["tasks"][1]["evidence"] == []
    assert [t["text"] for t in json.loads(await mcp_server.tasks("overdue", now=now))["tasks"]] == ["pay the fine"]
    assert [t["text"] for t in json.loads(await mcp_server.tasks("paper", now=now))["tasks"]] == ["read that paper"]


async def test_a_slot_is_not_a_deadline_and_an_ended_slot_reads_as_missed(tmp_path):
    from datetime import datetime, timedelta
    await _setup(tmp_path)
    noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    now = int(noon.timestamp() * 1000)
    friday = int((noon + timedelta(days=3)).replace(hour=9).timestamp() * 1000)
    tid, _ = await store.add_task("draft the deck", Provenance.USER_STATED, deadline=friday)
    await store.set_slot(tid, now - 30 * 60_000, 90)                  # 11:30 for 90 min: in progress
    old, _ = await store.add_task("call the bank", Provenance.USER_STATED)
    await store.set_slot(old, now - 3 * 3_600_000, 60)                # 09:00-10:00: ended, still open
    out = json.loads(await mcp_server.tasks(now=now))
    row = {t["id"]: t for t in out["tasks"]}
    assert row[tid]["lane"] not in ("today", "overdue")               # lanes follow the deadline only
    assert row[tid]["slot"] == {"start": f"{noon:%Y-%m-%d}T11:30", "minutes": 90}
    assert row[tid]["deadline"] == mcp_server._iso(friday) and "missed_slot" not in row[tid]
    assert row[old]["lane"] != "overdue" and "missed_slot" not in row[old]   # a missed slot is never overdue
    assert row[old]["slot"] == {"start": f"{noon:%Y-%m-%d}T09:00", "minutes": 60, "missed": True}
    assert out["today_slots"] == [old, tid]
    assert (await store.task_by_id(old)).slot_minutes == 60           # read-time only: nothing moved


async def test_a_missed_slot_from_today_stays_slotted_and_an_older_one_reads_as_missed_slot(tmp_path):
    from datetime import datetime, timedelta
    await _setup(tmp_path)
    noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    now = int(noon.timestamp() * 1000)
    today, _ = await store.add_task("draft the deck", Provenance.USER_STATED)
    await store.set_slot(today, int(noon.replace(hour=8).timestamp() * 1000), 60)        # today 08:00-09:00
    old, _ = await store.add_task("call the bank", Provenance.USER_STATED)
    await store.set_slot(old, int((noon - timedelta(days=1)).replace(hour=8).timestamp() * 1000), 60)
    out = json.loads(await mcp_server.tasks(now=now))
    row = {t["id"]: t for t in out["tasks"]}
    assert row[today]["slot"] == {"start": f"{noon:%Y-%m-%d}T08:00", "minutes": 60, "missed": True}
    assert "missed_slot" not in row[today] and today in out["today_slots"]
    assert "slot" not in row[old] and old not in out["today_slots"]
    assert row[old]["missed_slot"] == f"{noon - timedelta(days=1):%Y-%m-%d}T08:00"


async def test_tasks_keeps_todays_done_slots_for_the_day_page(tmp_path):
    from datetime import datetime, timedelta
    await _setup(tmp_path)
    noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    now = int(noon.timestamp() * 1000)
    done, _ = await store.add_task("draft the deck", Provenance.USER_STATED)
    await store.set_slot(done, now - 3 * 3_600_000, 60)                       # 09:00, then done
    await store.complete_task(done)
    old, _ = await store.add_task("call the bank", Provenance.USER_STATED)
    await store.set_slot(old, int((noon - timedelta(days=1)).timestamp() * 1000), 30)
    await store.complete_task(old)                                            # slotted yesterday: not today's
    out = json.loads(await mcp_server.tasks(now=now))
    assert [r["id"] for r in out["done_slots"]] == [done] and done not in [t["id"] for t in out["tasks"]]
    assert out["done_slots"][0]["slot"] == {"start": f"{noon:%Y-%m-%d}T09:00", "minutes": 60}
    assert out["done_slots"][0]["status"] == "done"


async def test_get_context_relates_by_name_and_evidence_and_computes_gaps(tmp_path):
    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    now = now_ms()
    tid, _ = await store.add_task("decide on coffee beans to buy while Mei is in Spain", Provenance.USER_STATED,
                                  deadline=now + 86_400_000, evidence=[Evidence(ref="m1", source="user")])
    await store.upsert_structured(StructuredEntry(id="people.mei", kind="people.mei", subject="Mei",
                                                  content="my partner", evidence=[Evidence(ref="m0", source="user")],
                                                  provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    await store.upsert_structured(StructuredEntry(id="preferences.coffee beans", kind="preferences.coffee beans", subject="coffee beans",
                                                  content="I like Kenyan AA", evidence=[Evidence(ref="m1", source="user")],
                                                  provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    await store.upsert_structured(StructuredEntry(id="people.ruth", kind="people.ruth", subject="Ruth",
                                                  content="PM on Crumbly", evidence=[Evidence(ref="s9", source="slack")],
                                                  provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    await store.upsert_structured(StructuredEntry(id="projects.valencia_trip", kind="projects.valencia_trip",
                                                  subject="Valencia trip", content="her week away",
                                                  evidence=[Evidence(ref="m1", source="user")],   # same evidence as the task
                                                  provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    from ken.agents.base import Proposal
    from ken.consent import gate
    await gate.apply(Proposal(intent="event", content="Mei flies to Spain on the 12th",
                              at=now, evidence=[Evidence(ref="m2", source="user")]),
                     conversation_id=bus.MAIN, author="kip")
    await gate.apply(Proposal(intent="event", content="found 3 shops in Valencia", at=now,
                              evidence=[Evidence(ref=tid, source="report")]),
                     conversation_id=bus.MAIN, author="kip")
    await store.upsert_structured(StructuredEntry(id="people.al", kind="people.al", subject="Al",
                                                  content="my barber", evidence=[Evidence(ref="b1", source="user")],
                                                  provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    out = json.loads(await mcp_server.get_context(tid, now=now))
    assert out["task"]["id"] == tid
    rel = {r["name"]: r["why"] for r in out["related"]}
    assert rel == {"mei": "named in the task", "coffee beans": "named in the task",
                   "valencia trip": "shares evidence with the task"}                  # Ruth, Al are not here
    assert {r["kind"] for r in out["related"]} == {"person", "preference", "project"}
    assert [e["what"] for e in out["events"]] == ["Mei flies to Spain on the 12th"]
    assert out["last_report"]["what"] == "found 3 shops in Valencia"
    assert out["questions_for_agent"] == [] or "Spain" in out["questions_for_agent"][0]   # Spain: not a known entity
    assert out["questions_for_user"] == []
    assert "error" in json.loads(await mcp_server.get_context("nope"))


async def test_get_context_respects_its_budget_oldest_events_first(tmp_path):
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence
    await _setup(tmp_path)
    now = now_ms()
    tid, _ = await store.add_task("write the quarterly report", Provenance.USER_STATED)
    for i in range(60):
        await gate.apply(Proposal(intent="event", content=f"report note {i} " + "x" * 200, at=now - i * 60_000,
                                  evidence=[Evidence(ref=f"e{i}", source="user")]),
                         conversation_id=bus.MAIN, author="kip")
    out = json.loads(await mcp_server.get_context(tid, now=now, budget_chars=3000))
    assert len(json.dumps(out)) <= 3300 and out["events"][0]["what"].startswith("report note 0")


async def test_who_is_and_find(tmp_path):
    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    now = now_ms()
    await store.upsert_structured(StructuredEntry(id="people.ruth_van_dalen", kind="people.ruth_van_dalen",
                                                  subject="Ruth van Dalen", content="PM on Crumbly, owns the toolbar question",
                                                  evidence=[Evidence(ref="s9", source="slack")],
                                                  provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    known = json.loads(await mcp_server.who_is("Ruth van Dalen"))
    assert known["known"] and known["kind"] == "person" and "toolbar" in known["note"]
    unknown = json.loads(await mcp_server.who_is("Zed"))
    assert unknown["known"] is False and unknown["find_out"] and unknown["then"].startswith("tell(")
    await store.add_task("ask Ruth about the toolbar", Provenance.USER_STATED)
    hits = json.loads(await mcp_server.find("toolbar"))
    assert [t["text"] for t in hits["tasks"]] == ["ask Ruth about the toolbar"]
    assert hits["beliefs"][0]["name"] == "ruth van dalen"


async def test_pending_and_decide_round_trip_through_the_engine(tmp_path):
    from ken.agents.assistant import Assistant
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence
    from ken.engine import Engine
    await _setup(tmp_path)
    await gate.apply(Proposal(intent="structured:people.jack", content="Jack is my manager",
                              evidence=[Evidence(ref="m1", source="user")]),
                     conversation_id=bus.MAIN, author="kip", action_id="act-1")
    (card,) = await mcp_server.pending_cards()
    assert card["kind"] == "people.jack" and card["action_id"] == "act-1" and card["type"] == "belief"
    assert card["evidence"] == [{"ref": "m1", "source": "user"}]          # a card shows where it came from
    assert (card["title"], card["subject"]) == ("Save person", "Jack")    # never an anonymous description
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)

    async def pump():
        for _ in range(40):
            await eng.pump_inbox(); await asyncio.sleep(0.05)
    t = asyncio.create_task(pump())
    out = json.loads(await mcp_server.decide(card["card_id"], True, edited="Jack is my manager; owns hiring", timeout=5))
    await t
    assert out["outcome"] == "applied" and out["edited"] is True
    assert await mcp_server.pending_cards() == []
    assert json.loads(await mcp_server.who_is("Jack"))["note"] == "Jack is my manager; owns hiring"
    assert json.loads(await mcp_server.decide(card["card_id"], True))["outcome"] == "no_such_card"
    # the tap is anchored to the card, so kip's history sees the decision
    taps = [a for a in await bus.take_unhandled_actions() if a.token]
    assert taps == [] or taps[0].anchor == card["card_id"]


async def test_decide_reports_the_engines_outcome_and_refuses_agent_tokens(tmp_path):
    from ken.agents.assistant import Assistant
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence
    from ken.engine import Engine
    await _setup(tmp_path)
    await gate.apply(Proposal(intent="structured:people.jack", content="Jack is my manager",
                              evidence=[Evidence(ref="m1", source="user")]),
                     conversation_id=bus.MAIN, author="kip", action_id="act-1")
    (card,) = await mcp_server.pending_cards()
    tok = mcp_server._CALLER.set("mcp:agent:codex")
    try:
        assert json.loads(await mcp_server.decide(card["card_id"], True))["outcome"] == "forbidden"
    finally:
        mcp_server._CALLER.reset(tok)
    assert await mcp_server.pending_cards() != []                                    # untouched
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)

    async def pump():
        for _ in range(40):
            await eng.pump_inbox(); await asyncio.sleep(0.05)
    t = asyncio.create_task(pump())
    out = json.loads(await mcp_server.decide(card["card_id"], False, timeout=5))
    await t
    assert out["outcome"] == "rejected"                                              # the gate's word
    assert await mcp_server.pending_cards() == []
    assert json.loads(await mcp_server.who_is("Jack"))["known"] is False


async def test_hydrate_plan_then_one_card_per_batch_then_per_item_decide(tmp_path):
    from ken.agents.assistant import Assistant
    from ken.engine import Engine
    await _setup(tmp_path)
    fresh = json.loads(await mcp_server.hydrate())
    assert fresh["state"] == "fresh" and fresh["have"] == {"people": 0, "projects": 0, "preferences": 0, "facts": 0, "assets": 0}
    assert len(fresh["plan"]) == 5 and fresh["open_batches"] == [] and fresh["batch_size"] == 10
    assert json.loads(await mcp_server.hydrate("not json"))["outcome"] == "invalid"
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)

    async def pump():
        for _ in range(60):
            await eng.pump_inbox(); await asyncio.sleep(0.05)
    t = asyncio.create_task(pump())
    items = [{"kind": "person", "name": "Ruth", "note": "PM on Crumbly"},
             {"kind": "person", "name": "Bob", "note": "a recruiter who cold-emailed"},
             {"kind": "project", "name": "Crumbly", "note": "the bakery app; toolbar redesign in flight"}]
    out = json.loads(await mcp_server.hydrate(json.dumps(items), source="slack"))
    await t
    assert out["outcome"] == "pending" and len(out["cards"]) == 1
    (card,) = await mcp_server.pending_cards()
    assert card["type"] == "batch" and [i["name"] for i in card["items"]] == ["Ruth", "Bob", "Crumbly"]
    assert card["card_id"] == out["cards"][0] and "3 new on one card" in out["result"][0]["text"]
    assert json.loads(await mcp_server.hydrate())["open_batches"] == [card["card_id"]]
    # a bad edit keeps the card open and says so
    t = asyncio.create_task(pump())
    bad = json.loads(await mcp_server.decide(card["card_id"], True, edited="{oops", timeout=5))
    await t
    assert bad["outcome"] == "edit_invalid" and len(await mcp_server.pending_cards()) == 1
    # keep Crumbly as proposed, edit Ruth, drop Bob
    t = asyncio.create_task(pump())
    out = json.loads(await mcp_server.decide(card["card_id"], True, timeout=5,
                                             edited=json.dumps([{"name": "Ruth", "note": "PM on Crumbly; owns the toolbar"},
                                                                {"name": "Bob", "drop": True}])))
    await t
    assert out["outcome"] == "applied" and await mcp_server.pending_cards() == []
    ruth = json.loads(await mcp_server.who_is("Ruth"))
    assert ruth["known"] and ruth["note"] == "PM on Crumbly; owns the toolbar" and ruth["evidence"] == []
    reader = json.loads(await mcp_server.who_is("Crumbly"))
    assert reader["known"] and reader["kind"] == "project" and reader["evidence"][0]["source"] == "slack"
    bob = json.loads(await mcp_server.who_is("Bob"))
    assert bob["known"] is False and bob["similar"] == []
    assert json.loads(await mcp_server.who_is("Ruth Jones"))["similar"] == ["Ruth"]   # a near match
    assert json.loads(await mcp_server.hydrate())["state"] == "partial"
    assert json.loads(await mcp_server.hydrate())["have"]["people"] == 1


async def test_pending_digest_is_short_plain_text_or_empty(tmp_path):
    from ken.agents.base import Proposal
    from ken.consent import gate
    await _setup(tmp_path)
    assert await mcp_server.pending_digest() == ""
    await gate.apply(Proposal(intent="structured:people.jack", content="Jack is my manager"),
                     conversation_id=bus.MAIN, author="kip")
    await gate.apply(Proposal(intent="batch:hydrate", content="Save 1 from slack?",
                              meta={"items": [{"kind": "people", "name": "Ruth", "note": "PM"}]}),
                     conversation_id=bus.MAIN, author="kip")
    text = await mcp_server.pending_digest()
    lines = text.splitlines()
    assert lines[0] == "Ken · 2 open questions"
    assert lines[1] == "1. [Save person] Jack is my manager" and lines[2].startswith("   id ")
    assert "2. [Save these?] Save 1 from slack?" in lines and "   - Ruth (person): PM" in lines
    assert lines[-1].startswith("Answer in any Ken-connected agent")
    # a judge-conflict card names the value a Yes replaces
    from ken.consent import judge
    from ken.contracts import Evidence, StructuredEntry
    await store.upsert_structured(StructuredEntry(id="people.ann", kind="people.ann", subject="Ann", content="a peer",
                                                  evidence=[Evidence(ref="m0", source="user")],
                                                  provenance=Provenance.USER_STATED, updated_at=now_ms()), change_note="t")

    class _Conflict:
        async def __call__(self, proposal, *, message, existing):
            return judge.conflict("people.ann", reason="differs")
    await gate.apply(Proposal(intent="structured:people.ann", content="Ann is my manager"),
                     conversation_id=bus.MAIN, author="kip", judge=_Conflict())
    card = next(c for c in await mcp_server.pending_cards() if c["kind"] == "people.ann")
    assert card["existing"] == "a peer"
    text = await mcp_server.pending_digest()
    assert "[Update person] Ann is my manager\n   now: a peer" in text
    # a backlog is capped, never an attachment-sized body
    for i in range(12):
        await gate.apply(Proposal(intent=f"structured:people.p{i}", content=f"person {i}"),
                         conversation_id=bus.MAIN, author="kip")
    text = await mcp_server.pending_digest()
    assert text.startswith("Ken · 15 open questions") and "… and 7 more: ken cards" in text
    assert len(text.encode()) < 4096


async def test_last_report_is_the_newest_logged_even_within_one_second_and_the_user_is_never_a_gap(tmp_path):
    from ken import config
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence
    await _setup(tmp_path)
    config.set_section_value("user", "name", "Alex")
    now = now_ms()
    tid, _ = await store.add_task("decide on coffee beans while Alex is in Spain", Provenance.USER_STATED)
    for what in ("found 3 shops", "bought a box, done"):
        await gate.apply(Proposal(intent="event", content=what, at=now, evidence=[Evidence(ref=tid, source="report")]),
                         conversation_id=bus.MAIN, author="kip")
    out = json.loads(await mcp_server.get_context(tid, now=now))
    assert out["last_report"]["what"] == "bought a box, done"
    assert not any("'Alex'" in q for q in out["questions_for_agent"])
    assert any("'Spain'" in q for q in out["questions_for_agent"])        # a place: still one wrong question at worst


async def test_decide_no_on_an_identity_card_returns_the_fresh_cards_id(tmp_path):
    from ken.agents.assistant import Assistant
    from ken.agents.base import Proposal
    from ken.consent import gate, judge
    from ken.contracts import Evidence, StructuredEntry
    from ken.engine import Engine
    await _setup(tmp_path)
    await store.upsert_structured(StructuredEntry(id="people.lucas", kind="people.lucas", subject="Lucas", content="Tallyhouse contact",
                                                  evidence=[Evidence(ref="m0", source="user")],
                                                  provenance=Provenance.USER_STATED, updated_at=1), change_note="t")

    class _Collapse:
        async def __call__(self, proposal, *, message, existing):
            return judge.collapse("people.lucas")
    await gate.apply(Proposal(intent="structured:people.lucas_tallyhouse", content="vendor"),
                     conversation_id=bus.MAIN, author="kip", judge=_Collapse())
    (card,) = await mcp_server.pending_cards()
    assert card["existing"] == "Tallyhouse contact"
    eng = Engine(); eng.register_agent(Assistant(llm=None), assistant=True)

    async def pump():
        for _ in range(40):
            await eng.pump_inbox(); await asyncio.sleep(0.05)
    t = asyncio.create_task(pump())
    out = json.loads(await mcp_server.decide(card["card_id"], False, timeout=5))
    await t
    assert out["outcome"] == "reasked" and "call pending()" in out["text"]
    (fresh,) = await mcp_server.pending_cards()
    assert out["cards"] == [fresh["card_id"]] and fresh["kind"] == "people.lucas_tallyhouse"


async def test_get_context_follows_exactly_one_hop_of_links_and_shows_urls(tmp_path):
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    now = now_ms()

    async def rec(kind, subject, content, url=None):
        await store.upsert_structured(StructuredEntry(id=kind, kind=kind, subject=subject, content=content, url=url,
                                                      evidence=[Evidence(ref=kind, source="user")],
                                                      provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    await rec("projects.recipes", "Recipes", "recipes in-app; Q4 P0")
    await rec("people.ruth", "Ruth", "the only Crumbly designer")
    await rec("assets.design_file", "Design file", "the Figma file", url="https://figma.com/file/abc")
    await rec("people.priya", "Priya", "eng lead")
    await store.set_links("projects.recipes", [{"dst": "people.ruth", "rel": "member"}, {"dst": "people.priya", "rel": "owner"}])
    await store.set_links("assets.design_file", [{"dst": "projects.recipes", "rel": "part_of"}])
    # the task names the project: one hop brings its people and its asset, with the url
    tid, _ = await store.add_task("prepare the Recipes demo for Friday", Provenance.USER_STATED)
    out = json.loads(await mcp_server.get_context(tid, now=now))
    rel = {r["name"]: r for r in out["related"]}                      # names are the de-slugged ids, lowercase
    assert rel["recipes"]["why"] == "named in the task"
    assert rel["ruth"]["why"] == "linked: ruth is a member of recipes" and rel["priya"]["why"] == "linked: priya is the owner of recipes"
    assert rel["design file"]["why"] == "linked: design file is part of recipes" and rel["design file"]["url"] == "https://figma.com/file/abc"
    # the task names a person: the project comes (one hop), the project's asset does NOT (two hops)
    tid2, _ = await store.add_task("ask Ruth about the toolbar", Provenance.USER_STATED)
    out = json.loads(await mcp_server.get_context(tid2, now=now))
    names = {r["name"]: r["why"] for r in out["related"]}
    assert names == {"ruth": "named in the task", "recipes": "linked: ruth is a member of recipes"}
    # who_is shows both directions
    w = json.loads(await mcp_server.who_is("Recipes"))
    assert {(lk["rel"], lk["name"]) for lk in w["links"]} == {("member", "ruth"), ("owner", "priya"), ("part_of", "design file")}
    assert {lk["text"] for lk in w["links"]} == {"ruth is a member of recipes", "priya is the owner of recipes", "design file is part of recipes"}
    f = json.loads(await mcp_server.who_is("Design file"))
    assert f["known"] and f["url"] == "https://figma.com/file/abc" and f["kind"] == "asset"
    # a card carries its links and url for the client, and the digest prints them
    await gate.apply(Proposal(intent="structured:assets.ledger", content="the finance sheet",
                              meta={"url": "https://sheets/1", "links": [{"dst": "projects.recipes", "rel": "part_of", "name": "Recipes"}]}),
                     conversation_id=bus.MAIN, author="kip")
    (card,) = await mcp_server.pending_cards()
    assert card["url"] == "https://sheets/1" and card["links"] == [{"rel": "part_of", "name": "Recipes"}]
    text = mcp_server.render_card(card)
    assert "   url: https://sheets/1" in text and "   link: part_of \u2192 Recipes" in text


async def test_get_context_ranks_by_strength_ignores_family_words_caps_notes_and_trims_the_weakest(tmp_path):
    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    now = now_ms()

    async def rec(kind, subject, content, at=now):
        await store.upsert_structured(StructuredEntry(id=kind, kind=kind, subject=subject, content=content,
                                                      evidence=[Evidence(ref=kind, source="user")],
                                                      provenance=Provenance.USER_STATED, updated_at=at), change_note="t")
    long = "x" * 900
    await rec("projects.crumbly_project_recipes_in_app", "Recipes", "recipes in-app " + long, at=1)      # oldest
    await rec("projects.crumbly_pre_order_browser", "Pre-order", "pre order " + long, at=2)
    await rec("projects.crumbly_tallyhouse_attribution", "Tallyhouse", "attribution " + long, at=3)
    await rec("people.ruth_van_dalen", "Ruth van Dalen", "designer " + long, at=4)
    await rec("assets.insights_recipes_dashboard", "Insights Recipes", "dashboard " + long, at=now)           # newest
    await store.set_links("projects.crumbly_project_recipes_in_app", [{"dst": "people.ruth_van_dalen", "rel": "member"}])
    tid, _ = await store.add_task("Crumbly: improve recipes-page exposure with Ruth van Dalen", Provenance.USER_STATED)
    out = json.loads(await mcp_server.get_context(tid, now=now))
    names = [r["name"] for r in out["related"]]
    # 'crumbly' is a family word (3 titles): Tallyhouse and Pre-order are NOT named by it alone
    assert "crumbly tallyhouse attribution" not in names and "crumbly pre order browser" not in names
    # strongest first: the person named in full, then the two records named by 'recipes' (a tie)
    assert names[0] == "ruth van dalen" and set(names[1:3]) == {"crumbly project recipes in app", "insights recipes dashboard"}
    # notes are capped, with the way to the full note
    assert all(len(r["note"]) <= 405 for r in out["related"]) and out["related"][0]["more"].startswith("who_is(")
    # a tight budget trims the WEAKEST, keeps the task's own person and project, and says it trimmed
    out = json.loads(await mcp_server.get_context(tid, now=now, budget_chars=1500))
    kept = [r["name"] for r in out["related"]]
    assert kept[0] == "ruth van dalen" and "crumbly project recipes in app" in kept and out.get("truncated") is True
    assert "crumbly" not in " ".join(out["questions_for_agent"]).lower()


async def test_a_task_named_only_by_a_family_word_still_gets_the_family_and_more_points_at_find_for_a_preference(tmp_path):
    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    now = now_ms()

    async def rec(kind, subject, content):
        await store.upsert_structured(StructuredEntry(id=kind, kind=kind, subject=subject, content=content,
                                                      evidence=[Evidence(ref=kind, source="user")],
                                                      provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    for k in ("crumbly_recipes", "crumbly_pre_order", "crumbly_tallyhouse"):
        await rec(f"projects.{k}", k, "a project")
    await rec("preferences.reporting", "you", "reports: " + "y" * 900)
    tid, _ = await store.add_task("Write the Crumbly Q4 update; keep the reporting style", Provenance.USER_STATED)
    out = json.loads(await mcp_server.get_context(tid, now=now))
    rel = {r["name"]: r for r in out["related"]}
    assert {n for n in rel if n.startswith("crumbly")} == {"crumbly recipes", "crumbly pre order", "crumbly tallyhouse"}
    assert all("family word 'crumbly'" in rel[n]["why"] for n in rel if n.startswith("crumbly"))
    assert rel["reporting"]["more"].startswith("find('reporting')")
    assert out["questions_for_agent"] == []


async def test_world_lists_what_ken_knows_with_links_and_the_tasks_that_name_each(tmp_path):
    from datetime import datetime

    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    now = int(datetime.now().replace(hour=12, minute=0, second=0, microsecond=0).timestamp() * 1000)
    midnight = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)  # not now-12h: DST days

    async def rec(kind, subject, content, url=None):
        await store.upsert_structured(StructuredEntry(id=kind, kind=kind, subject=subject, content=content, url=url,
                                                      evidence=[Evidence(ref=kind, source="user")],
                                                      provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    await rec("projects.recipes_in_app", "Recipes", "recipes in-app; Q4 P0")
    await rec("projects.ken", "Ken", "x" * 300)
    await rec("projects.tallyhouse_attribution_for_ios", "Tallyhouse", "attribution vendor")
    await rec("projects.node_js", "Node.js", "the runtime upgrade")
    await rec("people.ruth_van_dalen", "Ruth", "the only Crumbly designer")
    await rec("people.ann", "Ann", "a peer")
    await rec("assets.design_file", "Design file", "the Figma file", url="https://figma.com/file/abc")
    await rec("preferences.coffee beans", "coffee beans", "Cuban")
    await store.set_links("projects.recipes_in_app", [{"dst": "people.ruth_van_dalen", "rel": "member"}])
    await store.set_links("assets.design_file", [{"dst": "projects.recipes_in_app", "rel": "part_of"}])
    await store.set_links("people.ann", [{"dst": "assets.design_file", "rel": "owner"}])
    late, _ = await store.add_task("ask Ruth about the recipes in app toolbar", Provenance.USER_STATED, deadline=now - 3_600_000)
    await store.add_task("ship the app build", Provenance.USER_STATED)                 # 'app' alone names no project
    await store.add_task("write the recipes app changelog", Provenance.USER_STATED,     # two title words name it; date-only
                         deadline=midnight)                                       # today: due all day, not late at noon
    await store.add_task("fix the iOS crash for Ann", Provenance.USER_STATED)           # 'for' is glue: one word, no project
    await store.add_task("rent a van for the move", Provenance.USER_STATED)             # a lowercase 'van' is not a name
    await store.add_task("upgrade Node.js to 22", Provenance.USER_STATED)               # punctuation in a name still matches
    done, _ = await store.add_task("thank Ruth", Provenance.USER_STATED)
    await store.complete_task(done)                                                   # a done task is not counted

    w = json.loads(await mcp_server.world(now=now))
    assert w["counts"] == {"people": 2, "projects": 4, "assets": 1, "facts": 0, "preferences": 1} and w["truncated"] is False
    assert w["preferences"] == [{"name": "coffee beans", "note": "Cuban"}]                # the rules themselves, not a count
    assert json.loads(await mcp_server.world(now=now, budget_chars=10))["preferences"] == w["preferences"]  # never cut
    by = {p["name"]: p for p in w["projects"]}
    assert [p["name"] for p in w["projects"]][:2] == ["recipes in app", "node js"]      # late first, then open, then name
    recipes = by["recipes in app"]
    assert (recipes["open"], recipes["late"]) == (2, 1)
    assert recipes["tasks"][0] == {"id": late, "text": "ask Ruth about the recipes in app toolbar",
                                  "deadline": mcp_server._iso(now - 3_600_000), "late": True,
                                  "project": None, "owner": None, "evidence": []}   # the one task-row shape
    assert recipes["tasks"][1]["late"] is False                                        # the date-only task
    assert recipes["people"] == [{"name": "ruth van dalen", "rel": "member", "text": "ruth van dalen is a member of recipes in app"}]
    assert recipes["assets"] == [{"name": "design file", "rel": "part_of", "text": "design file is part of recipes in app",
                                 "url": "https://figma.com/file/abc"}] and recipes["projects"] == []
    assert by["tallyhouse attribution for ios"]["open"] == 0 and by["node js"]["open"] == 1
    ken = by["ken"]
    assert (ken["open"], ken["late"], ken["people"]) == (0, 0, []) and ken["note"].endswith(" …") and len(ken["note"]) < 250
    people = {p["name"]: p for p in w["people"]}
    ruth, ann = people["ruth van dalen"], people["ann"]
    assert (ruth["open"], ruth["late"]) == (1, 1)                                 # 'Ruth' as a name: yes; 'van': no
    assert [lk["name"] for lk in ruth["projects"]] == ["recipes in app"]
    assert ann["open"] == 1 and [lk["text"] for lk in ann["assets"]] == ["ann is the owner of design file"]
    (asset,) = w["assets"]
    assert (asset["name"], asset["url"]) == ("design file", "https://figma.com/file/abc")
    assert [lk["name"] for lk in asset["projects"]] == ["recipes in app"] and [lk["name"] for lk in asset["people"]] == ["ann"]
    assert json.loads(await mcp_server.who_is(recipes["name"]))["known"]                # a name round-trips to who_is
    # over budget: quiet rows lose their notes, then every row its task texts; the counts stay
    small = json.loads(await mcp_server.world(now=now, budget_chars=600))
    assert small["truncated"] is True and "note" not in {k for p in small["projects"] if not p["open"] for k in p}
    assert "tasks" not in small["projects"][0] and small["projects"][0]["open"] == 2


async def test_a_task_carries_its_project_from_capture_to_world(tmp_path):
    """The project on a task is a fact: set at capture by name, shown on the card, kept by the
    gate only while the project exists, counted by world() before any name match."""
    from ken.agents.assistant import Assistant, _parse_when
    from ken.agents.base import Context
    from ken.capabilities import MemoryReader, Proposer
    from ken.captures import Captures
    from ken.consent import gate
    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    now = now_ms()

    async def rec(kind, content):
        await store.upsert_structured(StructuredEntry(id=kind, kind=kind, subject="you", content=content,
                                                      evidence=[Evidence(ref=kind, source="user")],
                                                      provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    await rec("projects.crumbly_project_recipes_in_app", "recipes in-app; Q4 P0")
    await rec("projects.crumbly_pre_order", "the pre-order flow")
    recipes = "projects.crumbly_project_recipes_in_app"

    def cap(provenance):
        a = Assistant(llm=None)
        ctx = Context(conversation_id=bus.MAIN, now=now, message="")
        ctx.memory = MemoryReader(a.slice); ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
        return Captures(ctx, parse_when=_parse_when, provenance=provenance)

    # the user's own task: one distinctive word names the project; the text alone would not
    r = await cap(Provenance.USER_INSTRUCTED).add_task("hide retired default ones", project="Recipes")
    assert r.outcome == "task" and r.text.endswith("project: crumbly project recipes in app")
    (t,) = await store.open_tasks()
    assert t.project == recipes
    # a name Ken does not hold, or one that fits two projects: the task is saved on no project, and it says so
    r = await cap(Provenance.USER_INSTRUCTED).add_task("buy a lamp", project="Garden")
    assert r.outcome == "task" and "no project 'Garden'" in r.text
    r = await cap(Provenance.USER_INSTRUCTED).add_task("write the recap", project="crumbly")
    assert "more than one project" in r.text
    assert {x.text: x.project for x in await store.open_tasks()} == {
        "hide retired default ones": recipes, "buy a lamp": None, "write the recap": None}
    # the same task again with a DIFFERENT project: it keeps the one it has
    r = await cap(Provenance.USER_INSTRUCTED).add_task("hide retired default ones", project="pre order")
    assert r.outcome == "task_exists" and r.text.endswith("it keeps the project it already has")
    # an agent's task is a card; the card names the project, and a Yes writes it
    r = await cap(Provenance.WORLD).add_task("trace the picker load time", project="recipes")
    assert r.outcome == "pending"
    (card,) = await mcp_server.pending_cards()
    assert card["note"] == "project: crumbly project recipes in app"
    assert await gate.resolve(card["card_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    by = {x.text: x for x in await store.open_tasks()}
    assert by["trace the picker load time"].project == recipes

    # tasks() shows it; get_context leads with it; world() counts by it, not by the text
    rows = {x["text"]: x["project"] for x in json.loads(await mcp_server.tasks(now=now))["tasks"]}
    assert rows["hide retired default ones"] == "crumbly project recipes in app" and rows["buy a lamp"] is None
    ctxt = json.loads(await mcp_server.get_context(t.id, now=now))
    assert (ctxt["related"][0]["name"], ctxt["related"][0]["why"]) == ("crumbly project recipes in app", "the task's project")
    w = {p["name"]: p for p in json.loads(await mcp_server.world(now=now))["projects"]}
    assert w["crumbly project recipes in app"]["open"] == 2 and w["crumbly pre order"]["open"] == 0
    # a deliberate move is the one path that changes it
    assert await store.set_task_project(t.id, "projects.crumbly_pre_order")
    w = {p["name"]: p["open"] for p in json.loads(await mcp_server.world(now=now))["projects"]}
    assert w == {"crumbly project recipes in app": 1, "crumbly pre order": 1}
    # a forgotten project takes its label off its tasks; the tasks stay
    await store.delete_structured(recipes)
    assert {x.text: x.project for x in await store.open_tasks()}["trace the picker load time"] is None
    # a card that waited while its project was forgotten lands on no project
    await rec("projects.temp", "short-lived")
    await cap(Provenance.WORLD).add_task("clean the temp dir", project="temp")
    (card,) = await mcp_server.pending_cards()
    await store.delete_structured("projects.temp")
    assert await gate.resolve(card["card_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    assert {x.text: x.project for x in await store.open_tasks()}["clean the temp dir"] is None


async def test_a_project_name_must_be_distinctive_and_an_open_card_adopts_one(tmp_path):
    from ken.agents.assistant import Assistant, _parse_when
    from ken.agents.base import Context
    from ken.capabilities import MemoryReader, Proposer
    from ken.captures import Captures
    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    now = now_ms()
    for kind in ("projects.crumbly_project_recipes_in_app", "projects.crumbly_pre_order", "projects.garden"):
        await store.upsert_structured(StructuredEntry(id=kind, kind=kind, subject="you", content="x",
                                                      evidence=[Evidence(ref=kind, source="user")],
                                                      provenance=Provenance.USER_STATED, updated_at=now), change_note="t")
    recipes, preorder = "projects.crumbly_project_recipes_in_app", "projects.crumbly_pre_order"

    def cap(provenance):
        a = Assistant(llm=None)
        ctx = Context(conversation_id=bus.MAIN, now=now, message="")
        ctx.memory = MemoryReader(a.slice); ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
        return Captures(ctx, parse_when=_parse_when, provenance=provenance)

    # a generic word names nothing; a long name does not fall onto a short title inside it
    r = await cap(Provenance.USER_INSTRUCTED).add_task("write the recap", project="the project")
    assert r.outcome == "task" and "Ken has no project 'the project', so none was set" in r.text
    r = await cap(Provenance.USER_INSTRUCTED).add_task("plan the beds", project="Recipes redesign for the garden")
    assert "so none was set" in r.text
    assert {t.project for t in await store.open_tasks()} == {None}
    # an agent proposes a task with no project, then again WITH one: the open card adopts it
    r = await cap(Provenance.WORLD).add_task("trace the picker load time")
    assert r.outcome == "pending" and "project" not in r.text
    r = await cap(Provenance.WORLD).add_task("trace the picker load time", project="recipes")
    assert r.outcome == "pending" and r.text.endswith("project: crumbly project recipes in app")
    (card,) = await mcp_server.pending_cards()
    assert card["note"] == "project: crumbly project recipes in app"
    # a third time with a DIFFERENT project: the card keeps the one it has, and the reply says so
    r = await cap(Provenance.WORLD).add_task("trace the picker load time", project="pre order")
    assert r.text.endswith("it keeps the project it already has")
    assert (await mcp_server.pending_cards())[0]["note"] == "project: crumbly project recipes in app"
    # tasks(filter) finds a task by its project; a move to a project Ken does not hold is refused
    await cap(Provenance.USER_INSTRUCTED).add_task("hide retired default ones", project="recipes")
    rows = json.loads(await mcp_server.tasks("recipes in app", now=now))["tasks"]
    assert [x["text"] for x in rows] == ["hide retired default ones"]
    t = next(x for x in await store.open_tasks() if x.text == "hide retired default ones")
    assert not await store.set_task_project(t.id, "projects.typo") and await store.set_task_project(t.id, preorder)
    assert (await store.task_by_id(t.id)).project == preorder and recipes != preorder



def test_evidence_dicts_keeps_the_origin_and_the_newest_and_loads_tolerantly():
    from ken.contracts import Evidence
    ev = [Evidence(ref=f"m{i}", source="s") for i in range(5)]
    assert [e["ref"] for e in store.evidence_dicts(ev, 3)] == ["m0", "m3", "m4"]   # origin + newest two
    assert [e["ref"] for e in store.evidence_dicts(ev, 1)] == ["m4"]
    assert len(store.evidence_dicts(ev)) == 5
    # a stored row with an extra key, or a stray non-dict, must not take the readers down
    loaded = store.evidence_load('[{"ref": "m1", "source": "slack", "at": 1}, "junk", {"ref": "m2"}]')
    assert [(e.ref, e.source) for e in loaded] == [("m1", "slack"), ("m2", "")]


async def test_discard_closes_a_task_as_dropped_not_done(tmp_path):
    from ken.agents.base import Proposal
    from ken.consent import gate
    await _setup(tmp_path)
    now = now_ms()
    tid, _ = await store.add_task("buy the briefcase", Provenance.USER_STATED)
    kept, _ = await store.add_task("call mom", Provenance.USER_STATED)
    out = await gate.apply(Proposal(intent="discard", content=tid, provenance=Provenance.USER_INSTRUCTED,
                                    meta={"reason": "no longer needed"}), conversation_id=bus.MAIN, author="kip")
    assert out == "discarded"
    t = await store.task_by_id(tid)
    assert (t.status, t.drop_reason) == ("dropped", "no longer needed") and t.completed_at
    assert [x.id for x in await store.open_tasks()] == [kept]                      # it left the open list
    assert await store.done_since(now - 1000) == []                                # and is NOT counted as done
    (row,) = json.loads(await mcp_server.find("briefcase"))["tasks"]
    assert (row["status"], row["drop_reason"]) == ("dropped", "no longer needed")  # history says why
    # a second discard, or a discard of a missing task, changes nothing
    assert await gate.apply(Proposal(intent="discard", content=tid, provenance=Provenance.USER_INSTRUCTED),
                            conversation_id=bus.MAIN, author="kip") == "task_absent"


async def test_a_discard_from_an_agent_asks_first(tmp_path):
    from ken.agents.base import Proposal
    from ken.consent import gate
    await _setup(tmp_path)
    tid, _ = await store.add_task("investigate the Larder bug", Provenance.USER_STATED)
    out = await gate.apply(Proposal(intent="discard", content=tid, provenance=Provenance.WORLD,
                                    meta={"reason": "fixed upstream"}), conversation_id=bus.MAIN, author="kip")
    assert out == "pending"
    (card,) = await mcp_server.pending_cards()
    assert card["title"] == "Discard?" and "fixed upstream" in card["note"]
    assert (await store.task_by_id(tid)).status == "open"                          # nothing changed before the Yes


async def test_a_discard_report_supersedes_an_open_done_card_and_never_applies_the_other_verb(tmp_path):
    from ken.agents.base import Proposal
    from ken.consent import gate
    await _setup(tmp_path)
    tid, _ = await store.add_task("investigate the Larder bug", Provenance.USER_STATED)
    for intent in ("complete", "discard"):
        assert await gate.apply(Proposal(intent=intent, content=tid, provenance=Provenance.WORLD,
                                         meta={"reason": "line one\nline two " + "x" * 3000}),
                                conversation_id=bus.MAIN, author="kip") == "pending"
    (card,) = await mcp_server.pending_cards()                                      # one card, the newer verb
    assert card["title"] == "Discard?" and "\n" not in card["note"] and len(card["note"]) < 400
    assert "Yes drops it" in card["note"]                                           # the instruction survives a long reason
    ctx = json.loads(await mcp_server.get_context(tid))
    assert ctx["task"]["status"] == "open"
    await store.discard_task(tid, "dup")
    ctx = json.loads(await mcp_server.get_context(tid))                             # a dropped task stays reachable
    assert (ctx["task"]["status"], ctx["task"]["drop_reason"]) == ("dropped", "dup")


async def test_find_matches_words_not_one_phrase(tmp_path):
    """The pilot failure: an agent asks 'kitchen budget' or 'electrician booked'; no record holds
    that exact phrase, so a substring search found nothing and the agent answered 'unknown'."""
    from ken.contracts import Evidence, StructuredEntry
    await _setup(tmp_path)
    await store.upsert_structured(StructuredEntry(
        id="projects.kitchen_remodel", kind="projects.kitchen_remodel", subject="Kitchen remodel",
        content="Redo the kitchen with a 21,500 EUR budget after electrical work.",
        evidence=[Evidence(ref="m1", source="user")], provenance=Provenance.USER_STATED, updated_at=now_ms()),
        change_note="t")
    await store.add_task("Book an electrician for the kitchen remodel by April 1", Provenance.USER_STATED)
    await store.add_task("Pick an extractor hood", Provenance.USER_STATED)
    out = json.loads(await mcp_server.find("kitchen budget"))
    assert out["beliefs"] and out["beliefs"][0]["name"] == "kitchen remodel"
    out = json.loads(await mcp_server.find("When must the electrician be booked?"))
    assert out["tasks"][0]["text"].startswith("Book an electrician")          # stem: booked -> book
    assert all("hood" not in t["text"] for t in out["tasks"][:1])             # the best match first
    assert json.loads(await mcp_server.find("what is the"))["tasks"] == []   # only common words: nothing


async def test_typed_write_without_an_engine_is_queued_not_failed(tmp_path):
    await _setup(tmp_path)
    out = json.loads(await mcp_server.typed("add_task", {"text": "x"}, timeout=0.3))
    assert out["outcome"] == "queued" and "do not retry" in out["text"]


def test_server_reports_the_package_version():
    assert mcp_server.build_server()._lowlevel_server.version
