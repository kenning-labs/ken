"""Fixes from the v3 pilot: find puts what is true NOW before dated history, tasks(filter=)
does not silently miss, and a NEW record in the user's own words is saved without a card."""
import json
from datetime import datetime, timedelta

from ken import db, mcp_server
from ken.agents.assistant import Assistant, _parse_when
from ken.agents.base import Context
from ken.capabilities import MemoryReader, Proposer
from ken.captures import Captures
from ken.contracts import Provenance
from ken.conversation import bus
from ken.memory import store
from ken.util import now_ms


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


def _cap(prov: Provenance = Provenance.USER_INSTRUCTED) -> Captures:
    ctx = Context(conversation_id=bus.MAIN, now=now_ms(), message="")
    ctx.memory = MemoryReader(Assistant.slice)
    ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
    return Captures(ctx, parse_when=_parse_when, provenance=prov)


def _ago(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


# ── find: current before history ─────────────────────────────────────────────

async def test_find_lists_current_records_before_dated_history(tmp_path):
    await _setup(tmp_path)
    await _cap().log_event("pricing goes up to 10 USD a month", when=_ago(7))
    assert (await _cap().remember("project", "crumbly", "Crumbly launches March 17. Price is 8 USD a month.")).outcome == "applied"
    raw = await mcp_server.find("price month")
    out = json.loads(raw)
    assert raw.index('"beliefs"') < raw.index('"events"')            # what is true now comes first
    assert out["beliefs"] and out["events"]
    assert "history" in out["note"] and "current" in out["note"]       # the reader is told which wins


# ── tasks(filter=): no silent miss ───────────────────────────────────────────

async def test_tasks_filter_matches_the_projects_note(tmp_path):
    await _setup(tmp_path)
    await _cap().remember("project", "crumbly 2", "The Crumbly 2.0 launch, March 17.")
    await _cap().add_task("fix the logo on the hero image", project="crumbly 2")
    rows = json.loads(await mcp_server.tasks("launch"))["tasks"]
    assert [r["text"] for r in rows] == ["fix the logo on the hero image"]


async def test_tasks_filter_that_matches_nothing_shows_everything_and_says_so(tmp_path):
    await _setup(tmp_path)
    await _cap().add_task("send the press email to journalists")
    out = json.loads(await mcp_server.tasks("launch"))
    assert [r["text"] for r in out["tasks"]] == ["send the press email to journalists"]
    assert "matched no open task" in out["hint"]


# ── cards: the user's own new record needs no card ───────────────────────────

async def test_a_new_record_in_the_users_own_words_saves_without_a_card(tmp_path):
    await _setup(tmp_path)
    r = await _cap(Provenance.USER_INSTRUCTED).remember("person", "Felix", "Felix owns the landing page.")
    assert r.outcome == "applied"
    assert await mcp_server.pending_cards() == []
    assert (await store.get_structured("people.felix"))[0].content == "Felix owns the landing page."


async def test_agent_words_world_input_and_changes_still_card(tmp_path):
    await _setup(tmp_path)
    assert (await _cap(Provenance.AGENT_INFERRED).remember("fact", "budget", "The budget is 18k.")).outcome == "pending"
    assert (await _cap(Provenance.WORLD).remember("person", "Mara", "Mara is a designer.")).outcome == "pending"
    await _cap().remember("person", "Felix", "Felix owns the landing page.")
    r = await _cap().remember("person", "Felix", "Felix owns the pricing page.")     # a change to a record
    assert r.outcome == "pending"
    assert len(await mcp_server.pending_cards()) == 3


async def test_a_filter_that_only_matches_a_closed_task_says_so(tmp_path):
    await _setup(tmp_path)
    await _cap().add_task("finish the landing page draft")
    assert (await _cap().complete_task("finish the landing page draft")).outcome == "completed"
    await _cap().add_task("send the press email")
    out = json.loads(await mcp_server.tasks("landing page draft"))
    assert [r["text"] for r in out["closed_matches"]] == ["finish the landing page draft"]
    assert out["closed_matches"][0]["status"] == "done"
    assert "closed" in out["hint"]


async def test_add_task_tells_agents_what_is_not_a_task(tmp_path):
    srv = mcp_server.build_server()
    desc = next(t for t in await srv.list_tools() if t.name == "add_task").description
    assert "wish" in desc and "declined" in desc and "advice" in desc and "how-to" in desc


async def test_a_deadline_of_today_is_kept_after_nine(tmp_path, monkeypatch):
    await _setup(tmp_path)
    today = datetime.now().strftime("%Y-%m-%d")
    monkeypatch.setenv("KEN_NOW", f"{today}T15:00")               # after the 09:00 a date-only deadline maps to
    r = await _cap().add_task("file the expense report", deadline=today)
    assert r.outcome == "task" and "passed" not in r.text
    (t,) = await store.open_tasks()
    assert t.deadline is not None
    await _cap().add_task("call the bank")
    assert (await _cap().reschedule_task("call the bank", deadline=today)).outcome == "rescheduled"


async def test_add_task_says_when_is_the_old_name_of_deadline(tmp_path):
    srv = mcp_server.build_server()
    desc = next(t for t in await srv.list_tools() if t.name == "add_task").description
    assert "`when` is the old name of `deadline`" in desc


def test_server_instructions_say_what_a_task_is():
    ins = mcp_server.build_server().instructions.lower()
    assert "commitment" in ins and "advice" in ins


# ── an owner change names the person notes that still say otherwise ─────────

async def test_assign_names_a_person_note_that_still_claims_the_task(tmp_path):
    await _setup(tmp_path)
    await _cap().remember("person", "Omar", "Omar owns the Crumbly 2.0 release notes.")
    await _cap().remember("person", "Felix", "Felix owns the Crumbly 2.0 landing page.")
    await _cap().add_task("complete the Crumbly 2.0 release notes")
    r = await _cap().assign_task("release notes", owner="Nora")
    assert r.outcome == "assigned"
    assert "omar" in r.text.lower() and "remember" in r.text and "felix" not in r.text.lower()


async def test_assign_says_nothing_extra_when_no_note_disagrees(tmp_path):
    await _setup(tmp_path)
    await _cap().remember("person", "Omar", "Omar works on the API docs.")
    await _cap().add_task("complete the release notes")
    r = await _cap().assign_task("release notes", owner="Nora")
    assert "remember" not in r.text
