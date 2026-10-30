"""A task's text carries no deadline phrase; the date lives in the `deadline` field.

The benchmark pilot (kitchen scenario): a moved deadline left "by March 20, 2026" in the
text (agents read the text, answered the old date), and "by April 1, 2026" stayed only in
the text (deadline None, so lanes, overdue and the Day page missed it). Plus: an agent gave
up after one empty find.
"""
import asyncio
import json
from datetime import datetime, timedelta

import pytest

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


def _cap(now: int) -> Captures:
    ctx = Context(conversation_id=bus.MAIN, now=now, message="")
    ctx.memory = MemoryReader(Assistant.slice)
    ctx.propose = Proposer(conversation_id=bus.MAIN, author="kip")
    return Captures(ctx, parse_when=_parse_when, provenance=Provenance.USER_INSTRUCTED)


def _nine(d: datetime) -> int:
    return int(d.replace(hour=9, minute=0, second=0, microsecond=0).timestamp() * 1000)


def _long(d: datetime) -> str:                     # "March 20, 2027"
    return f"{d:%B} {d.day}, {d.year}"


async def test_reschedule_strips_the_old_date_from_the_text(tmp_path):
    """Bug 1: the deadline moved, the text kept the old date."""
    await _setup(tmp_path)
    now = now_ms()
    old, new = datetime.now() + timedelta(days=40), datetime.now() + timedelta(days=30)
    # a task as the old code stored it: the date in the text AND the field
    await store.add_task(f"Order floor tiles for the kitchen renovation by {_long(old)}",
                         Provenance.USER_STATED, deadline=_nine(old))
    r = await _cap(now).reschedule_task("floor tiles", deadline=new.strftime("%Y-%m-%d"))
    assert r.outcome == "rescheduled"
    (t,) = await store.open_tasks()
    assert t.deadline == _nine(new)
    assert t.text == "Order floor tiles for the kitchen renovation"
    row = json.loads(await mcp_server.tasks(now=now))["tasks"][0]          # what agents read
    assert row["text"] == "Order floor tiles for the kitchen renovation"
    ctx = json.loads(await mcp_server.get_context(t.id, now=now))
    assert ctx["task"]["text"] == "Order floor tiles for the kitchen renovation"


async def test_add_with_a_deadline_strips_the_phrase_the_field_wins(tmp_path):
    await _setup(tmp_path)
    when, said = datetime.now() + timedelta(days=30), datetime.now() + timedelta(days=50)
    await _cap(now_ms()).add_task(f"Order floor tiles by {_long(said)}", deadline=when.strftime("%Y-%m-%d"))
    (t,) = await store.open_tasks()
    assert (t.text, t.deadline) == ("Order floor tiles", _nine(when))


async def test_a_date_only_in_the_text_becomes_the_deadline(tmp_path):
    """Bug 2: 'Book an electrician by April 1, 2026' was stored with deadline None."""
    await _setup(tmp_path)
    due = datetime.now() + timedelta(days=60)
    r = await _cap(now_ms()).add_task(f"Book an electrician by {_long(due)}")
    assert r.outcome == "task"
    (t,) = await store.open_tasks()
    assert (t.text, t.deadline) == ("Book an electrician", _nine(due))


# ── the phrase parser: what it takes and what it leaves alone ─────────────────

_NOW = datetime(2026, 3, 10, 12, 0)  # noqa: DTZ001 — local, like the parser; a Tuesday


@pytest.mark.parametrize(("text", "clean", "date"), [
    ("Order tiles by March 20", "Order tiles", "2026-03-20"),
    ("Order tiles by March 20, 2026", "Order tiles", "2026-03-20"),
    ("Order tiles by Mar 20th 2027", "Order tiles", "2027-03-20"),
    ("Order tiles by 20 March", "Order tiles", "2026-03-20"),
    ("Order tiles by 2026-03-20", "Order tiles", "2026-03-20"),
    ("Order tiles by the 14th", "Order tiles", "2026-03-14"),
    ("Order tiles by the 5th", "Order tiles", "2026-04-05"),           # the 5th has passed: next month
    ("Pay rent by March 1", "Pay rent", "2027-03-01"),                 # passed this year: next year
    ("Book an electrician due April 1", "Book an electrician", "2026-04-01"),
    ("Book an electrician by April 1st", "Book an electrician", "2026-04-01"),
    ("Send the report before Friday", "Send the report", "2026-03-13"),
    ("Send the report by tomorrow", "Send the report", "2026-03-11"),
    ("Send the report (due April 1)", "Send the report", "2026-04-01"),
    ("Order tiles by March 20 for the kitchen", "Order tiles for the kitchen", "2026-03-20"),
    ("Order tiles by March 20 at 5pm", "Order tiles", "2026-03-20T17:00"),
    ("Order tiles, due: March 20.", "Order tiles", "2026-03-20"),
    ("Order tiles by the 14th of March", "Order tiles", "2026-03-14"),
    ("Order tiles by March 20 at 15:30", "Order tiles", "2026-03-20T15:30"),
    ("Order tiles by March 20. Then call Ann", "Order tiles. Then call Ann", "2026-03-20"),
])
def test_split_deadline_takes_clear_phrases(text, clean, date):
    from ken.util import split_deadline
    assert split_deadline(text, _NOW) == (clean, date)


@pytest.mark.parametrize("text", [
    "call at 3pm",                         # a time as content, no by/due
    "Book a table for four",
    "Read the book by Stephen King",       # 'by' without a date
    "Meet Bob on March 20",                # an event date, not a deadline phrase
    "Finish by next Friday",               # deliberately not handled (ambiguous)
    "Order tiles by February 30",          # not a real date: left alone
    "by March 20",                         # nothing left: keep the text
    "Pay Ann by Monday at 3",              # 'at 3' is neither am nor pm: left alone
])
def test_split_deadline_leaves_everything_else_alone(text):
    from ken.util import split_deadline
    assert split_deadline(text, _NOW) == (text, None)


# ── find: an empty result points at tasks() and world() ─────────────────────

async def test_empty_find_says_where_to_look_next(tmp_path):
    await _setup(tmp_path)
    await store.add_task("Book an electrician", Provenance.USER_STATED)
    empty = json.loads(await mcp_server.find("plumber invoice"))
    assert empty["tasks"] == [] and "tasks()" in empty["hint"] and "world()" in empty["hint"]
    assert "hint" in json.loads(await mcp_server.find("what is the"))       # only common words
    hit = json.loads(await mcp_server.find("electrician"))
    assert hit["tasks"] and "hint" not in hit


def test_find_description_steers_to_tasks_and_world():
    srv = mcp_server.build_server()
    desc = {t.name: t.description for t in asyncio.run(srv.list_tools())}
    assert "tasks()" in desc["find"] and "world()" in desc["find"] and "nothing" in desc["find"]
    assert "deadline" in desc["tasks"] and "due" in desc["tasks"]
    assert "person" in desc["world"] and "notes" in desc["world"]
