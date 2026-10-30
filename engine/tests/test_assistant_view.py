"""The Assistant's Today panel — the deterministic spec, and that run() publishes it."""
import json
from datetime import datetime

from ken import db
from ken.agents import assistant_view
from ken.agents.assistant import Assistant
from ken.agents.base import Context
from ken.capabilities import MemoryReader, Proposer, ViewPublisher
from ken.contracts import Provenance, Task
from ken.conversation import bus
from ken.memory import store

_NOW = int(datetime(2026, 8, 14, 9, 0).timestamp() * 1000)  # noqa: DTZ001 — local, by design


def _ctx(conversation_id, *, message="hi"):
    """A Context with the engine's slice-scoped capabilities. The panel publishes to
    the Assistant's OWN session ("main"), never the conversation it is answering in —
    so a misroute reply cannot leak the panel onto a foreign tab."""
    ctx = Context(conversation_id=conversation_id, now=_NOW, message=message,
                  history=[], tasks=[])
    ctx.memory = MemoryReader(Assistant.slice)
    ctx.propose = Proposer(conversation_id=conversation_id, author="Assistant")
    ctx.view = ViewPublisher("main")
    return ctx


class _Capture:
    """Fires the named capture tools, then replies — simulates the model capturing."""
    def __init__(self, calls):
        self.calls = calls

    async def complete(self, system, messages, *, tools=None):
        specs = {t.name: t for t in (tools or [])}
        for name, args in self.calls:
            await specs[name].fn(**args)
        return "done"


def _agenda_texts(view):
    return [i["text"] for b in view["blocks"] if b["type"] == "agenda" for i in b["items"]]


def _lane(view, label):
    (lanes,) = [b for b in view["blocks"] if b["type"] == "lanes"]
    return next(ln for ln in lanes["items"] if ln["label"] == label)


def test_build_view_is_the_bridge():
    view = assistant_view.build_view(
        [Task(id="a", text="finish report"), Task(id="b", text="call dentist")], _NOW)
    assert view["title"] == "today" and "Aug 14" in view["subtitle"]
    head = view["blocks"][0]
    assert head["type"] == "note" and head["strong"] and head["text"] == "Nothing for today."
    assert head["tail"] == "2 open"
    assert view["blocks"][1]["type"] == "timeline" and view["blocks"][1]["marks"] == []
    assert _lane(view, "someday")["count"] == 2                 # untimed + no ranker → someday
    assert _lane(view, "someday")["names"] == "finish report · call dentist"
    assert not [b for b in view["blocks"] if b["type"] == "agenda"]   # nothing first → no cards
    nudge = view["blocks"][-1]
    assert nudge["accent"] and nudge["text"] == "pick one for today: finish report?"


async def _view_of(conversation_id):
    async with db.get_db() as c:
        row = await (await c.execute(
            "SELECT body FROM views WHERE conversation_id=?", (conversation_id,))).fetchone()
    return json.loads(row["body"]) if row else None


def test_timed_tasks_land_by_time_and_the_overdue_one_is_a_red_card():
    hour = 3_600_000
    view = assistant_view.build_view([
        Task(id="p", text="pay the invoice", deadline=_NOW - 50 * hour),      # 2 days ago
        Task(id="m", text="call mom", deadline=_NOW + 2 * hour),              # today 11:00
        Task(id="w", text="dentist", deadline=_NOW + 3 * 24 * hour),          # this week
        Task(id="s", text="renew passport", deadline=_NOW + 30 * 24 * hour)], _NOW)  # someday
    assert view["blocks"][0]["text"] == "1 overdue, 1 for today."
    assert view["blocks"][1]["marks"] == [11 * 60]                # only today's task marks the strip
    assert [ _lane(view, l)["count"] for l in ("overdue", "today", "this week", "someday")] == [1, 1, 1, 1]
    (agenda,) = [b for b in view["blocks"] if b["type"] == "agenda"]
    first, second = agenda["items"]
    assert first["edge"] == "bad" and first["text"] == "pay the invoice" and "2d overdue" in first["when"]
    assert second["edge"] == "high" and second["text"] == "call mom" and "overdue" not in second["when"]
    assert view["blocks"][-1]["text"] == "ask kip to reschedule the 1 overdue, or tap done"


def test_scores_sort_the_untimed_backlog_into_lanes():
    # scores are aligned with the UNTIMED tasks; a timed task in between is placed by time.
    # Rounded to the rubric step: 2.0 ("today", confidently) IS today, not "this week".
    hour = 3_600_000
    tasks = [Task(id="a", text="urgent: fix prod"), Task(id="t", text="dentist", deadline=_NOW + 30 * 24 * hour),
             Task(id="b", text="book flights"), Task(id="c", text="learn Rust someday")]
    view = assistant_view.build_view(tasks, _NOW, scores=[2.0, 1.2, 0.2], ranked_at=_NOW)
    assert _lane(view, "today")["names"] == "urgent: fix prod"
    assert _lane(view, "this week")["names"] == "book flights"
    assert _lane(view, "someday")["names"] == "dentist · learn Rust someday"
    assert "ranked 09:00" in view["subtitle"]
    assert view["blocks"][-1]["text"] == "next up: urgent: fix prod"


def test_long_text_never_wraps_and_names_are_capped():
    tasks = [Task(id=str(i), text=f"task {i} " + "x" * 80) for i in range(6)]
    view = assistant_view.build_view(tasks, _NOW)
    names = _lane(view, "someday")["names"]
    assert names.endswith("· +4") and all(len(n) <= 24 for n in names.split(" · ")[:-1])
    assert _lane(view, "someday")["count"] == 6                   # the lane still shows the true total


def test_done_tasks_and_events_this_week_make_the_bars():
    from types import SimpleNamespace
    ev = [SimpleNamespace(summary="lunch with Sarah", at=_NOW - 86_400_000),
          SimpleNamespace(summary="ancient", at=_NOW - 30 * 86_400_000)]
    done = [Task(id="d1", text="shipped the coder agent", status="done", completed_at=_NOW - 3_600_000),
            Task(id="d2", text="old done thing", status="done", completed_at=_NOW - 20 * 86_400_000),
            Task(id="d3", text="done before completion times existed", status="done")]   # completed_at None
    view = assistant_view.build_view([], _NOW, events=ev, done=done)
    assert view["blocks"][0]["text"] == "Nothing open. Clear."
    assert view["blocks"][0]["tail"] == "0 open · 1 done this week · 1 happened"
    (bars,) = [b for b in view["blocks"] if b["type"] == "bars"]
    d, h = bars["items"]
    assert d["label"] == "done ·7d" and d["level"] == "good" and d["count"] == 1 and "coder agent" in d["tail"]
    assert h["label"] == "happened" and h["count"] == 1 and "lunch with Sarah" in h["tail"]


async def test_run_publishes_the_today_view(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    await store.add_task("water plants", Provenance.AGENT_INFERRED)
    await Assistant(llm=None).run(_ctx(bus.MAIN))          # even the stub path refreshes it
    view = await _view_of(bus.MAIN)
    assert "water plants" in _lane(view, "someday")["names"]


class _FakeRank:
    def __init__(self, scores): self.scores, self.seen = scores, None
    async def __call__(self, items, *, instructions, rubric):
        self.seen = items
        return self.scores


def test_a_date_only_task_is_due_all_day_not_overdue_at_midnight():
    midnight = int(datetime(2026, 8, 14).timestamp() * 1000)          # noqa: DTZ001 — today, 00:00
    yesterday = midnight - 86_400_000
    view = assistant_view.build_view([Task(id="p", text="renew passport", deadline=midnight),
                                      Task(id="q", text="file taxes", deadline=yesterday)], _NOW)
    assert _lane(view, "today")["names"] == "renew passport"       # due today, all day
    assert _lane(view, "overdue")["names"] == "file taxes"          # due yesterday → overdue since midnight
    assert view["blocks"][1]["marks"] == []                          # no clock mark for a date-only task
    (agenda,) = [b for b in view["blocks"] if b["type"] == "agenda"]
    assert agenda["items"][0]["when"].startswith("due ") and "overdue" in agenda["items"][0]["when"]
    assert agenda["items"][1]["when"] == "due Fri"


async def test_a_short_score_list_falls_back_to_unranked_lanes(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    await store.add_task("a", Provenance.AGENT_INFERRED)
    await store.add_task("b", Provenance.AGENT_INFERRED)
    await Assistant(llm=None, rank=_FakeRank([2.9])).run(_ctx(bus.MAIN))   # 1 score for 2 tasks
    view = await _view_of(bus.MAIN)
    assert _lane(view, "someday")["count"] == 2 and "ranked" not in view["subtitle"]   # published, unranked


async def test_the_refresh_cron_republishes_the_panel_and_says_nothing(tmp_path):
    from ken.agents.assistant import _REFRESH_CRON
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    await store.add_task("water plants", Provenance.AGENT_INFERRED)
    ctx = _ctx(bus.MAIN, message="")
    ctx.woke_by, ctx.cron = "schedule", _REFRESH_CRON
    assert await Assistant(llm=None).run(ctx) is None              # no brief, no advice
    assert "water plants" in _lane(await _view_of(bus.MAIN), "someday")["names"]


async def test_completing_a_task_stamps_the_time_and_shows_in_done(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    tid, _ = await store.add_task("water plants", Provenance.AGENT_INFERRED)
    assert await store.done_since(0) == []
    assert await store.complete_task(tid) is True
    (t,) = await store.done_since(0)
    assert t.id == tid and t.status == "done" and t.completed_at is not None
    assert await store.complete_task(tid) is False              # already done: no re-stamp
    await Assistant(llm=None).run(_ctx(bus.MAIN))
    view = await _view_of(bus.MAIN)
    assert "1 done this week" in view["blocks"][0]["tail"]
    (bars,) = [b for b in view["blocks"] if b["type"] == "bars"]
    assert bars["items"][0]["label"] == "done ·7d" and "water plants" in bars["items"][0]["tail"]


async def test_run_ranks_only_the_untimed_tasks(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    await store.add_task("water plants", Provenance.AGENT_INFERRED)
    await store.add_task("dentist", Provenance.AGENT_INFERRED, deadline=_NOW + 3_600_000)
    rank = _FakeRank([2.9])
    await Assistant(llm=None, rank=rank).run(_ctx(bus.MAIN))
    assert rank.seen == ["water plants"]                          # the timed one is placed by time
    view = await _view_of(bus.MAIN)
    assert "water plants" in _lane(view, "today")["names"]        # ranked 2.9 → today (with the 10:00 dentist)
    assert "ranked" in view["subtitle"]


async def test_panel_goes_to_the_assistants_own_tab_not_a_foreign_session(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    await store.add_task("water plants", Provenance.AGENT_INFERRED)
    # the Assistant answers a message in a session it does NOT own (misroute fallback)
    await Assistant(llm=None).run(_ctx("someone-elses-tab"))
    assert await _view_of(bus.MAIN) is not None          # the panel lands on its own tab
    assert await _view_of("someone-elses-tab") is None   # NOT the foreign tab


async def test_view_reflects_a_task_captured_this_turn(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    # ctx.tasks is empty at run start; the task is captured mid-run, and the panel is
    # rebuilt from FRESH open_tasks() — so it must already show the new task.
    await Assistant(llm=_Capture([("add_task", {"text": "buy milk"})])).run(
        _ctx(bus.MAIN, message="remind me to X"))
    view = await _view_of(bus.MAIN)
    assert "buy milk" in _lane(view, "someday")["names"]
