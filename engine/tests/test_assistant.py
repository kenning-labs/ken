"""Assistant memory loop — recall (the slice) + capture (propose -> consent gate).

No network: the LLM is faked. FakeLLM exercises recall (what the model is shown);
ToolCallingLLM exercises capture (the model deciding to call a capture tool).
"""

from ken import db
from ken.agents.assistant import Assistant
from ken.agents.base import Context, Proposal
from ken.capabilities import MemoryReader, Proposer, ViewPublisher
from ken.consent import gate
from ken.contracts import Provenance, StructuredEntry, Task
from ken.conversation import bus
from ken.memory import store


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


class FakeLLM:
    """Records what it was shown, returns a fixed reply, ignores tools."""
    def __init__(self, reply="ok"):
        self.reply = reply
        self.seen = None

    async def complete(self, system, messages, *, tools=None):
        self.seen = (system, messages, tools)
        return self.reply


class ToolCallingLLM:
    """Invokes the given capture tools with preset args — simulates the model
    deciding to capture — and records each tool's return, then returns a reply."""
    def __init__(self, calls, reply="done"):
        self.calls = calls          # [(tool_name, args_dict), ...]
        self.reply = reply
        self.results = []

    async def complete(self, system, messages, *, tools=None):
        self.seen_system = system
        self.seen_messages = messages
        specs = {t.name: t for t in (tools or [])}
        for name, args in self.calls:
            self.results.append(await specs[name].fn(**args))
        return self.reply


def _ctx(message="", *, tasks=None, structured=None, events=None, now=0):
    """A Context with the SAME slice-scoped capabilities the engine injects — so the
    tests exercise the real injection (live memory, gate-fronted writes, panel publish)
    rather than reaching store/gate directly."""
    ctx = Context(conversation_id=bus.MAIN, now=now, message=message, history=[],
                  tasks=tasks or [], structured=structured or [], events=events or [])
    ctx.memory = MemoryReader(Assistant.slice)
    ctx.propose = Proposer(conversation_id=bus.MAIN, author="Assistant")
    ctx.view = ViewPublisher("main")
    return ctx


# ── recall: the slice the model is shown ─────────────────────────────────────

async def test_recall_shows_open_tasks_and_saved_beliefs(tmp_path):
    await _setup(tmp_path)
    llm = FakeLLM()
    ctx = _ctx(
        "what's on my plate, and how do i like my mornings?",
        tasks=[Task(id="t1", text="call the dentist")],
        structured=[StructuredEntry(id="preferences.schedule", kind="preferences.schedule",
                                    subject="you", content="prefers mornings")])
    await Assistant(llm=llm).run(ctx)
    shown = llm.seen[1][-1]["content"]              # the folded-in user message
    assert "call the dentist" in shown             # the task is recalled
    assert "prefers mornings" in shown             # the saved preference is recalled


# ── capture: propose -> consent gate ─────────────────────────────────────────

async def test_add_task_captures_through_the_gate(tmp_path):
    await _setup(tmp_path)
    llm = ToolCallingLLM([("add_task", {"text": "buy running shoes"})])
    await Assistant(llm=llm).run(_ctx("remind me to buy running shoes"))
    tasks = await store.open_tasks()
    assert [t.text for t in tasks] == ["buy running shoes"]
    assert tasks[0].provenance is Provenance.AGENT_INFERRED   # captured, not user-typed


async def test_add_task_is_idempotent(tmp_path):
    await _setup(tmp_path)
    # the model re-proposes a task it already has (a ~75% behavior) -> no duplicate,
    # and it is told the truth, not "added"
    llm = ToolCallingLLM([("add_task", {"text": "pray tomorrow"}),
                          ("add_task", {"text": "pray tomorrow"})])
    await Assistant(llm=llm).run(_ctx("go"))
    assert [t.text for t in await store.open_tasks()] == ["pray tomorrow"]     # not two
    assert llm.results == ["added to your tasks", "already on your list"]      # truthful


async def test_add_task_dedup_ignores_case_and_surrounding_space(tmp_path):
    await _setup(tmp_path)
    id1, c1 = await store.add_task("Water Plants", Provenance.AGENT_INFERRED)
    id2, c2 = await store.add_task("  water plants  ", Provenance.AGENT_INFERRED)
    assert c1 and not c2 and id1 == id2                 # a trivial variant is not a new task
    assert len(await store.open_tasks()) == 1


async def test_add_task_folds_non_latin_case(tmp_path):
    await _setup(tmp_path)
    _, c1 = await store.add_task("Позвонить маме", Provenance.AGENT_INFERRED)
    _, c2 = await store.add_task("позвонить маме", Provenance.AGENT_INFERRED)   # Cyrillic case
    assert c1 and not c2 and len(await store.open_tasks()) == 1


async def test_add_task_adopts_a_schedule_but_never_reschedules(tmp_path):
    await _setup(tmp_path)
    id1, _ = await store.add_task("call mom", Provenance.AGENT_INFERRED)
    id2, created = await store.add_task("call mom", Provenance.AGENT_INFERRED, deadline=999)
    assert id1 == id2 and not created
    assert (await store.open_tasks())[0].deadline == 999   # adopted (was unscheduled)
    await store.add_task("call mom", Provenance.AGENT_INFERRED, deadline=555)
    assert (await store.open_tasks())[0].deadline == 999   # existing schedule NOT overwritten


async def test_add_task_with_a_time_schedules_it(tmp_path):
    from ken.agents.assistant import _parse_when
    await _setup(tmp_path)
    when = "2026-08-17T15:00"                                 # the model resolved "3pm tomorrow"
    llm = ToolCallingLLM([("add_task", {"text": "call mom", "when": when})])
    await Assistant(llm=llm).run(_ctx("remind me to call mom at 3pm tomorrow"))
    task = (await store.open_tasks())[0]
    assert task.text == "call mom" and task.deadline == _parse_when(when)


async def test_add_task_without_a_time_is_unscheduled(tmp_path):
    await _setup(tmp_path)
    await Assistant(llm=ToolCallingLLM([("add_task", {"text": "buy milk"})])).run(_ctx("buy milk"))
    assert (await store.open_tasks())[0].deadline is None


def test_parse_when_and_label():
    from ken.agents.assistant import _parse_when
    from ken.util import when_label
    assert _parse_when("") is None and _parse_when("not-a-date") is None    # graceful
    assert _parse_when(1500) is None                                       # non-string -> no crash
    ms = _parse_when("2026-08-17T15:30")
    assert ms is not None and "3:30pm" in when_label(ms)


def test_when_label_disambiguates_far_out_dates():
    from ken.util import when_label
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    near = when_label(2 * 86_400_000, 0)     # 2 days out -> weekday only
    far = when_label(10 * 86_400_000, 0)     # 10 days out -> adds a date
    assert not any(m in near for m in months) and any(m in far for m in months)


async def test_add_task_result_reflects_the_stored_time(tmp_path):
    await _setup(tmp_path)
    llm = ToolCallingLLM([("add_task", {"text": "call mom", "when": "2026-08-17T15:00"})])
    await Assistant(llm=llm).run(_ctx("call mom at 3pm tomorrow"))
    assert "scheduled for" in llm.results[0]           # reply anchored to the REAL stored time


async def test_add_task_with_a_past_time_promises_no_nudge(tmp_path):
    # a resolved time already in the past can never fire — the task still lands on the
    # list, but it is NOT scheduled and the reply never claims a reminder. (FIX 4)
    await _setup(tmp_path)
    now = 2_000_000_000_000                                    # ~2033, well after the date below
    past = "2001-01-01T09:00"                                  # long before `now`
    llm = ToolCallingLLM([("add_task", {"text": "call mom", "when": past})])
    await Assistant(llm=llm).run(_ctx("call mom at 9am", now=now))
    assert (await store.open_tasks())[0].deadline is None  # not scheduled — no false nudge
    assert "already passed" in llm.results[0]                   # the model is told the truth


async def test_add_task_flags_an_unreadable_time_not_claims_one(tmp_path):
    await _setup(tmp_path)
    llm = ToolCallingLLM([("add_task", {"text": "call mom", "when": "3PM tomorrow"})])   # not ISO
    await Assistant(llm=llm).run(_ctx("call mom at 3pm"))
    assert (await store.open_tasks())[0].deadline is None      # not stored
    assert "couldn't read that time" in llm.results[0]              # the model is TOLD, can't over-claim


async def test_add_task_reports_an_adopted_schedule(tmp_path):
    await _setup(tmp_path)
    await store.add_task("book the dentist", Provenance.AGENT_INFERRED)   # existing, un-timed
    llm = ToolCallingLLM([("add_task", {"text": "book the dentist", "when": "2026-08-17T15:00"})])
    await Assistant(llm=llm).run(_ctx("schedule the dentist for 3pm tomorrow"))
    assert (await store.open_tasks())[0].deadline is not None  # time adopted onto it
    assert "scheduled for" in llm.results[0]                        # and the reply says so, not a no-op


async def test_remember_preference_fast_tracks(tmp_path):
    await _setup(tmp_path)
    llm = ToolCallingLLM([("remember_preference",
                           {"topic": "workouts", "detail": "prefers morning workouts"})])
    await Assistant(llm=llm).run(_ctx("i like to train in the morning"))
    prefs = await store.get_structured("preferences.")
    assert len(prefs) == 1 and prefs[0].content == "prefers morning workouts"


async def test_remember_fact_asks_and_does_not_apply_yet(tmp_path):
    await _setup(tmp_path)
    llm = ToolCallingLLM([("remember_fact", {"topic": "son", "fact": "Leo"})])
    await Assistant(llm=llm).run(_ctx("my son's name is Leo"))
    # a fact is higher-stakes: NOT applied, a pending confirmation card is open instead
    assert await store.get_structured("facts.") == []
    async with db.get_db() as c:
        pending = await (await c.execute("SELECT kind, content FROM pending")).fetchall()
    assert [(r["kind"], r["content"]) for r in pending] == [("facts.son", "Leo")]


async def test_recall_renders_people_in_their_own_labeled_section(tmp_path):
    # people show under their own heading with a DE-SLUGGED name, so 'who is X?' reads a clean
    # roster — not people mixed anonymously among the user's own facts.
    await _setup(tmp_path)
    llm = FakeLLM()
    ctx = _ctx("who is cristian?",
               structured=[StructuredEntry(id="people.omar_haddad", kind="people.omar_haddad",
                                           subject="you", content="Android engineer")])
    await Assistant(llm=llm).run(ctx)
    shown = llm.seen[1][-1]["content"]
    assert "People in your world:" in shown
    assert "Omar Haddad: Android engineer" in shown            # de-slugged, not "omar_haddad"


async def test_remember_person_asks_through_the_gate(tmp_path):
    # the user tells kip who someone is -> remember_person proposes people.<slug>; a card goes
    # up (people is an ask-namespace), nothing is silently saved.
    await _setup(tmp_path)
    llm = ToolCallingLLM([("remember_person", {"name": "Dana", "note": "my manager; owns hiring"})])
    await Assistant(llm=llm).run(_ctx("Dana is my manager"))
    assert await store.get_structured("people.") == []           # not saved yet — carded
    async with db.get_db() as c:
        pending = await (await c.execute("SELECT kind, content FROM pending")).fetchall()
    assert [(r["kind"], r["content"]) for r in pending] == [("people.dana", "my manager; owns hiring")]


async def test_remember_project_asks_through_the_gate(tmp_path):
    # the user names a project -> remember_project proposes projects.<slug>; a card goes up
    # (projects is an ask-namespace), nothing is silently saved. No automatic discovery.
    await _setup(tmp_path)
    llm = ToolCallingLLM([("remember_project",
                           {"name": "Larder rollout", "note": "ship pre-order to Android"})])
    await Assistant(llm=llm).run(_ctx("the Larder rollout is a project I'm tracking"))
    assert await store.get_structured("projects.") == []          # not saved yet — carded
    async with db.get_db() as c:
        pending = await (await c.execute("SELECT kind, content FROM pending")).fetchall()
    assert [(r["kind"], r["content"]) for r in pending] == [("projects.larder_rollout", "ship pre-order to Android")]


async def test_kip_slice_grants_people_and_projects_but_not_ungranted(tmp_path):
    # kip READS people AND projects (slice-scoped + enforced); an ungranted namespace still raises.
    import pytest

    from ken.capabilities import MemoryReader, ScopeError
    await _setup(tmp_path)
    r = MemoryReader(Assistant.slice)
    assert await r.structured("people") == []          # granted → live read, no raise
    assert await r.structured("projects") == []        # now granted too
    with pytest.raises(ScopeError):
        await r.structured("goals")                    # not in kip's slice → raises, never lies with []


async def test_completing_a_fact_ask_applies_it(tmp_path):
    await _setup(tmp_path)
    await Assistant(llm=ToolCallingLLM(
        [("remember_fact", {"topic": "city", "fact": "Berlin"})])).run(_ctx("i live in Berlin"))
    async with db.get_db() as c:
        (ask_id,) = await (await c.execute("SELECT ask_id FROM pending")).fetchone()
    assert await gate.resolve(ask_id, approved=True, conversation_id=bus.MAIN) == "applied"
    facts = await store.get_structured("facts.")
    assert len(facts) == 1 and facts[0].content == "Berlin"


async def test_complete_task_by_text_match(tmp_path):
    await _setup(tmp_path)
    await store.add_task("water the plants", Provenance.AGENT_INFERRED)
    task = (await store.open_tasks())[0]
    # a distinctive part of the text, not a positional number
    llm = ToolCallingLLM([("complete_task", {"task": "water the plants"})])
    await Assistant(llm=llm).run(_ctx("done watering", tasks=[task]))
    assert await store.open_tasks() == []          # no longer open


# ── reschedule: the explicit re-time path ────────────────────────────────────

async def test_reschedule_moves_a_timed_task_and_clears_the_reminder(tmp_path):
    from datetime import datetime, timedelta

    from ken.agents.assistant import _parse_when
    await _setup(tmp_path)
    # relative to now so the reschedule's past-time floor never drops these (no date-bomb):
    # old already fired (yesterday), new is in the future so it must stick.
    old = _parse_when((datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT15:00"))
    tid, _ = await store.add_task("ring the plumber", Provenance.AGENT_INFERRED, deadline=old)
    await store.mark_reminded(tid, old)                       # the old time already fired
    task = (await store.open_tasks())[0]
    new = (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%dT18:00")
    llm = ToolCallingLLM([("reschedule_task", {"task": "ring the plumber", "when": new})])
    await Assistant(llm=llm).run(_ctx("move ring the plumber to sunday 6pm", tasks=[task]))
    assert (await store.open_tasks())[0].deadline == _parse_when(new)   # re-timed
    async with db.get_db() as c:
        row = await (await c.execute(
            "SELECT reminded_at FROM tasks WHERE id=?", (tid,))).fetchone()
    assert row["reminded_at"] is None                         # cleared -> a fresh nudge can fire
    assert "moved" in llm.results[0]


async def test_reschedule_asks_which_when_several_match(tmp_path):
    await _setup(tmp_path)
    await store.add_task("call the plumber", Provenance.AGENT_INFERRED)
    await store.add_task("call the plumber about the sink", Provenance.AGENT_INFERRED)
    tasks = await store.open_tasks()
    llm = ToolCallingLLM([("reschedule_task", {"task": "plumber", "when": "2026-08-23T18:00"})])
    await Assistant(llm=llm).run(_ctx("move the plumber to sunday", tasks=tasks))
    assert "several match" in llm.results[0] and "which one" in llm.results[0]


async def test_reschedule_no_match_is_honest(tmp_path):
    await _setup(tmp_path)
    await store.add_task("call the dentist", Provenance.AGENT_INFERRED)
    task = (await store.open_tasks())[0]
    llm = ToolCallingLLM([("reschedule_task", {"task": "buy groceries", "when": "2026-08-23T18:00"})])
    await Assistant(llm=llm).run(_ctx("move groceries", tasks=[task]))
    assert llm.results == ["no open task matches that"]


async def test_reschedule_to_a_past_time_is_refused(tmp_path):
    await _setup(tmp_path)
    now = 2_000_000_000_000                                    # ~2033
    await store.add_task("ring the plumber", Provenance.AGENT_INFERRED, deadline=now)
    task = (await store.open_tasks())[0]
    llm = ToolCallingLLM([("reschedule_task", {"task": "plumber", "when": "2001-01-01T09:00"})])
    await Assistant(llm=llm).run(_ctx("move it back", tasks=[task], now=now))
    assert (await store.open_tasks())[0].deadline == now  # unchanged — the past time is refused
    assert "already passed" in llm.results[0]


async def test_reschedule_unreadable_time_is_honest(tmp_path):
    await _setup(tmp_path)
    await store.add_task("ring the plumber", Provenance.AGENT_INFERRED)
    task = (await store.open_tasks())[0]
    llm = ToolCallingLLM([("reschedule_task", {"task": "plumber", "when": "sunday at 6"})])  # not ISO
    await Assistant(llm=llm).run(_ctx("move the plumber", tasks=[task]))
    assert (await store.open_tasks())[0].deadline is None   # not set
    assert "couldn't read that time" in llm.results[0]


# ── find_task: the on-demand read that reaches completed tasks ───────────────

async def test_find_task_reaches_completed_and_hides_the_id(tmp_path):
    await _setup(tmp_path)
    tid, _ = await store.add_task("book the dentist", Provenance.AGENT_INFERRED)
    await store.complete_task(tid)                            # done -> off the open list
    llm = ToolCallingLLM([("find_task", {"query": "dentist"})])
    await Assistant(llm=llm).run(_ctx("did i finish the dentist thing?"))
    result = llm.results[0]
    assert "book the dentist" in result and "done" in result   # reaches the completed task
    assert tid not in result                                    # never exposes the internal id


async def test_find_task_is_read_only_and_honest_on_no_match(tmp_path):
    await _setup(tmp_path)
    await store.add_task("call mom", Provenance.AGENT_INFERRED)
    llm = ToolCallingLLM([("find_task", {"query": "groceries"})])
    await Assistant(llm=llm).run(_ctx("is groceries done?"))
    assert llm.results == ["no task matches that"]
    assert len(await store.open_tasks()) == 1                   # read-only: nothing changed


# ── log_event: record something that HAPPENED (append-only) ──────────────────

async def test_log_event_records_an_event_at_now_when_no_time(tmp_path):
    await _setup(tmp_path)
    now = 1_700_000_000_000
    llm = ToolCallingLLM([("log_event", {"text": "had lunch with Sarah"})])
    await Assistant(llm=llm).run(_ctx("just had lunch with Sarah", now=now))
    events = await store.recent_events()
    assert [e.summary for e in events] == ["had lunch with Sarah"]
    assert events[0].at >= now                     # stamped 'now' (it happened as they told us)
    assert "logged" in llm.results[0]              # honest outcome relayed


async def test_log_event_uses_a_given_past_time(tmp_path):
    from ken.agents.assistant import _parse_when
    await _setup(tmp_path)
    when = "2026-08-17T09:00"                       # a time the model resolved from "this morning"
    llm = ToolCallingLLM([("log_event", {"text": "went for a run", "when": when})])
    await Assistant(llm=llm).run(_ctx("I ran this morning"))
    assert (await store.recent_events())[0].at == _parse_when(when)   # the time it HAPPENED


async def test_log_event_flags_an_unreadable_time_not_claims_one(tmp_path):
    # a non-ISO `when` cannot parse -> the event still lands (at now), but the reply must
    # be HONEST that the given time was not read (the silent-failure class). (FIX 1)
    await _setup(tmp_path)
    now = 1_700_000_000_000
    llm = ToolCallingLLM([("log_event", {"text": "went for a run", "when": "last Tuesday"})])
    await Assistant(llm=llm).run(_ctx("I went for a run last Tuesday", now=now))
    events = await store.recent_events()
    assert [e.summary for e in events] == ["went for a run"]      # it still landed
    assert events[0].at >= now                                    # stamped now, not a fake time
    assert "couldn't read that time" in llm.results[0]            # honest, never a claimed time


async def test_log_event_collapses_a_duplicate(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    await gate.apply(Proposal(intent="event", content="ran 10k"),
                     conversation_id=bus.MAIN, author="Assistant")
    existing = (await store.recent_events())[0]
    llm = ToolCallingLLM([("log_event", {"text": "ran 10k"})])
    await Assistant(llm=llm, judge=_StaticJudge(judge.collapse(existing.id))).run(_ctx("ran 10k"))
    assert len(await store.recent_events()) == 1                  # no duplicate stored
    assert "already logged that" in llm.results[0]               # told the truth, not "logged"


async def test_find_events_reaches_the_log_and_is_read_only(tmp_path):
    await _setup(tmp_path)
    await gate.apply(Proposal(intent="event", content="finished the report", at=5),
                     conversation_id=bus.MAIN, author="Assistant")
    llm = ToolCallingLLM([("find_events", {"query": "report"})])
    await Assistant(llm=llm).run(_ctx("when did I finish the report?"))
    assert "finished the report" in llm.results[0]               # found the past event
    assert len(await store.recent_events()) == 1                 # read-only: nothing changed


async def test_find_events_is_honest_on_no_match(tmp_path):
    await _setup(tmp_path)
    llm = ToolCallingLLM([("find_events", {"query": "moon landing"})])
    await Assistant(llm=llm).run(_ctx("did I see the moon landing?"))
    assert llm.results == ["no event matches that"]


async def test_recall_shows_recent_events(tmp_path):
    from ken.contracts import Event
    await _setup(tmp_path)
    llm = FakeLLM()
    ctx = _ctx("what have I done lately?",
               events=[Event(id="e1", summary="saw the dentist", at=5)])   # the preload snapshot
    await Assistant(llm=llm).run(ctx)
    shown = llm.seen[1][-1]["content"]
    assert "saw the dentist" in shown                            # the event is recalled in-prompt


def test_assistant_module_imports_neither_store_nor_gate():
    # the core invariant, asserted on the live module: the Assistant reaches memory ONLY
    # through the injected capabilities — it never binds store or gate.
    import ken.agents.assistant as a
    assert not hasattr(a, "store") and not hasattr(a, "gate")


# ── review regressions ───────────────────────────────────────────────────────

def test_slug_keeps_non_latin_and_separates():
    from ken.util import slug
    assert slug("Morning Workouts!") == "morning_workouts"
    assert slug("дочь") and slug("дочь") != slug("город")   # Cyrillic: distinct, not collapsed
    assert slug("🎉") != slug("✨")                           # wordless -> distinct hashes


async def test_non_latin_topics_do_not_collide(tmp_path):
    await _setup(tmp_path)
    llm = ToolCallingLLM([("remember_fact", {"topic": "дочь", "fact": "Мия"}),
                          ("remember_fact", {"topic": "город", "fact": "Берлин"})])
    await Assistant(llm=llm).run(_ctx("два факта"))
    async with db.get_db() as c:
        rows = await (await c.execute("SELECT kind FROM pending")).fetchall()
    kinds = {r["kind"] for r in rows}
    assert len(kinds) == 2 and "facts.note" not in kinds     # two distinct keys, not one


async def test_capture_tools_relay_the_true_outcome(tmp_path):
    await _setup(tmp_path)
    llm = ToolCallingLLM([("add_task", {"text": "x"}),
                          ("remember_preference", {"topic": "a", "detail": "b"}),
                          ("remember_fact", {"topic": "c", "fact": "d"})])
    await Assistant(llm=llm).run(_ctx("go"))
    assert llm.results == ["added to your tasks", "saved", "not saved yet — a confirmation card is up"]


async def test_complete_task_matches_a_distinctive_substring(tmp_path):
    await _setup(tmp_path)
    await store.add_task("Order a new briefcase for us", Provenance.AGENT_INFERRED)
    task = (await store.open_tasks())[0]
    # the user says "briefcase" — a distinctive part matches the full task
    await Assistant(llm=ToolCallingLLM([("complete_task", {"task": "briefcase"})])).run(
        _ctx("finished the briefcase", tasks=[task]))
    assert await store.open_tasks() == []


async def test_complete_task_ignores_a_non_matching_name(tmp_path):
    await _setup(tmp_path)
    await store.add_task("call the dentist", Provenance.AGENT_INFERRED)
    task = (await store.open_tasks())[0]
    llm = ToolCallingLLM([("complete_task", {"task": "buy groceries"})])   # unrelated
    await Assistant(llm=llm).run(_ctx("done shopping", tasks=[task]))
    assert len(await store.open_tasks()) == 1                # nothing wrongly completed
    assert llm.results == ["no open task matches that"]


async def test_complete_task_does_not_match_a_task_name_inside_a_longer_phrase(tmp_path):
    # a short task text ("Gym") must NOT be completed by a query that merely mentions it
    # (or negates it) — "skip gym" contains "gym", but the user did not finish it
    await _setup(tmp_path)
    await store.add_task("Gym", Provenance.AGENT_INFERRED)
    task = (await store.open_tasks())[0]
    llm = ToolCallingLLM([("complete_task", {"task": "skip gym"})])
    await Assistant(llm=llm).run(_ctx("no gym today, skipping it", tasks=[task]))
    assert len(await store.open_tasks()) == 1                # Gym stays open
    assert llm.results == ["no open task matches that"]


async def test_complete_task_tolerates_a_non_string_arg(tmp_path):
    await _setup(tmp_path)
    await store.add_task("call mom", Provenance.AGENT_INFERRED)
    task = (await store.open_tasks())[0]
    llm = ToolCallingLLM([("complete_task", {"task": 1})])   # old number habit -> graceful
    await Assistant(llm=llm).run(_ctx("done", tasks=[task]))
    assert len(await store.open_tasks()) == 1 and llm.results == ["no open task matches that"]


# ── the judged capture path (a judge is injected) ────────────────────────────

class _StaticJudge:
    """A deterministic judge for the assistant path: same decision every call."""
    def __init__(self, decision):
        self.decision = decision

    async def __call__(self, proposal, *, message, existing):
        return self.decision


async def test_assistant_attaches_the_message_as_evidence(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    llm = ToolCallingLLM([("add_task", {"text": "buy shoes"})])
    await Assistant(llm=llm, judge=_StaticJudge(judge.keep())).run(_ctx("please add buy shoes"))
    ev = (await store.open_tasks())[0].evidence
    assert ev and ev[0].source == "user" and ev[0].ref == "please add buy shoes"


async def test_assistant_relays_collapsed_truthfully(tmp_path):
    from ken.consent import judge
    await _setup(tmp_path)
    tid, _ = await store.add_task("call the plumber", Provenance.AGENT_INFERRED)
    llm = ToolCallingLLM([("add_task", {"text": "ring the plumber"})])
    await Assistant(llm=llm, judge=_StaticJudge(judge.collapse(tid))).run(_ctx("ring the plumber"))
    assert len(await store.open_tasks()) == 1                      # no duplicate
    assert "nothing new" in llm.results[0]                        # told the truth, not "added"


async def test_collapse_reply_reports_an_adopted_time(tmp_path):
    # an untimed task collapses with a new time -> a nudge WILL fire, so the reply must
    # disclose the time, never a blanket "nothing to add".
    from ken.agents.assistant import _parse_when
    from ken.consent import judge
    await _setup(tmp_path)
    tid, _ = await store.add_task("call mom", Provenance.AGENT_INFERRED)     # existing, untimed
    llm = ToolCallingLLM([("add_task", {"text": "call mom", "when": "2026-08-17T15:00"})])
    await Assistant(llm=llm, judge=_StaticJudge(judge.collapse(tid))).run(_ctx("call mom at 3pm"))
    assert (await store.open_tasks())[0].deadline == _parse_when("2026-08-17T15:00")
    assert "scheduled for" in llm.results[0] and "3pm" in llm.results[0]
    assert "nothing new" not in llm.results[0]                    # the reminder is disclosed


async def test_collapse_reply_reports_the_kept_time_when_a_retime_is_refused(tmp_path):
    # an already-timed task collapses with a DIFFERENT asked time -> adopt refuses to
    # re-time (correct); the reply must own that honestly, not hide the kept time.
    from ken.agents.assistant import _parse_when
    from ken.consent import judge
    await _setup(tmp_path)
    tid, _ = await store.add_task("call mom", Provenance.AGENT_INFERRED,
                                  deadline=_parse_when("2026-08-17T17:00"))   # 5pm
    llm = ToolCallingLLM([("add_task", {"text": "call mom", "when": "2026-08-17T15:00"})])  # asks 3pm
    await Assistant(llm=llm, judge=_StaticJudge(judge.collapse(tid))).run(_ctx("move call mom to 3pm"))
    assert (await store.open_tasks())[0].deadline == _parse_when("2026-08-17T17:00")  # unchanged
    assert "kept the existing" in llm.results[0] and "5pm" in llm.results[0]
    assert "3pm" not in llm.results[0]                            # never claims the time it did not apply


async def test_fuzzy_collapse_reply_reports_the_adopted_time(tmp_path):
    # the SURVIVING task text differs from the proposal (fuzzy dedup). The read-back must
    # anchor on the gate-returned target, not norm(proposal_text) — so it reports the real
    # adopted time, never a blanket "nothing new to add".
    from ken.agents.assistant import _parse_when
    from ken.consent import judge
    await _setup(tmp_path)
    tid, _ = await store.add_task("call the plumber", Provenance.AGENT_INFERRED)   # untimed
    llm = ToolCallingLLM([("add_task", {"text": "ring the plumber", "when": "2026-08-17T17:00"})])
    await Assistant(llm=llm, judge=_StaticJudge(judge.collapse(tid))).run(_ctx("ring plumber at 5"))
    assert (await store.open_tasks())[0].deadline == _parse_when("2026-08-17T17:00")
    assert "scheduled for" in llm.results[0] and "5pm" in llm.results[0]  # the adopted nudge
    assert "nothing new" not in llm.results[0]                            # not hidden as a no-op


async def test_assistant_relays_a_conflicting_preference_as_a_card_not_saved(tmp_path):
    # a NEW preference saves silently; a CONFLICTING one raises a card and is NOT saved.
    # The tool must relay that honestly — never "saved".
    from ken.consent import judge
    await _setup(tmp_path)
    seed = ToolCallingLLM([("remember_preference", {"topic": "coffee", "detail": "black"})])
    await Assistant(llm=seed, judge=_StaticJudge(judge.keep())).run(_ctx("I take it black"))
    assert "saved" in seed.results[0]                             # the new one WAS saved silently
    llm = ToolCallingLLM([("remember_preference", {"topic": "coffee", "detail": "milk and sugar"})])
    await Assistant(llm=llm, judge=_StaticJudge(judge.conflict("preferences.coffee"))).run(
        _ctx("actually milk and sugar"))
    assert "confirm" in llm.results[0] and "saved" not in llm.results[0].lower()  # a card, not a save
    assert [e.content for e in await store.get_structured("preferences.coffee")] == ["black"]  # untouched


async def test_a_raised_card_suppresses_the_assistant_message(tmp_path):
    # a card (a conflict, or a fact confirmation) IS the whole interaction: the assistant
    # posts NO message, so the user sees only Yes/No — never a chatty pre-claim.
    from ken.consent import judge
    await _setup(tmp_path)
    kept = await Assistant(
        llm=ToolCallingLLM([("remember_preference", {"topic": "coffee", "detail": "black"})]),
        judge=_StaticJudge(judge.keep())).run(_ctx("I take it black"))
    assert kept is not None                       # a plain, cardless save still speaks
    conflict = await Assistant(
        llm=ToolCallingLLM([("remember_preference", {"topic": "coffee", "detail": "milk"})]),
        judge=_StaticJudge(judge.conflict("preferences.coffee"))).run(_ctx("actually milk"))
    assert conflict.text == "" and conflict.body["silent"]          # a conflict card → silent
    assert conflict.body["result"][0]["outcome"] == "conflict"       # …but the result is data
    fact = await Assistant(
        llm=ToolCallingLLM([("remember_fact", {"topic": "dog", "fact": "your dog is Rex"})]),
        judge=_StaticJudge(judge.keep())).run(_ctx("my dog is Rex"))
    assert fact.text == "" and fact.body["silent"]                   # a fact confirmation card → silent


async def test_empty_model_reply_relays_the_capture_outcome(tmp_path):
    # a reasoning model sometimes runs a capture tool but returns NO final text — the
    # assistant must relay the real outcome, not the misleading "I didn't catch that".
    from ken.consent import judge
    await _setup(tmp_path)
    out = await Assistant(
        llm=ToolCallingLLM([("add_task", {"text": "buy milk"})], reply=""),
        judge=_StaticJudge(judge.keep())).run(_ctx("add buy milk"))
    assert out is not None and "catch that" not in out.text.lower()
    assert "added" in out.text.lower()             # the true outcome, relayed
    # a genuinely empty turn (no tool ran) still gets the honest "didn't catch that"
    out2 = await Assistant(llm=ToolCallingLLM([], reply="")).run(_ctx("uh"))
    assert "catch that" in out2.text.lower()


async def test_assistant_relays_a_judge_error_without_writing(tmp_path):
    from ken.consent.judge import JudgeError
    await _setup(tmp_path)

    class _Boom:
        async def __call__(self, proposal, *, message, existing):
            raise JudgeError("down")

    llm = ToolCallingLLM([("add_task", {"text": "book flights"})])
    await Assistant(llm=llm, judge=_Boom()).run(_ctx("book flights"))
    assert await store.open_tasks() == []                         # nothing written
    assert "error" in llm.results[0].lower()                      # honest, not a fake save


# ── the proactive brief (woken by the schedule, no message) ──────────────────

# ── degenerate paths ─────────────────────────────────────────────────────────

async def test_no_message_is_quiet(tmp_path):
    await _setup(tmp_path)
    assert await Assistant(llm=FakeLLM()).run(_ctx("")) is None


async def test_no_key_still_runs_as_a_stub(tmp_path):
    await _setup(tmp_path)
    adv = await Assistant(llm=None).run(_ctx("hello"))
    assert adv.body["outcome"] == "no_model" and "ken init" in adv.text   # never a fake echo


# ── a report on a task: framed for the model, evidence on the task, typed result ─────

async def test_a_report_on_a_task_frames_the_turn_and_returns_a_typed_result(tmp_path):
    from ken.util import now_ms
    await _setup(tmp_path)
    tid, _ = await store.add_task("decide on coffee beans for Spain", Provenance.USER_STATED)
    llm = ToolCallingLLM([("log_event", {"text": "ordered 2 Kenyan AA from a Valencia shop"}),
                          ("complete_task", {"task": "coffee beans for Spain"})],
                         reply="Done: ordered and closed.")
    ctx = _ctx("(via claude-code) found 3 shops in Valencia, ordered 2 Kenyan AA", now=now_ms())
    ctx.task_id = tid
    adv = await Assistant(llm=llm).run(ctx)
    assert "REPORT on the task 'decide on coffee beans for Spain'" in llm.seen_messages[-1]["content"]
    assert adv.body["result"] == [
        {"tool": "log_event", "outcome": "event", "text": "logged that"},
        {"tool": "complete_task", "outcome": "completed", "text": "marked done"}]
    assert adv.body["summary"] == "Done: ordered and closed." and adv.text == adv.body["summary"]
    (ev,) = await store.recent_events(5)
    assert tid in {e.ref for e in ev.evidence} and "report" in {e.source for e in ev.evidence}
    assert await store.open_tasks() == []
