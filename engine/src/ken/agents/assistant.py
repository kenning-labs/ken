"""The assistant — the front door and Ken's reference agent. (AGENTS.md)

The universal, connector-free agent that exercises the engine's core paradigms so
any plugin author can read it as the canonical example: it READS its memory (the
declared slice), CAPTURES durable facts through the consent gate (proposing, never
writing directly), and holds a grounded conversation. It is deliberately
UNOPINIONATED — generic primitives (tasks, preferences, facts, people), no methodology, no
habits, no coaching — so it is a useful floor for any user and never steps on a
specialised agent someone builds on top.

With an LLM it reasons and captures; without a key it answers outcome no_model, so
a self-hoster with no key still runs.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime

from .. import config
from ..captures import Captures
from ..contracts import RELS, Evidence
from ..tools.base import Tool
from ..util import GLUE, family_words, title_of, when_label
from . import assistant_view
from .base import Advice, Agent, OnMessage, OnSchedule, Slice

log = logging.getLogger(__name__)


def _parse_when(when: str) -> int | None:
    """A local ISO 8601 datetime from the model (e.g. 2026-08-17T15:00) -> ms epoch.
    Empty or unparseable -> None (the task is simply un-timed, never an error)."""
    if not when:
        return None
    try:
        return int(datetime.fromisoformat(when).timestamp() * 1000)   # naive -> local, by design
    except (ValueError, TypeError):                                   # bad string OR a non-string
        return None

# the gate's internal outcome tokens -> what the model may tell the user. Anything
# not here (a denial, a future outcome) becomes an honest failure, never a claimed
# save and never a leaked internal token. The judge adds routes the model must never
# narrate as a plain "saved": collapsed (a duplicate — nothing new stored), conflict (a
# confirm card is up, not yet saved), and judge_error (the check failed, so nothing was
# written). The judge NEVER drops a write, so there is no "filtered" outcome.
_LINKS_SCHEMA = {"type": "array", "description": "records this one is connected to, that Ken already has",
                 "items": {"type": "object", "properties": {
                     "kind": {"type": "string", "enum": ["person", "project", "asset"]},
                     "name": {"type": "string"},
                     "rel": {"type": "string", "enum": list(RELS)}},
                     "required": ["kind", "name", "rel"]}}

_SYSTEM = (
    "You are kip, a calm, warm personal companion with memory. You are shown what you "
    "already know about the user — their open tasks, the preferences and facts they have "
    "told you, the people in their world, and the projects they're tracking. That memory ALREADY "
    "EXISTS: use it to answer, and "
    "NEVER call a capture tool to re-add or restate something already shown. Capture only NEW "
    "information from the user's latest message.\n"
    "- Classify what the user tells you into ONE of five kinds before you capture it:\n"
    "  * A FUTURE thing the USER will do or must get done -> add_task (e.g. 'dentist tomorrow 3pm', "
    "'remind me to call the bank'). See the add_task rule below: most chat is NOT a task.\n"
    "  * A PAST happening, already done or experienced -> log_event (e.g. 'I saw the "
    "dentist today', 'had lunch with Sarah', 'went for a run this morning'). Use "
    "log_event, NOT add_task, for anything that already happened.\n"
    "  * An UPCOMING dated happening that is CONTEXT, not a to-do — a trip, a deadline, "
    "someone's leave, a launch (e.g. 'Alex leaves for Portugal on the 26th') -> log_event with "
    "`when` = that date and `until` = when it stops mattering. NOT add_task unless the user "
    "must do something about it.\n"
    "  * A CURRENT, durable fact or preference about the user -> remember_fact / "
    "remember_preference (e.g. 'my dentist is Dr. Smith', 'I like mornings').\n"
    "  * WHO A WORK/SOCIAL PERSON in their world is -> remember_person (e.g. 'Dana is my "
    "manager', 'my cofounder is Sam', 'Priya leads design'). A colleague, manager, "
    "collaborator, or contact — NOT the user, and NOT family (a family fact like 'my "
    "son's name is Leo' is remember_fact).\n"
    "  * A DOCUMENT, SHEET, REPO or DASHBOARD they work in, named with its url -> remember_asset, "
    "linked to its project.\n"
    "  * A PROJECT or initiative they are tracking -> remember_project (e.g. 'the Larder "
    "rollout is a project', 'we're calling it model v4'). A body of work — not a person, not "
    "a task, not a fact about the user.\n"
    "  BUT if what they finished is an EXISTING open task on their list, use complete_task "
    "to mark it done — NOT log_event. log_event is only for a standalone happening that "
    "was not a tracked task.\n"
    "- To remember something durable you MUST call a capture tool. NEVER say you saved, "
    "added, noted, or will remember something unless the tool's result confirmed it. Every "
    "capture tool RETURNS the true outcome — read it and tell the user exactly that. A "
    "capture can come back as a duplicate (nothing new stored), or as differing from what "
    "you already know (a confirm card goes up, so it is NOT saved yet) — in those cases "
    "never claim you saved it; relay what the tool said.\n"
    "- Your tools are the ONLY actions you can take. If the user asks for something no tool "
    "of yours can do — for example FORGETTING or DELETING something from memory — tell them "
    "plainly you cannot do that yet; NEVER say you did. Report only an action a tool you "
    "actually called performed.\n"
    "- A task WITH a time gets a nudge here in the app at that time — so for a timed task "
    "you may say you'll remind them then. A task with NO time just sits on their list "
    "(no nudge) — do NOT promise to remind them at a time you didn't set; ask for the "
    "time or say it's on their list. You still cannot email, text, or push.\n"
    "- add_task: an actionable to-do; pass `when` (a local ISO 8601 datetime like "
    "2026-08-17T15:00) ONLY if the user gave a specific time, else omit it. A date goes in "
    "`when`, NEVER in `text`: text 'Order floor tiles', when '2026-03-20' — not 'Order floor "
    "tiles by March 20'. "
    "remember_preference: how the "
    "user likes things — a NEW preference is saved quietly and immediately with no "
    "confirmation, so do not tell the user to confirm one; but a preference that DIFFERS "
    "from one you already hold puts up a Yes/No card and is NOT saved until they tap Yes, "
    "so then say you have asked them to confirm, never that you saved it — the tool's "
    "result tells you which happened. remember_fact: a personal fact — pass a "
    "complete statement as `fact` (e.g. 'your son's name is Leo'); do NOT ask them "
    "to confirm in words, just call it — a Yes/No card appears and the fact is saved ONLY "
    "after they tap Yes, so until then say you've asked them to confirm, never that you "
    "saved it. remember_person: WHO a WORK/SOCIAL person in their world is (a colleague, "
    "manager, collaborator, contact — not family) — pass the person's `name` and a COMPLETE "
    "`note` (e.g. 'my manager; owns hiring'); like remember_fact it puts up a card to confirm "
    "or edit and is NOT saved until they tap Yes, so say you've asked them to confirm, never "
    "that you saved it. remember_project: a body of WORK the user is tracking — pass the project "
    "`name` and a COMPLETE `note`; a confirm/edit card goes up, saved only after they tap Yes, so "
    "never claim a save until then. complete_task: ONLY when the user clearly FINISHED a task. discard_task: when a task was NOT done and stopped "
    "mattering (cancelled, no longer needed, a duplicate, a mistake), by the user's word or a report's, with the "
    "reason; never mark such a task done; "
    "pass the task's text. Never complete a task they did not mention. "
    "add_task `owner` / assign_task: who DOES the task — omit for the user's own; a person's name when "
    "the user hands it to someone ('Robert will send the contract'); assign_task moves an existing task. "
    "link_tasks: when the user says one task can only start after another ('X waits on Y', 'X only after "
    "Y'), record it as a dependency between the two tasks; add a missing task first only if the user "
    "committed to it; remove=true when it no longer waits. "
    "reschedule_task: change the time of a task the user already has — pass its text (or a "
    "distinctive part) and the new local ISO 8601 datetime; use this, not add_task, to MOVE "
    "an existing timed task. find_task: look up a task by text INCLUDING completed ones — "
    "your open-task list shows only open tasks, so use find_task to check whether something "
    "is already done; it reads only, it never changes anything. "
    "log_event: record something that HAPPENED — a past occurrence — with an optional "
    "`when` (local ISO 8601); omit `when` for just now. Events are append-only history, "
    "saved immediately with no confirmation card. find_events: look up past events by "
    "text; it reads only, it never changes anything.\n"
    "- When a person, project or asset is described TOGETHER with records Ken already has ('Robert "
    "is the designer on Recipes'), pass `links` so the connection is stored, not just the prose.\n"
    "- A PASTED document, plan, summary or thread is INPUT TO CAPTURE, never just context to hold: "
    "who owns what -> remember_person with links; the user's own assignments, targets and "
    "priorities -> remember_project (update the project's note) or remember_fact; a to-do the "
    "user committed to (the add_task rule below holds here too) -> add_task; a decision or outcome -> log_event. Say 'got it' only after "
    "the captures ran.\n"
    "- add_task ONLY when the USER states something THEY will do or must get done: a commitment, "
    "a plan, a request they accepted, 'remind me'. NEVER add a task for: a topic discussed; advice "
    "or steps YOU or any assistant suggested (even when the user repeats or agrees with them); a "
    "hypothetical; what ANOTHER person must do (a friend's deadline is not the user's task); a "
    "how-to or help question (coding, a recipe). A chat where the user gets HELP with a problem "
    "(debugging code, cooking, a how-to) yields NO task at all, not even their own 'I'll update the "
    "fixture and add a test case' about that problem: it is the fix in progress, not a to-do, unless "
    "they ask you to track or remind them. Put in the task only the action the user said; never add a step they "
    "did not state (they said 'I'll find the file': not 'find the file and send it to Ines'). When "
    "unsure, do not add the task.\n"
    "- Capture only what is clearly durable and worth keeping — not every passing remark. "
    "When in doubt, just reply; do not capture.\n"
    "- If a message clarifies that a NAME is a place, company, product or thing — not a "
    "person — remember_fact it under that name (topic 'Portugal', fact 'Portugal is a country; the "
    "travel destination'), so it is never asked about again.\n"
    "- Answer in a sentence or two. Be direct and warm. Never invent facts about the user."
)

_GIST = 110          # an index line: the first sentence of a note, clipped
_OPEN_MAX = 5        # records the message names, opened in full before the model runs
_OPEN_NOTE = 1500    # the most of one opened note


def _gist(text: str, cap: int = _GIST) -> str:
    """The first sentence of a note, as the one line the index shows."""
    first = re.split(r"(?<=[.!?])\s|\n", str(text or "").strip(), maxsplit=1)[0]
    return first if len(first) <= cap else first[:cap - 1].rstrip() + "…"


def _name(e) -> str:
    return title_of(e.kind).title() if e.kind.startswith(("people.", "projects.", "assets.")) else title_of(e.kind)


def _named(entries: list, msg: str) -> list:
    """The records a message names, by code: a person by any word of the name, anything else by
    every distinctive word of its title. Words every third title shares, and glue words, never count.
    ponytail: whole-word match on names; no aliases, no stemming."""
    msg = re.sub(r"^\(via [^)]*\)\s*", "", msg)        # the source tag names the sender, not the subject
    words = set(re.findall(r"[a-z0-9']{3,}", msg.lower()))
    out = []
    for prefix in ("people.", "projects.", "assets.", "facts.", "preferences."):
        group = [e for e in entries if e.kind.startswith(prefix)]
        skip = GLUE | family_words([title_of(e.kind) for e in group])
        for e in group:
            own = [w for w in title_of(e.kind).lower().split() if len(w) > 2 and w not in skip]
            if own and (any(w in words for w in own) if prefix == "people." else all(w in words for w in own)):
                out.append(e)
    return out[:_OPEN_MAX]


def _lookup(entries: list, name: str) -> list[dict]:
    """open_record: the full note of the records a name matches, best match first."""
    q = [w for w in re.findall(r"[a-z0-9']{2,}", str(name).lower())]
    scored = []
    for e in entries:
        title = title_of(e.kind).lower().split()
        hit = sum(1 for w in q if w in title)
        if hit:
            scored.append((-(hit / max(len(q), 1)), -hit, e))
    scored.sort(key=lambda x: x[:2])
    return [{"kind": e.kind, "name": _name(e), "note": e.content} for _, _, e in scored[:3]]


def _search(entries: list, query: str) -> list[dict]:
    """search_records: the records whose NOTE holds the query's words, most words first, each with
    the passage around the first hit. For when the first line of a note hides what the note holds.
    ponytail: plain substring count over a few hundred notes; an index if a world outgrows that."""
    words = [w for w in re.findall(r"[a-z0-9'-]{3,}", str(query).lower()) if w not in GLUE]
    scored = []
    for e in entries:
        low = e.content.lower()
        hits = [w for w in words if w in low]
        if hits:
            at = min(low.find(w) for w in hits)
            scored.append((-len(hits), e, e.content[max(0, at - 80):at + 240].strip()))
    scored.sort(key=lambda x: x[0])
    return [{"kind": e.kind, "name": _name(e), "passage": passage} for _, e, passage in scored[:5]]


def _render_memory(ctx, *, mode: str = "full", msg: str = "") -> str:
    """The declared slice, as plain text the model reads each turn — the memory-read
    paradigm. Tasks are listed by text (completion matches on text, not position).

    mode "full": every note in full (the original behaviour).
    mode "index": the skills pattern. Open tasks and preferences in full (the work, and standing
    instructions); every other record is ONE line, the first sentence of its note. The records
    the message names are opened in full by code; the rest the model opens with open_record."""
    index = mode == "index"
    lines: list[str] = []
    if ctx.tasks:
        lines.append("Open tasks:")
        lines += [f"  - {t.text}" + (f" (scheduled {when_label(t.deadline, ctx.now)})"
                                     if t.deadline else "") for t in ctx.tasks]
    if ctx.structured:
        # people and projects are their own views (de-slugged names), so 'who is X?' / 'what's
        # the state of Y?' read clean rosters — not mixed anonymously among the user's own facts.
        facts = [e for e in ctx.structured
                 if not e.kind.startswith(("people.", "projects."))]
        if facts:
            lines.append("What you know about the user:")
            lines += [f"  - {e.kind.split('.', 1)[-1]}: "
                      f"{_gist(e.content) if index and not e.kind.startswith('preferences.') else e.content}"
                      for e in facts]
        for prefix, heading in (("people.", "People in your world:"),
                                ("projects.", "Projects you're tracking:")):
            named = [e for e in ctx.structured if e.kind.startswith(prefix)]
            if named:
                lines.append(heading)
                lines += [f"  - {e.kind.split('.', 1)[-1].replace('_', ' ').title()}: "
                          f"{_gist(e.content) if index else e.content}" for e in named]
        if index:
            lines.insert(0, "Each record below shows only the FIRST LINE of its note. open_record(name) "
                            "returns the full note: call it before you answer in detail about a record, "
                            "or update one, when its full note is not shown under 'Opened for this message'. "
                            "A first line can hide what a note holds: when no line answers the question, "
                            "search_records(words) finds the notes that contain those words. Say that "
                            "something is not recorded only after a search found nothing.")
            opened = _named(ctx.structured, msg)
            if opened:
                lines.append("Opened for this message (full notes):")
                lines += [f"  - {_name(e)} [{e.kind.split('.', 1)[0]}]: {e.content[:_OPEN_NOTE]}" for e in opened]
    if ctx.events:
        # a bounded tail of the append-only log — recent happenings, newest first — so
        # the model can answer "what did I do lately?" without a tool call.
        lines.append("Recent events:")
        lines += [f"  - {_gist(e.summary, 160) if index else e.summary} ({when_label(e.at, ctx.now)})"
                  for e in ctx.events[:5]]
    return "\n".join(lines) if lines else "(nothing saved yet)"


_REFRESH_CRON = "*/15 * * * *"   # republish the Today panel; says nothing



# no model: say so plainly — never an echo that reads as if the report was understood
_NO_MODEL_TEXT = ("Not processed: no model is configured, so Ken cannot read free text. Run `ken init` to "
                  "choose one (or add an `llm:` section to config.yaml); the typed writes (add_task, "
                  "log_event, remember, ...) work without it.")

class Assistant(Agent):
    name = "kip"                         # the companion — its speaker tag, roster, @handle
                                         # (lowercase, matching every other agent handle)
    role = "your companion"
    default_assistant = True             # answers every un-addressed message
    # reactive chat + a quiet refresh so the Today panel's lanes/ages stay live
    triggers = [OnMessage(), OnSchedule(_REFRESH_CRON)]
    slice = Slice(structured=["preferences", "facts", "people", "projects", "assets"], tasks="open",
                  events="recent")
    # Defaults to the reasoning tier: a dogfood A/B showed the fast tier fabricated saves
    # and corrupted tool arguments, which the reasoning tier stops. Config-overridable
    # (overrides/tiers) for cost-sensitive self-hosters. The judge stays fast.
    model_tier = "reasoning"

    def __init__(self, llm=None, judge=None, rank=None) -> None:
        self._llm = llm
        self._judge = judge   # the machine skeptic; None -> the gate applies directly
        self._rank = rank     # a Decisions ranker for the Today lanes; None -> deterministic lanes

    async def act(self, ctx, proposal):
        """The acts kip performs itself after a Yes: mark done, discard, move, assign, link. All
        go through the same fast-track, audited edits the chat tools use — by task id."""
        verb = proposal.intent.split(":", 1)[1]
        meta = proposal.meta or {}
        tid = str(meta.get("task_id", ""))
        if verb == "mark_done":
            out = await ctx.propose.complete(tid)
            return "Marked done." if out == "completed" else "That task is no longer open."
        if verb == "discard":
            # an edit on the card is the user's own reason: it replaces the proposer's
            asked = f"{meta.get('to', '')} \u2014 discard?"
            mine = proposal.content.strip() if proposal.content.strip() != asked else ""
            out = await ctx.propose.discard(tid, mine or str(meta.get("reason", "")))
            return "Discarded. It is not counted as done." if out == "discarded" else "That task is no longer open."
        if verb == "assign":
            out = await ctx.propose.assign(tid, meta.get("owner") or None)
            return "Reassigned." if out == "assigned" else "That task is no longer open."
        if verb == "link_tasks":
            out = await ctx.propose.link(tid, str(meta.get("depends_on", "")), remove=bool(meta.get("remove")))
            return {"linked": "Linked.", "unlinked": "Unlinked.", "cycle": "That would make a cycle, so I left it.",
                    "link_absent": "It does not wait on that."}.get(out, "That task is no longer open.")
        if verb == "reschedule":
            at = int(meta.get("at", 0))
            if at <= ctx.now and meta.get("origin") == "maintainer":
                from ..maintain import tomorrow_at  # a late Yes on a stale card: the next 09:00
                at = tomorrow_at(ctx.now)
            out = await ctx.propose.reschedule(tid, at)
            if out == "rescheduled":
                return f"Moved to {when_label(at, ctx.now)}."
            if out == "reschedule_invalid":
                return "That time has already passed, so I left the task where it was."
            return "That task is no longer open."
        raise NotImplementedError(verb)

    @staticmethod
    async def _trail(ctx, kind: str, reason: str) -> None:
        if ctx.audit is not None:
            await ctx.audit(kind, reason)

    async def run(self, ctx):
        msg = (ctx.message or "").strip()
        if ctx.woke_by == "schedule" and ctx.cron == _REFRESH_CRON:
            out = None                              # the refresh wake: only the panel below
        elif not msg:
            out = None                              # an empty wake says nothing
        elif self._llm is None:
            out = Advice(text=_NO_MODEL_TEXT, body={"outcome": "no_model"})
        else:
            out = await self._chat(ctx, msg)

        # the Today panel is the Assistant's OWN standing state: ctx.view publishes to
        # the agent's own tab (self.session) — NOT ctx.conversation_id, which may be a
        # foreign session the Assistant is answering as the misroute fallback. Rebuilt
        # from FRESH tasks (ctx.memory is live) so a task captured this turn shows at
        # once. Best-effort: a panel glitch never breaks the reply.
        try:
            tasks = await ctx.memory.open_tasks()
            scores = ranked_at = None
            untimed = [t.text for t in tasks if not t.deadline]
            if self._rank and untimed:            # Decisions sorts the untimed backlog into lanes
                scores = await self._rank(untimed, instructions=assistant_view.RUBRIC_ASK,
                                          rubric=assistant_view.RUBRIC)
                if scores is not None and len(scores) != len(untimed):
                    scores = None                 # a short/odd answer → the deterministic floor
                ranked_at = ctx.now if scores is not None else None
            week_ago = ctx.now - 7 * 86_400_000
            done = await ctx.memory.done_since(week_ago)
            events = await ctx.memory.recent_events(200)      # the week, not the 10-row preload
            await ctx.view.publish(assistant_view.build_view(
                tasks, ctx.now, events=events, done=done, scores=scores, ranked_at=ranked_at))
        except Exception:  # noqa: BLE001, S110 — a panel is best-effort, never the reply
            pass
        return out

    async def _chat(self, ctx, msg):
        # the evidence behind every capture this turn: the inbound action (by id, so a reader
        # can find the exact message) and who sent it — an agent's report is not the user's word
        evidence = [Evidence(ref=getattr(ctx, "ref", "") or getattr(ctx, "action_id", "") or msg,
                             source=getattr(ctx, "source", "") or "user")]

        # capture tools: each PROPOSES; the gate is the single applier. The model
        # never writes memory directly — it routes through the judge and consent, and
        # the tool relays the TRUE outcome so the reply can never claim a save that was
        # collapsed, raised as a conflict, still pending, or blocked by a judge error.
        #
        # When a consent card goes up (a conflict, or a fact confirmation), the CARD is
        # the whole interaction: the assistant stays SILENT afterwards, so the user sees
        # only Yes/No — never a chatty reply that pre-claims the result ("I've updated…")
        # before they tap.
        card_raised = False
        # A reasoning model sometimes runs a capture tool but returns NO final text.
        # Every capture's TRUE outcome is recorded (name, result) so the reply can relay it
        # honestly — and so the turn's result is data (ReconcileResult), never model prose.
        results: list[dict] = []
        cap = Captures(ctx, judge=self._judge, message=msg, evidence=evidence,
                       provenance=getattr(ctx, "provenance", None), parse_when=_parse_when)
        if ctx.task_id:
            # a report on a task: what happened is evidence FOR that task (ref = task id)
            evidence.append(Evidence(ref=ctx.task_id, source="report"))

        async def _remember_preference(topic: str, detail: str):
            return await cap.remember("preference", topic, detail)

        async def _remember_fact(topic: str, fact: str):
            return await cap.remember("fact", topic, fact)

        async def _remember_person(name: str, note: str, links=None):
            return await cap.remember("person", name, note, links=links)

        async def _remember_project(name: str, note: str, links=None):
            return await cap.remember("project", name, note, links=links)

        async def _remember_asset(name: str, url: str, note: str, links=None):
            return await cap.remember("asset", name, note, links=links, url=url)

        def _rec(name: str, fn):
            # record each capture's outcome, so the empty-reply fallback can be honest and
            # the turn returns a typed result list; a raised card silences the reply
            async def wrapped(**kw):
                nonlocal card_raised
                r = await fn(**kw)
                if r.outcome in ("pending", "conflict"):
                    card_raised = True
                results.append({"tool": name, "outcome": r.outcome, "text": r.text})
                return r.text
            return wrapped

        tools = [
            Tool("add_task",
                 "Record a to-do the USER committed to (not advice, a help-chat step, or another "
                 "person's task). Pass `when` as a local ISO 8601 datetime "
                 "(e.g. 2026-08-17T15:00) ONLY if the user named a specific time; omit otherwise. "
                 "Put a deadline in `when`, never in `text` (no 'by March 20' in the text). "
                 "Pass `project` = the name of one of the user's projects ONLY when the to-do "
                 "clearly belongs to it; omit when unsure.",
                 {"type": "object", "properties": {
                     "text": {"type": "string"}, "when": {"type": "string"}, "project": {"type": "string"},
                     "owner": {"type": "string"}},
                  "required": ["text"]}, _rec("add_task", cap.add_task)),
            Tool("assign_task",
                 "Change who does an EXISTING task: `owner` = a person's name, or empty for the user. "
                 "Pass the task's text (or a distinctive part).",
                 {"type": "object", "properties": {"task": {"type": "string"}, "owner": {"type": "string"}},
                  "required": ["task", "owner"]}, _rec("assign_task", cap.assign_task)),
            Tool("remember_preference",
                 "Save how the user likes things — a standing taste, habit or rule (saved quietly, "
                 "no confirmation). NOT what they own, bought or did: that is remember_fact or "
                 "log_event. topic is a short key, detail is the preference.",
                 {"type": "object", "properties": {
                     "topic": {"type": "string"}, "detail": {"type": "string"}},
                  "required": ["topic", "detail"]}, _rec("remember_preference", _remember_preference)),
            Tool("remember_fact",
                 "Save a personal fact about the user; they are shown a Yes/No card to "
                 "confirm. topic is a short key; fact is the COMPLETE statement to confirm "
                 "and store, e.g. \"your son's name is Leo\".",
                 {"type": "object", "properties": {
                     "topic": {"type": "string"}, "fact": {"type": "string"}},
                  "required": ["topic", "fact"]}, _rec("remember_fact", _remember_fact)),
            Tool("remember_person",
                 "Save WHO A WORK OR SOCIAL PERSON in the user's world is, when the user tells you "
                 "(e.g. 'Dana is my manager', 'my cofounder is Sam', 'Priya leads design'). "
                 "They are shown a card to confirm or edit. name is the person; note is the "
                 "COMPLETE description to store, e.g. \"my manager; owns hiring\". Use this for a "
                 "colleague/manager/collaborator/contact — NOT for the user, and NOT for family "
                 "(a family fact like a child's name is remember_fact). Don't volunteer people the "
                 "user didn't mention.",
                 {"type": "object", "properties": {
                     "name": {"type": "string"}, "note": {"type": "string"}, "links": _LINKS_SCHEMA},
                  "required": ["name", "note"]}, _rec("remember_person", _remember_person)),
            Tool("remember_asset",
                 "Save a DOCUMENT, SHEET, REPO, DASHBOARD or other link the user works with, when they "
                 "name one with its url. A card confirms. name = what they call it; url = the link; note = "
                 "what it is; links = its project ({kind: 'project', name, rel: 'part_of'}).",
                 {"type": "object", "properties": {
                     "name": {"type": "string"}, "url": {"type": "string"}, "note": {"type": "string"},
                     "links": _LINKS_SCHEMA},
                  "required": ["name", "url", "note"]}, _rec("remember_asset", _remember_asset)),
            Tool("remember_project",
                 "Save a PROJECT or initiative the user is tracking, when they name one "
                 "(e.g. 'the Larder rollout is a project', 'we're calling it model v4'). They are "
                 "shown a card to confirm or edit. name is the project; note is the COMPLETE "
                 "description of the WORK, e.g. \"ship Larder pre-order to Android; in rollout/QA\". "
                 "A project is a body of work, not a person or a user-fact.",
                 {"type": "object", "properties": {
                     "name": {"type": "string"}, "note": {"type": "string"}, "links": _LINKS_SCHEMA},
                  "required": ["name", "note"]}, _rec("remember_project", _remember_project)),
            Tool("complete_task",
                 "Mark an open task done. Pass the task's text, or a distinctive part of "
                 "it — never a number.",
                 {"type": "object", "properties": {"task": {"type": "string"}},
                  "required": ["task"]}, _rec("complete_task", cap.complete_task)),
            Tool("discard_task",
                 "Drop an open task that was NOT done: cancelled, no longer needed, a duplicate, or added "
                 "by mistake. It closes as dropped, not as done. Pass the task's text (or a distinctive "
                 "part) and `reason` = why, in a few words.",
                 {"type": "object", "properties": {"task": {"type": "string"}, "reason": {"type": "string"}},
                  "required": ["task"]}, _rec("discard_task", cap.discard_task)),
            Tool("reschedule_task",
                 "Change the time of an EXISTING task. Pass the task's text (or a "
                 "distinctive part) and the new time as a local ISO 8601 datetime "
                 "(e.g. 2026-08-17T18:00).",
                 {"type": "object", "properties": {
                     "task": {"type": "string"}, "when": {"type": "string"}},
                  "required": ["task", "when"]}, _rec("reschedule_task", cap.reschedule_task)),
            Tool("link_tasks",
                 "Record that an EXISTING task waits on another: `task` can only start after `depends_on` "
                 "(each the task's text or a distinctive part). It reads as blocked until that one is closed. "
                 "`remove`: true drops the dependency. A cycle is refused.",
                 {"type": "object", "properties": {
                     "task": {"type": "string"}, "depends_on": {"type": "string"}, "remove": {"type": "boolean"}},
                  "required": ["task", "depends_on"]}, _rec("link_tasks", cap.link_tasks)),
            Tool("find_task",
                 "Look up a task by text, INCLUDING completed ones (your open-task list "
                 "already shows only open ones). Use it to check whether something is on "
                 "the list or already done. Returns each match's text, status, and time.",
                 {"type": "object", "properties": {"query": {"type": "string"}},
                  "required": ["query"]}, _rec("find_task", cap.find_task)),
            Tool("log_event",
                 "Record something that HAPPENED (a past occurrence), or an UPCOMING dated happening "
                 "that is context rather than a to-do (a trip, a deadline, a launch). Use this, NOT "
                 "add_task, for things already done/experienced. `when` = when it happens or happened "
                 "(local ISO 8601, e.g. 2026-08-17T15:00); omit for just now. `until` = when it "
                 "stops mattering (an offer, a trip, a deadline), same format; omit when it is history.",
                 {"type": "object", "properties": {
                     "text": {"type": "string"}, "when": {"type": "string"}, "until": {"type": "string"}},
                  "required": ["text"]}, _rec("log_event", cap.log_event)),
            Tool("find_events",
                 "Look up past events by text — the append-only log of things that "
                 "happened. Use it to check when something occurred. Returns each match's "
                 "summary and time. Read-only; it never changes anything.",
                 {"type": "object", "properties": {"query": {"type": "string"}},
                  "required": ["query"]}, _rec("find_events", cap.find_events)),
        ]

        # `assistant.preload: index` = the skills pattern: one line per record, full notes on demand
        mode = "index" if str(config.section("assistant").get("preload") or "full").lower() == "index" else "full"
        if mode == "index":
            entries = list(ctx.structured or [])

            def _open_record(name: str):
                found = _lookup(entries, name)
                return found or {"error": f"no record named '{name}'"}
            tools.append(Tool(
                "open_record",
                "Read the FULL note of a person, project, asset or fact by name. Your context lists only "
                "the first line of each. Read-only; it never changes anything.",
                {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                _open_record))
            tools.append(Tool(
                "search_records",
                "Find records by words INSIDE their notes (not only their names). Returns each match with "
                "the passage around the words. Use distinctive words from the question. Read-only.",
                {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                lambda query: _search(entries, query) or {"result": "no note contains those words"}))

        # history already ends with this same turn (built from the not-yet-handled
        # inbox row). Fold the memory into that last user turn — replace it rather
        # than exact-match strip it, so a normalized message can't duplicate the turn.
        framed = msg
        if ctx.task_id:
            task = next((t for t in await ctx.memory.open_tasks() if t.id == ctx.task_id), None)
            framed = (f"This is a REPORT on the task '{task.text if task else ctx.task_id}'. Record what "
                      "happened with log_event. Then complete_task ONLY if the report states the task's "
                      "outcome has HAPPENED (decided, delivered, finished). Research, findings, options, "
                      "a shortlist or a recommendation are NOT the outcome: the task stays open. If the "
                      "report says the task no longer matters (cancelled, a duplicate, not needed) and was "
                      "NOT done, discard_task it with that reason, never complete_task. If it "
                      "names a new time, reschedule_task it; if it names follow-up work, add_task it. "
                      f"The report: {msg}")
        folded = {"role": "user", "content": f"{_render_memory(ctx, mode=mode, msg=msg)}\n\n{framed}"}
        history = ctx.history
        prior = history[:-1] if history and history[-1]["role"] == "user" else history
        messages = [*prior, folded]

        # give the model the current local date/time so it can resolve "3pm tomorrow"
        now_local = datetime.fromtimestamp(ctx.now / 1000)  # noqa: DTZ006 — local, by design
        system = f"{_SYSTEM}\nThe current local date and time is {now_local:%A %Y-%m-%d %H:%M}."
        if ctx.said_at is not None:     # an older message: its relative dates are ITS, not today's
            said = datetime.fromtimestamp(ctx.said_at / 1000)  # noqa: DTZ006 — local, by design
            system += (f"\nThis message was said on {said:%A %Y-%m-%d %H:%M}: resolve every relative "
                       "date in it ('tomorrow', 'the 14th', 'next Friday', 'this week') against that "
                       "date, not today.")
        try:
            reply = (await self._llm.complete(system, messages, tools=tools)).strip()
        except Exception:  # a model error still owes the user a reply
            log.exception("assistant: the model call failed")   # the reason, not just the apology
            if card_raised:
                return Advice(text="", body={"result": results, "summary": "", "silent": True})
            return Advice(text="I couldn't reach the model — nothing was saved.", error=True,
                          body={"result": results, "summary": ""})
        body = {"result": results, "summary": reply}
        # a card is up → say nothing in the stream; the Yes/No card is the whole reply. The
        # typed result still travels, flagged `silent`, so an MCP caller gets its card ids
        # while the stream and the model's history show nothing but the card.
        if card_raised:
            return Advice(text="", body={**body, "silent": True})
        if reply:
            return Advice(text=reply, body=body)
        if results:
            # the model ran a capture but produced no text — relay the TRUE outcome,
            # never a misleading "I didn't catch that".
            last = results[-1]["text"]
            return Advice(text=last[0].upper() + last[1:], body=body)
        return Advice(text="I didn't catch that — could you rephrase?", body=body)
