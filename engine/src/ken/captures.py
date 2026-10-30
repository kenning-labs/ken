"""The six writes, as one module with two callers: kip's tool loop (the reconciler) and the
MCP typed tools. Each PROPOSES through the gate — the single applier — and returns the TRUE
outcome, so no caller can ever claim a save that was collapsed, carded, or refused.

`CaptureResult.text` is the honest one-line relay for a model or a user; `.outcome` is the
gate's token for code. Card ids are NOT here: the gate stamps every card with the action
that raised it, and the engine attaches `gate.cards_for(action_id)` to the reply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from . import db as _db
from .agents.base import Proposal
from .contracts import NOUN, RELS, Evidence, Provenance
from .memory import store
from .util import (
    GLUE,
    family_words,
    gone,
    local_ms,
    norm,
    now_ms,
    owner_label,
    slug,
    split_deadline,
    title_of,
    when_label,
)

OUTCOME = {"task": "added to your tasks", "task_exists": "already on your list",
           "applied": "saved", "pending": "not saved yet — a confirmation card is up",
           "collapsed": "already saved — nothing new to add",
           "conflict": "that differs from what I have — I've put up a card to confirm",
           "judge_error": "I hit an error checking that, so I didn't save it",
           "denied": "I can't record that from this source",
           "retired": "forgotten", "absent": "nothing saved under that name"}
BATCH = 10                       # ponytail: a guess; the first real sweep sets it
_DAY = 86_400_000


def _title_of(entry) -> str:
    subject = getattr(entry, "subject", "") or ""
    return subject if subject not in ("", "you") else entry.kind.split(".", 1)[-1].replace("_", " ")
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_SLOT_MINUTES = 24 * 60
_NS = {"preference": "preferences", "preferences": "preferences", "fact": "facts", "facts": "facts",
       "person": "people", "people": "people", "project": "projects", "projects": "projects",
       "asset": "assets", "assets": "assets"}
_KINDS = "person, project, preference, fact or asset"
_ME = {"", "me", "i", "you", "user", "the user", "myself", "self"}


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%dT%H:%M")   # noqa: DTZ006 — local, by design


_TASK_VERBS = frozenset({"complete", "finish", "send", "write", "make", "prepare", "prep", "draft", "review",
                         "update", "check", "call", "book", "plan", "start", "ping", "follow"})


def _parse_at(at) -> tuple[int | None, CaptureResult | None]:
    """A write's `at` (local ISO date or datetime: when the change HAPPENED) -> ms, or the
    `invalid` result that says why nothing changed. Empty = None: the write behaves as before.
    A date alone is the END of that day (the change happened some time in it), and no `at` is
    later than now: a future one would make every later dated write read as stale."""
    s = str(at or "").strip()
    if not s:
        return None, None
    try:
        ms = local_ms(s) + (_DAY - 1000 if _DATE_ONLY.match(s) else 0)
        return min(ms, now_ms()), None
    except (ValueError, TypeError):
        return None, CaptureResult(f"`at` must be a local ISO date or datetime like 2026-03-11T09:30, not "
                                   f"{s[:40]!r}; nothing changed", "invalid")


def _state(t) -> str:
    out = [t.status]
    if t.deadline:
        out.append(f"deadline {_iso(t.deadline)}")
    if t.owner:
        out.append(f"owner {owner_label(t.owner)}")
    return ", ".join(out)


def _twin_in(seen: set, ns: str, name: str) -> str | None:
    """A record already in this batch that IS this one: the same slug, or every token of the
    shorter name inside the longer ("Lucas" and "Lucas (Tallyhouse)")."""
    mine = {w for w in slug(name).split("_") if len(w) > 1}
    for s_ns, s_name in seen:
        if s_ns != ns:
            continue
        theirs = {w for w in slug(s_name).split("_") if len(w) > 1}
        small, big = (mine, theirs) if len(mine) <= len(theirs) else (theirs, mine)
        if slug(s_name) == slug(name) or (small and small <= big):
            return s_name
    return None


@dataclass
class CaptureResult:
    text: str                 # the honest relay
    outcome: str              # the gate's token ('task' | 'pending' | 'collapsed' | …)

    def __str__(self) -> str:
        return self.text


class Captures:
    """Bound to one turn: a context (its memory + proposer), the judge, the message that
    is the evidence, and the provenance the caller may write with (already clamped to
    its token's ceiling by the engine)."""

    def __init__(self, ctx, *, judge=None, message: str = "", evidence: list | None = None,
                 provenance: Provenance | None = None, parse_when=None) -> None:
        self.ctx = ctx
        self.judge = judge
        self.message = message
        self.evidence = evidence if evidence is not None else [Evidence(ref=message, source="user")]
        self.provenance = provenance
        self._parse_when = parse_when

    def _prov(self, default: Provenance) -> Provenance:
        return self.provenance or default

    def _meta(self) -> dict:
        # a write made while reporting on a task carries the task id, so its card can be
        # found from get_context(task).questions_for_user
        tid = getattr(self.ctx, "task_id", "")
        return {"task_id": tid} if tid else {}

    async def _propose(self, intent: str, content: str, at: int | None = None,
                       details: dict | None = None, meta: dict | None = None) -> str:
        return await self.ctx.propose.capture(
            Proposal(intent=intent, content=content, at=at, evidence=self.evidence,
                     provenance=self._prov(Provenance.AGENT_INFERRED), meta={**self._meta(), **(meta or {})}),
            judge=self.judge, message=self.message, details=details)

    # ── tasks ──────────────────────────────────────────────────────────────

    async def _stale(self, task: str, at: int | None, *, exact: bool = False):
        """A late, old write never wins: the `stale` result when the task it names (open first,
        else any status, so a done task is found too) changed AFTER `at`. None = go ahead.
        `exact`: match only the whole text (add_task, which never edits a task by a part of it)."""
        if at is None:
            return None
        q = norm(str(task))
        hits = [] if exact else await self._match_open(task)
        if len(hits) > 1:
            return None                               # several match: the ambiguity reply says so, not stale
        if not hits:
            found = [t for t in [await store.task_by_id(str(task).strip())] if t] or await store.find_tasks(str(task), limit=50)
            hits = [t for t in found if norm(t.text) == q or t.id == str(task).strip()] \
                or ([] if exact else [t for t in found if q and q in norm(t.text)][:2])
            hits = hits if len(hits) == 1 else []
        newer = sorted((t for t in hits if (t.changed_at or 0) > at), key=lambda t: -t.changed_at)
        if not newer:
            return None
        t = newer[0]
        await _db.audit("stale", t.id, f"write at {_iso(at)} older than the change at {_iso(t.changed_at)}")
        return CaptureResult(f"stale: '{t.text}' changed later ({_iso(t.changed_at)}) than this update "
                             f"({_iso(at)}), so nothing changed — it stays {_state(t)}", "stale")

    async def _owner(self, name: str) -> tuple[str | None, str]:
        """An owner NAME -> (what to store, what to say). The user (empty, 'me') is None. A person
        Ken holds is their record's kind (the exact slug, else the one name twin, as links resolve);
        anyone else is the name as given, and the reply says so — a task can be someone's before
        that person is saved."""
        name = " ".join(str(name or "").split())[:80]
        if name.lower() in _ME:
            return None, ""
        people = await self.ctx.memory.structured("people")
        kind = f"people.{slug(name)}"
        if any(e.kind == kind for e in people):
            return kind, ""
        twins = [e for e in people if _twin_in({("people", _title_of(e))}, "people", name)]
        if len(twins) == 1:
            return twins[0].kind, ""
        return name, f" — Ken has no person '{name}', so the owner is the name as given"

    def _said(self) -> datetime:
        """When the message was said (a dated tell), else now: what relative dates resolve from."""
        at = getattr(self.ctx, "said_at", None) or self.ctx.now
        return datetime.fromtimestamp(at / 1000)   # noqa: DTZ006 — local, by design

    def _due(self, when: str) -> int | None:
        """A due time from the model. A date alone, or exactly midnight, is a DAY, not a
        moment: it becomes 09:00 that day, so a nudge lands in the morning, never at 00:00."""
        at = self._parse_when(when) if (when and self._parse_when) else None
        if at is None:
            return None
        local = datetime.fromtimestamp(at / 1000)   # noqa: DTZ006 — local, by design
        if _DATE_ONLY.match(when.strip()) or (local.hour, local.minute, local.second) == (0, 0, 0):
            at = int(local.replace(hour=9, minute=0, second=0, microsecond=0).timestamp() * 1000)
        return at

    async def _project(self, name: str) -> tuple[str | None, str]:
        """A project NAME -> (its kind, what to say when it is none). The exact slug, else the ONE
        project whose title holds every distinctive word of the name ('Recipes' -> 'crumbly
        project recipes in app'). Glue, family and generic words ('for', 'crumbly', 'project')
        name nothing. No match, or more than one: no project, and the reply says so — a task
        never invents a project and never guesses between two."""
        name = str(name or "").strip()
        if not name:
            return None, ""
        kinds = [e.kind for e in await self.ctx.memory.structured("projects")]
        if f"projects.{slug(name)}" in kinds:
            return f"projects.{slug(name)}", ""
        skip = GLUE | family_words([title_of(k) for k in kinds]) | {"project", "projects"}
        want = {w for w in slug(name).split("_") if len(w) > 2 and w not in skip}
        hits = [k for k in kinds if want and want <= set(title_of(k).split())]
        if len(hits) == 1:
            return hits[0], ""
        return None, (f" — '{name}' fits more than one project, so none was set" if hits
                      else f" — Ken has no project '{name}', so none was set")

    # ponytail: `when` is the old name of `deadline`; drop the alias once every agent and
    # client sends `deadline` (kip's own tool schema and older clients).
    async def add_task(self, text: str, deadline: str = "", project: str = "", when: str = "",
                       owner: str = "", at: str = "") -> CaptureResult:
        # the text carries no deadline phrase: the date lives in the field. An explicit deadline
        # wins; a phrase in the text fills an empty one, resolved from when the message was said.
        at_ms, bad = _parse_at(at)
        if bad:
            return bad
        text, in_text = split_deadline(str(text), self._said())
        if stale := await self._stale(text, at_ms, exact=True):
            return stale                              # e.g. done since: an older "still open" never reopens it
        deadline = deadline or when or in_text or ""
        when = ""
        kind, said = await self._project(project)
        who, owner_note = await self._owner(owner)
        details: dict = {}
        r = await self._add_task(text, deadline or when, kind, details, owner=who)
        if r.outcome not in ("task", "task_exists", "pending"):
            return r                                  # collapsed or refused: no claim about a project
        if at_ms is not None and r.outcome != "pending":
            match = next((t for t in await self.ctx.memory.open_tasks() if norm(t.text) == norm(text)), None)
            if match:
                await store.stamp_task(match.id, at_ms)
        if who and r.outcome == "task_exists":
            held = next((t.owner for t in await self.ctx.memory.open_tasks() if norm(t.text) == norm(text)), who)
            owner_note = owner_note if held == who else " — it keeps the owner it already has"
        if owner_note:
            r = CaptureResult(r.text + owner_note, r.outcome)
        if kind:
            # say what is TRUE of what was stored: an existing task, or an open card, keeps the project it has
            if r.outcome == "pending":
                held = details.get("project")
            else:
                held = next((t.project for t in await self.ctx.memory.open_tasks() if norm(t.text) == norm(text)), None)
            said = f" — project: {title_of(kind)}" if held == kind else " — it keeps the project it already has"
        return CaptureResult(r.text + said, r.outcome) if said else r

    async def _add_task(self, text: str, when: str, kind: str | None, details: dict,
                        owner: str | None = None) -> CaptureResult:
        now = self.ctx.now
        at = self._due(when)
        # a resolved time already in the past can never fire a nudge — do NOT schedule
        # it (that would promise a reminder that never comes); say so honestly below.
        past = at is not None and gone(at, now)
        meta = {k: v for k, v in (("project", kind), ("owner", owner)) if v}
        out = await self._propose("task", text, None if past else at, details, meta=meta or None)
        if out == "collapsed":
            # a duplicate — folded onto an existing task. Anchor the read-back on the
            # SURVIVING task the gate named (its id), not on norm(text): a fuzzy collapse
            # folds onto a differently-worded task. Report its ACTUAL time.
            target = next((t for t in await self.ctx.memory.open_tasks()
                           if t.id == details.get("target_id")), None)
            stored = target.deadline if target else None
            if stored and at is not None and not past and at != stored:
                return CaptureResult(f"already on your list — I kept the existing {when_label(stored, now)}", out)
            if stored:
                return CaptureResult(f"already on your list — scheduled for {when_label(stored, now)}", out)
            return CaptureResult("already on your list — nothing new to add", out)
        if out not in ("task", "task_exists"):
            return CaptureResult(OUTCOME.get(out, "couldn't save that"), out)
        match = next((t for t in await self.ctx.memory.open_tasks() if norm(t.text) == norm(text)), None)
        stored = match.deadline if match else None
        verb = "added to your tasks" if out == "task" else "already on your list"
        if stored:
            return CaptureResult(f"{verb} — scheduled for {when_label(stored, now)}", out)
        if past:
            return CaptureResult(f"{verb} — but that time has already passed, so I set no reminder", out)
        if when:
            return CaptureResult(f"{verb} — I couldn't read that time, so it has no time set", out)
        return CaptureResult(verb, out)

    async def _match_open(self, task: str) -> list:
        # match by TEXT against the LIVE open tasks: exact norm first, else the phrase is a
        # distinctive part of the task text ("briefcase" -> "order a new briefcase"), never
        # the reverse (a short task text matches too many unrelated queries).
        # An id (from `tasks`) wins over any text match: it names exactly one task.
        q = norm(str(task))
        open_now = await self.ctx.memory.open_tasks()
        return [t for t in open_now if t.id == str(task).strip()] or [t for t in open_now if norm(t.text) == q] or [t for t in open_now if q and q in norm(t.text)]

    async def _one_open(self, task: str):
        """The single open task a text or id names, or the CaptureResult that says why not."""
        hits = await self._match_open(task)
        if not hits:
            return None, CaptureResult("no open task matches that", "task_absent")
        if len(hits) > 1:
            return None, CaptureResult("several match (" + "; ".join(t.text for t in hits) + ") — which one?", "ambiguous")
        return hits[0], None

    async def _guard(self, task: str, at: str):
        """`at` parsed and the stale check run: (at ms, None) to go ahead, else (None, the result)."""
        at_ms, bad = _parse_at(at)
        return (None, bad) if bad else (at_ms, await self._stale(task, at_ms))

    async def complete_task(self, task: str, at: str = "") -> CaptureResult:
        at_ms, stop = await self._guard(task, at)
        if stop:
            return stop
        hit, miss = await self._one_open(task)
        if miss:
            return miss
        out = await self.ctx.propose.complete(hit.id, provenance=self.provenance, evidence=self.evidence)
        if out == "completed" and at_ms is not None:
            await store.stamp_task(hit.id, at_ms, closed=True)
        if out == "pending":
            return CaptureResult("not marked yet — a done? card is up", out)
        return CaptureResult("marked done" if out == "completed" else "already gone", out)

    async def discard_task(self, task: str, reason: str = "", at: str = "") -> CaptureResult:
        at_ms, stop = await self._guard(task, at)
        if stop:
            return stop
        hit, miss = await self._one_open(task)
        if miss:
            return miss
        out = await self.ctx.propose.discard(hit.id, reason, provenance=self.provenance, evidence=self.evidence)
        if out == "discarded" and at_ms is not None:
            await store.stamp_task(hit.id, at_ms, closed=True)
        if out == "pending":
            return CaptureResult("not discarded yet — a discard? card is up", out)
        return CaptureResult("discarded, not counted as done" if out == "discarded" else "already gone", out)

    async def reschedule_task(self, task: str, deadline: str = "", when: str = "", at: str = "") -> CaptureResult:
        at_ms, stop = await self._guard(task, at)
        if stop:
            return stop
        at = self._due(deadline or when)                    # `when`: the deprecated alias (see add_task)
        if at is None:
            return CaptureResult("I couldn't read that time, so I didn't move it", "reschedule_invalid")
        if gone(at, self.ctx.now):
            return CaptureResult("that time has already passed, so I didn't set it", "reschedule_invalid")
        hit, miss = await self._one_open(task)
        if miss:
            return miss
        out = await self.ctx.propose.reschedule(hit.id, at, provenance=self.provenance, evidence=self.evidence)
        if out == "pending":
            return CaptureResult("not moved yet — a reschedule card is up", out)
        if out == "rescheduled":
            if at_ms is not None:
                await store.stamp_task(hit.id, at_ms)
            return CaptureResult(f"moved '{hit.text}' to {when_label(at, self.ctx.now)}", out)
        return CaptureResult("that task is no longer open", out)

    async def schedule_task(self, task: str, start: str = "", minutes=None, at: str = "") -> CaptureResult:
        """Set when the user works on a task: `start` (local ISO datetime) for `minutes`; an
        empty start clears it. A planning act: applied at once, no card. No default duration:
        minutes is always given. A slot the user set by hand is held: until it ends, only the
        user moves or clears it (an agent's write is refused with slot_held)."""
        _, stop = await self._guard(task, at)
        if stop:
            return stop
        hit, miss = await self._one_open(task)
        if miss:
            return miss
        by_user = self.provenance in (Provenance.USER_STATED, Provenance.USER_INSTRUCTED)
        if (not by_user and hit.slot_by == "user" and hit.slot_start and hit.slot_minutes
                and hit.slot_start + hit.slot_minutes * 60_000 > self.ctx.now):
            return CaptureResult(f"the user set the slot on '{hit.text}' by hand; it stays as it is", "slot_held")
        if not str(start or "").strip():
            await store.set_slot(hit.id, None, None)
            return CaptureResult(f"cleared the slot on '{hit.text}'", "slot_cleared")
        try:
            mins = int(minutes)
            if isinstance(minutes, bool) or mins != float(minutes):
                raise ValueError
        except (TypeError, ValueError):
            mins = 0
        if not 0 < mins <= MAX_SLOT_MINUTES:
            return CaptureResult(f"minutes is required: how long the work takes, a whole number 1..{MAX_SLOT_MINUTES}",
                                 "slot_invalid")
        at = self._parse_when(start) if self._parse_when else None
        if at is None or _DATE_ONLY.match(str(start).strip()):
            return CaptureResult("start must be a local ISO 8601 datetime like 2026-10-20T14:00", "slot_invalid")
        await store.set_slot(hit.id, at, mins, "user" if by_user else "agent")
        return CaptureResult(f"'{hit.text}' slotted {when_label(at, self.ctx.now)} for {mins} min", "slotted")

    async def assign_task(self, task: str, owner: str = "", at: str = "") -> CaptureResult:
        """Hand an open task to a person (or back to the user: empty or 'me')."""
        at_ms, stop = await self._guard(task, at)
        if stop:
            return stop
        hit, miss = await self._one_open(task)
        if miss:
            return miss
        who, note = await self._owner(owner)
        out = await self.ctx.propose.assign(hit.id, who, provenance=self.provenance, evidence=self.evidence)
        if out == "pending":
            return CaptureResult("not assigned yet — an assign? card is up", out)
        if out != "assigned":
            return CaptureResult("that task is no longer open", out)
        if at_ms is not None:
            await store.stamp_task(hit.id, at_ms)
        note += await self._owner_notes(hit.text, who)
        return CaptureResult(f"'{hit.text}' is now owned by {owner_label(who)}{note}" if who
                             else f"'{hit.text}' is yours again{note}", out)

    async def _owner_notes(self, text: str, who: str | None) -> str:
        """Person notes that still say someone else owns this task ('Omar owns the release
        notes' after it went to Nora): named in the reply so the caller updates them. The same
        fact in a task field and a note must not disagree silently."""
        words = {w for w in slug(text).split("_") if len(w) > 3 and w not in GLUE | _TASK_VERBS}
        if not words:
            return ""
        stale = [e for e in await self.ctx.memory.structured("people") if e.kind != who
                 and "own" in norm(e.content) and len(words & set(slug(e.content).split("_"))) >= min(2, len(words))]
        return "".join(f" — Ken's note on {title_of(e.kind)} still says: \u201c{e.content[:160]}\u201d; "
                       f"update it with remember if that changed" for e in stale[:3])

    async def link_tasks(self, task: str, depends_on: str = "", remove: bool = False, at: str = "") -> CaptureResult:
        """`task` waits on `depends_on` (both an id or a text); `remove` drops that dependency."""
        _, stop = await self._guard(task, at)
        if stop:
            return stop
        hit, miss = await self._one_open(task)
        if miss:
            return miss
        if remove:                                     # the dependency is one this task has, any status
            have = (await store.deps_map()).get(hit.id, [])
            q, key = norm(str(depends_on)), str(depends_on).strip()
            deps = [d for d in have if d["id"] == key] or [d for d in have if norm(d["text"]) == q] \
                or [d for d in have if q and q in norm(d["text"])]
            if len(deps) != 1:
                return CaptureResult(f"'{hit.text}' does not wait on that" if not deps else
                                     "several match (" + "; ".join(d["text"] for d in deps) + ") — which one?",
                                     "link_absent" if not deps else "ambiguous")
            dep_id, dep_text = deps[0]["id"], deps[0]["text"]
        else:
            dep, miss = await self._one_open(depends_on)
            if miss:
                return CaptureResult("the task it waits on: " + miss.text, miss.outcome)
            dep_id, dep_text = dep.id, dep.text
            if dep_id == hit.id or await store.depends_on_path(dep_id, hit.id):
                return CaptureResult(f"refused: '{dep_text}' already waits on '{hit.text}' (or is it), "
                                     "so this would make a cycle", "cycle")
        out = await self.ctx.propose.link(hit.id, dep_id, remove=remove, provenance=self.provenance,
                                          evidence=self.evidence)
        return CaptureResult({"pending": "not linked yet — a link tasks? card is up",
                              "linked": f"'{hit.text}' now waits on '{dep_text}'",
                              "unlinked": f"'{hit.text}' no longer waits on '{dep_text}'",
                              "cycle": "refused: that would make a cycle",
                              "link_absent": f"'{hit.text}' does not wait on that"}.get(out, "that task is no longer open"), out)

    # ── events ─────────────────────────────────────────────────────────────

    async def log_event(self, text: str, when: str = "", until: str = "", at: str = "") -> CaptureResult:
        """`until` = when this stops mattering (local ISO 8601): after it the event is hidden
        from context and search, never deleted. A date-only `until` means the END of that
        day. An unreadable or already-past `until` is dropped and said, never silently."""
        said, bad = _parse_at(at)                           # `at` = when it was reported; `when` = when it happened
        if bad:
            return bad
        parse = self._parse_when or (lambda _s: None)
        at = parse(when) if when else None
        if at is None:                                       # no time: when it was reported or said, else now
            at = said if said is not None else getattr(self.ctx, "said_at", None)
        exp = parse(until) if until else None
        if exp is not None and _DATE_ONLY.match(until.strip()):
            exp += _DAY - 1000                                               # 23:59:59 local
        notes = []
        if when and at is None:
            notes.append("I couldn't read that time, so I set it to just now")
        if until and exp is None:
            notes.append("I couldn't read the `until` time, so it never expires")
        elif exp is not None and exp < self.ctx.now:
            notes.append("that `until` has already passed, so I logged it without one")
            exp = None
        out = await self._propose("event", text, at, meta={"until": exp} if exp else None)
        if out == "event":
            return CaptureResult("logged that" + (" — " + "; ".join(notes) if notes else ""), out)
        if out == "collapsed":
            return CaptureResult("already logged that — nothing new to add", out)
        return CaptureResult(OUTCOME.get(out, "couldn't log that"), out)

    # ── beliefs ────────────────────────────────────────────────────────────

    async def _resolve_links(self, links, batch: set | None = None, cache: dict | None = None) -> tuple[list[dict], list[str]]:
        """Turn [{kind, name, rel}] into [{dst, rel, name}] against records that exist (or sit
        on the same batch card). The name shown is the RESOLVED record's title, so a fuzzy
        match is visible before the Yes. An unknown target or rel is dropped and SAID — never
        a dangling link."""
        out, notes = [], []
        cache = cache if cache is not None else {}
        for lk in links or []:
            lk = lk if isinstance(lk, dict) else {}
            kind, name, rel = (str(lk.get(k) or "").strip() for k in ("kind", "name", "rel"))
            ns = _NS.get(kind.lower())
            if ns is None or not name or rel not in RELS:
                notes.append(f"link to '{name or '?'}' dropped: needs kind ({_KINDS}), name, rel ({', '.join(RELS)})")
                continue
            target = f"{ns}.{slug(name)}"
            if ns not in cache:
                cache[ns] = await self.ctx.memory.structured(ns)
            existing = cache[ns]
            hit = next((e for e in existing if e.kind == target), None)
            if hit is None:                              # a name twin of exactly one existing record
                twins = [e for e in existing if _twin_in({(ns, _title_of(e))}, ns, name)]
                hit = twins[0] if len(twins) == 1 else None
            if hit is not None:
                out.append({"dst": hit.kind, "rel": rel, "name": _title_of(hit)})
            elif batch is not None and (ns, slug(name)) in batch:
                out.append({"dst": target, "rel": rel, "name": name})
            else:
                notes.append(f"link to '{name}' dropped: no such {NOUN.get(ns, ns)} yet — remember it first")
        return out, notes

    async def remember(self, kind: str, name: str, note: str, links=None, url: str = "", at: str = "") -> CaptureResult:
        """kind: preference, fact, person, project, asset. A new record in the user's own words is
        saved at once; a change to one, or an agent's words, is a confirm card. `name` is
        the topic or the entity; `note` is the COMPLETE statement to store. `links` =
        [{kind, name, rel}] to records that exist; `url` for an asset."""
        ns = _NS.get(str(kind).lower().strip())
        if ns is None:
            return CaptureResult(f"unknown kind {kind!r}: use {_KINDS}", "invalid")
        kind_id = f"{ns}.{slug(name)}"
        had = next((e for e in await self.ctx.memory.structured(ns) if e.kind == kind_id), None)
        at_ms, bad = _parse_at(at)
        if bad:
            return bad
        if stale := await self._stale_record(had, at_ms):
            return stale
        resolved, notes = await self._resolve_links(links)
        meta = {"links": resolved} if resolved else {}
        if url:
            meta["url"] = str(url).strip()
        out = await self._propose(f"structured:{kind_id}", note, meta=meta or None)
        if out == "applied" and at_ms is not None:
            await store.stamp_structured(kind_id, at_ms)
        if out == "collapsed" and resolved:
            notes.insert(0, f"{len(resolved)} link(s) saved onto the existing record")
        tail = (" — " + "; ".join(notes)) if notes else ""
        if out == "applied" and had is not None and norm(had.content) != norm(note):
            # a fast-track write onto an existing key REPLACES it: say so, never "saved"
            return CaptureResult(f"saved — this replaced your earlier note on '{name}' (the old one is kept in its history){tail}", out)
        return CaptureResult(OUTCOME.get(out, "couldn't save that") + tail, out)

    async def _stale_record(self, had, at: int | None):
        """A late, old write never replaces a record that was updated after it was said."""
        if at is None or had is None or not had.updated_at or had.updated_at <= at:
            return None
        await _db.audit("stale", had.kind, f"write at {_iso(at)} older than the update at {_iso(had.updated_at)}")
        return CaptureResult(f"stale: '{_title_of(had)}' was updated later ({_iso(had.updated_at)}) than this "
                             f"({_iso(at)}), so nothing changed — it stays: {had.content}", "stale")

    async def forget(self, kind: str, name: str, at: str = "") -> CaptureResult:
        ns = _NS.get(str(kind).lower().strip())
        if ns is None:
            return CaptureResult(f"unknown kind {kind!r}: use preference, fact, person or project", "invalid")
        at_ms, bad = _parse_at(at)
        if bad:
            return bad
        had = next((e for e in await self.ctx.memory.structured(ns) if e.kind == f"{ns}.{slug(name)}"), None)
        if stale := await self._stale_record(had, at_ms):
            return stale
        out = await self.ctx.propose.capture(
            Proposal(intent=f"retire:{ns}.{slug(name)}", content=f"forget {name}",
                     evidence=self.evidence, provenance=self._prov(Provenance.AGENT_INFERRED)),
            judge=None, message=self.message)
        return CaptureResult(OUTCOME.get(out, "couldn't forget that"), out)

    # ── hydrate: a batch of records into a fresh world ─────────────────────

    async def hydrate(self, items: list) -> CaptureResult:
        """Up to BATCH `{kind, name, note}` records from a sweep. An item bound for an EMPTY
        namespace has nothing to collapse onto, so it skips the judge and joins ONE batch
        card (keep, edit or drop per item on decide). An item bound for a namespace that
        already holds records is judged one by one, exactly like `remember`."""
        if not isinstance(items, list) or not items:
            return CaptureResult("hydrate needs a list of {kind, name, note} items", "invalid")
        if len(items) > BATCH:
            return CaptureResult(f"at most {BATCH} items per call — send the rest in another call", "invalid")
        # ponytail: one query per namespace (four) — the slice reader has no all-namespaces read
        filled = {ns for ns in set(_NS.values()) if await self.ctx.memory.structured(ns)}
        batch, lines, outcomes, seen = [], [], [], set()
        cache: dict = {}
        for it in items:
            it = it if isinstance(it, dict) else {}
            kind, name, note = (str(it.get(k) or "").strip() for k in ("kind", "name", "note"))
            ns = _NS.get(kind.lower())
            if ns is None or not name or not note:
                lines.append(f"- {name or '?'}: skipped — needs kind ({_KINDS}), name, note")
                outcomes.append("invalid")
                continue
            twin = _twin_in(seen, ns, name)
            if twin is not None:                         # two rows for one record: the first wins
                lines.append(f"- {name}: same as '{twin}' in this batch, skipped — send it with remember after")
                outcomes.append("invalid")
                continue
            seen.add((ns, name))
            if ns in filled:
                r = await self.remember(kind, name, note, links=it.get("links"), url=str(it.get("url") or ""))
                outcomes.append(r.outcome)
                lines.append(f"- {name}: {r.text}")
            else:
                item = {"kind": ns, "name": name, "note": note, "_links": it.get("links")}
                if it.get("url"):
                    item["url"] = str(it["url"]).strip()
                batch.append(item)
        # links resolve against what EXISTS plus what sits on this very card — never against an
        # item that was skipped or went to its own card
        card_keys = {(i["kind"], slug(i["name"])) for i in batch}
        for item in batch:
            resolved, notes = await self._resolve_links(item.pop("_links"), batch=card_keys, cache=cache)
            if resolved:
                item["links"] = resolved
            for n in notes:
                lines.append(f"- {item['name']}: {n}")
        if batch:
            source = self.evidence[0].source if self.evidence else "the sweep"
            n = len(batch)
            out = await self.ctx.propose.capture(
                Proposal(intent="batch:hydrate",
                         content=f"Save {n} new {'record' if n == 1 else 'records'} from {source}?",
                         evidence=self.evidence, provenance=self._prov(Provenance.AGENT_INFERRED),
                         meta={"items": batch, **self._meta()}),
                judge=None, message=self.message)
            outcomes.append(out)
            lines.insert(0, f"{n} new on one card — relay each item (keep, edit or drop), then "
                            "decide(card_id, true, edited='[{\"name\": …, \"note\": …}, {\"name\": …, \"drop\": true}]')"
                         if out == "pending" else f"couldn't card the batch ({out})")
        # the TRUE token: a card up → pending; every item landed → applied; none → failed; else partial
        ok = {"applied", "collapsed"}
        if any(o in ("pending", "conflict") for o in outcomes):
            outcome = "pending"
        elif outcomes and all(o in ok for o in outcomes):
            outcome = "applied"
        elif any(o in ok for o in outcomes):
            outcome = "partial"
        else:
            outcome = "failed"
        return CaptureResult("\n".join(lines), outcome)

    # ── reads (for the tool loop; the MCP server reads the store itself) ──

    async def find_task(self, query: str) -> CaptureResult:
        hits = await self.ctx.memory.find_tasks(str(query), include_done=True)
        if not hits:
            return CaptureResult("no task matches that", "none")
        lines = [f"- {t.text} · {t.status}" + (f" ({t.drop_reason})" if t.status == "dropped" and t.drop_reason else "")
                 + (f" · {when_label(t.deadline, self.ctx.now)}" if t.deadline else "")
                 for t in hits]
        return CaptureResult("\n".join(lines), "found")

    async def find_events(self, query: str) -> CaptureResult:
        hits = await self.ctx.memory.find_events(str(query))
        if not hits:
            return CaptureResult("no event matches that", "none")
        return CaptureResult("\n".join(f"- {e.summary} · {when_label(e.at, self.ctx.now)}" for e in hits), "found")
