"""Ken as an MCP server — the context engine's one interface for agents. Any session
(Claude Code, Codex, the Claude app through a connector, a long-running agent) reads the
user's world and reports back through the tools below; nothing here holds a model key or
writes memory directly. Reads hit the store. Every write is posted to the inbox and answered
by the engine — the single writer — with the TRUE outcome and the ids of the cards it raised.

Register once (stdio, same machine, same ~/.ken):
  claude mcp add ken -- ken-mcp

Or serve it over HTTP (`ken-mcp --http`, the `http:` section of config.yaml: host, port,
token, agent_tokens) and register from ANY machine, nothing to install there:
  claude mcp add --transport http ken https://<your-host>/mcp --header "Authorization: Bearer <token>"
A client that cannot send a header (a claude.ai custom connector) uses the token as a path
prefix instead: https://<your-host>/t/<token>/mcp — the secret is then in a URL, so rotate it after.
`http.token` is the USER's token (writes may carry the user's provenance); `http.agent_tokens`
({name: token}) are AGENT tokens, whose writes are world input at most — the engine clamps.
The same server answers `POST /` (text in, kip's reply out — the iPhone Shortcut) and
`GET /today`. The Bearer token is the only gate: set one before you expose it.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import importlib.metadata
import json
import re
import time
from collections import Counter
from contextvars import ContextVar
from datetime import datetime

from . import __version__, config, db
from .agents import assistant_view
from .captures import BATCH
from .consent import gate
from .contracts import Action, ActionKind, RenderableKind, link_sentence
from .conversation import bus
from .memory import store
from .util import GLUE as _GLUE
from .util import family_words, local_ms, new_id, norm, now_ms, owner_label, slug, title_of

_MAX_TEXT = 2000
_MAX_SOURCE = 60
NO_REPLY = "(no reply yet — is an engine running? start it with `ken`)"
QUEUED_NO_ENGINE = ("queued: the engine is not running, so this applies when the engine starts (`ken`) — "
                    "do not retry it")
QUEUED = "queued: the engine is busy and applies this when it gets to it — do not retry it"
CARD_RAISED = "the built-in assistant raised a consent card — relay it to the user and decide(card_id, approved)"

# who is calling: the credential that authenticated sets the provenance CEILING. `mcp:user`
# (the user's token, or stdio on their own machine) may write as the user; `mcp:agent:<name>`
# writes WORLD at most. The engine clamps; this only stamps the channel.
_CALLER: ContextVar[str] = ContextVar("ken_caller", default="mcp:user")


def _caller() -> str:
    return _CALLER.get()


def _evidence(items, n: int = 3) -> list[dict]:
    """The refs behind a record, as stored: a permalink or an id, and the source tag."""
    return store.evidence_dicts(items, n)


def _slot(t, now: int) -> dict:
    """The slot on the wire: {slot: {start, minutes}} while it has not ended or it started today
    (local), with missed: true once it ended on an open task; an ended slot from an earlier day on an
    open task reads as unslotted, reported as {missed_slot: start}. Computed, never stored."""
    if not (t.slot_start and t.slot_minutes):
        return {}
    slot = {"start": _iso(t.slot_start), "minutes": t.slot_minutes, **({"by": "user"} if t.slot_by == "user" else {})}
    if t.slot_start + t.slot_minutes * 60_000 > now:
        return {"slot": slot}
    if _local_day(t.slot_start) == _local_day(now):
        return {"slot": {**slot, "missed": True} if t.status == "open" else slot}
    return {"missed_slot": _iso(t.slot_start)} if t.status == "open" else {}


def _local_day(ms: int):
    return datetime.fromtimestamp(ms / 1000).date()  # noqa: DTZ006 — local, by design


def _deps(t, deps: dict | None) -> dict:
    """{depends_on: [{id, text, status}], blocked} when the task waits on others; blocked while
    any of them is still open. Nothing when it waits on none."""
    ds = (deps or {}).get(t.id)
    return {"depends_on": ds, "blocked": any(d["status"] == "open" for d in ds)} if ds else {}


def _task_row(t, now: int | None = None, deps: dict | None = None, **extra) -> dict:
    """One task on the wire, the same shape on every read: id, text, deadline, slot, project, owner
    (None = the user), depends_on + blocked (when it waits on others), evidence."""
    return {"id": t.id, "text": t.text, "deadline": _iso(t.deadline), **_slot(t, now or now_ms()),
            "project": title_of(t.project) if t.project else None, "owner": owner_label(t.owner),
            **_deps(t, deps), "evidence": _evidence(t.evidence), **extra}


def _tag(source: str) -> str:
    """`source` is model-written: one line, short, so the tag stays a tag."""
    return " ".join(str(source or "").split())[:_MAX_SOURCE] or "an agent"


async def _post(kind: ActionKind, text: str, *, meta: dict | None = None,
                timeout: float = 120.0, channel: str | None = None, token: str = "",
                anchor: str | None = None) -> dict:
    """Post one inbound action and wait for the engine's structured answer to THAT action
    (matched on the action id the engine echoes back), so a nudge or a turn typed elsewhere
    is never handed back as the answer. The body carries `text`, `outcome`, `result` and the
    ids of the cards this action raised. A card raised with no reply (kip went silent because
    the card IS the answer) is reported by id, never as a timeout."""
    action_id = new_id()
    start = await bus.last_seq(bus.MAIN)
    await bus.append_action(Action(id=action_id, conversation_id=bus.MAIN, kind=kind,
                                   channel=channel or _caller(), text=text, meta=meta or {},
                                   token=token, anchor=anchor, at=now_ms()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for _, r in await bus.read_outbox(bus.MAIN, start):
            body = dict(r.body or {})
            if r.kind is RenderableKind.ADVICE and body.get("reply_to") == action_id:
                if "cards" not in body:
                    body["cards"] = await gate.cards_for(action_id, bus.MAIN)
                return body
        await asyncio.sleep(0.4)
    cards = await gate.cards_for(action_id, bus.MAIN)
    if cards:
        return {"text": CARD_RAISED, "outcome": "pending", "cards": cards, "result": []}
    # the action stays in the inbox and applies when the engine gets to it: report it as
    # queued, never as a failure, so a client does not retry it into a duplicate
    return {"text": QUEUED if db.engine_is_running() else QUEUED_NO_ENGINE,
            "outcome": "queued", "cards": [], "result": []}


async def reply_to(text: str, channel: str, timeout: float = 120.0) -> str:
    """Plain-text talk-and-wait for the phone Shortcut: kip's reply, or the card notice."""
    body = await _post(ActionKind.MESSAGE, text, channel=channel, timeout=timeout)
    return str(body.get("text") or (CARD_RAISED if body.get("cards") else NO_REPLY))


async def tell(text: str, source: str = "a coding session", timeout: float = 90.0,
               task_id: str = "", provenance: str = "agent", ref: str = "", at: str = "") -> str:
    """Free text to kip (the reconciler), tagged with its source. With `task_id` it is a
    REPORT on that task: stored as an event with ref = task id, and kip decides close, move
    or follow-up. `at` (local ISO date or datetime) = when the text was SAID: kip resolves its
    relative dates against it and events default to it; empty = now. (`KEN_NOW` freezes the
    engine's clock for tests and benchmarks — see util.now_ms.) Returns the structured result
    as JSON: text, outcome, result, cards."""
    text = " ".join(str(text or "").split())[:_MAX_TEXT]
    if not text:
        return json.dumps({"text": "nothing to tell", "outcome": "empty", "cards": [], "result": []})
    meta = {"source": _tag(source), "provenance": provenance}
    if str(at or "").strip():
        try:
            meta["at"] = local_ms(at)
        except (ValueError, TypeError):
            return json.dumps({"text": f"`at` must be a local ISO date or datetime like 2026-03-11 or "
                                       f"2026-03-11T09:30, not {str(at)[:40]!r}; nothing was told",
                               "outcome": "invalid", "cards": [], "result": []})
    if task_id:
        meta["task_id"] = str(task_id)
    if ref:
        meta["ref"] = str(ref)[:2000]
    body = await _post(ActionKind.MESSAGE, f"(via {_tag(source)}) {text}", meta=meta, timeout=timeout)
    return json.dumps({k: body.get(k) for k in ("text", "outcome", "cards", "result", "summary") if k in body})


async def typed(tool: str, args: dict, *, source: str = "", provenance: str = "agent",
                timeout: float = 60.0, ref: str = "") -> str:
    """One typed write through the inbox: no model turn, judged, answered with the true
    outcome and the cards it raised. `ref` (a permalink) becomes the write's evidence ref."""
    meta = {"source": _tag(source) if source else "", "provenance": provenance}
    if ref:
        meta["ref"] = str(ref)[:2000]
    body = await _post(ActionKind.TOOL, json.dumps({"tool": tool, "args": args}), meta=meta, timeout=timeout)
    return json.dumps({k: body.get(k) for k in ("text", "outcome", "cards", "result") if k in body})


# ── reads: straight from the store, no model, no engine ───────────────────────────────

def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%dT%H:%M") if ms else None  # noqa: DTZ006


async def _lanes(tasks, now: int) -> dict:
    # the published panel holds kip's Decisions ranking of the untimed backlog (by task id); an
    # untimed task it has not seen yet lands in someday, as it would on the panel.
    view = await store.get_view(bus.MAIN) or {}
    lane_of = {tid: item.get("label") for b in view.get("blocks", []) or [] if b.get("type") == "lanes"
               for item in b.get("items", []) or [] for tid in item.get("ids", []) or []}
    step = {"today": 2, "this week": 1}
    scores = [step.get(lane_of.get(t.id), 0) for t in tasks if not t.deadline]
    return assistant_view.lanes_for(tasks, now, scores)


async def today(now: int | None = None) -> str:
    """The task lanes as kip last ranked them, as plain text (the phone Shortcut's view)."""
    now = now or now_ms()
    out: list[str] = []
    lanes = await _lanes(await store.open_tasks(), now)
    for label in ("overdue", "today", "this week", "someday"):
        if lanes[label]:
            out.append(f"{label}:")
            out += [f"  - {t.text}" for t in lanes[label]]
    return "\n".join(out) or "Nothing open. Clear."


DIGEST_CARDS = 8                 # a phone push stays under ~4 KB (ntfy's message cap) at this many


def render_card(c: dict) -> str:
    """One card as plain lines, for the digest and `ken cards`: the title says what a Yes
    does (an update names the value it replaces), then the question, the note's first line,
    the batch items, the id."""
    title = gate._card_label(c["kind"], c.get("existing"))[0]
    out = [f"[{title}] {c['question']}"]
    if c.get("existing") is not None:
        out.append(f"   now: {c['existing']}")
    if c.get("url"):
        out.append(f"   url: {c['url']}")
    for lk in c.get("links") or []:
        out.append(f"   link: {lk['rel']} \u2192 {lk['name']}")
    if c.get("note"):
        out.append("   " + str(c["note"]).splitlines()[0])
    if c.get("items"):
        out += ["   " + ln for ln in gate._batch_note(c["items"]).splitlines()]
    out.append(f"   id {c['card_id']}")
    return "\n".join(out)


async def pending_digest() -> str:
    """The open cards as short plain text — what a scheduled host routine, a phone Shortcut or
    `ken push` delivers at 09:00. Empty string when nothing waits (so nothing is sent); capped
    at DIGEST_CARDS cards so a backlog never turns the message into an attachment."""
    cards = await pending_cards()
    if not cards:
        return ""
    n = len(cards)
    out = [f"Ken · {n} open question{'s' if n > 1 else ''}"]
    for i, c in enumerate(cards[:DIGEST_CARDS], 1):
        out.append(f"{i}. " + render_card(c))
    if n > DIGEST_CARDS:
        out.append(f"… and {n - DIGEST_CARDS} more: ken cards")
    out.append("Answer in any Ken-connected agent (decide), or: ken decide <id> yes|no")
    return "\n".join(out)


async def pending_count(conversation_id: str = bus.MAIN) -> int:
    async with db.get_db() as c:
        cur = await c.execute("SELECT COUNT(*) FROM pending WHERE conversation_id=?", (conversation_id,))
        return int((await cur.fetchone())[0])


async def pending_cards(conversation_id: str = bus.MAIN) -> list[dict]:
    async with db.get_db() as c:
        cur = await c.execute(
            "SELECT ask_id, kind, content, ask_type, author, action_id, meta, created_at, "
            "origin, existing_snapshot, evidence FROM pending WHERE conversation_id=? ORDER BY created_at",
            (conversation_id,))
        rows = await cur.fetchall()
    out = []
    for r in rows:
        meta = json.loads(r["meta"]) if r["meta"] else {}
        # a judge-conflict card replaces a value that EXISTS: the client must show it
        existing = gate._snapshot_content(r["existing_snapshot"])   # None when nothing was snapshotted
        # the same title + named subject the in-engine card carries: "Save person" / "Bram Okafor",
        # so a card about a person is never an anonymous description on any surface
        title, derived = gate._card_label(r["kind"], existing)
        card = {"card_id": r["ask_id"], "question": r["content"], "kind": r["kind"], "title": title,
                "type": r["ask_type"] or "belief", "raised_by": r["author"], "existing": existing,
                "links": [{"rel": lk.get("rel"), "name": lk.get("name") or lk.get("dst")} for lk in meta.get("links") or []],
                "url": meta.get("url"),
                "task_id": meta.get("task_id"), "subject": meta.get("to") or derived,
                "note": meta.get("context"), "action_id": r["action_id"],
                "evidence": _evidence(store.evidence_load(r["evidence"])),
                "at": _iso(r["created_at"]),
                "answer": "decide(card_id, approved=true|false, edited='')"}
        if r["ask_type"] == "batch":
            card["items"] = meta.get("items") or []
            card["answer"] = ("ask keep / edit / drop per item, then decide(card_id, true, edited="
                              "'[{\"name\": …, \"note\": …}, {\"name\": …, \"drop\": true}]') "
                              "— items you leave out are kept as proposed; approved=false drops all")
        out.append(card)
    return out


HYDRATE_PLAN = [
    ("people: everyone the user exchanged direct messages with in the last 30 days (Slack DMs, "
     "email threads, shared docs, code reviews) — one line each: who they are to the user and what is live between them"),
    ("projects: docs, drives, channels and repos with activity in the last 30 days — one line each: "
     "what it is and where it stands"),
    "calendar: recurring meetings in the next 14 days — each one names a person or a project, not an event",
    "preferences: standing preferences the user has already stated to you (tools, hours, style)",
    ("assets: the docs, sheets, folders, repos and dashboards each project lives in — with the url and a link "
     "{kind: project, name, rel: part_of}; link a project's people with rel member or owner"),
]


async def hydrate(items: str | list = "", source: str = "hydrate", provenance: str = "agent") -> str:
    """No items: the sweep plan for a fresh world plus progress on a partial one. With items
    (a JSON list of {kind, name, note}, at most ten): one batch through the engine — into an
    empty namespace it skips the judge and comes back as ONE card; into a filled one each
    item is judged like `remember`."""
    if not items:
        counts = Counter(e.kind.split(".", 1)[0] for e in await store.get_structured())
        open_batches = [c["card_id"] for c in await pending_cards() if c["type"] == "batch"]
        have = {ns: counts.get(ns, 0) for ns in ("people", "projects", "preferences", "facts", "assets")}
        return json.dumps({
            "plan": HYDRATE_PLAN,
            "have": have,
            "state": "fresh" if not any(have.values()) else "partial",
            "open_batches": open_batches,
            "batch_size": BATCH,
            "send": "hydrate(items='[{\"kind\": \"person|project|preference|fact\", \"name\": …, "
                    "\"note\": \"one complete sentence\"}, …]') — then relay the card and decide it",
        })
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except ValueError:
            return json.dumps({"outcome": "invalid", "text": "items must be a JSON list of {kind, name, note}"})
    return await typed("hydrate", {"items": items}, source=source, provenance=provenance)


TASK_STATUSES = ("open", "done", "dropped", "all")


async def tasks(filter: str = "", now: int | None = None, status: str = "open") -> str:
    """Open tasks with id, due, age, project, evidence and the lane kip last published; plus how many
    questions wait for the user. `filter`: a lane name, or a substring of the text or the project.
    `status`: open (default) | done | dropped | all — closed rows carry status, completed, drop_reason."""
    now = now or now_ms()
    status = str(status or "open").strip().lower()
    if status not in TASK_STATUSES:
        return json.dumps({"error": f"status must be one of {', '.join(TASK_STATUSES)}, not {status[:20]!r}"})
    deps = await store.deps_map()
    all_open = await store.open_tasks() if status in ("open", "all") else []
    lanes = await _lanes(all_open, now)
    f = (filter or "").strip().lower()
    notes = {e.kind: f"{title_of(e.kind)} {e.content}".lower() for e in await store.get_structured("projects.")}
    words = _find_words(f)

    def hit(t) -> bool:
        """The filter as a phrase in the text or project, or every word of it in the text, the
        project's name or the project's note ('launch' finds the tasks of the launch project)."""
        project = title_of(t.project) if t.project else ""
        if f in t.text.lower() or f in project:
            return True
        hay = f"{norm(t.text)} {notes.get(t.project, '')}"
        return bool(words) and all(w in hay for w in words)

    rows = []
    lanes_all = [(label, t) for label in ("overdue", "today", "this week", "someday") for t in lanes[label]]
    missed = bool(f) and f not in ("overdue", "today", "this week", "someday") and status == "open" \
        and lanes_all and not any(hit(t) for _, t in lanes_all)
    for label, t in lanes_all:
        if f and not missed and f != label and not hit(t):
            continue
        rows.append(_task_row(t, now, deps, lane=label,
                              age=assistant_view._due_label(t, now, overdue=(label == "overdue")) or None,
                              created=_iso(t.created_at), **({"status": "open"} if status == "all" else {})))
    if status != "open":
        for t in await store.tasks_by_status(("done", "dropped") if status == "all" else (status,)):
            if f and not hit(t):
                continue
            rows.append(_task_row(t, now, deps, status=t.status, completed=_iso(t.completed_at),
                                  created=_iso(t.created_at),
                                  **({"drop_reason": t.drop_reason} if t.status == "dropped" else {})))
    # the day page's view: today's slots, by start (ahead, in progress or missed)
    day = datetime.fromtimestamp(now / 1000).strftime("%Y-%m-%d")  # noqa: DTZ006 — local, by design
    slotted = sorted((r for r in rows if (r.get("slot") or {}).get("start", "").startswith(day)),
                     key=lambda r: r["slot"]["start"])
    # today's done slots stay on the day page's timeline after a reload
    midnight = int(datetime.fromtimestamp(now / 1000).replace(hour=0, minute=0, second=0,  # noqa: DTZ006
                                                                microsecond=0).timestamp() * 1000)
    done = sorted((t for t in await store.done_since(midnight)
                   if t.slot_start and t.slot_minutes and midnight <= t.slot_start < midnight + 86_400_000),
                  key=lambda t: t.slot_start)
    done_slots = [_task_row(t, now, deps, status="done", slot={"start": _iso(t.slot_start), "minutes": t.slot_minutes, **({"by": "user"} if t.slot_by == "user" else {})})
                  for t in done]
    # a filter that misses the open tasks may name a CLOSED one ("is the draft still to do?"): say so
    closed = [_task_row(t, now, deps, status=t.status, completed=_iso(t.completed_at))
              for t in await store.tasks_by_status(("done", "dropped")) if hit(t)][:10] if missed else []
    return json.dumps({"tasks": rows, "today_slots": [r["id"] for r in slotted], "done_slots": done_slots,
                       **({"closed_matches": closed} if closed else {}),
                       "open_questions": await pending_count(),
                       "hint": (f"the filter {f!r} matched no open task, so every open task is listed"
                                + ("; closed_matches are done or dropped tasks it matched; " if closed else "; ")
                                if missed else "")
                               + "call get_context(task_id) before working on a task; "
                               "call pending() when open_questions > 0"})


_STOP = {"the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at", "by", "with", "from",
         "while", "when", "if", "my", "our", "his", "her", "their", "this", "that", "is", "are",
         "buy", "decide", "call", "review", "send", "fix", "write", "read", "start", "finish",
         "pay", "book", "order", "check", "ask", "plan", "prepare", "put", "do", "get"}
_MONTHS = {m.lower() for m in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}


def _names_in(text: str) -> list[str]:
    """Capitalised tokens that could be a person, place or project — the computed gap
    candidates. Never the first word (sentence case), never a month or weekday."""
    words = re.findall(r"[A-Za-z][\w'-]+", text)
    out: list[str] = []
    for i, w in enumerate(words):
        if i == 0 or not w[0].isupper() or w.lower() in _STOP or w.lower() in _MONTHS:
            continue
        if any(ch.isdigit() for ch in w):          # Q4, KR5, v4, O2: a label, never a person
            continue
        if w not in out:
            out.append(w)
    return out


def _title(e) -> str:
    return title_of(e.kind)


NOTE_CAP = 400                   # a context answer carries the gist; who_is(name) has the whole note


def _clip(text: str, cap: int) -> str:
    """The text itself when it fits (so `is` tells you nothing was cut), else its head."""
    return text if len(text) <= cap else text[:cap].rstrip() + " …"


def _related_item(e, why: str, score: int) -> dict:
    note = _clip(e.content, NOTE_CAP)
    item = {"kind": _noun(e.kind.split(".", 1)[0]), "name": _title(e), "note": note, "why": why,
            "evidence": _evidence(e.evidence),
            "updated": _iso(e.updated_at), "_kind": e.kind, "_score": score}
    if note is not e.content:
        ns = e.kind.split(".", 1)[0]
        item["more"] = (f"who_is('{_title(e)}')" if ns in ("people", "projects", "assets") else f"find('{_title(e)}')") + " for the full note"
    if e.url:
        item["url"] = e.url
    return item


def _named(title: str, text_l: str, family: frozenset = frozenset()) -> int:
    """How strongly the entity's name appears in the task text, as WHOLE words — 'Al' never
    matches 'all'. 0 = not named. The full title counts 3; each distinct non-family word 1."""
    t = title.lower()
    if len(t) > 2 and re.search(rf"\b{re.escape(t)}\b", text_l):
        return 3
    return sum(1 for w in set(t.split()) if len(w) > 2 and w not in family
               and re.search(rf"\b{re.escape(w)}\b", text_l))


def _noun(ns: str) -> str:
    return gate._NOUN.get(ns, ns)


async def get_context(task_id: str, now: int | None = None, budget_chars: int = 8000) -> str:
    """Everything that matters for ONE task, and nothing else: the task; the people,
    projects and preferences that relate to it (by name in the text, or by shared evidence),
    each with a reason and its evidence; the dated events in its window; the last report on
    it; the questions the engine has for the agent (names it cannot place) and for the user
    (open cards). Deterministic. Related is strongest first (named > shared evidence > linked)
    and the budget trims the weakest; events go oldest first; the answer says `truncated`."""
    now = now or now_ms()
    task = await store.task_by_id(task_id)          # open, done or dropped: history stays reachable
    if task is None:
        return json.dumps({"error": f"no task with id {task_id!r}; call tasks() first"})
    text_l = task.text.lower()
    task_refs = {ev.ref for ev in task.evidence}
    related: list[dict] = []
    known_titles: list[str] = []
    entries = await store.get_structured()
    family = frozenset(family_words([_title(e).lower() for e in entries]))
    for e in sorted(entries, key=lambda e: -(e.updated_at or 0)):
        title = _title(e)
        known_titles.append(title.lower())
        refs = {ev.ref for ev in e.evidence}
        strength = _named(title, text_l, family)
        if e.kind == task.project:                      # a fact on the task, stronger than any name
            related.append(_related_item(e, "the task's project", 20))
        elif strength:
            related.append(_related_item(e, "named in the task", 10 + strength))
        elif task.id in refs or (task_refs & refs):
            related.append(_related_item(e, "shares evidence with the task", 5))
    # a family word in the task ("the Crumbly Q1 update") whose family has no member
    # related yet: the family is the context there is, at the weakest strength, not nothing
    have = {r["_kind"] for r in related}
    for w in (w for w in family if re.search(rf"\b{re.escape(w)}\b", text_l)):
        members = [e for e in entries if w in _title(e).lower().split()]
        if members and not any(m.kind in have for m in members):
            for m in members:
                related.append(_related_item(m, f"named in the task (family word '{w}')", 1))
                have.add(m.kind)
    # ONE hop of explicit links from what was found, both directions — never further. The
    # reason is the link's own sentence, in its stored direction, so it reads right either way.
    by_kind = {e.kind: e for e in entries}
    found = {r["_kind"] for r in related}
    for lk in await store.links_of(sorted(found)):
        other = lk["dst"] if lk["src"] in found else lk["src"]
        if other in found or other not in by_kind or lk["src"] not in by_kind:
            continue
        why = link_sentence(_title(by_kind[lk["src"]]), lk["rel"], _title(by_kind[lk["dst"]]))
        related.append(_related_item(by_kind[other], f"linked: {why}", 1))
        found.add(other)
    # strongest first: the record the task names outranks one that shares evidence, which
    # outranks a linked hop; the budget below trims from the END, so the weakest goes first
    related.sort(key=lambda r: -r.pop("_score"))
    for r in related:
        r.pop("_kind", None)
    day = 86_400_000
    if task.deadline:
        lo, hi = task.deadline - 14 * day, task.deadline + 14 * day
    else:
        lo, hi = now - 14 * day, now + day
    names = [r["name"].lower() for r in related]
    words = [w for w in text_l.split() if len(w) > 3]
    # by ref, any age; newest LAST — two reports logged in the same second keep insert order
    reports = list(reversed(await store.events_citing(task.id)))
    events = []
    for ev in await store.recent_events(200, now=now):
        s_l = ev.summary.lower()
        if task.id in {x.ref for x in ev.evidence}:
            continue
        if lo <= ev.at <= hi and (any(w in s_l for w in words) or any(n in s_l for n in names)):
            events.append(ev)
    events.sort(key=lambda e: -e.at)
    me = config.section("user").get("name") or ""                 # the user is never a gap
    mine = {w.lower() for w in (me if isinstance(me, list) else str(me).split())}
    gaps = [n for n in _names_in(task.text) if n.lower() not in mine and n.lower() not in family
            and not any(n.lower() in t for t in known_titles)]
    cards = [c for c in await pending_cards()
             if c.get("task_id") == task.id or text_l in (c["question"] or "").lower()]
    out = {
        "task": {"id": task.id, "text": task.text, "status": task.status, "deadline": _iso(task.deadline),
                 **_slot(task, now), "owner": owner_label(task.owner), **_deps(task, await store.deps_map()),
                 "created": _iso(task.created_at),
                 **({"drop_reason": task.drop_reason} if task.status == "dropped" else {}),
                 "evidence": _evidence(task.evidence)},
        "related": related,
        "events": [{"what": e.summary, "when": _iso(e.at)} for e in events],
        "last_report": {"what": reports[-1].summary, "when": _iso(reports[-1].at)} if reports else None,
        "questions_for_agent": [f"I do not know who or what {n!r} is. Check your own sources "
                                "(Slack profile and recent threads, GitHub, calendar), then tell(..., source=...)"
                                for n in gaps],
        "questions_for_user": cards,
        "note": "nothing here is fetched live; every item cites the evidence it rests on",
    }
    # the budget: drop the oldest events first, then the WEAKEST related items; say so
    while len(json.dumps(out)) > budget_chars and (out["events"] or len(out["related"]) > 3):
        if out["events"]:
            out["events"].pop()
        else:
            out["related"].pop()
        out["truncated"] = True
    if len(json.dumps(out)) > budget_chars:
        out["truncated"] = True
    return json.dumps(out)


async def who_is(name: str) -> str:
    """One person or project by name, with its note, evidence and the recent events that
    mention it — or 'unknown; find out' with what to check."""
    key = slug(name)
    everything = await store.get_structured()
    entities = [e for e in everything if e.kind.split(".", 1)[0] in ("people", "projects")]
    hits = [e for e in entities if e.kind.split(".", 1)[-1] == key or _title(e).lower() == name.strip().lower()]
    if not hits:
        hits = [e for e in everything if e.kind.split(".", 1)[0] == "assets"
                and (e.kind.split(".", 1)[-1] == key or _title(e).lower() == name.strip().lower())]
    if not hits:
        parts = {w for w in name.lower().split() if len(w) > 2}
        similar = sorted({e.subject if e.subject not in ("", "you") else _title(e)
                          for e in entities if parts & set(_title(e).lower().split())})[:5]
        return json.dumps({"name": name, "known": False, "similar": similar,
                           "find_out": ["their Slack profile and your recent threads with them",
                                        "their GitHub profile and recent PRs", "your calendar for meetings"],
                           "then": "tell('<who they are, one or two sentences>', source='<where you looked>')"})
    e = hits[0]
    mentions = [{"what": ev.summary, "when": _iso(ev.at)} for ev in await store.recent_events(200)
                if _title(e).lower() in ev.summary.lower()][:5]
    by_kind = {x.kind: x for x in everything}
    links = []
    for lk in await store.links_of([e.kind]):
        if lk["src"] not in by_kind or lk["dst"] not in by_kind:
            continue
        other = by_kind[lk["dst"] if lk["src"] == e.kind else lk["src"]]
        links.append({"rel": lk["rel"], "name": _title(other), "kind": _noun(other.kind.split(".", 1)[0]),
                      "text": link_sentence(_title(by_kind[lk["src"]]), lk["rel"], _title(by_kind[lk["dst"]]))})
    out = {"name": _title(e), "kind": _noun(e.kind.split(".", 1)[0]), "known": True,
           "note": e.content, "updated": _iso(e.updated_at),
           "evidence": _evidence(e.evidence, 5),
           "links": links, "mentions": mentions}
    if e.url:
        out["url"] = e.url
    return json.dumps(out)


WORLD_NOTE = 240                 # a list row carries the gist; who_is(name) has the whole note
WORLD_TASKS = 5                  # task texts per entity; `open` and `late` count them all


async def world(now: int | None = None, budget_chars: int = 60_000) -> str:
    """What Ken knows, in one read: every project, person and asset with its links, the user's
    standing preferences, and for each project and person its
    open tasks: the tasks ON a project (then, for a task on no project, the ones that name it),
    and the tasks that name a person. Straight from the store: no model, no engine. The list a surface
    draws from; who_is(name) has one entity's full note, evidence and mentions."""
    now = now or now_ms()
    everything = await store.get_structured()
    by_kind = {e.kind: e for e in everything}
    ns_of = lambda kind: kind.split(".", 1)[0]
    ents = [e for e in everything if ns_of(e.kind) in ("projects", "people", "assets")]
    of: dict[str, list[dict]] = {}
    for lk in await store.all_links():
        if lk["src"] in by_kind and lk["dst"] in by_kind:
            of.setdefault(lk["src"], []).append(lk)
            of.setdefault(lk["dst"], []).append(lk)
    open_now = await store.open_tasks()
    deps = await store.deps_map()
    late_ids = {t.id for t in (await _lanes(open_now, now))["overdue"]}   # the same rule as tasks(): a date-only task is due all day
    # a task text, once: punctuation as spaces (so 'Node.js' meets the title 'node js'), and its names
    seen = [(t, re.sub(r"[\W_]+", " ", t.text.lower()), {n.lower() for n in _names_in(t.text)}) for t in open_now]
    family = frozenset(family_words([_title(e).lower() for e in ents if ns_of(e.kind) == "projects"]))

    def tasks_of(e) -> list:
        title = _title(e).lower()
        if ns_of(e.kind) == "projects":
            # the project ON the task is a fact. The name match (the full title, or two distinct
            # words of it, one when it has only one) fills in only for a task with no project.
            skip = family | _GLUE
            n = len({w for w in title.split() if len(w) > 2 and w not in skip})
            hit = [t for t, low, _ in seen if t.project == e.kind
                   or (not t.project and n and _named(title, low, skip) >= min(2, n))]
        else:
            # a person: the full name, or one name word written as a NAME in the task (capitalised,
            # not the first word) — 'rent a van' is not Ruth van Dalen's
            words = {w for w in title.split() if len(w) > 2 and w not in _GLUE}
            whole = re.compile(rf"\b{re.escape(title)}\b") if len(title) > 2 else None
            hit = [t for t, low, names in seen if (whole and whole.search(low)) or words & names]
        return sorted(hit, key=lambda t: (t.id not in late_ids, t.deadline is None, t.deadline or 0))

    def row(e) -> dict:
        out = {"name": _title(e), "note": _clip(e.content, WORLD_NOTE), "updated": _iso(e.updated_at)}
        if e.url:
            out["url"] = e.url
        return out

    def linked(e) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {"people": [], "projects": [], "assets": []}
        for lk in of.get(e.kind, []):
            other = by_kind[lk["dst"] if lk["src"] == e.kind else lk["src"]]
            if other.kind == e.kind or ns_of(other.kind) not in out:
                continue
            item = {"name": _title(other), "rel": lk["rel"],
                    "text": link_sentence(_title(by_kind[lk["src"]]), lk["rel"], _title(by_kind[lk["dst"]]))}
            if other.url:
                item["url"] = other.url
            out[ns_of(other.kind)].append(item)
        return {ns: sorted(items, key=lambda x: (x["name"], x["rel"])) for ns, items in out.items()}

    def with_tasks(e) -> dict:
        ts = tasks_of(e)
        return {**row(e), "open": len(ts), "late": sum(1 for t in ts if t.id in late_ids),
                "tasks": [_task_row(t, now, deps, late=t.id in late_ids) for t in ts[:WORLD_TASKS]], **linked(e)}

    busy = lambda r: (-r["late"], -r["open"], r["name"])
    counts = Counter(ns_of(e.kind) for e in everything)
    out = {"projects": sorted((with_tasks(e) for e in ents if ns_of(e.kind) == "projects"), key=busy),
           "people": sorted((with_tasks(e) for e in ents if ns_of(e.kind) == "people"), key=busy),
           "assets": sorted(({**row(e), **linked(e)} for e in ents if ns_of(e.kind) == "assets"), key=lambda r: r["name"]),
           # standing rules the user stated (hours, style, tools): small, and an agent planning the
           # day must check against them, so the budget below never cuts them
           "preferences": sorted(({"name": _title(e), "note": _clip(e.content, WORLD_NOTE)}
                                  for e in everything if ns_of(e.kind) == "preferences"), key=lambda r: r["name"]),
           "counts": {ns: counts.get(ns, 0) for ns in ("people", "projects", "assets", "facts", "preferences")},
           "truncated": False,
           "hint": "who_is(name) has one entity's full note, evidence and mentions"}
    # over budget: the quiet rows (no open task) lose their notes first, then every row its task texts
    quiet = [r for ns in ("projects", "people") for r in out[ns] if not r["open"]] + out["assets"]
    for cut in (lambda: [r.pop("note", None) for r in quiet],
                lambda: [r.pop("tasks", None) for ns in ("projects", "people") for r in out[ns]]):
        if len(json.dumps(out)) <= budget_chars:
            break
        cut()
        out["truncated"] = True
    return json.dumps(out)


_FIND_SKIP = frozenset({"the", "and", "for", "with", "from", "what", "when", "who", "where", "which", "how",
                        "does", "did", "was", "were", "are", "is", "has", "have", "had", "still", "now",
                        "my", "our", "we", "you", "your", "about", "that", "this", "any", "all", "to", "of",
                        "in", "on", "at", "by", "be", "do", "it", "a", "an"})


def _find_words(query: str) -> list[str]:
    """The query's searchable words: common words dropped, a light stem so 'booked' finds
    'book' and 'tiles' finds 'tile'."""
    out = []
    for w in re.findall(r"[\w']+", norm(query)):
        if w in _FIND_SKIP or len(w) < 3:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return list(dict.fromkeys(out))


FIND_HINT = "nothing matched these words; tasks() and world() show everything Ken holds"
FIND_NOTE = ("beliefs and open tasks are the current state; events are dated history, newest first — "
             "when an event and a belief disagree, the belief is current")


async def find(query: str) -> str:
    """Search across tasks (open and done), events, people, projects, preferences by WORDS:
    a hit holds at least one of the query's words, ranked by how many it holds; the exact
    phrase ranks first. 'kitchen budget' finds the project whose note states the budget."""
    q = norm(str(query))
    words = _find_words(query)
    if not q or not words:
        return json.dumps({"tasks": [], "events": [], "beliefs": [], "hint": FIND_HINT})

    def score(*texts: str) -> int:
        t = " ".join(norm(x or "") for x in texts)
        return sum(w in t for w in words) + (len(words) if q in t else 0)

    tasks, events = {}, {}
    for w in words:                                    # the store matches one substring at a time
        for t in await store.find_tasks(w, include_done=True, limit=200):
            tasks[t.id] = t
        for e in await store.find_events(w, limit=200):
            events[e.id] = e
    ranked_t = sorted(tasks.values(), key=lambda t: -score(t.text, t.project or ""))[:20]
    ranked_e = sorted(events.values(), key=lambda e: -score(e.summary))[:20]
    beliefs = [(score(_title(e), e.content), e) for e in await store.get_structured()]
    ranked_b = [e for n, e in sorted((x for x in beliefs if x[0]), key=lambda x: -x[0])][:20]
    deps = await store.deps_map()
    ts = [_task_row(t, None, deps, status=t.status, **({"drop_reason": t.drop_reason} if t.status == "dropped" else {}))
          for t in ranked_t]
    es = [{"what": e.summary, "when": _iso(e.at)} for e in sorted(ranked_e, key=lambda e: -(e.at or 0))]
    bs = [{"kind": e.kind, "name": _title(e), "note": e.content, "updated": _iso(e.updated_at)} for e in ranked_b]
    # what is true NOW first; events are dated history, newest first: an older event's value
    # may since have changed, and the reader must not quote it as current
    return json.dumps({"beliefs": bs, "tasks": ts, "events": es,
                       **({"note": FIND_NOTE} if es else {}), **({} if ts or es or bs else {"hint": FIND_HINT})})


EVENTS_DAYS = 30


async def events(since: str = "", until: str = "", now: int | None = None) -> str:
    """Live events (not past their `until`) whose time falls in [since, until], newest first.
    Local ISO dates or datetimes; a date-only `until` means the end of that day. Defaults: the
    last EVENTS_DAYS days up to now."""
    now = now or now_ms()
    day = 86_400_000
    try:
        lo = local_ms(since) if str(since or "").strip() else now - EVENTS_DAYS * day
        hi = now
        if str(until or "").strip():
            hi = local_ms(until) + (day - 1 if len(str(until).strip()) == 10 else 0)
    except (ValueError, TypeError):
        return json.dumps({"error": "since and until must be local ISO dates or datetimes like 2026-03-11"})
    return json.dumps({"events": [{"what": e.summary, "when": _iso(e.at), **({"until": _iso(e.until)} if e.until else {})}
                                  for e in await store.events_between(lo, hi, now=now)],
                       "since": _iso(lo), "until": _iso(hi)})


async def decide(card_id: str, approved: bool, edited: str = "", timeout: float = 30.0) -> str:
    """Answer one open card on the USER's behalf: the one tap every client makes, anchored to
    the card so the refusal enters kip's history. The outcome is the engine's own word
    (applied / rejected / restaled: a fresh card replaced it / no_pending), never inferred
    from a row disappearing. An agent token cannot decide: consent is the user's."""
    if _caller().startswith("mcp:agent"):
        return json.dumps({"card_id": card_id, "outcome": "forbidden",
                           "text": "an agent token cannot answer a card; relay it to the user's own client"})
    if await gate.open_ask(card_id) is None:
        return json.dumps({"card_id": card_id, "outcome": "no_such_card"})
    token = f"{'approve' if approved else 'reject'}:{card_id}"
    body = await _post(ActionKind.ACTION_RESPONSE, (edited or "").strip(), token=token, anchor=card_id,
                       timeout=timeout)
    out = body.get("outcome", "queued")
    res = {"card_id": card_id, "outcome": out, "edited": bool(edited), "cards": body.get("cards") or []}
    if out == "restaled":
        res["text"] = "the value changed under the card; a fresh card was raised — call pending()"
    elif out == "reasked":
        res["text"] = "not the same one — a fresh card asks to save it as new; call pending() and relay it"
    elif out == "queued":
        res["text"] = body.get("text") or QUEUED
    return json.dumps(res)


TOOL_NAMES = frozenset({"tasks", "get_context", "who_is", "find", "add_task", "complete_task", "discard_task",
                        "reschedule_task", "log_event", "remember", "forget", "tell", "pending", "decide",
                        "hydrate", "world", "schedule_task", "assign_task", "link_tasks", "events", "chat_history"})


_AT = "`at` = when this change HAPPENED (local ISO date or datetime), for a late report; a change older than the task's last change is refused as `stale` and nothing changes. Empty = now."


def _version() -> str:
    try:
        return importlib.metadata.version("ken-engine")
    except importlib.metadata.PackageNotFoundError:   # a bare checkout, not installed
        return __version__


def build_server():
    from mcp.server.mcpserver import MCPServer  # optional dep: pip install 'ken-engine[mcp]'
    srv = MCPServer(
        "ken", version=_version(),
        instructions=("Ken is the user's context engine: their tasks, people, projects, preferences "
                      "and what happened, kept current. Before you work on a task for the user, call "
                      "tasks() then get_context(task_id) and start from it — never ask the user to "
                      "retype context. When you finish or learn something, report it with "
                      "tell(text, source, task_id) or the typed writes. Every write returns the true "
                      "outcome and the ids of the cards it raised: relay each card's question to the "
                      "user in your own chat and answer it with decide(card_id, approved). Never claim "
                      "something was saved unless the outcome says so. On a fresh world, or when the "
                      "user says 'set up my world', call hydrate() and follow its plan. A task has two "
                      "times: `deadline` = by when (only a deadline can be overdue) and `slot` = when the "
                      "user works on it (schedule_task, start + minutes). A task is a commitment the user "
                      "made or accepted; advice or how-to steps from a chat are not tasks unless the user "
                      "says they will do them."))

    @srv.tool(name="tasks", description=(
        "The place to look for anything due or open: the user's open tasks with their CURRENT "
        "deadlines (a finished task: find). Each row: (id, text, lane, deadline, slot {start, minutes, by: 'user' when the user set it by hand} when set, missed: true "
        "inside a slot from today that ended with the task open, missed_slot = the start of such a slot "
        "from an earlier day, age, project, evidence [{ref, source}] "
        "= where each came from), today_slots = ids slotted today by start, done_slots = today's slotted "
        "tasks already done (full rows), and how many questions wait "
        "for them. Lanes follow the deadline. "
        "`filter` narrows by lane (overdue, today, this week, someday) or a substring of the text or the project. "
        "Each row has `owner` (null = the user) and, when it waits on other tasks, `depends_on` [{id, text, status}] "
        "and `blocked` (true while one is open). `status`: open (default) | done | dropped | all — closed rows "
        "carry status, completed, drop_reason."))
    async def tasks_tool(filter: str = "", status: str = "open") -> str:
        return await tasks(filter, status=status)

    @srv.tool(name="events", description=(
        "What happened in a date window, no word search: live events with their time, newest first. "
        "`since` / `until` = local ISO dates or datetimes; default the last 30 days."))
    async def events_tool(since: str = "", until: str = "") -> str:
        return await events(since, until)

    @srv.tool(name="get_context", description=(
        "Everything that matters for ONE task: the related people, projects and preferences with "
        "the reason and evidence for each; dated events in its window; the last report on it; "
        "questions for you (names Ken cannot place: go find out with your own tools, then tell) "
        "and questions for the user (open cards: relay them, then decide). Call it before working."))
    async def get_context_tool(task_id: str) -> str:
        return await get_context(task_id)

    @srv.tool(name="who_is", description=(
        "One person or project by name: note, evidence, recent mentions. Unknown → near matches and "
        "what to check; then tell(...) what you found."))
    async def who_is_tool(name: str) -> str:
        return await who_is(name)

    @srv.tool(name="world", description=(
        "The whole map of what Ken knows, as one list: every person, project and asset with its current "
        "notes, its links and its open "
        "tasks (counts: open, late): the tasks on a project, and the tasks that name a person; plus "
        "`preferences` (name, note): the user's standing rules, to respect when you plan. "
        "Read-only. Use who_is(name) for one entity in full."))
    async def world_tool() -> str:
        return await world()

    @srv.tool(name="find", description=(
        "A word search for something specific in tasks (open and done), events, people, projects, "
        "preferences: a hit holds at least one query word, best matches first. `beliefs` and open "
        "tasks are the CURRENT state; `events` are dated history, newest first — when they disagree, "
        "the belief is current. If it finds nothing, look in tasks() or world() before concluding "
        "Ken does not know."))
    async def find_tool(query: str) -> str:
        return await find(query)

    @srv.tool(name="add_task", description=(
        "Add a task. `deadline` = by when it must be done, a local ISO 8601 date or datetime, ONLY "
        "when one is known; a date alone means by that day. It is not a working time: set when the "
        "user works on it with schedule_task. `project` = the name of the user's project it belongs to "
        "(see world()), when you know it; an unknown name leaves the task on no project. `ref` = a "
        "permalink to the message or page this comes from; it becomes the task's evidence. "
        "`provenance`: 'user' when you relay the user's own instruction, else 'agent'. `owner` = who does it: "
        "a person's name, empty = the user (an unknown name is kept as given). Only a commitment the user "
        "made or accepted — not a wish or idea, small talk, a suggestion they declined, or advice and "
        "how-to steps from a chat. When unsure, it is not a task. `when` is the old name of `deadline`: leave it empty. " + _AT))
    async def add_task_tool(text: str, deadline: str = "", source: str = "", provenance: str = "agent", ref: str = "",
                            project: str = "", when: str = "", owner: str = "", at: str = "") -> str:
        # ponytail: `when` = deprecated alias of `deadline`; drop it once live agents send `deadline`
        return await typed("add_task", {"text": text, "deadline": deadline or when, "project": project,
                                        "owner": owner, "at": at},
                           source=source, provenance=provenance, ref=ref)

    @srv.tool(name="complete_task", description=(
        "Mark an open task done. Pass `task_id` (the id from `tasks`) when you have it: it is exact. "
        "Else `task` = its text (or a distinctive part). " + _AT))
    async def complete_task_tool(task: str = "", source: str = "", provenance: str = "agent", ref: str = "",
                                 task_id: str = "", at: str = "") -> str:
        return await typed("complete_task", {"task": task_id or task, "at": at}, source=source,
                           provenance=provenance, ref=ref)

    @srv.tool(name="discard_task", description=(
        "Close an open task that was NOT done: cancelled, no longer needed, a duplicate, a mistake. It "
        "closes as `dropped` with the reason, so history and done counts stay true. Never use "
        "complete_task for this. Pass `task_id` (the id from `tasks`) when you have it, else `task` = its text. " + _AT))
    async def discard_task_tool(task: str = "", reason: str = "", source: str = "", provenance: str = "agent",
                                ref: str = "", task_id: str = "", at: str = "") -> str:
        return await typed("discard_task", {"task": task_id or task, "reason": reason, "at": at}, source=source,
                           provenance=provenance, ref=ref)

    @srv.tool(name="reschedule_task", description=(
        "Move an open task's deadline (by when) to a new local ISO 8601 date or datetime; a date alone "
        "means by that day. Pass `task_id` (the id from `tasks`) when you have it, else `task` = its "
        "text. To set when the user works on it, use schedule_task. " + _AT))
    async def reschedule_task_tool(task: str = "", deadline: str = "", source: str = "", provenance: str = "agent",
                                   ref: str = "", task_id: str = "", when: str = "", at: str = "") -> str:
        # ponytail: `when` = deprecated alias of `deadline`; drop it once live agents send `deadline`
        return await typed("reschedule_task", {"task": task_id or task, "deadline": deadline or when, "at": at},
                           source=source, provenance=provenance, ref=ref)

    @srv.tool(name="assign_task", description=(
        "Change who does an open task: `owner` = a person's name (see world()); empty or 'me' = the user. "
        "An unknown name is kept as given. Pass `task_id` when you have it, else `task` = its text. " + _AT))
    async def assign_task_tool(task: str = "", owner: str = "", task_id: str = "", source: str = "",
                               provenance: str = "agent", ref: str = "", at: str = "") -> str:
        return await typed("assign_task", {"task": task_id or task, "owner": owner, "at": at},
                           source=source, provenance=provenance, ref=ref)

    @srv.tool(name="link_tasks", description=(
        "Say an open task waits on another: `task` (or `task_id`) waits on `depends_on` (an id or a text). "
        "It reads as blocked until that one is closed. `remove`: true drops the dependency. A cycle is "
        "refused (outcome `cycle`). " + _AT))
    async def link_tasks_tool(task: str = "", depends_on: str = "", task_id: str = "", remove: bool = False,
                              source: str = "", provenance: str = "agent", ref: str = "", at: str = "") -> str:
        return await typed("link_tasks", {"task": task_id or task, "depends_on": depends_on, "remove": remove,
                                          "at": at}, source=source, provenance=provenance, ref=ref)

    @srv.tool(name="schedule_task", description=(
        "Set when the user works on an open task: `start` = a local ISO 8601 datetime, `minutes` = how "
        "long (required, a whole number 1..1440; there is no default). An empty `start` clears the slot. "
        "Slots may overlap. A slot is planning, not world knowledge: it is applied at once for any "
        "caller, no card. It never changes the deadline; a slot that ends with the task open is "
        "reported as missed (today) or missed_slot (an earlier day). A slot the user set by hand (slot.by = "
        "\"user\") is held until it ends: only provenance \"user\" moves or clears it; any other write gets "
        "outcome slot_held and changes nothing. Pass `task_id` when you have it, else `task` = its text. " + _AT))
    async def schedule_task_tool(task: str = "", task_id: str = "", start: str = "", minutes: int = 0,
                                 source: str = "", provenance: str = "agent", ref: str = "", at: str = "") -> str:
        return await typed("schedule_task", {"task": task_id or task, "start": start, "minutes": minutes, "at": at},
                           source=source, provenance=provenance, ref=ref)

    @srv.tool(name="log_event", description=(
        "Record something that HAPPENED (append-only history), or an UPCOMING dated happening that is "
        "context rather than a to-do (a trip, a deadline). `when` local ISO 8601, omit for now. "
        "`until` = when it stops mattering (an offer, a trip, a deadline): after that it leaves "
        "context and search but is never deleted. `at` = when it was reported (local ISO); with no `when`, "
        "the event time is `at`."))
    async def log_event_tool(text: str, when: str = "", until: str = "", source: str = "",
                             provenance: str = "agent", ref: str = "", at: str = "") -> str:
        return await typed("log_event", {"text": text, "when": when, "until": until, "at": at}, source=source,
                           provenance=provenance, ref=ref)

    @srv.tool(name="remember", description=(
        "Save a belief: kind = preference | fact | person | project | asset. A new record in the user's "
        "own words (provenance 'user') is saved at once; a change to a record, or an agent's own "
        "words, is a card the user confirms. `name` = the topic or the entity; `note` = the COMPLETE statement. `links` = a "
        "JSON list of {kind, name, rel} to records Ken already has (rel: member | owner | asset | "
        "part_of | client), e.g. a project's people or an asset's project. `url` for an asset (a sheet, "
        "a repo, a doc, a dashboard). `at` = when this was stated (local ISO); a note older than the "
        "record's last update is refused as `stale`."))
    async def remember_tool(kind: str, name: str, note: str, links: str = "", url: str = "",
                            source: str = "", provenance: str = "agent", ref: str = "", at: str = "") -> str:
        args = {"kind": kind, "name": name, "note": note, "at": at}
        if links:
            try:
                args["links"] = json.loads(links)
            except ValueError:
                return json.dumps({"outcome": "invalid", "text": "links must be a JSON list of {kind, name, rel}"})
        if url:
            args["url"] = url
        return await typed("remember", args, source=source, provenance=provenance, ref=ref)

    @srv.tool(name="forget", description=(
        "Forget a saved belief by kind and name (a card confirms). `at` = when this was said; older than "
        "the record's last update = `stale`, nothing forgotten."))
    async def forget_tool(kind: str, name: str, source: str = "", provenance: str = "agent", at: str = "") -> str:
        return await typed("forget", {"kind": kind, "name": name, "at": at}, source=source, provenance=provenance)

    @srv.tool(name="tell", description=(
        "Tell Ken what happened, in plain words. With `task_id` it is a REPORT on that task: Ken "
        "records it, then closes, moves or adds a follow-up through cards. `ref` = a permalink to what "
        "you read. `at` = when this was said, for an older message (local ISO date or datetime); "
        "empty = now. Returns the outcome list and the card ids raised; relay the cards to the user "
        "and decide them."))
    async def tell_tool(text: str, source: str = "a coding session", task_id: str = "",
                        provenance: str = "agent", ref: str = "", at: str = "") -> str:
        return await tell(text, source, task_id=task_id, provenance=provenance, ref=ref, at=at)

    @srv.tool(name="chat_history", description=(
        "The recent chat thread one client had with Ken, oldest first: each message it sent with "
        "`source` = that client's tag, and Ken's reply: state answered (reply text), card (the answer was only "
        "cards, in pending), thinking (not handled yet) or none (no answer). "
        "Read-only. For a chat box: show it, then send new lines with tell(text, source)."))
    async def chat_history_tool(source: str, limit: int = 30) -> str:
        rows = await bus.chat(bus.MAIN, _tag(source), max(1, min(int(limit or 30), 100)))
        return json.dumps({"messages": [{**r, "at": _iso(r["at"])} for r in rows]})

    @srv.tool(name="hydrate", description=(
        "Set up a fresh world, or continue a partial one. Without items: the sweep plan (who and what "
        "to pull from your own connectors) and what Ken already has. With items: a JSON list of up to "
        "ten {kind: person|project|preference|fact|asset, name, note, links?: [{kind, name, rel}], url?} "
        "records; Ken answers with ONE card for the batch — relay each item (keep, edit, drop) and "
        "decide it with edited JSON."))
    async def hydrate_tool(items: str = "", source: str = "hydrate", provenance: str = "agent") -> str:
        return await hydrate(items, source=source, provenance=provenance)

    @srv.tool(name="pending", description=(
        "The open cards waiting for the user's decision: `title` (the action, e.g. Save person), `subject` "
        "(the named person/project when the question is only its description), `question`, and the "
        "evidence [{ref, source}]: show the "
        "user where a card came from (the source tag, the ref as a link when it is a URL) before they decide."))
    async def pending_tool() -> str:
        return json.dumps(await pending_cards())

    @srv.tool(name="decide", description=(
        "Answer one card on the user's behalf after asking them: approved true/false; `edited` = the "
        "user's corrected value on an approve."))
    async def decide_tool(card_id: str, approved: bool, edited: str = "") -> str:
        return await decide(card_id, approved, edited)

    return srv


def plain(md: str) -> str:
    """Flatten kip's markdown for plain-text surfaces (a phone notification, Speak Text)."""
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)             # [text](url) -> text
    md = re.sub(r"\*\*(\S|\S[^*\n]*\S)\*\*", r"\1", md)          # **bold**
    md = re.sub(r"\*(\S|\S[^*\n]*\S)\*", r"\1", md)              # *italic*; `5 * 3` survives
    md = md.replace("`", "")
    md = re.sub(r"^\s{0,3}#{1,6}\s*", "", md, flags=re.MULTILINE)        # # headers
    md = re.sub(r"^(\s*)[-*]\s+", r"\1• ", md, flags=re.MULTILINE)       # - list -> •
    return md.strip()


def message_of(body: str) -> str:
    """The message in a request body: `{"text": "..."}` or the raw text; one line, capped
    like a tool call — an interface never hands the engine an unbounded body."""
    text = str(body or "").strip()
    if text.startswith("{"):
        try:
            text = json.loads(text).get("text", text)
        except Exception:  # noqa: BLE001, S110 — a body that only looks like JSON is still a message
            pass
    return " ".join(str(text or "").split())[:_MAX_TEXT]


def _tokens(cfg: dict) -> dict[bytes, str]:
    """Every accepted bearer → the channel it stamps: the user token → `mcp:user`; each
    `agent_tokens` entry → `mcp:agent:<name>`."""
    out: dict[bytes, str] = {}
    if cfg.get("token"):
        out[f"Bearer {cfg['token']}".encode()] = "mcp:user"
    for name, tok in (cfg.get("agent_tokens") or {}).items():
        if tok:
            out[f"Bearer {tok}".encode()] = f"mcp:agent:{slug(str(name))}"
    return out


def http_app(token: str | None, host: str = "127.0.0.1", agent_tokens: dict | None = None):
    """One ASGI app: the MCP endpoint at /mcp, plus `POST /` (text in, plain reply out)
    and `GET /today` for clients that speak no MCP. With a token set, the Bearer check
    is the gate on every route and the transport's Host check is off (a tunnel rewrites
    the Host header); without one, the SDK keeps its loopback-only Host check. The token
    that matched stamps the caller's channel for the request (the provenance ceiling)."""
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse

    srv = build_server()

    @srv.custom_route("/", methods=["POST"])
    async def inbox(request: Request) -> PlainTextResponse:
        text = message_of((await request.body()).decode(errors="replace"))
        if not text:
            return PlainTextResponse("(empty message — no text in the request body)")
        return PlainTextResponse(plain(await reply_to(text, "http:client")))

    @srv.custom_route("/today", methods=["GET"])
    async def today_route(_: Request) -> PlainTextResponse:
        return PlainTextResponse(await today())

    @srv.custom_route("/pending", methods=["GET"])
    async def pending_route(_: Request) -> PlainTextResponse:
        return PlainTextResponse(await pending_digest() or "No open questions.")

    security = TransportSecuritySettings(enable_dns_rebinding_protection=False) if token else None
    app = srv.streamable_http_app(host=host, stateless_http=True, json_response=True,
                                  transport_security=security)
    accepted = _tokens({"token": token, "agent_tokens": agent_tokens or {}})
    prefix = f"/t/{token}" if token else None    # a client that cannot send a header (a
                                                 # claude.ai connector) puts the token in the path

    def _match(header: bytes) -> str | None:
        for want, channel in accepted.items():
            if len(header) == len(want) and hmac.compare_digest(header, want):
                return channel
        return None

    async def gated(scope, receive, send):
        channel = "mcp:user"
        if accepted and scope["type"] == "http":
            path = scope["path"]
            if prefix and path.startswith(prefix + "/") and hmac.compare_digest(path[:len(prefix)], prefix):
                scope = {**scope, "path": path[len(prefix):], "raw_path": path[len(prefix):].encode()}
            else:
                channel = _match(dict(scope["headers"]).get(b"authorization", b""))
                if channel is None:
                    resp = PlainTextResponse("unauthorized: the Bearer token does not match", 401)
                    return await resp(scope, receive, send)
        tok = _CALLER.set(channel)
        try:
            await app(scope, receive, send)
        finally:
            _CALLER.reset(tok)
    return gated


def main() -> None:
    ap = argparse.ArgumentParser(description="Ken's MCP server")
    ap.add_argument("--http", action="store_true",
                    help="serve over HTTP (config http: host/port/token) instead of stdio")
    args = ap.parse_args()
    db.configure()                       # the shared ~/.ken/ken.db, once per process
    asyncio.run(db.init_db())
    if not args.http:
        build_server().run("stdio")      # stdio on the user's own machine: the user ceiling
        return
    import uvicorn
    cfg = config.section("http")
    host, port = cfg.get("host", "127.0.0.1"), int(cfg.get("port", 8765))
    if not cfg.get("token") and host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"ken mcp: refusing to bind {host} with no http.token — anyone who "
                         "reaches this port could act as you. Set http.token in ~/.ken/config.yaml.")
    print(f"ken mcp listening on {host}:{port}  (token {'ON' if cfg.get('token') else 'off'}), "
          f"data in {db.db_path().parent}", flush=True)
    # host "*" = every interface, IPv4 AND IPv6 (uvicorn binds both when given no host; a
    # literal "::" is IPv6-only under asyncio). Needed where the network is IPv6-only.
    uvicorn.run(http_app(cfg.get("token"), host, cfg.get("agent_tokens") or {}),
                host=None if host == "*" else host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
