"""The consent gate — the single applier of agent-proposed writes. (ARCHITECTURE.md)

Toll by target-stakes. Stakes is a property of the target kind, not the
proposer's choice. Fast-track applies and logs, without interrupting. Ask emits
an approval request to the conversation, and records a pending row it can resolve
later on the user's yes/no. No batching in V1.

Each ask has a stable id. The tap token carries that id, and the pending row is
keyed by it, so a tap always resolves the exact question the card showed — a
superseded card names a deleted ask and is a safe no-op.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import replace
from types import SimpleNamespace

from .. import db as _db
from ..agents.base import Proposal
from ..contracts import (
    NOUN,
    ActionChoice,
    Provenance,
    Renderable,
    RenderableKind,
    StructuredEntry,
    orient,
)
from ..conversation import bus
from ..memory import store
from ..util import gone, new_id, norm, now_ms, owner_label, slug, stable_id, title_of, when_label
from . import judge as _judge

# Only these namespaces are low-stakes enough to apply without asking. Everything
# else -> ask. Deny-by-default: an agent cannot invent a fast-track namespace, so
# it cannot lower its own toll. (ARCHITECTURE.md rule 1)
_FAST_TRACK_NAMESPACES = frozenset({"patterns", "preferences"})


def consent_level(kind: str) -> str:
    """A declared property of the target, not the proposer's choice."""
    namespace = kind.split(".", 1)[0]
    return "fast_track" if namespace in _FAST_TRACK_NAMESPACES else "ask"


# a namespace -> the everyday noun a card names it by, so a card says WHAT is being decided
# ("Save person", "Forget project") instead of a bare "Approve?". Unknown namespaces -> "note".
_NOUN = NOUN                      # the shared vocabulary (contracts) — one place to add a namespace


def _link_note(meta: dict | None) -> str:
    """The card lines for a record's links and url, so a Yes is never a surprise."""
    meta = meta or {}
    lines = [f"link: {lk.get('rel')} \u2192 {lk.get('name') or lk.get('dst')}" for lk in meta.get("links") or []]
    if meta.get("url"):
        lines.insert(0, f"url: {meta['url']}")
    return "\n".join(lines)


async def _write_links(kind: str, meta: dict | None, evidence: list, conn=None) -> None:
    await store.set_links(kind, (meta or {}).get("links") or [], evidence=evidence, conn=conn)


async def _new_links(kind: str, links: list) -> list:
    """The links that are not stored yet, in either direction — so a re-mention with the same
    links asks nothing."""
    have = {(lk["src"], lk["dst"], lk["rel"]) for lk in await store.links_of([kind])}
    out = []
    for lk in links:
        o = orient(kind, str(lk.get("dst") or ""), str(lk.get("rel") or "")) if lk.get("dst") else None
        if o is not None and o not in have:
            out.append(lk)
    return out


async def _links_ask(proposal, kind: str, links: list, *, conversation_id: str, author: str) -> None:
    """One "Update X, add links" card per record. A second proposal with more links while
    the card is open MERGES into it and re-posts the card, so no link is lost in the wait."""
    ask_kind = f"links:{kind}"
    open_ask = await _open_ask_for(conversation_id, ask_kind)
    if open_ask is not None:
        meta = json.loads(open_ask["meta"]) if open_ask["meta"] else {}
        seen = {(lk["dst"], lk["rel"]) for lk in meta.get("links") or []}
        merged = list(meta.get("links") or []) + [lk for lk in links if (lk["dst"], lk["rel"]) not in seen]
        if len(merged) == len(seen):
            return                             # nothing new for the open card
        meta["links"] = merged
        async with _db.get_db() as c:
            await c.execute("UPDATE pending SET meta=? WHERE ask_id=?", (json.dumps(meta), open_ask["ask_id"]))
            await c.commit()
        await bus.append_renderable(_ask_card(
            open_ask["ask_id"], conversation_id=conversation_id, author=open_ask["author"],
            question=open_ask["content"], target=ask_kind, note="Add links:\n" + _link_note(meta)))
        return
    ask = replace(proposal, meta={**(getattr(proposal, "meta", None) or {}), "links": links})
    await _raise_ask(ask, kind=ask_kind, ask_type="links", conversation_id=conversation_id, author=author,
                     note="Add links:\n" + _link_note(ask.meta))


def _card_label(target: str, existing: str | None) -> tuple[str, str | None]:
    """The human action label + subject a consent card shows, derived from the target kind.
    Generic — every belief kind is `namespace.slug`. Returns (title, subject); subject is the
    de-slugged name for a NAMED entity (person/project) so the card can't be an anonymous note,
    and None where the card's own text already carries the name (a task, a retire)."""
    if target.startswith("task:"):
        return "New task", None
    if target.startswith("event:"):
        return "Log event?", None
    if target.startswith("batch:"):
        return "Save these?", None
    if target.startswith("identity:"):                             # identity:<target kind>:<own kind>
        return "Update " + _NOUN.get(target.split(":", 2)[1].split(".", 1)[0], "record"), None
    if target.startswith("links:"):                                # links:<record kind>: links only, the note stays
        rk = target.split(":", 1)[1]
        return "Update " + _NOUN.get(rk.split(".", 1)[0], "record"), rk.split(".", 1)[-1].replace("_", " ").title()
    if target.startswith("act:"):                                  # act:reply:<norm> → "Reply"
        verb = target.split(":", 2)[1].replace("_", " ")
        return {"mark done": "Mark done?", "discard": "Discard?", "reschedule": "Reschedule?", "assign": "Assign?",
                "link tasks": "Link tasks?"}.get(verb, verb.capitalize()), None
    if target.startswith("retire:"):
        return "Forget " + _NOUN.get(target.split(":", 1)[1].split(".", 1)[0], "note"), None
    ns = target.split(".", 1)[0]
    verb = "Update" if existing is not None else "Save"        # an existing value → a change, not a new save
    subject = (target.split(".", 1)[-1].replace("_", " ").strip().title()
               if ns in ("people", "projects") else None)      # named entities get a heading; topics don't
    return f"{verb} {_NOUN.get(ns, 'note')}", subject


# the inbound action a write belongs to: every card raised while it is set carries its id, so
# the caller (an MCP tool, a chat turn) can be told exactly which cards ITS call raised.
_ACTION: ContextVar[str | None] = ContextVar("ken_action_id", default=None)


async def cards_for(action_id: str, conversation_id: str | None = None) -> list[str]:
    """The open card ids raised by one inbound action, oldest first."""
    if not action_id:
        return []
    sql, args = "SELECT ask_id FROM pending WHERE action_id=?", [action_id]
    if conversation_id:
        sql += " AND conversation_id=?"; args.append(conversation_id)
    async with _db.get_db() as c:
        cur = await c.execute(sql + " ORDER BY created_at", args)
        return [r["ask_id"] for r in await cur.fetchall()]


async def apply(proposal, *, conversation_id: str, author: str,
                judge: _judge.Judge | None = None, message: str = "",
                details: dict | None = None, action_id: str | None = None) -> str:
    """The single applier of an agent-proposed write.

    When a `judge` is injected (the interactive path) it runs FIRST: the proposal
    passes the machine skeptic, then routes — keep (today's consent path), collapse
    (acknowledge, no duplicate, adopt a new time), or conflict (ask, with the existing
    belief snapshotted). The judge NEVER drops a write. A judge failure is LOUD: no
    write, a visible error, no deterministic fallback.

    Pass a mutable `details` dict to receive extra facts about the outcome the plain
    token cannot carry — on a collapse the gate writes `details['target_id']` (and, for
    a task, `details['deadline']`) so the caller can report the surviving entry's
    REAL time honestly, even when the collapse was fuzzy (target text ≠ proposal text).

    With no judge (the maintainer / non-interactive path — out of Phase-1 scope) the
    write applies directly, exactly as before. Returns the outcome token:
    'applied' | 'pending' | 'task' | 'task_exists' | 'event' | 'collapsed'
    | 'conflict' | 'judge_error' | 'retired' | 'absent' | 'denied'
    | 'completed' | 'rescheduled' | 'task_absent' | 'reschedule_invalid'
    (the fast-track task edits)."""
    # A retire (forget) never goes through the judge — it is a DELETE, not a belief to
    # skeptic-check. It has its own path: a fast-track namespace retires straight; a
    # user-owned belief (people/projects/…) is never silently dropped — it ASKS first
    # (a confirmation card), then deletes on approve. Universal: any agent's `retire:` is
    # confirmed the same way, for any belief the user owns.
    token = _ACTION.set(action_id) if action_id else None
    try:
        return await _apply_routed(proposal, conversation_id=conversation_id, author=author,
                                   judge=judge, message=message, details=details)
    finally:
        if token is not None:
            _ACTION.reset(token)


async def _apply_routed(proposal, *, conversation_id: str, author: str, judge, message: str,
                        details: dict | None) -> str:
    if proposal.intent.startswith("retire:"):
        return await _retire(proposal, conversation_id=conversation_id, author=author)
    # An ACTION (`act:<verb>`, e.g. act:reply) is never a memory write and never judged: it
    # ALWAYS cards, whoever proposes it. On approve the gate hands the proposal back to the
    # engine, which lets the proposing agent perform it — the one execution Ken does.
    if proposal.intent.startswith("act:"):
        # the key carries the TARGET (a stable id of meta: channel+ts …) as well as the text, so
        # the same draft to two threads is two cards, never one card that swallows the other.
        meta = getattr(proposal, "meta", None) or {}
        target = stable_id(json.dumps(meta, sort_keys=True))
        # the card must say WHERE the act lands and WHAT it answers — generic keys the
        # proposer fills (`to`: "#eng · Jack", `context`: the message being answered), so a
        # Yes is never a surprise. The engine never learns what "to" means.
        await _raise_ask(proposal, kind=f"{proposal.intent}:{norm(proposal.content)}:{target}",
                         ask_type="act", conversation_id=conversation_id, author=author,
                         subject=meta.get("to") or None, note=meta.get("context") or None)
        return "pending"
    # A hydrate BATCH (`batch:hydrate`) goes into namespaces that are empty: nothing to
    # collapse onto, so no judge — one card carries every item, and resolve applies keep,
    # edit or drop per item in one transaction. Never a silent write, whoever proposes it.
    if proposal.intent.startswith("batch:"):
        items = (getattr(proposal, "meta", None) or {}).get("items") or []
        if not items or not all(isinstance(i, dict) and all(isinstance(i.get(k), str) and i.get(k)
                                                             for k in ("kind", "name", "note")) for i in items):
            await _db.audit("batch_invalid", proposal.intent, f"by {author}: items need kind, name, note")
            return "invalid"
        await _raise_ask(proposal, kind=f"{proposal.intent}:{stable_id(json.dumps(items, sort_keys=True))}",
                         ask_type="batch", conversation_id=conversation_id, author=author,
                         note=_batch_note(items))
        return "pending"
    # The untrusted-source choke, BEFORE the judge routes: a task/event proposed from
    # WORLD-class content (e.g. a Slack agent reading someone else's message) is
    # prompt-injectable and must never fast-track OR silently collapse onto an existing
    # row. A task ASKS (a card the user approves); an event is DENIED — nothing may write
    # immutable history from untrusted content, and no agent proposes WORLD events today.
    # (A structured belief falls through to the judge here; its terminal write is carded for
    # WORLD in _apply — the ask-namespace path already cards, the fast-track path too — so no
    # WORLD structured write fast-tracks either.)
    if proposal.provenance is Provenance.WORLD:
        if proposal.intent == "task":
            project = await _project_of(getattr(proposal, "meta", None))
            ask_kind = f"task:{norm(proposal.content)}"
            note = f"project: {title_of(project)}" if project else None
            if note:     # `context` is what pending() and the push read back, so the project shows on every surface
                proposal = replace(proposal, meta={**(getattr(proposal, "meta", None) or {}), "context": note})
            # the same task proposed again while its card is open: the card adopts a project only
            # when it has none (a task's own rule), and the reply names the project the CARD holds
            open_ask = await _open_ask_for(conversation_id, ask_kind)
            if open_ask is not None and open_ask["content"] == proposal.content:
                old = json.loads(open_ask["meta"]) if open_ask["meta"] else {}
                if project and not old.get("project"):
                    async with _db.get_db() as c:
                        await c.execute("UPDATE pending SET meta=? WHERE ask_id=?",
                                        (json.dumps({**old, "project": project, "context": note}), open_ask["ask_id"]))
                        await c.commit()
                else:
                    project = await _project_of(old)
            if details is not None:
                details["project"] = project
            await _raise_ask(proposal, kind=ask_kind, ask_type="task",
                             conversation_id=conversation_id, author=author, note=note)
            return "pending"
        if proposal.intent == "event":
            # history from an untrusted source is never written silently — it asks, like a
            # task does, and lands only on the user's Yes (resolve's event branch)
            await _raise_ask(proposal, kind=f"event:{norm(proposal.content)}", ask_type="event",
                             conversation_id=conversation_id, author=author)
            return "pending"
        if proposal.intent in ("complete", "discard", "reschedule", "assign", "link_tasks"):
            # a task EDIT from untrusted content never lands silently either: it becomes the
            # same "done?" / "Discard?" / "Reschedule?" card the maintainer raises, performed by kip on Yes
            task = await store.task_by_id(proposal.content)
            if task is None or task.status != "open":
                return "task_absent"
            who = _reporter(proposal)
            # one open kip card per task: a second report (or the maintainer's card) does not
            # stack another; the reporting action is stamped onto the open card so its reply
            # carries the card id
            verb = {"complete": "mark_done", "discard": "discard", "reschedule": "reschedule", "assign": "assign",
                    "link_tasks": "link_tasks"}[proposal.intent]
            open_card = await _open_task_card(conversation_id, task.id)
            pm = getattr(proposal, "meta", None) or {}
            asks = {"assign": {"owner": pm.get("owner")}, "reschedule": {"at": proposal.at},
                    "link_tasks": {"depends_on": pm.get("depends_on"), "remove": bool(pm.get("remove"))}}.get(proposal.intent, {})
            if open_card is not None and (not str(open_card["kind"]).startswith(f"act:{verb}:") or any(
                    (json.loads(open_card["meta"] or "{}")).get(k) != v for k, v in asks.items())):
                # a DIFFERENT question is open on this task (done? vs discard?, or another owner,
                # time or dependency): the newer report supersedes it, so a Yes applies only it
                await withdraw(open_card["ask_id"], reason=f"superseded by a {verb} report")
                open_card = None
            if open_card is not None:
                if _ACTION.get():
                    async with _db.get_db() as c:
                        await c.execute("UPDATE pending SET action_id=? WHERE ask_id=?", (_ACTION.get(), open_card["ask_id"]))
                        await c.commit()
                return "pending"
            if proposal.intent == "discard":
                why = _reason(proposal)
                act = Proposal(intent="act:discard", content=f"{task.text} — discard?", evidence=list(proposal.evidence),
                               provenance=Provenance.AGENT_INFERRED,
                               meta={"task_id": task.id, "reason": why, "via": "kip", "to": task.text, "origin": "report",
                                     "context": f"{who} says it no longer matters" + (f": {why}" if why else "")
                                                + ". Yes drops it (not counted as done), No leaves it open."})
            elif proposal.intent in ("assign", "link_tasks"):
                if proposal.intent == "assign":
                    name = owner_label(pm.get("owner")) or "you"
                    ask, says = f"{task.text} \u2192 {name}", f"{name} does it"
                else:
                    dep = await store.task_by_id(str(pm.get("depends_on") or ""))
                    dep_text = dep.text if dep else "?"
                    ask = f"{task.text} {'no longer waits' if pm.get('remove') else 'waits'} on {dep_text}"
                    says = ask
                act = Proposal(intent=f"act:{verb}", content=ask, evidence=list(proposal.evidence),
                               provenance=Provenance.AGENT_INFERRED,
                               meta={**pm, "task_id": task.id, "via": "kip", "to": task.text, "origin": "report",
                                     "context": f"{who} reports: {says}. Yes applies it, No leaves the task as it is."})
            elif proposal.intent == "complete":
                act = Proposal(intent="act:mark_done", content=f"{task.text} — done?", evidence=list(proposal.evidence),
                               provenance=Provenance.AGENT_INFERRED,
                               meta={"task_id": task.id, "via": "kip", "to": task.text, "origin": "report",
                                     "context": f"{who} reports it done. Yes marks it done, No leaves it open."})
            else:
                if proposal.at is None or gone(proposal.at, now_ms()):
                    await store.log_action(author, "reschedule_refused", proposal.content, "missing or past time")
                    return "reschedule_invalid"
                when = when_label(proposal.at, now_ms())
                act = Proposal(intent="act:reschedule", content=f"{task.text} \u2192 {when}", evidence=list(proposal.evidence),
                               provenance=Provenance.AGENT_INFERRED,
                               meta={"task_id": task.id, "at": proposal.at, "via": "kip", "to": task.text, "origin": "report",
                                     "context": f"{who} reports a new time: {when}. Yes moves it there, No leaves it where it is."})
            return await _apply_routed(act, conversation_id=conversation_id, author="kip",
                                       judge=None, message=message, details=details)
    if judge is None:
        return await _apply(proposal, conversation_id=conversation_id, author=author)
    return await _judged_apply(proposal, conversation_id=conversation_id,
                               author=author, judge=judge, message=message, details=details)


# ── the judge orchestration (runs before the write) ──────────────────────────

def _reporter(proposal) -> str:
    """Who a WORLD report came from, for a card: the evidence source, never a channel label."""
    src = next((e.source for e in proposal.evidence if e.source), "") or ""
    return src if src and not src.startswith(("mcp:", "http:", "cli:")) else "an agent"


async def _judged_apply(proposal, *, conversation_id, author, judge, message,
                        details=None) -> str:
    existing = await _slice_for(proposal)
    try:
        decision = await judge(proposal, message=message, existing=existing)
    except Exception as exc:  # noqa: BLE001 — ANY judge failure is loud, never a write
        await _error(conversation_id,
                     "I could not check that safely, so I did not save it.")
        await _db.audit("judge_error", proposal.intent, repr(exc))
        return "judge_error"

    # a person or project is identified by the NAME an agent typed, so the same person
    # arrives under different ids ("Lucas", "Lucas (Tallyhouse)"). Any hint that it is an
    # existing record — the judge or a name twin — becomes an
    # "Update X?" card showing both versions; never a silent keep, never a silent merge.
    entry, why = _identity_target(proposal, decision, existing)
    if entry is not None:
        return await _identity_ask(proposal, entry, why, conversation_id=conversation_id, author=author)

    verdict = decision.verdict
    if verdict is _judge.Verdict.KEEP:
        return await _apply(proposal, conversation_id=conversation_id, author=author)
    if verdict is _judge.Verdict.COLLAPSE:
        return await _collapse(proposal, decision, conversation_id=conversation_id,
                               author=author, details=details)
    if verdict is _judge.Verdict.CONFLICT:
        return await _conflict(proposal, decision,
                               conversation_id=conversation_id, author=author)
    return await _apply(proposal, conversation_id=conversation_id, author=author)


_IDENTITY_NS = ("people", "projects")


def _title_of(entry) -> str:
    subject = getattr(entry, "subject", "") or ""
    return subject if subject not in ("", "you") else entry.kind.split(".", 1)[-1].replace("_", " ")


def _tokens(text: str) -> set[str]:
    return {w for w in slug(text).split("_") if len(w) > 1}      # slug first: "(Tallyhouse)" == "tallyhouse"


def _identity_target(proposal, decision, existing: list):
    """The existing person/project this proposal most likely IS (the entry), and why: the
    judge named it (a cross-id collapse or conflict), or its name is a twin (every token of the shorter name inside the longer: "Lucas" ⊂ "Lucas
    Tallyhouse", but not "Michael Davis" vs "Michael Novak"). (None, "") when nothing
    points at one — including when the proposal's OWN id already exists: that is a plain
    same-kind update, and a user's earlier "not the same" must stick."""
    if not proposal.intent.startswith("structured:"):
        return None, ""
    kind = proposal.intent.split(":", 1)[1]
    ns, _, name = kind.partition(".")
    ids = {e.id: e for e in existing}
    if ns not in _IDENTITY_NS or kind in ids:
        return None, ""
    d = decision
    if d.verdict in (_judge.Verdict.COLLAPSE, _judge.Verdict.CONFLICT) \
            and d.target_id and d.target_id != kind and d.target_id in ids:
        return ids[d.target_id], "the judge"
    mine = _tokens(name)
    twins = []
    for e in existing:
        theirs = _tokens(_title_of(e))
        small, big = (mine, theirs) if len(mine) <= len(theirs) else (theirs, mine)
        if small and small <= big:
            twins.append(e)
    if len(twins) == 1:                       # two or more twins: let the plain card ask
        return twins[0], "the same name"
    return None, ""


def _identity_note(title: str, noun: str, why: str) -> str:
    return (f"Looks like the same {noun} as \u201c{title}\u201d ({why}). Yes adds this note to that "
            f"record (edit to rewrite it); No saves a new {noun}.")


async def _identity_ask(proposal, entry, why: str, *, conversation_id: str, author: str) -> str:
    """One "Update <existing>?" card, both versions shown. Its own ask type and key
    (`identity:<target>:<own>`), so it never evicts a same-kind or forget card on the
    target and one pair asks once. Yes MERGES the note into the existing record; No
    re-asks it as a new record under its own id — a wrong hint loses nothing."""
    kind = proposal.intent.split(":", 1)[1]
    noun = _NOUN.get(kind.split(".", 1)[0], "record")
    title = _title_of(entry)
    note = _identity_note(title, noun, why)
    if _link_note(getattr(proposal, "meta", None)):
        note += "\n" + _link_note(getattr(proposal, "meta", None))
    ask = replace(proposal, meta={**(getattr(proposal, "meta", None) or {}), "context": note, "to": title})
    await _raise_ask(ask, kind=f"identity:{entry.kind}:{kind}", ask_type="identity",
                     conversation_id=conversation_id, author=author, origin="identity",
                     existing_snapshot=_snapshot_of(entry), existing=entry.content, note=note, subject=title)
    return "conflict"


async def _resolve_identity(ask_id: str, p, *, approved: bool, edit: str, author: str) -> str:
    """Close an identity ask. No → the proposal is still unsaved: re-ask under its own id
    ('reasked'). Yes → merge into the target unless it moved or vanished under the card
    ('restaled': a fresh card). The swap of one ask for the next is one transaction."""
    _, target, own = p["kind"].split(":", 2)
    meta = json.loads(p["meta"]) if p["meta"] else {}
    evidence = store.evidence_load(p["evidence"])
    carried = {k: v for k, v in meta.items() if k not in ("context", "to")}   # task_id etc. ride on
    common = {"evidence": evidence, "conversation_id": p["conversation_id"], "author": p["author"]}
    current = next((b for b in await store.get_structured(prefix=target) if b.kind == target), None)

    async def as_new(text: str) -> None:
        own_now = await _belief_snapshot(own)             # own may exist by now: show it, guard it
        await _swap_ask(ask_id, content=edit or p["content"], kind=own, ask_type="belief",
                        origin="judge-conflict" if own_now else "consent-stakes",
                        existing_snapshot=own_now, existing=_snapshot_content(own_now),
                        note=_link_note(carried) or None, meta=carried, text=text, **common)

    if not approved:
        await _outcome_row(ask_id, p, approved=False)
        await _db.audit("consent_reject", p["kind"], f"rejected by {author}: not the same")
        await as_new("Not the same — save it as new?")
        return "reasked"
    if current is None:
        await _db.audit("consent_restale", p["kind"], f"target gone under ask; {author}")
        await as_new("That record is gone — save it as new?")
        return "restaled"
    if not edit and _norm_or_none(current.content) != _norm_or_none(_snapshot_content(p["existing_snapshot"])):
        await _db.audit("consent_restale", p["kind"], f"target changed under ask; {author}")
        await _swap_ask(ask_id, content=p["content"], kind=p["kind"], ask_type="identity", origin="identity",
                        existing_snapshot=_snapshot_of(current), existing=current.content,
                        note=meta.get("context"), subject=meta.get("to"), meta=meta,
                        text="That changed since I asked — here is the current value to confirm.", **common)
        return "restaled"
    await _outcome_row(ask_id, p, approved=True)
    if edit:
        content = edit
    elif norm(p["content"]) in norm(current.content):
        content = current.content
    else:
        content = f"{current.content.rstrip('. ')}. {p['content']}"
    entry = StructuredEntry(
        id=target, kind=target, subject=current.subject, content=content,
        evidence=list(current.evidence) + ([] if edit else evidence),
        confidence=1.0 if edit else max(0.8, current.confidence),
        # a plain Yes endorses the merge: a user-authored record stays user-authored; a
        # weaker one (agent, world) rises to user-instructed. An edit is the user's own word.
        provenance=Provenance.USER_STATED if edit else (
            current.provenance if current.provenance in (Provenance.USER_STATED, Provenance.USER_INSTRUCTED)
            else Provenance.USER_INSTRUCTED))
    entry.url = current.url or meta.get("url") or None
    async with _db.get_db() as c:
        await store.upsert_structured(entry, change_note=f"merged {own}; approved by {author}", conn=c)
        await _write_links(target, meta, evidence, conn=c)
        await c.execute("DELETE FROM pending WHERE ask_id=?", (ask_id,))
        await c.commit()
    await _confirm(p["conversation_id"], "Saved your version." if edit else "Added to the existing record.")
    return "applied"


async def _swap_ask(old_ask_id: str, *, content: str, evidence: list, kind: str, ask_type: str, origin: str,
                    existing_snapshot: str | None, existing: str | None, note: str | None,
                    conversation_id: str, author: str, text: str, meta: dict | None = None,
                    subject: str | None = None) -> None:
    """Replace one open ask with the next in ONE transaction, then post the message and
    the card — a crash between the two leaves either ask, never neither."""
    proposal = SimpleNamespace(content=content, evidence=evidence, meta=meta or None, at=None)
    ask_id = new_id()
    async with _db.get_db() as c:
        await c.execute("DELETE FROM pending WHERE ask_id=?", (old_ask_id,))
        await _save_pending(ask_id, proposal, kind=kind, conversation_id=conversation_id, author=author,
                            origin=origin, existing_snapshot=existing_snapshot, ask_type=ask_type, conn=c)
        await c.commit()
    await _confirm(conversation_id, text)
    await bus.append_renderable(_ask_card(
        ask_id, conversation_id=conversation_id, author=author, question=content, target=kind, note=note,
        subject=subject, existing=existing, proposed=content if existing is not None else None))


async def _slice_for(proposal) -> list:
    """The bounded slice the judge compares against: open tasks for a task, recent
    events for an event, same-namespace beliefs for a structured proposal."""
    intent = proposal.intent
    if intent == "task":
        return await store.open_tasks()
    if intent == "event":
        return await store.recent_events()
    if intent.startswith("structured:"):
        namespace = intent.split(":", 1)[1].split(".", 1)[0]
        return await store.get_structured(prefix=namespace + ".")
    return []


async def _collapse(proposal, decision, *, conversation_id, author, details=None) -> str:
    """A confident duplicate: never store a second copy. But a collapse may drop the
    write ONLY when a real matching target is confirmed to exist. A falsy, garbled, or
    stale `target_id` must NEVER silently destroy a write — so when the claimed duplicate
    cannot be confirmed, we fall back to keep and write the proposal through the normal
    apply path.

    A STRUCTURED collapse is honoured only when the value is essentially identical to the
    stored belief (a genuine no-op). A different value for the same belief is not a
    duplicate — it is a conflict, so we route it to an ask instead of silently dropping
    the new value (Finding 5). Tasks/events keep the plain collapse behaviour.

    For a confirmed task duplicate we still adopt a new time and/or evidence onto the
    existing one (fuzzy dedup cannot fall through add_task's exact-norm path, and the
    never-silently-reschedule guard still holds), and we surface the surviving task's id
    (and its resulting time) through `details` so the caller reports it honestly. Never
    silent: the outcome acknowledges either the collapse or the write."""
    target = await _collapse_target(proposal, decision.target_id)
    if target is None:
        return await _apply(proposal, conversation_id=conversation_id, author=author)

    if proposal.intent.startswith("structured:"):
        kind = proposal.intent.split(":", 1)[1]
        # a structured belief's identity IS its kind. A collapse only holds within the SAME
        # kind (the same belief slot). If the judge named a DIFFERENT belief as the duplicate
        # — a new person whose note happens to match another person's ('my manager') — it is
        # NOT a duplicate of THIS belief: write it as its own, never silently drop it onto the
        # other. (The judge sees the whole namespace slice, so a cross-kind COLLAPSE is possible.)
        if getattr(target, "id", None) != kind:
            return await _apply(proposal, conversation_id=conversation_id, author=author)
        # same belief: only a truly-identical value is a duplicate; a different value for
        # the same belief is a conflict the user must confirm, never a silent no-op.
        if norm(str(getattr(target, "content", ""))) != norm(str(proposal.content)):
            return await _conflict(proposal, decision,
                                   conversation_id=conversation_id, author=author)
        # a duplicate note can still carry NEW links (ones not stored yet). In a consented
        # namespace — or from WORLD content anywhere — they ASK on their own links card, and a
        # Yes writes the links ONLY, never the note; a fast-track namespace lands them now.
        meta = getattr(proposal, "meta", None) or {}
        new = await _new_links(kind, meta.get("links") or [])
        if new and (consent_level(kind) != "fast_track" or proposal.provenance is Provenance.WORLD):
            await _links_ask(proposal, kind, new, conversation_id=conversation_id, author=author)
            return "pending"
        await _write_links(kind, meta, list(proposal.evidence))

    if proposal.intent == "task":
        await store.adopt_onto_task(decision.target_id, deadline=proposal.at,
                                    evidence=list(proposal.evidence),
                                    project=await _project_of(getattr(proposal, "meta", None)))
        if details is not None:
            # anchor the caller's read-back on the SURVIVING task, not the proposal text
            # (a fuzzy collapse folds onto a differently-worded task) — so its real
            # resulting time is reported honestly, never "nothing new to add". The
            # resulting time mirrors adopt_onto_task's rule (keep an existing time, else
            # take the proposed one) — no extra fetch needed.
            kept = getattr(target, "deadline", None)
            details["target_id"] = decision.target_id
            details["deadline"] = kept if kept is not None else proposal.at
    await store.log_action(author, "collapsed", proposal.intent,
                           decision.reason or "duplicate")
    return "collapsed"


async def _collapse_target(proposal, target_id):
    """The existing entry a collapse claims to duplicate, confirmed against the same
    bounded slice the judge saw — or None when `target_id` is falsy, garbled, or names a
    row that is gone (e.g. completed between slice-fetch and now). Generic over the noun
    objects: each carries an `id` (a structured belief's id is its kind)."""
    if not target_id:
        return None
    return next((e for e in await _slice_for(proposal)
                 if getattr(e, "id", None) == target_id), None)


async def _conflict(proposal, decision, *, conversation_id, author) -> str:
    """A clash with an existing belief → always ask (overriding fast-track). The
    pending row is tagged judge-conflict and snapshots the belief being challenged, so
    the card renders existing-vs-proposed and resolve can re-validate. Only structured
    beliefs can conflict; a task/event 'conflict' degrades to a plain keep.

    The kind IS the belief id in the V1 singleton model (id = kind, subject = 'you'),
    and a structured conflict is same-kind by construction — so the snapshot is always
    the real current belief for this proposal's own kind."""
    if not proposal.intent.startswith("structured:"):
        return await _apply(proposal, conversation_id=conversation_id, author=author)
    kind = proposal.intent.split(":", 1)[1]
    snapshot = await _belief_snapshot(kind)
    existing = json.loads(snapshot)["content"] if snapshot is not None else None
    await _raise_ask(proposal, kind=kind, conversation_id=conversation_id, author=author,
                     origin="judge-conflict", existing_snapshot=snapshot, existing=existing,
                     note=_link_note(getattr(proposal, "meta", None)) or None)
    return "conflict"


async def _belief_snapshot(kind: str) -> str | None:
    """A JSON snapshot of the current belief for a kind, or None when none exists."""
    match = next((b for b in await store.get_structured(prefix=kind) if b.kind == kind), None)
    return None if match is None else _snapshot_of(match)


def _snapshot_of(entry) -> str:
    return json.dumps({"kind": entry.kind, "content": entry.content,
                       "provenance": entry.provenance.value, "updated_at": entry.updated_at})


def _snapshot_content(snapshot: str | None) -> str | None:
    """The belief's stored value out of a JSON snapshot, or None. Used at resolve to
    tell whether the belief moved under an open conflict ask (the stale-approval guard)."""
    if not snapshot:
        return None
    try:
        return json.loads(snapshot).get("content")
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _norm_or_none(content: str | None) -> str | None:
    """A norm() comparison key for a belief value, or None when the belief is absent.
    The stale-approval guard compares with the SAME norm() the collapse path uses, so a
    cosmetic-only change ('black' -> 'Black') is NOT a real move, while a retired belief
    (None) still differs from any present value."""
    return norm(str(content)) if content is not None else None


async def _reraise_conflict(p, current_snapshot: str | None, *, stale_ask_id: str) -> None:
    """The belief moved under an open conflict ask (stale approval): raise a FRESH
    conflict against the NOW-current value, carrying the same proposed content, and tell
    the user it changed.

    The stale ask's DELETE and the fresh ask's INSERT commit in ONE transaction — the same
    discipline the apply path uses — so a crash between the two leaves EITHER the old ask
    or the new one, never neither: the pending proposal can never silently vanish. The
    user message and the fresh card are posted only after the row is durable. A
    SimpleNamespace stands in for the original proposal — _save_pending/_ask_card need only
    its content and evidence, and importing Proposal here would cycle back through agents."""
    await _swap_ask(stale_ask_id, content=p["content"], evidence=store.evidence_load(p["evidence"]),
                    kind=p["kind"], ask_type=p["ask_type"] or "belief", origin="judge-conflict",
                    existing_snapshot=current_snapshot, existing=_snapshot_content(current_snapshot), note=None,
                    meta=json.loads(p["meta"]) if p["meta"] else None,
                    conversation_id=p["conversation_id"], author=p["author"],
                    text="That changed since I asked — here is the current value to confirm.")


# ── forget (retire a belief) ─────────────────────────────────────────────────

async def _retire(proposal, *, conversation_id: str, author: str) -> str:
    """Delete a stored belief. A fast-track namespace deletes straight — but ONLY from a
    trusted source; a WORLD-provenance retire (untrusted, prompt-injectable) never silently
    deletes, it asks like any belief (mirrors the write-side WORLD choke). A user-owned belief
    (people/projects/…) is never silently dropped either — a confirmation card, delete on
    approve (see `resolve`'s retire branch). Absent belief → nothing to forget."""
    kind = proposal.intent.split(":", 1)[1]
    current = next((e for e in await store.get_structured(prefix=kind) if e.kind == kind), None)
    if current is None:
        return "absent"                          # nothing saved under that kind — nothing to forget
    trusted = proposal.provenance is not Provenance.WORLD
    if consent_level(kind) == "fast_track" and trusted:
        ok = await store.delete_structured(kind)
        if ok:
            await store.log_action(author, "forget_fast_track", kind, "retired (fast-track)")
        return "retired" if ok else "absent"
    # the card names WHAT is being forgotten (blank content would render a blank card): a
    # human question + the current value, so approve is an informed choice.
    name = kind.split(".", 1)[-1].replace("_", " ")
    ask = replace(proposal, content=f"Forget what I know about {name}?")
    await _raise_ask(ask, kind=f"retire:{kind}", ask_type="retire", note=current.content,
                     conversation_id=conversation_id, author=author)
    return "pending"


# ── the direct applier (today's path; used on 'keep' and with no judge) ───────

async def _apply(proposal, *, conversation_id: str, author: str) -> str:
    intent = proposal.intent

    if intent.startswith("structured:"):
        kind = intent.split(":", 1)[1]
        # a NEW record in the user's own words (their token, relayed as theirs) needs no card:
        # they just said it. A change to a record they have, agent words and WORLD input still ask.
        own_new = proposal.provenance in (Provenance.USER_INSTRUCTED, Provenance.USER_STATED) \
            and not any(e.kind == kind for e in await store.get_structured(kind)) \
            and not await _open_ask_for(conversation_id, kind)
        if consent_level(kind) == "fast_track" or own_new:
            # untrusted content must never fast-track silently, even into a fast-track
            # namespace: a WORLD-provenance preference/pattern cards like any other WORLD
            # write. KEEP is the only structured verdict that writes a new value here
            # (COLLAPSE no-ops or routes to a conflict card, CONFLICT already cards), so
            # guarding this branch closes the whole WORLD-structured silent-write hole.
            if proposal.provenance is Provenance.WORLD:
                await _raise_ask(proposal, kind=kind, conversation_id=conversation_id, author=author)
                return "pending"
            meta = getattr(proposal, "meta", None) or {}
            entry = StructuredEntry(
                id=kind,                       # V1: one belief per kind (singleton)
                kind=kind, subject="you", content=proposal.content,
                evidence=list(proposal.evidence), confidence=0.6,
                provenance=proposal.provenance, url=meta.get("url") or None)
            async with _db.get_db() as c:              # the record and its links land together
                await store.upsert_structured(entry, change_note=f"proposed by {author}", conn=c)
                await _write_links(kind, meta, list(proposal.evidence), conn=c)
                await c.commit()
            return "applied"

        await _raise_ask(proposal, kind=kind, conversation_id=conversation_id, author=author,
                         note=_link_note(getattr(proposal, "meta", None)) or None)
        return "pending"

    # (retire: is handled by `apply`'s own `_retire` path, before the judge — never here.)

    if intent == "task":
        # a WORLD-provenance task never reaches here — `apply`'s choke asks first (before
        # the judge), so a collapse/conflict verdict can't silently mutate a task either.
        _id, created = await store.add_task(
            proposal.content, proposal.provenance, deadline=proposal.at,
            evidence=list(proposal.evidence),
            project=await _project_of(getattr(proposal, "meta", None)),
            owner=(getattr(proposal, "meta", None) or {}).get("owner"))
        return "task" if created else "task_exists"   # idempotent: a re-proposed task is a no-op

    if intent == "complete":
        # a user-instructed task edit — fast-track (no judge, no card) but AUDITED like
        # any applied write, so it leaves a trail. The proposal names the task by id
        # (content); the store validates it only ever flips an EXISTING open task.
        ok = await store.complete_task(proposal.content)
        await store.log_action(author, "completed" if ok else "complete_noop",
                               proposal.content, "marked done" if ok else "not open")
        return "completed" if ok else "task_absent"

    if intent == "discard":
        # a user-instructed drop: the task was NOT done, it stopped mattering. Fast-track and
        # audited like complete, but a separate status so history and done counts stay true.
        why = _reason(proposal)
        ok = await store.discard_task(proposal.content, why)
        await store.log_action(author, "discarded" if ok else "discard_noop",
                               proposal.content, (why or "dropped") if ok else "not open")
        return "discarded" if ok else "task_absent"

    if intent == "reschedule":
        # a user-instructed re-time — the one path allowed to overwrite an existing
        # time. Fast-track but audited; validated to an existing open task by the store.
        # Floor (defense-in-depth for ANY proposer, not just the Assistant closure): a
        # missing time would null the schedule and a past time can never fire a nudge, so
        # refuse both LOUDLY with a non-success token rather than apply a dead schedule.
        if proposal.at is None or gone(proposal.at, now_ms()):
            await store.log_action(author, "reschedule_refused", proposal.content,
                                   "missing or past time")
            return "reschedule_invalid"
        ok = await store.reschedule(proposal.content, proposal.at)
        await store.log_action(author, "rescheduled" if ok else "reschedule_noop",
                               proposal.content, str(proposal.at) if ok else "not open")
        return "rescheduled" if ok else "task_absent"

    if intent == "assign":
        # a user-instructed hand-off: fast-track, audited, validated to an existing open task
        owner = (getattr(proposal, "meta", None) or {}).get("owner") or None
        ok = await store.set_task_owner(proposal.content, owner)
        await store.log_action(author, "assigned" if ok else "assign_noop", proposal.content,
                               (owner or "the user") if ok else "not open")
        return "assigned" if ok else "task_absent"

    if intent == "link_tasks":
        # task A waits on task B (or no longer does). A cycle is refused by the store.
        meta = getattr(proposal, "meta", None) or {}
        dep = str(meta.get("depends_on") or "")
        if meta.get("remove"):
            out = "unlinked" if await store.unlink_task(proposal.content, dep) else "link_absent"
        else:
            out = await store.link_task(proposal.content, dep)
        await store.log_action(author, out, proposal.content, f"depends on {dep}")
        return out

    if intent == "event":
        # an event gets the time it HAPPENED (proposal.at), not ingest time — falling
        # back to now only when the proposer left it unset.
        at = proposal.at if proposal.at is not None else now_ms()
        until = await _until_of((getattr(proposal, "meta", None) or {}).get("until"), proposal.intent)
        async with _db.get_db() as c:
            await c.execute(
                "INSERT INTO events (id, summary, at, provenance, evidence, until) VALUES (?,?,?,?,?,?)",
                (new_id(), proposal.content, at, proposal.provenance.value,
                 store.evidence_dump(list(proposal.evidence)), until))
            await c.commit()
        return "event"

    raise ValueError(f"unknown proposal intent: {intent!r}")


async def resolve(ask_id: str, *, approved: bool, conversation_id: str | None = None,
                  author: str = "you", edited: str | None = None,
                  details: dict | None = None) -> str:
    """Close a pending ask by its stable id, on the user's decision. The applier
    is deterministic: 'approve' applies the stored write, 'reject' drops it. A
    stale, superseded, or replayed tap finds no pending row and is a safe no-op.
    Returns: 'applied' | 'rejected' | 'no_pending' | 'restaled' | 'act' (an approved
    action: nothing written; `details` carries the proposal for the engine to dispatch).

    `edited` (approve only) is the user's own text, replacing the proposed value — a
    confirm-AND-correct. Because the user just authored it, the write becomes USER_STATED
    (trusted), not the untrusted source it came from, and the judge-conflict stale dance is
    skipped (a freshly-typed value is never stale). A retire has no editable value; it ignores
    `edited`.

    A genuine approve/reject posts a short outcome (Saved. / Not saved.) so an
    interface can clear its spinner. A no_pending outcome (superseded, already
    answered, or a crash-replayed tap) posts NOTHING — a confirmation there would
    contradict the real one already shown ('Saved.' then 'no longer open'). The
    interface clears its spinner for such a tap by noticing the ask left `pending`,
    not by a message from here.

    A judge-conflict Yes RE-VALIDATES first: if the belief moved since the card went up
    (someone changed it under the ask), the stored proposed value is stale — we do NOT
    apply it. We drop the stale ask, raise a FRESH conflict against the current value,
    and tell the user it changed. Returns 'restaled' in that case."""
    p = await _load_pending(ask_id)
    if p is None:
        return "no_pending"                    # silent: the real outcome was already posted
    if conversation_id is not None and p["conversation_id"] != conversation_id:
        return "no_pending"                    # a token resolves only its own conversation's ask
    if p["ask_type"] == "identity":
        return await _resolve_identity(ask_id, p, approved=approved, edit=(edited or "").strip(), author=author)
    if not approved:
        await _outcome_row(ask_id, p, approved=False)
        await _delete_pending(ask_id)
        await _db.audit("consent_reject", p["kind"], f"rejected by {author}")
        # a rejected forget KEEPS the belief; a rejected act is not done; a save isn't saved.
        await _confirm(p["conversation_id"], {"retire": "Kept.", "act": "Not sent.", "event": "Not logged.", "links": "Not linked."}
                       .get(p["ask_type"], "Not saved."))
        return "rejected"

    # an approved ACT writes nothing: clear the ask, hand the stored proposal back through
    # `details` (the engine dispatches it to the agent's act()). No confirmation here —
    # the agent's own outcome line is the confirmation.
    edit = (edited or "").strip()      # the user's own text — a confirm-AND-correct (approve only)

    # an approved ACT writes nothing and is NOT closed here: the engine performs it and closes
    # the ask (`close_act`) only when the agent succeeded, so a failed send can be retried with
    # a second tap. The user's edit replaces the draft — their words, USER_STATED.
    if p["ask_type"] == "act":
        await _outcome_row(ask_id, p, approved=True)
        await _db.audit("act_approved", p["kind"], f"by {author}; proposed by {p['author']}")
        if details is not None:
            details["proposal"] = Proposal(
                intent=p["kind"].split(":", 2)[0] + ":" + p["kind"].split(":", 2)[1],
                content=edit or p["content"], evidence=store.evidence_load(p["evidence"]),
                provenance=Provenance.USER_STATED if edit else Provenance.USER_INSTRUCTED,
                meta=json.loads(p["meta"]) if p["meta"] else {})
            details["author"] = p["author"]
            details["conversation_id"] = p["conversation_id"]
            details["ask_id"] = ask_id
        return "act"

    # stale-approval guard: a judge-conflict Yes only applies if the belief it was
    # raised against still holds. If the current belief moved under the open ask, the
    # proposed value was chosen against a value that no longer exists — never apply it
    # blindly. Re-raise a fresh conflict against the CURRENT value instead. An EDITED value
    # is skipped here — the user just typed it against what the card showed, so it is never
    # stale; their explicit value wins.
    if p["origin"] == "judge-conflict" and not edit:
        current = await _belief_snapshot(p["kind"])
        current_content = _snapshot_content(current)
        # already-satisfied: the current belief already IS the proposed value (norm-equal).
        # Re-raising "you have milk — save milk?" would be pointless. Treat it as applied
        # idempotently and relay an honest outcome — the belief is already what you want.
        if current_content is not None and norm(str(current_content)) == norm(str(p["content"])):
            await _outcome_row(ask_id, p, approved=True)
            await _delete_pending(ask_id)
            await _db.audit("consent_satisfied", p["kind"], f"already set; {author}")
            await _confirm(p["conversation_id"],
                           "That is already what you have — nothing to change.")
            return "applied"
        # stale-approval guard, compared with the SAME norm() the collapse path uses: a
        # cosmetic-only change ('black' -> 'Black') is NOT a real move and does not re-raise,
        # while a MEANINGFUL move (including the belief being retired to absent) re-raises a
        # fresh conflict against the current value — atomically (see _reraise_conflict).
        if _norm_or_none(current_content) != _norm_or_none(_snapshot_content(p["existing_snapshot"])):
            await _db.audit("consent_restale", p["kind"], f"belief changed under ask; {author}")
            await _reraise_conflict(p, current, stale_ask_id=ask_id)
            return "restaled"

    # a hydrate batch: keep, edit or drop PER ITEM (`edited` = JSON), every kept record written
    # and the ask cleared in ONE transaction — a crash can never leave half a batch saved.
    # Unreadable edits keep the card open and say so; nothing is guessed.
    if p["ask_type"] == "batch":
        try:
            edits = _batch_edits(edit)
        except ValueError:
            await _confirm(p["conversation_id"], "I could not read the edits, so the card stays open.")
            return "edit_invalid"
        items = (json.loads(p["meta"]) if p["meta"] else {}).get("items") or []
        unknown = sorted(set(edits) - {norm(i["name"]) for i in items})
        if unknown:                                            # a typo must not silently keep a record
            await _confirm(p["conversation_id"], f"No such item on this card: {', '.join(unknown)}. The card stays open.")
            return "edit_invalid"
        await _outcome_row(ask_id, p, approved=True)
        # the empty-namespace promise held at propose time; a record that exists NOW (another
        # batch, a remember in between) is never overwritten judge-free — kept unless the user
        # rewrote it on this card, which is their own explicit word
        existing = {e.kind for e in await store.get_structured()}
        kept, dropped, held = [], [], []
        async with _db.get_db() as c:
            for it in items:
                e = edits.get(norm(it["name"]), {})
                if e.get("drop"):
                    dropped.append(it["name"])
                    continue
                own = bool(e.get("note"))                       # the user rewrote this one
                kind = f"{it['kind']}.{slug(it['name'])}"
                if kind in existing and not own:
                    held.append(it["name"])
                    continue
                await store.upsert_structured(StructuredEntry(
                    id=kind, kind=kind, subject=it["name"], content=str(e.get("note") or it["note"]),
                    evidence=[] if own else store.evidence_load(p["evidence"]),
                    confidence=1.0 if own else 0.8,
                    provenance=Provenance.USER_STATED if own else Provenance.USER_INSTRUCTED,
                    url=it.get("url") or None),
                    change_note=f"hydrated; approved by {author}", conn=c)
                kept.append(it["name"])
            for it in items:                           # links last: a batch-mate now exists (or was dropped)
                if it["name"] in kept or it["name"] in held:
                    await _write_links(f"{it['kind']}.{slug(it['name'])}", it, store.evidence_load(p["evidence"]), conn=c)
            await c.execute("DELETE FROM pending WHERE ask_id=?", (ask_id,))
            await c.commit()
        await _db.audit("hydrate_batch", ask_id, json.dumps({"kept": kept, "dropped": dropped, "held": held}))
        await _confirm(p["conversation_id"], f"Saved {len(kept)}." + (f" Dropped {len(dropped)}." if dropped else "")
                       + (f" Already known, kept as is: {', '.join(held)}." if held else ""))
        return "applied"

    # a retire ask DELETES the belief (dispatched by ask_type). Delete the belief and clear the
    # ask in ONE transaction, so a crash can't half-forget. Absent already → still "applied" (the
    # end state the user asked for). p["kind"] is "retire:<belief kind>".
    await _outcome_row(ask_id, p, approved=True)   # past the stale guard: a REAL approve
    if p["ask_type"] == "retire":
        target = p["kind"].split(":", 1)[1]
        async with _db.get_db() as c:
            await store.delete_structured(target, conn=c)
            await c.execute("DELETE FROM pending WHERE ask_id=?", (ask_id,))
            await c.commit()
        await store.log_action(author, "consent_approve_forget", target, "forgotten via card")
        await _confirm(p["conversation_id"], "Forgotten.")
        return "applied"

    # a task ask applies as a TASK, not a belief (dispatched by ask_type, never by a kind
    # prefix that a belief kind could collide with). Add the task and clear the ask in ONE
    # transaction, so a crash can't leave a task the user then rejects (mirrors the belief path).
    if p["ask_type"] == "links":
        # links only: the record's note, subject, evidence and provenance are untouched. A
        # record retired under the card leaves nothing to link — say so, never guess.
        rk = p["kind"].split(":", 1)[1]
        meta = json.loads(p["meta"]) if p["meta"] else {}
        async with _db.get_db() as c:
            n = await store.set_links(rk, meta.get("links") or [], evidence=store.evidence_load(p["evidence"]), conn=c)
            await c.execute("DELETE FROM pending WHERE ask_id=?", (ask_id,))
            await c.commit()
        await store.log_action(author, "consent_approve_links", rk, f"{n} link(s)")
        await _confirm(p["conversation_id"], f"Linked ({n})." if n else "That record is gone — nothing linked.")
        return "applied"

    if p["ask_type"] == "event":
        # an approved event from an untrusted source: the user vouched for it, so it is theirs
        until = await _until_of((json.loads(p["meta"]) if p["meta"] else {}).get("until"), p["kind"])
        async with _db.get_db() as c:
            await c.execute(
                "INSERT INTO events (id, summary, at, provenance, evidence, until) VALUES (?,?,?,?,?,?)",
                (new_id(), edit or p["content"], p["at"] if p["at"] is not None else now_ms(),
                 (Provenance.USER_STATED if edit else Provenance.USER_INSTRUCTED).value, p["evidence"], until))
            await c.execute("DELETE FROM pending WHERE ask_id=?", (ask_id,))
            await c.commit()
        await store.log_action(author, "consent_approve_event", p["kind"], "logged via card")
        await _confirm(p["conversation_id"], "Logged.")
        return "applied"

    if p["ask_type"] == "task":
        # a time that passed while the card waited can never fire a nudge — drop it to None
        # (matches the assistant's add_task floor), so approving late doesn't queue a stale one.
        at = p["at"] if (p["at"] is None or p["at"] >= now_ms()) else None
        text = edit or p["content"]        # the user's edit replaces the proposed task text
        meta = json.loads(p["meta"]) if p["meta"] else {}
        project = await _project_of(meta)   # gone since the card? then none
        async with _db.get_db() as c:
            _id, created = await store.add_task(
                text, Provenance.USER_INSTRUCTED, deadline=at,
                evidence=store.evidence_load(p["evidence"]), project=project, owner=meta.get("owner"), conn=c)
            await c.execute("DELETE FROM pending WHERE ask_id=?", (ask_id,))
            await c.commit()
        await store.log_action(author, "consent_approve_task", text, "approved via card")
        await _confirm(p["conversation_id"], "Saved." if created else "Already on your list.")
        return "applied"

    # you approved -> the write is user-authored. apply the write and clear the
    # ask in ONE transaction, so a mid-write failure cannot lose the approval — it
    # stays pending and a re-tap re-applies. (fixes the delete-before-write gap)
    # An edit is the user STATING the fact in their own words: trusted (USER_STATED), full
    # confidence, and the proposal's external evidence no longer substantiates the stored value
    # (they rewrote it) so it is dropped. A plain approve endorses the proposal as-is — it keeps
    # the proposed evidence and the proposal-path confidence.
    content = edit or p["content"]
    meta = json.loads(p["meta"]) if p["meta"] else {}
    entry = StructuredEntry(
        id=p["kind"], kind=p["kind"], subject="you", content=content,
        evidence=[] if edit else store.evidence_load(p["evidence"]),
        confidence=1.0 if edit else 0.8,
        provenance=Provenance.USER_STATED if edit else Provenance.USER_INSTRUCTED,
        url=meta.get("url") or None)
    note = f"edited + approved by {author}" if edit else f"approved by {author}; proposed by {p['author']}"
    async with _db.get_db() as c:
        await store.upsert_structured(entry, change_note=note, conn=c)
        await _write_links(p["kind"], meta, store.evidence_load(p["evidence"]), conn=c)
        await c.execute("DELETE FROM pending WHERE ask_id=?", (ask_id,))
        await c.commit()
    await _confirm(p["conversation_id"], "Saved your version." if edit else "Saved.")
    return "applied"


async def _until_of(raw, ref: str) -> int | None:
    """An event's `until` as ms epoch, or None. Anything but a whole number (an ISO string
    from a caller that skipped the capture's parser) is dropped and audited — a TEXT in the
    INTEGER column would compare greater than every integer and never expire."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        await _db.audit("until_dropped", ref, f"not an integer: {raw!r}")
        return None
    return raw


def _batch_edits(edit: str) -> dict[str, dict]:
    """The per-item decisions on a batch card, keyed by the item's normalised name. Accepts a
    JSON list of {name, note?, drop?} or a {name: note | null} map (null = drop). Empty = keep
    all as proposed. Anything else raises ValueError."""
    if not edit:
        return {}
    data = json.loads(edit)
    if isinstance(data, dict):                     # {name: note | null}: null = drop, "" = keep
        data = [{"name": k, "drop": True} if v is None else
                ({"name": k, **v} if isinstance(v, dict) else {"name": k, "note": v}) for k, v in data.items()]
    if not isinstance(data, list) or not all(
            isinstance(d, dict) and isinstance(d.get("name"), str) and d.get("name")
            and isinstance(d.get("note", ""), str) and isinstance(d.get("drop", False), bool) for d in data):
        raise ValueError("edits must be a list of {name, note?: str, drop?: bool}")
    return {norm(d["name"]): {"note": d.get("note", "").strip(), "drop": d.get("drop", False)} for d in data}


def _batch_note(items: list) -> str:
    """The card's body for a batch: one line per item (plus its url and links), so the user
    decides in the open."""
    out = []
    for i in items:
        out.append(f"- {i['name']} ({_NOUN.get(i['kind'], i['kind'])}): {i['note']}")
        extra = _link_note(i)
        if extra:
            out += ["    " + ln for ln in extra.splitlines()]
    return "\n".join(out)


async def _outcome_row(ask_id: str, p, *, approved: bool) -> None:
    """The eval row `ken eval` reads: who proposed what kind of card, the tap, and how long it
    waited. JSON, so an author name with a space parses. Written only for a REAL outcome —
    never for a stale re-raise or a no_pending tap."""
    await _db.audit("card_outcome", ask_id, json.dumps({
        "author": p["author"], "ask_type": p["ask_type"] or "belief",
        "outcome": "approved" if approved else "rejected",
        "age_ms": max(0, now_ms() - (p["created_at"] or now_ms()))}))


async def withdraw(ask_id: str, *, reason: str) -> None:
    """Take an open ask back on purpose (its author changed its mind), audited. A tap on the
    old card is then a safe no-op. Never used on a card the user is mid-answer on — a stale
    tap resolves to no_pending, nothing is applied."""
    await _db.audit("card_withdrawn", ask_id, reason)
    await _delete_pending(ask_id)


async def close_act(ask_id: str) -> None:
    """Close an approved act's ask AFTER the agent performed it (the engine calls this). Until
    then the row stays, so a failed action can be retried with another tap."""
    await _delete_pending(ask_id)


async def _project_of(meta) -> str | None:
    """The project a task proposal names, if that project record EXISTS now. The gate is the
    single applier: a task never points at a project Ken does not hold (a stale card, a
    forgotten project, a caller that skipped the resolver)."""
    kind = str((meta or {}).get("project") or "")
    if not kind.startswith("projects."):
        return None
    return kind if any(e.kind == kind for e in await store.get_structured(prefix=kind)) else None


async def _raise_ask(proposal, *, kind: str, conversation_id: str, author: str,
                     origin: str = "consent-stakes", existing_snapshot: str | None = None,
                     existing: str | None = None, note: str | None = None,
                     ask_type: str = "belief", subject: str | None = None) -> None:
    """Record a pending ask and post its Yes/No card. Keeps at most one open ask per
    (conversation, target): an identical open ask is a no-op (no duplicate card), and a
    different proposal for the same target supersedes the old one — its card's ask id is
    now gone, so a tap on it resolves to a safe no-op. `origin`/`existing_snapshot` tag a
    judge-raised conflict so resolve can re-validate; `existing` (the current value) rides
    into the card body so the interface can render existing-vs-proposed."""
    open_ask = await _open_ask_for(conversation_id, kind)
    if open_ask is not None and open_ask["content"] == proposal.content:
        # identical ask already open — no duplicate card; but the action that re-raised it
        # (a retry, a second client) must still get the card id back in its reply
        if _ACTION.get():
            async with _db.get_db() as c:
                await c.execute("UPDATE pending SET action_id=? WHERE ask_id=?", (_ACTION.get(), open_ask["ask_id"]))
                await c.commit()
        return
    if open_ask is not None:
        await _delete_pending(open_ask["ask_id"])
    ask_id = new_id()
    await _save_pending(ask_id, proposal, kind=kind, conversation_id=conversation_id,
                        author=author, origin=origin, existing_snapshot=existing_snapshot,
                        ask_type=ask_type)
    await bus.append_renderable(_ask_card(
        ask_id, conversation_id=conversation_id, author=author,
        question=proposal.content, target=kind, note=note, subject=subject,
        existing=existing, proposed=proposal.content if existing is not None else None))


async def _error(conversation_id: str | None, text: str, *, author: str = "kip") -> None:
    """Post a LOUD, visible failure to the conversation — the judge-error surface. A
    dropped or failed write is never silent (ARCHITECTURE.md). Same shape as a
    confirmation so any interface renders it; distinct only in intent."""
    await _confirm(conversation_id, text, author=author)


async def _confirm(conversation_id: str | None, text: str, *, author: str = "kip") -> None:
    """Post a short outcome renderable, so an interface clears its spinner and shows
    the result. No-op when there is no conversation to answer in."""
    if conversation_id is None:
        return
    await bus.append_renderable(Renderable(
        id=new_id(), conversation_id=conversation_id, kind=RenderableKind.ADVICE,
        author=author, body={"text": text}, at=now_ms()))


async def open_ask(ask_id: str):
    """The pending row for an ask id, or None. Lets the engine tell an anchored
    reply to a live consent card apart from an ordinary message."""
    return await _load_pending(ask_id)


async def reprompt(ask_id: str) -> None:
    """Re-show an open ask's Yes/No card. Used when the user answers in words
    instead of tapping: we keep the ask alive rather than route the reply as a
    normal message, which would silently drop the consent. No-op if it is gone."""
    p = await _load_pending(ask_id)
    if p is None:
        return
    # a conflict card keeps its existing-vs-proposed framing on re-show.
    existing = _snapshot_content(p["existing_snapshot"])            # None when nothing was snapshotted
    meta = json.loads(p["meta"]) if p["meta"] else {}
    context = _batch_note(meta.get("items") or []) if p["ask_type"] == "batch" else meta.get("context")
    if p["ask_type"] != "batch" and _link_note(meta):
        context = (context + "\n" if context else "") + _link_note(meta)
    await bus.append_renderable(_ask_card(
        ask_id, conversation_id=p["conversation_id"], author=p["author"],
        question=p["content"], target=p["kind"], subject=meta.get("to") or None,
        note=(context + "\n" if context else "") + "Please tap Yes or No.",
        existing=existing, proposed=p["content"] if existing is not None else None))


# ── the ask card (one shape, reused by apply and reprompt) ────────────────────

def _ask_card(ask_id: str, *, conversation_id: str, author: str, question: str,
              target: str, note: str | None = None,
              existing: str | None = None, proposed: str | None = None,
              subject: str | None = None) -> Renderable:
    body = {"question": question, "target": target}
    title, derived = _card_label(target, existing)
    subject = subject or derived
    body["title"] = title                                     # the action, e.g. "New task" / "Save person"
    if subject:
        body["subject"] = subject                             # the named entity, e.g. "Omar Haddad"
    if not target.startswith("retire:"):
        body["editable"] = True                               # a value the user can correct in place (not a delete)
    if note:
        body["note"] = note
    # a judge-conflict card carries BOTH values in generic fields; the engine stays
    # agnostic (no domain-specific copy) — the interface renders existing-vs-proposed.
    if existing is not None:
        body["existing"] = existing
        body["proposed"] = proposed if proposed is not None else question
    return Renderable(
        id=ask_id, conversation_id=conversation_id, kind=RenderableKind.ASK,
        author=author, body=body,
        actions=[ActionChoice("Yes", f"approve:{ask_id}"),
                 ActionChoice("No", f"reject:{ask_id}")], at=now_ms())


# ── pending asks (the single applier's memory of an open question) ────────────

def _reason(proposal) -> str:
    """A discard's reason, flattened to one line and clamped ONCE here: it is free text, maybe
    from an agent, and it rides into a card note, the phone digest and the audit trail."""
    return " ".join(str((getattr(proposal, "meta", None) or {}).get("reason") or "").split())[:200]


async def _open_task_card(conversation_id: str, task_id: str):
    """kip's open act card on a task (done? / Discard? / Reschedule?), if any — one question per task."""
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT ask_id, kind, meta FROM pending WHERE conversation_id=? AND author='kip' AND ask_type='act' AND meta LIKE ?",
            (conversation_id, f'%"task_id": "{task_id}"%'))
        return await cur.fetchone()


async def _open_ask_for(conversation_id: str, kind: str):
    # a retire ask (`retire:X`) and an update ask (`X`) target the SAME belief — dedup them
    # together, so a "forget X" and an "update X" card can never both be open (approving both
    # would resurrect a just-forgotten belief). Match on the underlying target either way; the
    # newer supersedes the older in _raise_ask.
    # an identity ask (`identity:<target>:<own>`) is a question about OWN — it and a plain
    # ask for `own` are one open question, so a Yes on each can never save it twice
    if kind.startswith("links:"):                # a links ask matches only itself: the note is untouched
        async with _db.get_db() as c:
            cur = await c.execute("SELECT * FROM pending WHERE conversation_id=? AND kind=? LIMIT 1", (conversation_id, kind))
            return await cur.fetchone()
    if kind.startswith("identity:"):
        target = kind.split(":", 2)[2]
    else:
        target = kind.split(":", 1)[1] if kind.startswith("retire:") else kind
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT * FROM pending WHERE conversation_id=? AND (kind=? OR kind=? OR kind LIKE ?) LIMIT 1",
            (conversation_id, target, f"retire:{target}", f"identity:%:{target}"))
        return await cur.fetchone()


async def _save_pending(ask_id: str, proposal, *, kind: str, conversation_id: str,
                        author: str, origin: str = "consent-stakes",
                        existing_snapshot: str | None = None, ask_type: str = "belief",
                        conn=None) -> None:
    """Record a pending ask. Pass `conn` to run inside an existing transaction — the
    re-raise uses this to delete the stale ask and save the fresh one together, so a
    mid-write crash cannot lose the proposal. When `conn` is given, the caller commits."""
    meta = getattr(proposal, "meta", None) or None
    params = (ask_id, conversation_id, kind, proposal.content,
              store.evidence_dump(list(proposal.evidence)), author,
              origin, existing_snapshot, getattr(proposal, "at", None), ask_type,
              json.dumps(meta) if meta else None, _ACTION.get(), now_ms())
    sql = ("INSERT OR REPLACE INTO pending "
           "(ask_id, conversation_id, kind, content, evidence, author, origin, "
           "existing_snapshot, at, ask_type, meta, action_id, created_at) "
           "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)")
    if conn is not None:
        await conn.execute(sql, params)
        return
    async with _db.get_db() as c:
        await c.execute(sql, params)
        await c.commit()


async def _load_pending(ask_id: str):
    async with _db.get_db() as c:
        cur = await c.execute("SELECT * FROM pending WHERE ask_id=?", (ask_id,))
        return await cur.fetchone()


async def _delete_pending(ask_id: str) -> None:
    async with _db.get_db() as c:
        await c.execute("DELETE FROM pending WHERE ask_id=?", (ask_id,))
        await c.commit()
