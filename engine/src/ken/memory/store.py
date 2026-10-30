"""Memory — DAOs for the nouns (tasks, events) and the structured layer. The
consent gate is the only path that changes a *belief* (its content). Confidence is
a derived freshness signal, not a gated belief edit. (AGENTS.md)
"""
from __future__ import annotations

import json
from datetime import datetime

from .. import db as _db
from ..contracts import (
    Change,
    Event,
    Evidence,
    Provenance,
    StructuredEntry,
    Task,
    orient,
)
from ..util import new_id, norm, now_ms, split_deadline

# ── evidence serde (one shape, shared by the store and the consent gate) ────

def evidence_dicts(items, n: int | None = None) -> list[dict]:
    """`[{ref, source}]` — the one wire shape for evidence. With `n`, the origin (first) plus
    the newest ones, since writes append and the newest ref is the one a card rests on."""
    rows = [{"ref": e.ref, "source": e.source} for e in items]
    if n is not None and len(rows) > n:
        rows = rows[:1] + rows[-(n - 1):] if n > 1 else rows[-1:]
    return rows


def evidence_dump(evidence: list[Evidence]) -> str:
    return json.dumps(evidence_dicts(evidence))


def evidence_load(raw: str | None) -> list[Evidence]:
    # tolerant: an extra key in a stored row must not take every read surface down
    return [Evidence(ref=str(e.get("ref", "")), source=str(e.get("source", "")))
            for e in json.loads(raw or "[]") if isinstance(e, dict)]


async def bump(name: str, at: int, by: int = 1) -> None:
    """Increment an observability counter and stamp its last time. Cheap enough to
    call on every tick; the CLI reads these for the System dashboard."""
    async with _db.get_db() as c:
        await c.execute(
            "INSERT INTO stats (name, count, last_at) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET count=count+excluded.count, last_at=excluded.last_at",
            (name, by, at))
        await c.commit()


_AGENT_LOG_KEEP = 1000       # cap the activity trail; the dashboard reads a handful


async def log_action(agent: str, action: str, target: str | None = None,
                     detail: str | None = None) -> None:
    """Record what an agent DID — a scannable trail for background runs. Distinct
    from db.audit, which records what the engine dropped or failed to do. Pruned to
    the most recent _AGENT_LOG_KEEP rows so it can't grow without bound."""
    async with _db.get_db() as c:
        await c.execute(
            "INSERT INTO agent_log (id, at, agent, action, target, detail) VALUES (?,?,?,?,?,?)",
            (new_id(), now_ms(), agent, action, target, detail))
        await c.execute(
            "DELETE FROM agent_log WHERE at < "
            "(SELECT at FROM agent_log ORDER BY at DESC LIMIT 1 OFFSET ?)",
            (_AGENT_LOG_KEEP,))
        await c.commit()


async def refresh_view(conversation_id: str, build) -> None:
    """Rebuild and publish a conversation's standing panel from `build` (an async
    thunk returning the block spec). Any failure — building or writing — is swallowed,
    so a panel glitch never breaks the agent's reply. The one safe-publish path both
    agents use."""
    try:
        await set_view(conversation_id, await build())
    except Exception:  # noqa: BLE001, S110 — a panel is best-effort, never the reply
        pass


async def get_view(conversation_id: str) -> dict | None:
    """An agent's latest published panel spec, or None. Opaque to the store."""
    async with _db.get_db() as c:
        cur = await c.execute("SELECT body FROM views WHERE conversation_id=?", (conversation_id,))
        row = await cur.fetchone()
    return json.loads(row["body"]) if row else None


async def set_view(conversation_id: str, body: dict) -> None:
    """Store an agent's current rendered panel for a conversation (the latest, not
    a log). The engine stores it opaquely; an interface parses and draws it."""
    async with _db.get_db() as c:
        await c.execute(
            "INSERT INTO views (conversation_id, body, at) VALUES (?,?,?) "
            "ON CONFLICT(conversation_id) DO UPDATE SET body=excluded.body, at=excluded.at",
            (conversation_id, json.dumps(body), now_ms()))
        await c.commit()


async def get_meta(key: str) -> str | None:
    """Small durable agent state (kv), e.g. an agent's last check-in day."""
    async with _db.get_db() as c:
        cur = await c.execute("SELECT value FROM kv WHERE key=?", (key,))
        row = await cur.fetchone()
    return row["value"] if row else None


async def set_meta(key: str, value: str) -> None:
    async with _db.get_db() as c:
        await c.execute(
            "INSERT INTO kv (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, now_ms()))
        await c.commit()


# ── tasks ──────────────────────────────────────────────────────────────────

async def add_task(text: str, provenance: Provenance,
                   deadline: int | None = None,
                   evidence: list[Evidence] | None = None, *, project: str | None = None,
                   owner: str | None = None, conn=None) -> tuple[str, bool]:
    """Create an open task, or no-op if an identical open task already exists (matched
    on the normalized text) — an agent re-proposing a task it already has must not
    duplicate it. Returns (id, created): created is False when it already existed.

    Pass `conn` to run inside an existing transaction — the consent gate uses this to add
    the task and clear its pending ask together, so a mid-write failure cannot leave a task
    the user then rejects. When `conn` is given, the caller commits.

    `evidence` is persisted as-is (Step 1 plumbing: no requirement or validation yet).
    An existing task is never re-evidenced — only a newly created row stores it.
    `project` (a projects.<slug> kind) follows the schedule's rule: an existing task adopts
    one only when it has none; it is never silently moved to a different project."""
    key = norm(text)

    async def _do(c) -> tuple[str, bool]:
        cur = await c.execute("SELECT id, text, deadline, project, owner FROM tasks WHERE status='open'")
        for row in await cur.fetchall():
            if norm(row["text"]) != key:
                continue
            # adopt a schedule for a previously-unscheduled task; never silently
            # RE-schedule one that already has a time.
            if deadline is not None and row["deadline"] is None:
                await c.execute("UPDATE tasks SET deadline=? WHERE id=?",
                                (deadline, row["id"]))
            if project and row["project"] is None:
                await c.execute("UPDATE tasks SET project=? WHERE id=?", (project, row["id"]))
            if owner and row["owner"] is None:            # an owner, like a project: adopted, never moved
                await c.execute("UPDATE tasks SET owner=?, changed_at=? WHERE id=?", (owner, now_ms(), row["id"]))
            if conn is None:
                await c.commit()
            return row["id"], False
        tid = new_id()
        await c.execute(
            # changed_at stays NULL: a creation competes with no later edit. A dated add stamps it after.
            "INSERT INTO tasks (id, text, status, deadline, provenance, evidence, created_at, project, owner) "
            "VALUES (?,?,'open',?,?,?,?,?,?)",
            (tid, text, deadline, provenance.value,
             evidence_dump(evidence or []), now_ms(), project or None, owner or None),
        )
        if conn is None:
            await c.commit()
        return tid, True

    if conn is not None:
        return await _do(conn)
    async with _db.get_db() as c:
        return await _do(c)


async def adopt_onto_task(task_id: str, *, deadline: int | None = None,
                          evidence: list[Evidence] | None = None, project: str | None = None) -> bool:
    """Reconcile a fuzzy-duplicate proposal onto an existing open task — the collapse
    primitive. Fuzzy dedup cannot fall through `add_task`'s exact-norm path, so the
    gate names the target task by id and adopts a new time and/or evidence onto it.

    A schedule is adopted ONLY when the task has none yet — the same guard `add_task`
    keeps: an already-scheduled task is never silently re-timed. Evidence is adopted
    only when the task carries none yet, so the first 'why' behind it survives. Returns
    True when a row actually changed."""
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT deadline, evidence, project FROM tasks WHERE id=? AND status='open'", (task_id,))
        row = await cur.fetchone()
        if row is None:
            return False
        sets, args = [], []
        if deadline is not None and row["deadline"] is None:
            sets.append("deadline=?")
            args.append(deadline)
        if evidence and not evidence_load(row["evidence"]):
            sets.append("evidence=?")
            args.append(evidence_dump(evidence))
        if project and row["project"] is None:        # a project, like a time: adopted, never moved
            sets.append("project=?")
            args.append(project)
        if not sets:
            return False
        args.append(task_id)
        await c.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", args)
        await c.commit()
    return True


async def _close_task(task_id: str, status: str, reason: str | None = None) -> bool:
    """Close an open task as `done` or `dropped`. Validated: it only ever flips an existing open
    task, so a stale or forged token cannot invent or reopen one. Returns True if a row changed."""
    async with _db.get_db() as c:
        cur = await c.execute(
            "UPDATE tasks SET status=?, completed_at=?, drop_reason=?, changed_at=? WHERE id=? AND status='open'",
            (status, now_ms(), reason, now_ms(), task_id))
        await c.commit()
        return cur.rowcount > 0


async def complete_task(task_id: str) -> bool:
    """Mark an open task done."""
    return await _close_task(task_id, "done")


async def discard_task(task_id: str, reason: str = "") -> bool:
    """Close an open task as DROPPED, not done: it was not finished, it stopped mattering. Kept
    apart from complete_task so history and the done counts stay true."""
    return await _close_task(task_id, "dropped", " ".join(str(reason or "").split())[:500] or None)


async def reschedule(task_id: str, when_ms: int) -> bool:
    """Re-time an existing open task — the EXPLICIT reschedule primitive. Distinct
    from add_task/adopt, which never silently re-time a task that already has one; a
    reschedule is a deliberate user instruction, so it overwrites the time. Clearing
    reminded_at lets the new time fire a fresh nudge. A date phrase left in the text ("by
    March 20") is stripped: the field is the one date, so the text can never contradict it.
    Returns True if a row changed."""
    async with _db.get_db() as c:
        row = await (await c.execute("SELECT text FROM tasks WHERE id=?", (task_id,))).fetchone()
        text = split_deadline(row["text"], datetime.now())[0] if row else ""   # noqa: DTZ005 — only the text is used
        cur = await c.execute(
            "UPDATE tasks SET deadline=?, reminded_at=NULL, text=?, changed_at=? WHERE id=? AND status='open'",
            (when_ms, text, now_ms(), task_id))
        await c.commit()
        return cur.rowcount > 0


async def set_slot(task_id: str, start_ms: int | None, minutes: int | None, by: str | None = None) -> bool:
    """Set (or, with None, clear) when the user works on an open task, and who set it ('user' | 'agent').
    A planning act, not world knowledge: written at once. Overlaps are allowed. Returns True if a row changed."""
    async with _db.get_db() as c:
        cur = await c.execute("UPDATE tasks SET slot_start=?, slot_minutes=?, slot_by=? WHERE id=? AND status='open'",
                              (start_ms, minutes, by if start_ms else None, task_id))
        await c.commit()
        return cur.rowcount > 0


def _task(r) -> Task:
    """One tasks row -> Task. The single loader, so a new column lands everywhere at once."""
    return Task(id=r["id"], text=r["text"], status=r["status"],
                deadline=r["deadline"], provenance=Provenance(r["provenance"]),
                evidence=evidence_load(r["evidence"]), created_at=r["created_at"],
                completed_at=r["completed_at"], project=r["project"], drop_reason=r["drop_reason"],
                slot_start=r["slot_start"], slot_minutes=r["slot_minutes"], slot_by=r["slot_by"],
                owner=r["owner"], changed_at=r["changed_at"])


async def set_task_project(task_id: str, project: str | None) -> bool:
    """Put an open task on a project (a projects.<slug> kind), or on none. The one path
    that MOVES a task between projects: a deliberate edit, never a side effect of a capture.
    A project Ken does not hold is refused: a task never points at a record that is not there."""
    async with _db.get_db() as c:
        if project and not await (await c.execute("SELECT 1 FROM structured WHERE id=?", (project,))).fetchone():
            return False
        cur = await c.execute("UPDATE tasks SET project=? WHERE id=? AND status='open'",
                              (project or None, task_id))
        await c.commit()
        return cur.rowcount > 0


async def set_task_owner(task_id: str, owner: str | None) -> bool:
    """Hand an open task to someone (people.<slug> or a name as given), or back to the user
    (None). The one path that MOVES a task between owners. Returns True if a row changed."""
    async with _db.get_db() as c:
        cur = await c.execute("UPDATE tasks SET owner=?, changed_at=? WHERE id=? AND status='open'",
                              (owner or None, now_ms(), task_id))
        await c.commit()
        return cur.rowcount > 0


async def stamp_task(task_id: str, at: int, *, closed: bool = False) -> None:
    """Set when a task's state last changed to the time the change HAPPENED (a write's `at`),
    so a later write that happened earlier reads as stale. `closed`: completed_at too, so
    history shows when the work was done, not when it was reported."""
    async with _db.get_db() as c:
        await c.execute(f"UPDATE tasks SET changed_at=?{', completed_at=?' if closed else ''} WHERE id=?",
                        (at, at, task_id) if closed else (at, task_id))
        await c.commit()


async def stamp_structured(kind: str, at: int) -> None:
    """The same for a structured record: its updated_at becomes when the note was stated."""
    async with _db.get_db() as c:
        await c.execute("UPDATE structured SET updated_at=? WHERE id=?", (at, kind))
        await c.commit()


async def depends_on_path(start: str, target: str) -> bool:
    """True when `target` is reachable from `start` along dependencies (start waits on … on
    target). The cycle check: linking A on B is refused when A is reachable from B."""
    async with _db.get_db() as c:
        edges: dict[str, list[str]] = {}
        for r in await (await c.execute("SELECT task, depends_on FROM task_deps")).fetchall():
            edges.setdefault(r["task"], []).append(r["depends_on"])
    seen, todo = set(), [start]
    while todo:
        n = todo.pop()
        if n == target:
            return True
        if n not in seen:
            seen.add(n)
            todo += edges.get(n, [])
    return False


async def link_task(task_id: str, depends_on: str) -> str:
    """Task `task_id` waits on `depends_on`. Returns 'linked', 'cycle' (itself, or the
    dependency already waits on it, directly or not) or 'task_absent' (either task missing or
    the waiting one closed)."""
    if task_id == depends_on or await depends_on_path(depends_on, task_id):
        return "cycle"
    async with _db.get_db() as c:
        ok = await (await c.execute("SELECT 1 FROM tasks WHERE id=? AND status='open'", (task_id,))).fetchone()
        dep = await (await c.execute("SELECT 1 FROM tasks WHERE id=?", (depends_on,))).fetchone()
        if not (ok and dep):
            return "task_absent"
        await c.execute("INSERT OR IGNORE INTO task_deps (task, depends_on, at) VALUES (?,?,?)",
                        (task_id, depends_on, now_ms()))
        await c.commit()
    return "linked"


async def unlink_task(task_id: str, depends_on: str) -> bool:
    async with _db.get_db() as c:
        cur = await c.execute("DELETE FROM task_deps WHERE task=? AND depends_on=?", (task_id, depends_on))
        await c.commit()
        return cur.rowcount > 0


async def deps_map() -> dict[str, list[dict]]:
    """Every dependency, one read: {task id: [{id, text, status}]} of the tasks it waits on."""
    async with _db.get_db() as c:
        cur = await c.execute("SELECT d.task, t.id, t.text, t.status FROM task_deps d "
                              "JOIN tasks t ON t.id = d.depends_on ORDER BY d.at")
        out: dict[str, list[dict]] = {}
        for r in await cur.fetchall():
            out.setdefault(r["task"], []).append({"id": r["id"], "text": r["text"], "status": r["status"]})
    return out


async def tasks_by_status(statuses: tuple[str, ...], limit: int = 200) -> list[Task]:
    """Tasks in any of `statuses`, most recently closed (else created) first."""
    marks = ",".join("?" * len(statuses))
    async with _db.get_db() as c:
        cur = await c.execute(f"SELECT * FROM tasks WHERE status IN ({marks}) "
                              "ORDER BY COALESCE(completed_at, created_at) DESC LIMIT ?", (*statuses, limit))
        return [_task(r) for r in await cur.fetchall()]


async def done_since(since_ms: int) -> list[Task]:
    """Tasks marked done at or after `since_ms`, newest first — the week's completions.
    A task done before completion times existed has completed_at NULL and is not counted."""
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT * FROM tasks WHERE status='done' AND completed_at>=? "
            "ORDER BY completed_at DESC LIMIT 50", (since_ms,))
        rows = await cur.fetchall()
    return [_task(r) for r in rows]


async def task_by_id(task_id: str) -> Task | None:
    """One task by id, any status, or None."""
    async with _db.get_db() as c:
        cur = await c.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
        r = await cur.fetchone()
    return None if r is None else _task(r)


async def open_tasks() -> list[Task]:
    async with _db.get_db() as c:
        cur = await c.execute("SELECT * FROM tasks WHERE status='open' ORDER BY created_at")
        rows = await cur.fetchall()
    return [_task(r)
            for r in rows]


async def find_tasks(query: str, *, include_done: bool = True, limit: int = 20) -> list[Task]:
    """Text-search tasks by a normalized substring of `query`, newest first. Unlike
    open_tasks, this reaches COMPLETED (and dropped) tasks when `include_done` — the
    on-demand lookup for "is that done?" that the bounded open-view preload cannot
    answer. Generic substring match on norm(text); an empty query matches nothing."""
    q = norm(str(query))
    if not q:
        return []
    where = "" if include_done else " WHERE status='open'"
    async with _db.get_db() as c:
        cur = await c.execute(f"SELECT * FROM tasks{where} ORDER BY created_at DESC")
        rows = await cur.fetchall()
    hits = [_task(r)
            for r in rows if q in norm(r["text"])]
    return hits[:limit]


def _event(r) -> Event:
    return Event(id=r["id"], summary=r["summary"], at=r["at"], provenance=Provenance(r["provenance"]),
                 evidence=evidence_load(r["evidence"]), until=r["until"])


async def events_citing(ref: str, *, limit: int = 50) -> list[Event]:
    """Events whose evidence cites `ref` (a task id: its reports), newest first, expired or
    not. Matches the exact dump shape `evidence_dump` writes, so a ref never matches a
    substring of another id by accident."""
    needle = f'%"ref": "{ref}"%'
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT * FROM events WHERE evidence LIKE ? ORDER BY at DESC, rowid DESC LIMIT ?", (needle, limit))
        rows = await cur.fetchall()
    return [_event(r) for r in rows if _cites(r["evidence"], ref)]


def _cites(raw: str | None, ref: str) -> bool:
    return any(e.ref == ref for e in evidence_load(raw))


_LIVE = "(until IS NULL OR until >= ?)"      # the ONE boundary every read and the gauge share


async def recent_events(limit: int = 10, *, now: int | None = None) -> list[Event]:
    """The most recent LIVE events, newest first — the bounded slice the judge dedups a
    proposed event against. Generic: it knows nothing about what any event means."""
    now = now or now_ms()
    async with _db.get_db() as c:
        cur = await c.execute(f"SELECT * FROM events WHERE {_LIVE} ORDER BY at DESC LIMIT ?", (now, limit))
        rows = await cur.fetchall()
    return [_event(r) for r in rows]


async def find_events(query: str, *, limit: int = 20, now: int | None = None) -> list[Event]:
    """Text-search LIVE events by a normalized substring of `query`, newest first — the
    on-demand "when did that happen?" lookup the bounded recent-events preload cannot
    answer. Generic substring match on norm(summary); an empty query matches nothing."""
    q = norm(str(query))
    if not q:
        return []
    now = now or now_ms()
    async with _db.get_db() as c:
        cur = await c.execute(f"SELECT * FROM events WHERE {_LIVE} ORDER BY at DESC", (now,))
        rows = await cur.fetchall()
    return [_event(r) for r in rows if q in norm(r["summary"])][:limit]


async def events_between(lo: int, hi: int, *, now: int | None = None, limit: int = 200) -> list[Event]:
    """LIVE events whose time falls in [lo, hi], newest first."""
    now = now or now_ms()
    async with _db.get_db() as c:
        cur = await c.execute(f"SELECT * FROM events WHERE {_LIVE} AND at BETWEEN ? AND ? ORDER BY at DESC LIMIT ?",
                              (now, lo, hi, limit))
        return [_event(r) for r in await cur.fetchall()]


async def expired_events(now: int | None = None) -> int:
    """How many events have stopped mattering (a gauge for the maintainer's audit row)."""
    now = now or now_ms()
    async with _db.get_db() as c:
        cur = await c.execute(f"SELECT count(*) FROM events WHERE NOT {_LIVE}", (now,))
        return (await cur.fetchone())[0]


async def latest_report(task_id: str) -> Event | None:
    """The report on a task that was LOGGED last (insert order, not `at`: a report told
    today about last week is still the newest word on the task). Expired or not."""
    needle = f'%"ref": "{task_id}"%'
    async with _db.get_db() as c:
        cur = await c.execute("SELECT * FROM events WHERE evidence LIKE ? ORDER BY rowid DESC LIMIT 5", (needle,))
        rows = await cur.fetchall()
    return next((_event(r) for r in rows if _cites(r["evidence"], task_id)), None)


async def due_reminders(now: int, limit: int = 20) -> list[Task]:
    """Open tasks whose scheduled time has arrived and that have not been reminded yet
    — the engine nudges each once, oldest first. Capped so one tick can't flood the
    tab; any overflow fires on the next tick."""
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT * FROM tasks WHERE status='open' AND deadline IS NOT NULL "
            "AND deadline <= ? AND reminded_at IS NULL ORDER BY deadline LIMIT ?",
            (now, limit))
        rows = await cur.fetchall()
    return [_task(r)
            for r in rows]


async def mark_reminded(task_id: str, at: int) -> None:
    """Stamp a task's reminder as fired — the fire-once guard, restart-safe."""
    async with _db.get_db() as c:
        await c.execute("UPDATE tasks SET reminded_at=? WHERE id=?", (at, task_id))
        await c.commit()


# ── the maintained structured layer (evidence-backed, versioned) ───────────

_UPSERT_STRUCTURED_SQL = (
    "INSERT INTO structured "
    "(id, kind, subject, content, evidence, confidence, provenance, changelog, updated_at, url) "
    "VALUES (?,?,?,?,?,?,?, json_array(json(?)), ?, ?) "
    "ON CONFLICT(id) DO UPDATE SET content=excluded.content, subject=excluded.subject, "
    "evidence=excluded.evidence, confidence=excluded.confidence, "
    "provenance=excluded.provenance, updated_at=excluded.updated_at, "
    "url=COALESCE(excluded.url, structured.url), "
    "changelog=json_insert(COALESCE(structured.changelog,'[]'), '$[#]', "
    "json_set(json(?), '$.prev', structured.content))"
)


async def upsert_structured(entry: StructuredEntry, *, change_note: str, conn=None) -> None:
    """Insert a new belief, or revise one and append to its changelog. Only the
    consent gate should call this. The changelog is appended atomically in SQL
    (no read-modify-write), so concurrent writers cannot lose an entry; a revision's
    entry carries `prev`, the content it replaced, so no version is ever lost.

    Pass `conn` to run inside an existing transaction — the consent gate uses this
    to apply the write and clear the pending ask together, so a mid-write failure
    cannot lose an approval. When `conn` is given, the caller commits."""
    change = Change(at=now_ms(), by=entry.provenance, note=change_note)
    change_json = json.dumps({"at": change.at, "by": change.by.value, "note": change.note})
    params = (entry.id, entry.kind, entry.subject, entry.content,
              evidence_dump(entry.evidence), entry.confidence, entry.provenance.value,
              change_json, now_ms(), entry.url, change_json)
    if conn is not None:
        await conn.execute(_UPSERT_STRUCTURED_SQL, params)
        return
    async with _db.get_db() as c:
        await c.execute(_UPSERT_STRUCTURED_SQL, params)
        await c.commit()


async def _unown(c, kind: str) -> None:
    """Tasks owned by a record being removed keep its name as given: still that person's,
    never silently the user's, and never taken over by a later record with the same slug."""
    row = await (await c.execute("SELECT subject FROM structured WHERE id=?", (kind,))).fetchone()
    if row is not None:
        await c.execute("UPDATE tasks SET owner=? WHERE owner=?", (row["subject"] or kind.split(".", 1)[-1], kind))


async def delete_structured(kind: str, *, conn=None) -> bool:
    """Retire a maintained belief. Returns whether a row was actually removed
    (the id is the kind — one belief per kind). Pass `conn` to run inside an existing
    transaction (the consent-approve-forget path deletes belief + pending together);
    when `conn` is given the caller commits."""
    if conn is not None:
        await conn.execute("DELETE FROM links WHERE src=? OR dst=?", (kind, kind))   # links die with the record
        await conn.execute("UPDATE tasks SET project=NULL WHERE project=?", (kind,))  # its tasks stay, on no project
        await _unown(conn, kind)
        cur = await conn.execute("DELETE FROM structured WHERE id=?", (kind,))
        return cur.rowcount > 0
    async with _db.get_db() as c:
        await c.execute("DELETE FROM links WHERE src=? OR dst=?", (kind, kind))
        await c.execute("UPDATE tasks SET project=NULL WHERE project=?", (kind,))
        await _unown(c, kind)
        cur = await c.execute("DELETE FROM structured WHERE id=?", (kind,))
        await c.commit()
        return cur.rowcount > 0


async def get_structured(prefix: str | None = None) -> list[StructuredEntry]:
    q, args = "SELECT * FROM structured", []
    if prefix:
        q += " WHERE kind LIKE ?"
        args.append(prefix + "%")
    async with _db.get_db() as c:
        cur = await c.execute(q, args)
        rows = await cur.fetchall()
    out = []
    for r in rows:
        ev = evidence_load(r["evidence"])
        out.append(StructuredEntry(
            id=r["id"], kind=r["kind"], subject=r["subject"], content=r["content"],
            evidence=ev, confidence=r["confidence"],
            provenance=Provenance(r["provenance"]), updated_at=r["updated_at"], url=r["url"]))
    return out


# ── links: one hop between records, never a graph walk ──────────────────────

async def set_links(src: str, links: list[dict], *, evidence: list[Evidence] | None = None, conn=None) -> int:
    """Add `links` ({dst, rel}) between `src` and each dst, stored in the rel's fixed direction
    (contracts.orient). Adds, never replaces: a link is a fact that stands until a record is
    retired. A link to a record that does not exist (a dropped batch item) is skipped, so no
    link ever dangles. Returns how many were written."""
    rows = []
    for lk in links or []:
        o = orient(src, str(lk.get("dst") or ""), str(lk.get("rel") or "")) if lk.get("dst") and lk["dst"] != src else None
        if o is not None:
            rows.append((o[0], o[1], o[2], evidence_dump(evidence or []), now_ms()))
    if not rows:
        return 0
    sql = ("INSERT OR IGNORE INTO links (src, dst, rel, evidence, at) "
           "SELECT ?, ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM structured WHERE id=?) "
           "AND EXISTS (SELECT 1 FROM structured WHERE id=?)")
    n = 0
    if conn is not None:
        for r in rows:
            n += (await conn.execute(sql, (*r, r[0], r[1]))).rowcount
        return n
    async with _db.get_db() as c:
        for r in rows:
            n += (await c.execute(sql, (*r, r[0], r[1]))).rowcount
        await c.commit()
    return n


async def all_links() -> list[dict]:
    """Every link: [{src, dst, rel}]. One read for a caller that lists the whole world."""
    async with _db.get_db() as c:
        return [dict(r) for r in await (await c.execute("SELECT src, dst, rel FROM links")).fetchall()]


async def links_of(kinds: list[str]) -> list[dict]:
    """Every link that touches any of `kinds`, either end: [{src, dst, rel}]."""
    if not kinds:
        return []
    marks = ",".join("?" * len(kinds))
    async with _db.get_db() as c:
        cur = await c.execute(f"SELECT src, dst, rel FROM links WHERE src IN ({marks}) OR dst IN ({marks})",
                              (*kinds, *kinds))
        return [dict(r) for r in await cur.fetchall()]
