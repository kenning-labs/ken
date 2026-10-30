"""Slice-scoped capabilities the engine injects onto a Context. (AGENTS.md)

The core invariant: a first-party agent touches memory ONLY through capabilities the
engine hands it — it imports NEITHER `store` NOR `gate` NOR `db`. The engine (the
composition layer) is the single holder of those; it builds one small set of
capabilities per agent, from that agent's declared Slice + session, and attaches them
to the Context.

Honest framing: this is capability INJECTION plus first-party discipline plus
auditability of a well-behaved plugin. It is NOT a sandbox — Python cannot stop a
malicious in-process plugin (real containment needs process isolation, out of scope).
The value is a clean, copyable reference pattern and a real read/write scope, enforced
for first-party code by the import guard (tests/test_agent_import_guard.py).

Three capabilities:
  - MemoryReader  (ctx.memory)  — READ-ONLY, always LIVE, slice-scoped.
  - Proposer      (ctx.propose) — the WRITE path, through the judge + consent gate.
  - ViewPublisher (ctx.view)    — publish the standing panel (interface output).
"""
from __future__ import annotations

import asyncio

from . import config as _config
from . import db as _db
from .agents.base import Proposal, Slice
from .consent import gate
from .contracts import Provenance
from .memory import store

_bucket_locks: dict[str, asyncio.Lock] = {}   # one lock per bucket — serialize its writers


def _bucket_lock(name: str) -> asyncio.Lock:
    lock = _bucket_locks.get(name)
    if lock is None:
        lock = _bucket_locks[name] = asyncio.Lock()
    return lock


class ScopeError(RuntimeError):
    """A read outside the agent's declared slice. LOUD by doctrine (AGENTS.md):
    an out-of-scope read raises, it never returns a silent empty result that would let
    a mis-scoped agent quietly believe the memory is empty."""


class MemoryReader:
    """A READ-ONLY, always-LIVE, slice-scoped view of memory. Every method hits the
    store NOW (never a stale snapshot), so a task captured this same turn is visible.
    A read outside the declared slice RAISES (ScopeError) rather than lying with [].

    Built by the engine from the agent's Slice: `tasks` (a truthy grant like "open")
    grants the tasks noun — which is what lets `find_tasks` reach COMPLETED tasks even
    though the preload is bounded to the open view; `structured` grants exactly the
    listed namespaces."""

    def __init__(self, slice_: Slice) -> None:
        self._tasks_granted = bool(slice_.tasks)          # the tasks noun (not just the preload)
        self._events_granted = bool(slice_.events)        # the events noun (append-only log)
        self._namespaces = frozenset(slice_.structured)   # granted structured namespaces
        self._panels = frozenset(slice_.panels)           # other agents' panels readable live

    async def open_tasks(self):
        """Every open task, live. Requires the tasks scope."""
        if not self._tasks_granted:
            raise ScopeError("this agent's slice does not grant the tasks scope")
        return await store.open_tasks()

    async def done_since(self, since_ms: int):
        """Tasks completed at or after `since_ms`, newest first, live. Requires the tasks scope."""
        if not self._tasks_granted:
            raise ScopeError("this agent's slice does not grant the tasks scope")
        return await store.done_since(since_ms)

    async def find_tasks(self, query: str, include_done: bool = True):
        """Text-search tasks (reaching completed ones when include_done), live.
        Requires the tasks scope."""
        if not self._tasks_granted:
            raise ScopeError("this agent's slice does not grant the tasks scope")
        return await store.find_tasks(query, include_done=include_done)

    async def recent_events(self, limit: int = 10):
        """The most recent events, newest first, live. Requires the events scope."""
        if not self._events_granted:
            raise ScopeError("this agent's slice does not grant the events scope")
        return await store.recent_events(limit)

    async def find_events(self, query: str):
        """Text-search the append-only event log, live. Requires the events scope."""
        if not self._events_granted:
            raise ScopeError("this agent's slice does not grant the events scope")
        return await store.find_events(query)

    async def panel(self, session: str):
        """Another agent's latest published panel spec (its `views` row), live, or None
        when it has not published yet. An ungranted session raises."""
        if session not in self._panels:
            raise ScopeError(
                f"panel {session!r} is not in this agent's slice "
                f"(granted: {sorted(self._panels) or 'none'})")
        return await store.get_view(session)

    async def structured(self, namespace: str):
        """Every belief in one GRANTED namespace, live. An ungranted namespace raises."""
        if namespace not in self._namespaces:
            raise ScopeError(
                f"namespace {namespace!r} is not in this agent's slice "
                f"(granted: {sorted(self._namespaces) or 'none'})")
        return await store.get_structured(namespace + ".")


class Proposer:
    """The WRITE path. An agent NEVER writes memory or imports the gate — it proposes,
    and the gate is the single applier.

      - capture(...)    the judged capture path (add_task / remember_*): the proposal
                        passes the machine skeptic then routes through consent, exactly
                        as before — just reached through the injected capability.
      - complete(id)    /
      - reschedule(...) user-instructed task edits. These become gate proposals too
                        (new fast-track intents): no judge, no card, but AUDITED/logged
                        like any applied write — closing the seam where they used to
                        bypass the gate with a direct store write. Applied by task id.
    """

    def __init__(self, *, conversation_id: str, author: str, action_id: str | None = None) -> None:
        self._cid = conversation_id
        self._author = author
        self.action_id = action_id      # the inbound action this turn answers; cards carry it

    async def capture(self, proposal: Proposal, *, judge=None, message: str = "",
                      details: dict | None = None) -> str:
        """Route a capture proposal through the judge + consent gate. Returns the gate's
        outcome token (the caller relays the TRUE outcome — never a claimed save)."""
        return await gate.apply(proposal, conversation_id=self._cid, author=self._author,
                                judge=judge, message=message, details=details,
                                action_id=self.action_id)

    async def complete(self, task_id: str, *, provenance=None, evidence=None) -> str:
        """Mark a task done, by id — a fast-track, audited edit for the user's own agents;
        WORLD content (an agent token) asks with a "done?" card instead. Returns
        'completed', 'pending' or 'task_absent'."""
        return await gate.apply(
            Proposal(intent="complete", content=task_id, evidence=list(evidence or []),
                     provenance=provenance or Provenance.AGENT_INFERRED),
            conversation_id=self._cid, author=self._author, action_id=self.action_id)

    async def discard(self, task_id: str, reason: str = "", *, provenance=None, evidence=None) -> str:
        """Close a task as dropped (not done), by id, with the reason — a fast-track, audited edit
        for the user's own agents; WORLD content asks with a "Discard?" card. Returns
        'discarded', 'pending' or 'task_absent'."""
        return await gate.apply(
            Proposal(intent="discard", content=task_id, evidence=list(evidence or []),
                     provenance=provenance or Provenance.AGENT_INFERRED, meta={"reason": reason}),
            conversation_id=self._cid, author=self._author, action_id=self.action_id)

    async def reschedule(self, task_id: str, when: int, *, provenance=None, evidence=None) -> str:
        """Re-time a task, by id — a fast-track, audited edit for the user's own agents;
        WORLD content asks with a "Reschedule?" card. Returns 'rescheduled', 'pending'
        or 'task_absent'."""
        return await gate.apply(
            Proposal(intent="reschedule", content=task_id, at=when, evidence=list(evidence or []),
                     provenance=provenance or Provenance.AGENT_INFERRED),
            conversation_id=self._cid, author=self._author, action_id=self.action_id)


    async def assign(self, task_id: str, owner: str | None, *, provenance=None, evidence=None) -> str:
        """Hand a task to `owner` (people.<slug>, a name, or None = the user), by id — fast-track
        and audited for the user's own agents; WORLD content asks with an "Assign?" card. Returns
        'assigned', 'pending' or 'task_absent'."""
        return await gate.apply(
            Proposal(intent="assign", content=task_id, evidence=list(evidence or []),
                     provenance=provenance or Provenance.AGENT_INFERRED, meta={"owner": owner}),
            conversation_id=self._cid, author=self._author, action_id=self.action_id)

    async def link(self, task_id: str, depends_on: str, *, remove: bool = False, provenance=None,
                   evidence=None) -> str:
        """Task `task_id` waits on `depends_on` (or, with remove, no longer does), by id — fast-track
        and audited; WORLD content asks with a "Link tasks?" card. Returns 'linked', 'unlinked',
        'cycle', 'link_absent', 'pending' or 'task_absent'."""
        return await gate.apply(
            Proposal(intent="link_tasks", content=task_id, evidence=list(evidence or []),
                     provenance=provenance or Provenance.AGENT_INFERRED,
                     meta={"depends_on": depends_on, "remove": bool(remove)}),
            conversation_id=self._cid, author=self._author, action_id=self.action_id)


class ViewPublisher:
    """Publish the agent's standing panel to its OWN session. This is interface output,
    not memory — so it lives here, off ctx.memory. Best-effort: a write failure is
    swallowed so a panel glitch never breaks the agent's reply."""

    def __init__(self, session: str) -> None:
        self._session = session

    async def publish(self, view: dict) -> None:
        try:
            await store.set_view(self._session, view)
        except Exception:  # noqa: BLE001, S110 — a panel is best-effort, never the reply
            pass


def _as_list(d: dict, key: str) -> list:
    """The current value under `key` as a list. Absent OR a hand-edited yaml null (`watch:`)
    is an empty list; a real scalar (set via .set) raises rather than being iterated
    char-by-char — mixing set() and add()/remove() on one key is a bug."""
    cur = d.get(key)
    if cur is None:
        return []
    if not isinstance(cur, list):
        raise ValueError(f"config key {key!r} holds a {type(cur).__name__}, not a list — "  # noqa: TRY004
                         f"use set() for scalars, add()/remove() for lists")
    return list(cur)


class Bucket:
    """One config bucket (~/.ken/state/<name>.yaml). Reads are sync + live (a plain file
    read each call); writes are atomic and AUDITED. Write on a read-only bucket raises."""

    def __init__(self, name: str, *, writable: bool, author: str) -> None:
        self._name = name
        self._writable = writable
        self._author = author

    def get(self, key: str, default=None):
        return _config.read_bucket(self._name).get(key, default)

    async def set(self, key: str, value) -> None:
        await self._mutate(lambda d: d.__setitem__(key, value), f"set {key}={value!r}")

    async def add(self, key: str, item) -> None:
        def op(d):
            lst = [x for x in _as_list(d, key) if x != item]
            lst.append(item)                       # idempotent append (no dup)
            d[key] = lst
        await self._mutate(op, f"add {item!r} to {key}")

    async def remove(self, key: str, item) -> None:
        await self._mutate(lambda d: d.__setitem__(key, [x for x in _as_list(d, key) if x != item]),
                           f"remove {item!r} from {key}")

    async def _mutate(self, op, reason: str) -> None:
        if not self._writable:
            raise ScopeError(f"config bucket {self._name!r} is read-only for this agent")
        async with _bucket_lock(self._name):       # serialize this bucket's writers — no lost update
            data = _config.read_bucket(self._name)
            op(data)                               # validate + apply in-memory (may raise → nothing persisted)
            _config.write_bucket(self._name, data)  # the real effect — the FILE is the source of truth
        try:                                       # audit is best-effort history AFTER a real write:
            await _db.audit("config", f"{self._name}:{self._author}", reason)   # never a phantom row for a write that didn't land
        except Exception as e:  # noqa: BLE001 — a logging miss must not fail an already-applied change
            print(f"ken: config change applied but audit failed: {e!r}", flush=True)


class ConfigStore:
    """Scoped access to named config buckets, built from an agent's declared Config grant.
    `ctx.config("slack")` returns a Bucket handle; a bucket the agent did not declare (in
    read OR write) RAISES — loud, by the same doctrine as an out-of-slice memory read. A
    WRITE grant implies read (you can see what you may change); a read-only grant (in read,
    not write) reads live but refuses every mutation."""

    def __init__(self, *, read: list[str], write: list[str], author: str) -> None:
        self._read = frozenset(read)
        self._write = frozenset(write)
        self._author = author

    def __call__(self, bucket: str) -> Bucket:
        if bucket not in self._read and bucket not in self._write:
            raise ScopeError(
                f"config bucket {bucket!r} is not in this agent's grant "
                f"(read: {sorted(self._read) or 'none'}; write: {sorted(self._write) or 'none'})")
        return Bucket(bucket, writable=bucket in self._write, author=self._author)
