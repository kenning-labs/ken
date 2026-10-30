"""SQLite — the store and the interface bus.

One file, WAL mode, in ~/.ken. Async via aiosqlite. This module owns the schema
and the connection; every other module reads and writes through it. The same
file carries the domain stores *and* the outbox/inbox — so an interface only has
to share this file, never call the engine. (ARCHITECTURE.md)
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path

try:
    import fcntl  # POSIX only; the engine lock degrades to a no-op without it
except ImportError:                       # pragma: no cover — non-POSIX (e.g. Windows)
    fcntl = None                          # type: ignore[assignment]

import aiosqlite

from .util import new_id, now_ms

DEFAULT_HOME = "~/.ken"
_DB_PATH: Path | None = None

SCHEMA = """
PRAGMA journal_mode = WAL;

-- the nouns ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tasks (
  id            TEXT PRIMARY KEY,
  text          TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'open',   -- open | done | dropped
  deadline      INTEGER,                        -- by when (ms epoch); a date alone is stored as 09:00
  slot_start    INTEGER,                        -- when the user works on it (ms epoch), with slot_minutes
  slot_minutes  INTEGER,                        -- how long that slot lasts; NULL with slot_start = no slot
  slot_by       TEXT,                           -- who set the slot: user (by hand, held from agents) | agent
  reminded_at   INTEGER,                        -- when its reminder fired (fire-once guard)
  completed_at  INTEGER,                        -- when it was closed: done or dropped (NULL while open)
  drop_reason   TEXT,                           -- why it was discarded (status 'dropped' only)
  project       TEXT,                           -- the structured kind of its project (NULL = none)
  owner         TEXT,                           -- who does it: NULL = the user, else people.<slug> or a name as given
  changed_at    INTEGER,                        -- when its state last changed (a write's `at`, else write time)
  provenance    TEXT NOT NULL,
  evidence      TEXT,                            -- json: [{ref, source}] behind the task
  created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  id          TEXT PRIMARY KEY,
  summary     TEXT NOT NULL,
  at          INTEGER NOT NULL,
  provenance  TEXT NOT NULL,
  evidence    TEXT                              -- json: [{ref, source}] behind the event
);

-- the maintained structured layer: evidence-backed, versioned
CREATE TABLE IF NOT EXISTS structured (
  id          TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,                  -- namespaced, e.g. patterns.recovery
  subject     TEXT NOT NULL,
  content     TEXT NOT NULL,
  evidence    TEXT,                            -- json: [{ref, source}]
  confidence  REAL NOT NULL DEFAULT 0.5,
  provenance  TEXT NOT NULL,
  changelog   TEXT,                            -- json: [{at, by, note}]
  updated_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_structured_kind ON structured(kind);

-- explicit links between records: one hop, both directions, a closed `rel` list. Not a graph:
-- get_context follows exactly one hop from what it already found. Written only by the gate.
CREATE TABLE IF NOT EXISTS links (
  src       TEXT NOT NULL,                    -- the record that holds the link (a structured id)
  dst       TEXT NOT NULL,                    -- the record it points to
  rel       TEXT NOT NULL,                    -- member | owner | asset | part_of | client
  evidence  TEXT,                             -- json: [{ref, source}]
  at        INTEGER NOT NULL,
  PRIMARY KEY (src, dst, rel)
);
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);

-- task A waits on task B: one row per dependency. Blocked = some dependency still open.
CREATE TABLE IF NOT EXISTS task_deps (
  task        TEXT NOT NULL,                  -- the waiting task
  depends_on  TEXT NOT NULL,                  -- the task it waits on
  at          INTEGER NOT NULL,
  PRIMARY KEY (task, depends_on)
);

-- the interface-agnostic conversation -----------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
  id          TEXT PRIMARY KEY,
  title       TEXT,
  agent       TEXT,                            -- owning agent, or null for main
  parent_id   TEXT,                            -- subthread parent (null = root)
  anchor      TEXT,                            -- renderable id it hangs off
  state       TEXT NOT NULL DEFAULT 'open',    -- open | closed
  created_at  INTEGER NOT NULL
);

-- outbox: the renderable log. the same renderable id may recur (versions).
CREATE TABLE IF NOT EXISTS outbox (
  seq             INTEGER PRIMARY KEY AUTOINCREMENT,
  id              TEXT NOT NULL,               -- stable renderable id
  conversation_id TEXT NOT NULL,
  kind            TEXT NOT NULL,               -- advice | state_view | ask (a reminder is
                                               -- an advice; its body.type marks the flavor)
  author          TEXT NOT NULL,
  body            TEXT NOT NULL,               -- json
  actions         TEXT,                        -- json: [{label, token}]
  deliver_by      INTEGER,
  at              INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outbox_conv ON outbox(conversation_id, seq);

-- inbox: user actions, normalized by the interface
CREATE TABLE IF NOT EXISTS inbox (
  seq             INTEGER PRIMARY KEY AUTOINCREMENT,
  id              TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  kind            TEXT NOT NULL,               -- message | command | action_response
  channel         TEXT NOT NULL,               -- opaque handle {interface}:{address}
  text            TEXT,
  token           TEXT,
  anchor          TEXT,                         -- renderable id replied to
  at              INTEGER NOT NULL,
  handled         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_inbox_unhandled ON inbox(handled, seq);
CREATE INDEX IF NOT EXISTS idx_inbox_conv ON inbox(conversation_id, seq);

-- per-channel delivery cursor into the outbox
CREATE TABLE IF NOT EXISTS channel_cursors (
  channel         TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  last_seq        INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (channel, conversation_id)
);

-- audited drops and notable engine events (a drop is never a silent continue)
CREATE TABLE IF NOT EXISTS audit (
  id      TEXT PRIMARY KEY,
  kind    TEXT NOT NULL,                       -- e.g. drop, agent_run
  ref     TEXT,
  reason  TEXT,
  at      INTEGER NOT NULL
);

-- consent asks awaiting a yes/no. each ask has a STABLE id; the tap token carries
-- that id, so a tap always resolves the exact question the card showed — never a
-- newer, unseen one. the single applier writes a row here when the toll is 'ask'
-- and, on approval, deletes it in the same transaction as the write.
CREATE TABLE IF NOT EXISTS pending (
  ask_id          TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  kind            TEXT NOT NULL,               -- the target, e.g. goals.race
  content         TEXT NOT NULL,
  evidence        TEXT,                         -- json: [{ref, source}]
  author          TEXT NOT NULL,               -- the proposing agent
  origin          TEXT,                         -- why the ask was raised: consent-stakes | judge-conflict | identity
  existing_snapshot TEXT,                       -- json snapshot of the belief being replaced (conflicts)
  at              INTEGER,                       -- a task ask carries its scheduled time here (belief asks: NULL)
  ask_type        TEXT,                          -- 'task' | 'belief' | 'retire' | 'act' — how resolve applies it (unambiguous, not a kind prefix)
  meta            TEXT,                          -- json: an act proposal's arguments
  created_at      INTEGER NOT NULL
);
-- at most one open ask per (conversation, target): the dedup / supersede lookup
CREATE INDEX IF NOT EXISTS idx_pending_target ON pending(conversation_id, kind);

-- key/value: connector cursors, migration flags
CREATE TABLE IF NOT EXISTS kv (
  key         TEXT PRIMARY KEY,
  value       TEXT,
  updated_at  INTEGER NOT NULL
);

-- counters for observability: ticks, agent runs, connector polls (read by the CLI)
CREATE TABLE IF NOT EXISTS stats (
  name        TEXT PRIMARY KEY,
  count       INTEGER NOT NULL DEFAULT 0,
  last_at     INTEGER
);

-- current view per conversation: an agent's rendered panel spec (opaque json the
-- engine never parses). One row per conversation — the latest, not a log. An
-- interface reads it and draws the primitives; the engine stays agent-agnostic.
CREATE TABLE IF NOT EXISTS views (
  conversation_id TEXT PRIMARY KEY,
  body            TEXT NOT NULL,          -- json: {title, blocks:[...]}
  at              INTEGER NOT NULL
);

-- agent activity: what an agent DID (created/updated/retired a pattern, checked
-- in, or hit an error), so a background run leaves a scannable trail. Distinct
-- from `audit`, which records what the engine DROPPED or failed to do.
CREATE TABLE IF NOT EXISTS agent_log (
  id      TEXT PRIMARY KEY,
  at      INTEGER NOT NULL,
  agent   TEXT NOT NULL,
  action  TEXT NOT NULL,          -- pattern_created | pattern_updated | pattern_retired | error
  target  TEXT,                   -- what it acted on, e.g. a pattern id
  detail  TEXT                    -- the finding, the error, a short note
);
CREATE INDEX IF NOT EXISTS idx_agent_log_at ON agent_log(at);
"""


def home_dir() -> Path:
    """The data directory: ``KEN_HOME`` when set, else ``~/.ken``."""
    return Path(os.path.expanduser(os.environ.get("KEN_HOME") or DEFAULT_HOME))


def configure(path: str | None = None) -> Path:
    """Set the database file path. Creates the parent directory.

    Default is ``ken.db`` under ``home_dir()`` unless an explicit ``path`` is given."""
    global _DB_PATH
    if path is None:
        home = home_dir()
        path = home / "ken.db"
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        home.chmod(0o700)                 # ~/.ken holds keys and your data: owner only
    p = Path(os.path.expanduser(str(path)))
    p.parent.mkdir(parents=True, exist_ok=True)
    for f in (p, p.parent / "config.yaml"):  # tighten files an older version left 0644
        if f.exists():
            f.chmod(0o600)
    _DB_PATH = p
    return p


def db_path() -> Path:
    return _DB_PATH if _DB_PATH is not None else configure()


# ── engine singleton lock ────────────────────────────────────────────────────
# Exactly one engine may tick a given ~/.ken at a time (two would double-drain the
# inbox). An `flock` on this file enforces it: the holder keeps the fd for its whole
# life, and the OS releases the lock the instant that process exits — crash included —
# which is the race-free "is the engine alive?" signal a heartbeat can't give (a busy
# engine still holds the lock; a dead one has already released it).
# Both calls fail OPEN — no flock support (non-POSIX) or an unopenable lockfile means
# "no lock", so the engine still runs and a front-end spawns as it did before the lock.

_NO_LOCK = -1     # sentinel: acquired "successfully" but without a real lock (fail-open)


def engine_lock_path() -> Path:
    return db_path().parent / "engine.lock"


def acquire_engine_lock() -> int | None:
    """Take the exclusive engine lock for this process. Returns an fd to HOLD for the
    process's life (never close it), `_NO_LOCK` if locking is unavailable (non-POSIX, or
    the lockfile can't be opened) so the engine still runs unguarded, or None if another
    engine genuinely owns it (the caller should exit).

    Retries briefly: `engine_is_running()` probes by taking the lock and releasing it, so
    a starting engine can momentarily collide with a probe — a few retries clear that
    transient hold without a spurious exit; a real owner stays held and we give up."""
    if fcntl is None:
        return _NO_LOCK
    try:
        fd = os.open(engine_lock_path(), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return _NO_LOCK                   # can't open the lockfile → run unguarded, don't crash
    for attempt in range(5):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            if attempt < 4:
                time.sleep(0.02)          # maybe a probe's transient hold — retry, don't exit
    os.close(fd)
    return None                           # held for real → another engine owns it


def engine_is_running() -> bool:
    """True if some process holds the engine lock — so a front-end should attach, not
    spawn. Probes by trying to take it non-blocking, then releasing at once. No flock
    support or an unopenable lockfile reads as not-running (spawn as before)."""
    if fcntl is None:
        return False
    try:
        fd = os.open(engine_lock_path(), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False                      # got it → free → no engine holds it
    except OSError:
        return True                       # can't → held → an engine is running
    finally:
        os.close(fd)


_BUSY_TIMEOUT_MS = 5000   # wait this long for a WAL write lock before raising 'locked'


@asynccontextmanager
async def get_db():
    """One connection, row access by name. A busy timeout lets a write wait for the
    other process (the CLI shares this WAL file) instead of failing at once with
    'database is locked'."""
    p = db_path()
    if not p.exists():                    # create it owner-only; SQLite's -wal/-shm copy this mode
        os.close(os.open(p, os.O_CREAT | os.O_RDWR, 0o600))
    conn = await aiosqlite.connect(p)
    conn.row_factory = aiosqlite.Row
    await conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    try:
        yield conn
    finally:
        await conn.close()


async def init_db() -> None:
    """Create the schema. Idempotent."""
    async with get_db() as conn:
        await conn.executescript(SCHEMA)
        # tasks.scheduled_for was both 'deadline' and 'when I work on it'; it meant the deadline.
        # Rename it once; the work time is the new slot (never guessed from old data).
        cur = await conn.execute("PRAGMA table_info(tasks)")
        if "scheduled_for" in [r[1] for r in await cur.fetchall()]:
            try:
                await conn.execute("ALTER TABLE tasks RENAME COLUMN scheduled_for TO deadline")
            except sqlite3.OperationalError as exc:
                if "no such column" not in str(exc):
                    raise                  # else another process (engine + MCP booting together) won the race
        await _add_column(conn, "tasks", "slot_start", "INTEGER")
        await _add_column(conn, "tasks", "slot_minutes", "INTEGER")
        await _add_column(conn, "tasks", "slot_by", "TEXT")
        # add-column migrations for DBs created before a column existed (CREATE TABLE
        # IF NOT EXISTS never alters an existing table).
        if await _add_column(conn, "tasks", "reminded_at", "INTEGER"):
            # a task already PAST-due at upgrade time must not surprise-nudge on the
            # first tick — treat it as already reminded. Future-scheduled tasks keep
            # reminded_at NULL and fire normally at their time.
            await conn.execute(
                "UPDATE tasks SET reminded_at=? WHERE deadline IS NOT NULL "
                "AND deadline <= ?", (now_ms(), now_ms()))
        # evidence on the other two nouns, and the two pending columns Steps 2-3
        # consume. All nullable — an existing row migrates to NULL cleanly; the
        # write paths supply the value going forward. No backfill needed.
        await _add_column(conn, "tasks", "evidence", "TEXT")
        # completion time: a task done before this column existed has no known time —
        # leave it NULL (never fake one), so "done this week" counts only real completions.
        await _add_column(conn, "tasks", "completed_at", "INTEGER")
        await _add_column(conn, "tasks", "project", "TEXT")     # the project a task belongs to: a fact, not a name match
        await _add_column(conn, "tasks", "drop_reason", "TEXT")  # why a task was discarded (status 'dropped')
        await _add_column(conn, "tasks", "owner", "TEXT")        # NULL = the user
        await _add_column(conn, "tasks", "changed_at", "INTEGER")  # NULL on old rows = never stale
        await _add_column(conn, "events", "evidence", "TEXT")
        await _add_column(conn, "pending", "origin", "TEXT")
        await _add_column(conn, "pending", "existing_snapshot", "TEXT")
        await _add_column(conn, "pending", "at", "INTEGER")   # a task ask carries its scheduled time
        await _add_column(conn, "pending", "ask_type", "TEXT") # 'task' | 'belief' — how resolve applies it
        await _add_column(conn, "pending", "meta", "TEXT")     # an act proposal's arguments (json)
        await _add_column(conn, "pending", "action_id", "TEXT")  # the inbound action that raised it
        await _add_column(conn, "inbox", "meta", "TEXT")          # json: task_id / provenance / source
        await _add_column(conn, "events", "until", "INTEGER")     # when it stops mattering: hides, never deletes
        await _add_column(conn, "structured", "url", "TEXT")      # an asset's link (a sheet, a repo, a doc)
        # the connector layer is gone (agents fetch through their own tools); drop its
        # tables on a DB that still has them. Idempotent.
        for t in ("raw_events", "metrics", "notes"):
            await conn.execute(f"DROP TABLE IF EXISTS {t}")
        await conn.commit()


async def _add_column(conn, table: str, column: str, decl: str) -> bool:
    """Add a column if the table lacks it. Returns True only when it just added it,
    so a one-time backfill can run exactly on the migrating startup."""
    cur = await conn.execute(f"PRAGMA table_info({table})")
    if column in [r[1] for r in await cur.fetchall()]:
        return False
    try:
        await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc):
            raise
        return False      # another process (engine + MCP server booting together) won the race
    return True


async def audit(kind: str, ref: str | None, reason: str) -> None:
    """Record a drop or a notable engine event. One writer for every module, so
    the audit shape lives in exactly one place (a drop is never a silent continue)."""
    async with get_db() as c:
        await c.execute(
            "INSERT INTO audit (id, kind, ref, reason, at) VALUES (?,?,?,?,?)",
            (new_id(), kind, ref, reason, now_ms()))
        await c.commit()
