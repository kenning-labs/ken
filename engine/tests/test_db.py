"""The schema builds, and it is idempotent."""
from ken import db

EXPECTED_TABLES = {
    "tasks", "events", "structured",
    "conversations", "outbox", "inbox", "channel_cursors", "audit", "kv",
}
GONE_TABLES = {"raw_events", "metrics", "notes"}   # the connector layer, dropped on migration


async def _tables() -> set:
    async with db.get_db() as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return {r[0] for r in await cur.fetchall()}


def test_ken_home_env_sets_the_default_db(tmp_path, monkeypatch):
    monkeypatch.setenv("KEN_HOME", str(tmp_path / "home"))
    assert db.configure() == tmp_path / "home" / "ken.db"      # env → default db
    # an explicit path still wins over the env override
    assert db.configure(str(tmp_path / "x.db")) == tmp_path / "x.db"


def test_default_home_is_dot_ken(tmp_path, monkeypatch):
    from ken import config
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KEN_HOME", raising=False)
    assert db.configure() == tmp_path / ".ken" / "ken.db"            # fresh install
    assert config.config_path() == tmp_path / ".ken" / "config.yaml"


def test_engine_lock_is_exclusive_and_frees_on_close(tmp_path):
    """One engine at a time: a held flock blocks a second acquire and reads as running;
    releasing (the OS does this on process exit) frees it."""
    import os
    db.configure(str(tmp_path / "ken.db"))
    assert db.engine_is_running() is False        # nothing holds it yet
    fd = db.acquire_engine_lock()
    assert fd is not None
    assert db.engine_is_running() is True          # a separate fd can't take it
    assert db.acquire_engine_lock() is None        # nor can a second engine
    os.close(fd)                                   # the OS does this on exit; do it explicitly here
    assert db.engine_is_running() is False         # freed


async def test_schema_creates_all_tables(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    assert EXPECTED_TABLES <= await _tables()


async def test_init_db_is_idempotent(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    await db.init_db()  # must not raise
    assert EXPECTED_TABLES <= await _tables()


async def test_migration_drops_the_connector_tables(tmp_path):
    # a DB from before the cut still carries raw_events/metrics/notes; init_db drops
    # them (idempotently) and leaves the nouns untouched.
    db.configure(str(tmp_path / "ken.db"))
    async with db.get_db() as conn:
        await conn.executescript(
            "CREATE TABLE metrics (id TEXT PRIMARY KEY, value REAL);"
            "CREATE TABLE notes (id TEXT PRIMARY KEY, text TEXT);"
            "CREATE TABLE raw_events (id TEXT PRIMARY KEY);"
            "INSERT INTO metrics VALUES ('m', 1.0);")
        await conn.commit()
    await db.init_db()
    await db.init_db()
    tables = await _tables()
    assert EXPECTED_TABLES <= tables and not (GONE_TABLES & tables)


async def _columns(table: str) -> set:
    async with db.get_db() as conn:
        cur = await conn.execute(f"PRAGMA table_info({table})")
        return {r[1] for r in await cur.fetchall()}


async def test_migration_adds_columns_and_preserves_rows(tmp_path):
    # a DB built before this step: tasks/events lack `evidence`, pending lacks the two
    # Step 2-3 columns. init_db must add them AND leave the existing rows intact.
    db.configure(str(tmp_path / "ken.db"))
    async with db.get_db() as conn:
        await conn.executescript(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, text TEXT NOT NULL, "
            "  status TEXT NOT NULL DEFAULT 'open', scheduled_for INTEGER, "
            "  reminded_at INTEGER, provenance TEXT NOT NULL, created_at INTEGER NOT NULL);"
            "CREATE TABLE events (id TEXT PRIMARY KEY, summary TEXT NOT NULL, "
            "  at INTEGER NOT NULL, provenance TEXT NOT NULL);"
            "CREATE TABLE pending (ask_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, "
            "  kind TEXT NOT NULL, content TEXT NOT NULL, evidence TEXT, author TEXT NOT NULL, "
            "  created_at INTEGER NOT NULL);")
        await conn.execute(
            "INSERT INTO tasks (id, text, scheduled_for, provenance, created_at) VALUES "
            "('t1','book the dentist',1760950800000,'agent_inferred',10)")
        await conn.execute(
            "INSERT INTO events (id, summary, at, provenance) VALUES "
            "('e1','ran 10k',20,'agent_inferred')")
        await conn.execute(
            "INSERT INTO pending (ask_id, conversation_id, kind, content, author, created_at) "
            "VALUES ('a1','conv','goals.race','run a 10k','Goal',30)")
        await conn.commit()

    await db.init_db()

    assert "evidence" in await _columns("tasks")
    assert "evidence" in await _columns("events")
    assert {"origin", "existing_snapshot"} <= await _columns("pending")

    async with db.get_db() as conn:
        # existing rows survive; the new columns read back as NULL on them
        (text, ev) = await (await conn.execute(
            "SELECT text, evidence FROM tasks WHERE id='t1'")).fetchone()
        assert (text, ev) == ("book the dentist", None)
        # the old scheduled_for IS the deadline now, unchanged; no slot is ever guessed
        row = await (await conn.execute(
            "SELECT deadline, slot_start, slot_minutes FROM tasks WHERE id='t1'")).fetchone()
        assert tuple(row) == (1760950800000, None, None)
        (origin, snap) = await (await conn.execute(
            "SELECT origin, existing_snapshot FROM pending WHERE ask_id='a1'")).fetchone()
        assert (origin, snap) == (None, None)

    # second run is a clean no-op — columns already present, rows unchanged
    await db.init_db()
    assert "evidence" in await _columns("tasks") and "scheduled_for" not in await _columns("tasks")
    async with db.get_db() as conn:
        (n,) = await (await conn.execute("SELECT COUNT(*) FROM tasks")).fetchone()
        assert n == 1


async def test_get_db_sets_a_busy_timeout(tmp_path):
    # a shared WAL file: a write must WAIT for the other process's lock, not fail at
    # once with 'database is locked'. The busy timeout is what makes it wait. (FIX 2)
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    async with db.get_db() as conn:
        cur = await conn.execute("PRAGMA busy_timeout")
        (ms,) = await cur.fetchone()
    assert ms == db._BUSY_TIMEOUT_MS == 5000


async def test_private_files_are_owner_only(tmp_path, monkeypatch):
    """~/.ken holds API keys and your data: the dir is 0700, the db and config 0600."""
    from ken import config
    monkeypatch.setenv("KEN_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(mode=0o755)
    (tmp_path / "home" / "config.yaml").write_text("llm: {}\n")
    (tmp_path / "home" / "config.yaml").chmod(0o644)                # left loose by an old version
    db.configure()
    await db.init_db()
    config.set_section_value("llm", "api_key", "SECRET")
    config.write_bucket("slack", {"a": 1})
    home = tmp_path / "home"
    assert home.stat().st_mode & 0o777 == 0o700
    for f in ("ken.db", "config.yaml", "state/slack.yaml"):
        assert (home / f).stat().st_mode & 0o777 == 0o600, f
