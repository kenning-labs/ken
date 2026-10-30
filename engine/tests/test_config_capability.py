"""ctx.config — scoped, audited, agent-writable state buckets.

A bucket is a plain yaml file under ~/.ken/state/ (separate from config.yaml, so nothing
agent-writable touches secrets). The ConfigStore is built from an agent's declared grant:
a granted bucket reads/writes, an ungranted one raises, a read-only one refuses writes.
Every write is atomic and leaves an audit row."""
import pytest

from ken import config, db
from ken.capabilities import ConfigStore, ScopeError


def _store(read, write):
    return ConfigStore(read=read, write=write, author="tester")


def test_bucket_file_roundtrip(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    assert config.read_bucket("slack") == {}                 # absent → empty
    config.write_bucket("slack", {"watch": ["#eng"]})
    assert config.read_bucket("slack") == {"watch": ["#eng"]}
    # it lives under state/, NOT in config.yaml
    assert (config.state_dir() / "slack.yaml").exists()
    assert not config.config_path().exists()                 # secrets file untouched


def test_ungranted_bucket_raises(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    store = _store(read=["slack"], write=["slack"])
    with pytest.raises(ScopeError):
        store("github")                                      # not in the grant → loud


async def test_write_is_scoped_atomic_and_audited(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    b = _store(read=["slack"], write=["slack"])("slack")
    await b.add("watch", "#eng")
    await b.add("watch", "#eng")                             # idempotent — no dup
    await b.add("watch", "#oncall")
    await b.remove("watch", "#eng")
    assert config.read_bucket("slack") == {"watch": ["#oncall"]}
    # each mutation left an audit row (history + undo)
    async with db.get_db() as c:
        rows = await (await c.execute("SELECT kind, ref, reason FROM audit WHERE kind='config'")).fetchall()
    assert len(rows) == 4 and rows[0]["ref"] == "slack:tester"
    assert any("add '#eng' to watch" in r["reason"] for r in rows)


async def test_read_only_bucket_refuses_writes(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    b = _store(read=["slack"], write=[])("slack")            # read granted, write not
    assert b.get("watch", []) == []                          # read works
    with pytest.raises(ScopeError):
        await b.add("watch", "#eng")                         # write refused


def test_bucket_name_cannot_traverse_to_secrets(tmp_path):
    """The headline invariant: a bucket name is a NAME, never a path. '../config' must not
    resolve to ~/.ken/config.yaml (which holds the api key)."""
    db.configure(str(tmp_path / "ken.db"))
    config.set_section_value("llm", "api_key", "SECRET")     # write the secrets file
    config.write_bucket("slack", {"watch": ["#eng"]})        # ensure state/ exists
    for bad in ["../config", "../../etc/passwd", "a/b", "with.dot"]:
        with pytest.raises(ValueError):
            config.read_bucket(bad)
        with pytest.raises(ValueError):
            config.write_bucket(bad, {})


def test_empty_grant_raises_on_any_bucket(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    with pytest.raises(ScopeError):
        _store(read=[], write=[])("slack")                   # an agent with no grant → loud, not None


async def test_write_grant_implies_read(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    b = _store(read=[], write=["slack"])("slack")            # write-only grant
    await b.add("watch", "#eng")
    assert b.get("watch") == ["#eng"]                        # can read what it may change


async def test_add_on_a_scalar_key_raises_not_mangles(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    b = _store(read=["slack"], write=["slack"])("slack")
    await b.set("tone", "terse")                             # a scalar preference
    with pytest.raises(ValueError):
        await b.add("tone", "x")                             # not iterated char-by-char
    assert b.get("tone") == "terse"                          # unchanged


async def test_add_on_a_hand_edited_null_key_works(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    config.write_bucket("slack", {"watch": None})            # `watch:` typed by hand → yaml null
    b = _store(read=["slack"], write=["slack"])("slack")
    await b.add("watch", "#eng")                             # coerced to [], not a crash
    assert b.get("watch") == ["#eng"]


async def test_concurrent_writes_do_not_lose_updates(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    b = _store(read=["slack"], write=["slack"])("slack")
    import asyncio
    await asyncio.gather(b.add("watch", "x"), b.add("watch", "y"), b.add("watch", "z"))
    assert sorted(b.get("watch")) == ["x", "y", "z"]         # the per-bucket lock keeps all three


def test_bucket_name_with_trailing_newline_rejected(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    with pytest.raises(ValueError):
        config.read_bucket("slack\n")                        # `$` would allow it; fullmatch does not
