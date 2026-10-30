"""Phase D — the dev-loop seam: config-disable a built-in, and live reload.

`agents.disabled` turns a shipped built-in OFF without deleting repo files (the default
assistant is exempt — it must always answer). `ken reload` rebuilds the registry onto
the LIVE engine via `adopt_registry`, so a dropped/edited agent is picked up without a
restart. These build the real engine (`app.build_engine`) so the actual gates run.
"""
from ken import app, db
from ken.memory import store

_COACH = '''
from ken.agents.base import Agent, OnMessage, Slice, Advice
class Coach(Agent):
    name = "coach"
    role = "nags you kindly"
    session = "coach"
    triggers = [OnMessage()]
    slice = Slice()
    async def run(self, ctx):
        return Advice("pick the oldest")
'''


def _prep(tmp_path, config_yaml=""):
    db.configure(str(tmp_path / "ken.db"))
    (tmp_path / "config.yaml").write_text(config_yaml)


def _drop_agent(tmp_path, stem, body):
    (tmp_path / "agents").mkdir(exist_ok=True)
    (tmp_path / "agents" / f"{stem}.py").write_text(body)


# ── config-disable a built-in ────────────────────────────────────────────────

def test_config_disables_an_agent(tmp_path):
    _prep(tmp_path, "agents:\n  disabled: [coach]\n")
    _drop_agent(tmp_path, "coach", _COACH)
    eng = app.build_engine()
    assert "coach" not in eng._by_name            # the disabled agent is OFF...
    assert "kip" in eng._by_name                   # ...Kip is untouched
    assert eng.discovery_skips == []               # a chosen disable is not a failure


def test_disable_accepts_a_bare_string_and_a_handle_slug(tmp_path):
    # a single string (not a list), matched by the same slug rule as the @handle.
    _prep(tmp_path, "agents:\n  disabled: COACH\n")
    _drop_agent(tmp_path, "coach", _COACH)
    eng = app.build_engine()
    assert "coach" not in eng._by_name


def test_the_default_assistant_cannot_be_disabled(tmp_path):
    _prep(tmp_path, "agents:\n  disabled: [kip]\n")
    eng = app.build_engine()
    assert eng.assistant is not None and eng.assistant.name == "kip"   # Kip still answers
    assert "kip" in eng._by_name


# ── live reload (what `ken reload` triggers) ────────────────────────────────

def test_adopt_registry_swaps_in_a_fresh_registry(tmp_path):
    _prep(tmp_path, "")
    _drop_agent(tmp_path, "coach", _COACH)
    live = app.build_engine()
    assert "coach" in live._by_name
    live._seeded = True

    _prep(tmp_path, "agents:\n  disabled: [coach]\n")   # rebuild with coach off
    live.adopt_registry(app.build_engine())
    assert "coach" not in live._by_name            # the live engine reflects the rebuild
    assert live._seeded is False                   # re-seed next tick


async def test_reload_picks_up_a_new_dropin_and_clears_the_flag(tmp_path):
    _prep(tmp_path, "")
    await db.init_db()
    live = app.build_engine()
    assert "coach" not in live._by_name

    _drop_agent(tmp_path, "coach", _COACH)         # author drops a new agent...
    await store.set_meta(app.RELOAD_FLAG, "1")      # ...and runs `ken reload`
    await app._reload(live)

    assert "coach" in live._by_name                 # picked up without a restart
    assert not await store.get_meta(app.RELOAD_FLAG)  # the request was consumed


async def test_reload_republishes_the_roster(tmp_path):
    _prep(tmp_path, "")
    await db.init_db()
    live = app.build_engine()
    _drop_agent(tmp_path, "coach", _COACH)
    await app._reload(live)
    import json

    from ken.memory import store
    roster = json.loads(await store.get_meta("roster") or "[]")
    assert any(a["name"] == "coach" for a in roster)   # a client reading the roster sees the new agent


async def _count_audit(kind):
    from ken import db as _db
    async with _db.get_db() as c:
        cur = await c.execute("SELECT COUNT(*) AS n FROM audit WHERE kind=?", (kind,))
        return (await cur.fetchone())["n"]


async def test_a_permanently_broken_dropin_is_trailed_once_across_reloads(tmp_path):
    # a broken file that survives many reloads must not spam an identical row each time.
    _prep(tmp_path, "")
    await db.init_db()
    _drop_agent(tmp_path, "broken", "this is ::: not python\n")
    live = app.build_engine()
    await app._announce_registry(live)           # startup trail
    await app._reload(live)                       # reload 1
    await app._reload(live)                       # reload 2 — same broken file
    assert await _count_audit("plugin_skipped") == 1   # trailed exactly once
