"""Observability counters — the data behind the CLI System dashboard."""
from ken import db
from ken.agents.base import Advice, Agent, OnMessage, Slice
from ken.engine import Engine
from ken.memory import store


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


async def _stats():
    async with db.get_db() as c:
        cur = await c.execute("SELECT name, count, last_at FROM stats")
        return {r["name"]: (r["count"], r["last_at"]) for r in await cur.fetchall()}


async def test_bump_counts_and_stamps(tmp_path):
    await _setup(tmp_path)
    await store.bump("run:X", 100)
    await store.bump("run:X", 200, by=2)
    assert (await _stats())["run:X"] == (3, 200)      # summed, last time kept


class Speaker(Agent):
    triggers = [OnMessage()]
    slice = Slice()

    async def run(self, ctx):
        return Advice(text="hi")


class Quiet(Agent):
    triggers = [OnMessage()]
    slice = Slice()
    session = "q"

    async def run(self, ctx):
        return None


async def test_run_agent_records_ran_and_spoke(tmp_path):
    await _setup(tmp_path)
    await Engine().run_agent(Speaker(), message="yo")
    s = await _stats()
    assert s["run:Speaker"][0] == 1 and s["spoke:Speaker"][0] == 1


async def test_a_quiet_run_counts_ran_but_not_spoke(tmp_path):
    await _setup(tmp_path)
    await Engine().run_agent(Quiet())
    s = await _stats()
    assert s["run:Quiet"][0] == 1 and "spoke:Quiet" not in s


async def test_tick_writes_a_heartbeat(tmp_path):
    await _setup(tmp_path)
    await Engine().tick()
    s = await _stats()
    assert s["tick"][0] == 1 and s["tick"][1] is not None
