"""tell(at=...) — an older message keeps its own date — and the KEN_NOW frozen clock."""
import asyncio
import json

from ken import db, mcp_server
from ken.agents.assistant import Assistant
from ken.contracts import Provenance
from ken.conversation import bus
from ken.engine import Engine
from ken.memory import store
from ken.util import local_ms, now_ms


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


class ModelThatReadsTheDate:
    """Stands in for kip's model: resolves 'by the 14th' from the date the prompt gives it,
    and logs the happening with no `when` (so it must default to when it was said)."""
    async def complete(self, system, messages, *, tools=None):
        self.system = system
        specs = {t.name: t for t in tools or []}
        said = system.split("This message was said on ")[1].split()[1]          # 'YYYY-MM-DD'
        await specs["add_task"].fn(text="send Mei the deck", when=said[:8] + "14T17:00")
        await specs["log_event"].fn(text="call with Mei")
        return "noted"


async def test_dated_tell_resolves_against_the_message_day(tmp_path, monkeypatch):
    await _setup(tmp_path)
    monkeypatch.setenv("KEN_NOW", "2026-03-11T12:00")                  # the harness: session day
    llm = ModelThatReadsTheDate()
    eng = Engine()
    eng.register_agent(Assistant(llm=llm), assistant=True)

    async def pump():
        for _ in range(60):
            await eng.pump_inbox()
            await asyncio.sleep(0.05)
    t = asyncio.create_task(pump())
    out = json.loads(await mcp_server.tell("call with Mei: deck by the 14th", "import",
                                           provenance="user", at="2026-03-11T09:30", timeout=5))
    t.cancel()
    assert out["text"] == "noted", out
    assert "This message was said on Wednesday 2026-03-11 09:30" in llm.system
    assert "The current local date and time is Wednesday 2026-03-11 12:00" in llm.system
    (task,) = await store.open_tasks()
    assert task.deadline == local_ms("2026-03-14T17:00")
    async with db.get_db() as c:
        rows = await (await c.execute("SELECT summary, at FROM events")).fetchall()
        pend = await (await c.execute("SELECT content, at FROM pending")).fetchall()
    ats = {r["at"] for r in [*rows, *pend] if "Mei" in r[0]}
    assert ats == {local_ms("2026-03-11T09:30")}                       # the event: when it was said


async def test_bad_at_is_rejected_and_nothing_is_queued(tmp_path):
    await _setup(tmp_path)
    out = json.loads(await mcp_server.tell("hello", "x", at="the 14th", timeout=0.5))
    assert out["outcome"] == "invalid" and "nothing was told" in out["text"]
    assert await bus.take_unhandled_actions() == []
    await mcp_server.tell("hello", "x", at="2026-03-11", timeout=0.2)
    ((_, a),) = await bus.take_unhandled_actions()
    assert a.meta["at"] == local_ms("2026-03-11")


async def test_ken_now_moves_the_lanes(tmp_path, monkeypatch):
    await _setup(tmp_path)
    await store.add_task("pay the fine", Provenance.USER_STATED, deadline=local_ms("2026-03-14T10:00"))
    monkeypatch.setenv("KEN_NOW", "2026-03-13")
    assert now_ms() == local_ms("2026-03-13")
    (row,) = json.loads(await mcp_server.tasks())["tasks"]
    assert row["lane"] != "overdue"
    monkeypatch.setenv("KEN_NOW", "2026-03-15T08:00")                  # read per call: no restart
    (row,) = json.loads(await mcp_server.tasks())["tasks"]
    assert row["lane"] == "overdue"
