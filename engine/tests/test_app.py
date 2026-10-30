"""The composition root: the wired engine answers a bare message end to end.

This is the Python side of the CLI seam — a message in the inbox, one tick, an
advice in the outbox — with the stub assistant the running app registers.
"""
import asyncio

from ken import db
from ken.app import _MAX_BACKOFF, _backoff, build_engine, tick_guarded
from ken.contracts import Action, ActionKind
from ken.conversation import bus
from ken.util import new_id, now_ms


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


# ── the daemon survives one bad tick (FIX 2) ─────────────────────────────────

class _BoomEngine:
    """A stand-in engine whose tick always raises — a transient DB error, say."""
    async def tick(self):
        raise RuntimeError("database is locked")


class _OkEngine:
    async def tick(self):
        return None


async def _tick_errors() -> int:
    async with db.get_db() as c:
        (n,) = await (await c.execute(
            "SELECT COUNT(*) FROM audit WHERE kind='tick_error'")).fetchone()
    return n


async def test_tick_guarded_audits_a_failure_and_does_not_raise(tmp_path):
    await _setup(tmp_path)
    ok = await tick_guarded(_BoomEngine())     # must NOT propagate — the loop lives on
    assert ok is False                         # a caught failure reports False
    assert await _tick_errors() == 1           # surfaced, never swallowed silently


async def test_tick_guarded_returns_true_on_success(tmp_path):
    await _setup(tmp_path)
    assert await tick_guarded(_OkEngine()) is True
    assert await _tick_errors() == 0           # a healthy tick writes no error row


async def test_tick_guarded_can_skip_the_audit_to_throttle_a_streak(tmp_path):
    await _setup(tmp_path)
    assert await tick_guarded(_BoomEngine(), audit=False) is False
    assert await _tick_errors() == 0           # logged, but NOT audited — no per-tick flood


async def test_tick_guarded_propagates_cancellation(tmp_path):
    await _setup(tmp_path)

    class _CancelEngine:
        async def tick(self):
            raise asyncio.CancelledError

    raised = False
    try:
        await tick_guarded(_CancelEngine())
    except asyncio.CancelledError:
        raised = True
    assert raised                              # Ctrl-C / shutdown is never swallowed


def test_backoff_grows_then_caps():
    assert _backoff(0, 1.0) == 1.0             # healthy: the normal interval
    assert _backoff(1, 1.0) == 1.0             # a single failure: still just the interval
    assert _backoff(2, 1.0) == 2.0             # then it grows
    assert _backoff(3, 1.0) == 4.0
    assert _backoff(100, 1.0) == _MAX_BACKOFF  # capped, never unbounded


async def test_stub_assistant_answers_a_bare_message(tmp_path):
    await _setup(tmp_path)
    eng = build_engine()

    await bus.append_action(Action(
        id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.MESSAGE,
        channel="cli:local", text="hello ken", at=now_ms()))
    await eng.tick()

    out = await bus.read_outbox(bus.MAIN, 0)
    assert any(r.kind.value == "advice" and r.body.get("outcome") == "no_model"
               for _, r in out)


async def test_empty_message_stays_quiet(tmp_path):
    await _setup(tmp_path)
    eng = build_engine()

    await bus.append_action(Action(
        id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.MESSAGE,
        channel="cli:local", text="   ", at=now_ms()))
    await eng.tick()

    assert await bus.read_outbox(bus.MAIN, 0) == []
