"""OnSchedule — cron matching, and the engine firing a time-driven agent once."""
# ruff: noqa: DTZ001 — naive datetimes are intentional: cron runs on local wall-clock
from datetime import datetime

import pytest

from ken import db
from ken.agents.base import Advice, Agent, OnSchedule, Slice
from ken.conversation import bus
from ken.engine import Engine
from ken.schedule import cron_matches


def test_cron_matches_daily():
    assert cron_matches("0 7 * * *", datetime(2026, 3, 2, 7, 0))
    assert not cron_matches("0 7 * * *", datetime(2026, 3, 2, 7, 1))
    assert not cron_matches("0 7 * * *", datetime(2026, 3, 2, 8, 0))


def test_cron_matches_step_and_list():
    assert cron_matches("*/15 * * * *", datetime(2026, 3, 2, 9, 30))
    assert not cron_matches("*/15 * * * *", datetime(2026, 3, 2, 9, 31))
    assert cron_matches("0 9,17 * * *", datetime(2026, 3, 2, 17, 0))


def test_cron_matches_weekday_sunday_is_zero_or_seven():
    assert datetime(2026, 3, 2).weekday() == 0           # a Monday (python Mon=0)
    assert cron_matches("0 9 * * 1", datetime(2026, 3, 2, 9, 0))       # Monday = cron 1
    assert not cron_matches("0 9 * * 1", datetime(2026, 3, 3, 9, 0))   # Tuesday
    assert datetime(2026, 3, 8).weekday() == 6           # a Sunday
    assert cron_matches("0 9 * * 0", datetime(2026, 3, 8, 9, 0))       # Sunday = 0
    assert cron_matches("0 9 * * 7", datetime(2026, 3, 8, 9, 0))       # ...and 7


def test_step_counts_from_the_field_minimum():
    assert cron_matches("0 0 */15 * *", datetime(2026, 3, 1, 0, 0))       # day 1
    assert cron_matches("0 0 */15 * *", datetime(2026, 3, 16, 0, 0))      # day 16
    assert not cron_matches("0 0 */15 * *", datetime(2026, 3, 15, 0, 0))  # NOT day 15
    assert cron_matches("0 0 1 */3 *", datetime(2026, 4, 1, 0, 0))        # month 4
    assert not cron_matches("0 0 1 */3 *", datetime(2026, 3, 1, 0, 0))    # NOT month 3


def test_bad_cron_raises():
    with pytest.raises(ValueError):
        cron_matches("0 7 * *", datetime(2026, 3, 2, 7, 0))


class Ping(Agent):
    triggers = [OnSchedule("0 7 * * *")]
    slice = Slice()
    session = "main"

    async def run(self, ctx):
        return Advice(text="ping")


class BadCron(Agent):
    triggers = [OnSchedule("not a cron")]
    slice = Slice()
    session = "bad"

    async def run(self, ctx):
        return Advice(text="should never fire")


async def _engine(tmp_path, *agents):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    eng = Engine()
    for a in agents:
        eng.register_agent(a)
    return eng


async def test_scheduled_agent_fires_once_per_occurrence(tmp_path):
    eng = await _engine(tmp_path, Ping())
    seven = int(datetime(2026, 3, 2, 7, 0, 0).timestamp() * 1000)   # local 07:00
    await eng._run_scheduled(seven)
    await eng._run_scheduled(seven + 3_000)        # same minute -> no second fire
    await eng._run_scheduled(seven + 61_000)       # 07:01 -> occurrence already fired
    out = await bus.read_outbox(bus.MAIN, 0)
    assert len([r for _, r in out if r.body.get("text") == "ping"]) == 1


async def test_catch_up_fires_a_recently_missed_occurrence(tmp_path):
    # nothing ran at 07:00 (box asleep); a tick at 07:05 must still fire it, once
    eng = await _engine(tmp_path, Ping())
    seven = int(datetime(2026, 3, 2, 7, 0, 0).timestamp() * 1000)
    await eng._run_scheduled(seven + 5 * 60_000)   # 07:05
    await eng._run_scheduled(seven + 6 * 60_000)   # 07:06 -> no refire
    out = await bus.read_outbox(bus.MAIN, 0)
    assert len([r for _, r in out if r.body.get("text") == "ping"]) == 1


async def test_a_bad_cron_is_isolated_not_fatal(tmp_path):
    # a malformed cron in one agent must be audited and skipped, not abort the run
    eng = await _engine(tmp_path, Ping(), BadCron())
    seven = int(datetime(2026, 3, 2, 7, 0, 0).timestamp() * 1000)
    await eng._run_scheduled(seven)                # must not raise
    out = await bus.read_outbox(bus.MAIN, 0)
    assert any(r.body.get("text") == "ping" for _, r in out)   # the good agent still fired
    async with db.get_db() as c:
        rows = await (await c.execute("SELECT kind FROM audit WHERE kind='bad_cron'")).fetchall()
    assert rows                                                 # the bad cron was audited
