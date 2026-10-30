"""`@name` routing — a leading `@handle` addresses that agent (mention stripped);
an unknown handle stays text and falls through to the default assistant."""
from ken import db
from ken.agents.base import Advice, Agent, OnMessage, Slice
from ken.contracts import Action, ActionKind
from ken.conversation import bus
from ken.engine import Engine
from ken.util import new_id, now_ms


class _Kip(Agent):
    name = "kip"
    default_assistant = True
    triggers = [OnMessage()]
    slice = Slice()
    seen = None

    async def run(self, ctx):
        type(self).seen = ctx.message
        return Advice("kip")


class _Spy(Agent):
    name = "spy"
    session = "spy"
    triggers = [OnMessage()]
    slice = Slice()
    seen = None

    async def run(self, ctx):
        type(self).seen = ctx.message
        return Advice("spy")


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


def _msg(text):
    return Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.MESSAGE,
                  channel="cli:local", text=text, anchor=None, at=now_ms())


def _engine():
    _Kip.seen = _Spy.seen = None
    eng = Engine()
    eng.register_agent(_Kip())
    eng.register_agent(_Spy())
    return eng


async def test_at_mention_routes_to_the_addressed_agent(tmp_path):
    await _setup(tmp_path)
    eng = _engine()
    await eng._route_message(_msg("@spy weather in Oslo"))
    assert _Spy.seen == "weather in Oslo"      # addressed, mention stripped
    assert _Kip.seen is None                   # Kip not run


async def test_no_mention_routes_to_the_default_assistant(tmp_path):
    await _setup(tmp_path)
    eng = _engine()
    await eng._route_message(_msg("what's on my plate?"))
    assert _Kip.seen == "what's on my plate?"
    assert _Spy.seen is None


async def test_unknown_mention_falls_through_as_text(tmp_path):
    await _setup(tmp_path)
    eng = _engine()
    await eng._route_message(_msg("@nobody hello there"))
    assert _Kip.seen == "@nobody hello there"  # unknown handle stays literal → Kip
    assert _Spy.seen is None


async def test_mention_tolerates_punctuation_and_bare_handle(tmp_path):
    await _setup(tmp_path)
    for text, rest in [("@spy, weather please", "weather please"),
                       ("@spy: weather", "weather"),
                       ("@spy?", ""),
                       ("@spy", ""),
                       ("@spy   hi", "hi")]:
        eng = _engine()
        await eng._route_message(_msg(text))
        assert _Spy.seen == rest and _Kip.seen is None, (text, _Spy.seen)
