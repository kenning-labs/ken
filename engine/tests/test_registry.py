"""Drop-file discovery — the plug-and-play seam.

A fixture agent written into a temp dir is discovered, built, registered, and
attributed by its `name`; a broken file is skipped (recorded, not fatal); a
duplicate name collides loudly. The built-in Kip is always discovered too. No DB
needed — discovery is pure wiring.
"""
from ken import registry
from ken.agents.base import agent_name
from ken.engine import Engine

_COACH = '''
from ken.agents.base import Agent, OnMessage, Slice, Advice
class Coach(Agent):
    name = "coach"
    role = "nags you kindly"
    session = "coach"                 # its own tab (Kip owns "main")
    triggers = [OnMessage()]
    slice = Slice(tasks="open")
    async def run(self, ctx):
        return Advice("pick the oldest")
'''


def _write(d, stem, body):
    (d / f"{stem}.py").write_text(body)


def _agent_src(name, session, extra=""):
    return (
        "from ken.agents.base import Agent, OnMessage, Slice, Advice\n"
        "class A(Agent):\n"
        f"    name = {name!r}\n"
        f"    session = {session!r}\n"
        f"    {extra}\n"
        "    triggers = [OnMessage()]\n"
        "    slice = Slice()\n"
        "    async def run(self, ctx): return Advice('hi')\n"
    )


def _discover(tmp_path):
    eng = Engine()
    skips = registry.discover_agents(eng, build=lambda cls: cls(), extra_dirs=[tmp_path])
    return eng, skips


def test_built_in_kip_is_always_discovered(tmp_path):
    eng, skips = _discover(tmp_path)
    assert "kip" in eng._by_name and eng.assistant is eng._by_name["kip"]
    assert skips == []


def test_discovers_a_dropin_agent_and_attributes_it_by_name(tmp_path):
    _write(tmp_path, "coach", _COACH)
    eng, skips = _discover(tmp_path)
    assert skips == []
    assert "coach" in eng._by_name and "kip" in eng._by_name    # drop-in + built-in
    coach = eng._by_name["coach"]
    assert agent_name(coach) == "coach"
    assert coach.session == "coach"                             # its own tab (declared)
    assert eng.assistant is not coach                          # not the default


def test_a_broken_dropin_is_skipped_not_fatal(tmp_path):
    _write(tmp_path, "broken", "this is not valid python :::\n")
    eng, skips = _discover(tmp_path)
    assert any("broken" in src for src, _ in skips)            # the failure is recorded
    assert "kip" in eng._by_name                               # the good agents still register


def test_a_file_with_no_agent_is_ignored(tmp_path):
    _write(tmp_path, "notes", "greeting = 'hello'\n")
    eng, skips = _discover(tmp_path)
    assert skips == []                                         # not an error — nothing to register
    assert set(eng._by_name) == {"kip"}                       # only the built-in; notes added none


def test_a_duplicate_name_collides_loudly(tmp_path):
    _write(tmp_path, "dupe", _COACH.replace('name = "coach"', 'name = "kip"'))
    eng, skips = _discover(tmp_path)
    assert any("kip" in err and "already registered" in err for _, err in skips)
    assert "coach" not in eng._by_name                         # it was the renamed one


def test_one_broken_file_does_not_hide_the_good_ones(tmp_path):
    # per-file isolation: a broken module sorted first must NOT drop later good ones.
    _write(tmp_path, "aaa_broken", "this is ::: not python\n")
    _write(tmp_path, "coach", _COACH)
    eng, skips = _discover(tmp_path)
    assert "coach" in eng._by_name and "kip" in eng._by_name   # good ones survive
    assert any("aaa_broken" in src for src, _ in skips)         # broken recorded


def test_a_second_default_assistant_collides_loudly(tmp_path):
    _write(tmp_path, "boss", _agent_src("boss", "boss", "default_assistant = True"))
    eng, skips = _discover(tmp_path)
    assert eng.assistant.name == "kip"                          # Kip stays the default
    assert any("default_assistant" in err for _, err in skips)


def test_handle_is_a_slug_of_the_name(tmp_path):
    from ken.agents.base import agent_handle
    _write(tmp_path, "day", _agent_src("Day Organizer", "day"))
    eng, skips = _discover(tmp_path)
    assert skips == []
    agent = eng._by_name["Day Organizer"]
    assert agent_handle(agent) == "day-organizer"              # clean, single token
    assert "day-organizer" in eng._by_handle


def test_case_only_distinct_names_collide_on_handle(tmp_path):
    # "Coach" and "coach" are distinct names but slug to the same handle — refuse it.
    _write(tmp_path, "a", _agent_src("Coach", "c1"))
    _write(tmp_path, "b", _agent_src("coach", "c2"))
    eng, skips = _discover(tmp_path)
    assert "coach" in eng._by_handle                            # exactly one wins
    assert any("collides" in err for _, err in skips)


async def test_publish_roster_round_trips_to_the_ui(tmp_path):
    # the engine publishes the registered agents; the UI (a separate process) reads
    # them back from kv['roster'].
    import json

    from ken import db
    from ken.memory import store
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    eng = Engine()
    registry.discover_agents(eng, build=lambda cls: cls(), extra_dirs=[])   # only Kip
    await eng.publish_roster()
    roster = json.loads(await store.get_meta("roster"))
    assert any(a["name"] == "kip" and a["role"] == "your companion" and a["default"]
               for a in roster)
