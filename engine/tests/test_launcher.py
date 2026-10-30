"""The `ken` command: the config helpers, and the arg-parser routing.

`db.configure(tmp_path/…)` ties `config.config_path()` to the temp dir, so each
test reads and writes its own config, never your real ~/.ken one.
"""
import yaml

from ken import config, db
from ken.__main__ import build_parser

# --- config helpers ---------------------------------------------------------

def test_has_api_key_false_when_missing(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    assert config.has_api_key() is False


def test_has_api_key_false_for_placeholder(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    config.config_path().write_text(yaml.safe_dump({"llm": {"api_key": "YOUR_OPENAI_API_KEY"}}))
    assert config.has_api_key() is False





# --- arg-parser routing -----------------------------------------------------

def test_no_subcommand_is_the_engine():
    assert build_parser().parse_args([]).cmd is None      # main() treats None as `run`


def test_cards_and_decide_route():
    assert build_parser().parse_args(["cards"]).cmd == "cards"
    ns = build_parser().parse_args(["decide", "abc", "yes", "--edited", "x"])
    assert (ns.cmd, ns.card_id, ns.answer, ns.edited) == ("decide", "abc", "yes", "x")


def test_run_routes_to_engine():
    assert build_parser().parse_args(["run"]).cmd == "run"


def test_serve_routes_to_engine():
    ns = build_parser().parse_args(["serve"])
    assert ns.cmd == "serve"
    assert ns.interval == 1.0   # serve carries the same engine args as run


# --- cards from the terminal -------------------------------------------------

def test_ken_cards_prints_the_open_cards_and_how_to_answer(tmp_path, capsys):
    import asyncio

    from ken.__main__ import _cmd_cards, _cmd_decide, system_snapshot
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.conversation import bus
    db.configure(str(tmp_path / "ken.db"))

    async def seed():
        await db.init_db()
        await gate.apply(Proposal(intent="structured:people.jack", content="Jack is my manager"),
                         conversation_id=bus.MAIN, author="kip")
        await gate.apply(Proposal(intent="batch:hydrate", content="Save 1 from slack?",
                                  meta={"items": [{"kind": "people", "name": "Robert", "note": "PM"}]}),
                         conversation_id=bus.MAIN, author="kip")
        return await system_snapshot()
    assert asyncio.run(seed())["counts"]["pending"] == 2
    _cmd_cards()
    out = capsys.readouterr().out
    assert "[Save person] Jack is my manager" in out and "[Save these?] Save 1 from slack?" in out
    assert "   - Robert (person): PM" in out and "ken decide <id> yes|no" in out
    # no engine holds the lock here: decide refuses before it queues anything
    _cmd_decide("abc", "yes", "")
    assert "no engine running" in capsys.readouterr().out


# --- push ---------------------------------------------------------------------

def test_ken_push_sends_the_digest_only_when_cards_wait(tmp_path, capsys, monkeypatch):
    import asyncio

    import httpx

    from ken.__main__ import _cmd_push
    from ken.agents.base import Proposal
    from ken.consent import gate
    from ken.conversation import bus
    db.configure(str(tmp_path / "ken.db"))
    asyncio.run(db.init_db())
    sent = []

    def fake_post(url, *, content, headers, timeout):
        if "down" in url:
            raise httpx.ConnectError("no route", request=httpx.Request("POST", url))
        sent.append((url, content.decode(), headers))
        return httpx.Response(200, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake_post)
    config.set_section_value("push", "url", "https://ntfy.sh/ken-test")
    config.set_section_value("push", "headers", {"Title": "Ken", "Priority": 5})    # yaml int, as ntfy documents
    assert _cmd_push(dry_run=False) == 0
    assert "nothing sent" in capsys.readouterr().out and sent == []        # nothing waits → nothing sent
    asyncio.run(gate.apply(Proposal(intent="structured:people.jack", content="Jack is my manager"),
                           conversation_id=bus.MAIN, author="kip"))
    assert _cmd_push(dry_run=True) == 0
    assert "(dry run)" in capsys.readouterr().out and sent == []
    assert _cmd_push(dry_run=False) == 0
    (url, body, headers) = sent[0]
    assert url == "https://ntfy.sh/ken-test" and body.startswith("Ken · 1 open question")
    assert headers["Title"] == "Ken" and headers["Priority"] == "5" and headers["Content-Type"].startswith("text/plain")
    out = capsys.readouterr().out
    assert "sent → ntfy.sh (200)" in out and "ken-test" not in out.splitlines()[-1]   # the topic never hits the log
    config.set_section_value("push", "url", "https://down.example/x")
    assert _cmd_push(dry_run=False) == 1
    assert "push failed: down.example: ConnectError" in capsys.readouterr().out
