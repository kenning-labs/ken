"""`ken` — the one command. With no subcommand it runs the engine:

  ken                 the long-lived tick loop (== run)
  ken run             start the long-lived tick loop
  ken serve           alias of `run` — the always-on headless engine
  ken init            first-time setup: home, database, model, Claude Code, background service
  ken service install|uninstall|status   run the engine in the background (launchd / systemd)
  ken init-db         create the schema, then exit
  ken tick-once       run exactly one tick, then exit
  ken cards           the open consent cards
  ken push            send the open cards to push.url (run it from a 09:00 scheduler)
  ken decide ID yes|no [--edited …]   answer one card (the same tap any client makes)
  ken maintain        run the maintainer pass now

Dev-loop commands for authoring plug-in agents/connectors (drop a file into
``~/.ken/agents`` or ``~/.ken/connectors`` — see the `example` files):

  ken eval            the card scoreboard: approve rate per agent + card type, and ignored cards
  ken agents          list the discovered agents + connectors, and any load-skips
  ken agent test NAME [MESSAGE]   run one agent once and print what it returns
  ken reload          tell the running engine to rebuild its registry (no restart)

There is no terminal UI: the interfaces are the MCP server (`ken-mcp`) and any client
that reads the outbox and writes the inbox. Everything shares one SQLite file (~/.ken/ken.db).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from . import config, db, service
from .agents.base import Advice, agent_handle, agent_name, handle_slug
from .app import RELOAD_FLAG, build_engine, run
from .memory import store


def _cmd_agents() -> None:
    """`ken agents` — the plug-in doctor: what discovery actually registered, and why
    anything was skipped. Pure wiring, no DB — safe to run anytime."""
    eng = build_engine()
    print("Agents")
    for a in eng.agents:
        tag = "  (default)" if a is eng.assistant else ""
        role = getattr(a, "role", "") or ""
        print(f"  {agent_name(a):<16} @{agent_handle(a):<16} {role}{tag}")
    if eng.discovery_skips:
        print("Skipped")
        for src, err in eng.discovery_skips:
            print(f"  {src}\n      {err}")


def _cmd_agent_test(name: str, message: str) -> None:
    """`ken agent test NAME [MESSAGE]` — run one agent once and print its output. It
    assembles the real slice context and calls the agent, but does NOT post the reply
    into a conversation or apply a proposal — a smoke test, not a live turn. (A tool the
    agent calls that writes will still act; most authored agents only return Advice.)"""
    async def _go() -> None:
        await db.init_db()
        eng = build_engine()
        agent = eng._by_handle.get(handle_slug(name)) or eng._by_name.get(name)
        if agent is None:
            print(f"ken: no agent named {name!r}. Run `ken agents` to see the roster.")
            return
        ctx = await eng._assemble(agent, message=message)
        out = await agent.run(ctx)
        label = agent_name(agent)
        if out is None:
            print(f"{label}: (quiet — the agent chose not to respond)")
        elif isinstance(out, Advice):
            print(f"{label}: {out.text}")
        else:
            print(f"{label}: [proposal — not applied in a test] {out}")
    asyncio.run(_go())


async def eval_rows(now: int | None = None) -> dict:
    """The scoreboard behind `ken eval`: per (author, ask_type) counts from the
    `card_outcome` audit rows, plus `ignored` — cards still pending after 24h."""
    from .util import now_ms
    now = now or now_ms()
    rows: dict = {}
    async with db.get_db() as c:
        cur = await c.execute("SELECT reason FROM audit WHERE kind='card_outcome'")
        for (reason,) in await cur.fetchall():
            try:
                d = json.loads(reason)
            except (json.JSONDecodeError, TypeError):
                continue                                  # a pre-JSON row: skip, never miscount
            key = (str(d.get("author", "")), str(d.get("ask_type", "belief")))
            r = rows.setdefault(key, {"cards": 0, "approved": 0, "rejected": 0, "ignored": 0})
            r["cards"] += 1
            if d.get("outcome") in ("approved", "rejected"):
                r[d["outcome"]] += 1
        cur = await c.execute("SELECT author, ask_type FROM pending WHERE created_at < ?",
                              (now - 86_400_000,))
        for author, ask_type in await cur.fetchall():
            r = rows.setdefault((author, ask_type or "belief"),
                                {"cards": 0, "approved": 0, "rejected": 0, "ignored": 0})
            r["ignored"] += 1
    return rows


def _cmd_eval() -> None:
    """`ken eval` — print the card scoreboard."""
    async def _go() -> None:
        await db.init_db()
        rows = await eval_rows()
        if not rows:
            print("no cards answered yet — tap a few and come back")
            return
        print(f"{'agent':<12}{'card':<10}{'cards':>6}{'yes':>5}{'no':>5}{'rate':>7}{'ignored':>9}")
        for (author, kind), r in sorted(rows.items()):
            answered = r["approved"] + r["rejected"]
            rate = f"{r['approved'] / answered:.0%}" if answered else "-"
            print(f"{author:<12}{kind:<10}{r['cards']:>6}{r['approved']:>5}{r['rejected']:>5}"
                  f"{rate:>7}{r['ignored']:>9}")
    asyncio.run(_go())


async def system_snapshot() -> dict:
    """Read-only counters for `ken status`: the ``stats`` rows and the store counts."""
    async with db.get_db() as c:
        stats = [{"name": r["name"], "count": r["count"], "last_at": r["last_at"]}
                 for r in await (await c.execute("SELECT name, count, last_at FROM stats ORDER BY name")).fetchall()]
        counts = {}
        for key, sql in (("tasks", "SELECT COUNT(*) FROM tasks WHERE status='open'"),
                         ("events", "SELECT COUNT(*) FROM events"),
                         ("structured", "SELECT COUNT(*) FROM structured"),
                         ("inbox", "SELECT COUNT(*) FROM inbox WHERE handled=0"),
                         ("pending", "SELECT COUNT(*) FROM pending")):
            counts[key] = (await (await c.execute(sql)).fetchone())[0]
    return {"stats": stats, "counts": counts}


def _cmd_status() -> None:
    """`ken status` — the engine's vital signs, plain print: heartbeat, per-agent
    ran/spoke, and the memory counts."""
    from .util import real_ms

    async def _go() -> None:
        await db.init_db()
        snap = await system_snapshot()
        now = real_ms()                     # the heartbeat is wall clock (see Engine.tick)
        by = {s["name"]: s for s in snap["stats"]}
        tick = by.get("tick")
        alive = bool(tick and tick.get("last_at") and now - tick["last_at"] < 30_000)
        print("engine   ", "live" if alive else "stale",
              f"· last beat {_ago(tick.get('last_at') if tick else None, now)}")
        print("agents")
        for name, s_ in sorted((n[4:], s_) for n, s_ in by.items() if n.startswith("run:")):
            spoke = by.get(f"spoke:{name}", {}).get("count", 0)
            print(f"  {name:<12} ran {s_['count']:<5} spoke {spoke:<5} {_ago(s_.get('last_at'), now)}")
        c_ = snap["counts"]
        print("memory   ", f"{c_['tasks']} open tasks · {c_['events']} events · "
                          f"{c_['structured']} beliefs · {c_['inbox']} unhandled · {c_['pending']} open cards")
    asyncio.run(_go())


def _cmd_cards() -> None:
    """`ken cards` — the open consent cards, one per line, then how to answer."""
    from .mcp_server import pending_cards

    async def _go() -> None:
        await db.init_db()
        cards = await pending_cards()
        if not cards:
            print("no open cards")
            return
        from .mcp_server import render_card
        for c in cards:
            print(render_card(c))
        print("answer: ken decide <id> yes|no [--edited '<your version, or the batch edits JSON>']")
    asyncio.run(_go())


def _cmd_push(dry_run: bool) -> int:
    """`ken push` — the open cards as one short message, delivered to `push.url` from
    ~/.ken/config.yaml (any endpoint that takes a text body: an ntfy topic, a webhook, a
    Shortcut URL) with `push.headers` if set. Nothing waits → nothing is sent. Meant for a
    09:00 scheduler (launchd, cron); `--dry-run` prints without sending."""
    import httpx

    from .mcp_server import pending_digest

    async def _digest() -> str:
        await db.init_db()
        return await pending_digest()
    text = asyncio.run(_digest())
    push = config.section("push")
    url = str(push.get("url") or "")
    no_url = ("push.url is not set — nothing can be sent. Set it under `push: url:` in "
              f"{config.config_path()} (e.g. an ntfy topic URL)")
    if not text:
        print("no open cards — nothing sent" + ("" if url else f"\n({no_url})"))
        return 0
    print(text)
    if not url:
        print(f"({no_url}; printed only)")
        return 0
    if dry_run:
        print("(dry run)")
        return 0
    raw = push.get("headers")
    headers = {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}   # yaml ints → str
    host = url.split("//", 1)[-1].split("/", 1)[0]               # never log the path: an ntfy topic is a secret
    try:
        r = httpx.post(url, content=text.encode(), timeout=20,
                       headers={"Content-Type": "text/plain; charset=utf-8", **headers})
    except httpx.HTTPError as exc:
        print(f"push failed: {host}: {exc.__class__.__name__}: {exc}")
        return 1
    print(f"sent → {host} ({r.status_code})")
    return 0 if r.is_success else 1


def _cmd_decide(card_id: str, answer: str, edited: str) -> None:
    """`ken decide ID yes|no` — the same tap any client makes, through the engine (it must
    be running: the reply is the gate's own outcome)."""
    from .mcp_server import decide

    async def _go() -> None:
        await db.init_db()
        if not db.engine_is_running():         # fail fast: never queue an answer to apply later, unseen
            print("no engine running — start `ken` first, then decide")
            return
        out = json.loads(await decide(card_id, answer == "yes", edited=edited or ""))
        print(f"{out['outcome']}" + (f" — {out['text']}" if out.get("text") else ""))
    asyncio.run(_go())


def _cmd_activity() -> None:
    """`ken activity` — the last 20 audit rows, newest first."""
    from .util import now_ms

    async def _go() -> None:
        await db.init_db()
        now = now_ms()
        async with db.get_db() as c:
            rows = await (await c.execute(
                "SELECT kind, ref, reason, at FROM audit ORDER BY at DESC LIMIT 20")).fetchall()
        if not rows:
            print("nothing yet — agents run in the background")
        for r in rows:
            print(f"{_ago(r['at'], now):>8}  {r['kind']:<18} {str(r['ref'] or '')[:24]:<24} "
                  f"{str(r['reason'] or '')[:60]}")
    asyncio.run(_go())


def _ago(at, now: int) -> str:
    if not at:
        return "never"
    s = max(0, (now - at) // 1000)
    return f"{s}s ago" if s < 60 else f"{s // 60}m ago" if s < 3600 else f"{s // 3600}h ago"


def _cmd_reload() -> None:
    """`ken reload` — set the flag the running engine consumes on its next tick to
    rebuild its registry. Picks up new/edited drop-ins without restarting the app."""
    async def _go() -> None:
        await db.init_db()
        await store.set_meta(RELOAD_FLAG, "1")
    asyncio.run(_go())
    print("ken: reload requested — the running engine will rebuild on its next tick "
          "(nothing running? the next start discovers fresh anyway).")


_EXAMPLE_URL = "https://github.com/kenning-labs/ken/blob/main/engine/config.example.yaml"
_CONFIG_HEADER = f"# Ken config. More options (tiers, http, push, user): {_EXAMPLE_URL}\n"


def _ask_llm(args) -> dict | None:
    """The llm section for a new config: from the flags, else ONE question on a TTY.
    {} = no model for now. None = no answer possible (not a TTY and no flag), no key given,
    or --claude-code without `claude` on PATH."""
    if args.no_model:
        return {}   # no llm section: no judge, no `tell` parsing; typed MCP writes still work
    if args.claude_code:
        if not shutil.which("claude"):
            print("ken init: `claude` is not on PATH — install Claude Code and run `claude` once to "
                  "log in, then run `ken init --claude-code` again. Nothing written.", file=sys.stderr)
            return None
        return {"provider": "claude-code"}
    if args.api_key:
        return ({"provider": "openai", "api_key": args.api_key}
                | ({"base_url": args.base_url} if args.base_url else {}))
    if not sys.stdin.isatty():
        print("ken init: no terminal to ask in. Pass one of: --claude-code, --api-key KEY "
              "(and --base-url URL for another OpenAI-compatible API), --no-model.", file=sys.stderr)
        return None
    has_claude = bool(shutil.which("claude"))
    default = "1"
    print("Which model should Ken use for its own small calls?")
    print("  1) an OpenAI API key (recommended: also turns on OpenAI Decisions)")
    print("  2) your Claude Code login (no API key)" + ("" if has_claude else "  [claude not found on PATH]"))
    print("  3) another OpenAI-compatible API (base URL + key)")
    print("  4) no model for now (typed writes work; no judge, no `tell` parsing)")
    import getpass
    try:
        choice = ""
        while choice not in ("1", "2", "3", "4"):
            choice = input(f"Choose 1-4 [{default}]: ").strip() or default
        if choice == "4":
            return {}
        if choice == "2":
            return {"provider": "claude-code"}
        base_url = input("Base URL (e.g. https://host/v1): ").strip() if choice == "3" else ""
        key = getpass.getpass("API key (hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not key:
        print("ken init: no key given — nothing written. (No model for now? `ken init --no-model`.)",
              file=sys.stderr)
        return None
    return {"provider": "openai", "api_key": key} | ({"base_url": base_url} if base_url else {})


def _register_claude(args) -> None:
    """Offer `claude mcp add ken`. Never fails the init: on any problem, print the command."""
    claude = shutil.which("claude")
    if args.no_claude or not claude:
        if args.add_to_claude:
            print("ken: `claude` is not on PATH — skipped the Claude Code step")
        return
    sibling = Path(sys.executable).with_name("ken-mcp")            # same venv as this `ken`
    mcp = str(sibling) if sibling.exists() else shutil.which("ken-mcp")
    if not mcp:
        print("ken: ken-mcp not found — install with `pip install 'ken-engine[mcp]'`, then run "
              "`claude mcp add --scope user ken -- ken-mcp`")
        return
    cmd = [claude, "mcp", "add", "--scope", "user", "ken"]
    if os.environ.get("KEN_HOME"):                                   # the MCP must read the same home
        cmd += ["-e", f"KEN_HOME={db.db_path().parent}"]
    cmd += ["--", mcp]
    by_hand = " ".join(["claude", *cmd[1:]])
    if not args.add_to_claude:
        if not sys.stdin.isatty():
            print(f"ken: to use Ken from Claude Code, run: {by_hand}")
            return
        try:
            yes = input("Add Ken to Claude Code (all projects)? [y/N]: ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            yes = False
            print()
        if not yes:
            print(f"ken: skipped. Later, run: {by_hand}")
            return
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"ken: `claude mcp add` failed ({exc.__class__.__name__}). Run by hand: {by_hand}")
        return
    out = (r.stdout + r.stderr).strip()
    if r.returncode == 0:
        print("ken: added to Claude Code as the `ken` MCP server")
    elif "already exists" in out.lower():
        print("ken: Claude Code already has a `ken` MCP server — left unchanged")
    else:
        print(f"ken: `claude mcp add` failed: {out.splitlines()[-1] if out else r.returncode}\n"
              f"     Run by hand: {by_hand}")


def _offer_service(args) -> bool:
    """Offer `ken service install`. True = the engine now runs in the background."""
    if args.no_service:
        return False
    if not args.service:
        if not sys.stdin.isatty():
            print("ken: to run Ken in the background, run: ken service install")
            return False
        try:
            yes = input("Run Ken in the background (starts at login)? [Y/n]: ").strip().lower() in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            yes = False
            print()
        if not yes:
            print("ken: skipped. Later, run: ken service install")
            return False
    return service.main("install") == 0


def _cmd_init(args) -> int:
    """`ken init` — one guided first run: home + database, the model, Claude Code."""
    cfg = config.config_path()                                     # creates the 0700 home
    llm = None
    if cfg.exists():
        print(f"ken: {cfg} already exists — left unchanged")
    else:
        llm = _ask_llm(args)
        if llm is None:
            return 2
    asyncio.run(db.init_db())                                      # == `ken init-db`
    print(f"initialized {db.db_path()}")
    if llm is not None:
        config.write_private(cfg, _CONFIG_HEADER + (yaml.safe_dump({"llm": llm}, sort_keys=False) if llm else ""))
        print(f"ken: wrote {cfg} (owner-only)" + ("" if llm else f" — no model; to add one later, see {_EXAMPLE_URL}"))
        if llm.get("base_url"):     # another host rarely serves the OpenAI default model names
            print(f"ken: set your models under llm.tiers (fast, reasoning); see {_EXAMPLE_URL}")
    _register_claude(args)
    if _offer_service(args):
        print("\nNext: in Claude Code, say: \"set up my world\".")
    else:
        print("\nNext:\n  1. Start the engine: `ken` (keep it running).\n"
              "  2. In Claude Code, say: \"set up my world\".")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ken")
    sub = ap.add_subparsers(dest="cmd")   # not required — no subcommand → the engine

    for name, help_ in (("run", "start the engine tick loop"),
                        ("serve", "run the engine as an always-on service (== run)")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--db", help="database path (default ~/.ken/ken.db)")
        p.add_argument("--interval", type=float, default=1.0, help="seconds per tick")

    setupp = sub.add_parser(
        "init", help="first-time setup: home, database, model, Claude Code, background service",
        description="Create the Ken home (~/.ken or KEN_HOME) and database, write a minimal "
                    "config.yaml with your model choice (never overwrites one), and offer to "
                    "add Ken to Claude Code and to run it in the background. Asks on a terminal; flags make it scriptable.")
    model = setupp.add_mutually_exclusive_group()
    model.add_argument("--claude-code", action="store_true", help="use your Claude Code login (no API key)")
    model.add_argument("--api-key", metavar="KEY", help="use an OpenAI (or compatible) API key")
    model.add_argument("--no-model", action="store_true",
                       help="no model for now: typed writes work; no judge, no `tell` parsing")
    setupp.add_argument("--base-url", metavar="URL", help="with --api-key: another OpenAI-compatible API")
    claude = setupp.add_mutually_exclusive_group()
    claude.add_argument("--add-to-claude", action="store_true", help="run `claude mcp add ken` without asking")
    claude.add_argument("--no-claude", action="store_true", help="skip the Claude Code step")
    bg = setupp.add_mutually_exclusive_group()
    bg.add_argument("--service", action="store_true", help="install the background service without asking")
    bg.add_argument("--no-service", action="store_true", help="skip the background-service step")

    servicep = sub.add_parser("service", help="run the engine in the background, started at login")
    servicep.add_argument("action", choices=["install", "uninstall", "status"])

    initp = sub.add_parser("init-db", help="create the schema and exit")
    initp.add_argument("--db", help="database path (default ~/.ken/ken.db)")

    oncep = sub.add_parser("tick-once", help="run exactly one tick, then exit")
    oncep.add_argument("--db", help="database path (default ~/.ken/ken.db)")

    agentsp = sub.add_parser("agents", help="list discovered agents")
    agentsp.add_argument("--db", help="database path (default ~/.ken/ken.db)")

    agentp = sub.add_parser("agent", help="author tools for a single agent")
    agentp.add_argument("action", choices=["test"], help="what to do")
    agentp.add_argument("name", help="the agent's name or @handle")
    # nargs="*" keeps `--db` working anywhere on the line; for a message that itself
    # starts with a dash, use argparse's `--` escape: `ken agent test c -- --hi there`.
    agentp.add_argument("message", nargs="*",
                        help="the message to send it (prefix with `--` if it starts with '-')")
    agentp.add_argument("--db", help="database path (default ~/.ken/ken.db)")

    cardsp = sub.add_parser("cards", help="the open consent cards")
    cardsp.add_argument("--db", help="database path (default ~/.ken/ken.db)")

    pushp = sub.add_parser("push", help="send the open cards to push.url (a 09:00 scheduler runs this)")
    pushp.add_argument("--dry-run", action="store_true", help="print the message, send nothing")
    pushp.add_argument("--db", help="database path (default ~/.ken/ken.db)")

    decidep = sub.add_parser("decide", help="answer one card: ken decide ID yes|no")
    decidep.add_argument("card_id")
    decidep.add_argument("answer", choices=["yes", "no"])
    decidep.add_argument("--edited", default="", help="your corrected value (or the batch edits JSON) on a yes")
    decidep.add_argument("--db", help="database path (default ~/.ken/ken.db)")

    maintp = sub.add_parser("maintain", help="run the maintainer pass now (done? and still-real? cards)")
    maintp.add_argument("--db", help="database path (default ~/.ken/ken.db)")

    evalp = sub.add_parser("eval", help="the card scoreboard; or `eval reconcile` / `eval context` (the gated evals)")
    evalp.add_argument("suite", nargs="?", choices=["reconcile", "context"],
                       help="reconcile: real reports → typed outcomes (real tiers); context: get_context cases")
    evalp.add_argument("--cases", help="a cases .jsonl (default: evals/<suite>.jsonl in the repo)")
    evalp.add_argument("--db", help="database path (default ~/.ken/ken.db)")

    for name, help_ in (("status", "engine heartbeat, agents"),
                        ("activity", "what your agents did, newest first")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--db", help="database path (default ~/.ken/ken.db)")

    reloadp = sub.add_parser("reload", help="tell the running engine to rebuild")
    reloadp.add_argument("--db", help="database path (default ~/.ken/ken.db)")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    if getattr(args, "db", None):
        db.configure(args.db)

    if args.cmd == "cards":
        _cmd_cards()
        return

    if args.cmd == "decide":
        _cmd_decide(args.card_id, args.answer, args.edited)
        return

    if args.cmd == "push":
        raise SystemExit(_cmd_push(args.dry_run))

    if args.cmd == "init":
        if args.base_url and not args.api_key:
            raise SystemExit("ken init: --base-url needs --api-key")
        raise SystemExit(_cmd_init(args))

    if args.cmd == "service":
        raise SystemExit(service.main(args.action))

    if args.cmd == "init-db":
        asyncio.run(db.init_db())
        print(f"initialized {db.db_path()}")
        return

    if args.cmd == "tick-once":
        async def _once() -> None:
            await db.init_db()
            await build_engine().tick()
        asyncio.run(_once())
        return

    if args.cmd == "agents":
        _cmd_agents()
        return

    if args.cmd == "agent":       # only action is "test" (argparse-constrained)
        _cmd_agent_test(args.name, " ".join(args.message).strip() or "hello")
        return

    if args.cmd == "reload":
        _cmd_reload()
        return

    if args.cmd == "status":
        _cmd_status()
        return

    if args.cmd == "activity":
        _cmd_activity()
        return

    if args.cmd == "eval":
        if args.suite:
            from .evals import main as _eval_main
            raise SystemExit(_eval_main(args.suite, args.cases))
        _cmd_eval()
        return

    if args.cmd == "maintain":
        async def _m() -> None:
            await db.init_db()
            from .maintain import run_once
            d = await run_once()
            print(f"ken: maintainer raised {d['loop_cards']} done? card(s), {d['stale_cards']} still-real? "
                  f"card(s); {d['expired_events']} event(s) expired")
        asyncio.run(_m())
        return

    # "run", its "serve" alias, and the bare `ken`: the long-lived engine loop.
    interval = getattr(args, "interval", 1.0)
    if not config.has_api_key():
        print("ken: no model configured — `tell` reports are not processed (typed writes still work) until you add one "
              "(or set llm.provider: claude-code to use your Claude Code login)")
    print(f"ken: data in {db.db_path().parent}")
    print(f"ken engine: ticking every {interval}s on {db.db_path()}")
    try:
        asyncio.run(run(interval))
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
