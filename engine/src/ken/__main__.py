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
import shlex
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


# ---- the `ken init` screens: plain ANSI, no dependency. Color and the spinner only on a real
# terminal (not piped, not NO_COLOR, not TERM=dumb); otherwise the same lines in plain text.

def _color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _color() else text


_CLAY = "38;2;184;90;60"          # the brand clay, #B85A3C
_MARKS = {"ok": ("32", "✓"), "no": ("31", "✗"), "note": ("33", "!"), "skip": ("2", "·")}
_WORDMARK = ("██╗  ██╗███████╗███╗   ██╗", "██║ ██╔╝██╔════╝████╗  ██║", "█████╔╝ █████╗  ██╔██╗ ██║",
             "██╔═██╗ ██╔══╝  ██║╚██╗██║", "██║  ██╗███████╗██║ ╚████║", "╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝")


class _Step:
    """One live step: results collect while it works; `end` prints one line plus any notes."""
    current: "_Step | None" = None

    def __init__(self, title: str):
        import threading
        self.title, self.results, self._stop, self._spin = title, [], threading.Event(), None
        _Step.current = self

    def work(self) -> None:
        """Spin while Ken works (never during a question). No-op without color."""
        import threading
        import time
        if not _color() or self._spin:
            return

        def spin() -> None:
            frames, i = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", 0
            while not self._stop.is_set():
                sys.stdout.write(f"\r  {_c(_CLAY, frames[i % 10])} {_c('1', self.title)} {_c('2', 'working…')}\033[K")
                sys.stdout.flush()
                i += 1
                time.sleep(0.08)
        self._spin = threading.Thread(target=spin, daemon=True)
        self._spin.start()

    def end(self) -> None:
        if self._spin:
            self._stop.set()
            self._spin.join()
            sys.stdout.write("\r\033[K")
        _Step.current = None
        rank = {"no": 0, "ok": 1, "skip": 2, "note": 3}
        main = min(self.results, key=lambda r: rank[r[0]]) if self.results else ("ok", "")
        code, mark = _MARKS[main[0]]
        first, *more = main[1].split("\n")
        print(f"  {_c(code, mark)} {_c('1', f'{self.title:<12}')} {_c('2', first) if main[0] == 'ok' else first}")
        for line in more:
            print(f"      {_c('2', line.strip())}")
        for kind, text in self.results:
            if (kind, text) is main or (kind, text) == main:
                continue
            code, mark = _MARKS[kind]
            first, *rest = text.split("\n")
            print(f"      {_c(code, mark)} {first}" + "".join(f"\n        {line.strip()}" for line in rest))


def _say(kind: str, text: str) -> None:
    """A result: ✓ done, ✗ failed, ! worth knowing, · skipped. Inside a step it waits for the step's line."""
    if _Step.current is not None:
        _Step.current.results.append((kind, text))
        return
    code, mark = _MARKS[kind]
    first, *rest = text.split("\n")
    print(f"  {_c(code, mark)} {first}" + "".join(f"\n    {line.strip()}" for line in rest))


def _tilde(path) -> str:
    p, home = str(path), str(Path.home())
    return "~" + p[len(home):] if p == home or p.startswith(home + os.sep) else p


def _banner() -> None:
    from . import __version__
    print()
    if sys.stdout.isatty() and shutil.get_terminal_size((80, 24)).columns >= 40:
        bar = ("   ", "   ", "███", "███", "███", "   ")
        for row, b in zip(_WORDMARK, bar):
            print(f"  {_c('1', row)} {_c(_CLAY, b)}")
        print(f"\n  the current state of your life, for every AI you use  {_c('2', f'· v{__version__}')}\n")
    else:
        print(f"  {_c('1', 'ken')}{_c(_CLAY, '▮')}  the current state of your life, for every AI you use"
              f"  {_c('2', f'· v{__version__}')}\n")


def _model_label(llm: dict) -> str:
    from urllib.parse import urlparse
    if not llm:
        return "none yet: Ken saves what apps send, but cannot check it"
    if llm.get("provider") == "claude-code":
        return "your Claude Code login"
    if llm.get("base_url"):
        return f"an OpenAI-compatible API ({urlparse(llm['base_url']).hostname or llm['base_url']})"
    return "an OpenAI API key"


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
    print("  Ken uses a model to check what your apps write and to read plain-text notes. Which one?")
    print("    1) an OpenAI API key (recommended)")
    print("    2) your Claude Code login (no API key)" + ("" if has_claude else "  [claude not found on PATH]"))
    print("    3) another OpenAI-compatible API (base URL + key)")
    print("    4) no model for now: Ken saves what your apps send, but cannot check it or read plain-text notes")
    import getpass
    try:
        choice = ""
        while choice not in ("1", "2", "3", "4"):
            choice = input(f"  Choose 1-4 [{default}]: ").strip() or default
        if choice == "4":
            return {}
        if choice == "2":
            return {"provider": "claude-code"}
        base_url = input("  Base URL (e.g. https://host/v1): ").strip() if choice == "3" else ""
        key = getpass.getpass("  API key (hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not key:
        print("ken init: no key given — nothing written. (No model for now? `ken init --no-model`.)",
              file=sys.stderr)
        return None
    return {"provider": "openai", "api_key": key} | ({"base_url": base_url} if base_url else {})


def _claude_entry(claude: str, name: str, mcp: str) -> tuple[str, str, str]:
    """What Claude Code's MCP server `name` is, from `claude mcp get`: (state, label, remove command).
    state: "absent" | "this" (runs this ken-mcp on this home) | "foreign" (known to be something else)
    | "unknown" (the call failed, or a URL we cannot match to this engine). The label shows the
    command or the URL's host only, never the URL's path, user info or headers (tokens live there)."""
    from urllib.parse import urlparse
    remove = f"claude mcp remove {name}"
    try:
        r = subprocess.run([claude, "mcp", "get", name], capture_output=True, text=True, timeout=60,
                           check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return "unknown", "an unreadable server", remove
    if r.returncode != 0:
        no_such = "no mcp server named" in (r.stdout + r.stderr).lower()
        return ("absent" if no_such else "unknown"), "an unreadable server", remove
    fields: dict[str, str] = {}
    for line in r.stdout.splitlines():
        text = line.strip()
        if text.startswith("To remove this server, run:"):
            remove = text.partition(":")[2].strip()
        elif text.startswith("KEN_HOME="):
            fields["KEN_HOME"] = text.partition("=")[2]
        else:
            key, sep, val = text.partition(":")
            if sep and key in ("Command", "URL"):
                fields[key] = val.strip()
    if "URL" in fields:
        host = urlparse(fields["URL"]).hostname or "?"
        if host == "kenninglabs.com" or host.endswith(".kenninglabs.com"):
            return "foreign", f"Ken Cloud ({host})", remove
        return "unknown", f"a server at {host}", remove   # maybe this engine over HTTP: cannot tell
    if "Command" in fields:
        command = fields["Command"]
        path = shutil.which(command) or command
        same_file = os.path.realpath(path) == os.path.realpath(mcp)
        home = os.path.realpath(os.path.expanduser(fields.get("KEN_HOME") or "~/.ken"))
        same_home = home == os.path.realpath(db.db_path().parent)
        if same_file and same_home:
            return "this", command, remove
        if same_file:
            return "foreign", f"this ken-mcp on another data folder ({home})", remove
        return "foreign", command, remove
    return "unknown", "a server of another kind", remove


def _register_claude(args) -> tuple[str | None, str]:
    """Offer `claude mcp add ken`. Never fails the init: on any problem, print the command.
    Returns (the name Claude Code reaches THIS engine under, or None; the command to connect by hand).
    Never removes or replaces an existing server."""
    claude = shutil.which("claude")
    sibling = Path(sys.executable).with_name("ken-mcp")            # same venv as this `ken`
    found = str(sibling) if sibling.exists() else shutil.which("ken-mcp")
    mcp = found or "ken-mcp"

    def add_args(name: str) -> list[str]:
        cmd = ["mcp", "add", "--scope", "user", name]
        if os.environ.get("KEN_HOME"):                               # the MCP must read the same home
            cmd += ["-e", f"KEN_HOME={db.db_path().parent}"]
        return cmd + ["--", mcp]

    def hand(name: str) -> str:
        return shlex.join(["claude", *add_args(name)])

    by_hand = hand("ken")
    if args.no_claude:
        _say("skip", "skipped")
        return None, by_hand
    if not claude:
        _say("skip", "Claude Code not found (`claude` is not on PATH)")
        return None, by_hand
    if not found:
        _say("no", f"ken-mcp not found — install with `pip install 'ken-engine[mcp]'`, then run: {by_hand}")
        return None, by_hand
    if not args.add_to_claude:
        if not sys.stdin.isatty():
            _say("skip", f"not asked (no terminal). To use Ken from Claude Code, run: {by_hand}")
            return None, by_hand
        try:
            yes = input("  Add Ken to Claude Code (all projects)? [y/N]: ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            yes = False
            print()
        if not yes:
            _say("skip", f"skipped. Later, run: {by_hand}")
            return None, by_hand

    if _Step.current:
        _Step.current.work()
    remove_ken = ""
    for name in ("ken", "ken-local"):
        by_hand = hand(name)
        try:
            r = subprocess.run([claude, *add_args(name)], capture_output=True, text=True,
                               timeout=60, check=False, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as exc:
            _say("no", f"`claude mcp add` failed ({exc.__class__.__name__}). Run by hand: {by_hand}")
            return None, by_hand
        out = (r.stdout + r.stderr).strip()
        if r.returncode == 0:
            _say("ok", f"added to Claude Code as the `{name}` MCP server")
            if remove_ken:
                _say("note", "Claude Code now has two Ken servers. To use only this one, turn off `ken`\n"
                     f"(and any claude.ai Ken connector) in /mcp, or remove it: {remove_ken}")
            return name, by_hand
        if "already exists" not in out.lower():
            _say("no", f"`claude mcp add` failed: {out.splitlines()[-1] if out else r.returncode}\n"
                 f"Run by hand: {by_hand}")
            return None, by_hand
        state, label, remove = _claude_entry(claude, name, mcp)
        if state == "this":
            _say("ok", f"Claude Code already reaches this engine as `{name}`")
            return name, by_hand
        if state != "foreign":          # unknown (or gone meanwhile): never add a possible duplicate
            _say("note", f"Claude Code's `{name}` is {label}; ken could not tell whether it is this engine —\n"
                 "left unchanged.")
            if name == "ken":
                _say("note", f"If it is not, add this engine as `ken-local`: {hand('ken-local')}")
                return None, hand("ken-local")
            return None, by_hand
        if name == "ken":
            _say("note", f"Claude Code's `ken` is {label}, not this engine — left unchanged.\n"
                 "Using `ken-local` for this engine.")
            remove_ken = remove
    _say("no", f"Claude Code's `ken-local` is {label}, not this engine — left unchanged.\n"
         f"To connect this engine, free that name ({remove}), then run: {by_hand}")
    return None, f"{remove} && {by_hand}"


def _offer_service(args) -> bool:
    """Offer `ken service install`. True = the engine now runs in the background."""
    import contextlib
    import io
    if args.no_service:
        _say("skip", "skipped")
        return False
    if not args.service:
        if not sys.stdin.isatty():
            _say("skip", "not asked (no terminal). To run Ken in the background, run: ken service install")
            return False
        try:
            yes = input("  Run Ken in the background (starts at login)? [Y/n]: ").strip().lower() in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            yes = False
            print()
        if not yes:
            _say("skip", "skipped. Later, run: ken service install")
            return False
    if _Step.current:
        _Step.current.work()
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        ok = service.main("install") == 0
    if not ok:
        print(said.getvalue(), end="")
        _say("no", "could not install the background service (see above). Start the engine by hand: ken")
        return False
    log = next((ln.split("log:", 1)[1].strip() for ln in said.getvalue().splitlines() if "log:" in ln), "")
    _say("ok", "runs now and at every login" + (f"\nlog: {_tilde(log)}" if log else ""))
    return True


def _cmd_init(args) -> int:
    """`ken init` — one guided first run: the model, home + database, Claude Code, the background."""
    cfg = config.config_path()                                     # creates the 0700 home
    _banner()
    step = _Step("Model")
    llm = None
    if cfg.exists():
        _say("ok", f"{_tilde(cfg)} already exists — left unchanged")
        try:
            have = (yaml.safe_load(cfg.read_text()) or {}).get("llm") or {}
        except (OSError, yaml.YAMLError):
            have = {}
    else:
        try:
            llm = _ask_llm(args)
        finally:
            if llm is None:
                _Step.current = None
        if llm is None:
            return 2
        have = llm
        _say("ok", _model_label(llm))
    step.end()
    step = _Step("Your data")
    step.work()
    asyncio.run(db.init_db())                                      # == `ken init-db`
    if llm is not None:
        config.write_private(cfg, _CONFIG_HEADER + (yaml.safe_dump({"llm": llm}, sort_keys=False) if llm else ""))
        if not llm:
            _say("note", f"no model; to add one later, see {_EXAMPLE_URL}")
        elif llm.get("base_url"):     # another host rarely serves the OpenAI default model names
            _say("note", f"set your models under llm.tiers (fast, reasoning); see {_EXAMPLE_URL}")
    _say("ok", f"initialized {_tilde(db.db_path())}, only you can read it")
    step.end()
    step = _Step("Claude Code")
    name, by_hand = _register_claude(args)
    step.end()
    step = _Step("Background")
    running = _offer_service(args)
    step.end()
    _summary(running, have, name, by_hand)
    return 0


def _summary(running: bool, llm: dict, name: str | None, by_hand: str) -> None:
    """The last screen: what is true now, and what to do next."""
    sibling = Path(sys.executable).with_name("ken-mcp")
    mcp = str(sibling) if sibling.exists() else (shutil.which("ken-mcp") or "ken-mcp")
    on_path = shutil.which("ken-mcp")
    if on_path and os.path.realpath(on_path) == os.path.realpath(mcp):
        mcp = "ken-mcp"                                            # the short name works from any shell
    rows = [("Engine", "live in the background · starts at login" if running else "not running yet"),
            ("Data", f"{_tilde(db.db_path().parent)} (only you can read it)"),
            ("Model", _model_label(llm)),
            ("Apps", f"Claude Code → `{name}`" if name else "not connected yet")]
    steps = [] if running else ["Start the engine: `ken` (keep it running)."]
    if name:
        steps.append('In Claude Code, say: "set up my world".'
                     + ("" if name == "ken" else f" (There Ken is the `{name}` server.)"))
    else:
        steps.append(f"Connect Claude Code: {by_hand}  (other apps: see the README)")
        steps.append('Then, in Claude Code, say: "set up my world".')
    head = (_c("32;1", "✓ Ken is ready.") if running and name
            else _c("33;1", "! Ken is installed. A step is left."))
    print(f"\n  {head}\n")
    for key, val in rows:
        print(f"    {_c('2', f'{key:<7}')} {val}")
    if len(steps) == 1:
        print(f"\n  {_c(_CLAY, '→')} {steps[0]}")
    else:
        print("\n  " + _c("1", "Next:") + "".join(f"\n    {i}. {s}" for i, s in enumerate(steps, 1)))
    print(f"\n  {_c('2', f'Cursor, Codex or any MCP client: add a stdio server that runs {_tilde(mcp)}')}\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ken")
    from . import __version__
    ap.add_argument("--version", action="version", version=f"ken-engine {__version__}")
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
