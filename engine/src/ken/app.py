"""Composition root and runner — the one long-lived process. (ARCHITECTURE.md)

`build_engine` wires the running app: it registers the connectors and agents the
deployment ships. `run` drives `tick()` on an interval — poll, pump the inbox,
wake agents — forever. An interface (the CLI) is a *separate* process on the same
SQLite file; it never imports this.
"""
from __future__ import annotations

import asyncio
import inspect
import logging

from . import config, db, registry
from .agents.base import handle_slug
from .consent.judge import DecisionsJudge, FallbackJudge, LlmJudge
from .decisions import build_decisions
from .engine import Engine
from .llm import build_llm, resolve_model
from .memory import store
from .rank import DecisionsRanker
from .util import real_ms


def _disabled_handles() -> set:
    """The slugged agent handles the user turned off via ``agents.disabled`` in
    config (a bare string or a list). Uses the same ``handle_slug`` rule as the
    ``@handle``, so a disable entry resolves an agent the same way an address does."""
    raw = config.section("agents").get("disabled") or []
    if isinstance(raw, str):
        raw = [raw]
    return {handle_slug(x) for x in raw if isinstance(x, str)}


def discover_into(eng: Engine) -> Engine:
    """(Re)populate an engine's registry by DISCOVERY: every `Agent` subclass in the
    built-in package and in ``~/.ken/agents`` is found, built, and registered — no
    hand-editing here. Each agent is handed only the deps its ``__init__`` declares (an
    Advice-only drop-in takes none; the Assistant takes llm + judge). A broken drop-in
    is skipped, never fatal.

    Shared by ``build_engine`` (a fresh engine at startup) and ``ken reload`` (a
    rebuild onto the live engine). Sets ``eng.discovery_skips``."""
    llm_cfg = config.section("llm")

    # the judge always runs on the configured FAST tier, independent of the agent's own
    # tier. With no key its LLM is None, so no judge is injected and the gate applies
    # directly — a self-hoster with no key still runs.
    judge_llm = build_llm(llm_cfg, model=resolve_model(llm_cfg, tier="fast", agent="Judge"))
    judge = LlmJudge(judge_llm) if judge_llm is not None else None
    # OpenAI Decisions: on by default with an OpenAI key (`decisions.enabled: false` turns it
    # off; any other provider leaves it off). When on it is
    # the PRIMARY judge and the ranker; the LLM judge answers only when Decisions fails.
    decisions = build_decisions(config.section("decisions"), llm_cfg)
    rank = DecisionsRanker(decisions) if decisions is not None else None
    if decisions is not None:
        dj = DecisionsJudge(decisions)
        judge = FallbackJudge(dj, judge) if judge is not None else dj
    eng.judge = judge                          # typed writes (the TOOL inbox path) are judged too
    disabled = _disabled_handles()

    def build(cls):
        # a config-disabled built-in is turned OFF here (None → skipped, not a failure).
        # the default assistant is exempt — disabling it would leave sessions with no
        # fallback — so a disable of Kip is ignored and noted, never silently obeyed.
        name = getattr(cls, "name", "") or cls.__name__
        if handle_slug(name) in disabled:
            if getattr(cls, "default_assistant", False):
                _log.warning("ignoring agents.disabled %r — the default assistant "
                             "cannot be disabled", name)
            else:
                return None
        params = inspect.signature(cls.__init__).parameters
        kwargs = {}
        if "llm" in params:
            model = resolve_model(llm_cfg, tier=cls.model_tier, agent=cls.__name__)
            effort = llm_cfg.get("reasoning_effort") if cls.model_tier == "reasoning" else None
            kwargs["llm"] = build_llm(llm_cfg, model=model, effort=effort)
        if "judge" in params:
            kwargs["judge"] = judge
        if "rank" in params:                  # a Decisions ranker for panels; None when off
            kwargs["rank"] = rank
        return cls(**kwargs)

    base_dir = config.config_path().parent                    # ~/.ken
    # the drop-in root always exists, so "drop a .py in here" is true on a fresh
    # install — no mkdir for the user before their first agent.
    (base_dir / "agents").mkdir(parents=True, exist_ok=True)
    eng.discovery_skips = registry.discover_agents(
        eng, build=build, extra_dirs=[base_dir / "agents"])
    return eng


def build_engine() -> Engine:
    """A fresh engine with its registry fully discovered. (See ``discover_into``.)"""
    return discover_into(Engine())


_log = logging.getLogger("ken.engine")

_MAX_BACKOFF = 60.0   # cap the sleep between retries on a persistent failure

RELOAD_FLAG = "reload_requested"   # kv key: `ken reload` sets it, the loop consumes it
_RELOAD_POLL_MS = 3000   # how often the loop checks the flag — rare + manual, so not per-tick


async def _announce_registry(eng) -> None:
    """Publish the registry to the DB + audit any load-skips — what a client reads. Run
    at startup and again after a reload, so the roster a client shows is never stale."""
    await eng.audit_discovery_skips()   # a bad drop-in leaves a trail, never silent
    await eng.publish_roster()          # a client reads the REAL registered agents
    await eng.prune_orphan_views()      # drop panels for agents that are no longer registered


async def _reload(eng) -> None:
    """Rebuild the registry onto the LIVE engine — pick up new/edited drop-ins without
    a restart. Discovery runs on a throwaway engine first; only a clean result is
    adopted, so a broken edit leaves the running companion exactly as it was."""
    await store.set_meta(RELOAD_FLAG, "")   # consume first, so a failure can't loop
    try:
        fresh = build_engine()
    except Exception as exc:  # noqa: BLE001 — a bad edit must not take the daemon down
        _log.exception("reload: discovery failed; keeping the current registry")
        await db.audit("reload_failed", None, repr(exc))
        return
    eng.adopt_registry(fresh)               # next tick re-seeds sessions (_seeded reset)
    await _announce_registry(eng)
    await db.audit("reloaded", None, f"{len(eng.agents)} agents")
    _log.info("reloaded: %d agents", len(eng.agents))


async def tick_guarded(eng, *, audit: bool = True) -> bool:
    """Run one tick, but never let a single failure kill the daemon. Returns True on
    success, False on a caught failure. A transient error (e.g. 'database is locked'
    when the CLI holds the write lock) is always logged; it is audited only when
    `audit` is set — the caller passes False after the first failure of a streak, so
    a persistent fault does not flood the audit table with a row every second. A
    CancelledError (Ctrl-C / shutdown) propagates untouched."""
    try:
        await eng.tick()
        return True
    except asyncio.CancelledError:
        raise                                       # never swallow a shutdown
    except Exception as exc:  # noqa: BLE001 — one bad tick must not stop the whole daemon
        _log.exception("tick failed; continuing")
        if audit:
            try:
                await db.audit("tick_error", None, repr(exc))
            except Exception:  # noqa: BLE001 — even the audit write can hit the same lock
                _log.exception("could not audit the tick failure")
        return False


def _backoff(fails: int, interval: float) -> float:
    """Sleep before the next tick: the normal interval while healthy, then exponential
    growth over a streak of failures, capped — so a persistent fault stops busy-looping
    at ~1 Hz. A single failure (fails=1) still waits just the normal interval."""
    if fails <= 0:
        return interval
    return min(interval * (2 ** (fails - 1)), _MAX_BACKOFF)


async def run(interval: float = 1.0) -> None:
    """Create the schema if needed, then tick forever. Ctrl-C stops it.

    Exactly one engine may tick a given ~/.ken — two would double-drain the inbox. We
    take the singleton lock first and exit quietly if another engine already holds it
    (e.g. a launchd service, when `ken` is also run by hand). The fd is held for the
    whole process; the OS frees it on exit."""
    lock = db.acquire_engine_lock()
    if lock is None:
        print("ken engine: another engine already owns this ~/.ken — exiting.", flush=True)
        return
    await db.init_db()
    eng = build_engine()
    await store.set_meta(RELOAD_FLAG, "")   # drop any request left over from before boot
    await _announce_registry(eng)
    fails = 0
    last_reload_check = 0
    while True:
        # `ken reload` is rare + manual, so poll its flag every few seconds, not every
        # tick. GUARD it like a tick: the CLI that set the flag may still hold the write
        # lock, so a 'database is locked' from this read or the reload's writes must be
        # caught here — otherwise it would escape the loop and kill the always-on daemon.
        now = real_ms()      # a poll throttle: wall clock, even under KEN_NOW
        if now - last_reload_check >= _RELOAD_POLL_MS:
            last_reload_check = now
            try:
                if await store.get_meta(RELOAD_FLAG):
                    await _reload(eng)      # `ken reload` requested a rebuild
            except asyncio.CancelledError:
                raise                        # never swallow a shutdown
            except Exception:  # noqa: BLE001 — a reload fault must not stop the daemon
                _log.exception("reload check failed; continuing")
        # audit only the first failure of a streak; back off so a permanent fault does
        # not busy-loop and flood the audit table (~86k rows/day at 1 Hz).
        ok = await tick_guarded(eng, audit=(fails == 0))
        fails = 0 if ok else fails + 1
        await asyncio.sleep(_backoff(fails, interval))
