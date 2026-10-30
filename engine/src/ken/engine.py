"""Engine — wire the parts and run the loop. One long-lived process.

The loop, minimally: pump the inbox, then wake agents on their schedule; an agent
fetches what it needs through its own tools, returns advice (-> outbox) or a
proposal (-> consent). Interfaces drain the outbox and fill the inbox on their own.

V1 keeps this a plain object with an explicit `tick()`, so it is trivial to
drive from a test. The always-on daemon is a thin wrapper over `tick()`.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

from .agents.base import (
    Advice,
    Agent,
    Config,
    Context,
    OnSchedule,
    Proposal,
    agent_handle,
    agent_name,
)
from .capabilities import ConfigStore, MemoryReader, Proposer, ViewPublisher
from .consent import gate
from .contracts import ActionKind, Evidence, Provenance, Renderable, RenderableKind
from .conversation import bus
from .llm import reset_tool_reporter, set_tool_reporter
from .memory import store
from .schedule import cron_matches
from .util import new_id, now_ms, real_ms

_HEARTBEAT_MS = 15_000            # write the liveness heartbeat at most this often
_SCHED_CATCHUP_MIN = 240          # a scheduled occurrence can fire up to this late (box asleep)
_REMINDER_STALE_MS = 4 * 3_600_000  # a reminder more than 4h late is retired, not fired
# `@handle rest…` — a leading mention addresses one agent by its slug handle. The
# handle may be followed by whitespace, trailing punctuation, or nothing at all
# (a bare `@example` still addresses it), and the rest — possibly empty — is the
# message. Only a LEADING @ counts, so "email@x" / "a @b" are never mentions.
_MENTION = re.compile(r"^\s*@([a-z0-9-]+)[\s,:;.!?]*(.*)$", re.IGNORECASE | re.DOTALL)


def _progress_step(tool: str, args: dict) -> str:
    """A compact single-line label for a live tool call — the tool name plus its most telling
    arg (channel / query / who / text), for the interface's 'thinking' detail. Newlines are
    flattened so a multi-line arg can't break the one-line spinner row."""
    hint = ""
    if isinstance(args, dict):
        for k in ("channel", "query", "name_or_id", "text"):
            v = args.get(k)
            if v is not None and str(v).strip():
                hint = " ".join(str(v).split())[:48]   # collapse any newlines/runs of whitespace
                break
    return f"{tool} {hint}".strip()


def _said_at(action) -> int | None:
    """tell(at=...): when the message was said, ms epoch (validated by the MCP server)."""
    at = (action.meta or {}).get("at")
    return at if isinstance(at, int) and not isinstance(at, bool) else None


def _provenance_of(action) -> Provenance | None:
    """The provenance this action's writes may carry. The CHANNEL is the ceiling (an
    interface stamps it from the credential that authenticated the call: `mcp:user`,
    `mcp:agent:<name>`, `cli:local`…); `meta.provenance` is what the caller asked for. An
    ask above the ceiling is clamped and audited by the caller's outcome line — an agent
    token can never write as the user. None = each capture's own default."""
    if str(action.channel or "").startswith("mcp:agent"):
        return Provenance.WORLD                     # an agent token: world input, whatever it asked (or did not)
    asked = str((action.meta or {}).get("provenance") or "").lower()
    if not asked:
        return None
    if asked == "user":
        return Provenance.USER_INSTRUCTED           # the user's own instruction, relayed by a client
    return Provenance.AGENT_INFERRED                # an agent's own words under the user's ceiling


class Engine:
    def __init__(self) -> None:
        self.agents: list[Agent] = []
        self._by_session: dict[str, Agent] = {}   # session (conversation) -> owning agent
        self._by_name: dict[str, Agent] = {}      # name -> agent (attribution)
        self._by_handle: dict[str, Agent] = {}    # @handle -> agent (Phase-B routing)
        self.assistant: Agent | None = None       # fallback for an unowned session
        self.judge = None                          # the fast-tier judge for typed writes (app sets it)
        self._last_heartbeat = 0                   # throttles the liveness write
        self._last_sched_min = -1                  # schedule is checked once a minute
        self._seeded = False
        self.discovery_skips: list[tuple[str, str]] = []   # drop-ins that failed to load
        self._audited_skips: set = set()   # (src, err) already trailed — dedup across reloads

    def register_agent(self, a: Agent, *, assistant: bool = False) -> None:
        # one session, one owning agent — a silent overwrite would misroute every
        # message on that tab, so refuse a collision loudly at wire-up time. Names
        # collide loudly too (a duplicate would shadow attribution + @routing).
        name = agent_name(a)
        handle = agent_handle(a)
        is_default = assistant or getattr(a, "default_assistant", False)
        if a.session in self._by_session:
            raise ValueError(
                f"session {a.session!r} already owned by "
                f"{agent_name(self._by_session[a.session])}; "
                f"{name} must declare a distinct session")
        if name in self._by_name:
            raise ValueError(f"agent name {name!r} is already registered")
        if handle in self._by_handle:
            raise ValueError(
                f"@{handle} (from {name!r}) collides with "
                f"{agent_name(self._by_handle[handle])!r}")
        # exactly one default: a second one would silently steal every unaddressed
        # message + the reminder attribution, so refuse it loudly.
        if is_default and self.assistant is not None:
            raise ValueError(
                f"{name} sets default_assistant, but {agent_name(self.assistant)} "
                f"is already the default")
        self.agents.append(a)
        self._by_session[a.session] = a
        self._by_name[name] = a
        self._by_handle[handle] = a
        if is_default:
            self.assistant = a

    def adopt_registry(self, other: Engine) -> None:
        """Swap in another engine's freshly discovered registry — the live-reload seam
        (``ken reload``). Discovery is done on a throwaway engine and only its finished
        registry is adopted here, so a discovery that half-fails never leaves THIS
        engine partly wired. Sessions re-seed on the next tick."""
        self.agents = other.agents
        self._by_session = other._by_session
        self._by_name = other._by_name
        self._by_handle = other._by_handle
        self.assistant = other.assistant
        self.discovery_skips = other.discovery_skips
        self._seeded = False

    async def audit_discovery_skips(self) -> None:
        """Record any drop-in agent that failed to load — a bad plugin is skipped at
        boot, never silent: it leaves a ``plugin_skipped`` trail in the activity log.
        Each distinct ``(src, err)`` is trailed once: a `ken reload` that re-discovers
        the same permanently-broken file does not re-write an identical row every time."""
        for src, err in self.discovery_skips:
            if (src, err) in self._audited_skips:
                continue
            self._audited_skips.add((src, err))
            await self._audit("plugin_skipped", src, err)

    async def publish_roster(self) -> None:
        """Write the registered agents (name · role · @handle) to a meta row, so the
        client — a separate process — can show the REAL roster instead of a hardcoded
        list, even before an agent has run."""
        roster = [{"name": agent_name(a), "handle": agent_handle(a),
                   "role": getattr(a, "role", ""), "default": a is self.assistant}
                  for a in self.agents]
        await store.set_meta("roster", json.dumps(roster))

    async def prune_orphan_views(self) -> None:
        """Drop published panels for a session no LIVE agent owns — a removed drop-in leaves its
        `views` row behind, and the palette builds `/<session>` panel commands from that table, so
        a stale `/github` would linger after @github is gone. MAIN (the assistant's /today) is kept.
        Runs with `publish_roster` on startup and every reload, so the palette follows the roster."""
        from . import db as _db
        live = set(self._by_session) | {bus.MAIN}     # _by_session is keyed by each live agent's session
        async with _db.get_db() as c:
            rows = await (await c.execute("SELECT DISTINCT conversation_id FROM views")).fetchall()
            stale = [r["conversation_id"] for r in rows if r["conversation_id"] not in live]
            for s in stale:
                await c.execute("DELETE FROM views WHERE conversation_id=?", (s,))
            if stale:
                await c.commit()

    async def seed_sessions(self) -> None:
        """Make each agent's session exist so an interface can list the tabs even
        before anyone has spoken. Idempotent. A strictly increasing created_at
        keeps the tabs in registration order (the assistant first) instead of
        tying when several seed within the same millisecond."""
        base = now_ms()
        for i, a in enumerate(self.agents):
            await bus.ensure_conversation(a.session, title=agent_name(a),
                                          agent=agent_name(a), created_at=base + i)

    # ── reason ───────────────────────────────────────────────────────────────

    async def _assemble(self, agent: Agent, *, message: str = "",
                        conversation_id: str | None = None, woke_by: str = "",
                        cron: str = "", action_id: str = "") -> Context:
        s = agent.slice
        cid = conversation_id or agent.session
        ctx = Context(conversation_id=cid, now=now_ms(), message=message, woke_by=woke_by,
                      cron=cron)
        ctx.history = await bus.history(cid, author=agent_name(agent))
        # the slice-scoped capabilities: LIVE read (memory), the gate-fronted write
        # (propose), and the panel publish (view). The engine is the ONLY holder of
        # store/gate; the agent reaches them only through these. The reply lands in
        # `cid`; the panel always publishes to the agent's OWN session (its tab).
        ctx.memory = MemoryReader(s)
        ctx.propose = Proposer(conversation_id=cid, author=agent_name(agent), action_id=action_id or None)
        ctx.view = ViewPublisher(agent.session)
        grant = getattr(agent, "config", None)
        if isinstance(grant, str):                        # sugar: `config = "slack"` = read+write it
            grant = Config(read=[grant], write=[grant])
        elif grant is None:                               # no grant → empty; any ctx.config(x) raises ScopeError
            grant = Config()
        elif not isinstance(grant, Config):
            raise TypeError(f"{agent_name(agent)}.config must be a str or Config, "
                            f"not {type(grant).__name__}")
        ctx.config = ConfigStore(read=grant.read, write=grant.write, author=agent_name(agent))
        # an audit trail an agent may leave (e.g. what a brief dropped) — tagged by the agent,
        # so the module never imports db. Best-effort; a logging miss never fails a run.
        who = agent_name(agent)

        async def _agent_audit(kind: str, reason: str) -> None:
            try:
                await self._audit(kind, who, reason)
            except Exception:  # noqa: BLE001, S110 — never fail a run over a log row
                pass
        ctx.audit = _agent_audit
        if s.structured:
            for ns in s.structured:
                ctx.structured += await store.get_structured(ns + ".")
        if s.tasks:
            ctx.tasks = await store.open_tasks()
        if s.events:
            ctx.events = await store.recent_events()
        return ctx

    async def run_agent(self, agent: Agent, *, message: str = "",
                        conversation_id: str | None = None, woke_by: str = "",
                        cron: str = "", reply_to: str = "", task_id: str = "",
                        provenance=None, source: str = "", ref: str = "",
                        said_at: int | None = None) -> str:
        """Run one agent. Handle its advice or proposal. Returns the outcome. The
        reply lands in `conversation_id` when given (a fallback answers in the
        session the user spoke in), otherwise the agent's own session. `reply_to`
        is the inbound action's id, echoed into the advice body so an interface
        that waits for the answer to ITS message can tell it from a nudge."""
        cid = conversation_id or agent.session
        name = agent_name(agent)
        ctx = await self._assemble(agent, message=message, conversation_id=cid, woke_by=woke_by,
                                   cron=cron, action_id=reply_to)
        ctx.task_id, ctx.provenance, ctx.action_id, ctx.source = task_id, provenance, reply_to, source
        ctx.ref, ctx.said_at = ref, said_at
        # Publish live tool-call progress to a per-conversation kv while the agent runs, so an
        # interface can show what it's doing under the "thinking" spinner. ONLY for a user-driven
        # turn (a message) — that's when a spinner is actually waiting; a proactive run (cron/data)
        # on the same MAIN conversation must not paint its steps under the user's wait. The value
        # carries a timestamp so a stale write (engine killed before the clear) reads as not-live.
        steps: list[str] = []
        token = None
        if message:
            async def _report(tool: str, args: dict) -> None:
                steps.append(_progress_step(tool, args))
                await store.set_meta(f"progress:{cid}",
                                     json.dumps({"at": now_ms(), "steps": steps[-6:]}))
            token = set_tool_reporter(_report)
        try:
            out = await agent.run(ctx)
        finally:
            if token is not None:
                reset_tool_reporter(token)
                await store.set_meta(f"progress:{cid}", "")   # clear the live steps when the run ends
        await store.bump(f"run:{name}", now_ms())
        if out is not None:
            await store.bump(f"spoke:{name}", now_ms())   # produced output, not a quiet run
        if out is None:
            return "quiet"
        if isinstance(out, Advice):
            body = {**(getattr(out, "body", None) or {}), "text": out.text}
            if reply_to:
                body["reply_to"] = reply_to
                body["cards"] = await gate.cards_for(reply_to, cid)   # the cards THIS turn raised
            if getattr(out, "error", False):
                body["type"] = "error"   # a loud failure — the interface flags it (▲)
            await bus.append_renderable(Renderable(
                id=new_id(), conversation_id=cid,
                kind=RenderableKind.ADVICE, author=agent_name(agent),
                body=body, at=now_ms()))
            return "advice"
        if isinstance(out, Proposal):
            return await gate.apply(
                out, conversation_id=cid, author=agent_name(agent), action_id=reply_to or None)
        return "quiet"

    # ── respond (the inbound half) ───────────────────────────────────────────

    async def pump_inbox(self) -> None:
        """Drain the inbox — a durable work queue — and acknowledge the batch.

        Semantics are at-least-once: act on each action, then ack the whole batch
        (mark handled). An action is acked even when its dispatch fails, so one
        bad action never jams the queue (the blocked-bot retry storm); the
        failure is audited, not swallowed. A crash between acting and the ack
        replays the action on restart — but the critical effects are idempotent
        (a re-approve resolves to 'no_pending', a re-'done' is a no-op), so only a
        rare duplicate chat reply can slip through, an accepted V1 risk."""
        done: list[int] = []
        for seq, action in await bus.take_unhandled_actions():
            try:
                await self._dispatch_action(action)
            except Exception as exc:  # noqa: BLE001 — one bad action must not jam the rest
                await self._audit("action_error", action.id, repr(exc))
            done.append(seq)
        await bus.mark_handled_many(done)

    async def _dispatch_action(self, action) -> None:
        if action.kind == ActionKind.MESSAGE:
            await self._route_message(action)
        elif action.kind == ActionKind.ACTION_RESPONSE:
            await self._dispatch_token(action)
        elif action.kind == ActionKind.COMMAND:
            await self._dispatch_command(action)
        elif action.kind == ActionKind.TOOL:
            await self._dispatch_tool(action)

    # the typed writes an interface may post without a model turn: one capture each
    _TOOLS = frozenset({"add_task", "complete_task", "discard_task", "reschedule_task", "log_event",
                        "remember", "forget", "hydrate", "schedule_task", "assign_task", "link_tasks"})

    async def _dispatch_tool(self, action) -> None:
        """A typed write from an interface (the MCP tools): `{"tool": name, "args": {…}}` in
        the action text. No model turn — the named capture runs through the judge and the
        gate on kip's behalf, and the reply carries the TRUE outcome plus the ids of the
        cards THIS action raised. Provenance follows the action's channel ceiling."""
        from .agents.assistant import _parse_when
        from .captures import Captures
        cid = action.conversation_id
        try:
            spec = json.loads(action.text or "{}")
            tool, args = str(spec.get("tool", "")), dict(spec.get("args") or {})
        except (ValueError, TypeError, AttributeError):
            tool, args = "", {}
        agent = self.assistant
        if agent is None or tool not in self._TOOLS:
            await self._audit("tool_unknown", action.id, tool or (action.text or "")[:80])
            await self._reply(cid, action.id, f"no such tool: {tool!r}", outcome="invalid")
            return
        source = str((action.meta or {}).get("source") or action.channel)
        # a typed write needs no history, no preload: the minimal context (live memory +
        # the gated write path), not the assistant's full chat assembly
        ctx = Context(conversation_id=cid, now=now_ms(), message=f"{tool} {json.dumps(args, sort_keys=True)}")
        ctx.memory = MemoryReader(agent.slice)
        ctx.propose = Proposer(conversation_id=cid, author=agent_name(agent), action_id=action.id)
        ctx.task_id = str((action.meta or {}).get("task_id") or "")
        ctx.provenance = _provenance_of(action)
        cap = Captures(ctx, judge=self.judge, message=ctx.message,
                       evidence=[Evidence(ref=str((action.meta or {}).get("ref") or action.id), source=source)],
                       provenance=ctx.provenance, parse_when=_parse_when)
        try:
            r = await getattr(cap, tool)(**args)
        except TypeError as exc:                       # wrong arguments: say so, write nothing
            await self._reply(cid, action.id, f"bad arguments for {tool}: {exc}", outcome="invalid")
            return
        except Exception as exc:  # noqa: BLE001 — a failed write is loud, never silent
            await self._audit("tool_error", action.id, f"{tool}: {exc!r}")
            await self._reply(cid, action.id, f"couldn't {tool}: {exc}", outcome="error")
            return
        await store.bump(f"run:{agent_name(agent)}", now_ms())
        await self._reply(cid, action.id, r.text, outcome=r.outcome, tool=tool, via=source)

    async def _reply(self, cid: str, action_id: str, text: str, *, outcome: str,
                     tool: str = "", via: str = "", silent: bool = False) -> None:
        """The structured answer to one inbound action: text for a human, `outcome` and
        `result` for code, and the cards this action raised. Posted as kip. `silent` draws
        nothing in a stream and enters no model history (an empty text does the latter)."""
        body = {"reply_to": action_id, "text": text, "outcome": outcome,
                "cards": await gate.cards_for(action_id, cid),
                "result": [{"tool": tool, "outcome": outcome, "text": text}] if tool else []}
        if via:
            body["via"] = via
        if silent:
            body["silent"] = True
        await bus.append_renderable(Renderable(
            id=new_id(), conversation_id=cid, kind=RenderableKind.ADVICE, author="kip",
            body=body, at=now_ms()))

    async def _dispatch_command(self, action) -> None:
        """V1 has no slash commands the engine handles itself — an interface owns its own
        (`/today`, `/clear`…). Anything that reaches here is audited, never guessed at."""
        await self._audit("command_unhandled", action.id, (action.text or "").strip())

    async def _route_message(self, action) -> None:
        """Route a message to an agent. A leading ``@handle`` addresses that agent
        directly (the mention is stripped, and its reply lands in the CURRENT
        conversation, not its own tab); otherwise it goes to the agent that owns the
        session, falling back to the assistant. A typed reply to a live consent ask
        keeps the ask alive instead. (ARCHITECTURE.md)"""
        # a typed reply to a live consent ask must not be routed as a normal
        # message (that silently drops the consent). Keep the ask alive.
        if action.anchor and await gate.open_ask(action.anchor) is not None:
            await gate.reprompt(action.anchor)
            return

        text = action.text or ""
        agent = None
        m = _MENTION.match(text)
        if m:                                    # `@handle rest…`
            target = self._by_handle.get(m.group(1).lower())
            if target is not None:
                agent, text = target, m.group(2)   # addressed agent + stripped text
            # an unknown @handle is left as ordinary text and routed normally below.

        if agent is None:
            agent = self._by_session.get(action.conversation_id)
        if agent is None:
            # no agent owns this session: the assistant answers, but in the
            # session the user spoke in — never silently in the main tab — and
            # the misroute is audited so it leaves a trail.
            agent = self.assistant
            if agent is None:
                await self._audit("no_route", action.id,
                                  f"no agent owns session {action.conversation_id!r}")
                return
            await self._audit("misroute", action.id,
                              f"session {action.conversation_id!r} unowned; "
                              f"{agent_name(agent)} answered")
        await self.run_agent(agent, message=text,
                             conversation_id=action.conversation_id, reply_to=action.id,
                             task_id=str((action.meta or {}).get("task_id") or ""),
                             provenance=_provenance_of(action),
                             source=str((action.meta or {}).get("source") or action.channel or ""),
                             ref=str((action.meta or {}).get("ref") or ""),
                             said_at=_said_at(action))

    async def _dispatch_token(self, action) -> None:
        """A tapped token calls one internal, validated tool — never an external
        side effect (ARCHITECTURE.md). An approve/reject closes a
        consent ask; done marks a task; an unknown token does nothing but audit."""
        token = action.token or ""
        if token.startswith(("approve:", "reject:")):
            ask_id = token.split(":", 1)[1]
            if str(action.channel or "").startswith("mcp:agent"):
                # consent is the USER's: an agent token can raise a card, never answer one
                await self._audit("consent_forbidden", action.id, f"{action.channel}: {token}")
                await self._reply(action.conversation_id, action.id, "", outcome="forbidden", silent=True)
                return
            details: dict = {}
            stamp = gate._ACTION.set(action.id)      # a card raised by this decision (reasked, restaled) is ITS card
            try:
                out = await gate.resolve(ask_id, approved=token.startswith("approve:"),
                                         conversation_id=action.conversation_id, edited=action.text,
                                         details=details)
            finally:
                gate._ACTION.reset(stamp)
            if out == "act":
                await self._act(details)
            # the true outcome for a client that decided by tool (applied / rejected / restaled /
            # no_pending / act); silent, the gate already posted the human confirmation
            await self._reply(action.conversation_id, action.id, "", outcome=out, silent=True)
        elif token.startswith("done:"):
            task_id = token.split(":", 1)[1]
            if not await store.complete_task(task_id):
                await self._audit("token_noop", action.id, token)
        else:
            await self._audit("unknown_token", action.id, token)

    async def _act(self, details: dict) -> None:
        """Perform an approved `act:` proposal: hand it to the agent named in
        `meta["via"]` (a delegate) or else the proposer, post its outcome line, and audit.
        The agent's failure is caught and shown — it never aborts the tick."""
        proposal = details["proposal"]
        cid = details["conversation_id"]
        name = proposal.meta.get("via") or details["author"]
        agent = self._by_name.get(name)
        if agent is None:
            # nobody can ever perform it: close the ask so it cannot be tapped forever
            if details.get("ask_id"):
                await gate.close_act(details["ask_id"])
            await self._audit("act_failed", proposal.intent, f"no agent named {name!r}")
            await bus.append_renderable(Renderable(
                id=new_id(), conversation_id=cid, kind=RenderableKind.ADVICE, author="kip",
                body={"type": "error", "text": f"Couldn't do that — no agent named {name}."},
                at=now_ms()))
            return
        ctx = await self._assemble(agent, conversation_id=cid)
        try:
            line = await agent.act(ctx, proposal)
        except Exception as exc:  # noqa: BLE001 — an agent's action must not abort the tick
            # the ask stays open (the gate did not close it), so a second Yes retries
            await self._audit("act_failed", proposal.intent, f"{name}: {exc!r}")
            await bus.append_renderable(Renderable(
                id=new_id(), conversation_id=cid, kind=RenderableKind.ADVICE, author=name,
                body={"type": "error", "text": f"Couldn't do that: {exc} — tap Yes to retry."},
                at=now_ms()))
            return
        if details.get("ask_id"):
            await gate.close_act(details["ask_id"])       # done: now the card is closed
        await self._audit("act_done", proposal.intent, f"{name}: {line}")
        await bus.append_renderable(Renderable(
            id=new_id(), conversation_id=cid, kind=RenderableKind.ADVICE, author=name,
            body={"text": str(line or "Done.")}, at=now_ms()))

    # ── the loop ─────────────────────────────────────────────────────────────

    async def tick(self) -> None:
        """One turn: hear the user (pump the inbox), then the once-a-minute time-driven
        work (cron agents + due reminders)."""
        if not self._seeded:
            await self.seed_sessions()
            self._seeded = True
        # heartbeat FIRST (throttled), so the header goes live the moment the loop is
        # running. A truly dead engine is still caught by the process supervisor.
        hb = real_ms()        # liveness stays on the wall clock, even under KEN_NOW
        if hb - self._last_heartbeat >= _HEARTBEAT_MS:
            self._last_heartbeat = hb
            await store.bump("tick", hb)
        await self.pump_inbox()
        now = now_ms()
        # time-driven work: check once a minute — cron agents + due task reminders
        now_min = now // 60_000
        if now_min != self._last_sched_min:
            self._last_sched_min = now_min
            await self._run_scheduled(now)
            await self._run_reminders(now)
            await self._run_maintainer(now)

    async def _run_maintainer(self, now: int) -> None:
        """The daily maintenance pass (stale-task cards); one failure never aborts the tick."""
        try:
            from . import maintain
            await maintain.run_scheduled(now)
        except Exception as exc:  # noqa: BLE001 — audited, never fatal
            await self._audit("maintain_error", "maintainer", repr(exc))

    async def _run_reminders(self, now: int) -> None:
        """Nudge each open task whose scheduled time has arrived, exactly once. The
        nudge is the Assistant's, posted to its tab as an ADVICE renderable whose
        structured body flags it a reminder (`type: reminder`) — so an interface can
        style a timed alarm apart from ordinary chat WITHOUT a fourth renderable kind,
        and the engine never formats the text (no glyph). Post BEFORE marking, so a
        failed post re-fires next tick rather than silently drops.

        A reminder too stale to matter (the box was down for hours) is NOT nudged — a
        wake-up never dumps days of old alarms — but it is never a SILENT drop either:
        it is retired with an audit row, so the miss leaves a trail."""
        cid = self.assistant.session if self.assistant else bus.MAIN
        author = agent_name(self.assistant) if self.assistant else "kip"
        for task in await store.due_reminders(now):
            try:
                if now - task.deadline <= _REMINDER_STALE_MS:   # fresh enough to nudge
                    await bus.append_renderable(Renderable(
                        id=new_id(), conversation_id=cid, kind=RenderableKind.ADVICE,
                        author=author,
                        body={"type": "reminder", "text": task.text,
                              "at": task.deadline}, at=now))
                else:                                                # too stale to nudge
                    late_h = (now - task.deadline) // 3_600_000
                    await self._audit("reminder_stale", task.id,
                                      f"retired without a nudge; {late_h}h late")
                await store.mark_reminded(task.id, now)
            except Exception as exc:  # noqa: BLE001 — one bad reminder must not abort the tick
                await self._audit("reminder_error", task.id, repr(exc))

    async def _run_scheduled(self, now: int) -> None:
        """Fire each agent whose OnSchedule cron had an occurrence in the recent
        window, at most once per occurrence. Bounded catch-up (_SCHED_CATCHUP_MIN) so
        a brief still fires if the box was asleep at the exact minute. Dedup keys on
        the LOCAL occurrence time, so a DST fall-back hour never double-fires. A
        malformed cron is audited and skipped — one bad plugin cron cannot abort the
        tick."""
        now_min = now // 60_000
        for agent in self.agents:
            name = agent_name(agent)
            for trig in agent.triggers:
                if not isinstance(trig, OnSchedule):
                    continue
                stamp = None
                try:                                     # newest occurrence in the window
                    for m in range(now_min, now_min - _SCHED_CATCHUP_MIN, -1):
                        dt = datetime.fromtimestamp(m * 60)  # noqa: DTZ006 — local wall clock, by design
                        if cron_matches(trig.cron, dt):
                            stamp = dt.strftime("%Y%m%d%H%M")
                            break
                except ValueError as exc:
                    await self._audit("bad_cron", name, f"{trig.cron!r}: {exc}")
                    continue
                key = f"sched:{name}:{trig.cron}"
                if stamp is None or stamp == await store.get_meta(key):
                    continue                             # no recent occurrence, or already fired
                await store.set_meta(key, stamp)
                try:
                    # a scheduled wake is proactive too — surface it in the one visible
                    # stream (bus.MAIN), tagged by name, so it isn't lost in a hidden tab.
                    await self.run_agent(agent, conversation_id=bus.MAIN, woke_by="schedule",
                                         cron=trig.cron)
                except Exception as exc:  # noqa: BLE001 — one agent must not abort the tick
                    await self._audit("agent_error", name, repr(exc))

    async def _audit(self, kind: str, ref: str, reason: str) -> None:
        from . import db as _db
        await _db.audit(kind, ref, reason)
