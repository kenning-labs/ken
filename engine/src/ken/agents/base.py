"""Agent base — the public agent API. (AGENTS.md)

The one rule: an agent never writes to memory. It returns Advice, or it emits a
Proposal that re-enters the pipeline judged. An agent declares its triggers and
its slice; the engine assembles the context and hands it over.

Evals live with the agent, not the engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..contracts import Event, Evidence, Provenance, StructuredEntry, Task

# ── what an agent is woken by ──────────────────────────────────────────────

@dataclass
class OnMessage:
    """Woken by a user message anchored to this agent."""


@dataclass
class OnSchedule:
    cron: str                            # e.g. "0 7 * * *"


# ── what an agent may read ─────────────────────────────────────────────────

@dataclass
class Slice:
    """The declared read. Slow-maintained (structured) + fast-fresh (tasks,
    events). The engine assembles exactly this into the
    preloaded snapshot AND uses it to scope the live `ctx.memory` capability.

    Each field grants BOTH a bounded preload and a read scope — do not let the string
    under-report the grant:
      structured  the namespaces preloaded AND readable live via ctx.memory.structured;
                  an ungranted namespace raises. e.g. ["preferences", "facts"].
      tasks       a truthy value (e.g. "open") preloads that bounded task view AND
                  grants the tasks NOUN — so ctx.memory.find_tasks can reach COMPLETED
                  tasks that the bounded open preload never contains. None -> no tasks.
      events      the bounded event view to preload (V1: preload only). e.g. "today".
      panels      other agents' published panels readable live via ctx.memory.panel()."""
    structured: list[str] = field(default_factory=list)   # e.g. ["patterns", "world"]
    tasks: str | None = None                           # e.g. "open" — preload + tasks noun
    events: str | None = None                          # e.g. "today"
    panels: list[str] = field(default_factory=list)       # other agents' published panels readable
                                                          # live via ctx.memory.panel(session)


@dataclass
class Config:
    """Declared config-bucket access: named state buckets (~/.ken/state/<bucket>.yaml)
    an agent may read and/or write through `ctx.config`. Access is a DECLARED grant, so
    one bucket can be shared by many agents (and a connector) and one agent can span many
    buckets. Sugar: setting an agent's `config` to a bare string ("slack") grants read+write
    of that one bucket. WRITE is the escalation surface (it widens what a connector acts on),
    so keep it explicit and lean on the consent gate for agent-inferred changes."""
    read: list[str] = field(default_factory=list)
    write: list[str] = field(default_factory=list)


# ── the assembled context an agent reads (built by the engine from the slice) ─

@dataclass
class Context:
    conversation_id: str
    now: int
    structured: list[StructuredEntry] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)   # recent events, when the slice grants
    message: str = ""            # the user text, when woken by a message
    woke_by: str = ""            # "" = a message · "schedule" = a cron
    cron: str = ""               # which OnSchedule cron fired, when woke_by == "schedule"
    history: list[dict] = field(default_factory=list)   # LLM-ready [{role, content}]
    task_id: str = ""            # set when the message is a REPORT on one task (tell with task_id)
    action_id: str = ""          # the inbound action this turn answers (evidence ref for its captures)
    ref: str = ""                # a caller-given evidence ref (a permalink) that outranks the action id
    source: str = ""             # who sent it: the tell() source tag, or the channel
    said_at: int | None = None   # ms epoch the message was SAID (tell(at=...)); None = now
    provenance: Any = None       # the provenance this turn's writes carry (clamped by the caller's
                                 # token ceiling); None = each capture's default
    # ── injected capabilities (built by the engine from the slice + session) ──
    # The preloaded structured/tasks/events above are the default-prompt SNAPSHOT —
    # the reliability floor. These capabilities are the LIVE + WRITE + PUBLISH paths
    # tools use. Typed loosely (Any) so this agent-facing module never imports the
    # engine's store/gate — the composition layer holds those. See capabilities.py.
    memory: Any = None           # MemoryReader — read-only, live, slice-scoped
    propose: Any = None          # Proposer — the write path, through the gate
    view: Any = None             # ViewPublisher — publish the standing panel
    config: Any = None           # ConfigStore — scoped read/write of state buckets (when granted)
    audit: Any = None            # async (kind, reason) -> None — leave an audit trail, tagged by agent


# ── what an agent returns ──────────────────────────────────────────────────

@dataclass
class Advice:
    """No state change. Rendered to an interface."""
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    error: bool = False   # a loud failure (e.g. the model was unreachable) — nothing
                          # was saved; the interface flags it (▲) and offers a retry
    body: dict = field(default_factory=dict)   # extra structured fields for the interface
                                               # (e.g. a brief's sections); text stays the fallback


@dataclass
class Proposal:
    """A change the agent wants. Re-enters the pipeline, judged. Becomes a
    task, an event, or a structured entry — never a note."""
    intent: str                          # e.g. "structured:patterns.recovery"
    content: str
    at: int | None = None             # ms epoch — a scheduled time, for task/event
    evidence: list[Evidence] = field(default_factory=list)
    provenance: Provenance = Provenance.AGENT_INFERRED
    meta: dict = field(default_factory=dict)   # an `act:` proposal's arguments (e.g. where to send)


class Agent(ABC):
    # An agent declares its IDENTITY (name/role), WHEN it wakes (triggers), WHAT it
    # reads (slice), which tab it owns (session) and which model tier it wants. It
    # builds its tools and prompt inline in run() — see the Assistant for the pattern.
    #
    # `name` is the single identity string: the speaker tag (`✦ <name>`), the roster
    # entry, and the `@name` handle. It defaults to the class name (see `agent_name`);
    # a drop-in sets it explicitly. `role` is the one-line roster description. Set
    # `default_assistant = True` on the one agent that answers un-addressed messages.
    name: str = ""                       # identity; blank -> the class name
    role: str = ""                       # one-line roster description
    default_assistant: bool = False      # the fallback for an unowned/unaddressed message
    triggers: list                       # OnMessage / OnSchedule
    slice: Slice
    session: str = "main"                # the conversation this agent owns (its tab)
    model_tier: str = "fast"             # fast | reasoning; config maps tier -> model
    config: Config | str | None = None # declared config-bucket grant; a bare string = read+write it

    @abstractmethod
    async def run(self, ctx) -> Advice | Proposal | None:
        """Reason over the assembled context. Return advice, or a proposal, or
        nothing. Never write directly."""
        ...

    async def act(self, ctx, proposal: Proposal) -> str:
        """Perform an `act:<verb>` proposal the user APPROVED on its card — the one
        execution Ken does, and only after a tap. The engine calls this on the
        proposing agent (or the agent named in `proposal.meta["via"]`). Return a short
        outcome line for the user. Default: the agent declared no actions."""
        raise NotImplementedError(f"{agent_name(self)} declares no act:{proposal.intent}")


def agent_name(a: Agent) -> str:
    """The identity string for attribution + display — the declared `name`, or the
    class name when a drop-in didn't set one. One resolver, used everywhere so the
    engine never tags by `type().__name__` directly again."""
    return getattr(a, "name", "") or type(a).__name__


def handle_slug(name: str) -> str:
    """A name → an addressable ``@handle`` token: lowercased, non-alnum runs collapsed
    to one '-', ASCII. "Day Organizer" -> "day-organizer". This is the handle rule —
    distinct from ``ken.util.slug`` (underscore + unicode, used for memory keys) — so
    a `@handle` and a config-disable entry resolve an agent by the SAME rule, matching
    the ``[a-z0-9-]+`` the mention parser accepts."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def agent_handle(a: Agent) -> str:
    """The `@handle` for routing/display — a handle_slug of the name. Handles collide
    (see register_agent) so two names that slug the same are refused rather than
    becoming an ambiguous `@handle`."""
    return handle_slug(agent_name(a))
