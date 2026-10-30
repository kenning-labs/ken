"""Ken core contracts — the fixed language every module speaks.

These types are the schema-bound heart of the engine. Everything else is
replaceable or extensible around them. Change these and everything breaks.

Locked by: AGENTS.md, AGENTS.md, ARCHITECTURE.md, ARCHITECTURE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ───────────────────────────── the nouns ─────────────────────────────────
# The closed set of storable things: tasks, events, beliefs. No custom tables.

class Provenance(str, Enum):
    """Who authored a write. Sets the consent toll. (ARCHITECTURE.md)"""
    USER_STATED = "user_stated"          # you said it -> fast-track
    USER_INSTRUCTED = "user_instructed"  # a button/command -> deterministic
    AGENT_INFERRED = "agent_inferred"    # a guess -> gated by target stakes
    WORLD = "world"                      # someone else's words (a message) -> judged / carded


@dataclass
class Task:
    """A thing to do. Actionable. Resolved, not maintained."""
    id: str
    text: str
    status: str = "open"                 # open | done | dropped
    deadline: int | None = None       # by when (ms epoch); only a deadline can be overdue
    slot_start: int | None = None     # when the user works on it (ms epoch); None = no slot
    slot_minutes: int | None = None   # how long that slot lasts (> 0 with slot_start)
    slot_by: str | None = None        # who set the slot: 'user' (by hand: an agent leaves it alone) | 'agent'
    provenance: Provenance = Provenance.WORLD
    evidence: list[Evidence] = field(default_factory=list)   # why it exists
    created_at: int = 0
    completed_at: int | None = None   # when it was closed, done or dropped (ms epoch); None while open
    drop_reason: str | None = None    # why it was discarded; only with status 'dropped'
    project: str | None = None        # the structured kind of its project (projects.<slug>); None = none
    owner: str | None = None          # who does it: None = the user, else people.<slug> or a name as given
    changed_at: int | None = None     # when its state last changed (ms epoch); a write with an older `at` is stale


@dataclass
class Event:
    """A thing that happened. Awareness only. Immutable history."""
    id: str
    summary: str
    at: int
    provenance: Provenance = Provenance.WORLD
    evidence: list[Evidence] = field(default_factory=list)   # why it was recorded
    until: int | None = None           # ms epoch when it stops mattering; past = hidden from reads, never deleted


@dataclass
class Evidence:
    """One item behind a belief, carrying its own source. (AGENTS.md)"""
    ref: str                             # id of a capture / a message / a task
    source: str


@dataclass
class Change:
    """One versioned entry in a structured entry's changelog."""
    at: int
    by: Provenance
    note: str


# ── links between records: a closed vocabulary with a fixed direction ────────
# A link is stored ONE way, so a reader never guesses: "<src> is <phrase> <dst>".
#   member   person  → project            robert is a member of recipes
#   owner    person  → project | asset    robert is the owner of the ledger
#   part_of  project | asset → project    the ledger is part of recipes
#   client   person | project → project   acme is a client of recipes
# `asset` is accepted as input (a project naming its asset) and stored as part_of the other way.
RELS = ("member", "owner", "asset", "part_of", "client")
REL_PHRASE = {"member": "a member of", "owner": "the owner of", "part_of": "part of", "client": "a client of"}
_REL_ENDS = {"member": (("people",), ("projects",)),
             "owner": (("people",), ("projects", "assets")),
             "part_of": (("projects", "assets"), ("projects",)),
             "client": (("people", "projects"), ("projects",))}
NOUN = {"people": "person", "projects": "project", "preferences": "preference", "facts": "fact",
        "assets": "asset", "patterns": "note"}


def orient(a: str, b: str, rel: str) -> tuple[str, str, str] | None:
    """The stored (src, dst, rel) for a link between records `a` and `b` (structured ids),
    whichever way the caller named them — or None when the pair does not fit the rel."""
    if rel == "asset":
        rel = "part_of"
    ends = _REL_ENDS.get(rel)
    if ends is None:
        return None
    ns_a, ns_b = a.split(".", 1)[0], b.split(".", 1)[0]
    if ns_a in ends[0] and ns_b in ends[1]:
        return a, b, rel
    if ns_b in ends[0] and ns_a in ends[1]:
        return b, a, rel
    return None


def link_sentence(src_title: str, rel: str, dst_title: str) -> str:
    return f"{src_title} is {REL_PHRASE.get(rel, rel)} {dst_title}"


@dataclass
class StructuredEntry:
    """A curated, maintained belief in the model of you. Evidence-backed and
    versioned. One generic store, open namespaced `kind`. (AGENTS.md)"""
    id: str
    kind: str                            # e.g. "patterns.recovery", "goals.race"
    subject: str
    content: str
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.5
    provenance: Provenance = Provenance.AGENT_INFERRED
    changelog: list[Change] = field(default_factory=list)
    updated_at: int = 0
    url: str | None = None            # an asset's link (a sheet, a repo, a doc); None for everything else


# ───────────────────────────── the interface seam ─────────────────────────
# Two queues of typed things on an interface-agnostic Conversation. The engine
# never formats. An interface formats for its medium. (ARCHITECTURE.md)

class RenderableKind(str, Enum):
    ADVICE = "advice"          # an agent's authored text (a reminder is an advice whose
                               # body carries type=reminder — the flavor is in the body,
                               # not a distinct kind, so the interface styles it from there)
    STATE_VIEW = "state_view"  # your state, rendered verbatim
    ASK = "ask"                # a consent approval request


@dataclass
class ActionChoice:
    """A pre-bound choice on a renderable. `token` maps to an internal
    validated tool — never an external side effect."""
    label: str
    token: str


@dataclass
class Renderable:
    """One item to show. Typed and medium-agnostic — never a formatted string.
    Re-send with the same `id` to update it (latest wins)."""
    id: str
    conversation_id: str
    kind: RenderableKind
    author: str                          # the agent that spoke; shown as a tag
    body: dict                           # kind-specific: text, card, question...
    actions: list[ActionChoice] = field(default_factory=list)
    deliver_by: int | None = None
    at: int = 0


class ActionKind(str, Enum):
    MESSAGE = "message"                  # free text (or transcribed voice)
    COMMAND = "command"                  # a slash-command
    ACTION_RESPONSE = "action_response"  # a tapped choice / answered ask
    TOOL = "tool"                        # a typed write from an interface: text = {"tool", "args"}


@dataclass
class Action:
    """One user action. The interface normalizes local input into this."""
    id: str
    conversation_id: str
    kind: ActionKind
    channel: str                         # opaque handle "{interface}:{address}"
    text: str = ""                       # message/command text; on an action_response an OPTIONAL
                                         # edited value — set only on an approve, means confirm-and-correct
    token: str = ""                      # for action_response
    anchor: str | None = None         # the renderable id this replies to
    at: int = 0
    meta: dict = field(default_factory=dict)   # optional: task_id (a report), provenance, source
