# AGENTS.md — write an agent for Ken

This file is for a coding agent (Claude Code, etc.) or a person extending Ken. It is
the **real contract**, verified against the code.

It covers agents that run *inside* the engine. An agent can also run outside it, as any MCP
client (Claude Code, Codex, Hermes, a script); that kind needs no contract beyond the MCP
tools.

## The model in four lines

- One **engine** loops over a local **SQLite bus** (`~/.ken/ken.db`). Everything is there.
- **Agents** wake on a message or a cron, fetch what they need through their own **tools**
  (an API, an MCP server, a file), and **advise** — they don't write. Writes go through the
  consent gate as a `Proposal`; the one execution (`act:`) happens only after the user taps Yes.
- Memory is three nouns: **tasks**, **events**, and **beliefs** (people, projects, facts,
  preferences). An agent declares the *slice* of it that it reads.
- You extend it by **dropping one `.py` file** into `~/.ken/agents/`. Ken discovers it on
  start (or `ken reload`).

## Rules that make a drop-in work (read these first)

1. **Absolute imports only** — `from ken.agents.base import Agent`, never `from ..agents`.
   A relative import breaks the moment the file is copied into `~/.ken/`.
2. **One concrete class** per file (a subclass of `Agent`).
3. **Secrets live in `~/.ken/config.yaml`**, read with `ken.config.section("<name>")`; never
   in the agent file.
4. **Never touch the core.** A plugin is one self-contained file. Don't edit `engine/`.
5. **Never crash the loop.** On a bad read an agent returns `None` (or an `Advice(error=True)`).
   Errors are data, not exceptions.

## Add an agent (reason + speak)

An agent declares **what it reads** (`slice`) and **when it wakes** (`triggers`), then
`run(ctx)` returns `Advice` (speak), a `Proposal` (suggest a write), or `None` (stay quiet).

```python
# ~/.ken/agents/streak.py — @streak: cheers your commit streak. Deterministic, no LLM.
from __future__ import annotations
import subprocess
from ken.agents.base import Advice, Agent, OnMessage, OnSchedule, Slice


class Streak(Agent):
    name = "streak"                     # the speaker tag, the roster entry, the @streak handle
    role = "cheers your commit streak"  # one-line roster description
    session = "streak"                  # the conversation this agent owns
    triggers = [
        OnMessage(),                    # @streak how am I doing?
        OnSchedule("0 18 * * *"),       # a 6pm check-in, every day
    ]
    slice = Slice()                     # reads no memory; it fetches what it needs itself

    async def run(self, ctx):
        out = subprocess.run(["git", "log", "--since=midnight", "--oneline"],
                             capture_output=True, text=True).stdout
        n = len(out.splitlines())
        if not n:
            return None                 # nothing to say → silence, not noise
        return Advice(f"🔥 {n} commits today — keep it alive.")
```

**Triggers** (any mix): `OnMessage()` · `OnSchedule("0 18 * * *")` (cron). `ctx.woke_by` is
`""` for a message, `"schedule"` for a cron, and `ctx.cron` names which cron fired — branch on
them to behave differently when proactive vs addressed.

**Slice** — what gets preloaded into `ctx`. Declare only what you read:
`Slice(tasks="open", events="today", structured=["preferences"], panels=["main"])`.

**`ctx`** gives you: `ctx.tasks`, `ctx.events`, `ctx.structured` (as your slice granted),
`ctx.memory` (the live, scoped reader — `open_tasks()`, `find_tasks()`, `structured(ns)`,
`panel(session)`), `ctx.message` (the user text), `ctx.woke_by`, `ctx.cron`, and `ctx.now` (ms).

**`ctx.config` — self-tuning config (opt-in).** Declare a `config` grant and the engine injects
a scoped handle to named **state buckets** (`~/.ken/state/<bucket>.yaml`) — plain yaml, separate
from your secrets in `config.yaml`. A bucket is shared: any agent granted it can read/write,
and a change takes effect live (no reload):

```python
config = "slack"                                   # sugar: read + write the "slack" bucket
# or, spanning buckets:  config = Config(read=["slack", "calendar"], write=["slack"])
...
ch = ctx.config("slack").get("watch", [])          # read (live)
await ctx.config("slack").add("watch", "eng")      # write — atomic + audited
```
An ungranted bucket, or a write to a read-only one, **raises** (like an out-of-slice read).
Writes are atomic and audited (best-effort history + undo). Keep secrets out of buckets; keep
`write` grants lean — agent-inferred changes should still go through the consent gate (a
`Proposal`), not a silent `ctx.config` write.

**Return exactly one of:**
- `Advice(text)` — speak. Set `error=True` for a loud failure (nothing saved, offers retry).
- `Proposal(intent, content, at=None, meta=None)` — suggest a write; it goes through the consent
  gate. `intent` is `"task"`, `"event"`, `"structured:<namespace>.<key>"`, or `"act:<verb>"`.
  `at` is an optional ms-epoch time for a task/event. Content another person wrote (a Slack
  message) carries `provenance=Provenance.WORLD`, so the gate always cards it. **Never write
  memory directly** — this is the only write path.
- `None` — stay quiet.

### Want to act? Propose `act:<verb>` and implement `act()`

An `act:` proposal always shows a card and never touches memory. When the user taps Yes, the
engine calls **your** `async def act(self, ctx, proposal) -> str` with the same proposal
(`proposal.meta` carries your arguments), and posts the string you return. A failure keeps the
card so the user can tap again. This is the one execution Ken does — after a tap, in the
agent's own code (for example, an `act:reply` that posts a reply as the user).

### Want a ranker? Accept `rank` and it's injected

Add a `rank` parameter to `__init__` and, when OpenAI Decisions is on (the default with an
OpenAI key), the engine hands you a callable: `await rank(texts, instructions="how soon…",
rubric=["someday", "this week", "today", "now"])` returns one calibrated score per text
(0 .. len(rubric)-1; the rubric holds 2 to 10 levels), or `None` on failure. When Decisions is off it is `None` — keep a deterministic order as the floor.
Kip (the built-in assistant agent) uses it in its Today panel to sort the untimed backlog into lanes.

### Want the model? Accept `llm` and it's injected

Add an `llm` parameter to `__init__` and the engine hands you a configured client keyed to
your `model_tier` (`"fast"` | `"reasoning"`):

```python
class Advisor(Agent):
    name = "advisor"; role = "..."; session = "advisor"
    triggers = [OnMessage()]
    slice = Slice(tasks="open")
    model_tier = "fast"

    def __init__(self, llm=None):
        self._llm = llm

    async def run(self, ctx):
        reply = (await self._llm.complete(
            "You are a terse advisor. One sentence.",
            [{"role": "user", "content": ctx.message}],
            tools=[],                    # optional list of Tool(...)
        )).strip()
        return Advice(reply) if reply else None
```

A **Tool** the model calls on demand: `Tool(name, description, parameters=<JSON schema>, fn=<sync/async fn>)`,
passed as `tools=[...]` to `complete`. A tool is how an agent reaches the world — an HTTP API,
an MCP server, a local command. Ken ships no connector layer on purpose: the fetch is yours;
the judge, the gate, and the memory are Ken's.

## Publish a panel (optional)

An agent can keep a standing panel with `await ctx.view.publish(view)`. The engine stores the spec;
an interface draws it (the built-in assistant's Today panel is what `ken-mcp`'s `GET /today` and `tasks()` lanes read).
Blocks, in order:

| block | fields |
|---|---|
| `statrow` | `items: [{label, value, unit?, trend?: up\|down\|flat, good?: bool, vs?}]` |
| `note` | `text` (a dim hint); `strong: true` makes it a heading, `accent: true` a call-to-action; `tail?` a dim clause after it |
| `agenda` | `items: [{text, mark?, edge?, meta?, badge?, when?, sub?, dim?}]` — `mark` replaces the caret (`●`), `edge: high\|mid` draws a coloured bar down the item (a card), `meta` is a dim tail (a channel), `badge` an accent tag (`@you`), `when` a dim time, `sub` a dim line under it, `dim` greys the whole row |
| `lanes` | `items: [{label, level: high\|mid\|low\|off, count, names}]` — a board: one lane per tier, a cell per item, then the names in that lane |
| `bars` | `items: [{label, level, count, tail?}]` — one small labelled bar per row (`level: bad` is red) |
| `divider` | — |

Prefer labels and glyphs over raw numbers: a viewer should read a panel without a legend. For
example, rank messages into three tiers (`● reply now · ◐ reply · ○ glance`) instead of
showing the model's 0–3 score.

## Test loop

```bash
ken agents                 # lists discovered agents, and any load error
ken agent test streak "hi" # run one agent once, print its reply
ken reload                 # tell a running engine to rebuild (or just restart ken)
```

If your file doesn't appear: check `ken agents` for the load error (usually a relative
import). Fix, `ken reload`, retry.

## For a coding agent, in one paragraph

To add an agent to Ken: write **one** `.py` file with **absolute** `from ken...` imports,
defining **one** class that subclasses `Agent`. It sets `name`, `triggers` (`OnMessage()` /
`OnSchedule(cron)`), `slice`, and implements `async def run(self, ctx)` returning
`Advice(text)`, a `Proposal`, or `None`; it fetches its own data through `Tool`s (an API, an
MCP server); add an `llm` param to `__init__` for a model, `rank` for a Decisions ranker, and
`act()` to perform an approved `act:` proposal. Put it in `~/.ken/agents/`. Verify with
`ken agents`. Do not edit anything under `engine/`.
