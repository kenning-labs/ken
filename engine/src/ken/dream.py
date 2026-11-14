"""The Dreamer — Ken's overnight agent. Once a night it reads everything Ken knows with the
read tools and a reasoning model, then:

  ask    something only the user can decide (a conflict, stuck work, a task the facts made
         pointless) -> an Inbox item
  look   a pattern worth knowing, no decision needed (a slot that never holds) -> an Inbox item
  tidy   an additive fix to a record (a fact a note lacks, a missing link) -> a confirm card
         through the gate: proposed, never applied, until the user trusts it

Asks and looks together are capped at CAP a night, tidy cards at TIDY_CAP; a quiet night says nothing. Nothing the
model writes is taken on its word: an item that cites an id Ken does not hold is dropped, and
a tidy fact must rest on an event or a chat message at the time it names (a task's status is
never proof). Closed items and their outcomes go back into the next night's prompt, so the
same question is never asked twice. Off unless `dreamer.enabled`; runs on the first tick after
`dreamer.hour` (default 03:00 local), in the background, so the engine keeps answering.
`ken dream` runs it now.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta

from . import config
from . import db as _db
from .agents.base import Context, Slice
from .capabilities import MemoryReader, Proposer
from .captures import Captures
from .contracts import Evidence, Provenance
from .conversation import bus
from .llm import build_llm, resolve_model
from .memory import store
from .tools.base import Tool
from .util import new_id, norm, now_ms, slug

CAP = 6                  # asks + looks a night, at most
TIDY_CAP = 3             # tidy cards a night: each one is a tap while tidy is proposed, not applied
EXPIRE_DAYS = 7          # an item nobody answered closes itself after a week
HISTORY_DAYS = 14        # how far back the prompt shows past items and their outcomes
HOPS = 16                # tool rounds a night may take
TOOL_CAP = 60_000        # chars of one tool answer the model sees
_DAY = 86_400_000
_ID = re.compile(r"\b(?=[0-9]*[a-f])[0-9a-f]{8,32}\b")   # needs a letter: a Slack ts or a date is not an id
_KINDS = {"person": "people", "project": "projects", "asset": "assets"}
_running: asyncio.Task | None = None

PROMPT = """You are the Dreamer, Ken's overnight agent. Ken is the user's context engine: their people,
projects, things (assets), tasks, events and preferences, shared by every AI they use. Tonight you
read everything Ken knows and do three jobs.

1. Tidy: additive, clearly right fixes to a person's, project's or asset's record.
   - Add a fact the note lacks, when an EVENT or a CHAT MESSAGE states it. A task's status is never
     proof of a fact (a closed task does not tell you how it went).
   - Add a missing link (a person is a member of a project, an asset is part of a project) when the
     tasks or notes show it.
   Never rewrite or remove the user's words. Each tidy becomes a card the user confirms.
2. Ask (needs the user): something only the user can decide. Conflicts (two records that look like
   one, two notes that disagree, a note that is out of date), stuck work (many late tasks), a task the
   facts already answered or made pointless, people the user waits on for a long time.
3. Worth a look (no decision): patterns across days: tasks missed again and again, a time of day
   that never works, things nobody used for a long time.

How to work:
- Start with world, tasks (open, then status=done with since = two weeks ago), events (the last 30
  days, answers=true to see what the user already decided on cards), chat, and pending. Use who_is,
  find and get_context to check a hunch before you report it.
- Report a finding only if the user would act on it or be glad to know it. A quiet night is a good night.
- Every claim cites its evidence: full task ids, event times, record names, short quotes. Copy ids
  exactly from the tool answers. Never guess numbers: count them.
- Never ask again about something in "Already raised" below, or something the user answered on a card.
- At most {cap} items across asks and looks together, the most important first. Fewer is fine.
- At most {tidy_cap} tidy items, the most useful first.
- Write like a calm, sharp colleague: plain words, short sentences, no hype. Address the user as "you".
- Each ask has 2 or 3 quick answers you could carry out, the first one your recommendation.

Reply with ONLY one JSON object:
{{"asks":[{{"title":"…","preview":"one line","body":["paragraph"],"evidence":["…"],"quick":["…"],"why_now":"…"}}],
 "looks":[{{"title":"…","preview":"…","body":["…"],"evidence":["…"],"quick":["…"]}}],
 "tidy":[{{"kind":"person|project|asset","name":"the record's exact name","add":"the sentence(s) to append, or empty",
          "link":{{"kind":"person|project|asset","name":"…","rel":"member|owner|asset|part_of|client"}} or null,
          "why":"…","source":{{"event":"<the event's time, as the tool gave it>"}} or {{"chat":"<the message's time>"}}}}],
 "skipped":["what you noticed but did not report, and why, one line each"]}}"""


def _cfg() -> dict:
    return config.section("dreamer")


def _llm():
    """The Dreamer's model: `dreamer.llm` keys over the main `llm` section, so it can run on another
    host (an OpenAI-compatible one: base_url, api_key, api_style: chat) with its own `model` and
    `reasoning_effort`. Without `model`, the reasoning tier."""
    cfg = {**config.section("llm"), **(_cfg().get("llm") or {})}
    model = cfg.get("model") or resolve_model(cfg, tier="reasoning", agent="Dreamer")
    return build_llm(cfg, model=model, effort=cfg.get("reasoning_effort"))


async def _chat(limit: int = 60) -> str:
    """What was said to Ken lately, by any client or agent, oldest first: the source tag is in the text."""
    limit = max(1, min(int(limit or 60), 200))
    async with _db.get_db() as c:
        cur = await c.execute("SELECT at, text FROM inbox WHERE conversation_id=? AND kind='message' "
                              "ORDER BY seq DESC LIMIT ?", (bus.MAIN, limit))
        rows = list(reversed(await cur.fetchall()))
    return json.dumps({"messages": [{"at": _iso(r["at"]), "text": r["text"]} for r in rows]})


def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%dT%H:%M") if ms else None  # noqa: DTZ006


def _tools() -> list[Tool]:
    from . import mcp_server as m  # the same reads every client gets; no model, no writes

    def clip(fn):
        async def run(**kw):
            out = await fn(**kw)
            return out if len(out) <= TOOL_CAP else out[:TOOL_CAP] + ' …(cut; narrow the read)'
        return run

    async def pending():
        return json.dumps(await m.pending_cards())

    s = lambda **p: {"type": "object", "properties": {k: {"type": v} for k, v in p.items()}}
    return [
        Tool("world", "Every project, person and asset with notes, links and open tasks; the preferences.", s(), clip(m.world)),
        Tool("tasks", "Tasks. status: open (default) | done | dropped | all. since: a local ISO date for closed "
             "tasks. filter: a lane or words.", s(filter="string", status="string", since="string"), clip(m.tasks)),
        Tool("events", "What happened in a window, newest first. since/until: local ISO dates. answers=true adds "
             "the user's answers to cards.", s(since="string", until="string", answers="boolean"), clip(m.events)),
        Tool("who_is", "One person, project or asset in full: note, evidence, links, mentions.", s(name="string"), clip(m.who_is)),
        Tool("find", "Search tasks, events and records by words.", s(query="string"), clip(m.find)),
        Tool("get_context", "Everything that matters for one task, by its id.", s(task_id="string"), clip(m.get_context)),
        Tool("chat", "What the user and their agents said to Ken lately, oldest first.", s(limit="integer"), clip(_chat)),
        Tool("pending", "The cards waiting for the user's decision.", s(), clip(pending)),
    ]


async def _history(now: int) -> list[dict]:
    async with _db.get_db() as c:
        cur = await c.execute("SELECT night, kind, body, status, outcome FROM dream_items WHERE created_at>=? "
                              "ORDER BY created_at", (now - HISTORY_DAYS * _DAY,))
        rows = await cur.fetchall()
    return [{"night": r["night"], "kind": r["kind"], "title": json.loads(r["body"]).get("title"),
             "status": r["status"], **({"outcome": r["outcome"]} if r["outcome"] else {})} for r in rows]


def _parse(text: str) -> dict:
    try:
        night = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        return {}
    return night if isinstance(night, dict) else {}


async def _known_ids() -> set[str]:
    tasks = await store.open_tasks() + await store.tasks_by_status(("done", "dropped"), limit=100_000)
    async with _db.get_db() as c:
        cards = [r["ask_id"] for r in await (await c.execute("SELECT ask_id FROM pending")).fetchall()]
    return {t.id for t in tasks} | set(cards)


def _bad_ids(item, known: set[str]) -> list[str]:
    """Every id-like token the item cites that is not the prefix of an id Ken holds."""
    return sorted(c for c in set(_ID.findall(json.dumps(item))) if not any(k.startswith(c) for k in known))


async def _source_ok(src) -> bool:
    """A tidy fact's source must exist: an event, or a chat message, at the minute it names."""
    if not isinstance(src, dict):
        return False
    for key in ("event", "chat"):
        when = str(src.get(key) or "").strip()
        if not when:
            continue
        try:
            at = int(datetime.fromisoformat(when[:16]).timestamp() * 1000)
        except ValueError:
            return False
        lo, hi = at, at + 60_000
        if key == "event":
            return any(lo <= e.at < hi for e in await store.events_between(lo - 1, hi))
        async with _db.get_db() as c:
            cur = await c.execute("SELECT 1 FROM inbox WHERE conversation_id=? AND kind='message' AND at>=? AND at<?",
                                  (bus.MAIN, lo, hi))
            return await cur.fetchone() is not None
    return False


async def _tidy(group: list[dict], night: str) -> str:
    """Every tidy item for ONE record -> one confirm card (or why not): a second card on the same
    record would replace the first. The note is the old one with each checked `add` appended:
    the model's words never replace the user's."""
    t = group[0]
    ns = _KINDS.get(str(t.get("kind") or "").lower())
    name = str(t.get("name") or "").strip()
    if not ns or not name:
        return "no record named"
    had = next((e for e in await store.get_structured(ns + ".") if e.kind == f"{ns}.{slug(name)}"), None)
    if had is None:
        return f"no {t.get('kind')} {name!r}"
    adds, links, why, notes = [], [], [], []
    for t in group:
        add = " ".join(str(t.get("add") or "").split())
        if add and norm(add) in norm(" ".join([had.content, *adds])):
            add = ""                                     # the note says it already
        if add and not await _source_ok(t.get("source")):
            notes.append("a fact with no event or chat message behind it was left out")
            add = ""
        if add:
            adds.append(add)
        if isinstance(t.get("link"), dict):
            links.append(t["link"])
        if add or isinstance(t.get("link"), dict):
            why.append(" ".join(str(t.get("why") or "").split())[:300])
    if not adds and not links:
        return "nothing to add" + (f" ({'; '.join(notes)})" if notes else "")
    note = " ".join([had.content.rstrip(), *adds])
    ctx = Context(conversation_id=bus.MAIN, now=now_ms())
    ctx.memory = MemoryReader(Slice(structured=list(_KINDS.values()), tasks="open"))
    ctx.propose = Proposer(conversation_id=bus.MAIN, author="Dreamer")
    cap = Captures(ctx, message=f"Dreamer, night of {night}: {' '.join(why)}",
                   evidence=[Evidence(ref=f"dreamer:{night}", source="Dreamer")],
                   provenance=Provenance.AGENT_INFERRED)
    r = await cap.remember(group[0]["kind"], name, note, links=links or None)
    return r.outcome + (f" ({'; '.join(notes)})" if notes else "")


async def run_once(now: int | None = None, llm=None) -> dict:
    """One night: the model reads, the engine checks, asks and looks go to the Inbox, tidy to cards.
    Returns what it did; the same goes to one `dream` audit row."""
    now = now or now_ms()
    llm = llm or _llm()
    if llm is None:
        return {"error": "no model"}
    night = datetime.fromtimestamp(now / 1000).strftime("%Y-%m-%d")  # noqa: DTZ006 — local, by design
    async with _db.get_db() as c:                     # a week unanswered: it closes itself
        await c.execute("UPDATE dream_items SET status='closed', outcome='expired', closed_at=? "
                        "WHERE status='open' AND created_at<?", (now, now - EXPIRE_DAYS * _DAY))
        await c.commit()
    local = datetime.fromtimestamp(now / 1000)  # noqa: DTZ006
    context = (f"It is now {local:%A %d %B %Y, %H:%M} local time. Two weeks ago was "
               f"{(local - timedelta(days=14)):%Y-%m-%d}. Slots that ended before now have passed.\n\n"
               f"Already raised (do not raise again; an outcome is the user's answer):\n"
               f"{json.dumps(await _history(now), ensure_ascii=False) or '[]'}")
    text = await llm.complete(PROMPT.format(cap=CAP, tidy_cap=TIDY_CAP), [{"role": "user", "content": context}],
                              tools=_tools(), max_hops=HOPS)
    out = _parse(text)
    known, dropped = await _known_ids(), []
    kept: list[tuple[str, dict]] = []
    for kind, key in (("ask", "asks"), ("look", "looks")):
        for item in out.get(key) or []:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            if not item.get("evidence"):
                dropped.append(f"{item['title'][:60]}: no evidence")
            elif bad := _bad_ids(item, known):
                dropped.append(f"{item['title'][:60]}: unknown ids {bad}")
            else:
                kept.append((kind, item))
    dropped += [f"{i['title'][:60]}: over the cap of {CAP}" for _, i in kept[CAP:]]
    kept = kept[:CAP]
    async with _db.get_db() as c:
        for kind, item in kept:
            body = {k: item.get(k) for k in ("title", "preview", "body", "evidence", "quick", "why_now") if item.get(k)}
            await c.execute("INSERT INTO dream_items (id, night, kind, body, created_at) VALUES (?,?,?,?,?)",
                            (new_id(), night, kind, json.dumps(body, ensure_ascii=False), now))
        await c.commit()
    groups: dict[tuple, list[dict]] = {}
    for t in out.get("tidy") or []:
        if isinstance(t, dict):
            groups.setdefault((str(t.get("kind")).lower(), slug(str(t.get("name") or ""))), []).append(t)
    tidy = [f"{g[0].get('name')}: over the cap of {TIDY_CAP}" for g in list(groups.values())[TIDY_CAP:]]
    for group in list(groups.values())[:TIDY_CAP]:
        name = group[0].get("name")
        try:
            tidy.append(f"{name}: {await _tidy(group, night)}")
        except Exception as exc:  # noqa: BLE001 — one bad tidy never loses the night
            tidy.append(f"{name}: failed {exc!r}")
    done = {"night": night, "asks": sum(k == "ask" for k, _ in kept), "looks": sum(k == "look" for k, _ in kept),
            "tidy": tidy, "dropped": dropped, "skipped": out.get("skipped") or [],
            **({} if out else {"error": "the model gave no JSON", "text": text[:500]})}
    await _db.audit("dream", night, json.dumps(done, ensure_ascii=False))
    return done


async def _guarded(now: int) -> None:
    try:
        await run_once(now)
    except Exception as exc:  # noqa: BLE001 — a failed night is audited, never fatal
        await _db.audit("dream_error", "dreamer", repr(exc))


async def run_scheduled(now: int | None = None) -> bool:
    """Start tonight's run once per local day, on the first tick at or after `dreamer.hour`. It runs
    in the background: a night takes minutes and the engine must keep answering. True = started."""
    global _running
    cfg = _cfg()
    if not cfg.get("enabled") or (_running is not None and not _running.done()):
        return False
    now = now or now_ms()
    local = datetime.fromtimestamp(now / 1000)  # noqa: DTZ006 — local, by design
    if local.hour < int(cfg.get("hour", 3)):
        return False
    stamp = local.strftime("%Y-%m-%d")
    if await store.get_meta("dream:day") == stamp:
        return False
    await store.set_meta("dream:day", stamp)   # stamped at the start: a failed night waits for tomorrow
    _running = asyncio.create_task(_guarded(now))
    return True


async def items(status: str = "open") -> list[dict]:
    """The Inbox, newest first: open items, or closed ones (with how they closed), or all."""
    where = {"open": "WHERE status='open'", "closed": "WHERE status='closed'"}.get(status, "")
    async with _db.get_db() as c:
        cur = await c.execute(f"SELECT * FROM dream_items {where} ORDER BY created_at DESC, rowid LIMIT 200")
        rows = await cur.fetchall()
    return [{"id": r["id"], "night": r["night"], "kind": r["kind"], **json.loads(r["body"]), "status": r["status"],
             **({"outcome": r["outcome"], "closed": _iso(r["closed_at"])} if r["status"] == "closed" else {})}
            for r in rows]


async def close(item_id: str, outcome: str) -> bool:
    """Close one open item with the user's answer (a quick answer, or their own words)."""
    outcome = " ".join(str(outcome or "").split())[:500] or "closed"
    async with _db.get_db() as c:
        cur = await c.execute("UPDATE dream_items SET status='closed', outcome=?, closed_at=? WHERE id=? AND status='open'",
                              (outcome, now_ms(), str(item_id)))
        await c.commit()
        return cur.rowcount == 1
