"""The Dreamer — Ken's overnight agent. Once a night it reads everything Ken knows with the
read tools and a reasoning model, then:

  ask    something only the user can decide (a conflict, stuck work, a task the facts made
         pointless) -> an Inbox item
  look   a pattern worth knowing, no decision needed (a slot that never holds) -> an Inbox item
  tidy   an additive fix to a record (a fact a note lacks, a missing link) -> an Inbox item the
         user applies (inbox_apply) or skips: proposed, never applied on its own

Asks and looks together are capped at CAP a night; a quiet night says nothing. Nothing the
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
from .consent import gate
from .conversation import bus
from .llm import build_llm, resolve_model
from .memory import store
from .tools.base import Tool
from .util import new_id, norm, now_ms, slug

CAP = 6                  # asks + looks a night, at most
EXPIRE_DAYS = 7          # an item nobody answered closes itself after a week
HISTORY_DAYS = 90        # how far back the prompt shows past items, and repeats are refused
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
   Never rewrite or remove the user's words. Each tidy waits in the user's Inbox until they apply it.
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
    """The Dreamer's model: `dreamer.llm` when set, as a WHOLE llm section of its own (no key of the
    main one leaks in: a main `provider: claude-code` must not turn a DeepSeek block into claude -p),
    else the main `llm`. Its `model`, else the reasoning tier; its `reasoning_effort`."""
    cfg = _cfg().get("llm") or config.section("llm")
    model = cfg.get("model") or resolve_model(cfg, tier="reasoning", agent="Dreamer")
    return build_llm(cfg, model=model, effort=cfg.get("reasoning_effort"))


async def _chat(limit: int = 60) -> str:
    """What was said to Ken lately, by any client or agent, oldest first: the source tag is in the text."""
    limit = max(1, min(int(limit or 60), 200))
    async with _db.get_db() as c:
        cur = await c.execute("SELECT at, text FROM inbox WHERE conversation_id=? AND kind='message' "
                              "ORDER BY seq DESC LIMIT ?", (bus.MAIN, limit))
        rows = list(reversed(await cur.fetchall()))
    from .mcp_server import _iso
    return json.dumps({"messages": [{"at": _iso(r["at"]), "text": r["text"]} for r in rows]})


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


_ID_SQL = ("SELECT 1 FROM tasks WHERE id LIKE ?1 || '%' OR evidence LIKE '%' || ?1 || '%' "
           "UNION ALL SELECT 1 FROM pending WHERE ask_id LIKE ?1 || '%' "
           "UNION ALL SELECT 1 FROM audit WHERE kind='card_outcome' AND ref LIKE ?1 || '%' "
           "UNION ALL SELECT 1 FROM dream_items WHERE id LIKE ?1 || '%' "
           "UNION ALL SELECT 1 FROM events WHERE id LIKE ?1 || '%' OR evidence LIKE '%' || ?1 || '%' "
           "UNION ALL SELECT 1 FROM structured WHERE evidence LIKE '%' || ?1 || '%' LIMIT 1")


async def _bad_ids(item) -> list[str]:
    """Every id-like token the item cites that names nothing Ken holds: a task (open or closed), a
    card (open or answered), an Inbox item, an event, or an evidence ref on a record."""
    bad = []
    async with _db.get_db() as c:
        for tok in sorted(set(_ID.findall(json.dumps(item)))):
            if await (await c.execute(_ID_SQL, (tok,))).fetchone() is None:
                bad.append(tok)
    return bad


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", norm(text)) if len(w) > 3}


async def _source_ok(src, add: str) -> bool:
    """A tidy fact's source must exist (an event, expired or not, or a chat message, at the minute
    it names) and share at least two real words with the fact, so a time picked from an unrelated
    event does not pass.
    ponytail: a word overlap, not a judge; a paraphrase with no shared noun is refused (sparse beats false)."""
    if not isinstance(src, dict):
        return False
    for key, sql in (("event", "SELECT summary AS t FROM events WHERE at>=? AND at<?"),
                     ("chat", "SELECT text AS t FROM inbox WHERE kind='message' AND at>=? AND at<?")):
        when = str(src.get(key) or "").strip()
        if not when:
            continue
        try:
            at = int(datetime.fromisoformat(when[:16]).timestamp() * 1000)
        except ValueError:
            return False
        async with _db.get_db() as c:
            rows = await (await c.execute(sql, (at, at + 60_000))).fetchall()
        return any(len(_words(r["t"] or "") & _words(add)) >= 2 for r in rows)
    return False


async def _tidy(group: list[dict], open_records: set) -> tuple[str, dict | None]:
    """Every tidy item for ONE record -> one proposed fix for the Inbox (or why not). It holds only
    what to ADD; inbox_apply appends it to the note as it is when the user presses Apply, so the
    model's words never replace the user's and a note edited since the night is not lost."""
    t = group[0]
    ns = _KINDS.get(str(t.get("kind") or "").lower())
    name = str(t.get("name") or "").strip()
    if not ns or not name:
        return "no record named", None
    had = next((e for e in await store.get_structured(ns + ".") if e.kind == f"{ns}.{slug(name)}"), None)
    if had is None:
        return f"no {t.get('kind')} {name!r}", None
    if had.kind in open_records:
        return "a fix for it already waits in the Inbox", None
    adds, links, why, notes = [], [], [], []
    for t in group:
        add = " ".join(str(t.get("add") or "").split())
        if add and norm(add) in norm(" ".join([had.content, *adds])):
            add = ""                                     # the note says it already
        if add and not await _source_ok(t.get("source"), add):
            notes.append("a fact with no event or chat message behind it was left out")
            add = ""
        if add:
            adds.append(add)
        if isinstance(t.get("link"), dict):
            links.append(t["link"])
        if add or isinstance(t.get("link"), dict):
            why.append(" ".join(str(t.get("why") or "").split())[:300])
    tail = f" ({'; '.join(notes)})" if notes else ""
    if not adds and not links:
        return "nothing to add" + tail, None
    title = f"Add to {_title_of(had)}" if adds else f"Link {_title_of(had)}"
    item = {"title": title, "preview": (adds[0] if adds else _link_line(_title_of(had), links[0]))[:160],
            "record": {"kind": str(group[0]["kind"]).lower(), "name": _title_of(had), "id": had.kind},
            "add": " ".join(adds), "links": links, "why": " ".join(w for w in why if w)}
    return "proposed" + tail, item


def _title_of(e) -> str:
    return e.subject or e.kind.split(".", 1)[-1].replace("_", " ")


def _link_line(name: str, lk: dict) -> str:
    return f"{name} is {lk.get('rel', 'linked to')} {lk.get('name', '')}".strip()


async def run_once(now: int | None = None, llm=None) -> dict:
    """One night: the model reads, the engine checks, and asks, looks and tidy go to the Inbox.
    Returns what it did; the same goes to one `dream` audit row."""
    now = now or now_ms()
    llm = llm or _llm()
    if llm is None:
        await _db.audit("dream_error", "dreamer", "no model: set dreamer.llm (or llm) with a real api_key")
        return {"error": "no model"}
    night = datetime.fromtimestamp(now / 1000).strftime("%Y-%m-%d")  # noqa: DTZ006 — local, by design
    # the Dreamer never keeps a card in the shared queue ("Ken wants to check" is for writes from
    # outside); cards an older version raised go back, and tonight proposes them in the Inbox
    async with _db.get_db() as c:
        old = [r["ask_id"] for r in await (await c.execute("SELECT ask_id FROM pending WHERE author='Dreamer'")).fetchall()]
    for ask_id in old:
        await gate.withdraw(ask_id, reason="the Dreamer proposes fixes in the Inbox, not as cards")
    async with _db.get_db() as c:                     # a week unanswered: it closes itself
        await c.execute("UPDATE dream_items SET status='closed', outcome='expired', closed_at=? "
                        "WHERE status='open' AND created_at<?", (now, now - EXPIRE_DAYS * _DAY))
        await c.commit()
    local = datetime.fromtimestamp(now / 1000)  # noqa: DTZ006
    history = await _history(now)
    context = (f"It is now {local:%A %d %B %Y, %H:%M} local time. Two weeks ago was "
               f"{(local - timedelta(days=14)):%Y-%m-%d}. Slots that ended before now have passed.\n\n"
               f"Already raised (do not raise again; an outcome is the user's answer):\n"
               f"{json.dumps(history, ensure_ascii=False)}")
    text = await llm.complete(PROMPT.format(cap=CAP), [{"role": "user", "content": context}],
                              tools=_tools(), max_hops=HOPS)
    out = _parse(text)
    dropped = []
    seen = {norm(h["title"] or "") for h in history}
    kept: list[tuple[str, dict]] = []
    for kind, key in (("ask", "asks"), ("look", "looks")):
        for item in out.get(key) or []:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            if not item.get("evidence"):
                dropped.append(f"{item['title'][:60]}: no evidence")
            elif norm(item["title"]) in seen:
                dropped.append(f"{item['title'][:60]}: raised before")
            elif bad := await _bad_ids(item):
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
    tidy, fixes = [], []
    open_records = {json.loads(i["body"]).get("record", {}).get("id") for i in await _open("tidy")}
    for group in groups.values():
        name = group[0].get("name")
        try:
            said, item = await _tidy(group, open_records)
        except Exception as exc:  # noqa: BLE001 — one bad tidy never loses the night
            said, item = f"failed {exc!r}", None
        tidy.append(f"{name}: {said}")
        if item:
            fixes.append(item)
    async with _db.get_db() as c:
        for item in fixes:
            await c.execute("INSERT INTO dream_items (id, night, kind, body, created_at) VALUES (?,?,?,?,?)",
                            (new_id(), night, "tidy", json.dumps(item, ensure_ascii=False), now))
        await c.commit()
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


async def _open(kind: str) -> list:
    async with _db.get_db() as c:
        return await (await c.execute("SELECT body FROM dream_items WHERE status='open' AND kind=?", (kind,))).fetchall()


async def get(item_id: str) -> dict | None:
    async with _db.get_db() as c:
        r = await (await c.execute("SELECT * FROM dream_items WHERE id=?", (str(item_id),))).fetchone()
    return None if r is None else {"id": r["id"], "kind": r["kind"], "status": r["status"], **json.loads(r["body"])}


async def items(status: str = "open") -> list[dict]:
    """The Inbox, newest first: open items, or closed ones (with how they closed), or all."""
    where = {"open": "WHERE status='open'", "closed": "WHERE status='closed'"}.get(status, "")
    async with _db.get_db() as c:
        cur = await c.execute(f"SELECT * FROM dream_items {where} ORDER BY created_at DESC, rowid LIMIT 200")
        rows = await cur.fetchall()
    from .mcp_server import _iso
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
