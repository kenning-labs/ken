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
from .contracts import link_sentence, orient
from .conversation import bus
from .llm import build_llm, resolve_model
from .memory import store
from .tools.base import Tool
from .util import display_name, new_id, norm, now_ms, plain_dates, sentences, slug, title_of

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
- Every claim cites its evidence in `evidence`: full task ids, event times, record names, short quotes. Copy
  ids exactly from the tool answers. In the title, preview, body, quick answers and why_now, name a task by its
  words, never by its id: the user reads those. Never guess numbers: count them.
- Never ask again about something in "Already raised" below, or something the user answered on a card.
- At most {cap} items across asks and looks together, the most important first. Fewer is fine.
- Write like a calm, sharp colleague: plain words, short sentences, no hype. Address the user as "you".
  Use the user's words, not Ken's: "planned time", not "slot"; a record's name, not "(world)" or a tool's name.
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

    s = _schema
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
    out = []
    for r in rows:
        b = json.loads(r["body"])
        out.append({"night": r["night"], "kind": r["kind"], "title": b.get("title"), **({"add": b["add"]} if b.get("add") else {}),
                    "status": r["status"], **({"outcome": r["outcome"]} if r["outcome"] else {})})
    return out


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


async def _tidy(group: list[dict], open_records: set, refused: dict) -> tuple[str, dict | None]:
    """Every tidy item for ONE record -> one proposed fix for the Inbox (or why not). It holds only
    what to ADD, sentence by sentence; inbox_apply appends the ones the note still lacks when the
    user presses Apply, so the model's words never replace the user's and a later edit is kept.
    A fact the user skipped before, or a link to a record Ken does not hold, is never proposed."""
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
    adds, links, why, sources, notes = [], [], [], [], []
    for t in group:
        add = " ".join(str(t.get("add") or "").split())
        if add and norm(add) in norm(" ".join([had.content, *adds])):
            add = ""                                     # the note says it already
        if add and norm(add) in refused.get(had.kind, set()):
            notes.append("a fact you skipped before was left out")
            add = ""
        if add and not await _source_ok(t.get("source"), add):
            notes.append("a fact with no event or chat message behind it was left out")
            add = ""
        if add:
            adds.append(add)
            sources.append(t["source"])
        lk = t.get("link") if isinstance(t.get("link"), dict) else None
        if lk and not await _exists(lk):
            notes.append(f"a link to {lk.get('name')!r}, which Ken does not hold, was left out")
            lk = None
        if lk:
            links.append({**lk, "text": _link_text(had.kind, lk)})
        if add or lk:
            why.append(" ".join(str(t.get("why") or "").split())[:300])
    tail = f" ({'; '.join(notes)})" if notes else ""
    if not adds and not links:
        return "nothing to add" + tail, None
    title = title_of(had.kind)
    item = {"title": f"Add to {title}" if adds else f"Link {title}",
            "preview": (adds[0] if adds else links[0]["text"])[:160],
            "record": {"kind": str(group[0]["kind"]).lower(), "name": title, "id": had.kind},
            "add": " ".join(adds), "adds": adds, "links": links, "sources": sources, "why": " ".join(w for w in why if w)}
    return "proposed" + tail, item


async def _exists(lk: dict) -> bool:
    ns = _KINDS.get(str(lk.get("kind") or "").lower())
    kind = f"{ns}.{slug(str(lk.get('name') or ''))}" if ns else ""
    return bool(ns) and any(e.kind == kind for e in await store.get_structured(ns + "."))


async def _refused(now: int) -> dict:
    """record kind -> the facts the user skipped (or let expire) in the history window, normalized."""
    out: dict = {}
    async with _db.get_db() as c:
        rows = await (await c.execute("SELECT body FROM dream_items WHERE kind='tidy' AND status='closed' "
                                      "AND (outcome LIKE 'skipped%' OR outcome='expired') AND created_at>=?", (now - HISTORY_DAYS * _DAY,))).fetchall()
    for r in rows:
        b = json.loads(r["body"])
        out.setdefault(b.get("record", {}).get("id"), set()).update(norm(x) for x in b.get("adds") or [b.get("add") or ""])
    return out


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
    if not await store.get_meta("dream:cards_withdrawn"):          # once: the cards the first canary raised
        async with _db.get_db() as c:
            old = [r["ask_id"] for r in await (await c.execute("SELECT ask_id FROM pending WHERE author='Dreamer'")).fetchall()]
        for ask_id in old:
            await gate.withdraw(ask_id, reason="the Dreamer proposes fixes in the Inbox, not as cards")
        await store.set_meta("dream:cards_withdrawn", "1")
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
    refused = await _refused(now)
    for group in groups.values():
        name = group[0].get("name")
        try:
            said, item = await _tidy(group, open_records, refused)
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
    return None if r is None else {"id": r["id"], "night": r["night"], "kind": r["kind"], "status": r["status"],
                                   **json.loads(r["body"])}


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


CHAT_PROMPT = """You are the Dreamer, Ken's overnight agent. You raised one item in the user's Inbox, and now
the user talks with you about it. Ken is their context engine: tasks, people, projects and what happened.

The item (kind ask = needs a decision, look = worth knowing, tidy = a proposed fix to a record):
{item}

It is now {now} local time.

How to answer:
- When the user picks an answer or tells you what to do, carry it out with the tools, then call `close` with
  one short line that says what was decided and done. A quick answer they tapped is such an instruction.
- Do only what the user asked. If it is unclear which tasks they mean, ask one short question first.
- Your earlier messages list what you already did ("[done: ...]"). Never do it twice.
- Report only what the tools confirmed. A tool that did not apply did not happen: say so.
- Name tasks by their words. Never show the user an id.
- When the talk reaches a decision with nothing to do ("leave it"), call `close` too.
{tidy}
- Plain words, short sentences, no hype. Address the user as "you". Two to four sentences."""

TIDY_RULES = """- This is a tidy item. You cannot save it: the user presses Apply under the item for that. If they want it
  saved, tell them to press Apply. `revise_fix` changes the proposed sentences in their words. `skip`
  drops the fix when they do not want it. Never write a note yourself."""


def _schema(**p) -> dict:
    return {"type": "object", "properties": {k: {"type": v} for k, v in p.items()}}


_sentences = sentences


async def chat(item_id: str, text: str, *, judge=None, action_id: str | None = None) -> dict:
    """One user turn on one open Inbox item: the Dreamer answers with the read tools, the task writes
    (the user's own instruction, so they apply at once and are audited), `close`, and for a tidy item
    `revise_fix` and `skip` (never Apply: that is the user's tap). The thread is stored on the item,
    each Dreamer message with the actions it took. A model failure still records what was done.
    Returns {reply, outcome, closed, actions}; the engine posts it as the answer to `action_id`."""
    from .agents.assistant import _parse_when
    from .agents.base import Context, Slice
    from .capabilities import MemoryReader, Proposer
    from .captures import Captures
    from .contracts import Evidence, Provenance

    item = await get(item_id)
    if item is None or item["status"] != "open":
        return {"reply": f"There is no open Inbox item {item_id!r}.", "outcome": "no_item", "closed": False, "actions": []}
    text = " ".join(str(text or "").split())[:2000]
    if not text:
        return {"reply": "Say what you want to do with it.", "outcome": "invalid", "closed": False, "actions": []}
    llm = _llm()
    if llm is None:
        return {"reply": "The Dreamer has no model, so it cannot answer.", "outcome": "no_model", "closed": False, "actions": []}
    now, tidy = now_ms(), item["kind"] == "tidy"
    ctx = Context(conversation_id=bus.MAIN, now=now, message=text)
    ctx.memory = MemoryReader(Slice(structured=list(_KINDS.values()) + ["preferences", "facts"], tasks="open", events="recent"))
    ctx.propose = Proposer(conversation_id=bus.MAIN, author="Dreamer", action_id=action_id)   # its cards answer THIS turn
    cap = Captures(ctx, judge=judge, message=text, evidence=[Evidence(ref=f"inbox:{item_id}", source="Dreamer Inbox")],
                   provenance=Provenance.USER_INSTRUCTED, parse_when=_parse_when)
    actions: list[dict] = []
    state = {"closed": None, "fix": None}

    def act(name, fn):
        async def run(**kw):
            r = await fn(**kw)
            actions.append({"tool": name, "outcome": r.outcome, "text": r.text})
            return r.text
        return run

    async def close_item(outcome: str):
        state["closed"] = outcome
        return "closed"

    async def skip(reason: str = ""):
        state["closed"] = "skipped" + (f": {reason}" if str(reason or "").strip() else "")
        return "skipped"

    async def revise_fix(add: str):
        add = " ".join(str(add or "").split())
        if not add:
            return "a fix needs at least one sentence; to drop it, call skip"
        state["fix"] = add
        return "the fix now reads: " + add

    tools = _tools() + [
        Tool("complete_task", "Mark a task done. task = its id (or exact text).", _schema(task="string"),
             act("complete_task", cap.complete_task)),
        Tool("discard_task", "Drop a task (not done), with a short reason.", _schema(task="string", reason="string"),
             act("discard_task", cap.discard_task)),
        Tool("reschedule_task", "Set a task's deadline: a local ISO date or datetime.", _schema(task="string", deadline="string"),
             act("reschedule_task", cap.reschedule_task)),
        Tool("schedule_task", "Book a work slot: start = local ISO datetime, minutes; start '' takes it off the day.",
             _schema(task="string", start="string", minutes="integer"), act("schedule_task", cap.schedule_task)),
        Tool("add_task", "Add a task the user asked for. deadline = local ISO date or datetime, optional.",
             _schema(text="string", deadline="string", project="string"), act("add_task", cap.add_task)),
    ] + ([Tool("revise_fix", "Replace the proposed sentences with the user's version.", _schema(add="string"), revise_fix),
          Tool("skip", "Drop this fix: the user does not want it. reason = their words, short.", _schema(reason="string"), skip)]
         if tidy else
         [Tool("close", "Close this item with one short line: what was decided and done.", _schema(outcome="string"), close_item)])
    thread = list(item.get("thread") or []) + [{"role": "user", "text": text, "at": now}]
    shown = {k: v for k, v in item.items() if k not in ("thread", "status", "id")}
    system = CHAT_PROMPT.format(item=json.dumps(shown, ensure_ascii=False), tidy=TIDY_RULES if tidy else "",
                                now=datetime.fromtimestamp(now / 1000).strftime("%A %d %B %Y, %H:%M"))  # noqa: DTZ006

    def said(m) -> str:                                   # an earlier turn, with what it did
        done = "; ".join(f"{a['tool']}: {a['text']}" for a in m.get("actions") or [])
        return m["text"] + (f"\n[done: {done}]" if done else "")
    messages = [{"role": "user", "content": m["text"]} if m["role"] == "user" else {"role": "assistant", "content": said(m)}
                for m in thread]
    outcome = None
    try:
        reply = (await llm.complete(system, messages, tools=tools)).strip()
    except Exception as exc:  # noqa: BLE001 — a failed model still records what was already done
        await _db.audit("dream_chat_error", item_id, repr(exc))
        done = " ".join(a["text"] for a in actions)
        reply = "The Dreamer's model failed before it finished." + (f" Already done: {done}" if done else " Nothing was changed.")
        outcome, state["closed"] = "error", None
    if not reply:
        done = [a["text"] for a in actions]
        reply = " ".join(done) if done else ("Closed." if state["closed"] else "I have no answer to that. Try again.")
    thread.append({"role": "dreamer", "text": reply, "at": now_ms(), **({"actions": actions} if actions else {})})
    still_open = (await get(item_id) or {}).get("status") == "open"      # another client may have closed it meanwhile
    fields = {"thread": thread, **(fix_fields(state["fix"]) if state["fix"] is not None and still_open else {})}
    await _patch(item_id, fields)
    closed = bool(state["closed"]) and still_open and await close(item_id, state["closed"])
    return {"reply": reply, "outcome": outcome or ("closed" if closed else "answered"), "closed": closed, "actions": actions}


def fix_fields(text: str) -> dict:
    """A fix's sentences in new words: the fields an item stores for them (inbox_chat and inbox_apply)."""
    return {"add": text, "adds": _sentences(text), "preview": text[:160]}


async def set_fix(item_id: str, text: str) -> dict | None:
    """Store the user's own version of a fix's sentences on its open item; the item as it now is, or None."""
    await _patch(item_id, fix_fields(text))
    return await get(item_id)


async def _patch(item_id: str, fields: dict) -> None:
    """Merge fields into an item's body: one write transaction, so two clients never undo each other."""
    async with _db.get_db() as c:
        await c.execute("BEGIN IMMEDIATE")
        r = await (await c.execute("SELECT body FROM dream_items WHERE id=?", (item_id,))).fetchone()
        if r is None:
            await c.rollback()
            return
        await c.execute("UPDATE dream_items SET body=? WHERE id=?",
                        (json.dumps({**json.loads(r["body"]), **fields}, ensure_ascii=False), item_id))
        await c.commit()


def _link_text(kind: str, lk: dict) -> str:
    """One proposed link as the sentence the user reads, in the link's stored direction
    ('sashko potapov is the owner of ken gtm'), whichever record the model put it on."""
    other = f"{_KINDS.get(str(lk.get('kind') or '').lower(), '')}.{slug(str(lk.get('name') or ''))}"
    o = orient(kind, other, str(lk.get("rel") or ""))
    if o is None:
        return f"{title_of(kind)} and {lk.get('name')} ({lk.get('rel')})"
    return link_sentence(title_of(o[0]), o[2], title_of(o[1]))


# an id in brackets right after a quoted name: the name already says it
_ID_AFTER_NAME = re.compile(r"([\"\u201d])\s*\(\s*(?:tasks?\s+)?(\b(?=[0-9]*[a-f])[0-9a-f]{8,32}\b)\s*\)")
# an id right before a quoted name ('Task <id> "Ken: ship it"'): the name already says it
_ID_BEFORE_NAME = re.compile(r"(?:\b[Tt]asks?\s+)?\b((?=[0-9]*[a-f])[0-9a-f]{8,32})\b:?\s+(?=[\"\u201c])")
# "task <id>" or a bare id: one pass, so a name put in is never scanned again
_ANY_ID = re.compile(r"(?:\b[Tt]asks?\s+)?(\b(?=[0-9]*[a-f])[0-9a-f]{8,32}\b)")
_KEEP = ("id", "record", "sources", "night", "closed")    # never shown as text, never rewritten
_EXACT = ("quick", "add", "adds")    # sent back (a quick answer) or written as is (a fix): exact dates stay


async def _task_names() -> list[tuple[str, str]]:
    async with _db.get_db() as c:
        return [(r["id"], " ".join((r["text"] or "").split())) for r in await (await c.execute("SELECT id, text FROM tasks")).fetchall()]


async def readable(value):
    """The Inbox as a person reads it, for a client that shows it as is (the web page asks for it with
    inbox(readable=true); an AI client gets the ids and can act on them). Each task id the Dreamer cited
    becomes the task's quoted name; an id in brackets right after a quoted name is dropped. Unknown ids,
    other numbers and the item's own fields stay. A fix stored before link sentences existed gets them,
    and its preview with them. Raw timestamps read as '13 Nov, 17:45'; a fix names its record as a heading
    ('Ken gtm', 'Artem Kuzmich') and carries `show`: its question, button labels and the receipt text.
    One query for all task names, one pass per string."""
    tasks, cache = await _task_names(), {}

    def name(tok: str) -> str | None:
        if tok not in cache:
            hits = [t for i, t in tasks if i.startswith(tok)]
            cache[tok] = None if len(hits) != 1 else "\u201c" + (hits[0] if len(hits[0]) <= 70 else hits[0][:68].rstrip() + "\u2026") + "\u201d"
        return cache[tok]

    def text(v: str, dates: bool) -> str:
        v = _ID_AFTER_NAME.sub(lambda m: m.group(1) if name(m.group(2)) else m.group(0), v)
        v = _ID_BEFORE_NAME.sub(lambda m: "" if name(m.group(1)) else m.group(0), v)
        v = _ANY_ID.sub(lambda m: name(m.group(1)) or m.group(0), v)
        return plain_dates(v) if dates else v

    def walk(v, dates=True):
        if isinstance(v, str):
            return text(v, dates)
        if isinstance(v, list):
            return [walk(x, dates) for x in v]
        if isinstance(v, dict):
            return {k: (x if k in _KEEP else walk(x, dates and k not in _EXACT)) for k, x in v.items()}
        return v

    def one(item):
        out = walk(item)
        rec = (out.get("record") or {}).get("id")
        if rec and out.get("links") and not all(lk.get("text") for lk in out["links"]):
            out["links"] = [lk if lk.get("text") else {**lk, "text": _link_text(rec, lk)} for lk in out["links"]]
            if not out.get("add"):                        # a link-only fix: its preview is the link sentence
                out["preview"] = out["links"][0]["text"][:160]
        if rec and out.get("kind") == "tidy":             # the record by its display name, and the fix as a question
            shown = display_name(rec)
            out["record"] = {**out["record"], "name": shown}
            out["title"] = ("Add to " if out.get("add") else "Link ") + shown
            n = len(out.get("links") or [])
            also = f", with {n} link{'s' if n > 1 else ''}" if out.get("add") and n else ""
            out["show"] = ({"question": f"Add this to {shown}{also}?", "yes": "Yes, add it", "no": "Skip",
                            "done": [f"Added to {shown}{also}", "Skipped it"]}
                           if out.get("add") else
                           {"question": f"Link {shown}?", "yes": "Yes, link it", "no": "Skip", "done": [f"Linked {shown}", "Skipped it"]})
        return out
    return [one(i) for i in value] if isinstance(value, list) else one(value) if isinstance(value, dict) else walk(value)
