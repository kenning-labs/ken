"""The Conversation bus — the outbox and inbox on one SQLite file.

The engine appends renderables to the outbox and reads actions from the inbox.
Interfaces drain the outbox (per-channel cursor) and fill the inbox. The engine
never calls an interface. (ARCHITECTURE.md)
"""
from __future__ import annotations

import json

from .. import db as _db
from ..contracts import Action, ActionChoice, ActionKind, Renderable, RenderableKind
from ..util import now_ms

MAIN = "main"  # V1: one flat conversation


async def ensure_conversation(cid: str = MAIN, *, title: str | None = None,
                              agent: str | None = None,
                              created_at: int | None = None) -> None:
    # backfill a null title/agent on conflict, so a row a plain writer created
    # first (title null) still gets its label when the engine later seeds it.
    async with _db.get_db() as c:
        await c.execute(
            "INSERT INTO conversations (id, title, agent, created_at) VALUES (?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "title=COALESCE(conversations.title, excluded.title), "
            "agent=COALESCE(conversations.agent, excluded.agent)",
            (cid, title, agent, created_at if created_at is not None else now_ms()),
        )
        await c.commit()


# ── outbox ──────────────────────────────────────────────────────────────────

async def append_renderable(r: Renderable) -> int:
    await ensure_conversation(r.conversation_id)
    async with _db.get_db() as c:
        cur = await c.execute(
            "INSERT INTO outbox "
            "(id, conversation_id, kind, author, body, actions, deliver_by, at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (r.id, r.conversation_id, r.kind.value, r.author,
             json.dumps(r.body),
             json.dumps([{"label": a.label, "token": a.token} for a in r.actions]),
             r.deliver_by, r.at or now_ms()),
        )
        await c.commit()
        return cur.lastrowid


async def read_outbox(conversation_id: str, after_seq: int = 0) -> list[tuple[int, Renderable]]:
    """The log after a cursor. An interface delivers these and advances its cursor."""
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT * FROM outbox WHERE conversation_id=? AND seq>? ORDER BY seq",
            (conversation_id, after_seq),
        )
        rows = await cur.fetchall()
    return [(r["seq"], _renderable(r)) for r in rows]


def _renderable(r) -> Renderable:
    return Renderable(
        id=r["id"], conversation_id=r["conversation_id"],
        kind=RenderableKind(r["kind"]), author=r["author"],
        body=json.loads(r["body"]),
        actions=[ActionChoice(**a) for a in json.loads(r["actions"] or "[]")],
        deliver_by=r["deliver_by"], at=r["at"])


async def last_seq(conversation_id: str) -> int:
    """The current tail of the log — a cursor for "anything after now" without reading it."""
    async with _db.get_db() as c:
        cur = await c.execute("SELECT COALESCE(MAX(seq), 0) FROM outbox WHERE conversation_id=?",
                              (conversation_id,))
        return (await cur.fetchone())[0]


async def recent(conversation_id: str, kind: RenderableKind, limit: int = 50, *,
                 since: int = 0, body_type: str | None = None) -> list[Renderable]:
    """The newest `limit` renderables of one kind, newest first — the log is append-only and
    grows forever, so a reader that wants "the latest X" must not scan it all. `since`
    (ms) and `body_type` (the body's `type` field, e.g. "brief") filter in SQL, so a
    chatty day can never push the one row wanted out of the window."""
    sql = "SELECT * FROM outbox WHERE conversation_id=? AND kind=? AND at>=?"
    args: list = [conversation_id, kind.value, since]
    if body_type is not None:
        sql += " AND json_extract(body, '$.type')=?"
        args.append(body_type)
    async with _db.get_db() as c:
        cur = await c.execute(sql + " ORDER BY seq DESC LIMIT ?", (*args, limit))
        rows = await cur.fetchall()
    return [_renderable(r) for r in rows]


async def history(conversation_id: str, author: str | None = None,
                  limit: int = 20) -> list[dict]:
    """Recent turns as LLM-ready messages: user lines from the inbox, agent advice
    from the outbox, merged in time order. `author` is the agent that will read
    this: its own advice is tagged `assistant`; another agent's advice is tagged
    `user` and labelled, so one agent never treats another's words as its own.

    A consent-card tap (an `action_response` anchored to an ASK renderable) is the
    user's decision, so it is a `user` turn for every agent — the one place a refused
    proposal becomes visible to the model instead of being re-proposed. Stamped at the
    tap's own time, so it sorts before a message typed right after it."""
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT at, author, body FROM outbox WHERE conversation_id=? AND kind='advice'",
            (conversation_id,))
        out = await cur.fetchall()
        cur = await c.execute(
            "SELECT at, text FROM inbox WHERE conversation_id=? AND kind='message'",
            (conversation_id,))
        inb = await cur.fetchall()
        cur = await c.execute(
            # the card the tap answered: the latest outbox row with the anchored id (a
            # reprompt re-sends the same id). No card row -> not a consent tap -> skipped.
            "SELECT i.at, i.token, i.text, "
            "  (SELECT o.body FROM outbox o WHERE o.id=i.anchor AND o.kind='ask' "
            "   ORDER BY o.seq DESC LIMIT 1) AS card "
            "FROM inbox i WHERE i.conversation_id=? AND i.kind='action_response' "
            "AND (i.token LIKE 'approve:%' OR i.token LIKE 'reject:%')",
            (conversation_id,))
        taps = await cur.fetchall()
    turns: list[tuple[int, str, str]] = []
    for r in out:
        text = json.loads(r["body"]).get("text", "")
        if not text:                      # a silent reply (a card was the whole answer) is no turn
            continue
        if author is None or r["author"] == author:
            turns.append((r["at"], "assistant", text))
        else:
            turns.append((r["at"], "user", f"[{r['author']}] {text}"))
    for r in inb:
        turns.append((r["at"], "user", r["text"] or ""))
    for r in taps:
        if r["card"]:
            turns.append((r["at"], "user", _tap_line(json.loads(r["card"]), r["token"], r["text"])))
    turns.sort(key=lambda t: t[0])
    return [{"role": role, "content": content} for _, role, content in turns[-limit:]]


async def chat(conversation_id: str, tag: str, limit: int = 30) -> list[dict]:
    """One client's own chat thread, oldest first: each message it sent (tagged "(via <tag>) ") with the
    reply that answered it (outbox advice whose body.reply_to is the message id). Other senders' turns in
    the same conversation (agents, coding sessions) are left out. Each turn says where its answer stands:
    state "answered" (reply text), "card" (the whole answer was cards: reply None, cards = how many),
    "thinking" (not handled yet) or "none" (handled, no answer); bad = the reply reports a failure."""
    prefix = f"(via {tag}) "
    async with _db.get_db() as c:
        # an exact, case-sensitive prefix (LIKE would read _ and % in a tag as wildcards)
        cur = await c.execute(
            "SELECT id, at, text, handled FROM inbox WHERE conversation_id=? AND kind='message' "
            "AND substr(text, 1, ?)=? ORDER BY seq DESC LIMIT ?", (conversation_id, len(prefix), prefix, limit))
        msgs = list(reversed(await cur.fetchall()))
        if not msgs:
            return []
        ids = [m["id"] for m in msgs]
        cur = await c.execute(
            "SELECT body FROM outbox WHERE conversation_id=? AND kind='advice' AND at>=? "
            f"AND json_extract(body, '$.reply_to') IN ({','.join('?' * len(ids))})",
            (conversation_id, msgs[0]["at"], *ids))
        replies: dict[str, dict] = {}
        for r in await cur.fetchall():
            body = json.loads(r["body"])
            had = replies.setdefault(body["reply_to"], {"text": None, "cards": 0, "bad": False})
            had["text"] = body.get("text") or had["text"]
            had["bad"] = had["bad"] or body.get("type") == "error"     # a loud failure (the model was down)
            had["cards"] += len(body.get("cards") or [])
    out = []
    for m in msgs:
        r = replies.get(m["id"]) or {"text": None, "cards": 0, "bad": False}
        state = ("answered" if r["text"] else "card" if r["cards"] else "none" if m["handled"] else "thinking")
        out.append({"id": m["id"], "at": m["at"], "text": (m["text"] or "")[len(prefix):], "reply": r["text"],
                    "cards": r["cards"], "state": state, "bad": r["bad"]})
    return out


def _tap_line(card: dict, token: str, edited: str | None) -> str:
    """One plain line for the model from the ASK body the tap answered, e.g.
    [you tapped No on the card "Update person Dana: manager (you had: colleague)"].
    An edited approve names the value the user typed — that is what got saved."""
    def _flat(s) -> str:   # one line, no stray quotes, so the bracketed shape survives
        return " ".join(str(s or "").split()).replace('"', "'")
    what = _flat(card.get("title", ""))
    if card.get("subject"):
        what += f" {_flat(card['subject'])}"
    value = edited or card.get("proposed") or card.get("question", "")
    what += f": {_flat(value)}"
    if card.get("existing") is not None:
        what += f" (you had: {_flat(card['existing'])})"
    tap = "Yes" if token.startswith("approve:") else "No"
    verb = "edited and tapped Yes" if (edited and tap == "Yes") else f"tapped {tap}"
    return f'[you {verb} on the card "{what}"]'


async def author_of(renderable_id: str) -> str | None:
    """The author of a renderable, by its stable id. A reply anchored to it
    routes back to this author. The latest version wins if the id recurred."""
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT author FROM outbox WHERE id=? ORDER BY seq DESC LIMIT 1",
            (renderable_id,))
        row = await cur.fetchone()
    return row["author"] if row else None


# ── inbox ────────────────────────────────────────────────────────────────────

async def append_action(a: Action) -> int:
    await ensure_conversation(a.conversation_id)
    async with _db.get_db() as c:
        meta = getattr(a, "meta", None) or None
        cur = await c.execute(
            "INSERT INTO inbox "
            "(id, conversation_id, kind, channel, text, token, anchor, at, handled, meta) "
            "VALUES (?,?,?,?,?,?,?,?,0,?)",
            (a.id, a.conversation_id, a.kind.value, a.channel, a.text, a.token,
             a.anchor, a.at or now_ms(), json.dumps(meta) if meta else None),
        )
        await c.commit()
        return cur.lastrowid


async def take_unhandled_actions() -> list[tuple[int, Action]]:
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT * FROM inbox WHERE handled=0 ORDER BY seq")
        rows = await cur.fetchall()
    return [(r["seq"], Action(
        id=r["id"], conversation_id=r["conversation_id"],
        kind=ActionKind(r["kind"]), channel=r["channel"], text=r["text"] or "",
        token=r["token"] or "", anchor=r["anchor"], at=r["at"],
        meta=_meta_of(r))) for r in rows]


def _meta_of(r) -> dict:
    try:
        raw = r["meta"]
    except (IndexError, KeyError):
        return {}
    if not raw:
        return {}
    try:
        m = json.loads(raw)
        return m if isinstance(m, dict) else {}
    except ValueError:
        return {}


async def mark_handled(seq: int) -> None:
    async with _db.get_db() as c:
        await c.execute("UPDATE inbox SET handled=1 WHERE seq=?", (seq,))
        await c.commit()


async def mark_handled_many(seqs: list[int]) -> None:
    """Acknowledge a batch of actions in one statement and one commit — the pump
    acks its whole drained batch at once instead of one connection per action."""
    if not seqs:
        return
    placeholders = ",".join("?" for _ in seqs)
    async with _db.get_db() as c:
        await c.execute(f"UPDATE inbox SET handled=1 WHERE seq IN ({placeholders})", seqs)
        await c.commit()


# ── per-channel cursor ───────────────────────────────────────────────────────

async def get_cursor(channel: str, conversation_id: str = MAIN) -> int:
    async with _db.get_db() as c:
        cur = await c.execute(
            "SELECT last_seq FROM channel_cursors WHERE channel=? AND conversation_id=?",
            (channel, conversation_id))
        row = await cur.fetchone()
    return row["last_seq"] if row else 0


async def set_cursor(channel: str, last_seq: int, conversation_id: str = MAIN) -> None:
    async with _db.get_db() as c:
        await c.execute(
            "INSERT INTO channel_cursors (channel, conversation_id, last_seq) "
            "VALUES (?,?,?) ON CONFLICT(channel, conversation_id) "
            "DO UPDATE SET last_seq=excluded.last_seq",
            (channel, conversation_id, last_seq))
        await c.commit()
