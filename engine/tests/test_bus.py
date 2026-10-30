"""The conversation bus — outbox author lookup and the inbox primitives.

The engine speaks through the outbox and hears through the inbox. These tests
pin the two seams the pump depends on: `author_of` (who to route a reply to) and
the take/mark handshake (drain once, never twice).
"""
from ken import db
from ken.contracts import Action, ActionKind, Renderable, RenderableKind
from ken.conversation import bus
from ken.util import new_id, now_ms


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


async def test_author_of_returns_the_latest_speaker(tmp_path):
    await _setup(tmp_path)
    rid = new_id()
    # the same renderable id, reprinted by two authors — latest wins
    await bus.append_renderable(Renderable(
        id=rid, conversation_id=bus.MAIN, kind=RenderableKind.ADVICE,
        author="Daily", body={"text": "v1"}, at=now_ms()))
    await bus.append_renderable(Renderable(
        id=rid, conversation_id=bus.MAIN, kind=RenderableKind.ADVICE,
        author="Coach", body={"text": "v2"}, at=now_ms()))
    assert await bus.author_of(rid) == "Coach"
    assert await bus.author_of("no-such-id") is None


async def test_inbox_take_mark_drains_once(tmp_path):
    await _setup(tmp_path)
    a = Action(id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.MESSAGE,
               channel="cli:local", text="hi", at=now_ms())
    await bus.append_action(a)

    pending = await bus.take_unhandled_actions()
    assert [x.text for _, x in pending] == ["hi"]

    seq, _ = pending[0]
    await bus.mark_handled(seq)
    assert await bus.take_unhandled_actions() == []          # not handed out twice


async def test_history_tags_advice_by_author(tmp_path):
    await _setup(tmp_path)
    await bus.append_renderable(Renderable(
        id=new_id(), conversation_id=bus.MAIN, kind=RenderableKind.ADVICE,
        author="Assistant", body={"text": "hi from me"}, at=1))
    await bus.append_renderable(Renderable(
        id=new_id(), conversation_id=bus.MAIN, kind=RenderableKind.ADVICE,
        author="CoachAgent", body={"text": "recovery 80"}, at=2))
    await bus.append_action(Action(
        id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.MESSAGE,
        channel="cli:local", text="hey", at=3))

    # reading as the Assistant: its own line is 'assistant'; another agent's is
    # a labelled 'user' line, so it never treats Coach's words as its own.
    hist = await bus.history(bus.MAIN, author="Assistant")
    assert {"role": "assistant", "content": "hi from me"} in hist
    assert {"role": "user", "content": "[CoachAgent] recovery 80"} in hist
    assert {"role": "user", "content": "hey"} in hist


async def test_chat_keeps_one_clients_thread_with_its_replies(tmp_path):
    await _setup(tmp_path)
    mine, other, silent = new_id(), new_id(), new_id()
    for aid, text, at in [(mine, "(via Ken Day) what's next?", 1), (other, "(via claude-code) shipped the fix", 2),
                          (silent, "(via Ken Day) nina is the backend contractor", 3),
                          (new_id(), "(via Ken_Day) a wildcard-shaped tag", 4), (new_id(), "(via ken day) other case", 5)]:
        await bus.append_action(Action(id=aid, conversation_id=bus.MAIN, kind=ActionKind.MESSAGE,
                                       channel="http:client", text=text, at=at))
    failed = new_id()
    await bus.append_action(Action(id=failed, conversation_id=bus.MAIN, kind=ActionKind.MESSAGE,
                                   channel="http:client", text="(via Ken Day) and now?", at=5))
    await bus.append_renderable(Renderable(id=new_id(), conversation_id=bus.MAIN, kind=RenderableKind.ADVICE, author="Assistant",
                                           at=5, body={"text": "I couldn't reach the model.", "reply_to": failed, "type": "error"}))
    for reply_to, text, at in [(mine, "Review Nina's mockups at 11.", 6), (other, "Noted.", 7), (silent, "", 8)]:
        await bus.append_renderable(Renderable(id=new_id(), conversation_id=bus.MAIN, kind=RenderableKind.ADVICE,
                                               author="Assistant", at=at,
                                               body={"text": text, "reply_to": reply_to, "cards": ["c1"] if not text else []}))
    waiting = new_id()
    await bus.append_action(Action(id=waiting, conversation_id=bus.MAIN, kind=ActionKind.MESSAGE,
                                   channel="http:client", text="(via Ken Day) still on it?", at=9))

    thread = await bus.chat(bus.MAIN, "Ken Day")
    assert [(m["text"], m["reply"], m["state"]) for m in thread] == [
        ("what's next?", "Review Nina's mockups at 11.", "answered"),   # its own line, prefix stripped, with the reply
        ("nina is the backend contractor", None, "card"),               # a card was the whole answer
        ("and now?", "I couldn't reach the model.", "answered"),          # a failure: answered, but bad
        ("still on it?", None, "thinking"),                             # not handled yet
    ]                       # the coding session, "Ken_Day" (no wildcard) and "ken day" (case) are not in it
    assert [m["bad"] for m in thread] == [False, False, True, False]
    assert await bus.chat(bus.MAIN, "nobody") == []
