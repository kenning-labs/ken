"""A record changes by statements, not by whole notes (the 17 Nov benchmark run: 59 of 75 additions to a
note were judged a conflict and carded; a refused price came back in new words inside a true update).
The judge here is scripted: it answers per statement, as the real ones do."""
import json

import httpx

from ken import db, mcp_server
from ken.agents import Proposal
from ken.consent import gate
from ken.consent.judge import DecisionsJudge, FallbackJudge, JudgeError, LlmJudge
from ken.contracts import Evidence, Provenance, StructuredEntry
from ken.conversation import bus
from ken.memory import store

KIND = "projects.pagewise"
NOTE = "Pagewise 2.0 launches on March 10. Pricing stays at 8 USD a month."


class Script:
    """A judge that answers each new statement by a word in it: {'12 USD': ('contradicts', 1)}; else new."""
    def __init__(self, **by_word):
        self.by, self.calls = by_word, []

    async def __call__(self, proposal, *, message, existing):
        raise AssertionError("the whole-note judge must not run for a record that exists")

    async def statements(self, new, old, refused, *, record, message):
        self.calls.append({"new": new, "old": old, "refused": refused})
        return [next((v for w, v in self.by.items() if w.replace("_", " ") in s), ("new", None)) for s in new]


async def _world(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    await store.upsert_structured(StructuredEntry(id=KIND, kind=KIND, subject="Pagewise", content=NOTE,
                                                  provenance=Provenance.USER_STATED), change_note="test")


async def _write(text, judge, provenance=Provenance.USER_INSTRUCTED, source="chat", details=None):
    return await gate.apply(Proposal(intent=f"structured:{KIND}", content=text, provenance=provenance,
                                     evidence=[Evidence(ref="x", source=source)]),
                            conversation_id=bus.MAIN, author="kip", judge=judge, message=text, details=details)


async def _note():
    return next(e for e in await store.get_structured(KIND) if e.kind == KIND).content


async def test_the_users_own_new_statement_is_added_with_no_card(tmp_path):
    await _world(tmp_path)
    did = {}
    out = await _write(NOTE + " Nora owns the release notes.", Script(), details=did)
    assert out == "applied" and did == {"added": 1, "asked": 0, "refused": 0}
    assert await _note() == NOTE + " Nora owns the release notes."       # added; nothing rewritten or removed
    assert await mcp_server.pending_cards() == []
    # the same words again: nothing new, no judge call, no card
    j = Script()
    assert await _write("Nora owns the release notes.", j) == "collapsed" and j.calls == []


async def test_a_contradiction_is_a_card_that_shows_two_sentences_even_for_the_user(tmp_path):
    await _world(tmp_path)
    out = await _write("Pagewise 2.0 launches on March 17. Felix owns the landing page.",
                       Script(March_17=("contradicts", 0)))
    assert out == "conflict"
    assert await _note() == NOTE + " Felix owns the landing page."        # the addition landed; the date waits
    (card,) = await mcp_server.pending_cards()
    assert card["question"] == "Pagewise 2.0 launches on March 17."
    assert card["existing"] == "Pagewise 2.0 launches on March 10."       # one sentence, not the whole note
    assert card["statements"] == [{"op": "replace", "text": "Pagewise 2.0 launches on March 17.",
                                   "replaces": "Pagewise 2.0 launches on March 10."}]
    assert await gate.resolve(card["card_id"], approved=True, conversation_id=bus.MAIN) == "applied"
    assert await _note() == "Pagewise 2.0 launches on March 17. Pricing stays at 8 USD a month. Felix owns the landing page."


async def test_an_agents_statements_wait_on_one_card_and_a_true_part_does_not_carry_a_false_one(tmp_path):
    await _world(tmp_path)
    out = await _write("The launch slipped to March 17. The price is now 12 USD a month.",
                       Script(March_17=("contradicts", 0), **{"12_USD": ("contradicts", 1)}),
                       provenance=Provenance.WORLD, source="slack")
    assert out == "conflict" and await _note() == NOTE                    # nothing lands before the user says so
    (card,) = await mcp_server.pending_cards()
    assert [s["op"] for s in card["statements"]] == ["replace", "replace"]
    # the user keeps the date and drops the price
    assert await gate.resolve(card["card_id"], approved=True, conversation_id=bus.MAIN,
                              edited=json.dumps([{"i": 1, "drop": True}])) == "applied"
    assert await _note() == "The launch slipped to March 17. Pricing stays at 8 USD a month."
    assert await store.refused(KIND) == ["The price is now 12 USD a month."]


async def test_a_refused_statement_does_not_come_back_in_the_same_or_other_words(tmp_path):
    await _world(tmp_path)
    j = Script(**{"12_USD": ("contradicts", 1)})
    await _write("The price is now 12 USD a month.", j, provenance=Provenance.WORLD, source="slack")
    (card,) = await mcp_server.pending_cards()
    assert await gate.resolve(card["card_id"], approved=False, conversation_id=bus.MAIN) == "rejected"
    # the same words: refused with no model call and no card
    j2, did = Script(), {}
    assert await _write("The price is now 12 USD a month.", j2, provenance=Provenance.WORLD, source="slack",
                        details=did) == "refused"
    assert j2.calls == [] and did["refused"] == 1
    # other words, inside a true update: the judge sees the refused list; the true part still gets its card
    j3 = Script(confirmed=("refused", 0), March_17=("contradicts", 0))
    out = await _write("The launch moved to March 17. Kim confirmed a price of 12 USD.", j3,
                       provenance=Provenance.AGENT_INFERRED, source="slack")
    assert j3.calls[0]["refused"] == ["The price is now 12 USD a month."]
    (card,) = await mcp_server.pending_cards()
    assert out == "conflict" and [s["text"] for s in card["statements"]] == ["The launch moved to March 17."]
    # the user can still say it themselves: their words are not held back, and it is no longer refused
    j4 = Script(confirmed=("refused", 0))
    assert await _write("Kim confirmed a price of 12 USD.", j4) == "applied"
    assert (await _note()).endswith("Kim confirmed a price of 12 USD.") and await store.refused(KIND) == []


async def test_an_agents_new_statement_asks_with_plain_words_and_a_preference_lands_at_once(tmp_path):
    await _world(tmp_path)
    assert await _write("Omar is back on 20 Feb.", Script(), provenance=Provenance.AGENT_INFERRED) == "pending"
    (card,) = await mcp_server.pending_cards()
    assert card["show"]["question"] == "Add this to Pagewise?" and card["existing"] is None
    assert await gate.resolve(card["card_id"], approved=True, conversation_id=bus.MAIN, edited="Omar returns 20 Feb.") == "applied"
    assert (await _note()).endswith("Omar returns 20 Feb.")                # their own words for the one statement
    # a low-stakes namespace: a trusted agent's addition needs no card
    await store.upsert_structured(StructuredEntry(id="preferences.meetings", kind="preferences.meetings", subject="you",
                                                  content="No meetings before 10.", provenance=Provenance.USER_STATED),
                                  change_note="test")
    out = await gate.apply(Proposal(intent="structured:preferences.meetings", content="Fridays are meeting-free.",
                                    provenance=Provenance.AGENT_INFERRED), conversation_id=bus.MAIN, author="kip",
                           judge=Script(), message="")
    assert out == "applied"


async def test_a_judge_failure_is_loud_and_the_real_judges_answer_per_statement(tmp_path):
    await _world(tmp_path)

    class Boom(Script):
        async def statements(self, *a, **k):
            raise JudgeError("down")
    assert await _write("Nora owns the release notes.", Boom()) == "judge_error" and await _note() == NOTE

    class LLM:
        async def complete(self, system, messages, **_):
            return 'ok {"verdicts":[{"k":0,"verdict":"contradicts","i":1},{"k":1,"verdict":"same","i":99},{"k":2,"verdict":"refused","i":0}]}'
    got = await LlmJudge(LLM()).statements(["a", "b", "c"], ["x", "y"], ["r"], record="Pagewise", message="")
    assert got == [("contradicts", 1), ("new", None), ("refused", 0)]      # an index out of range reads as new

    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": [{"name": "k0", "type": "choice", "choice": "contradicts:0", "confidence": 0.9},
                                                     {"name": "k1", "type": "choice", "choice": "new", "confidence": 0.8}]})
    from ken.decisions import Decisions
    dj = DecisionsJudge(Decisions("sk-test", transport=httpx.MockTransport(handler)))
    got = await FallbackJudge(dj, LlmJudge(LLM())).statements(["March 17.", "Nora owns it."], ["March 10."], [],
                                                             record="Pagewise", message="")
    assert got == [("contradicts", 0), ("new", None)]
    assert [c["value"] for c in seen["body"]["questions"][0]["choices"]] == ["new", "same:0", "contradicts:0"]
