"""Regression tests — one per bug found in the high-effort code review.

Each test names the defect it locks down. If any regresses, the loop breaks in
the exact way the review predicted.
"""
from ken import db
from ken.agents import Proposal
from ken.consent import gate
from ken.consent.gate import consent_level
from ken.contracts import Provenance, StructuredEntry
from ken.conversation import bus
from ken.memory import store


async def _setup(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()


# 1 — an agent cannot lower its own consent toll (deny-by-default) ────────────

def test_consent_denies_unknown_namespace_by_default():
    assert consent_level("patterns.recovery") == "fast_track"
    assert consent_level("goals.race") == "ask"
    assert consent_level("made_up_ns.x") == "ask"          # not proposer's choice


async def test_agent_made_up_namespace_asks_not_writes(tmp_path):
    await _setup(tmp_path)
    p = Proposal(intent="structured:secretgoal.x",
                 content="a sneaky high-stakes belief",
                 provenance=Provenance.AGENT_INFERRED)
    out = await gate.apply(p, conversation_id=bus.MAIN, author="Sneaky")
    assert out == "pending"
    assert await store.get_structured("secretgoal.") == []


# 7 — a structured revision refreshes provenance ─────────────────────────────

async def test_structured_update_refreshes_provenance(tmp_path):
    await _setup(tmp_path)
    e1 = StructuredEntry(id="patterns.x", kind="patterns.x", subject="you",
                         content="v1", provenance=Provenance.AGENT_INFERRED)
    await store.upsert_structured(e1, change_note="first")
    e2 = StructuredEntry(id="patterns.x", kind="patterns.x", subject="you",
                         content="v2", provenance=Provenance.USER_STATED)
    await store.upsert_structured(e2, change_note="second")
    (only,) = await store.get_structured("patterns.")
    assert only.provenance is Provenance.USER_STATED and only.content == "v2"


# 10 — the changelog append does not lose an entry ───────────────────────────

async def test_changelog_accumulates_on_update(tmp_path):
    await _setup(tmp_path)
    import json
    e = StructuredEntry(id="patterns.y", kind="patterns.y", subject="you",
                        content="c", provenance=Provenance.AGENT_INFERRED)
    await store.upsert_structured(e, change_note="first")
    await store.upsert_structured(e, change_note="second")
    async with db.get_db() as c:
        cur = await c.execute("SELECT changelog FROM structured WHERE id=?", ("patterns.y",))
        row = await cur.fetchone()
    log = json.loads(row["changelog"])
    assert [x["note"] for x in log] == ["first", "second"]
