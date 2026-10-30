"""The two evals that gate the context engine's model paths. Deterministic cases, a score,
a threshold; a PR that lowers the score does not merge.

  ken eval reconcile   real reports → the typed outcomes kip must produce (runs the REAL
                        configured tiers; never a substitute model)
  ken eval context     real tasks → the items get_context must include, exclude, and ask about
                        (no model: deterministic)

Cases live in engine/evals/*.jsonl, one JSON object per line. Each case seeds a fresh
temporary database, so cases never see each other or the user's real world.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import db
from .agents.base import Proposal
from .consent import gate
from .contracts import Evidence, Provenance, StructuredEntry
from .conversation import bus
from .memory import store
from .util import now_ms

CASES_DIR = Path(__file__).resolve().parents[2] / "evals"
THRESHOLD = 0.8


@dataclass
class CaseResult:
    id: str
    passed: bool
    why: str = ""
    got: dict = field(default_factory=dict)


def load_cases(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


async def _seed(seed: dict, now: int) -> dict[str, str]:
    """Seed a fresh DB from a case: tasks (text → id), people/projects/preferences, events."""
    ids: dict[str, str] = {}
    day = 86_400_000
    # every seeded row gets its OWN evidence ref unless the case names one: shared refs are
    # how get_context relates things, so an accidental common ref would relate everything
    for n, t in enumerate(seed.get("tasks", []) or []):
        text = t if isinstance(t, str) else t["text"]
        due = None
        if isinstance(t, dict) and t.get("due_in_days") is not None:
            due = now + int(t["due_in_days"]) * day
        ref = t.get("ref", f"seed-task-{n}") if isinstance(t, dict) else f"seed-task-{n}"
        tid, _ = await store.add_task(text, Provenance.USER_STATED, deadline=due,
                                      evidence=[Evidence(ref=ref, source="seed")])
        ids[text] = tid
    for ns, entries in (seed.get("beliefs") or {}).items():
        for name, note in entries.items():
            kind = f"{ns}.{name}"
            ref = note.get("ref", f"seed-{kind}") if isinstance(note, dict) else f"seed-{kind}"
            content = note["note"] if isinstance(note, dict) else note
            await store.upsert_structured(StructuredEntry(
                id=kind, kind=kind, subject=name.replace("_", " "), content=content,
                evidence=[Evidence(ref=ref, source="seed")], provenance=Provenance.USER_STATED,
                updated_at=now), change_note="eval seed")
    for e in seed.get("events", []) or []:
        at = now - int(e.get("days_ago", 0)) * day
        ref = e.get("ref", "seed")
        if ref == "__task__":                       # "a report on the case's task"
            ref = next(iter(ids.values()), "seed")
        await gate.apply(Proposal(intent="event", content=e["what"], at=at,
                                  evidence=[Evidence(ref=ref, source="report" if e.get("ref") == "__task__" else "seed")]),
                         conversation_id=bus.MAIN, author="kip")
    return ids


# ── reconcile: report in → typed outcomes out ─────────────────────────────────────────

def _check_reconcile(case: dict, result: list[dict], cards: list[str]) -> CaseResult:
    got = {r["tool"]: r["outcome"] for r in result}
    exp = case.get("expect", {})
    for tool, outcome in (exp.get("must") or {}).items():
        want = outcome if isinstance(outcome, list) else [outcome]
        if got.get(tool) not in want:
            return CaseResult(case["id"], False, f"expected {tool}={want}, got {got.get(tool)!r}", got)
    for tool in exp.get("must_not") or []:
        if tool in got:
            return CaseResult(case["id"], False, f"{tool} must not run, ran with {got[tool]!r}", got)
    any_of = exp.get("any_of") or {}
    if any_of and not any(got.get(t) in (o if isinstance(o, list) else [o]) for t, o in any_of.items()):
        return CaseResult(case["id"], False, f"none of {any_of} happened; got {got}", got)
    if "cards" in exp and len(cards) != int(exp["cards"]):
        return CaseResult(case["id"], False, f"expected {exp['cards']} card(s), got {len(cards)}", got)
    return CaseResult(case["id"], True, "", got)


async def run_reconcile(path: Path | None = None, *, assistant_factory=None, verbose: bool = True) -> float:
    """Run every reconcile case against a fresh DB and the assistant `assistant_factory()`
    returns (default: the real configured one). Prints one line per case and the score."""
    cases = load_cases(path or CASES_DIR / "reconcile.jsonl")
    if assistant_factory is None:
        from .app import build_engine
        eng = build_engine()
        if eng.assistant is None or getattr(eng.assistant, "_llm", None) is None:
            raise SystemExit("ken eval reconcile: no model key configured (llm.api_key) — "
                             "this eval runs the real tiers, never a substitute")
        assistant = eng.assistant
        assistant_factory = lambda: assistant
    from .engine import Engine, _provenance_of
    results: list[CaseResult] = []
    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            db.configure(f"{tmp}/eval.db")
            await db.init_db()
            now = now_ms()
            ids = await _seed(case.get("seed") or {}, now)
            eng = Engine()
            assistant = assistant_factory()
            eng.register_agent(assistant, assistant=True)
            action_id = f"eval-{case['id']}"
            ctx = await eng._assemble(assistant, message=f"(via {case.get('source', 'an agent')}) {case['report']}",
                                      conversation_id=bus.MAIN, action_id=action_id)
            ctx.task_id = ids.get(case.get("task") or "", "")
            # the ceiling is the CHANNEL (which token called), the ask is the provenance:
            # default = the user's own machine (stdio), asking as an agent → AGENT_INFERRED
            fake = type("A", (), {"meta": {"provenance": case.get("provenance", "agent")},
                                  "channel": "mcp:agent:eval" if case.get("channel") == "agent" else "mcp:user"})()
            ctx.provenance = _provenance_of(fake)
            try:
                adv = await assistant.run(ctx)
                result = list((getattr(adv, "body", None) or {}).get("result") or [])
            except Exception as exc:  # noqa: BLE001 — a crash is a failed case, never a crashed eval
                results.append(CaseResult(case["id"], False, f"crashed: {exc!r}"))
                continue
            if getattr(adv, "error", False):          # the model was unreachable: say so, loudly
                results.append(CaseResult(case["id"], False, f"model error: {adv.text}"))
                continue
            cards = await gate.cards_for(action_id, bus.MAIN)
            results.append(_check_reconcile(case, result, cards))
    return _report("reconcile", results, verbose)


# ── context: task in → the right items out ────────────────────────────────────────────

def _check_context(case: dict, out: dict) -> CaseResult:
    exp = case.get("expect", {})
    names = {r["name"].lower() for r in out.get("related", [])}
    gaps = " ".join(out.get("questions_for_agent", [])).lower()
    events = " ".join(e["what"].lower() for e in out.get("events", []))
    for n in exp.get("related_include") or []:
        if n.lower() not in names:
            return CaseResult(case["id"], False, f"missing related {n!r}; got {sorted(names)}", out)
    for n in exp.get("related_exclude") or []:
        if n.lower() in names:
            return CaseResult(case["id"], False, f"{n!r} must not be related", out)
    for n in exp.get("gaps_include") or []:
        if f"'{n.lower()}'" not in gaps:
            return CaseResult(case["id"], False, f"missing gap {n!r}; got {gaps!r}", out)
    for n in exp.get("gaps_exclude") or []:
        if f"'{n.lower()}'" in gaps:
            return CaseResult(case["id"], False, f"{n!r} must not be a gap", out)
    for w in exp.get("events_include") or []:
        if w.lower() not in events:
            return CaseResult(case["id"], False, f"missing event {w!r}", out)
    if exp.get("last_report") is not None:
        got = (out.get("last_report") or {}).get("what", "")
        if exp["last_report"].lower() not in got.lower():
            return CaseResult(case["id"], False, f"last_report {got!r} != {exp['last_report']!r}", out)
    return CaseResult(case["id"], True, "", out)


async def run_context(path: Path | None = None, *, verbose: bool = True) -> float:
    from .mcp_server import get_context
    cases = load_cases(path or CASES_DIR / "context.jsonl")
    results: list[CaseResult] = []
    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            db.configure(f"{tmp}/eval.db")
            await db.init_db()
            now = now_ms()
            ids = await _seed(case.get("seed") or {}, now)
            tid = ids.get(case["task"])
            if tid is None:
                results.append(CaseResult(case["id"], False, "the task must be in seed.tasks"))
                continue
            out = json.loads(await get_context(tid, now=now))
            results.append(_check_context(case, out))
    return _report("context", results, verbose)


def _report(name: str, results: list[CaseResult], verbose: bool) -> float:
    passed = sum(r.passed for r in results)
    score = passed / len(results) if results else 0.0
    if verbose:
        for r in results:
            print(f"  {'ok  ' if r.passed else 'FAIL'} {r.id}" + (f"  — {r.why}" if r.why else ""))
        print(f"{name}: {passed}/{len(results)} = {score:.0%}  (threshold {THRESHOLD:.0%})  "
              f"{'PASS' if score >= THRESHOLD else 'BELOW THRESHOLD'}")
    return score


def main(suite: str, cases: str | None = None) -> int:
    path = Path(cases) if cases else None
    score = asyncio.run(run_reconcile(path) if suite == "reconcile" else run_context(path))
    return 0 if score >= THRESHOLD else 1
