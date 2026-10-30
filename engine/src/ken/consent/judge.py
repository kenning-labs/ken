"""The judge — the machine skeptic in front of the consent gate. (ARCHITECTURE.md)

An agent write earns its place before it lands: it passes the judge, then the
consent checkpoint, then storage. The judge only ever *chooses* an outcome for a
proposal — it NEVER authors content, and it NEVER drops one. Its three verdicts:

- keep      — new and worth keeping.
- collapse  — a confident duplicate of an existing entry (named by target_id).
- conflict  — clashes with an existing belief (named by target_id) → the gate asks.

The real judge is LLM-backed on the user's configured **fast tier** (structured
JSON: a verdict enum + an optional target id + a one-line reason). With
`decisions.enabled`, OpenAI Decisions judges first and the LLM judge is only the
fallback when a Decisions call fails (FallbackJudge). The gate injects
it, so a test can inject a deterministic fake and exercise every route with no
network. The judge is domain-free: it reasons over the closed noun set and the
generic reconcile targets, never over any one domain's specifics.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

_log = logging.getLogger("ken.judge")


class Verdict(str, Enum):
    KEEP = "keep"
    COLLAPSE = "collapse"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Decision:
    """The judge's whole output. It carries NO content — the gate stores the
    proposal's content, never anything the judge produced. `target_id` names the
    existing entry a collapse duplicates or a conflict clashes with. The judge never
    drops a write: every proposal is kept, collapsed onto a duplicate, or raised as a
    conflict for the user to confirm."""
    verdict: Verdict
    target_id: str | None = None
    reason: str = ""


def keep(*, reason: str = "") -> Decision:
    return Decision(Verdict.KEEP, reason=reason)


def collapse(target_id: str, *, reason: str = "") -> Decision:
    return Decision(Verdict.COLLAPSE, target_id=target_id, reason=reason)


def conflict(existing_id: str, *, reason: str = "") -> Decision:
    return Decision(Verdict.CONFLICT, target_id=existing_id, reason=reason)


class JudgeError(RuntimeError):
    """The judge could not reach a verdict (model outage, or an unparseable answer).
    The gate treats this as a LOUD failure: no write, a visible error. No fallback."""


class Judge(Protocol):
    """The dependency the gate runs first for an agent proposal. Injected, not built
    inside the gate, so tests substitute a deterministic fake with no network."""
    async def __call__(self, proposal, *, message: str, existing: list) -> Decision: ...


# ── the LLM-backed judge (fast tier, structured JSON) ────────────────────────

_SYSTEM = (
    "You are a strict memory skeptic. An agent proposes to store one item. Decide its "
    "fate — you NEVER write, edit, or drop content, you only choose ONE verdict:\n"
    "- keep: the item is new and worth keeping.\n"
    "- collapse: the item duplicates an existing entry — set target_id to that entry's "
    "id. Only when you are confident it is the same thing with the SAME value; when "
    "unsure, keep.\n"
    "- conflict: the item contradicts an existing belief (a different claim or a different "
    "value for the same subject) — set target_id to that belief's id. For a preference or "
    "fact, the same subject with a DIFFERENT value is a conflict, never a collapse.\n"
    "An EVENT is an append-only record of something that HAPPENED — it has no current "
    "value to contradict, so an event is only ever keep or collapse, NEVER conflict.\n"
    "Reply with ONE JSON object and nothing else: "
    '{"verdict":"keep|collapse|conflict","target_id":"<id or null>",'
    '"reason":"<one short line>"}'
)


def _render_existing(existing: list) -> str:
    """The bounded slice, as plain lines the model reads. Each line leads with the id
    the model must echo back as target_id. Generic over the noun objects."""
    lines = []
    for e in existing:
        eid = getattr(e, "id", None) or getattr(e, "kind", "?")
        label = getattr(e, "text", None) or getattr(e, "content", None) \
            or getattr(e, "summary", "")
        kind = getattr(e, "kind", "")
        lines.append(f"- [{eid}] {kind + ': ' if kind else ''}{label}".rstrip())
    return "\n".join(lines) if lines else "(nothing comparable stored yet)"


class LlmJudge:
    """The real judge. Wraps an LLM pinned to the fast tier. Parses a strict JSON
    verdict; anything it cannot parse into a known verdict raises JudgeError, so the
    gate fails loud rather than guessing."""

    def __init__(self, llm) -> None:
        self._llm = llm

    async def __call__(self, proposal, *, message: str, existing: list) -> Decision:
        prompt = (
            f"User's latest message:\n{message or '(none)'}\n\n"
            f"Proposed to store ({proposal.intent}):\n{proposal.content}\n\n"
            f"Existing comparable entries:\n{_render_existing(existing)}"
        )
        try:
            raw = await self._llm.complete(_SYSTEM, [{"role": "user", "content": prompt}])
        except Exception as exc:
            raise JudgeError(f"judge model failed: {exc}") from exc
        return _parse(raw)


def _parse(raw: str) -> Decision:
    """A verdict from the model's JSON. We read ONLY verdict / target_id / reason —
    never any content field, so the model cannot smuggle stored text through the
    judge. An unknown or missing verdict is a JudgeError (loud, no guess)."""
    text = (raw or "").strip()
    if text.startswith("```"):                       # tolerate a fenced block
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError) as exc:
        raise JudgeError(f"judge gave no JSON verdict: {raw!r}") from exc
    try:
        verdict = Verdict(str(data.get("verdict", "")).strip().lower())
    except ValueError as exc:
        raise JudgeError(f"judge gave an unknown verdict: {data.get('verdict')!r}") from exc
    target = data.get("target_id")
    target = str(target) if target not in (None, "", "null") else None
    return Decision(verdict, target_id=target, reason=str(data.get("reason", "")))


# ── the Decisions judge (OpenAI Decisions: typed choice, calibrated confidence) ──

_DECIDE_INSTRUCTIONS = (
    "An agent proposes to store one item in the user's memory. Pick its fate: keep it as "
    "new, collapse it onto the existing entry it duplicates (same thing, SAME value), or "
    "flag it as a conflict with the existing belief it contradicts (same subject, DIFFERENT "
    "value). When unsure between keep and collapse, keep."
)
_DECIDE_GROUNDED = (
    "The proposed content is supported by what the user actually said: it is neither "
    "invented nor distorted, and it adds nothing the user did not state."
)
_DECIDE_MAX_EXISTING = 120   # 1 + 2 per entry <= 241 choices, a bounded question


class DecisionsJudge:
    """The judge as ONE typed `choice` question to OpenAI Decisions: the verdict comes
    back as a value with a calibrated confidence — nothing to parse, no malformed JSON
    possible. Values: `keep`, plus `collapse:<id>` and `conflict:<id>` per existing entry
    (an event never conflicts). Confidence rides in `Decision.reason` so the audit trail
    shows it. The same request also asks one `predicate` — is the proposal GROUNDED in
    what the user said? — whose probability rides in the reason too
    (`decisions 0.90 grounded 0.12`): the silent-failure class (a fabricated or distorted
    save) becomes visible in the audit trail. Asked only when there is a user message to
    ground against. `client` is a `ken.decisions.Decisions` (or any object with the same
    `decide`), injected — no network in tests."""

    def __init__(self, client) -> None:
        self._client = client

    async def __call__(self, proposal, *, message: str, existing: list) -> Decision:
        if len(existing) > _DECIDE_MAX_EXISTING:   # it would see only part of the slice: a
            raise JudgeError(                       # duplicate beyond the cut would save silently
                f"decisions judge: {len(existing)} entries to compare, more than {_DECIDE_MAX_EXISTING}")
        choices = {"keep": "new: not the same thing as any existing entry"}
        rows = []
        for e in existing[:_DECIDE_MAX_EXISTING]:
            eid = str(getattr(e, "id", None) or getattr(e, "kind", "?"))
            label = getattr(e, "text", None) or getattr(e, "content", None) \
                or getattr(e, "summary", "")
            rows.append({"id": eid, "kind": getattr(e, "kind", ""), "text": label})
            choices[f"collapse:{eid}"] = f"duplicates [{eid}] (same thing, same value): {label}"
            if proposal.intent != "event":
                choices[f"conflict:{eid}"] = f"contradicts [{eid}] (same subject, different value): {label}"
        evidence = {"user_message": message or "", "proposed": {"intent": proposal.intent,
                    "content": proposal.content}, "existing": rows}
        # nothing stored to compare (the first task, event or person): keep is the only
        # choice, and the API rejects a choice question with fewer than 2 choices (a 400)
        compare = bool(existing)
        questions = [{"type": "choice", "name": "verdict", "instructions": _DECIDE_INSTRUCTIONS,
                      "choices": [{"value": v, "description": d} for v, d in choices.items()]}] if compare else []
        if message:
            questions.append({"type": "predicate", "name": "grounded",
                              "instructions": _DECIDE_GROUNDED})
        if not questions:
            return keep(reason="decisions nothing to compare")
        answers = await self._client.decide(questions, evidence)
        if compare:
            ans = (answers or {}).get("verdict") or {}
            if ans.get("type") != "choice":     # outage, HTTP error or refusal: loud, no guess
                cause = ans.get("type") or getattr(self._client, "last_error", None) or "no answer"
                raise JudgeError(f"decisions judge failed: {cause}")
            choice, conf = str(ans.get("choice")), float(ans.get("confidence") or 0.0)
            if choice not in choices:                   # cannot happen by contract; loud anyway
                raise JudgeError(f"decisions gave an unknown value: {choice!r}")
        g = (answers or {}).get("grounded") or {}
        grounded = g.get("probability") if g.get("type") == "predicate" else None
        verdict, _, target = choice.partition(":") if compare else ("keep", "", "")
        reason = (f"decisions {conf:.2f}" if compare else "decisions nothing to compare") + (
            f" grounded {float(grounded):.2f}" if grounded is not None else "")
        return Decision(Verdict(verdict), target_id=target or None, reason=reason)


class FallbackJudge:
    """Ask `primary`; when it cannot reach a verdict (a JudgeError: HTTP error, timeout,
    refusal), ask `fallback` instead. A fallback failure propagates as before — loud."""

    def __init__(self, primary: Judge, fallback: Judge) -> None:
        self._primary, self._fallback = primary, fallback

    async def __call__(self, proposal, *, message: str, existing: list) -> Decision:
        try:
            return await self._primary(proposal, message=message, existing=existing)
        except Exception as exc:  # noqa: BLE001 — any primary failure (a JudgeError, a malformed
            _log.warning("primary judge failed, using the fallback: %r", exc)  # reply) falls back
            return await self._fallback(proposal, message=message, existing=existing)
