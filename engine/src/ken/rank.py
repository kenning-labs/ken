"""A ranker — one calibrated score per item from OpenAI Decisions.

The generic "how much / how soon" question a panel needs to sort things without a
reasoning model: send the items as the evidence, one `score` question per item on a
fixed rubric (2-10 levels, low -> high), get a number each. Built by default whenever the
LLM is OpenAI with a real key, unless `decisions.enabled: false` (see app.py); an agent that accepts
`rank` in __init__ gets it injected, else None, and falls back to its deterministic
order. Never raises — a failure is LOGGED and returns None, so a panel is never broken
by the ranker and a broken key never fails silently.
"""
from __future__ import annotations

import logging

_log = logging.getLogger("ken.rank")


_MAX_ITEMS = 400

class DecisionsRanker:
    def __init__(self, client) -> None:
        self._client = client     # a ken.decisions.Decisions (or any object with `decide`)

    async def __call__(self, items: list[str], *, instructions: str,
                       rubric: list[str]) -> list[float] | None:
        """Scores aligned with `items` (0 .. len(rubric)-1, expected value), or None."""
        if not items:
            return []
        if len(items) > _MAX_ITEMS:              # the API takes at most 400 questions
            _log.warning("decisions ranker: %d items, more than %d; using the plain order",
                         len(items), _MAX_ITEMS)
            return None
        evidence = {"items": [{"i": i, "text": t} for i, t in enumerate(items)]}
        levels = [{"label": r, "description": r} for r in rubric]
        questions = [{"type": "score", "name": f"s{i}", "levels": levels,
                      "instructions": f"For item i={i} only: {instructions}"}
                     for i in range(len(items))]
        answers = await self._client.decide(questions, evidence)
        try:
            return [float(answers[f"s{i}"]["score"]) for i in range(len(items))]
        except Exception as exc:  # noqa: BLE001 — None, a refusal: never the panel's failure…
            _log.warning("ranker failed (%d items): %r", len(items), exc)   # …but never silent
            return None
