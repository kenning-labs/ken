"""OpenAI Decisions — typed questions (predicate / choice / score) with calibrated answers.

`POST /v1/decisions` over httpx (the installed openai SDK has no `decisions` resource).
On by default whenever the LLM is OpenAI with a real `llm.api_key`; `decisions.enabled:
false` turns it off, and any other provider or base_url leaves it off. Its own `decisions.api_key`
keeps it on whatever the main model is (the main model on DeepSeek, the judge on OpenAI Decisions). `decide` never raises: an HTTP error, a timeout or
a malformed reply is LOGGED and returns None, so each caller picks its own fallback.

`decisions.provider: jev` answers the same questions with TypeSafe's Jev (`POST /v1/systemone`): the
same `decide`, the same answer shapes, so the judge and the ranker do not know which one they talk to.
"""
from __future__ import annotations

import json
import logging
from urllib.parse import urlparse

import httpx

from .llm import is_claude_code

_log = logging.getLogger("ken.decisions")

DEFAULT_MODEL = "gpt-6-luna"
_OPENAI_URL = "https://api.openai.com/v1"
_TIMEOUT = 10.0


class Decisions:
    label = "decisions"      # the audit reason names who judged: "decisions 0.91" / "jev 0.91"
    _path = "/decisions"

    def __init__(self, api_key: str, *, base_url: str = _OPENAI_URL,
                 model: str = DEFAULT_MODEL, transport=None) -> None:
        self._key, self._url, self.model = api_key, base_url.rstrip("/"), model
        self._transport = transport      # tests inject an httpx.MockTransport — no network
        self.last_error: str | None = None   # the cause of the last failed call, for the judge's error

    def _body(self, questions: list[dict], input) -> dict:
        return {"model": self.model, "questions": questions,
                "input": input if isinstance(input, str) else json.dumps(input, ensure_ascii=False)}

    def _parse(self, questions: list[dict], data: dict) -> dict[str, dict]:
        return {a["name"]: a for a in data["answers"]}

    async def decide(self, questions: list[dict], input) -> dict[str, dict] | None:
        """{question name: answer dict} or None on any failure. `input` is the shared
        evidence: a string, or a dict/list sent as JSON text."""
        try:
            body = self._body(questions, input)
            async with httpx.AsyncClient(transport=self._transport, timeout=_TIMEOUT) as c:
                r = await c.post(self._url + self._path, json=body,
                                 headers={"Authorization": f"Bearer {self._key}"})
                if r.is_error:      # the body names the bad field; headers and key stay out
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                self.last_error = None
                return self._parse(questions, r.json())
        except Exception as exc:  # noqa: BLE001 — best-effort; the caller falls back…
            self.last_error = repr(exc)[:600]
            _log.warning("%s call failed (%d questions): %r", self.label, len(questions), exc)  # …loudly
            return None


JEV_MODEL = "jev-latest"
_JEV_URL = "https://api.typesafe.ai/v1"


class Jev(Decisions):
    """TypeSafe Jev behind the Decisions interface (`POST /v1/systemone`). Ken's questions map one to one:
    choice -> choice (criteria {value: description}; no description = the value alone), predicate -> noul
    (the probability of yes), score -> score (criteria = the levels in order; the answer is the expected
    level, from 0). `state` takes the evidence as it is (Jev reads a string, an object or a list).
    The answers come back in the shapes the callers already read. The callers' limits (241 choices, 400
    score questions a call) are inside Jev's (255 criteria; no stated cap on questions)."""
    label = "jev"
    _path = "/systemone"

    def __init__(self, api_key: str, *, base_url: str = _JEV_URL, model: str = JEV_MODEL, transport=None) -> None:
        super().__init__(api_key, base_url=base_url, model=model, transport=transport)

    @staticmethod
    def _question(q: dict) -> dict:
        kind, text = q["type"], q.get("instructions")
        if kind == "choice":
            return {"type": "choice", "instructions": text,
                    "criteria": {str(c["value"]): c.get("description") for c in q["choices"]}}
        if kind == "score":
            return {"type": "score", "instructions": text,
                    "criteria": [lv.get("description") or lv.get("label") for lv in q["levels"]]}
        if kind == "predicate":
            return {"type": "noul", "instructions": text}
        raise ValueError(f"no Jev question for type {kind!r}")

    def _body(self, questions: list[dict], input) -> dict:
        return {"model": self.model, "state": input, "questions": {q["name"]: self._question(q) for q in questions}}

    def _parse(self, questions: list[dict], data: dict) -> dict[str, dict]:
        """Each answer by the type of the question WE asked (never by a field Jev may leave out). An answer
        that is missing or not an object is left out: the caller sees no answer for that question only."""
        out, answers = {}, data["answers"]
        for q in questions:
            a = answers.get(q["name"])
            if not isinstance(a, dict):
                continue
            if q["type"] == "predicate":            # the callers read a predicate's `probability`
                if a.get("noul") is not None:
                    out[q["name"]] = {"name": q["name"], "type": "predicate", "probability": a["noul"]}
            else:
                out[q["name"]] = {**a, "name": q["name"], "type": q["type"]}
        return out


def build_decisions(cfg: dict, llm_cfg: dict, *, transport=None) -> Decisions | None:
    """A Decisions client when `decisions.api_key` is set, or else when the LLM is OpenAI with a
    real key, unless `decisions.enabled` is false (it is on by default); else None.
    `decisions.provider: jev` gives the Jev client (its own key is required); any other unknown
    provider turns it off with a warning. Only an
    EXPLICIT `enabled: true` that cannot work logs why: a claude-code or other provider stays
    off quietly."""
    raw = cfg.get("enabled", True)
    if isinstance(raw, str):                    # a quoted "false" / "off" means off too
        raw = raw.strip().lower() not in ("false", "off", "no", "0", "")
    explicit = raw is True and "enabled" in cfg
    if raw is False:
        return None
    own = str(cfg.get("api_key") or "")
    provider = str(cfg.get("provider") or "openai").strip().lower()
    if provider == "jev":                           # TypeSafe Jev: its own key, its own host and model
        if not own or own.startswith("YOUR_"):
            _log.warning("decisions.provider is jev but decisions.api_key is not set; it stays off")
            return None
        model, base = str(cfg.get("model") or JEV_MODEL), str(cfg.get("base_url") or _JEV_URL)
        if not model.startswith("jev"):             # a model left over from OpenAI Decisions
            _log.warning("decisions.model %r is not a Jev model; using %s", model, JEV_MODEL)
            model = JEV_MODEL
        if (urlparse(base).hostname or "").endswith("openai.com"):   # never a TypeSafe key to OpenAI
            _log.warning("decisions.base_url %r is OpenAI's; using %s", base, _JEV_URL)
            base = _JEV_URL
        return Jev(own, base_url=base, model=model, transport=transport)
    if provider != "openai":                        # a typo must not send another service's key to OpenAI
        _log.warning("decisions.provider %r is unknown (openai or jev); it stays off", provider)
        return None
    if own and not own.startswith("YOUR_"):          # its own key: the main model may be anything
        return Decisions(own, base_url=str(cfg.get("base_url") or _OPENAI_URL),
                         model=cfg.get("model") or DEFAULT_MODEL, transport=transport)
    key = str(llm_cfg.get("api_key") or "")
    base = str(llm_cfg.get("base_url") or _OPENAI_URL)
    provider = str(llm_cfg.get("provider") or "openai").lower()
    why = ("llm.provider is claude-code" if is_claude_code(llm_cfg)
           else f"llm.provider is {provider!r}, not openai" if provider != "openai"
           else "llm.api_key is not set" if not key or key.startswith("YOUR_")
           else f"llm.base_url {base!r} is not OpenAI" if urlparse(base).hostname != "api.openai.com"
           else None)
    if why:
        if not explicit:
            return None
        _log.warning("decisions.enabled is set but %s; OpenAI Decisions stays off", why)
        return None
    return Decisions(key, base_url=base, model=cfg.get("model") or DEFAULT_MODEL,
                     transport=transport)
