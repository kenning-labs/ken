"""OpenAI Decisions — typed questions (predicate / choice / score) with calibrated answers.

`POST /v1/decisions` over httpx (the installed openai SDK has no `decisions` resource).
On by default whenever the LLM is OpenAI with a real `llm.api_key`; `decisions.enabled:
false` turns it off, and any other provider or base_url leaves it off. `decide` never raises: an HTTP error, a timeout or
a malformed reply is LOGGED and returns None, so each caller picks its own fallback.
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
    def __init__(self, api_key: str, *, base_url: str = _OPENAI_URL,
                 model: str = DEFAULT_MODEL, transport=None) -> None:
        self._key, self._url, self.model = api_key, base_url.rstrip("/"), model
        self._transport = transport      # tests inject an httpx.MockTransport — no network
        self.last_error: str | None = None   # the cause of the last failed call, for the judge's error

    async def decide(self, questions: list[dict], input) -> dict[str, dict] | None:
        """{question name: answer dict} or None on any failure. `input` is the shared
        evidence: a string, or a dict/list sent as JSON text."""
        body = {"model": self.model, "questions": questions,
                "input": input if isinstance(input, str) else json.dumps(input, ensure_ascii=False)}
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=_TIMEOUT) as c:
                r = await c.post(f"{self._url}/decisions", json=body,
                                 headers={"Authorization": f"Bearer {self._key}"})
                if r.is_error:      # the body names the bad field; headers and key stay out
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                self.last_error = None
                return {a["name"]: a for a in r.json()["answers"]}
        except Exception as exc:  # noqa: BLE001 — best-effort; the caller falls back…
            self.last_error = repr(exc)[:600]
            _log.warning("decisions call failed (%d questions): %r", len(questions), exc)  # …loudly
            return None


def build_decisions(cfg: dict, llm_cfg: dict, *, transport=None) -> Decisions | None:
    """A Decisions client when the LLM is OpenAI with a real key, unless `decisions.enabled`
    is false (it is on by default); else None. Only an EXPLICIT `enabled: true` that cannot
    work logs why: a claude-code or other provider stays off quietly."""
    raw = cfg.get("enabled", True)
    if isinstance(raw, str):                    # a quoted "false" / "off" means off too
        raw = raw.strip().lower() not in ("false", "off", "no", "0", "")
    explicit = raw is True and "enabled" in cfg
    if raw is False:
        return None
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
