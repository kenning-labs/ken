"""LLM client — provider-agnostic, OpenAI by default; headless Claude Code as the no-key option. (AGENTS.md)

Agents call `complete(system, messages, tools=...)`. No key -> build_llm returns
None and the agent falls back to its no-LLM behaviour. Tests inject a fake, so
they never call a real API.

Model names live in config, not in code. An agent declares a *tier* (fast /
reasoning); a tier maps to a model; a per-agent override wins over the tier.

Tools read our own memory, so the model composes facts it is given rather than
inventing them. OpenAI reasoning models reject function tools on
/v1/chat/completions, so OpenAI runs its tool loop on /v1/responses instead
(keeping reasoning on); the plain-completion path stays on chat.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import shutil
from typing import Protocol

from .capabilities import ScopeError

# Built-in tier -> model, per provider. Config overrides these; with no config, these apply.
_DEFAULT_TIERS = {"fast": "gpt-5.6-luna", "reasoning": "gpt-5.6-terra"}
_CLAUDE_CODE_TIERS = {"fast": "sonnet", "reasoning": "opus"}
_MAX_TOOL_HOPS = 8   # on /v1/responses tools run ONE per hop (parallel_tool_calls off), so a
                     # tool-rich turn needs headroom; the chat loop may run several per hop

# A per-run hook the tool loop calls (await) with (tool_name, args) BEFORE each tool runs, so a
# host can publish live progress ("thinking" detail). It's a ContextVar so it's task-local: each
# agent run sets its own, with no bleed between concurrently-running agents. Default None = off.
_tool_reporter: contextvars.ContextVar = contextvars.ContextVar("ken_tool_reporter", default=None)


def set_tool_reporter(fn):
    """Install the tool-progress hook for the current run; returns a token for reset_tool_reporter."""
    return _tool_reporter.set(fn)


def reset_tool_reporter(token) -> None:
    _tool_reporter.reset(token)


async def _report_tool(name: str, args: dict) -> None:
    """Fire the installed progress reporter (if any) with a tool about to run. Best-effort:
    a reporter error is swallowed so live-progress can never break the tool loop."""
    reporter = _tool_reporter.get()
    if reporter is not None:
        try:
            await reporter(name, args)
        except Exception:   # noqa: BLE001, S110 — progress is cosmetic, never fatal
            pass


class LLM(Protocol):
    async def complete(self, system: str, messages: list[dict], *,
                       tools: list | None = None) -> str: ...


# ── the tool executor ────────────────────────────────────────────────────────

async def _run(fn, args: dict) -> str:
    try:
        result = fn(**args)
        if hasattr(result, "__await__"):
            result = await result
    except ScopeError:
        # an out-of-scope read is a PROGRAMMING error in the agent's slice declaration,
        # not a recoverable tool failure. Never paraphrase it back to the model (which
        # would let a mis-scoped agent narrate around it) — let it surface LOUDLY.
        raise
    except Exception as exc:  # noqa: BLE001 — a bad tool call is fed back, never raised
        return json.dumps({"error": f"tool failed: {exc}"})
    return result if isinstance(result, str) else json.dumps(result)


# ── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAILLM:
    def __init__(self, api_key: str, *, model: str, base_url: str | None = None,
                 effort: str | None = None, api_style: str = "responses") -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._effort = effort   # reasoning effort (reasoning models only); None = model default
        # "responses" (default): the tool loop runs on /v1/responses. "chat": it runs on
        # /v1/chat/completions, for OpenAI-compatible hosts that have no /v1/responses.
        self._api_style = api_style

    async def complete(self, system: str, messages: list[dict], *,
                       tools: list | None = None) -> str:
        if not tools:
            content, _ = await self._chat([{"role": "system", "content": system}, *messages], None)
            return content
        if self._api_style == "chat":
            return await self._chat_loop(system, messages, tools)
        return await self._respond(system, messages, tools)

    async def _chat_loop(self, system: str, messages: list[dict], tools: list, *,
                         max_hops: int = _MAX_TOOL_HOPS) -> str:
        """Tool loop on /v1/chat/completions. The whole transcript is resent each hop:
        the assistant's tool_calls message, then one `tool` message per call (several calls
        per hop are allowed, each gets its message in order). At the hop cap every call is
        already answered, so one more tools-off turn forces the final text. Differences from
        `_respond`: no reasoning effort is sent (compatible hosts reject the field), and the
        model may call several tools in one hop."""
        specs = {t.name: t for t in tools}
        schemas = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                     "parameters": t.parameters}} for t in tools]
        convo = [{"role": "system", "content": system}, *messages]
        for _ in range(max_hops):
            resp = await self._client.chat.completions.create(
                model=self._model, messages=convo, tools=schemas)
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                return (msg.content or "").strip()
            convo.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                for tc in calls]})
            for tc in calls:                         # every call gets a tool message, in order
                name, args = tc.function.name, _load_args(tc.function.arguments)
                await _report_tool(name, args)
                out = (await _run(specs[name].fn, args) if name in specs else f"unknown tool: {name}")
                convo.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        # hop cap hit with every call already answered: force a tools-off final answer
        convo.append({"role": "user", "content": "Answer now with what you have; do not call more tools."})
        resp = await self._client.chat.completions.create(model=self._model, messages=convo)
        return (resp.choices[0].message.content or "").strip()

    async def _chat(self, convo: list[dict], schemas: list | None):
        # No temperature: reasoning models reject a non-default value.
        resp = await self._client.chat.completions.create(
            model=self._model, messages=convo, tools=schemas)
        msg = resp.choices[0].message
        calls = [{"id": tc.id, "name": tc.function.name, "args": _load_args(tc.function.arguments)}
                 for tc in (msg.tool_calls or [])]
        return (msg.content or "").strip(), calls

    async def _respond(self, system: str, messages: list[dict], tools: list, *,
                       max_hops: int = _MAX_TOOL_HOPS) -> str:
        """Tool loop on /v1/responses — the endpoint that allows function tools on
        a reasoning model. Server-side state is chained by previous_response_id, so
        each hop sends only the new tool outputs, not the whole transcript."""
        specs = {t.name: t for t in tools}
        schemas = [{"type": "function", "name": t.name, "description": t.description,
                    "parameters": t.parameters} for t in tools]
        reason = {"reasoning": {"effort": self._effort}} if self._effort else {}
        resp = await self._client.responses.create(
            model=self._model, instructions=system, input=messages, tools=schemas,
            parallel_tool_calls=False, **reason)   # one call per turn — every output is fed back, no mismatch
        for _ in range(max_hops):
            calls = [it for it in resp.output if getattr(it, "type", None) == "function_call"]
            if not calls:
                return (resp.output_text or "").strip()
            outputs = []
            for c in calls:
                spec = specs.get(c.name)
                args = _load_args(c.arguments)
                await _report_tool(c.name, args)
                out = (await _run(spec.fn, args) if spec else f"unknown tool: {c.name}")
                outputs.append({"type": "function_call_output",
                                "call_id": c.call_id, "output": out})
            resp = await self._client.responses.create(
                model=self._model, previous_response_id=resp.id,
                input=outputs, tools=schemas, parallel_tool_calls=False, **reason)
        # hop cap hit. The final hop's response is never re-examined by the loop above: if the
        # model already answered in text (no pending call), return that — don't burn a round-trip
        # re-asking and discard a good answer for a likely-lazier one.
        pending = [it for it in resp.output if getattr(it, "type", None) == "function_call"]
        if not pending:
            return (resp.output_text or "").strip()
        # a call is still pending — the API rejects the next turn unless every call gets an
        # output, so satisfy them first, THEN force a tools-off final answer (no tools → text).
        outputs = []
        for c in pending:
            args = _load_args(c.arguments)
            await _report_tool(c.name, args)
            out = (await _run(specs[c.name].fn, args) if c.name in specs
                   else f"unknown tool: {c.name}")
            outputs.append({"type": "function_call_output", "call_id": c.call_id, "output": out})
        resp = await self._client.responses.create(
            model=self._model, previous_response_id=resp.id,
            input=[*outputs, {"role": "user", "content": "Answer now with what you have; do not call more tools."}],
            **reason)
        return (resp.output_text or "").strip()


# ── headless Claude Code ─────────────────────────────────────────────────────

_CC_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
# the model answers a tool turn through this shape: one call, or the final text
_CC_TURN = {"type": "object", "properties": {
    "call": {"type": "object", "properties": {"name": {"type": "string"}, "args": {"type": "object"}},
             "required": ["name", "args"]},
    "final": {"type": "string"}}}
_CC_FINAL = {"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]}


class ClaudeCodeLLM:
    """`claude -p`: the user's own Claude Code login does the work, no API key. One process per
    completion. A tool loop is one process per hop on a resumed session, the model answering
    through a JSON schema (a call, or the final text). Slower than an API call by a few seconds
    per hop; the right trade for a self-hoster who already pays for Claude.
    ponytail: the tool loop re-parses the model's JSON each hop; a hop cap bounds the processes."""

    def __init__(self, *, model: str, effort: str | None = None, binary: str | None = None,
                 timeout: float = 180.0) -> None:
        self._model = model
        self._effort = effort if effort in _CC_EFFORTS else None
        self._bin = binary or shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
        self._timeout = timeout
        self._cwd = os.path.expanduser("~/.ken")      # no project settings or CLAUDE.md leak in

    async def complete(self, system: str, messages: list[dict], *,
                       tools: list | None = None) -> str:
        prompt = _flatten(messages)
        if not tools:
            out = await self._run(prompt, system=system, persist=False)
            return str(out.get("result") or "").strip()
        specs = {t.name: t for t in tools}
        manifest = "\n".join(f"- {t.name}: {t.description}\n  args schema: {json.dumps(t.parameters)}" for t in tools)
        sys2 = (system + "\n\nYou have these tools. To use one, answer {\"call\": {\"name\": <tool>, \"args\": {...}}}: "
                "ONE call per answer; its result comes back as the next message. When you have what you "
                "need, answer {\"final\": <your reply>}.\nTools:\n" + manifest)
        out = await self._run(prompt, system=sys2, schema=_CC_TURN, persist=True)
        sid = out.get("session_id")
        for _ in range(_MAX_TOOL_HOPS):
            turn = out.get("structured_output") or _load_args(str(out.get("result") or ""))
            call = turn.get("call") if isinstance(turn, dict) else None
            if not call or not call.get("name"):
                return str((turn or {}).get("final") or out.get("result") or "").strip()
            name, args = str(call["name"]), call.get("args") if isinstance(call.get("args"), dict) else {}
            await _report_tool(name, args)
            result = await _run(specs[name].fn, args) if name in specs else f"unknown tool: {name}"
            out = await self._run(f"Result of {name}:\n{result}", resume=sid, schema=_CC_TURN, persist=True)
        out = await self._run("Answer now with what you have; do not call more tools.",
                              resume=sid, schema=_CC_FINAL, persist=True)
        turn = out.get("structured_output") or {}
        return str(turn.get("final") or out.get("result") or "").strip()

    async def _run(self, prompt: str, *, system: str | None = None, schema: dict | None = None,
                   resume: str | None = None, persist: bool) -> dict:
        # no --bare: it skips the keychain login. --tools "" keeps the built-in tools off;
        # --strict-mcp-config keeps the user's MCP servers out of a judge call.
        args = [self._bin, "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
                "--model", self._model]
        if self._effort:
            args += ["--effort", self._effort]
        if system and not resume:
            args += ["--system-prompt", system]
        if schema:
            args += ["--json-schema", json.dumps(schema)]
        if resume:
            args += ["--resume", resume]
        elif not persist:
            args += ["--no-session-persistence"]
        proc = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=self._cwd)
        try:
            raw, err = await asyncio.wait_for(proc.communicate(prompt.encode()), self._timeout)
        except TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude -p timed out after {self._timeout:.0f}s")
        try:
            out = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            raise RuntimeError(f"claude -p returned no JSON (exit {proc.returncode}): {err.decode()[:200]}")
        if proc.returncode or out.get("is_error"):
            raise RuntimeError(f"claude -p failed: {out.get('result') or err.decode()[:200]}")
        return out


def _flatten(messages: list[dict]) -> str:
    """Chat messages as one prompt. A lone user message passes through unchanged."""
    def text(c):
        if isinstance(c, list):
            return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
        return str(c or "")
    if len(messages) == 1 and messages[0].get("role") == "user":
        return text(messages[0].get("content"))
    return "\n\n".join(f"{m.get('role', 'user').capitalize()}: {text(m.get('content'))}" for m in messages)


def _load_args(raw: str | None) -> dict:
    """Parse tool-call arguments defensively — malformed JSON becomes empty args
    (the tool then reports the error through _run) rather than aborting the turn.

    The fast model occasionally emits concatenated objects ("Extra data" — two objects
    back-to-back); raw_decode recovers the first. On the happy path the parsed dict is
    returned UNCHANGED: we never rewrite a value the model validly emitted, because a
    leaked "}},{"-style run and a legitimate value like "call mom :]" are indistinguishable
    once both are valid JSON. If a cheap-model self-hoster does leak such a run, the
    garbled text is visible on their list (and deletable) — strictly safer than silently
    truncating real content. The reasoning default doesn't produce that artifact."""
    text = raw or "{}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:                                    # recover the FIRST object of a concatenated / trailing-garbage payload
            parsed, _ = json.JSONDecoder().raw_decode(text)
        except (json.JSONDecodeError, ValueError):
            return {}
    except TypeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def resolve_model(llm_cfg: dict, *, tier: str, agent: str) -> str:
    """The model for an agent: a per-agent override wins, else the tier's model,
    else the built-in default for that tier."""
    override = (llm_cfg.get("overrides") or {}).get(agent)
    if override:
        return override
    tier_model = (llm_cfg.get("tiers") or {}).get(tier)
    defaults = _CLAUDE_CODE_TIERS if is_claude_code(llm_cfg) else _DEFAULT_TIERS
    return tier_model or defaults.get(tier, defaults["fast"])


def is_claude_code(llm_cfg: dict) -> bool:
    return str(llm_cfg.get("provider") or "").replace("_", "-").lower() == "claude-code"


def build_llm(llm_cfg: dict, *, model: str | None = None,
              effort: str | None = None) -> LLM | None:
    """An LLM for a resolved model, or None if no real key is set. `provider: claude-code`
    needs no key: it runs the user's own Claude Code login."""
    if is_claude_code(llm_cfg):
        return ClaudeCodeLLM(model=model or _CLAUDE_CODE_TIERS["fast"],
                             effort=effort,
                             binary=llm_cfg.get("claude_bin"))
    key = llm_cfg.get("api_key")
    if not key or key.startswith("YOUR_"):
        return None
    return OpenAILLM(key, model=model or _DEFAULT_TIERS["fast"],
                     base_url=llm_cfg.get("base_url"), effort=effort,
                     api_style=str(llm_cfg.get("api_style") or "responses"))
