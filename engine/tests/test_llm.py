"""The LLM wiring — build_llm gating and the assistant's real/stub split.

No network: a fake LLM stands in for the real client.
"""
import json

from ken import db
from ken.agents.assistant import Assistant
from ken.agents.base import Context
from ken.contracts import Action, ActionKind
from ken.conversation import bus
from ken.engine import Engine
from ken.llm import OpenAILLM, _load_args, _run, build_llm, resolve_model
from ken.tools.base import Tool
from ken.util import new_id, now_ms


class FakeLLM:
    def __init__(self, reply="a real reply"):
        self.reply = reply
        self.seen = None

    async def complete(self, system, messages, *, tools=None):
        self.seen = (system, messages, tools)
        return self.reply


# ── build_llm gating ─────────────────────────────────────────────────────────

def test_build_llm_none_without_a_real_key():
    assert build_llm({}) is None
    assert build_llm({"api_key": ""}) is None
    assert build_llm({"api_key": "YOUR_OPENAI_API_KEY"}) is None   # placeholder


def test_build_llm_uses_the_given_model():
    llm = build_llm({"api_key": "sk-test"}, model="gpt-5.6-terra")
    assert isinstance(llm, OpenAILLM)   # constructed, never called here
    assert llm._model == "gpt-5.6-terra"


# ── model resolution: tier + per-agent override ──────────────────────────────

def test_tier_maps_to_its_model():
    cfg = {"tiers": {"fast": "m-fast", "reasoning": "m-reason"}}
    assert resolve_model(cfg, tier="reasoning", agent="SampleAgent") == "m-reason"
    assert resolve_model(cfg, tier="fast", agent="Assistant") == "m-fast"


def test_override_beats_the_tier_for_that_agent_only():
    cfg = {"tiers": {"reasoning": "m-reason"}, "overrides": {"SampleAgent": "m-special"}}
    assert resolve_model(cfg, tier="reasoning", agent="SampleAgent") == "m-special"
    assert resolve_model(cfg, tier="reasoning", agent="Assistant") == "m-reason"


def test_builtin_defaults_when_no_config():
    assert resolve_model({}, tier="fast", agent="Assistant") == "gpt-5.6-luna"
    assert resolve_model({}, tier="reasoning", agent="SampleAgent") == "gpt-5.6-terra"


# ── defensive tool-arg parsing (_load_args) ──────────────────────────────────

def test_load_args_parses_valid_json():
    assert _load_args('{"text": "buy milk", "when": "2026-08-17T15:00"}') == \
        {"text": "buy milk", "when": "2026-08-17T15:00"}


def test_load_args_recovers_the_first_of_concatenated_objects():
    # the fast model sometimes emits two objects back-to-back ("Extra data") — take the first
    assert _load_args('{"a": 1}{"b": 2}') == {"a": 1}


def test_load_args_total_garbage_becomes_empty():
    assert _load_args("not json at all") == {}
    assert _load_args(None) == {}


def test_load_args_non_dict_becomes_empty():
    assert _load_args("[1, 2, 3]") == {}     # a valid JSON array is not tool args
    assert _load_args('"just a string"') == {}


def test_load_args_leaves_valid_values_untouched():
    # the happy path NEVER rewrites a value the model validly emitted — including ones
    # that end in a bracket-bearing structural run (which an over-eager cleaner would maul)
    for val in ("call mom :]", "read chapter [3]", "pay the bill]",
                "grab dinner :}", "the {plan}"):
        assert _load_args(json.dumps({"text": val})) == {"text": val}


# ── the tool executor (_run) ──────────────────────────────────────────────────

async def test_run_tool_error_is_fed_back_not_raised():
    async def boom(**_kw):
        raise ValueError("nope")

    out = await _run(boom, {})
    assert "tool failed" in out      # the error came back as a result, not an exception


async def test_run_scope_error_propagates_not_paraphrased():
    # a ScopeError is a PROGRAMMING error in the agent's slice, not a recoverable tool
    # failure — it must surface loudly, never be fed back to the model as a "tool failed"
    # string the agent could narrate around.
    import pytest

    from ken.capabilities import ScopeError

    async def out_of_scope(**_kw):
        raise ScopeError("namespace 'facts' is not in this agent's slice")

    with pytest.raises(ScopeError):
        await _run(out_of_scope, {})


# ── the /v1/responses tool loop (OpenAILLM._respond) ─────────────────────────
# A fake responses client: OpenAILLM.__init__ builds a real AsyncOpenAI (no network
# until called), so we just swap _client. `script(i, kw)` returns the i-th response.

class _Resp:
    def __init__(self, output, output_text="", id="r"):
        self.output, self.output_text, self.id = output, output_text, id

class _Call:
    type = "function_call"
    def __init__(self, name="ping", call_id="c1", arguments="{}"):
        self.name, self.call_id, self.arguments = name, call_id, arguments

class _Responses:
    def __init__(self, script):
        self.turns, self._script = [], script
    async def create(self, **kw):
        self.turns.append(kw)
        return self._script(len(self.turns) - 1, kw)

class _Client:
    def __init__(self, script):
        self.responses = _Responses(script)


def _fake_llm(script):
    llm = OpenAILLM("sk-test", model="m")
    llm._client = _Client(script)
    return llm


async def test_respond_reports_each_tool_to_the_reporter():
    # the tool loop calls the installed progress reporter with (name, args) BEFORE running a tool,
    # so an interface can show live 'thinking' detail. It's task-local (ContextVar) and best-effort.
    from ken.llm import reset_tool_reporter, set_tool_reporter
    seen = []
    async def rep(name, args):
        seen.append((name, args))
    async def ping(**kw):
        return "pong"
    tool = Tool("ping", "ping",
                {"type": "object", "properties": {"channel": {"type": "string"}}}, ping)

    def script(i, kw):   # hop 0 calls ping(channel=x); hop 1 answers in text
        return (_Resp([_Call(name="ping", call_id="c0", arguments='{"channel": "x"}')], "")
                if i == 0 else _Resp([], "done"))

    llm = _fake_llm(script)
    token = set_tool_reporter(rep)
    try:
        out = await llm._respond("sys", [{"role": "user", "content": "hi"}], [tool])
    finally:
        reset_tool_reporter(token)
    assert out == "done"
    assert seen == [("ping", {"channel": "x"})]   # reporter saw the tool + args, before it ran


def test_progress_step_labels_a_tool_with_its_key_arg():
    from ken.engine import _progress_step
    assert _progress_step("channel_history", {"channel": "monetization"}) == "channel_history monetization"
    assert _progress_step("search", {"query": "ios release"}) == "search ios release"
    assert _progress_step("unread", {}) == "unread"                 # no telling arg → just the name
    # a multi-line arg is flattened so it can't break the one-line spinner row
    assert _progress_step("search", {"query": "line one\nline two"}) == "search line one line two"


async def test_respond_hop_cap_answers_pending_calls_then_forces_a_reply():
    # regression: when the loop exhausts its hop budget with a call STILL pending, the forced
    # final turn must feed a function_call_output for that call (the API rejects a follow-up
    # that leaves a function_call unanswered) AND pass no tools. It used to send only a user
    # message -> "No tool output found for function call ..." 400.
    ran = []
    async def ping(**kw):
        ran.append(kw)
        return "pong"
    tool = Tool("ping", "ping", {"type": "object", "properties": {}}, ping)

    def script(i, kw):
        if "tools" not in kw:              # the tools-off forced answer
            return _Resp([], "final answer")
        return _Resp([_Call(call_id=f"c{i}")], "")   # never stops calling -> exhaust hops

    llm = _fake_llm(script)
    out = await llm._respond("sys", [{"role": "user", "content": "hi"}], [tool], max_hops=3)

    assert out == "final answer"
    forced = llm._client.responses.turns[-1]
    assert "tools" not in forced                       # tools off -> the model must reply in text
    outs = [i for i in forced["input"] if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert outs and all(o.get("call_id") for o in outs)  # the pending call was answered, not dropped
    assert ran                                           # and its fn actually ran


async def test_respond_returns_text_once_the_model_stops_calling():
    async def ping(**kw):
        return "pong"
    tool = Tool("ping", "ping", {"type": "object", "properties": {}}, ping)

    def script(i, kw):
        return _Resp([_Call(call_id="c0")], "") if i == 0 else _Resp([], "done")

    llm = _fake_llm(script)
    out = await llm._respond("sys", [{"role": "user", "content": "hi"}], [tool])
    assert out == "done"
    assert len(llm._client.responses.turns) == 2   # one tool round-trip, then the text answer


async def test_respond_keeps_the_final_hops_answer_without_a_forced_call():
    # if the last hop already returned text (no pending call), the hop-cap tail must return
    # that answer — not burn another create and discard it for a likely-lazier re-answer.
    async def ping(**kw):
        return "pong"
    tool = Tool("ping", "ping", {"type": "object", "properties": {}}, ping)

    def script(i, kw):
        return _Resp([_Call(call_id=f"c{i}")], "") if i < 2 else _Resp([], "answered on last hop")

    llm = _fake_llm(script)
    out = await llm._respond("sys", [{"role": "user", "content": "hi"}], [tool], max_hops=2)
    assert out == "answered on last hop"
    assert len(llm._client.responses.turns) == 3   # initial + 2 hops; NO extra forced create


# ── the assistant's two modes ────────────────────────────────────────────────

class BoomLLM:
    async def complete(self, system, messages, *, tools=None):
        raise RuntimeError("boom")


class EmptyLLM:
    async def complete(self, system, messages, *, tools=None):
        return "   "

async def test_assistant_stub_without_llm():
    a = Assistant(llm=None)
    ctx = Context(conversation_id=bus.MAIN, now=0, message="hello",
                  history=[{"role": "user", "content": "hello"}])
    adv = await a.run(ctx)
    assert adv.body == {"outcome": "no_model"} and "not processed" in adv.text.lower()


async def test_assistant_uses_the_llm_and_passes_history():
    llm = FakeLLM("You slept well.")
    a = Assistant(llm=llm)
    hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"},
            {"role": "user", "content": "how did I sleep?"}]
    ctx = Context(conversation_id=bus.MAIN, now=0, message="how did I sleep?", history=hist)
    adv = await a.run(ctx)
    assert adv.text == "You slept well."
    seen = llm.seen[1]
    assert seen[:2] == hist[:2]                       # prior turns pass through
    assert "how did I sleep?" in seen[-1]["content"]  # current turn, memory folded in


async def test_empty_message_stays_quiet():
    a = Assistant(llm=FakeLLM())
    ctx = Context(conversation_id=bus.MAIN, now=0, message="   ")
    assert await a.run(ctx) is None


async def test_assistant_llm_error_still_replies():
    ctx = Context(conversation_id=bus.MAIN, now=0, message="hi",
                  history=[{"role": "user", "content": "hi"}])
    adv = await Assistant(llm=BoomLLM()).run(ctx)
    # loud failure: flagged as an error (▲ + retry), says nothing was saved, never silent
    assert adv is not None and adv.error and "model" in adv.text.lower()


async def test_assistant_empty_reply_asks_to_rephrase():
    ctx = Context(conversation_id=bus.MAIN, now=0, message="hi",
                  history=[{"role": "user", "content": "hi"}])
    adv = await Assistant(llm=EmptyLLM()).run(ctx)
    assert adv is not None and adv.text.strip() and "rephrase" in adv.text.lower()


# ── through the engine: history is assembled from the conversation ───────────

async def test_engine_feeds_history_to_the_assistant(tmp_path):
    db.configure(str(tmp_path / "ken.db"))
    await db.init_db()
    llm = FakeLLM("hi back")
    eng = Engine()
    eng.register_agent(Assistant(llm=llm), assistant=True)

    await bus.append_action(Action(
        id=new_id(), conversation_id=bus.MAIN, kind=ActionKind.MESSAGE,
        channel="cli:local", text="hi ken", at=now_ms()))
    await eng.pump_inbox()

    out = await bus.read_outbox(bus.MAIN, 0)
    assert any(r.body.get("text") == "hi back" for _, r in out)
    # the engine assembled the current message into history for the model
    assert "hi ken" in llm.seen[1][-1]["content"]


# ── headless Claude Code backend ─────────────────────────────────────────────

from ken.llm import ClaudeCodeLLM, _flatten, is_claude_code


def test_build_llm_picks_claude_code_by_provider_with_no_key():
    cfg = {"provider": "claude-code"}
    llm = build_llm(cfg, model=resolve_model(cfg, tier="reasoning", agent="X"))
    assert isinstance(llm, ClaudeCodeLLM) and llm._model == "opus"      # its own tier defaults
    assert resolve_model(cfg, tier="fast", agent="X") == "sonnet"
    assert resolve_model({"provider": "claude_code", "tiers": {"fast": "haiku"}}, tier="fast", agent="X") == "haiku"
    assert is_claude_code({"provider": "openai"}) is False


def test_flatten_passes_a_lone_user_message_through():
    assert _flatten([{"role": "user", "content": "hi"}]) == "hi"
    assert _flatten([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]) == "User: a\n\nAssistant: b"
    assert _flatten([{"role": "user", "content": [{"type": "text", "text": "x"}]}]) == "x"


async def test_claude_code_plain_completion_runs_one_process(monkeypatch):
    llm = ClaudeCodeLLM(model="sonnet", effort="bogus")
    calls = []

    async def fake_run(prompt, **kw):
        calls.append((prompt, kw))
        return {"result": '{"verdict":"keep"}', "session_id": "s1"}
    monkeypatch.setattr(llm, "_run", fake_run)
    out = await llm.complete("sys", [{"role": "user", "content": "judge this"}])
    assert out == '{"verdict":"keep"}' and llm._effort is None            # an unknown effort is dropped
    assert calls == [("judge this", {"system": "sys", "persist": False})]


async def test_claude_code_tool_loop_calls_then_answers(monkeypatch):
    llm = ClaudeCodeLLM(model="opus")
    seen = []

    async def fake_run(prompt, **kw):
        seen.append((prompt, kw.get("resume")))
        if len(seen) == 1:
            return {"session_id": "s9", "structured_output": {"call": {"name": "lookup", "args": {"q": "jack"}}}}
        return {"session_id": "s9", "structured_output": {"final": "Jack is your manager."}}
    monkeypatch.setattr(llm, "_run", fake_run)
    tool = Tool(name="lookup", description="find a person", parameters={"type": "object"}, fn=lambda q: {"found": q})
    out = await llm.complete("sys", [{"role": "user", "content": "who is jack?"}], tools=[tool])
    assert out == "Jack is your manager."
    assert seen[0] == ("who is jack?", None)
    assert seen[1][1] == "s9" and seen[1][0].startswith("Result of lookup:") and '"found": "jack"' in seen[1][0]


async def test_claude_code_unknown_tool_is_fed_back_and_hop_cap_forces_an_answer(monkeypatch):
    llm = ClaudeCodeLLM(model="opus")
    n = {"calls": 0}

    async def fake_run(prompt, **kw):
        n["calls"] += 1
        if kw.get("schema") == {"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]}:
            return {"structured_output": {"final": "enough"}}
        return {"session_id": "s", "structured_output": {"call": {"name": "nope", "args": {}}}}
    monkeypatch.setattr(llm, "_run", fake_run)
    tool = Tool(name="lookup", description="d", parameters={}, fn=lambda: "x")
    assert await llm.complete("sys", [{"role": "user", "content": "go"}], tools=[tool]) == "enough"
    assert n["calls"] == 1 + 8 + 1                                       # first turn, 8 hops, the forced answer


# ── the /v1/chat/completions tool loop (OpenAILLM._chat_loop, llm.api_style: chat) ──
# A fake chat client: `script(i, kw)` returns the i-th completion.

class _TC:
    type = "function"
    def __init__(self, id="c0", name="ping", arguments="{}"):
        class _F:
            pass
        self.id, self.function = id, _F()
        self.function.name, self.function.arguments = name, arguments

class _Completion:
    def __init__(self, content="", tool_calls=None):
        class _Msg:
            pass
        class _Choice:
            pass
        m = _Msg()
        m.content, m.tool_calls = content, tool_calls
        ch = _Choice()
        ch.message = m
        self.choices = [ch]

class _Completions:
    def __init__(self, script):
        self.turns, self._script = [], script
    async def create(self, **kw):
        self.turns.append(kw)
        return self._script(len(self.turns) - 1, kw)

class _ChatClient:
    def __init__(self, script):
        class _Chat:
            pass
        self.chat = _Chat()
        self.chat.completions = _Completions(script)


def _fake_chat_llm(script):
    llm = OpenAILLM("sk-test", model="m", api_style="chat")
    llm._client = _ChatClient(script)
    return llm


def test_build_llm_reads_api_style():
    assert build_llm({"api_key": "sk-test"})._api_style == "responses"
    assert build_llm({"api_key": "sk-test", "api_style": "chat"})._api_style == "chat"


async def test_chat_loop_feeds_tool_results_back_as_tool_messages():
    ran = []
    async def ping(**kw):
        ran.append(kw)
        return "pong"
    tool = Tool("ping", "ping", {"type": "object", "properties": {"channel": {"type": "string"}}}, ping)

    def script(i, kw):
        return (_Completion("", [_TC(id="c0", arguments='{"channel": "x"}')]) if i == 0
                else _Completion("done"))

    llm = _fake_chat_llm(script)
    out = await llm.complete("sys", [{"role": "user", "content": "hi"}], tools=[tool])
    assert out == "done"
    assert ran == [{"channel": "x"}]
    turns = llm._client.chat.completions.turns
    assert len(turns) == 2
    assert turns[0]["tools"][0]["function"]["name"] == "ping"   # function-tool shape
    msgs = turns[1]["messages"]
    assert msgs[-2]["role"] == "assistant" and msgs[-2]["tool_calls"][0]["id"] == "c0"
    assert msgs[-1] == {"role": "tool", "tool_call_id": "c0", "content": "pong"}


async def test_chat_loop_hop_cap_forces_a_tools_off_answer():
    async def ping(**kw):
        return "pong"
    tool = Tool("ping", "ping", {"type": "object", "properties": {}}, ping)

    def script(i, kw):
        if "tools" not in kw:                      # the forced final answer
            return _Completion("final answer")
        return _Completion("", [_TC(id=f"c{i}")])  # never stops calling

    llm = _fake_chat_llm(script)
    out = await llm._chat_loop("sys", [{"role": "user", "content": "hi"}], [tool], max_hops=3)
    assert out == "final answer"
    turns = llm._client.chat.completions.turns
    assert len(turns) == 4 and "tools" not in turns[-1]
    msgs = turns[-1]["messages"]
    assert msgs[-1]["role"] == "user" and "do not call more tools" in msgs[-1]["content"]
    assert msgs[-2]["role"] == "tool"                 # the last call was answered before the forced turn


async def test_chat_loop_answers_every_parallel_call_in_order():
    async def ping(**kw):
        return "pong"
    tool = Tool("ping", "ping", {"type": "object", "properties": {}}, ping)

    def script(i, kw):
        return _Completion("", [_TC(id="c0"), _TC(id="c1")]) if i == 0 else _Completion("done")

    llm = _fake_chat_llm(script)
    assert await llm.complete("sys", [{"role": "user", "content": "hi"}], tools=[tool]) == "done"
    msgs = llm._client.chat.completions.turns[1]["messages"]
    assert [c["id"] for c in msgs[-3]["tool_calls"]] == ["c0", "c1"]
    assert [m["tool_call_id"] for m in msgs[-2:]] == ["c0", "c1"]


async def test_chat_loop_unknown_tool_is_fed_back():
    async def ping(**kw):
        return "pong"
    tool = Tool("ping", "ping", {"type": "object", "properties": {}}, ping)

    def script(i, kw):
        return _Completion("", [_TC(id="c0", name="nope", arguments=None)]) if i == 0 else _Completion("done")

    llm = _fake_chat_llm(script)
    assert await llm.complete("sys", [{"role": "user", "content": "hi"}], tools=[tool]) == "done"
    msgs = llm._client.chat.completions.turns[1]["messages"]
    assert msgs[-2]["tool_calls"][0]["function"]["arguments"] == "{}"   # None echoes as "{}"
    assert msgs[-1] == {"role": "tool", "tool_call_id": "c0", "content": "unknown tool: nope"}
