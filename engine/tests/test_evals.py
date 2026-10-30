"""The two gated evals: the context suite runs for real (deterministic); the reconcile
runner is exercised with a scripted assistant so CI needs no model key."""
import asyncio
import json

from ken import evals
from ken.agents.assistant import Assistant


def test_context_eval_runs_the_repo_cases_above_threshold(capsys):
    score = asyncio.run(evals.run_context())
    out = capsys.readouterr().out
    assert "context:" in out and score >= evals.THRESHOLD, out


def test_context_eval_catches_a_wrong_expectation(tmp_path):
    cases = tmp_path / "c.jsonl"
    cases.write_text(json.dumps({"id": "x", "seed": {"tasks": ["call Robert"], "beliefs": {"people": {"robert": "PM"}}},
                                 "task": "call Robert", "expect": {"related_exclude": ["robert"]}}) + "\n")
    assert asyncio.run(evals.run_context(cases, verbose=False)) == 0.0


class _Scripted:
    """An LLM that runs a fixed tool script per case, keyed on a word in the report."""
    def __init__(self, script): self.script = script
    async def complete(self, system, messages, *, tools=None):
        specs = {t.name: t for t in (tools or [])}
        text = messages[-1]["content"]
        for key, calls in self.script.items():
            if key in text:
                for name, args in calls:
                    await specs[name].fn(**args)
                return "ok"
        return "nothing to do"


def test_reconcile_runner_scores_scripted_outcomes(tmp_path, capsys):
    cases = tmp_path / "r.jsonl"
    cases.write_text("\n".join([
        json.dumps({"id": "done", "seed": {"tasks": ["ship it"]}, "task": "ship it", "report": "Shipped it, mark done.",
                    "expect": {"must": {"complete_task": "completed"}, "cards": 0}}),
        json.dumps({"id": "must-not", "seed": {"tasks": ["ship it"]}, "task": "ship it", "report": "Started it.",
                    "expect": {"must": {"log_event": "event"}, "must_not": ["complete_task"]}}),
        json.dumps({"id": "agent-card", "source": "codex", "channel": "agent", "seed": {},
                    "report": "Add a task: review PR 12.", "expect": {"must": {"add_task": "pending"}, "cards": 1}}),
    ]) + "\n")
    script = {"mark done": [("complete_task", {"task": "ship it"})],
              "Started": [("log_event", {"text": "started it"}), ("complete_task", {"task": "ship it"})],
              "review PR": [("add_task", {"text": "review PR 12"})]}
    score = asyncio.run(evals.run_reconcile(cases, assistant_factory=lambda: Assistant(llm=_Scripted(script))))
    out = capsys.readouterr().out
    assert "ok   done" in out and "FAIL must-not" in out and "ok   agent-card" in out, out
    assert abs(score - 2 / 3) < 1e-9
