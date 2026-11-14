"""What a client reads: task names, never ids. Ids stay stored for the checks and the chat's actions."""
import json

from ken import dream, mcp_server
from ken.contracts import Provenance, StructuredEntry
from ken.memory import store

from .test_dream import FakeLLM, _ask, _setup


async def test_ids_become_task_names_in_every_shape_the_model_writes(tmp_path):
    task, _ = await _setup(tmp_path)                 # "book the consulate appointment"
    tid = task.id
    item = _ask("Start the consulate booking?",
                f'Task {tid}: created 2026-10-28, blocked false',
                f"Tasks {tid[:8]} (2026-10-05): open with no deadline")
    item["body"] = [f'The task "book the consulate appointment" ({tid}) is open.',
                    f"Review it (task {tid[:8]}, from 4 Nov) and the PR {tid[:8]}.",
                    "A Slack ts 1793138851.805299 and an unknown id 83c4b6b4 stay as they are."]
    item["evidence"].append("unknown 83c4b6b4")              # never a known id: the night would drop it, but test the read
    await dream.run_once(llm=FakeLLM({"asks": [{**item, "body": ["clean"], "evidence": item["evidence"][:2]}]}))
    (raw,) = await dream.items()
    raw["body"] = item["body"]                                # what a model might still write in prose
    out = await dream.readable(raw)
    name = "“book the consulate appointment”"
    assert out["body"][0] == 'The task "book the consulate appointment" is open.'
    assert out["body"][1] == f"Review it (from 4 Nov) and the PR {name}."
    assert out["body"][2] == "A Slack ts 1793138851.805299 and an unknown id 83c4b6b4 stay as they are."
    assert out["evidence"] == [f"{name}: created 2026-10-28, blocked false", f"{name} (2026-10-05): open with no deadline"]
    assert out["id"] == raw["id"]                             # the item's own id is never touched
    stored = await dream.get(raw["id"])
    assert tid in json.dumps(stored["evidence"])              # storage keeps the ids


async def test_the_inbox_tool_serves_names_and_link_sentences_the_right_way_round(tmp_path):
    task, _ = await _setup(tmp_path)
    await store.upsert_structured(StructuredEntry(id="people.sashko", kind="people.sashko", subject="sashko", content="You.",
                                                  provenance=Provenance.USER_STATED), change_note="test")
    await dream.run_once(llm=FakeLLM({"asks": [_ask("x", f"task {task.id}")], "tidy": [
        {"kind": "project", "name": "emigration", "add": "", "link": {"kind": "person", "name": "sashko", "rel": "owner"}}]}))
    srv = mcp_server.build_server()
    out = await srv.call_tool("inbox", {"status": "open"})
    text = json.dumps(out, default=str, ensure_ascii=False)
    assert task.id not in text and "book the consulate appointment" in text
    fix = next(i for i in await dream.items() if i["kind"] == "tidy")
    assert fix["preview"] == "sashko is the owner of emigration"     # the person owns the project, not the other way
