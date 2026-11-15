"""What a person reads: task names, never ids (inbox(readable=true), the web page's view). An AI client gets the
ids by default, so it can act on the tasks. Ids always stay stored."""
import json

from ken import dream, mcp_server
from ken.contracts import Provenance, StructuredEntry
from ken.memory import store

from .test_dream import FakeLLM, _ask, _setup

NAME = "“book the consulate appointment”"


def _short(tid: str) -> str:
    """The shortest prefix of 8+ characters with a letter in it: an all-digit token is never taken
    for an id (a Slack timestamp is not one), and about 2% of random ids start with 8 digits."""
    return next(tid[:n] for n in range(8, len(tid) + 1) if any(c in "abcdef" for c in tid[:n]))


async def _one_ask(tmp_path):
    task, _ = await _setup(tmp_path)                 # "book the consulate appointment"
    await dream.run_once(llm=FakeLLM({"asks": [_ask("x", f"Task {task.id}: created 2026-10-28",
                                                    f"Tasks {_short(task.id)} (2026-10-05): open")]}))
    (item,) = await dream.items()
    return task.id, item


async def test_every_shape_the_model_writes_reads_as_a_name(tmp_path):
    tid, item = await _one_ask(tmp_path)
    item["body"] = [f'The task "book the consulate appointment" ({tid}) is open.',      # a name, then its id: dropped
                    f"Review it (task {_short(tid)}, from 4 Nov) and the PR {_short(tid)}.",    # never lost: the name in its place
                    f"Two are stale ({_short(tid)}).",
                    "A Slack ts 1793138851.805299 and an unknown id 83c4b6b4 stay as they are."]
    out = await dream.readable(item)
    assert out["body"] == ['The task "book the consulate appointment" is open.',
                           f"Review it ({NAME}, from 4 Nov) and the PR {NAME}.",
                           f"Two are stale ({NAME}).",
                           "A Slack ts 1793138851.805299 and an unknown id 83c4b6b4 stay as they are."]
    assert out["evidence"] == [f"{NAME}: created 28 Oct", f"{NAME} (5 Oct): open"]      # dates as a person says them
    assert out["id"] == item["id"] and "links" not in out          # its own fields stay, and no field is added
    assert tid in json.dumps((await dream.get(item["id"]))["evidence"])   # storage keeps the ids


async def test_a_name_put_in_is_never_rewritten_again(tmp_path):
    tid, item = await _one_ask(tmp_path)
    await store.add_task(f"review PR {tid[:10]}", Provenance.USER_STATED)    # a task whose name holds an id-like word
    other = next(t for t in await store.open_tasks() if t.text.startswith("review PR"))
    item["body"] = [f"See task {other.id}."]
    assert (await dream.readable(item))["body"] == [f"See “review PR {tid[:10]}”."]


async def test_ai_clients_get_ids_and_the_page_gets_names(tmp_path):
    tid, _ = await _one_ask(tmp_path)
    srv = mcp_server.build_server()
    raw = json.dumps(await srv.call_tool("inbox", {}), default=str, ensure_ascii=False)
    page = json.dumps(await srv.call_tool("inbox", {"readable": True}), default=str, ensure_ascii=False)
    assert tid in raw                                               # Claude can call complete_task on it
    assert tid not in page and "book the consulate appointment" in page


async def test_an_older_link_fix_gets_its_sentence_and_preview_the_right_way_round(tmp_path):
    await _setup(tmp_path)
    await store.upsert_structured(StructuredEntry(id="people.sashko", kind="people.sashko", subject="sashko", content="You.",
                                                  provenance=Provenance.USER_STATED), change_note="test")
    await dream.run_once(llm=FakeLLM({"tidy": [
        {"kind": "project", "name": "emigration", "add": "", "link": {"kind": "person", "name": "sashko", "rel": "owner"}}]}))
    (fix,) = await dream.items()
    assert fix["preview"] == "sashko is the owner of emigration"     # a new fix: right from the night
    # a fix stored before link sentences existed: no task id anywhere, the old backwards preview
    old = {**fix, "preview": "emigration is owner sashko", "links": [{k: v for k, v in fix["links"][0].items() if k != "text"}]}
    out = await dream.readable(old)
    assert out["links"][0]["text"] == "sashko is the owner of emigration"
    assert out["preview"] == "sashko is the owner of emigration"


def test_raw_timestamps_read_as_a_person_says_them():
    from ken.util import plain_dates
    assert plain_dates('Event 2026-11-13T17:45: "rent is fine"') == '13 Nov, 17:45: "rent is fine"'
    assert plain_dates("deadline 2026-11-09T18:00:00+02:00, missed 2026-11-12") == "deadline 9 Nov, 18:00, missed 12 Nov"
    assert plain_dates("PR #2569 and 2026-13-40 stay") == "PR #2569 and 2026-13-40 stay"


async def test_a_fix_names_its_record_as_a_heading_and_asks_a_question(tmp_path):
    await _setup(tmp_path)
    await store.upsert_structured(StructuredEntry(id="people.sashko", kind="people.sashko", subject="sashko", content="You.",
                                                  provenance=Provenance.USER_STATED), change_note="test")
    await dream.run_once(llm=FakeLLM({"tidy": [
        {"kind": "project", "name": "emigration", "add": "", "link": {"kind": "person", "name": "sashko", "rel": "owner"}}]}))
    (fix,) = await dream.items()
    out = await dream.readable(fix)
    assert out["title"] == "Link Emigration" and out["record"]["name"] == "Emigration"
    assert out["record"]["id"] == fix["record"]["id"] and out["night"] == fix["night"]      # keys stay as stored
    assert out["show"] == {"question": "Link Emigration?", "yes": "Yes, link it", "no": "Skip", "done": "Linked Emigration"}
    assert "show" not in fix and fix["title"] == "Link emigration"                          # an AI client gets the stored item


def test_every_card_kind_asks_in_plain_words():
    from ken.mcp_server import card_show
    assert card_show("people.michael", "Michael", "iOS lead.") == {
        "question": "Update what Ken knows about Michael?", "yes": "Yes, update", "no": "No, keep it",
        "done": ["Updated Michael", "Kept Michael as it was"]}
    assert card_show("act:discard:send the pr", None, None)["question"] == "Drop this task?"
    assert card_show("task:book tickets", None, None)["yes"] == "Yes, add it"
    assert card_show("people.nina", "Nina", None)["question"] == "Save this about Nina?"
    assert card_show("act:something_new:x", None, None)["yes"] == "Yes"                    # an unknown act still reads
