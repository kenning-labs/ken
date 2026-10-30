"""The assistant's index preload: one line per record, named records opened by code, the rest on demand."""
from types import SimpleNamespace

from ken.agents.assistant import _gist, _lookup, _named, _render_memory, _search


def _e(kind, content):
    return SimpleNamespace(kind=kind, content=content)


LONG = "Ship in-app recipes; the primary Q4 initiative. " + "Background detail. " * 60
WORLD = [
    _e("projects.crumbly_recipes", LONG),
    _e("projects.crumbly_growth", "Onboarding and paywall experiments for Crumbly."),
    _e("projects.crumbly_explore", "The Explore surface."),
    _e("people.bram_okafor", "Mobile engineer on the shop app; works on growth. Long history follows. " * 5),
    _e("preferences.writing_style", "Write in Simplified Technical English. Always. " * 6),
    _e("facts.son", "Your son is Leo. He was born in 2019 and likes train sets."),
]
CTX = SimpleNamespace(tasks=[SimpleNamespace(text="call Bram", deadline=None)], structured=WORLD, events=[], now=0)


def test_gist_is_the_first_sentence_clipped():
    assert _gist("One. Two.") == "One."
    assert _gist("x" * 300).endswith("…") and len(_gist("x" * 300)) == 110


def test_index_mode_is_one_line_per_record_but_keeps_tasks_and_preferences_whole():
    full, index = _render_memory(CTX), _render_memory(CTX, mode="index", msg="hello")
    assert len(index) < len(full)
    assert "Background detail." in full and "Background detail." not in index      # the long note is one line
    assert "Crumbly Recipes: Ship in-app recipes; the primary Q4 initiative." in index
    assert "Always. Write in Simplified Technical English. Always." in index       # a preference stays whole
    assert "son: Your son is Leo." in index and "train sets" not in index
    assert "call Bram" in index and "open_record(name)" in index
    assert "Opened for this message (full notes):" not in index                     # the message names nothing


def test_a_named_record_is_opened_in_full_by_code():
    out = _render_memory(CTX, mode="index", msg="what do we know about Bram and the recipes project?")
    opened = out.split("Opened for this message (full notes):")[1]
    assert "Bram Okafor [people]" in opened and "Long history follows." in opened
    assert "Crumbly Recipes [projects]" in opened and "Background detail." in opened
    assert "Growth" not in opened


def test_family_words_do_not_open_every_project():
    # "crumbly" is in three project titles: alone it names none of them
    assert _named(WORLD, "what is new in crumbly?") == []
    assert [e.kind for e in _named(WORLD, "the growth work")] == ["projects.crumbly_growth"]


def test_open_record_returns_the_full_note_best_match_first():
    got = _lookup(WORLD, "recipes")
    assert got[0]["kind"] == "projects.crumbly_recipes" and got[0]["note"] == LONG
    assert _lookup(WORLD, "nobody") == []


def test_search_finds_a_record_by_words_inside_its_note():
    world = WORLD + [_e("facts.problem_report_ownership", "Customer problem reports from Slack. " + "x " * 40 +
                        "Michael owns all checkout-flow bugs on iOS.")]
    (hit, *_) = _search(world, "who owns checkout-flow bugs")
    assert hit["kind"] == "facts.problem_report_ownership" and "Michael owns all checkout-flow bugs" in hit["passage"]
    assert _search(world, "zeppelin") == []


def test_the_source_tag_does_not_open_the_sender():
    world = WORLD + [_e("people.alex_morgan", "The user.")]
    assert _named(world, "(via Alex, test) remind me of the deck") == []
