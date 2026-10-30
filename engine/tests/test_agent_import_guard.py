"""First-party discipline: no file under agents/ imports the engine's privileged
modules. An agent reads/writes/publishes ONLY through the injected capabilities
(ctx.memory / ctx.propose / ctx.view). The composition layer (engine, capabilities,
consent, memory) holds store/gate/db — agents must not.

This is a first-party LINT, not a sandbox: Python cannot stop a malicious in-process
plugin (real containment needs process isolation, out of scope). Its value is catching
an accidental direct import that would quietly re-open the seam.
"""
import ast
import pathlib

AGENTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "ken" / "agents"

# The privileged MODULES an agent must never reach — matched on the module itself, by
# its LAST dotted component, so the guard is robust to ALIASING: the bound name (asname)
# is irrelevant, only which module object is imported. `db` in particular has no unique
# path token, so a bound-name check let `from .. import db as d` / `import ken.db as foo`
# slip through and hand out raw SQL that bypasses gate/judge/scope — this catches them.
_FORBIDDEN_MODULES = {"store", "gate", "db"}


def _last(dotted: str) -> str:
    return dotted.rsplit(".", 1)[-1]


def _forbidden_imports(source: str) -> set[str]:
    """The privileged imports a source pulls in, as human-readable strings (empty when
    clean). Matches on the imported MODULE (never the bound name), so every aliased form
    is caught: `from ..memory import store as s`, `from ..consent import gate as g`,
    `from .. import db as d`, `import ken.db as foo`, `import ken.memory.store as m`,
    and `from ..db import audit` (anything imported OUT OF a forbidden module)."""
    hits: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            # `from ..db import audit` — the MODULE itself is forbidden
            if mod and _last(mod) in _FORBIDDEN_MODULES:
                hits.add(f"from {'.' * node.level}{mod} import ...")
            # `from .. import db` / `from ..memory import store` — the imported name is a
            # submodule; its last component is what gets bound regardless of the alias.
            for a in node.names:
                if _last(a.name) in _FORBIDDEN_MODULES:
                    hits.add(f"from {'.' * node.level}{mod} import {a.name}")
        elif isinstance(node, ast.Import):
            for a in node.names:               # `import ken.db as foo` — bind is aliased, module is not
                if _last(a.name) in _FORBIDDEN_MODULES:
                    hits.add(f"import {a.name}")
    return hits


def test_no_agent_file_imports_a_privileged_module():
    offenders = {p.name: sorted(hits) for p in AGENTS_DIR.glob("*.py")
                 if (hits := _forbidden_imports(p.read_text()))}
    assert offenders == {}, f"agents must not import store/gate/db: {offenders}"


def test_the_guard_catches_a_planted_import():
    # every form the guard must flag — a regression here would let the seam re-open
    assert _forbidden_imports("from ..memory import store\n")
    assert _forbidden_imports("from ..consent import gate\n")
    assert _forbidden_imports("from .. import db\n")
    assert _forbidden_imports("import ken.db as _db\n")
    assert _forbidden_imports("import ken.memory.store\n")
    assert _forbidden_imports("from ..memory.store import open_tasks\n")


def test_the_guard_is_robust_to_aliasing():
    # ALIASING must not smuggle a privileged module past the guard — the rawest one,
    # db, has no unique path token, so these forms previously slipped through and handed
    # out raw SQL that bypasses gate/judge/scope. They must ALL fail the guard now.
    assert _forbidden_imports("from .. import db as d\n")            # relative db, aliased
    assert _forbidden_imports("import ken.db as foo\n")             # absolute db, aliased
    assert _forbidden_imports("from ..db import audit\n")           # anything OUT OF db
    assert _forbidden_imports("from ..memory import store as s\n")  # store, aliased
    assert _forbidden_imports("from ..consent import gate as g\n")  # gate, aliased
    assert _forbidden_imports("import ken.memory.store as m\n")     # store submodule, aliased


def test_the_guard_passes_clean_agent_imports():
    # the capabilities + ordinary helpers an agent MAY import must not trip the guard
    assert _forbidden_imports(
        "from ..contracts import Evidence\n"
        "from ..tools.base import Tool\n"
        "from ..util import norm, slug, when_label\n"
        "from .base import Advice, Agent, Proposal, Slice\n") == set()
