"""Drop-file discovery — the plug-and-play seam.

An agent is *registered* by simply existing as a file with an `Agent` subclass, in
one of two roots:

  * the built-in package ``ken.agents`` — ships with the product (Kip, and later
    the reference `example`); a fork/downstream bundles its own here.
  * ``~/.ken/agents`` — the self-hoster's personal drop-ins; never touches source.

``discover_agents`` imports each module in those roots, finds the concrete `Agent`
subclasses **defined there** (not imported), builds each via the caller's `build`
(which injects its declared deps), and registers it on the engine. A file that
fails to import, or a class that fails to register, is collected as a skip — never
raised — so one bad drop-in can't take the companion down. The caller audits the
skips once the DB is up (discovery itself is synchronous, called from build_engine).
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
from collections.abc import Callable
from pathlib import Path

from .agents.base import Agent


def _subclasses(module, base: type) -> list[type]:
    """Concrete `base` subclasses DEFINED in this module (imported ones are skipped
    via ``__module__``, so importing the base class itself never re-registers it)."""
    out = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (issubclass(obj, base) and obj is not base
                and not inspect.isabstract(obj)
                and obj.__module__ == module.__name__):
            out.append(obj)
    return out


def _module_importers(package, extra_dirs) -> list[tuple[str, Callable[[], object]]]:
    """A ``(source_label, import_fn)`` per candidate module — built-ins first, then the
    drop-in dirs. Enumerating the candidates is cheap and safe; the risky part (the
    actual import) is deferred to ``import_fn`` so the caller can isolate EACH file.
    So one un-importable module never hides the others (or drops Kip)."""
    out: list[tuple[str, Callable[[], object]]] = []
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        leaf = info.name.rsplit(".", 1)[-1]
        if leaf.startswith("_") or leaf in ("base",):    # skip dunders + the base class
            continue
        out.append((info.name, lambda n=info.name: importlib.import_module(n)))
    for d in extra_dirs:
        path = Path(d)
        if not path.is_dir():
            continue
        for f in sorted(path.glob("*.py")):
            if f.stem.startswith("_"):
                continue
            def _imp(f=f):
                spec = importlib.util.spec_from_file_location(f"ken_dropin_{f.stem}", f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)             # may raise — isolated per file
                return mod
            out.append((str(f), _imp))
    return out


def discover_agents(eng, *, build: Callable[[type], Agent],
                    extra_dirs=()) -> list[tuple[str, str]]:
    """Find, build, and register every agent in the built-in package + `extra_dirs`.
    Returns the ``(source, error)`` skips (empty on a clean run) for the caller to
    audit. Isolation is PER FILE and PER CLASS: one bad module or agent is recorded
    and skipped — it never aborts the rest, and a broken built-in never drops Kip.
    ``build`` may return ``None`` to mean "intentionally OFF" (a config-disabled
    built-in) — that agent is silently skipped, not recorded as a failure."""
    from . import agents as agents_pkg

    skipped: list[tuple[str, str]] = []
    for source, import_fn in _module_importers(agents_pkg, extra_dirs):
        try:
            mod = import_fn()
        except Exception as exc:  # noqa: BLE001 — one bad file, skipped; the rest load
            skipped.append((source, repr(exc)))
            continue
        for cls in _subclasses(mod, Agent):
            try:
                agent = build(cls)                   # None → intentionally OFF (disabled)
                if agent is not None:
                    eng.register_agent(agent)
            except Exception as exc:  # noqa: BLE001 — one bad agent, skipped
                skipped.append((f"{source}::{cls.__name__}", repr(exc)))
    return skipped
