"""User config — private, read at runtime, never committed.

`config.yaml` beside the database (~/.ken). A missing file is fine. Tying the path
to the db dir also isolates tests: a temp-db test reads that dir's config, not yours.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from . import db

_SAFE_BUCKET = re.compile(r"^[A-Za-z0-9_-]+$")   # no '.', '/', '..' — a bucket name is never a path


def write_private(p: Path, text: str) -> None:
    """Write a file only its owner can read (0600). The config holds API keys and tokens."""
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)                       # O_CREAT's mode does not touch an existing file
    with os.fdopen(fd, "w") as f:
        f.write(text)


def config_path(path: str | None = None) -> Path:
    if path:
        return Path(os.path.expanduser(path))
    return db.db_path().parent / "config.yaml"


def _read_yaml_dict(p: Path) -> dict[str, Any]:
    """A yaml file as a dict, or {} if absent, empty, or not a mapping."""
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    return data if isinstance(data, dict) else {}


def load(path: str | None = None) -> dict[str, Any]:
    """The config as a dict, or {} if the file is absent or empty."""
    return _read_yaml_dict(config_path(path))


def section(name: str, path: str | None = None) -> dict[str, Any]:
    """One top-level section as a dict, or {} if absent."""
    value = load(path).get(name)
    return value if isinstance(value, dict) else {}


def has_api_key(path: str | None = None) -> bool:
    """True when a real OpenAI key is set under `llm.api_key`.

    Mirrors `llm.build_llm`: an empty value or the `YOUR_...` example placeholder
    counts as *no* key, so a freshly copied config.example.yaml still onboards."""
    llm = section("llm", path)
    if str(llm.get("provider") or "").replace("_", "-").lower() == "claude-code":
        return True                                   # the user's Claude Code login is the key
    key = llm.get("api_key")
    return bool(key) and not str(key).startswith("YOUR_")


def set_section_value(name: str, key: str, value: Any, path: str | None = None) -> Path:
    """Set one leaf `<name>.<key> = value`, merging into any existing config.

    A pure config mutation (no subprocess, no I/O beyond the one file), so it is
    unit-testable. It loads the whole document, sets just this one leaf, and dumps
    it back — every other key the user has is preserved. Creates `~/.ken/` if the
    file's parent directory does not exist yet."""
    p = config_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load(path)
    sect = data.get(name)
    if not isinstance(sect, dict):
        sect = {}
    sect[key] = value
    data[name] = sect
    write_private(p, yaml.safe_dump(data, sort_keys=False))
    return p


# ── agent-writable state buckets (~/.ken/state/<bucket>.yaml) ────────────────
# A named config bucket, SEPARATE from config.yaml — so nothing agent-writable ever
# touches your secrets. A connector reads its bucket each poll (live, no reload); an
# agent reads/writes it through the scoped `ctx.config` capability. Plain yaml on
# purpose: `cat ~/.ken/state/slack.yaml` to inspect, hand-edit to override.

def state_dir() -> Path:
    return config_path().parent / "state"


def _bucket_path(name: str) -> Path:
    """The file for a bucket. A name is a NAME, never a path — reject anything that could
    traverse out of state/ (e.g. '../config' would otherwise resolve to your secrets file).
    `fullmatch` anchors the whole string, so a trailing newline can't slip past `$`."""
    if not _SAFE_BUCKET.fullmatch(name):
        raise ValueError(f"invalid config bucket name {name!r} (allowed: letters, digits, '-', '_')")
    return state_dir() / f"{name}.yaml"


def read_bucket(name: str) -> dict[str, Any]:
    """One state bucket as a dict, or {} if it does not exist yet."""
    return _read_yaml_dict(_bucket_path(name))


def write_bucket(name: str, data: dict[str, Any]) -> Path:
    """Replace a state bucket, atomically (temp + rename) so a concurrent reader never sees
    a torn file. A per-process temp name avoids two writers colliding, and is cleaned up if
    the write fails. The caller owns audit (the ctx.config capability logs each write)."""
    p = _bucket_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{name}.{os.getpid()}.tmp")
    try:
        write_private(tmp, yaml.safe_dump(data, sort_keys=False))
        tmp.replace(p)
    except Exception:
        tmp.unlink(missing_ok=True)                # no orphaned dotfile on a failed write
        raise
    return p
