"""Tool base — an internal capability an agent can call while it reasons.
(AGENTS.md)

V1 tools read our own memory (a trend, a series, a day's detail) — deterministic,
offline, no fabrication. A tool declares a JSON-schema for its args (sent to the
model so it calls the tool correctly); the LLM tool loop runs `fn` and feeds the
result back. A bad call is returned to the model as an error, never raised.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict            # JSON schema (object) for the arguments
    fn: Callable                # sync or async; returns a str or JSON-able value
