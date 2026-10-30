"""The consent gate — the single applier of agent-proposed writes. (ARCHITECTURE.md)

Toll by target-stakes: fast-track (apply + log, no interrupt) vs ask (interrupt).
Stakes is a declared property of the target kind, not the proposer's choice.
No batching in V1. TODO(engine)."""
