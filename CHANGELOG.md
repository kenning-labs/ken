# Changelog

## 0.1.0 — 2026-10-31

The first public release.

- Ken is a local context engine behind an MCP server. It keeps your tasks, people,
  projects, preferences and what happened in one SQLite file under `~/.ken/`.
- Any MCP client (Claude Code, Claude Desktop, others) reads that context and writes
  back through typed tools. Every write goes through a consent gate: the engine judges
  it, and anything it is not sure of becomes a card you approve or reject.
- A maintainer pass keeps the store current: it asks "done?" and "still real?" about
  stale tasks.
- Agents extend it: any MCP client works, and one `.py` file in `~/.ken/agents/` runs inside
  the engine.
- Setup in two commands: `ken init` asks for a model (an OpenAI key, your Claude Code login,
  another OpenAI-compatible API, or none for now with `--no-model`), offers to add Ken to
  Claude Code, and offers to run the engine in the background (`ken service`: launchd on
  macOS, a systemd user service on Linux).
- A write sent while the engine is stopped answers `queued` and applies when the engine starts.
