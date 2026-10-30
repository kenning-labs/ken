# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report a vulnerability privately through GitHub: open the repository's
**Security** tab and choose **Report a vulnerability** (GitHub private advisory).
This keeps the report confidential while we work on a fix.

Include, as far as you can:

- what the issue is and the impact you see,
- the steps or a minimal case to reproduce it,
- the version or commit you were on, and your OS.

We aim to acknowledge a report within a few days and to keep you updated as we
investigate. Once a fix ships, we're glad to credit you unless you prefer to
stay anonymous.

## Scope

Ken runs on your own machine, on your own API key, and stores its data locally
in `~/.ken/`. It talks only to the LLM provider you configure. Ken has no connector layer:
your agents and MCP clients write to it. Reports that concern this local-first
surface — the engine, the `ken` / `ken-mcp` commands, the MCP server and its
tokens, the agent contract, credential handling — are in scope.
