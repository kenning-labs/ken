# Contributing to Ken

Thanks for your interest in Ken — a local context engine behind an MCP server
that you run on your own machine and extend. Contributions are welcome: bug
reports, fixes, docs, and new agents.

## License

Ken is licensed under the **GNU Affero General Public License v3.0** (see
[`LICENSE`](LICENSE)). By contributing, you agree that your contributions are
licensed under the same terms, as set out in the [CLA](CLA.md).

## Contributor License Agreement (CLA)

Before we can merge your first pull request, you accept our
[Contributor License Agreement](CLA.md). The CLA Assistant bot asks you in the pull
request; it takes one click and covers all your future contributions.

The CLA keeps Ken open under the AGPL-3.0 and also lets Kenning Labs offer it under
other terms (for example to companies that cannot use the AGPL). You keep the copyright
in your work.

## Working on the code

The engine lives in `engine/` and needs Python 3.11 or newer (a stock Mac ships 3.9).

```bash
cd engine
python3.12 -m venv .venv    # any python3.11+ works && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest   # run the tests
.venv/bin/python -m ruff check .
```

- Keep changes focused; one concern per pull request.
- Add or update tests for behavior you change.
- Run the tests and the linter before opening a PR.

## Writing an agent

Ken is built to be extended. There is no connector layer: an agent fetches its own
data through tools.

- An agent inside the engine is one `.py` file dropped into `~/.ken/agents/`; Ken
  discovers it on start. The contract is [AGENTS.md](AGENTS.md), and
  `engine/src/ken/agents/assistant.py` (kip) is the full worked example.
- An agent outside the engine is any MCP client; it needs only the MCP tools.

## Reporting bugs

Open an issue with what you did, what you expected, and what happened. Include
your OS and Python version. For anything security-sensitive, see
[`SECURITY.md`](SECURITY.md) instead of filing a public issue.
