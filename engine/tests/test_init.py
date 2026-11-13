"""`ken init` — the first-run setup. Never calls the real `claude`."""
from __future__ import annotations

import stat
import subprocess
import sys

import pytest
import yaml

from ken import __main__ as cli
from ken import db


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("KEN_HOME", str(h))
    monkeypatch.setattr(db, "_DB_PATH", None)
    return h


@pytest.fixture(autouse=True)
def no_real_service(tmp_path, monkeypatch):
    """Enter at the service prompt means yes: keep that install in a tmp HOME, never launchctl."""
    from ken import service
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setattr(service, "_run", lambda cmd: subprocess.CompletedProcess(cmd, 0, "", ""))


def _init(monkeypatch, *flags, claude=None):
    """Run `ken init FLAGS` with `claude` on PATH or not; return the exit code."""
    monkeypatch.setattr(cli.shutil, "which", lambda name: claude if name == "claude" else None)
    monkeypatch.setattr(sys, "argv", ["ken", "init", *flags])
    with pytest.raises(SystemExit) as e:
        cli.main()
    return e.value.code


def test_claude_code_writes_private_config_and_db(home, monkeypatch, capsys):
    assert _init(monkeypatch, "--claude-code", "--no-claude", claude="/bin/claude") == 0
    cfg = home / "config.yaml"
    assert yaml.safe_load(cfg.read_text())["llm"] == {"provider": "claude-code"}
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert (home / "ken.db").exists()
    assert "set up my world" in capsys.readouterr().out


def test_api_key_is_written_but_never_printed(home, monkeypatch, capsys):
    assert _init(monkeypatch, "--api-key", "sk-secret-123", "--no-claude") == 0
    assert yaml.safe_load((home / "config.yaml").read_text())["llm"]["api_key"] == "sk-secret-123"
    out = capsys.readouterr()
    assert "sk-secret-123" not in out.out + out.err


def test_existing_config_is_not_overwritten(home, monkeypatch, capsys):
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text("llm:\n  api_key: mine\n")
    assert _init(monkeypatch, "--claude-code", "--no-claude") == 0
    assert (home / "config.yaml").read_text() == "llm:\n  api_key: mine\n"
    assert "already exists" in capsys.readouterr().out


def test_no_tty_and_no_flag_exits_with_the_flag_hint(home, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert _init(monkeypatch, "--no-claude") != 0
    assert "--claude-code" in capsys.readouterr().err
    assert not (home / "config.yaml").exists()


def test_no_claude_never_runs_claude(home, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("claude must not run")
    monkeypatch.setattr(cli.subprocess, "run", boom)
    assert _init(monkeypatch, "--claude-code", "--no-claude", claude="/bin/claude") == 0


def _fake_venv(tmp_path, monkeypatch):
    """A ken-mcp beside a fake sys.executable, as in a real venv's bin/."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ken-mcp").touch()
    monkeypatch.setattr(sys, "executable", str(bindir / "python"))


def _fake_claude(monkeypatch, servers):
    """A fake `claude mcp add/get` over a dict name -> `claude mcp get` body. Returns the calls."""
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        verb, name = cmd[2], cmd[5] if cmd[2] == "add" else cmd[3]
        if verb == "add":
            if name in servers:
                return subprocess.CompletedProcess(cmd, 1, "", f"MCP server {name} already exists in user config")
            servers[name] = f"  Command: {cmd[-1]}"
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if name in servers:
            return subprocess.CompletedProcess(cmd, 0, f"{name}:\n{servers[name]}\n", "")
        return subprocess.CompletedProcess(cmd, 1, "", f'No MCP server named "{name}".')
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls


def test_add_to_claude_runs_mcp_add(home, tmp_path, monkeypatch, capsys):
    calls = _fake_claude(monkeypatch, {})
    _fake_venv(tmp_path, monkeypatch)
    assert _init(monkeypatch, "--claude-code", "--add-to-claude", claude="/bin/claude") == 0
    cmd = calls[0]
    assert cmd[:6] == ["/bin/claude", "mcp", "add", "--scope", "user", "ken"]
    assert f"KEN_HOME={home}" in cmd and cmd[-1].endswith("ken-mcp")
    out = capsys.readouterr().out
    assert "as the `ken` MCP server" in out and "set up my world" in out


def test_a_cloud_ken_is_kept_and_this_engine_becomes_ken_local(home, tmp_path, monkeypatch, capsys):
    secret = "https://api.kenninglabs.com/u/u-1/t/SECRET/mcp"
    calls = _fake_claude(monkeypatch, {"ken": f"  Type: http\n  URL: {secret}\n  Headers:\n    Authorization: Bearer TOK"})
    _fake_venv(tmp_path, monkeypatch)
    assert _init(monkeypatch, "--claude-code", "--add-to-claude", claude="/bin/claude") == 0
    assert not any(c[2] == "remove" for c in calls)                  # the cloud entry is never removed
    out = capsys.readouterr().out
    assert "Ken Cloud (api.kenninglabs.com), not this engine" in out
    assert "as the `ken-local` MCP server" in out and "`ken-local` server" in out
    assert "SECRET" not in out and "TOK" not in out                  # never the URL path or headers


def test_an_existing_entry_for_this_engine_counts_as_connected(home, tmp_path, monkeypatch, capsys):
    _fake_venv(tmp_path, monkeypatch)
    _fake_claude(monkeypatch, {"ken": f"  Command: {tmp_path / 'bin' / 'ken-mcp'}"})
    assert _init(monkeypatch, "--claude-code", "--add-to-claude", claude="/bin/claude") == 0
    out = capsys.readouterr().out
    assert "already reaches this engine as `ken`" in out and "set up my world" in out


def test_not_connected_prints_the_connect_command_in_next(home, monkeypatch, capsys):
    assert _init(monkeypatch, "--claude-code", "--no-claude", "--no-service", claude="/bin/claude") == 0
    out = capsys.readouterr().out
    assert "Connect Claude Code: claude mcp add --scope user ken" in out


def test_version(monkeypatch, capsys):
    from ken import __version__
    monkeypatch.setattr(sys, "argv", ["ken", "--version"])
    with pytest.raises(SystemExit):
        cli.main()
    assert capsys.readouterr().out.strip() == f"ken-engine {__version__}"


def test_failed_registration_does_not_fail_init(home, tmp_path, monkeypatch, capsys):
    _fake_venv(tmp_path, monkeypatch)

    def fake_run(cmd, **k):
        raise OSError("nope")
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert _init(monkeypatch, "--claude-code", "--add-to-claude", claude="/bin/claude") == 0
    assert "Run by hand: claude mcp add --scope user ken" in capsys.readouterr().out


def test_pressing_enter_picks_the_openai_key(home, monkeypatch, capsys):
    import getpass
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")          # Enter at every prompt
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "sk-typed-456")
    assert _init(monkeypatch, "--no-claude", claude="/usr/bin/claude") == 0
    llm = yaml.safe_load((home / "config.yaml").read_text())["llm"]
    assert llm == {"provider": "openai", "api_key": "sk-typed-456"}
    assert "sk-typed-456" not in capsys.readouterr().out


def test_claude_code_without_claude_on_path_writes_nothing(home, monkeypatch, capsys):
    assert _init(monkeypatch, "--claude-code", "--no-claude") != 0
    assert "not on PATH" in capsys.readouterr().err
    assert not (home / "config.yaml").exists()


def test_no_model_writes_a_working_config_and_the_db(home, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert _init(monkeypatch, "--no-model", "--no-claude") == 0
    assert "llm" not in (yaml.safe_load((home / "config.yaml").read_text()) or {})
    assert (home / "ken.db").exists()


def test_menu_choice_no_model(home, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "4")
    assert _init(monkeypatch, "--no-claude") == 0
    assert (home / "config.yaml").exists() and (home / "ken.db").exists()


def test_push_dry_run_without_url_names_the_missing_setting(home, monkeypatch, capsys):
    from ken import mcp_server

    async def one_card():
        return "1 open card"
    monkeypatch.setattr(mcp_server, "pending_digest", one_card)
    assert cli._cmd_push(dry_run=True) == 0
    assert "push.url is not set" in capsys.readouterr().out
