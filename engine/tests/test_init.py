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


def test_add_to_claude_runs_mcp_add_and_tolerates_already_exists(home, tmp_path, monkeypatch, capsys):
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "MCP server ken already exists in user config")
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    _fake_venv(tmp_path, monkeypatch)
    assert _init(monkeypatch, "--claude-code", "--add-to-claude", claude="/bin/claude") == 0
    cmd = calls[0]
    assert cmd[:6] == ["/bin/claude", "mcp", "add", "--scope", "user", "ken"]
    assert f"KEN_HOME={home}" in cmd and cmd[-1].endswith("ken-mcp")
    assert "already has" in capsys.readouterr().out


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
