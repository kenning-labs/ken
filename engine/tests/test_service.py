"""`ken service` — the background engine. Writes into a tmp HOME and never runs the
real launchctl/systemctl: every external command goes through `service._run`."""
from __future__ import annotations

import plistlib
import subprocess
import sys

import pytest

from ken import __main__ as cli
from ken import db, service


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tmp HOME + KEN_HOME with a config (Ken is set up), and a recording fake runner."""
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    ken_home = tmp_path / "kenhome"
    ken_home.mkdir(mode=0o700)
    (ken_home / "config.yaml").write_text("llm: {}\n")
    monkeypatch.setenv("KEN_HOME", str(ken_home))
    monkeypatch.setattr(db, "_DB_PATH", None)
    monkeypatch.setattr(service.shutil, "which",
                        lambda name: {"claude": "/opt/claude/bin/claude", "ken": "/opt/uv/bin/ken"}.get(name))
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(service, "_run", fake_run)
    return {"home": tmp_path / "user", "ken": ken_home, "calls": calls}


def _mac(monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "darwin")


def _linux(monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "linux")


def test_mac_install_writes_the_plist_and_bootstraps(env, monkeypatch):
    _mac(monkeypatch)
    assert service.main("install") == 0
    plist = env["home"] / "Library/LaunchAgents/com.kenninglabs.ken.plist"
    p = plistlib.loads(plist.read_bytes())
    assert p["Label"] == "com.kenninglabs.ken"
    assert p["ProgramArguments"] == ["/opt/uv/bin/ken", "run"]
    assert p["RunAtLoad"] is True and p["KeepAlive"] is True
    assert p["EnvironmentVariables"]["KEN_HOME"] == str(env["ken"].resolve())
    assert "/opt/claude/bin" in p["EnvironmentVariables"]["PATH"].split(":")
    log = str(env["ken"].resolve() / "logs" / "engine.log")
    assert p["StandardOutPath"] == log and p["StandardErrorPath"] == log
    assert (env["ken"] / "logs").is_dir()
    uid = service.os.getuid()
    assert env["calls"][-1] == ["launchctl", "bootstrap", f"gui/{uid}", str(plist)]
    assert ["launchctl", "bootout", f"gui/{uid}/com.kenninglabs.ken"] in env["calls"]


def test_without_a_ken_script_it_falls_back_to_this_python(env, monkeypatch):
    _mac(monkeypatch)
    monkeypatch.setattr(service.shutil, "which", lambda name: None)
    assert service.main("install") == 0
    p = plistlib.loads((env["home"] / "Library/LaunchAgents/com.kenninglabs.ken.plist").read_bytes())
    assert p["ProgramArguments"] == [sys.executable, "-m", "ken", "run"]


def test_mac_uninstall_boots_out_and_removes_the_plist(env, monkeypatch):
    _mac(monkeypatch)
    service.main("install")
    env["calls"].clear()
    assert service.main("uninstall") == 0
    assert not (env["home"] / "Library/LaunchAgents/com.kenninglabs.ken.plist").exists()
    assert env["calls"] == [["launchctl", "bootout", f"gui/{service.os.getuid()}/com.kenninglabs.ken"]]


def test_mac_status_shows_the_pid_and_the_last_log_line(env, monkeypatch, capsys):
    _mac(monkeypatch)
    service.main("install")
    (env["ken"] / "logs" / "engine.log").write_text("ken: data in x\nken engine: ticking every 1.0s\n")
    monkeypatch.setattr(service, "_run", lambda cmd: subprocess.CompletedProcess(
        cmd, 0, '{\n\t"PID" = 4242;\n\t"Label" = "com.kenninglabs.ken";\n};\n', ""))
    assert service.main("status") == 0
    out = capsys.readouterr().out
    assert "4242" in out and "ticking every 1.0s" in out


def test_linux_install_writes_the_unit_and_enables_it(env, monkeypatch, capsys):
    _linux(monkeypatch)
    assert service.main("install") == 0
    unit = (env["home"] / ".config/systemd/user/ken.service").read_text()
    assert "ExecStart=/opt/uv/bin/ken run" in unit
    assert f"Environment=KEN_HOME={env['ken'].resolve()}" in unit
    assert "Restart=on-failure" in unit and "RestartSec=5" in unit and "WantedBy=default.target" in unit
    assert env["calls"] == [["systemctl", "--user", "daemon-reload"],
                            ["systemctl", "--user", "enable", "--now", "ken.service"]]
    assert "loginctl enable-linger" in capsys.readouterr().out


def test_linux_uninstall_disables_and_removes_the_unit(env, monkeypatch):
    _linux(monkeypatch)
    service.main("install")
    env["calls"].clear()
    assert service.main("uninstall") == 0
    assert not (env["home"] / ".config/systemd/user/ken.service").exists()
    assert env["calls"] == [["systemctl", "--user", "disable", "--now", "ken.service"],
                            ["systemctl", "--user", "daemon-reload"]]


def test_install_refuses_before_ken_init(env, monkeypatch, capsys):
    _mac(monkeypatch)
    (env["ken"] / "config.yaml").unlink()
    assert service.main("install") == 2
    assert "ken init" in capsys.readouterr().err
    assert env["calls"] == []


def test_install_refuses_while_a_terminal_engine_runs(env, monkeypatch, capsys):
    _mac(monkeypatch)
    monkeypatch.setattr(service.db, "engine_is_running", lambda: True)
    assert service.main("install") == 2
    assert "stop" in capsys.readouterr().err.lower()
    assert env["calls"] == []


def test_unsupported_os(env, monkeypatch, capsys):
    monkeypatch.setattr(service.sys, "platform", "win32")
    assert service.main("install") == 2
    assert "not supported" in capsys.readouterr().err
    assert env["calls"] == []


def test_service_parses():
    assert cli.build_parser().parse_args(["service", "status"]).action == "status"


# ── `ken init` offers the service as its last step ───────────────────────────

def _init(monkeypatch, *flags):
    monkeypatch.setattr(sys, "argv", ["ken", "init", "--no-model", "--no-claude", *flags])
    with pytest.raises(SystemExit) as e:
        cli.main()
    return e.value.code


def test_init_service_flag_installs_without_asking(env, monkeypatch, capsys):
    _mac(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("must not ask"))
    assert _init(monkeypatch, "--service") == 0
    assert (env["home"] / "Library/LaunchAgents/com.kenninglabs.ken.plist").exists()
    assert "keep it running" not in capsys.readouterr().out


def test_init_no_service_flag_skips(env, monkeypatch, capsys):
    _mac(monkeypatch)
    assert _init(monkeypatch, "--no-service") == 0
    assert env["calls"] == []
    assert "keep it running" in capsys.readouterr().out


def test_init_without_a_tty_prints_the_command(env, monkeypatch, capsys):
    _mac(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert _init(monkeypatch) == 0
    assert env["calls"] == []
    assert "ken service install" in capsys.readouterr().out


def test_init_enter_on_a_tty_installs(env, monkeypatch):
    _mac(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert _init(monkeypatch) == 0
    assert env["calls"][-1][:2] == ["launchctl", "bootstrap"]
