"""`ken service install|uninstall|status` — run the engine in the background, started at
login: a launchd agent on macOS, a systemd user unit on Linux.

The service runs the same `ken run` you would type, pinned to this KEN_HOME. Two engines
never tick one database: the engine lock (db.acquire_engine_lock) makes a second engine
exit at once, and install refuses while a terminal engine holds that lock.
"""
from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import db

LABEL = "com.kenninglabs.ken"
UNIT = "ken.service"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """The one door to launchctl/systemctl (tests replace it). Never raises."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def _command() -> list[str]:
    # The `ken` script on PATH, not sys.executable: `uv tool upgrade` (or a reinstall on a
    # new Python) can rebuild the tool venv and move its python, but keeps the ~/.local/bin/ken
    # shim the user types. Not resolved, so the plist follows the shim, not today's target.
    # No `ken` on PATH (a dev venv run by path) → this python, which is all we can name.
    ken = shutil.which("ken")
    return [os.path.abspath(ken), "run"] if ken else [sys.executable, "-m", "ken", "run"]


def _home() -> Path:
    return db.home_dir().resolve()


def _env() -> dict[str, str]:
    """KEN_HOME pinned (absolute), and a PATH with `claude` on it: a login service gets a
    bare PATH, and the claude-code provider runs `claude -p`."""
    dirs = ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    claude = shutil.which("claude")
    if claude:
        dirs.insert(0, os.path.dirname(os.path.abspath(claude)))
    return {"KEN_HOME": str(_home()), "PATH": ":".join(dirs),
            "PYTHONUNBUFFERED": "1"}            # the log shows each line as it happens


def _log() -> Path:
    return _home() / "logs" / "engine.log"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / UNIT


def _err(msg: str) -> int:
    print(f"ken service: {msg}", file=sys.stderr)
    return 2


def _check_install(service_file: Path) -> int:
    """0 = go ahead. Refuse before `ken init`, and while a terminal engine runs: with no
    service file yet, a held engine lock can only be an engine someone started by hand."""
    if not (db.home_dir() / "config.yaml").exists():
        return _err(f"Ken is not set up in {db.home_dir()} — run `ken init` first.")
    if not service_file.exists() and db.engine_is_running():
        return _err("an engine already runs (in a terminal?). Stop it (Ctrl-C), then run "
                    "`ken service install` again.")
    return 0


def _mac_install() -> int:
    plist = plist_path()
    if rc := _check_install(plist):
        return rc
    _log().parent.mkdir(parents=True, exist_ok=True)
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_bytes(plistlib.dumps({
        "Label": LABEL, "ProgramArguments": _command(), "EnvironmentVariables": _env(),
        "RunAtLoad": True, "KeepAlive": True,
        "StandardOutPath": str(_log()), "StandardErrorPath": str(_log())}))
    domain = f"gui/{os.getuid()}"
    _run(["launchctl", "bootout", f"{domain}/{LABEL}"])        # a reinstall: "not loaded" is fine
    r = _run(["launchctl", "bootstrap", domain, str(plist)])
    if r.returncode != 0:
        return _err(f"launchctl bootstrap failed: {(r.stderr or r.stdout).strip() or r.returncode}")
    print(f"ken service: installed {plist} — the engine runs now and at every login.\n"
          f"  log: {_log()}")
    return 0


def _mac_uninstall() -> int:
    _run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"])
    plist_path().unlink(missing_ok=True)
    print("ken service: removed — run `ken` in a terminal to start the engine by hand.")
    return 0


def _mac_status() -> int:
    if not plist_path().exists():
        print("ken service: not installed (`ken service install`)")
        return 1
    r = _run(["launchctl", "list", LABEL])
    pid = re.search(r'"PID" = (\d+);', r.stdout)
    state = (f"running (pid {pid[1]})" if pid else "loaded, not running") if r.returncode == 0 else "not loaded"
    print(f"ken service: installed, {state}")
    lines = _log().read_text(errors="replace").splitlines() if _log().exists() else []
    if lines:
        print(f"  last log: {lines[-1]}")
    return 0


def _linux_install() -> int:
    unit = unit_path()
    if rc := _check_install(unit):
        return rc
    unit.parent.mkdir(parents=True, exist_ok=True)
    env = "".join(f"Environment={k}={v}\n" for k, v in _env().items())
    unit.write_text(f"[Unit]\nDescription=Ken engine\n\n[Service]\nExecStart={' '.join(_command())}\n"
                    f"{env}Restart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n")
    for cmd in (["systemctl", "--user", "daemon-reload"], ["systemctl", "--user", "enable", "--now", UNIT]):
        r = _run(cmd)
        if r.returncode != 0:
            return _err(f"`{' '.join(cmd)}` failed: {(r.stderr or r.stdout).strip() or r.returncode}")
    print(f"ken service: installed {unit} — the engine runs now and at every login.\n"
          f"  logs: journalctl --user -u ken\n"
          f"  on a server, to keep it running after you log out: loginctl enable-linger $USER")
    return 0


def _linux_uninstall() -> int:
    _run(["systemctl", "--user", "disable", "--now", UNIT])
    unit_path().unlink(missing_ok=True)
    _run(["systemctl", "--user", "daemon-reload"])
    print("ken service: removed — run `ken` in a terminal to start the engine by hand.")
    return 0


def _linux_status() -> int:
    if not unit_path().exists():
        print("ken service: not installed (`ken service install`)")
        return 1
    r = _run(["systemctl", "--user", "is-active", UNIT])
    print(f"ken service: installed, {r.stdout.strip() or 'unknown'} (logs: journalctl --user -u ken)")
    return 0


_ACTIONS = {
    "darwin": {"install": _mac_install, "uninstall": _mac_uninstall, "status": _mac_status},
    "linux": {"install": _linux_install, "uninstall": _linux_uninstall, "status": _linux_status},
}


def main(action: str) -> int:
    actions = _ACTIONS.get(sys.platform)
    if not actions:
        return _err(f"a background service is not supported on {sys.platform}. "
                    "Run `ken` in a terminal and keep it open.")
    return actions[action]()
