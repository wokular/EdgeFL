"""Launch, track, and tear down the Python servers.

The harness owns each node_server / aggregator as a subprocess.Popen handle, giving crash
detection via .poll(), per-node stdout+stderr captured to a log file, and clean teardown.

We parse the generated .env with dotenv_values() and inject it via Popen(env=...) rather
than depending on a dotenv CLI on PATH. That's what dotenv run does internally. The
servers' own bare load_dotenv() then finds no .env in cwd and no-ops, leaving our injected
vars in place.

EdgeLake master/operators and Postgres are Docker containers, handled in infra.py; this
module is only the Python servers.
"""

import os
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass

from dotenv import dotenv_values

from platform_components.lib.logger.error_handling import get_logger

logger = get_logger(__name__)


# uvicorn app targets for each server type.
NODE_APP = "platform_components.node.node_server:app"
AGGREGATOR_APP = "platform_components.aggregator.aggregator_server:app"


@dataclass
class ServerProcess:
    """One launched Python server (a node_server or the aggregator)."""

    name: str                       # logical name, eg "node1" / "aggregator"
    app: str                        # uvicorn app target
    port: int                       # uvicorn HTTP port
    env_path: str                   # generated .env driving this process
    edgefl_dir: str                 # cwd for the process (the edgefl/ package root)
    python_bin: str = "python3"
    log_path: str | None = None     # where stdout/stderr is captured

    def __post_init__(self):
        # runtime-only handles, kept out of the dataclass fields
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def start(self) -> None:
        """Launch the server. Idempotent guard: refuses to double-start."""
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError(f"{self.name} already running (pid={self._proc.pid})")

        # child env: current env + repo PYTHONPATH + the rendered .env
        child_env = {**os.environ}
        child_env["PYTHONPATH"] = self.edgefl_dir
        child_env["PYTHONUNBUFFERED"] = "1"
        rendered = dotenv_values(self.env_path)
        child_env.update({k: v for k, v in rendered.items() if v is not None})

        cmd = [
            self.python_bin, "-m", "uvicorn", self.app,
            "--host", "0.0.0.0", "--port", str(self.port),
        ]

        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._log_fh = open(self.log_path, "w")
            stdout = stderr = self._log_fh
        else:
            stdout = stderr = None

        logger.info(f"[{self.name}] launching on :{self.port} (env={os.path.basename(self.env_path)})")
        self._proc = subprocess.Popen(
            cmd,
            cwd=self.edgefl_dir,
            env=child_env,
            stdout=stdout,
            stderr=stderr,
            # new process group so teardown can signal the whole tree (uvicorn workers)
            start_new_session=True,
        )

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def is_alive(self) -> bool:
        """True if the process is still running."""
        return self._proc is not None and self._proc.poll() is None

    def returncode(self) -> int | None:
        return self._proc.poll() if self._proc else None

    def stop(self, timeout: float = 10.0) -> None:
        """Graceful SIGTERM to the process group, escalating to SIGKILL after timeout."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f"[{self.name}] did not exit on SIGTERM; sending SIGKILL")
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._proc.wait(timeout=timeout)
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
        logger.info(f"[{self.name}] stopped (rc={self.returncode()})")


def _port_holder(port: int) -> str:
    """Describe whatever is listening on a TCP port: 'pid 123 (<command>)'."""
    out = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
        capture_output=True, text=True,
    ).stdout
    pids = re.findall(r"^p(\d+)", out, re.MULTILINE)
    if not pids:
        return "an unknown process"
    cmd = subprocess.run(["ps", "-p", pids[0], "-o", "command="],
                         capture_output=True, text=True).stdout.strip()
    return f"pid {pids[0]} ({cmd[:100]})"


def busy_ports(ports: list[int]) -> dict[int, str]:
    """The subset of ports that cannot be bound, each with a description of its holder."""
    busy = {}
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                busy[port] = _port_holder(port)
    return busy


def find_stray_servers() -> list[tuple[int, int | None, str]]:
    """(pid, port, name) for every running uvicorn of the aggregator/node apps, whoever
    launched it (a harness run, an interrupted run, or a manual start)."""
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        pid_s, _, cmd = line.strip().partition(" ")
        tokens = cmd.split()
        # Only a Python process whose arguments are literally `-m uvicorn <app>`: matching
        # the text anywhere in a command line would also hit shells and editors that
        # merely contain it.
        if not tokens or not os.path.basename(tokens[0]).lower().startswith("python"):
            continue
        name = None
        for app, label in ((AGGREGATOR_APP, "aggregator"), (NODE_APP, "node")):
            if ["-m", "uvicorn", app] in [tokens[i:i + 3] for i in range(len(tokens) - 2)]:
                name = label
        if name is None or int(pid_s) == os.getpid():
            continue
        m = re.search(r"--port (\d+)", cmd)
        found.append((int(pid_s), int(m.group(1)) if m else None, name))
    return found


def stop_stray_servers(timeout: float = 10.0) -> list[tuple[int, int | None, str]]:
    """SIGTERM every aggregator/node uvicorn found, SIGKILL any that outlive the timeout.
    Returns what was found."""
    servers = find_stray_servers()
    for pid, _, _ in servers:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + timeout
    while time.time() < deadline and any(_pid_alive(p) for p, _, _ in servers):
        time.sleep(0.3)
    for pid, port, name in servers:
        if _pid_alive(pid):
            logger.warning(f"{name} (pid {pid}, port {port}) ignored SIGTERM; sending SIGKILL")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return servers


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class ServerGroup:
    """A set of ServerProcesses for one run: the aggregator (if any) plus all
    node_servers. Handles start order, bulk teardown, and crash scanning."""

    def __init__(self) -> None:
        self._servers: list[ServerProcess] = []

    def add(self, server: ServerProcess) -> ServerProcess:
        self._servers.append(server)
        return server

    @property
    def servers(self) -> list[ServerProcess]:
        return list(self._servers)

    def start_all(self, stagger_s: float = 0.3) -> None:
        """Start every server, staggered so port binding doesn't thrash."""
        for s in self._servers:
            s.start()
            if stagger_s:
                time.sleep(stagger_s)

    def dead_servers(self) -> list[ServerProcess]:
        """Servers that have exited. Empty means healthy."""
        return [s for s in self._servers if not s.is_alive()]

    def stop_all(self, timeout: float = 10.0) -> None:
        """Tear down every server, reverse order so nodes stop before the aggregator."""
        for s in reversed(self._servers):
            s.stop(timeout=timeout)
