"""AnyLog infra backend: the PRIMARY deployment path on Apple Silicon
(AnylogNetworkSetup/AnyLog-Setup.md). Drives the AnyLog-co/docker-compose repo
(branch pre-develop) instead of the EdgeLake Makefile.

Differences from the EdgeLake backend (infra.py) that make this one simpler:
  - Containers run with Docker Desktop HOST networking, so every node is
    127.0.0.1:<port> from everywhere (and Postgres shards, when operators use psql) —
    no bridge-IP discovery per boot.
  - Requires the ANYLOG_LICENSE env var. The generated node configs carry
    `LICENSE_KEY=$ANYLOG_LICENSE`, which docker compose (>= 2.24) interpolates from
    the environment when it reads the env file, so the key never lands on disk. A
    stale/expired license shows up as the master refusing to start its processes.

pre-develop layout: one `docker-makefiles/<type>/node_configs.env` per node, started with
`make up ANYLOG_TYPE=<type>` (deploy.sh regenerates the compose file on every `up`).
The harness leaves the committed `anylog-master` / `anylog-operator1` configs untouched
and generates its own `edgefl-master` and `edgefl-operator<i>` dirs from them, patching
name/ports/LEDGER_CONN and giving EVERY operator its own CLUSTER_NAME — operators sharing
a cluster replicate each other's data, which would silently break per-Node data-shard
isolation. Container names stay `master` / `operator<i>`.
"""

import json
import os
import re
import subprocess
import time

import requests

from platform_components.lib.logger.error_handling import get_logger
from .identity import NodeIdentity, master_ports, node_identities
from .infra import psql_host_port, start_postgres, stop_postgres
from .readiness import wait_edgelake_process, wait_edgelake_status

logger = get_logger(__name__)

MASTER_TYPE = "edgefl-master"


def _compose_dir(host) -> str:
    d = getattr(host, "ANYLOG_COMPOSE_DIR", None)
    if not d or not os.path.isfile(
        os.path.join(d, "docker-makefiles", "anylog-master", "node_configs.env")
    ):
        raise RuntimeError(
            "host_profile.ANYLOG_COMPOSE_DIR must point at a checkout of "
            "github.com/AnyLog-co/docker-compose (branch pre-develop, the layout with "
            "docker-makefiles/anylog-master/node_configs.env). "
            "See AnylogNetworkSetup/AnyLog-Setup.md sections 3-5."
        )
    return d


def require_license(node_count: int = 0) -> None:
    if not os.environ.get("ANYLOG_LICENSE"):
        raise RuntimeError(
            "ANYLOG_LICENSE is not set in the environment. The generated node configs "
            "reference $ANYLOG_LICENSE; export it (usually from your shell profile) "
            "before running the harness. See AnyLog-Setup.md section 4."
        )
    key = os.environ["ANYLOG_LICENSE"]
    if "{" in key and '"' not in key[key.index("{"):]:
        raise RuntimeError(
            "ANYLOG_LICENSE has lost the quotes inside its JSON part (it should look like "
            "<hex>{\"company\": \"...\", ...}). This happens when the value is wrapped in "
            "double quotes in the shell profile, so the shell eats the inner ones, and AnyLog "
            "then rejects it ('Wrong license info format'). Wrap it in single quotes instead: "
            "export ANYLOG_LICENSE='<hex>{\"company\": ...}'  then open a new shell."
        )
    _check_license_terms(key, node_count)


def _check_license_terms(key: str, node_count: int) -> None:
    """Catch an expired key or too few licensed nodes before anything starts. The hex part
    is a signature over the JSON part, so these numbers can only be changed by AnyLog."""
    try:
        terms = json.loads(key[key.index("{"):])
    except (ValueError, json.JSONDecodeError):
        raise RuntimeError(
            "ANYLOG_LICENSE has no readable JSON part after the hex signature. It should "
            "look like <hex>{\"company\": \"...\", \"expiration\": \"...\", \"max_nodes\": N}."
        )
    expiration = str(terms.get("expiration", ""))
    if expiration and expiration < time.strftime("%Y-%m-%d"):
        raise RuntimeError(f"ANYLOG_LICENSE expired on {expiration}. Ask AnyLog for a new key.")
    max_nodes = terms.get("max_nodes")
    needed = node_count + 1  # operators + master
    if node_count and isinstance(max_nodes, int) and max_nodes < needed:
        print(
            f"[infra] WARNING: ANYLOG_LICENSE allows {max_nodes} nodes but this run starts "
            f"{needed} (master + {node_count} operators). Ask AnyLog for a larger key. "
            "Do NOT edit max_nodes yourself: it breaks the signature and AnyLog rejects the "
            "whole key.",
            flush=True,
        )


def _check_license_active(rest_conn: str) -> bool:
    """Fail fast when the master booted but did not activate the license. Returns False
    when the status could not be read at all (unknown, not a failure)."""
    text = None
    for attempt in range(6):
        try:
            text = requests.get(
                f"http://{rest_conn}",
                # Without an AnyLog/curl User-Agent the REST server frames a text reply as
                # JSON and sends an empty chunked body ('Response ended prematurely').
                headers={"command": "get license", "User-Agent": "AnyLog/1.23"},
                timeout=20,
            ).text
            break
        except requests.RequestException as e:
            logger.warning(f"could not read license status from master (try {attempt + 1}/6): {e}")
            time.sleep(5)
    if text is None:
        return False
    if "missing" in text.lower():
        raise RuntimeError(
            "AnyLog master started but did not activate the license ('Activation license "
            "is missing'); network-wide queries will fail. The usual cause is an "
            "ANYLOG_LICENSE whose JSON part was edited (e.g. max_nodes), which invalidates "
            "the signature: the boot log shows 'Wrong license key'. Use the key exactly as "
            "AnyLog issued it. To check by hand: "
            "curl -X POST http://" + rest_conn + " -H \"command: set license where "
            "activation_key=$ANYLOG_LICENSE\""
        )
    return True


def _run(cmd: list[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    logger.info(f"$ {' '.join(cmd)}  (cwd={cwd})")
    # No stdin and no controlling terminal: deploy.sh prompts (license acceptance, clean
    # confirmation) must fail fast here instead of hanging an unattended run.
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    if proc.returncode != 0:
        msg = f"command failed (rc={proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        if check:
            raise RuntimeError(msg)
        logger.warning(msg)
    return proc


def _make(target: str, node_type: str, compose_dir: str, host, check: bool = True):
    """`make <target> ANYLOG_TYPE=<node_type>` in the compose repo. PROMPT_LICENSE=false
    keeps the Makefile from forcing the interactive license-acceptance prompt."""
    cmd = ["make", target, f"ANYLOG_TYPE={node_type}", "PROMPT_LICENSE=false"]
    tag = getattr(host, "ANYLOG_TAG", None)
    if tag:
        cmd.append(f"TAG={tag}")
    return _run(cmd, cwd=compose_dir, check=check)


# Node config generation

def _cfg_dir(compose_dir: str, node_type: str) -> str:
    return os.path.join(compose_dir, "docker-makefiles", node_type)


def _read_template(compose_dir: str, committed_type: str) -> str:
    with open(os.path.join(_cfg_dir(compose_dir, committed_type), "node_configs.env")) as f:
        return f.read()


def operator_db_type(compose_dir: str, host) -> str:
    """DB_TYPE (psql|sqlite) for the operators: host_profile.OPERATOR_DB_TYPE if set,
    else the anylog-operator1 template's value (default psql if the key is missing)."""
    override = getattr(host, "OPERATOR_DB_TYPE", None)
    if override:
        return override.lower()
    for line in _read_template(compose_dir, "anylog-operator1").splitlines():
        if line.split("=", 1)[0].strip() == "DB_TYPE":
            return line.split("=", 1)[1].split("#", 1)[0].strip().strip('"').lower()
    return "psql"


def _patched_config(template_text: str, values: dict[str, str], disable: tuple = ()) -> str:
    """template with KEY=value lines replaced (keys the template lacks are appended) and
    any `disable` keys commented out."""
    remaining = dict(values)
    lines = []
    for line in template_text.splitlines():
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            line = f"{key}={remaining.pop(key)}"
        elif key in disable:
            line = f"# {line}  # disabled by the harness, see _operator_values"
        lines.append(line)
    lines.extend(f"{key}={value}" for key, value in remaining.items())
    return "\n".join(lines) + "\n"


# The committed configs bind TCP to the NIC's address (NIC_TYPE=eth0), which under Docker
# Desktop is the VM's 192.168.65.x — unreachable via host.docker.internal, so operators
# can't sync with the master ("Connection refused"). OVERLAY_IP replaces the local IP, and
# 127.0.0.1 is what the roy-local configs used.
_OVERLAY_IP = "127.0.0.1"


def _master_values(host) -> dict[str, str]:
    rest, tcp = master_ports()
    return {
        "NODE_NAME": '"master"',
        "CONTAINER_NAME": "master",
        "LICENSE_KEY": "$ANYLOG_LICENSE",
        "LEDGER_CONN": f"{_docker_host(host)}:{tcp}",
        "ANYLOG_SERVER_PORT": str(tcp),
        "ANYLOG_REST_PORT": str(rest),
        "OVERLAY_IP": _OVERLAY_IP,
        "TCP_BIND": "false",
        # the committed master config points at a Postgres on someone's LAN
        "DB_TYPE": "sqlite",
    }


def _operator_values(ident: NodeIdentity, logical_db: str, db_type: str, host) -> dict[str, str]:
    _, master_tcp = master_ports()
    name = ident.operator_container
    values = {
        "NODE_NAME": f'"{name}"',
        "CONTAINER_NAME": name,
        "LICENSE_KEY": "$ANYLOG_LICENSE",
        "LEDGER_CONN": f"{_docker_host(host)}:{master_tcp}",
        "ANYLOG_SERVER_PORT": str(ident.edgelake_tcp_port),
        "ANYLOG_REST_PORT": str(ident.edgelake_rest_port),
        "OVERLAY_IP": _OVERLAY_IP,
        "TCP_BIND": "false",
        # host networking: every operator needs its own broker port or they collide
        "ANYLOG_BROKER_PORT": str(ident.edgelake_rest_port + 1),
        # one cluster per operator: cluster peers replicate data, benchmarks need shards
        "CLUSTER_NAME": f'"cluster-{name}"',
        "DEFAULT_DBMS": logical_db,
        "DB_TYPE": db_type,
    }
    if db_type == "psql":
        # 127.0.0.1 is the container itself; reach the shard via its published host port
        values.update({
            "DB_IP": _docker_host(host),
            "DB_PORT": str(psql_host_port(ident.index)),
            "DB_USER": host.PSQL_USER,
            "DB_PASSWD": host.PSQL_PASSWORD,
        })
    return values


# START_DATE: the deployment-scripts' set_params.al builds the HA data-consumer start date
# as `- + $START_DATE + d`, which came out as the literal '-$START_DATEd' on the os-dev
# branch and aborted the operator's config policy before it declared the Operator
# process. Unset, the script's own default (-30d) applies, which is what START_DATE=30
# asked for anyway. Not re-verified against the pre-develop scripts, so still disabled.
_OPERATOR_DISABLE = ("START_DATE",)


def _docker_host(host) -> str:
    return getattr(host, "DOCKER_INTERNAL_HOST", "host.docker.internal")


def generate_node_configs(
    identities: list[NodeIdentity], compose_dir: str, logical_db: str, host
) -> None:
    """Create/refresh docker-makefiles/edgefl-master/ and edgefl-operator<i>/ (each with a
    node_configs.env) from the committed anylog-master / anylog-operator1 templates."""
    db_type = operator_db_type(compose_dir, host)
    outputs = {
        MASTER_TYPE: _patched_config(
            _read_template(compose_dir, "anylog-master"), _master_values(host)
        ),
    }
    operator_template = _read_template(compose_dir, "anylog-operator1")
    for ident in identities:
        outputs[operator_type(ident)] = _patched_config(
            operator_template, _operator_values(ident, logical_db, db_type, host),
            disable=_OPERATOR_DISABLE,
        )

    for node_type, text in outputs.items():
        cfg_dir = _cfg_dir(compose_dir, node_type)
        os.makedirs(cfg_dir, exist_ok=True)
        with open(os.path.join(cfg_dir, "node_configs.env"), "w") as f:
            f.write(text)
        logger.info(f"generated AnyLog config dir {node_type}")


def operator_type(ident: NodeIdentity) -> str:
    return f"edgefl-{ident.operator_container}"


# Tier orchestration (same interface as infra.py)

def _say(t0: float, msg: str) -> None:
    """Progress line for the person watching the terminal (the logger output isn't shown)."""
    print(f"[infra +{time.time() - t0:.0f}s] {msg}", flush=True)


# Each container clones AnyLog's deployment-scripts from GitHub on every boot, so a boot
# takes ~40 s on a good connection and many minutes on a slow one (one clone ran at
# ~54 KiB/s). Too short a wait fails a stack that was about to come up.
BOOT_TIMEOUT_S = 600.0


def _ensure_node_healthy(
    container: str, rest_conn: str, process_name: str, timeout_s: float = BOOT_TIMEOUT_S,
    t0: float | None = None,
) -> None:
    """Deep readiness with one self-heal: the node's deployment script is one-shot and
    sometimes loses a boot-time race (TCP bind, master reachability), leaving a node
    that answers REST but runs no TCP/Operator process. A container restart re-runs
    the script; observed to fix it reliably. Raise if even the restart doesn't."""
    if wait_edgelake_process(rest_conn, process_name, timeout_s=timeout_s):
        return
    logger.warning(
        f"{container}: '{process_name}' process not running after {timeout_s}s; "
        f"restarting the container once (one-shot startup script likely lost a race)"
    )
    if t0 is not None:
        _say(t0, f"{container}: '{process_name}' not running after {timeout_s:.0f}s, restarting it once")
    subprocess.run(["docker", "restart", container], capture_output=True, text=True)
    if not wait_edgelake_process(rest_conn, process_name, timeout_s=timeout_s):
        raise RuntimeError(
            f"{container} still unhealthy after a restart ('{process_name}' not "
            f"running). Check its logs: docker logs {container} / REST `get error log`."
        )


def infra_up(node_count: int, host, repo_root: str, logical_db: str = "mnist_fl") -> None:
    """Postgres shards (only when operators use DB_TYPE=psql) -> AnyLog master ->
    operators, gating each layer on DEEP readiness (processes running, not just REST
    answering). Requires ANYLOG_LICENSE in the environment."""
    t0 = time.time()
    require_license(node_count)
    compose_dir = _compose_dir(host)
    identities = node_identities(node_count)
    generate_node_configs(identities, compose_dir, logical_db, host)
    _say(t0, f"generated configs for master + {node_count} operator(s)")

    # sqlite operators keep their data inside the AnyLog container; no shard needed
    db_type = operator_db_type(compose_dir, host)
    if db_type == "psql":
        for ident in identities:
            start_postgres(ident.index, repo_root, host)
            _say(t0, f"postgres{ident.index} started ({ident.index}/{node_count})")
    else:
        logger.info(f"operator DB_TYPE={db_type}: skipping Postgres shard containers")
        _say(t0, f"operator DB_TYPE={db_type}: skipping Postgres shards")

    _say(t0, "starting master (first run pulls the image, which can take minutes)...")
    _make("up", MASTER_TYPE, compose_dir, host)
    master_rest, _ = master_ports()
    _say(t0, "master container up, waiting for it to answer...")
    if not wait_edgelake_status(f"{host.EDGELAKE_HOST}:{master_rest}", timeout_s=BOOT_TIMEOUT_S):
        raise RuntimeError(
            "AnyLog master never answered `get status`. If it is running but idle, "
            "check the license (expired ANYLOG_LICENSE presents exactly like this)."
        )
    _ensure_node_healthy("master", f"{host.EDGELAKE_HOST}:{master_rest}", "TCP", t0=t0)
    _say(t0, "master healthy")
    if _check_license_active(f"{host.EDGELAKE_HOST}:{master_rest}"):
        _say(t0, "license active")
    else:
        _say(t0, "WARNING: could not read the master's license status; continuing")

    for ident in identities:
        _say(t0, f"starting {ident.operator_container} ({ident.index}/{node_count})...")
        _make("up", operator_type(ident), compose_dir, host)

    for ident in identities:
        _say(t0, f"waiting for {ident.operator_container} to declare its Operator "
                 f"({ident.index}/{node_count})...")
        _ensure_node_healthy(
            ident.operator_container,
            f"{host.EDGELAKE_HOST}:{ident.edgelake_rest_port}",
            "Operator",
            t0=t0,
        )
        _say(t0, f"{ident.operator_container} healthy ({ident.index}/{node_count})")
    _say(t0, f"infra up: master + {node_count} operator(s)")
    logger.info(f"AnyLog infra up: master + {node_count} operator(s)")


def _remove_node_containers(node_name: str) -> None:
    """Fallback for when `make down` could not run (compose file or config gone): a
    leftover container, even a stopped init sidecar, keeps its volumes from being removed."""
    # compose names the init sidecar <project>-<service>-1; the project is the compose
    # file's directory, docker-compose-files
    subprocess.run(
        ["docker", "rm", "-f", node_name, f"{node_name}-debug",
         f"docker-compose-files-{node_name}-init-1"],
        capture_output=True, text=True,
    )


def _remove_node_volumes(node_name: str) -> None:
    """Remove a node's compose volumes regardless of project prefix (compose prefixes
    with the compose-file directory name)."""
    proc = subprocess.run(["docker", "volume", "ls", "-q"], capture_output=True, text=True)
    pattern = re.compile(rf"(^|_){re.escape(node_name)}-(anylog|blockchain|data|local-scripts|ext-lib|external-lib)$")
    doomed = [v for v in proc.stdout.split() if pattern.search(v)]
    if doomed:
        subprocess.run(["docker", "volume", "rm", "-f", *doomed], capture_output=True, text=True)
        logger.info(f"removed volumes: {doomed}")


def infra_down(node_count: int, host, repo_root: str, remove_volumes: bool = False) -> None:
    compose_dir = _compose_dir(host)
    for ident in reversed(node_identities(node_count)):
        _make("down", operator_type(ident), compose_dir, host, check=False)
        _remove_node_containers(ident.operator_container)
        if remove_volumes:
            _remove_node_volumes(ident.operator_container)
    _make("down", MASTER_TYPE, compose_dir, host, check=False)
    _remove_node_containers("master")
    if remove_volumes:
        _remove_node_volumes("master")
    # The compose repo's deploy script adds this license-page server (port 32000) to the first
    # master it deploys, with restart: always, so `make down` leaves it running. The next
    # infra-up recreates it. Anything else on this machine only loses the license web page.
    subprocess.run(["docker", "rm", "-f", "anylog-licenses"], capture_output=True, text=True)
    # shards may exist from an earlier psql setup, so always stop them
    for ident in reversed(node_identities(node_count)):
        stop_postgres(ident.index, repo_root, remove_volume=remove_volumes)
    logger.info(f"AnyLog infra down for {node_count} node(s) (volumes removed: {remove_volumes})")


def infra_rebuild(node_count: int, host, repo_root: str, logical_db: str = "mnist_fl",
                  seed_fn=None) -> None:
    infra_down(node_count, host, repo_root, remove_volumes=True)
    infra_up(node_count, host, repo_root, logical_db=logical_db)
    if seed_fn is not None:
        seed_fn()
