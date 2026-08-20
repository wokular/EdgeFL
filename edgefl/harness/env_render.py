"""Env rendering: merge host profile + derived identity + dataset + config into one
.env per node (and one for the aggregator), then write them to disk.

The app reads everything via os.getenv(), so generated env files are the contract
between the harness and the servers. They're disposable artifacts written under
<run_dir>/env/ and gitignored.

This module is pure (config + identity + host -> dict/text) plus a thin writer; it does
not launch processes.
"""

import os

from .config import BenchmarkConfig, AggregationMode
from .datasets import get_dataset
from .identity import NodeIdentity, node_identities


# Values fixed for every node regardless of host/config.
_CONSTANTS = {
    "TRAINING_APPLICATION_DIR": "edgefl/platform_components/data_handlers",
    "TMP_DIR": "edgefl/tmp_dir/",
    "FILE_WRITE_DESTINATION": "edgefl/file_write",
    "DOCKER_FILE_WRITE_DESTINATION": "/app/file_write",
    "EDGELAKE_DOCKER_RUNNING": "True",
}


def _host_common(host) -> dict[str, str]:
    """Host-profile-derived keys shared by node and aggregator env files."""
    return {"GITHUB_DIR": host.GITHUB_DIR}


def render_node_env(
    config: BenchmarkConfig,
    identity: NodeIdentity,
    host,
    index_name: str,
) -> dict[str, str]:
    """Build the env mapping for a single node.

    index_name is the blockchain index for this run (config.run_id under light teardown),
    used as REPLICA_INDEX so every run is isolated on-chain.
    """
    ds = get_dataset(config.dataset)
    is_dfl = config.is_aggregator_for_node(identity.index)
    mode = (
        AggregationMode.DECENTRALIZED.value
        if is_dfl
        else AggregationMode.CENTRALIZED.value
    )

    env: dict[str, str] = {}
    env.update(_CONSTANTS)
    env.update(_host_common(host))

    # dataset profile
    env["MODULE_NAME"] = ds.module_name
    env["MODULE_FILE"] = ds.module_file
    env["LOGICAL_DATABASE"] = ds.logical_database
    env["TRAIN_TABLE"] = ds.train_table
    env["TEST_TABLE"] = ds.test_table

    # derived identity
    env["SERVER_TYPE"] = "node"
    env["REPLICA_NAME"] = identity.replica_name
    env["REPLICA_INDEX"] = index_name
    env["EDGELAKE_DOCKER_CONTAINER_NAME"] = identity.operator_container
    env["EXTERNAL_IP"] = f"{host.EDGELAKE_HOST}:{identity.edgelake_rest_port}"
    env["EXTERNAL_TCP_IP_PORT"] = f"{host.EDGELAKE_HOST}:{identity.edgelake_tcp_port}"

    # config (per run)
    env["AGGREGATION_MODE"] = mode
    env["MIN_PARAMS"] = str(config.min_params)
    env["DRIFT_HANDLING"] = config.drift_handling.value
    env["ROUND_LAG_THRESHOLD"] = str(config.round_lag_threshold)
    env["DRIFT_THRESHOLD"] = str(config.drift_threshold)
    env["PAUSE_MAX_SECONDS"] = str(config.pause_max_seconds)
    env["SELF_START"] = str(config.self_start)
    # Exactly one node bootstraps round 1 — multiple cold-starters race and publish
    # duplicate round-1 RoundStarts (see dflFeatures.txt OPEN QUESTIONS). Node 1 is
    # always in the DFL cohort for pure-DFL runs, the only mode where cold_start is
    # valid (config validation enforces that).
    env["COLD_START"] = str(config.cold_start and is_dfl and identity.index == 1)

    # rollback. auto-rollback is CFL-only (config validation enforces that), so DFL
    # nodes in a hybrid run get it forced off rather than inheriting the run setting.
    env["ROLLBACK_ENABLED"] = str(config.rollback_enabled)
    env["ROLLBACK_AUTO_ENABLED"] = str(config.rollback_auto_enabled and not is_dfl)
    env["ROLLBACK_PATIENCE_ROUNDS"] = str(config.rollback_patience_rounds)
    env["ROLLBACK_MIN_DELTA"] = str(config.rollback_min_delta)
    env["ROLLBACK_ALLOW_MANUAL"] = str(config.rollback_allow_manual)
    env["ROLLBACK_LOG_EVENTS"] = str(config.rollback_log_events)

    # benchmarking: all nodes stream metrics to operator1's REST endpoint, which writes
    # to the benchmarkfl sqlite db (the benchmark team's pipeline).
    node1 = node_identities(1)[0]
    env["BENCHMARK_ENABLED"] = "True"
    env["BENCHMARK_REST_CONN"] = f"{host.EDGELAKE_HOST}:{node1.edgelake_rest_port}"

    return env


def render_aggregator_env(
    config: BenchmarkConfig,
    host,
    index_name: str,
) -> dict[str, str]:
    """Build the env mapping for the central aggregator (CFL and hybrid runs)."""
    ds = get_dataset(config.dataset)
    master_rest = host.MASTER_REST_PORT
    master_tcp = host.MASTER_TCP_PORT

    env: dict[str, str] = {}
    env.update(_CONSTANTS)
    env.update(_host_common(host))

    env["MODULE_NAME"] = ds.module_name
    env["MODULE_FILE"] = ds.module_file
    env["SERVER_TYPE"] = "aggregator"
    env["AGG_NAME"] = "agg"
    env["REPLICA_INDEX"] = index_name
    env["EDGELAKE_DOCKER_CONTAINER_NAME"] = "master"
    env["EXTERNAL_IP"] = f"{host.EDGELAKE_HOST}:{master_rest}"
    env["EXTERNAL_TCP_IP_PORT"] = f"{host.EDGELAKE_HOST}:{master_tcp}"

    node1 = node_identities(1)[0]
    env["BENCHMARK_ENABLED"] = "True"
    env["BENCHMARK_REST_CONN"] = f"{host.EDGELAKE_HOST}:{node1.edgelake_rest_port}"

    return env


def to_env_text(env: dict[str, str]) -> str:
    """Serialize an env mapping to .env text. Values are quoted for paths/spaces; dotenv
    strips the quotes on load."""
    lines = [f'{k}="{v}"' for k, v in env.items()]
    return "\n".join(lines) + "\n"


def write_env_file(env: dict[str, str], path: str) -> str:
    """Write an env mapping to `path`, creating parent dirs. Returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(to_env_text(env))
    return path


def render_run_env_files(
    config: BenchmarkConfig,
    host,
    run_dir: str,
    index_name: str | None = None,
) -> dict[str, str]:
    """Render and write every env file for a run under <run_dir>/env/.

    Returns {logical_name: written_path}, e.g. {"node1": ".../env/node1.env", ...}. The
    aggregator entry is present only when the config needs one.
    """
    idx = index_name or config.run_id
    env_dir = os.path.join(run_dir, "env")
    written: dict[str, str] = {}

    for identity in node_identities(config.node_count):
        env = render_node_env(config, identity, host, idx)
        path = os.path.join(env_dir, f"{identity.replica_name}.env")
        written[identity.replica_name] = write_env_file(env, path)

    if config.needs_central_aggregator:
        env = render_aggregator_env(config, host, idx)
        path = os.path.join(env_dir, "aggregator.env")
        written["aggregator"] = write_env_file(env, path)

    return written
