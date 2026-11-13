"""Single-Run lifecycle: env render → launch → readiness → init → train →
completion poll → CSV collect → teardown. Plus the mid-run failure rules
(process death / timeout / stall → FAILED, partial CSV, one retry after a
full rebuild).

Assumes the Docker infra for the tier (master + operators 1..N + Postgres shards,
seeded with data) is already up — that's infra.py's job, driven by the CLI. This module
owns only the Python servers and the run itself, so a runner crash never strands
containers.

Run artifacts land under <repo>/edgefl/harness/runs/<timestamp>_<run_id>_a<attempt>/:
  env/         generated .env per server (the exact contract that launched it)
  logs/        per-server stdout+stderr
  results.csv  raw fl_benchmarks rows (partial on failure)
  manifest.json  config + status + timings — the machine-readable run record
"""

import dataclasses
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import requests

from platform_components.lib.logger.error_handling import get_logger
from .completion import CompletionMonitor, PollResult
from .config import AggregationMode, BenchmarkConfig
from .env_render import render_run_env_files
from .identity import AGGREGATOR_PORT, node_identities, node_identity
from .process import AGGREGATOR_APP, NODE_APP, ServerGroup, ServerProcess, save_kept_servers
from .readiness import wait_all_http_up, wait_edgelake_status
from .results import collect_run_csv

logger = get_logger(__name__)

POLL_INTERVAL_S = 5
# After completion, give the final aggregation + the Benchmarker queue a beat before
# the settle-polling CSV collection takes over (results.collect_run_csv waits out the
# streaming buffer itself).
METRIC_FLUSH_GRACE_S = 5
INIT_TIMEOUT_S = 600  # /init loads the data handler on every node; scales with tier


class RunStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED_SETUP = "FAILED_SETUP"              # servers/operators never became ready
    FAILED_PROCESS_DEATH = "FAILED_PROCESS_DEATH"
    FAILED_TIMEOUT = "FAILED_TIMEOUT"
    FAILED_STALL = "FAILED_STALL"


@dataclass
class RunResult:
    run_id: str
    attempt: int
    status: RunStatus
    reason: str
    run_dir: str
    csv_path: str | None
    csv_rows: int
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.COMPLETED


def _run_dir_for(repo_root: str, config: BenchmarkConfig, attempt: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(
        repo_root, "edgefl", "harness", "runs", f"{stamp}_{config.run_id}_a{attempt}"
    )


def _build_server_group(
    config: BenchmarkConfig, host, repo_root: str, run_dir: str, env_paths: dict[str, str]
) -> ServerGroup:
    edgefl_dir = os.path.join(repo_root, "edgefl")
    log_dir = os.path.join(run_dir, "logs")
    group = ServerGroup()

    # Aggregator first: start_all launches in order, stop_all in reverse, so nodes
    # always die before the aggregator they report to.
    if config.needs_central_aggregator:
        group.add(ServerProcess(
            name="aggregator",
            app=AGGREGATOR_APP,
            port=AGGREGATOR_PORT,
            env_path=env_paths["aggregator"],
            edgefl_dir=edgefl_dir,
            python_bin=host.PYTHON_BIN,
            log_path=os.path.join(log_dir, "aggregator.log"),
        ))

    for identity in node_identities(config.node_count):
        group.add(ServerProcess(
            name=identity.replica_name,
            app=NODE_APP,
            port=identity.uvicorn_port,
            env_path=env_paths[identity.replica_name],
            edgefl_dir=edgefl_dir,
            python_bin=host.PYTHON_BIN,
            log_path=os.path.join(log_dir, f"{identity.replica_name}.log"),
        ))
    return group


def _print_log_hints(group: ServerGroup, run_dir: str) -> None:
    """Tell the person watching where to follow the run live: the aggregator's log (CFL)
    and the first node's, plus a glob for everything."""
    log_dir = os.path.join(run_dir, "logs")
    names = [s.name for s in group.servers]
    picks = [n for n in names if n == "aggregator"] + [n for n in names if n != "aggregator"][:1]
    print("\nservers launched; follow the run live in another terminal:", flush=True)
    for name in picks:
        print(f"  tail -f {log_dir}/{name}.log", flush=True)
    print(f"  tail -f {log_dir}/*.log   # all servers\n", flush=True)


def _check_operators_ready(config: BenchmarkConfig, host) -> list[str]:
    """Quick gate that the tier's operators (and master) answer `get status`.
    Returns the endpoints that are NOT ready."""
    endpoints = [f"{host.EDGELAKE_HOST}:{host.MASTER_REST_PORT}"] + [
        f"{host.EDGELAKE_HOST}:{i.edgelake_rest_port}"
        for i in node_identities(config.node_count)
    ]
    return [ep for ep in endpoints if not wait_edgelake_status(ep, timeout_s=30)]


def _init_cfl(config: BenchmarkConfig, group: ServerGroup, index: str) -> None:
    """Drive the aggregator's /init + /start-training. DFL runs skip this — their nodes
    self-start off SELF_START/COLD_START env."""
    agg_url = f"http://localhost:{AGGREGATOR_PORT}"
    node_urls = [s.url for s in group.servers if s.name != "aggregator"]

    r = requests.post(
        f"{agg_url}/init",
        json={"nodeUrls": node_urls, "index": index},
        timeout=INIT_TIMEOUT_S,
    )
    r.raise_for_status()
    logger.info(f"[{index}] aggregator /init done: {r.text[:200]}")

    r = requests.post(
        f"{agg_url}/start-training",
        json={
            "totalRounds": config.total_rounds,
            "minParams": config.min_params,
            "index": index,
        },
        timeout=30,
    )
    r.raise_for_status()
    logger.info(f"[{index}] training started ({config.total_rounds} rounds)")


def _wipe_run_state(host) -> None:
    """Light teardown: clear the file-transfer artifacts. Blockchain isolation comes
    free from the fresh index name per run, and data stays in Postgres."""
    for rel in ("edgefl/file_write", "edgefl/tmp_dir"):
        path = os.path.join(host.GITHUB_DIR, rel)
        if not os.path.isdir(path):
            continue
        for entry in os.listdir(path):
            full = os.path.join(path, entry)
            try:
                shutil.rmtree(full) if os.path.isdir(full) else os.remove(full)
            except OSError as e:
                logger.warning(f"could not remove {full}: {e}")


def _write_manifest(run_dir: str, config: BenchmarkConfig, result: RunResult) -> None:
    manifest = {
        "run_id": result.run_id,
        "attempt": result.attempt,
        "status": result.status.value,
        "reason": result.reason,
        "csv_path": result.csv_path,
        "csv_rows": result.csv_rows,
        "duration_s": round(result.duration_s, 1),
        "config": {
            k: (v.value if isinstance(v, Enum) else v)
            for k, v in dataclasses.asdict(config).items()
        },
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def run_once(
    config: BenchmarkConfig, host, repo_root: str, attempt: int = 1, keep_servers: bool = False
) -> RunResult:
    """Execute one Run end-to-end against already-running infra. With keep_servers, a
    COMPLETED run leaves its servers up (recorded for the next harness command to stop);
    any failure still tears them down."""
    if (
        config.aggregation_mode == AggregationMode.DECENTRALIZED
        and not (config.self_start and config.cold_start)
    ):
        raise ValueError(
            "pure DFL runs need self_start=True and cold_start=True: with no central "
            "aggregator, nodes must self-init and node1 must bootstrap round 1."
        )

    index = config.run_id
    run_dir = _run_dir_for(repo_root, config, attempt)
    started = time.time()
    logger.info(f"=== RUN {index} (attempt {attempt}) -> {run_dir} ===")

    def finish(status: RunStatus, reason: str, csv_path=None, csv_rows=0) -> RunResult:
        result = RunResult(
            run_id=index, attempt=attempt, status=status, reason=reason,
            run_dir=run_dir, csv_path=csv_path, csv_rows=csv_rows,
            duration_s=time.time() - started,
        )
        _write_manifest(run_dir, config, result)
        logger.info(f"=== RUN {index} finished: {status.value} ({reason}) ===")
        return result

    not_ready = _check_operators_ready(config, host)
    if not_ready:
        return finish(
            RunStatus.FAILED_SETUP,
            f"EdgeLake not ready: {not_ready}. Bring infra up first "
            f"(python -m harness infra-up --nodes {config.node_count}).",
        )

    env_paths = render_run_env_files(config, host, run_dir, index_name=index)
    group = _build_server_group(config, host, repo_root, run_dir, env_paths)
    kept = False

    try:
        group.start_all()
        _print_log_hints(group, run_dir)
        failed_urls = wait_all_http_up([s.url for s in group.servers], timeout_s=120)
        if failed_urls:
            return finish(
                RunStatus.FAILED_SETUP,
                f"servers never answered HTTP: {failed_urls} (see {run_dir}/logs)",
            )

        if config.needs_central_aggregator:
            try:
                _init_cfl(config, group, index)
            except requests.RequestException as e:
                return finish(RunStatus.FAILED_SETUP, f"aggregator init failed: {e}")

        monitor = CompletionMonitor(
            f"{host.EDGELAKE_HOST}:{host.MASTER_REST_PORT}", config, index
        )
        query_conn = f"{host.EDGELAKE_HOST}:{node_identity(1).edgelake_rest_port}"
        csv_path = os.path.join(run_dir, "results.csv")

        while True:
            time.sleep(POLL_INTERVAL_S)

            dead = group.dead_servers()
            if dead:
                names = [f"{s.name}(rc={s.returncode()})" for s in dead]
                _, rows = collect_run_csv(query_conn, index, csv_path)
                return finish(
                    RunStatus.FAILED_PROCESS_DEATH,
                    f"server(s) died mid-run: {names} (see {run_dir}/logs)",
                    csv_path, rows,
                )

            poll = monitor.poll()
            if poll == PollResult.RUNNING:
                continue

            if poll == PollResult.COMPLETE:
                time.sleep(METRIC_FLUSH_GRACE_S)
                _, rows = collect_run_csv(query_conn, index, csv_path)
                kept = keep_servers
                return finish(
                    RunStatus.COMPLETED,
                    f"completed in {monitor.elapsed_s:.0f}s", csv_path, rows,
                )

            # TIMEOUT or STALLED: partial, flagged result
            _, rows = collect_run_csv(query_conn, index, csv_path)
            status = (
                RunStatus.FAILED_TIMEOUT if poll == PollResult.TIMEOUT
                else RunStatus.FAILED_STALL
            )
            return finish(
                status,
                f"{poll.value} after {monitor.elapsed_s:.0f}s "
                f"(timeout={config.completion_timeout_s}s, "
                f"no_progress={config.no_progress_timeout_s}s)",
                csv_path, rows,
            )
    finally:
        if kept:
            # The file-transfer artifacts stay too: stopping the kept servers wipes them.
            save_kept_servers(group, index, run_dir)
        else:
            group.stop_all()
            _wipe_run_state(host)


def run_with_retry(
    config: BenchmarkConfig, host, repo_root: str, rebuild_fn=None, keep_servers: bool = False
) -> list[RunResult]:
    """The plan's failure policy: one retry after a fresh Full rebuild, then hard-fail.
    rebuild_fn (from the CLI/suite layer) tears down and rebuilds the Docker infra +
    reseeds data; without one the retry runs against the existing infra."""
    first = run_once(config, host, repo_root, attempt=1, keep_servers=keep_servers)
    if first.ok:
        return [first]

    logger.warning(f"[{config.run_id}] attempt 1 failed ({first.status.value}); retrying once")
    if rebuild_fn is not None:
        rebuild_fn()
    second = run_once(config, host, repo_root, attempt=2, keep_servers=keep_servers)
    if not second.ok:
        logger.error(f"[{config.run_id}] hard failure after retry ({second.status.value})")
    return [first, second]
