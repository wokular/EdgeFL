"""Run-completion detection: poll the blockchain ledger for per-mode done signals.

No clean programmatic completion signal exists (/start-training returns immediately and
DFL nodes only stop once TOTAL_ROUNDS is reached), so the harness polls the master's ledger — the same query surface
the servers themselves use — with a per-mode predicate:

  CFL:    the aggregator's RoundStart reached total_rounds AND >= min_params submodels
          exist at total_rounds. The plan's bare "RoundStart hits total_rounds" fires
          when the final round *starts* (the aggregator publishes RoundStart, then waits
          for submodels), so the submodel clause keeps us from collecting results before
          the last round has trained.
  DFL:    all node_count nodes published a submodel at total_rounds (unanimity — the
          measurement bar, even though the algorithm itself progresses on MIN_PARAMS).
  hybrid: both of the above (the cohorts share one model, so unanimity spans all nodes).

CompletionMonitor adds the two failure detectors from the plan on top of the predicate:
a wall-clock completion timeout and a no-progress stall (ledger fingerprint unchanged
for no_progress_timeout_s). Process death is the runner's job — it holds the Popen
handles.
"""

import time
from enum import Enum

import requests

from platform_components.lib.logger.error_handling import get_logger
from .config import AggregationMode, BenchmarkConfig

logger = get_logger(__name__)

ANYLOG_UA = {"User-Agent": "AnyLog/1.23"}


def blockchain_get(rest_conn: str, index: str, condition: str = "") -> list[dict]:
    """Fetch the policies at `index`, unwrapped to their inner dicts. rest_conn is
    host:port of any EdgeLake node (the harness polls the master)."""
    cmd = f"blockchain get {index}"
    if condition:
        cmd += f" where {condition}"
    r = requests.get(
        f"http://{rest_conn}", headers={**ANYLOG_UA, "command": cmd}, timeout=10
    )
    r.raise_for_status()
    data = r.json() or []
    return [item[index] for item in data if index in item]


def submodel_publishers_at_round(rest_conn: str, index: str, round_number: int) -> set[str]:
    """Distinct node names that published a submodel at round_number."""
    policies = blockchain_get(
        rest_conn, index, f"round_number = {round_number} and node_type = training"
    )
    return {p.get("node") for p in policies if p.get("node")}


def max_round_started(rest_conn: str, index: str, node_type: str | None = None) -> int:
    """Highest RoundStart round_number at index (0 if none). node_type filters to
    'aggregator' for the CFL predicate; None counts any RoundStart (progress signal)."""
    condition = "policy_type = RoundStart"
    if node_type:
        condition += f" and node_type = {node_type}"
    policies = blockchain_get(rest_conn, index, condition)
    return max((int(p.get("round_number", 0)) for p in policies), default=0)


def count_submodels(rest_conn: str, index: str) -> int:
    """Total submodel policies at index — part of the progress fingerprint."""
    return len(blockchain_get(rest_conn, index, "node_type = training"))


def is_run_complete(rest_conn: str, index: str, config: BenchmarkConfig) -> bool:
    """The per-mode completion predicate against the live ledger."""
    total = config.total_rounds

    if config.aggregation_mode == AggregationMode.CENTRALIZED:
        return (
            max_round_started(rest_conn, index, node_type="aggregator") >= total
            and len(submodel_publishers_at_round(rest_conn, index, total))
            >= min(config.min_params, config.node_count)
        )

    if config.aggregation_mode == AggregationMode.DECENTRALIZED:
        publishers = submodel_publishers_at_round(rest_conn, index, total)
        return len(publishers) >= config.node_count

    # hybrid: centralized cohort's aggregator finished its rounds AND every node
    # (both cohorts — one shared model) published at total_rounds.
    return (
        max_round_started(rest_conn, index, node_type="aggregator") >= total
        and len(submodel_publishers_at_round(rest_conn, index, total))
        >= config.node_count
    )


def progress_fingerprint(rest_conn: str, index: str) -> tuple[int, int]:
    """(max RoundStart round, total submodel count) — any change counts as progress."""
    return (
        max_round_started(rest_conn, index),
        count_submodels(rest_conn, index),
    )


class PollResult(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    STALLED = "stalled"      # ledger unchanged for no_progress_timeout_s
    TIMEOUT = "timeout"      # completion_timeout_s wall clock expired


class CompletionMonitor:
    """Stateful poller for one run: completion predicate + stall + timeout.

    The runner calls poll() on its own cadence; this object only tracks time and the
    last ledger fingerprint. Ledger query errors are treated as "no new progress
    observed" rather than failures — transient REST hiccups mid-run are expected.
    """

    def __init__(self, rest_conn: str, config: BenchmarkConfig, index: str | None = None):
        self.rest_conn = rest_conn
        self.config = config
        self.index = index or config.run_id
        self.started_at = time.time()
        self._last_fingerprint: tuple[int, int] | None = None
        self._last_progress_at = self.started_at

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def poll(self) -> PollResult:
        try:
            if is_run_complete(self.rest_conn, self.index, self.config):
                return PollResult.COMPLETE
            fingerprint = progress_fingerprint(self.rest_conn, self.index)
        except requests.RequestException as e:
            logger.warning(f"[{self.index}] ledger poll failed (transient?): {e}")
            fingerprint = None

        now = time.time()
        if fingerprint is not None and fingerprint != self._last_fingerprint:
            self._last_fingerprint = fingerprint
            self._last_progress_at = now
            logger.info(
                f"[{self.index}] progress: max_round={fingerprint[0]} "
                f"submodels={fingerprint[1]} elapsed={self.elapsed_s:.0f}s"
            )

        if now - self.started_at > self.config.completion_timeout_s:
            return PollResult.TIMEOUT
        if now - self._last_progress_at > self.config.no_progress_timeout_s:
            return PollResult.STALLED
        return PollResult.RUNNING
