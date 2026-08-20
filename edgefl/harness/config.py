"""BenchmarkConfig: the typed per-run parameter set.

A run is one end-to-end benchmark execution; a config is its parameters. Suites build
lists of configs by comprehension. Each config yields a deterministic run_id, which
doubles as the fresh blockchain index name under light teardown.
"""

from dataclasses import dataclass, field
from enum import Enum


class AggregationMode(str, Enum):
    CENTRALIZED = "centralized"      # CFL: central aggregator drives rounds
    DECENTRALIZED = "decentralized"  # DFL: nodes self-aggregate
    HYBRID = "hybrid"                # mixed cohorts, one shared model


class DriftHandling(str, Enum):
    NONE = "none"
    SKIP = "skip"
    PAUSE = "pause"


class TeardownStrategy(str, Enum):
    LIGHT = "light"   # default: fresh index + run-state reset
    FULL = "full"     # opt-in: full infra rebuild


@dataclass(frozen=True)
class BenchmarkConfig:
    # core
    node_count: int                                  # tier size (5/10/15/30)
    aggregation_mode: AggregationMode
    dataset: str = "mnist"
    total_rounds: int = 10
    min_params: int = 1                              # DFL aggregate threshold / CFL minParams

    # DFL drift handling
    drift_handling: DriftHandling = DriftHandling.NONE
    round_lag_threshold: int = 3                     # skip mode
    drift_threshold: int = 3                         # pause mode
    pause_max_seconds: int = 60                      # pause mode

    # DFL self-start
    self_start: bool = False
    cold_start: bool = False

    # rollback (CFL only — DFL peer aggregation isn't guarded yet)
    rollback_enabled: bool = True             # allow /rollback at all
    rollback_auto_enabled: bool = False       # auto-rollback on accuracy regression
    rollback_patience_rounds: int = 3
    rollback_min_delta: float = 0.0
    rollback_allow_manual: bool = True
    rollback_log_events: bool = True

    # hybrid only: number of centralized nodes; the rest run decentralized. ignored for
    # pure modes.
    hybrid_centralized: int = 0

    # lifecycle
    teardown: TeardownStrategy = TeardownStrategy.LIGHT
    completion_timeout_s: int = 1800                 # wall-clock escape hatch
    no_progress_timeout_s: int = 300                 # stall detector

    def __post_init__(self):
        if self.node_count < 1:
            raise ValueError(f"node_count must be >= 1, got {self.node_count}")
        if self.min_params < 1:
            raise ValueError(f"min_params must be >= 1, got {self.min_params}")
        if self.min_params > self.node_count:
            raise ValueError(
                f"min_params ({self.min_params}) > node_count ({self.node_count})"
            )
        if self.total_rounds < 1:
            raise ValueError(f"total_rounds must be >= 1, got {self.total_rounds}")
        if self.aggregation_mode == AggregationMode.HYBRID:
            if not (0 < self.hybrid_centralized < self.node_count):
                raise ValueError(
                    f"hybrid run needs 0 < hybrid_centralized < node_count; got "
                    f"hybrid_centralized={self.hybrid_centralized}, node_count={self.node_count}"
                )
        if self.cold_start and self.aggregation_mode != AggregationMode.DECENTRALIZED:
            raise ValueError("cold_start only valid in decentralized mode")
        if self.rollback_auto_enabled and self.aggregation_mode == AggregationMode.DECENTRALIZED:
            raise ValueError(
                "rollback_auto_enabled is not supported in decentralized mode yet "
                "(DFL peer aggregation loads weights outside the rollback guard)"
            )
        if self.rollback_patience_rounds < 1:
            raise ValueError(
                f"rollback_patience_rounds must be >= 1, got {self.rollback_patience_rounds}"
            )

    @property
    def run_id(self) -> str:
        """Deterministic, filesystem- and index-safe id for this run.

        Doubles as the fresh blockchain index name under light teardown, so every run is
        isolated on-chain. Encodes the parameters that distinguish runs.
        """
        mode = self.aggregation_mode.value[:3]   # cen / dec / hyb
        parts = [
            self.dataset,
            mode,
            f"n{self.node_count}",
            f"mp{self.min_params}",
            f"r{self.total_rounds}",
        ]
        if self.aggregation_mode == AggregationMode.HYBRID:
            parts.append(f"h{self.hybrid_centralized}")
        if self.drift_handling != DriftHandling.NONE:
            parts.append(f"drift-{self.drift_handling.value}")
        if self.cold_start:
            parts.append("cold")
        return "_".join(parts)

    def is_aggregator_for_node(self, node_index: int) -> bool:
        """Does node node_index (1-based) run as a DFL aggregator under this config?

        centralized: never. decentralized: always. hybrid: nodes after the centralized
        cohort.
        """
        if self.aggregation_mode == AggregationMode.CENTRALIZED:
            return False
        if self.aggregation_mode == AggregationMode.DECENTRALIZED:
            return True
        # nodes 1..hybrid_centralized are centralized, rest decentralized
        return node_index > self.hybrid_centralized

    @property
    def needs_central_aggregator(self) -> bool:
        """True if this run launches a central aggregator process (CFL or hybrid)."""
        return self.aggregation_mode in (
            AggregationMode.CENTRALIZED,
            AggregationMode.HYBRID,
        )
