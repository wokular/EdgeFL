"""Harness CLI. Run from the edgefl/ package root:

    cd edgefl
    python -m harness infra-up --nodes 5 --seed     # Docker stack + data for a tier
    python -m harness run --mode cfl --nodes 5      # one Run against that infra
    python -m harness suite --tiers 5,10            # full sweep, tiered teardown
    python -m harness infra-down --nodes 5          # stop containers (keep data)
    python -m harness stop                          # stop leftover aggregator/node servers

Needs edgefl/harness/host_profile.py (copy host_profile.example.py and edit).
--dry-run on run/suite prints the plan and renders env files without
touching Docker or launching anything.
"""

import argparse
import os
import sys

from .config import AggregationMode, BenchmarkConfig, DriftHandling
from .env_render import render_run_env_files
from .identity import AGGREGATOR_PORT, node_identities

# harness/ -> edgefl/ -> repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_host_profile():
    try:
        from . import host_profile
        return host_profile
    except ImportError:
        sys.exit(
            "missing edgefl/harness/host_profile.py (per-machine values).\n"
            "Create it with:\n"
            "    cp edgefl/harness/host_profile.example.py edgefl/harness/host_profile.py\n"
            "then edit GITHUB_DIR (and anything else that differs on this machine)."
        )


def _backend(host):
    """The Docker infra backend module selected by the host profile: anylog (primary
    on Apple Silicon) or edgelake. Both expose infra_up/infra_down/infra_rebuild."""
    name = getattr(host, "INFRA_BACKEND", "edgelake")
    if name == "anylog":
        from . import infra_anylog as backend
    elif name == "edgelake":
        from . import infra as backend
    else:
        sys.exit(f"unknown INFRA_BACKEND '{name}' in host_profile.py (anylog|edgelake)")
    return backend


def tier_sweep(node_count: int) -> list[BenchmarkConfig]:
    """The default Configs swept per Tier. Edit here to change the suite; single-run
    experiments go through `run` flags instead."""
    n = node_count
    return [
        # CFL baseline: aggregator waits for every node each round
        BenchmarkConfig(n, AggregationMode.CENTRALIZED, min_params=n),
        # DFL baseline: unanimity threshold
        BenchmarkConfig(n, AggregationMode.DECENTRALIZED, min_params=n,
                        self_start=True, cold_start=True),
        # DFL drift variants: aggregate at n-1 so drift handling has room to act
        BenchmarkConfig(n, AggregationMode.DECENTRALIZED, min_params=max(1, n - 1),
                        drift_handling=DriftHandling.SKIP,
                        self_start=True, cold_start=True),
        BenchmarkConfig(n, AggregationMode.DECENTRALIZED, min_params=max(1, n - 1),
                        drift_handling=DriftHandling.PAUSE,
                        self_start=True, cold_start=True),
    ]


def _config_from_args(args) -> BenchmarkConfig:
    mode = {
        "cfl": AggregationMode.CENTRALIZED,
        "dfl": AggregationMode.DECENTRALIZED,
        "hybrid": AggregationMode.HYBRID,
    }[args.mode]
    is_dfl = mode == AggregationMode.DECENTRALIZED
    return BenchmarkConfig(
        node_count=args.nodes,
        aggregation_mode=mode,
        dataset=args.dataset,
        total_rounds=args.rounds,
        min_params=args.min_params if args.min_params else args.nodes,
        drift_handling=DriftHandling(args.drift),
        # pure DFL can only start via self-start + a single cold-start bootstrap
        self_start=is_dfl,
        cold_start=is_dfl,
        hybrid_centralized=args.hybrid_centralized,
        completion_timeout_s=args.completion_timeout,
        no_progress_timeout_s=args.no_progress_timeout,
    )


def _describe(config: BenchmarkConfig) -> str:
    return (
        f"{config.run_id}: {config.aggregation_mode.value} n={config.node_count} "
        f"rounds={config.total_rounds} min_params={config.min_params} "
        f"drift={config.drift_handling.value}"
    )


def _dry_run(configs: list[BenchmarkConfig], host) -> None:
    """Print the plan and render the env files it would launch with."""
    dry_dir = os.path.join(REPO_ROOT, "edgefl", "harness", "runs", "_dryrun")
    for config in configs:
        print(f"\n[dry-run] {_describe(config)}")
        run_dir = os.path.join(dry_dir, config.run_id)
        env_paths = render_run_env_files(config, host, run_dir, index_name=config.run_id)
        if config.needs_central_aggregator:
            print(f"  aggregator      :{AGGREGATOR_PORT}  {env_paths['aggregator']}")
        for identity in node_identities(config.node_count):
            print(
                f"  {identity.replica_name:<6} uvicorn :{identity.uvicorn_port}  "
                f"operator {identity.operator_container} "
                f"rest:{identity.edgelake_rest_port} tcp:{identity.edgelake_tcp_port}  "
                f"{env_paths[identity.replica_name]}"
            )
    print(f"\n[dry-run] env files rendered under {dry_dir}; nothing was launched.")


def _print_results(results) -> None:
    print("\n=== results ===")
    for r in results:
        print(
            f"  {r.status.value:<22} {r.run_id} (attempt {r.attempt}) "
            f"rows={r.csv_rows} {r.duration_s:.0f}s\n"
            f"    {r.reason}\n    -> {r.run_dir}"
        )
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"  {len(failed)} attempt(s) did not complete.")


def cmd_infra_up(args, host):
    from .seed import seed_all
    _backend(host).infra_up(args.nodes, host, REPO_ROOT)
    if args.seed:
        seed_all(args.nodes, host, REPO_ROOT, dataset=args.dataset)
    print(f"infra up for {args.nodes} node(s)" + (" (seeded)" if args.seed else ""))


def cmd_infra_down(args, host):
    _backend(host).infra_down(args.nodes, host, REPO_ROOT, remove_volumes=args.volumes)


def cmd_seed(args, host):
    from .seed import seed_all
    seed_all(args.nodes, host, REPO_ROOT, dataset=args.dataset)


def cmd_stop(args, host):
    from .process import stop_stray_servers
    stopped = stop_stray_servers()
    if not stopped:
        print("no aggregator/node servers running")
    for pid, port, name in stopped:
        print(f"stopped {name} (pid {pid}, port {port})")


def _ensure_ports_free(node_count: int, needs_aggregator: bool) -> None:
    """Exit before touching anything if a server port is taken. Checking here rather than
    in the run: a failed attempt triggers a full infra rebuild + retry, and that would
    wipe the stack over a problem a rebuild can't fix."""
    from .process import busy_ports
    ports = ([AGGREGATOR_PORT] if needs_aggregator else []) + [
        i.uvicorn_port for i in node_identities(node_count)
    ]
    busy = busy_ports(ports)
    if busy:
        lines = "\n".join(f"  :{port} is held by {who}" for port, who in sorted(busy.items()))
        sys.exit(
            f"cannot start servers, {len(busy)} port(s) already in use:\n{lines}\n"
            f"Stop leftover servers with: python -m harness stop"
        )


def cmd_run(args, host):
    config = _config_from_args(args)
    if args.dry_run:
        _dry_run([config], host)
        return
    _ensure_ports_free(config.node_count, config.needs_central_aggregator)
    from .runner import run_once, run_with_retry
    if args.no_retry:
        results = [run_once(config, host, REPO_ROOT)]
    else:
        results = run_with_retry(config, host, REPO_ROOT, rebuild_fn=_rebuild_fn(config, host, args.dataset))
    _print_results(results)
    if not results[-1].ok:
        sys.exit(1)


def _rebuild_fn(config: BenchmarkConfig, host, dataset: str):
    def rebuild():
        from .seed import seed_all
        _backend(host).infra_rebuild(
            config.node_count, host, REPO_ROOT,
            seed_fn=lambda: seed_all(config.node_count, host, REPO_ROOT, dataset=dataset),
        )
    return rebuild


def cmd_suite(args, host):
    tiers = [int(t) for t in args.tiers.split(",")]
    sweeps = {tier: tier_sweep(tier) for tier in tiers}

    if args.dry_run:
        for tier in tiers:
            print(f"\n=== tier {tier} (forced-Full rebuild + seed before it runs) ===")
            _dry_run(sweeps[tier], host)
        return

    from .runner import run_with_retry
    from .seed import seed_all

    # sweeps include CFL configs, so the aggregator port is needed too
    _ensure_ports_free(max(tiers), needs_aggregator=True)

    all_results = []
    for tier in tiers:
        # Crossing into a tier is always a Full rebuild + reseed (plan: Light's premise
        # breaks when the node count changes; not Config-disableable).
        print(f"\n=== tier {tier}: full rebuild + seed ===")
        _backend(host).infra_rebuild(
            tier, host, REPO_ROOT,
            seed_fn=lambda t=tier: seed_all(t, host, REPO_ROOT, dataset=args.dataset),
        )
        for config in sweeps[tier]:
            print(f"\n--- {_describe(config)} ---")
            all_results += run_with_retry(
                config, host, REPO_ROOT, rebuild_fn=_rebuild_fn(config, host, args.dataset)
            )

    _print_results(all_results)
    if any(not r.ok for r in all_results[-1:]):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="python -m harness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--dataset", default="mnist")

    p = sub.add_parser("infra-up", help="start Postgres + master + operators for a tier")
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--seed", action="store_true", help="insert the dataset after startup")
    add_common(p)
    p.set_defaults(fn=cmd_infra_up)

    p = sub.add_parser("infra-down", help="stop the tier's containers")
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--volumes", action="store_true",
                   help="also remove volumes (blockchain + Postgres data)")
    p.set_defaults(fn=cmd_infra_down)

    p = sub.add_parser("seed", help="insert the dataset into operators 1..N")
    p.add_argument("--nodes", type=int, required=True)
    add_common(p)
    p.set_defaults(fn=cmd_seed)

    p = sub.add_parser("stop", help="stop leftover aggregator/node uvicorn servers")
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser("run", help="execute one Run against running infra")
    p.add_argument("--mode", choices=["cfl", "dfl", "hybrid"], required=True)
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--min-params", type=int, default=None,
                   help="aggregate threshold (default: node count)")
    p.add_argument("--drift", choices=[d.value for d in DriftHandling], default="none")
    p.add_argument("--hybrid-centralized", type=int, default=0,
                   help="hybrid only: size of the centralized cohort")
    p.add_argument("--completion-timeout", type=int, default=1800)
    p.add_argument("--no-progress-timeout", type=int, default=300)
    p.add_argument("--no-retry", action="store_true",
                   help="fail immediately instead of retrying once after a full rebuild")
    p.add_argument("--dry-run", action="store_true")
    add_common(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("suite", help="sweep the default Configs across Tiers")
    p.add_argument("--tiers", default="5", help="comma-separated node counts, e.g. 5,10,15")
    p.add_argument("--dry-run", action="store_true")
    add_common(p)
    p.set_defaults(fn=cmd_suite)

    args = parser.parse_args()
    host = _load_host_profile()
    args.fn(args, host)


if __name__ == "__main__":
    main()
