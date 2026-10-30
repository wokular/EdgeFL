"""Data seeding: load the dataset into each Node's operator via the existing
store_data.py script (one invocation per operator, matching the README's manual step).

Seeding is NOT part of every run — data persists in Postgres across light teardowns.
It's needed after a full rebuild (fresh volumes) and when standing up a tier for the
first time. Re-seeding an already-seeded operator duplicates rows, so the suite only
seeds when it just rebuilt, and the CLI exposes it as an explicit command otherwise.
"""

import os
import subprocess
import time

from platform_components.lib.logger.error_handling import get_logger
from .datasets import get_dataset
from .identity import NodeIdentity, node_identities

logger = get_logger(__name__)

# store_data.py inserts num_rounds batches of num_rows train rows (test = 20% of train).
# These defaults mirror the script's own.
DEFAULT_NUM_ROUNDS = 20
DEFAULT_NUM_ROWS = 50

_SEED_SCRIPTS = {
    # dataset name -> path of the insert script relative to the repo root
    "mnist": "edgefl/data/mnist/store_data.py",
}


def seed_operator(
    identity: NodeIdentity,
    host,
    repo_root: str,
    dataset: str = "mnist",
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    num_rows: int = DEFAULT_NUM_ROWS,
) -> None:
    """Insert the dataset into one operator's data shard."""
    if dataset not in _SEED_SCRIPTS:
        raise KeyError(
            f"no seed script registered for dataset '{dataset}'. "
            f"Known: {sorted(_SEED_SCRIPTS)}. Add one to harness/seed.py."
        )
    ds = get_dataset(dataset)
    script = os.path.join(repo_root, _SEED_SCRIPTS[dataset])
    conn = f"{host.EDGELAKE_HOST}:{identity.edgelake_rest_port}"

    cmd = [
        host.PYTHON_BIN, script, conn,
        "--db-name", ds.logical_database,
        "--num-rounds", str(num_rounds),
        "--num-rows", str(num_rows),
    ]
    logger.info(f"seeding {dataset} into {identity.operator_container} ({conn})")
    proc = subprocess.run(
        cmd, cwd=os.path.dirname(script), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"seed failed for {identity.operator_container} (rc={proc.returncode}):\n"
            f"{proc.stderr.strip()[-2000:]}"
        )


def seed_all(
    node_count: int,
    host,
    repo_root: str,
    dataset: str = "mnist",
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    num_rows: int = DEFAULT_NUM_ROWS,
) -> None:
    """Seed operators 1..node_count. Sequential on purpose — N parallel MNIST inserts
    into N Postgres shards on one laptop just thrash."""
    start = time.time()
    for identity in node_identities(node_count):
        print(f"[seed {identity.index}/{node_count}] {identity.operator_container}: "
              f"inserting {dataset} ({num_rounds} batches x {num_rows} rows)...", flush=True)
        t0 = time.time()
        seed_operator(identity, host, repo_root, dataset, num_rounds, num_rows)
        print(f"[seed {identity.index}/{node_count}] {identity.operator_container}: "
              f"done in {time.time() - t0:.0f}s", flush=True)
    print(f"seeded {dataset} into {node_count} operator(s) in {time.time() - start:.0f}s", flush=True)
    logger.info(f"seeded {dataset} into {node_count} operator(s)")
