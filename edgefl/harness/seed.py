"""Data seeding: load the dataset into each Node's operator via the existing
store_data.py script (one invocation per operator, matching the README's manual step).

Seeding is NOT part of every run — data persists in Postgres across light teardowns.
It's needed after a full rebuild (fresh volumes) and when standing up a tier for the
first time. Re-seeding an already-seeded operator duplicates rows, so the suite only
seeds when it just rebuilt, and the CLI exposes it as an explicit command otherwise.
"""

import os
import re
import subprocess
import time

import requests

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
    "medmnist": "edgefl/data/medmnist/publish_medmnist_info.py",
}

# medmnist: slides inserted per round per operator, so each node's workload stays the same
# at every tier. The demo's 3-operator defaults are 1536 train / 192 test per round.
MEDMNIST_TRAIN_PER_NODE = 512
MEDMNIST_TEST_PER_NODE = 64


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


def _publish_medmnist(identities, host, repo_root: str, first_round: int, last_round: int) -> None:
    """One publisher run: it splits BloodMNIST into per-operator shards and inserts rounds
    first_round..last_round of scalar rows (filename, label, round_number). The PNGs are
    written under edgefl/data/medmnist/published_images/ for the nodes to read.
    The first run downloads the dataset (bloodmnist.npz, md5-checked) next to the script."""
    ds = get_dataset("medmnist")
    script = os.path.join(repo_root, _SEED_SCRIPTS["medmnist"])
    conns = ",".join(f"{host.EDGELAKE_HOST}:{i.edgelake_rest_port}" for i in identities)
    cmd = [
        host.PYTHON_BIN, "-u", script, conns,
        "--db-name", ds.logical_database,
        "--train-table", ds.train_table,
        "--test-table", ds.test_table,
        "--first-round", str(first_round),
        "--num-rounds", str(last_round),
        "--samples-per-round", str(MEDMNIST_TRAIN_PER_NODE * len(identities)),
        "--test-samples-per-round", str(MEDMNIST_TEST_PER_NODE * len(identities)),
    ]
    print(f"[seed] medmnist: round(s) {first_round}-{last_round} across {len(identities)} "
          f"operator(s) ({MEDMNIST_TRAIN_PER_NODE} train / {MEDMNIST_TEST_PER_NODE} test "
          f"slides per operator per round)...", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=os.path.dirname(script), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-30:]
        print(f"[seed] {line}", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(
            f"medmnist publisher failed (rc={proc.returncode}):\n" + "\n".join(tail[-12:])
        )


def _rows_per_round(host, ident, table: str) -> dict[int, int]:
    """Rows this operator holds per round_number. The rows live in partition tables
    (par_<table>_...), which a plain count on the logical table does not see."""
    db = get_dataset("medmnist").logical_database
    base = f"http://{host.EDGELAKE_HOST}:{ident.edgelake_rest_port}"

    def ask(command: str) -> str:
        # an AnyLog/curl User-Agent is needed for a text reply (see infra_anylog)
        return requests.get(
            base, headers={"command": command, "User-Agent": "AnyLog/1.23"}, timeout=120
        ).text

    counts: dict[int, int] = {}
    for part in set(re.findall(rf"par_{re.escape(table)}_[A-Za-z0-9_]+", ask(f"get tables where dbms={db}"))):
        text = ask(f'sql {db} format=table "select round_number, count(*) from {part} group by round_number"')
        for m in re.finditer(r"^\s*(\d+)\s+(\d+)\s*$", text, re.MULTILINE):
            counts[int(m[1])] = counts.get(int(m[1]), 0) + int(m[2])
    return counts


def _incomplete_operators(identities, host, num_rounds: int) -> dict:
    """operator -> rounds (1..num_rounds) where either table has under half its rows.
    AnyLog rejects the first files an operator writes into a fresh table (error 174 in
    its `get error log`), which loses whole early rounds on one or two operators."""
    ds = get_dataset("medmnist")
    needs = ((ds.train_table, MEDMNIST_TRAIN_PER_NODE), (ds.test_table, MEDMNIST_TEST_PER_NODE))
    bad = {}
    for ident in identities:
        missing = set()
        for table, per_round in needs:
            try:
                have = _rows_per_round(host, ident, table)
            except requests.RequestException as e:
                logger.warning(f"could not read row counts from {ident.operator_container}: {e}")
                continue
            missing |= {r for r in range(1, num_rounds + 1) if have.get(r, 0) < per_round / 2}
        if missing:
            bad[ident] = sorted(missing)
    return bad


def _seed_medmnist(node_count: int, host, repo_root: str, num_rounds: int) -> None:
    identities = node_identities(node_count)
    _publish_medmnist(identities, host, repo_root, 1, num_rounds)
    for attempt in (1, 2):
        print("[seed] waiting 20s for the operators to flush, then checking every round...", flush=True)
        time.sleep(20)
        bad = _incomplete_operators(identities, host, num_rounds)
        if not bad:
            print("[seed] every operator has data for every round", flush=True)
            return
        for ident, rounds in bad.items():
            print(f"[seed] {ident.operator_container} is missing round(s) {rounds}", flush=True)
        if attempt == 2:
            break
        low, high = min(min(r) for r in bad.values()), max(max(r) for r in bad.values())
        print(f"[seed] re-publishing rounds {low}-{high} to those operator(s) only", flush=True)
        _publish_medmnist(list(bad), host, repo_root, low, high)
    print("[seed] WARNING: some rounds are still missing after one repair pass; a run that "
          "reaches them will stall on those nodes. Inspect `get error log` on the operator.",
          flush=True)


def seed_all(
    node_count: int,
    host,
    repo_root: str,
    dataset: str = "mnist",
    num_rounds: int | None = None,
    num_rows: int = DEFAULT_NUM_ROWS,
) -> None:
    """Seed operators 1..node_count with `num_rounds` rounds of data (default: the dataset's
    own, DatasetProfile.seed_rounds). MNIST goes operator by operator, sequentially on
    purpose — N parallel inserts into N Postgres shards on one laptop just thrash."""
    if num_rounds is None:
        num_rounds = get_dataset(dataset).seed_rounds
    if dataset == "medmnist":
        _seed_medmnist(node_count, host, repo_root, num_rounds)
        logger.info(f"seeded medmnist into {node_count} operator(s)")
        return
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
