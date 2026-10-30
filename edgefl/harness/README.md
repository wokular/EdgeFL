# EdgeFL Large-Scale Evaluation Harness

Runs CFL/DFL federated-learning benchmarks at varying scale (Tiers of 5/10/15/30
Nodes) on a single machine, and writes per-Run result CSVs. Design decisions are in
`../platform_components/benchmarking/HARNESS_PLAN.md`.

Quick vocabulary: a **Node** is one participant (node_server + operator + Postgres); a
**Run** is one end-to-end benchmark execution; a **Config** is its parameter set
(mode, node count, rounds, ...); a **Tier** is a node-count band (5/10/15/30); the
**Harness** is this orchestration layer; the **Benchmarker** is the metrics pipeline
it consumes (`platform_components/benchmarking/`).

Validated end-to-end on Apple Silicon: 5-Node CFL/DFL (July 2026), and 10-Node CFL/DFL
(August 2026), AnyLog backend, MNIST. See `UPSTREAM_PR_NOTES.md` for what to reconcile
with [PR #105](https://github.com/royshadmon/EdgeFL/pull/105) before an upstream submission.

New here? [QUICKSTART.md](QUICKSTART.md) has the short version of setup and the
infra-up, seed, run commands.

## One-time setup

1. **Docker Desktop** with **host networking enabled**
   (Settings → Resources → Network → "Enable host networking"). The AnyLog stack
   binds everything to 127.0.0.1 — no host networking means nothing can talk.
2. **AnyLog license + image + compose repo** — follow
   `AnylogNetworkSetup/AnyLog-Setup.md` sections 2–5:
   - an `anylogco/anylog-network` image for the tag in `host_profile.ANYLOG_TAG`
     (unset = the compose repo's Makefile default). The repo is private: `make login`
     in the compose repo, or `docker load < AnyLog-ARM.tar` for a pre-built tag.
   - clone `github.com/AnyLog-co/docker-compose` on branch **`pre-develop`** and point
     `host_profile.ANYLOG_COMPOSE_DIR` at it. No config edits are needed: the harness
     generates `docker-makefiles/edgefl-master/` and `edgefl-operator<i>/` from the
     committed `anylog-master` / `anylog-operator1` configs. (The guide's roy-local
     steps no longer apply.) Add `docker-makefiles/edgefl-*/` to the repo's
     `.git/info/exclude` so the generated dirs stay out of `git status`.
   - license: `pre-develop`'s `deploy.sh` normally runs an interactive acceptance
     prompt on first `make up`. The harness skips it by putting
     `LICENSE_KEY=$ANYLOG_LICENSE` in the generated configs (deploy.sh's own
     "key already in the config" path). Run one `make up` by hand in the compose repo
     first if you need to accept the license terms yourself.
   - export `ANYLOG_LICENSE` from your shell profile. **An expired license shows up
     as the master idling with no processes** — check expiry first when nothing works.
3. **Python env** — the repo venv with `requirements.txt` plus
   `torch torchvision tensorflow scikit-learn python-dotenv uvicorn fastapi`.
4. **Host profile** — per-machine values:

   ```bash
   cp edgefl/harness/host_profile.example.py edgefl/harness/host_profile.py
   # edit GITHUB_DIR, ANYLOG_COMPOSE_DIR; PYTHON_BIN should point at the venv python
   ```

## Running

Everything is `python -m harness ...` from the `edgefl/` directory (venv active,
`ANYLOG_LICENSE` exported).

```bash
# Bring up master + N operators + N Postgres shards, then insert MNIST into each shard
python -m harness infra-up --nodes 5 --seed

# One Run against that infra (CFL here; results + logs land under harness/runs/)
python -m harness run --mode cfl --nodes 5 --rounds 10

# DFL (self-start + cold-start are implied by the mode)
python -m harness run --mode dfl --nodes 5 --rounds 10

# The default Config sweep across Tiers (full rebuild + reseed at each tier boundary)
python -m harness suite --tiers 5,10

# Print the plan + render env files, but launch nothing
python -m harness suite --tiers 5 --dry-run

# Stop containers (add --volumes to also wipe blockchain + Postgres data)
python -m harness infra-down --nodes 5

# Stop leftover aggregator/node_server processes (they run on the host, not in Docker,
# so infra-down doesn't touch them). A normal `run` cleans up after itself; this is
# for runs that were interrupted or servers started by hand.
python -m harness stop
```

`run` and `suite` check that the server ports (8080 aggregator, 8081.. nodes) are free
before doing anything, and exit naming whatever holds each one. They check first
because a failed attempt triggers a full infra rebuild + retry, which would wipe the
stack over a problem a rebuild can't fix.

The default per-tier sweep lives in `tier_sweep()` in `__main__.py` — edit it there.

## What a Run produces

`edgefl/harness/runs/<utc-timestamp>_<run_id>_a<attempt>/`:

| artifact | meaning |
|---|---|
| `env/*.env` | the generated env file each server was launched with |
| `logs/*.log` | per-server stdout+stderr (uvicorn + training logs) |
| `results.csv` | raw `fl_benchmarks` rows for this run_id (long format) |
| `manifest.json` | config + status + reason + timings — the machine-readable record |

CSV columns: `node, training_index, round_number, metric_name, metric_value, time`
(+ AnyLog bookkeeping columns). `training_index` == `run_id` == the blockchain index,
so every run is isolated on-chain and in the results table.

Timing metrics recorded per round: `polling_time_s`, `training_time_s`,
`total_round_time_s` per node; `aggregation_time_s` (per DFL node, or `node_name=agg`
for CFL); `first_to_last_arrival_s` + `straggling_node_id` (CFL aggregator).
`round_accuracy` is the accuracy team's metric and appears when their inference
call-sites are active.

**DFL note:** DFL nodes never stop on their own — the harness tears down once all N
published a submodel at `total_rounds`, but rows for a few extra rounds usually land
in the CSV (the network keeps training during collection). Filter
`round_number <= total_rounds` when analyzing.

## How completion/failure is decided

- **CFL complete** — aggregator RoundStart reached `total_rounds` AND ≥ `min_params`
  submodels exist at that round (RoundStart alone only means the last round *started*).
- **DFL complete** — all N nodes published a submodel at `total_rounds` (unanimity).
- **FAILED_PROCESS_DEATH** — a server's Popen exited mid-run.
- **FAILED_TIMEOUT / FAILED_STALL** — wall clock (`--completion-timeout`, default
  1800s) or no ledger progress (`--no-progress-timeout`, default 300s).
- Failures still write a partial `results.csv`, then the run is retried once after a
  full infra rebuild (`--no-retry` disables). A second failure is recorded and the
  suite moves on.

## Backends

`host_profile.INFRA_BACKEND` selects the Docker stack:

- **`anylog`** (primary, validated) — drives the AnyLog-co/docker-compose Makefile,
  host networking, needs `ANYLOG_LICENSE`. Targets the compose repo's `pre-develop`
  branch, where each node is one `docker-makefiles/<type>/node_configs.env` started with
  `make up ANYLOG_TYPE=<type>`. The harness generates `edgefl-master` and
  `edgefl-operator<i>` (container names `master` / `operator<i>`) from the committed
  `anylog-master` and `anylog-operator1` configs, so edit those templates (or
  `_master_values` / `_operator_values` in `infra_anylog.py`) to change settings for
  every node; edits to the generated dirs are overwritten on each `infra-up`.
  `DB_TYPE` in `anylog-operator1/node_configs.env` is inherited by all operators; with
  `sqlite` the harness skips the Postgres shard containers (`infra-down` still stops
  any left over from an earlier psql setup). Switching `DB_TYPE` needs
  `infra-down --volumes` first. The master is always sqlite. **MNIST needs `psql` for
  the operators**: on sqlite the seed fails (AnyLog turns the long `image` text into an
  unquoted `.blob` token the INSERT can't parse, seen on both the `ucsc-arm` and
  `2.1.2610` images), so set `OPERATOR_DB_TYPE = "psql"` in `host_profile.py`.
  Every operator gets its **own CLUSTER_NAME** — operators sharing a cluster
  replicate each other's data, which silently breaks per-Node shard isolation (the
  committed operator2 demo config shares a cluster for HA; don't copy that pattern).
- **`edgelake`** (open-source fallback, NOT fully working) — brings up containers and
  the master ledger fine, but the EdgeLake image's one-shot deployment script fails to
  self-configure operators (no DBMS connect, no Operator process; its policy push
  races container networking at boot and never retries). Known gap left for a future
  pass — see "Handoff" below. Config knobs for it (`DOCKER_INTERNAL_HOST`) are already
  plumbed.

## Troubleshooting

| symptom | likely cause |
|---|---|
| master up but no processes / operators can't sync | expired `ANYLOG_LICENSE` |
| `connection refused` to any 32X49 port | Docker Desktop host networking disabled |
| operator has no `mnist_fl` in `get databases` | Postgres shard not up, or wrong `DB_PORT` in its generated config |
| seed succeeds but `select count(*)` shows 0 | rows sit in partitions — query with `destination: network`, or check `get rows count where dbms = mnist_fl` (par_ tables) |
| `FAILED_SETUP: EdgeLake not ready` from `run` | bring infra up first: `python -m harness infra-up --nodes N` |
| run completes but CSV is thin | results settle-polling capped out; rerun collection manually via `harness.results.collect_run_csv` |
| operators sit at Operator `Not declared`; `get error log` shows `Failed to retrieve date and time ... '-$START_DATEd'` | each container clones AnyLog's `deployment-scripts` on first boot (branch `os-dev` under roy-local), and its `set_params.al` mishandled `START_DATE`. The harness comments `START_DATE` out of the generated operator configs. If you start operators by hand, remove `START_DATE=30` from the node's `node_configs.env`. |
| an operator shows `Cluster Member: False` / duplicate-IP errors in `get error log` | stale duplicate `cluster-operator*` policies accumulated on the blockchain from repeated container restarts over time (AnyLog's own declare-policy step isn't idempotent). No in-place fix — `infra-down --volumes` + fresh `infra-up` clears it. Don't leave the stack running unattended for weeks; tear it down between sessions. |

Useful direct probes (any node, REST):

```bash
curl -s http://127.0.0.1:32149 -H "User-Agent: AnyLog/1.23" -H "command: get processes"
curl -s http://127.0.0.1:32149 -H "User-Agent: AnyLog/1.23" -H "command: get databases"
curl -s http://127.0.0.1:32049 -H "User-Agent: AnyLog/1.23" -H "command: blockchain get <run_id>"
```

## Handoff — what's left (deliberately)

- **Hybrid runs** — config, env rendering, and completion predicate exist
  (`--mode hybrid --hybrid-centralized K`) but have never been executed; HARNESS_PLAN
  ordered hybrid last. Known caveat: which RoundStart a listener picks up each round
  is nondeterministic when both cohorts are publishing one — see the prereq note on
  step 9 in `HARNESS_PLAN.md`.
- **Tier 30 validation** — the machinery is tier-agnostic (ports/configs derive from
  the node index; tier-30 port math is collision-checked). Tiers 5, 10, and 15 have
  all run CFL+DFL clean on real hardware (Aug 2026); 30 is the plan's stretch tier,
  unvalidated. Watch RAM: each node_server loads TensorFlow.
- **EdgeLake backend** — see gap above. Options: drive operator config explicitly over
  REST (`connect dbms` + cluster/operator policies + `run operator`), or retry the
  image's deployment script once networking is up.
- **Plotting** — out of scope by decision; CSVs are long-format so a pandas/matplotlib
  layer can read past suites without re-running.
- **Datasets beyond MNIST** — the platform itself already supports this (data handlers
  are loaded dynamically via `MODULE_NAME`/`MODULE_FILE`; `chest_xrays_bbox` and
  `winniio` data handlers + env files already exist alongside MNIST's), but the
  harness's `datasets.py` registry only has an MNIST `DatasetProfile` wired up. Adding
  another is small: a new `DatasetProfile` entry (module name/file, logical db, train/
  test table names) plus a `seed.py` entry pointing at that dataset's insertion script.
  One wrinkle: `winniio_db_script.py` writes straight to Postgres with raw SQL rather
  than through AnyLog's REST insert like `store_data.py`/`chest_xrays_bbox_db_script.py`
  do, so wiring in Winniio specifically needs a slightly different `seed_operator` path,
  not just a registry line. chest_xrays_bbox would be the easier first addition — same
  REST-insert pattern as MNIST.
- **Failure thresholds as Config knobs, multi-person distributed runs** — future
  directions listed at the end of HARNESS_PLAN.md.
