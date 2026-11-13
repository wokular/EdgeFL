# Harness quickstart

Bring up N nodes, load MNIST, run a federated-learning benchmark, take it all down.
For how it works and the design decisions, see [README.md](README.md).

All commands run from the `edgefl/` directory with the repo venv's Python. If
`python -m harness` isn't found, use `../.venv/bin/python -m harness ...` (a venv that
was moved on disk keeps stale paths in `activate` and its console scripts).

## One-time setup

1. **Docker Desktop** with host networking enabled (Settings > Resources > Network).
2. **Compose repo**: clone `github.com/AnyLog-co/docker-compose`, check out
   **`pre-develop`**, and point `ANYLOG_COMPOSE_DIR` in `host_profile.py` at it.
   Add `docker-makefiles/edgefl-*/` to that repo's `.git/info/exclude`.
3. **Image access**: the image repository is private. Log in once from the compose repo
   (`make login DOCKER_LOGIN=<token from AnyLog>`; it replaces your `docker.io`
   credentials).
4. **License key** in your shell profile, wrapped in **single** quotes so the JSON
   quotes survive:

   ```bash
   export ANYLOG_LICENSE='<hex>{"company": "...", "expiration": "...", "type": "...", "max_nodes": 500}'
   ```

   Double quotes outside make the shell strip the inner ones, and AnyLog rejects the key
   (`Wrong license info format`). The harness checks this and stops with a message.
   Open a new shell after editing.
5. **Host profile**:

   ```bash
   cp harness/host_profile.example.py harness/host_profile.py
   ```

   Set `GITHUB_DIR`, `PYTHON_BIN` (the venv python), `ANYLOG_COMPOSE_DIR`, and
   `OPERATOR_DB_TYPE = "psql"`. MNIST needs Postgres on the operators: with sqlite the
   seed fails.

## Run a benchmark

```bash
cd edgefl

# 1. Containers: master + N operators (+ N Postgres shards). Prints progress per stage.
python -m harness infra-up --nodes 5

# 2. Load MNIST into every operator (adds rows each time: don't run it twice).
python -m harness seed --nodes 5

# 3. One run. CFL = central aggregator, DFL = nodes aggregate among themselves.
python -m harness run --mode cfl --nodes 5 --rounds 10
python -m harness run --mode dfl --nodes 5 --rounds 10
```

`infra-up --nodes 5 --seed` does steps 1 and 2 together.

**Check the seed before running.** Each operator should hold 1000 train and 200 test
rows. An operator can lose its whole first seed (its partition tables never get created),
which later shows up as a node failing to init:

```bash
for i in $(seq 1 5); do p=$((32049 + i*100)); printf "op$i "; curl -s -m 20 http://127.0.0.1:$p -H "User-Agent: AnyLog/1.23" -H "command: get rows count where dbms = mnist_fl" | grep -E "par_mnist_(train|test)" | awk -F'|' '{printf "%s ", $3}'; echo; done
```

An operator that prints nothing is empty. Re-seeding just that one is safe:
`cd data/mnist && python store_data.py 127.0.0.1:<rest-port> --db-name mnist_fl`.

## MedMNIST on sqlite

BloodMNIST stores only scalar rows (filename, label, round) in AnyLog and keeps the PNGs
on disk, so it runs on sqlite operators, which is the compose repo's default. In
`host_profile.py` set `OPERATOR_DB_TYPE = "sqlite"` (or delete the line), then rebuild:

```bash
python -m harness infra-down --nodes 10 --volumes
python -m harness infra-up --nodes 10 --dataset medmnist
python -m harness seed --nodes 10 --dataset medmnist      # --seed-rounds N to change the 10 default
python -m harness run --mode cfl --nodes 10 --rounds 3 --dataset medmnist --keep-servers
```

`infra-up --nodes 10 --dataset medmnist --seed` does the second and third commands together.

**Pass `--dataset medmnist` to `infra-up`, `seed` and `run`.** The default is `mnist`. The
dataset's database (`mydb` for MedMNIST, `mnist_fl` for MNIST) is created by `infra-up`, so
the operators can only serve the dataset they were brought up for. If they were started
without `--dataset medmnist`, `seed` streams rows into a database that does not exist (they
are counted as sent but never stored) and `run` fails every node with
`Error fetching datasets: 'Query'`. `seed` and `run` now stop with the rebuild commands
when they see that mismatch.

After a run that was started with the wrong dataset: stop it (`Ctrl-C`, then
`python -m harness stop` for any leftover servers), and re-run the block above.

**Check the seed before running.** Each operator should hold about 80-90% of 5,120 train
and 640 test rows (some rows are lost on sqlite; that is fine for the demo). The rows are
in partition tables, so a plain `select count(*)` returns 0. Use:

```bash
curl -s http://127.0.0.1:32149 -H "User-Agent: AnyLog/1.23" -H "command: get rows count where dbms=mydb and table=medmnist_train"
```

(operator `i` is on port `32{i}49`; operator 10 is on `33049`.)

- The first `seed` downloads `bloodmnist.npz` (md5-checked) into `edgefl/data/medmnist/`,
  then writes the PNGs to `edgefl/data/medmnist/published_images/`. Each operator gets
  512 train and 64 test slides per round.
- `--seed-rounds` must be at least the run's `--rounds`, or later rounds find no data.
- The seed sends about 400 rows per second in total. The AnyLog server closes every
  connection, so each row uses one local port and macOS holds it for ~30 s; faster than
  ~550 rows/s runs out of ports (`Errno 49 Can't assign requested address`).
  A full 10-round, 10-node seed takes about 2.5 minutes.
- MNIST still needs `psql`; seeding it on sqlite is refused with a message.
- After a `--keep-servers` run, use the GUI's **Inference** step: choose **MedMNIST**, set
  the header Node to a node (for example `localhost:8081`), and upload a PNG from
  `edgefl/data/medmnist/eval_images/`.

## Watch it and get results

Each run writes `edgefl/harness/runs/<timestamp>_<run_id>_a<attempt>/`:

| file | what |
|---|---|
| `logs/*.log` | stdout of the aggregator and each node server |
| `results.csv` | benchmark rows for the run |
| `manifest.json` | status, reason, timings |

```bash
# follow every server of the newest run
tail -f "$(ls -dt harness/runs/*/ | head -1)"logs/*.log
```

A healthy round logs `Step 2 Complete`, `Step 3 Complete` on each node and
`Step 4 Complete: model parameters aggregated` on the aggregator. Probe a container
directly over REST (node *i* answers on `32{i}49`, the master on `32049`):

```bash
curl -s http://127.0.0.1:32149 -H "User-Agent: AnyLog/1.23" -H "command: get error log" | tail -20
```

## Stop

```bash
python -m harness stop                       # leftover aggregator/node servers (host processes)
python -m harness infra-down --nodes 5       # containers, keep data
python -m harness infra-down --nodes 5 --volumes   # containers + blockchain + Postgres data
```

`infra-down` only touches Docker. A normal `run` stops its own servers; `stop` is for
interrupted runs. Use `--volumes` whenever you change a node config or the license key,
so old blockchain state doesn't carry over.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `cannot start servers, N port(s) already in use` | leftover servers hold 8080+: `python -m harness stop` |
| `ANYLOG_LICENSE has lost the quotes...` | single-quote the key in your profile (setup step 4), open a new shell |
| `pull access denied for anylogco/anylog-network` | not logged in to the image repository (setup step 3) |
| node `Failed to initialize node ... Response ended prematurely` | that operator is empty: check the seed (above) |
| `operator1 has no 'mydb' database` | operators were started for another dataset: `infra-down --volumes`, then `infra-up --dataset medmnist` |
| node `Error fetching datasets: 'Query'` | `run` used the default `mnist` dataset against MedMNIST data (or the reverse): pass the same `--dataset` everywhere |
| seed prints thousands of `Errno 49 Can't assign requested address` | local ports exhausted by too many short connections; the publisher paces itself now, re-run `seed` |
| `FAILED_SETUP: EdgeLake not ready` | run `infra-up` first |
| `FAILED_PROCESS_DEATH` | open the run's `logs/` and `manifest.json` for which server died and why |
| seed fails with `unrecognized token` in the operator error log | operators are on sqlite: set `OPERATOR_DB_TYPE = "psql"` and rebuild with `--volumes` |

## Inference

The harness has no inference command, and a normal `run` stops its servers when it
finishes. Add `--keep-servers` to leave them up after a completed run:

```bash
python -m harness run --mode cfl --nodes 5 --rounds 10 --keep-servers
```

When the run completes it prints each node's URL and a ready-made `curl`, for example
`curl -X POST http://localhost:8081/inference/<run_id>` (the index is the run id).
To use the GUI instead:

```bash
cd gui/edgefl-gui && npm start    # opens http://localhost:3000
```

In the GUI open **Inference** and set the top-right URL to a **node** (for example
`http://localhost:8081`), not the aggregator at 8080.

The kept servers are separate processes, not tied to the terminal: closing it won't stop
them. They stop on **your next harness command** (any command, including `run`,
`infra-down` or `stop`), or with `python -m harness stop`. A failed run never keeps its
servers. With `--mode dfl`, kept nodes keep training rounds after the run completes.

## Not yet validated

Run end to end on the `pre-develop` layout: bring-up, seed and a CFL `harness run` with
`--keep-servers` and inference, for MNIST on Postgres (10 nodes) and for MedMNIST on
sqlite (3 nodes, 3 rounds). Not run on it: DFL, hybrid, `suite`, MedMNIST at 10 nodes.

Known issues: the license does not activate (the master logs `Wrong license key`), so
network-wide queries fail, though training and inference do not need them. A MedMNIST
seed can lose roughly a fifth of its train rows (AnyLog logs `JSON file name: time
section with incorrect length`); check the row counts after seeding.
