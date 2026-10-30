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
| `FAILED_SETUP: EdgeLake not ready` | run `infra-up` first |
| `FAILED_PROCESS_DEATH` | open the run's `logs/` and `manifest.json` for which server died and why |
| seed fails with `unrecognized token` in the operator error log | operators are on sqlite: set `OPERATOR_DB_TYPE = "psql"` and rebuild with `--volumes` |

## Inference

The harness doesn't run inference, and its servers are gone when a run ends. To run
inference, start the servers by hand (see `Demo-READMEs/MNIST.md`) and call
`POST http://localhost:<node-port>/inference/<index>` on each node after training.

## Not yet validated

Bring-up, seed and the manual MNIST demo have been run on the `pre-develop` layout.
A full `harness run` and `suite` on it have not.
