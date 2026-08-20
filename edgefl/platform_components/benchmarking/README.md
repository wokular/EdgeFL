# EdgeFL Benchmarking

Per-round timing and accuracy metrics for the FL lifecycle, streamed into a single
time-series table (`benchmarkfl.fl_benchmarks`) that can be queried with SQL.

Design constraints the subsystem holds to:

- **Non-invasive** — instrumentation adds a dict write and a queue put on the training
  path; the network I/O happens on a daemon thread.
- **Never fatal** — a benchmarking failure cannot block or crash an FL loop. The
  accessor swallows construction errors and returns an inert benchmarker.
- **Off by default, env-configured** — no code change to move the collector or disable
  collection.

---

## What it measures

One row per measurement, tagged with `training_index`, `round_number`, and `node`.

**Per training node** (tagged `node1`, `node2`, …):

| Metric | Meaning |
|---|---|
| `polling_time_s` | Time spent waiting for the round-start signal on the blockchain. |
| `training_time_s` | Time spent in `train_model_params()` for that round. |
| `total_round_time_s` | End-to-end round time for that node (polling + training + publish). |
| `round_accuracy` | Local model accuracy after this round's training. |

**Per aggregator** (tagged `agg`):

| Metric | Meaning |
|---|---|
| `aggregation_time_s` | Time to fuse the collected updates into the new global model. |
| `first_to_last_arrival_s` | Spread between the earliest and latest node *publishing* its update — the straggler gap. |
| `straggling_node_id` | Numeric id of the node that published last (`-1` if unresolved). |

DFL runs also record `aggregation_time_s` per node, since each node aggregates for
itself.

---

## Configuration

Set in each process's env file — every node **and** the aggregator.

| Variable | Values | Default | Effect |
|---|---|---|---|
| `BENCHMARK_ENABLED` | `True`/`False` | `False` | Master switch for this process. When off, `record_simple_metric()` returns immediately. |
| `BENCHMARK_REST_CONN` | `host:port` | *(unset)* | REST address of the collector. In the default setup every process points at the same operator. |
| `BENCHMARK_FALLBACK` | `True`/`False` | `False` | Only consulted when `BENCHMARK_REST_CONN` is unset. Sends metrics to this process's own `EXTERNAL_IP` instead of a central collector. |

Two behaviors worth knowing:

- **Fallback is never implicit.** Leaving `BENCHMARK_REST_CONN` unset does *not*
  silently write to the node's own address; that requires `BENCHMARK_FALLBACK=True`.
- **The collector must run an Operator process.** A master node accepts the PUT and
  drops the rows silently. The aggregator's own `EXTERNAL_IP` is the master, which is
  exactly why it must be pointed at an operator explicitly. The `Benchmarker`
  probes for a running Operator at startup and disables itself with a warning if
  there isn't one.

If enabled but no target resolves, the process logs which variable to set and disables
benchmarking for itself. It does not crash.

---

## What changed in this branch

The `Benchmarker` class itself arrived earlier via the `fl-evals` cherry-pick. This
branch closes four gaps between it and the benchmarking work on `juanalvv/EdgeFL`.

### 1. `round_accuracy` is now recorded

Previously declared in `Benchmarker.metrics` but never emitted by anything — the
harness had no accuracy instrumentation at all, so confirming that training actually
converged meant hand-calling `/inference/{index}` per node before teardown.

The rollback feature had already changed `train_model_params()` to return a dict
carrying `initial_accuracy`/`final_accuracy`. `node_server.py` now mirrors
`final_accuracy` into `benchmarkfl` next to the existing `push_accuracy()` write, so
accuracy lands in every harness CSV alongside the timing metrics.

`node_accuracy` remains the source of truth for rollback decisions; this is the same
number in the benchmarking long format, not a second measurement.

### 2. `first_to_last_arrival_s` / `straggling_node_id` now measure node lateness

**This is a semantic fix, not a refactor — figures from before and after are not
comparable.**

The previous implementation timestamped each submodel when the *aggregator finished
fetching* it. That measures the aggregator's poll cadence rather than node lateness:
when several nodes land in the same poll — the common case — the spread collapses to
`0.000s`, and the "straggler" is whichever link the fetch loop happened to return last.

Nodes now stamp `published_ts` into their own submodel policy
(`Node.add_node_params()`), and the aggregator reads it back off the ledger. The gap is
now the real interval between the first and last node finishing their round.

Three details that matter:

- `published_ts` is stamped **once, before the insert-retry loop**, so a retry re-sends
  a byte-identical policy and `check_policy_inserted()` can still match it.
- Policy values are **cast with `_as_float()` on read**. AnyLog returns policy fields as
  strings even when inserted as numbers — the same reason every `round_number` read in
  this codebase is wrapped in `int()`. Without the cast, the span subtraction raises
  `TypeError`, which the poll loop's broad `except` would swallow into an infinite
  retry, hanging aggregation with a misleading "Waiting for file" log.
- Params whose policy carries no usable `published_ts` are skipped rather than
  defaulted, so a node running an older build degrades to "not counted" instead of
  poisoning the span with a bogus timestamp.

The straggler block is additionally wrapped in its own `try/except`. It sits on the
aggregation critical path, where the enclosing handler only retries the poll — so an
escaping error would stall the round rather than surface. Benchmarking must not be able
to do that.

The straggler id now comes from the policy's `node` field instead of being parsed out
of the submodel filename, which retires `_node_id_from_params_link()`.

### 3. `BENCHMARK_FALLBACK` supported

Third env flag from the upstream work, wired into the existing lazy accessor. Enables
the distributed topology where each process writes to its own operator instead of one
central collector. The harness always sets `BENCHMARK_REST_CONN` explicitly, so this
only affects manual runs.

### 4. Benchmarking config added to the manual env files

`env_files/mnist/*.env` had no benchmarking variables, so hand-run experiments
collected nothing. All four node files and the aggregator now carry a benchmarking
block pointing at operator1, mirroring how the rollback block is laid out.

### Deliberately not taken from upstream

The upstream branch constructs a `Benchmarker` eagerly at module import in both server
files. This branch keeps the lazy `get_benchmarker()` accessor instead: the harness
launches servers as subprocesses and injects env afterward, so import-time construction
both races the env and makes the module unimportable for tooling without
`BENCHMARK_REST_CONN` set. Metrics, names, and payload shape are identical either way.

---

## Running it

### Via the harness

Benchmarking is on automatically — `env_render.py` points every node and the aggregator
at node1's operator. From `edgefl/`, with the venv active:

```
python -m harness infra-up --nodes 5 --seed
python -m harness run --mode cfl --nodes 5 --rounds 10
python -m harness infra-down --nodes 5
```

Each run directory gets a `results.csv` of the raw `fl_benchmarks` rows for that
`training_index`, plus a `manifest.json`. See `harness/README.md` for the full runbook.

### By hand

1. Bring up EdgeLake and insert the MNIST data as usual.
2. Start the aggregator and node servers from `env_files/mnist/`. The benchmarking
   block is already configured to send to operator1 at `127.0.0.1:32149` — adjust if
   your operator1 REST port differs. (`mnist4.env` points `EXTERNAL_IP` at a non-local
   host; check its collector address before using it.)
3. Init the index and run rounds normally.
4. Read the metrics back from operator1:

```
sql benchmarkfl "select * from fl_benchmarks"
sql benchmarkfl "select node, metric_name, metric_value from fl_benchmarks where round_number = 3"
sql benchmarkfl "select round_number, avg(metric_value) from fl_benchmarks where metric_name = 'round_accuracy' group by round_number"
```

The `benchmarkfl` database is connected automatically at startup if absent.

### Sanity checks

- Every round should produce 4 rows per node plus 3 aggregator rows in CFL.
- `first_to_last_arrival_s` should be non-zero with more than one node. A constant
  `0.000s` means nodes are publishing without `published_ts` — check they're running
  this build.
- `round_accuracy` should trend upward across rounds and match the `node_accuracy`
  table's `final_accuracy` for the same round.

### If no metrics appear at all

The first suspect is the startup capability probe, not the instrumentation.
`Benchmarker._verify_target_availability()` parses `get processes` output and disables
benchmarking outright if it can't find a line whose first column is `Operator` in a
`running` state. Upstream has this probe commented out, so a parsing difference across
EdgeLake versions would show up here and not there. It logs
`"...has no Operator process running and can't ingest data"` when it trips.

Check what the collector actually reports:

```
curl -H "User-Agent: AnyLog/1.23" -H "command: get processes" http://127.0.0.1:32149
```

If the Operator row is present and running but the probe still disables, the parser —
not your deployment — is what needs adjusting.

---

## Interaction with rollback

The two features are independent but visible in the same table, which is the point of
landing them together:

- A rollback rewinds a node's weights, so its `round_accuracy` **drops** at the
  rollback round and re-climbs afterward. That trace is the clearest confirmation a
  rollback actually took effect.
- Auto-rollback reads `node_accuracy`, not `fl_benchmarks` — benchmarking is purely an
  observer and cannot influence a rollback decision.
- A rolled-back node skips its `initParams` download and trains from stale weights, so
  its `training_time_s` for that round is typically slightly lower.
- Rollback holds the round barrier (`_node_ready`) while restoring, which shows up as
  elevated `polling_time_s` on that node for the affected round.