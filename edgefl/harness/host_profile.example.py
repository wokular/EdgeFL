"""Host profile: per-machine values the harness can't derive.

Copy this to host_profile.py (same dir) and edit for your machine. host_profile.py is
gitignored; this example is the committed template.

    cp edgefl/harness/host_profile.example.py edgefl/harness/host_profile.py

The harness imports host_profile.py, merges these with derived per-node identity and the
run config, and renders one .env per node.
"""

# Absolute path to the EdgeFL repo root on this machine (becomes GITHUB_DIR).
GITHUB_DIR = "/Users/myname/Desktop/Code/EdgeFL"

# Host where the EdgeLake master + operators are reachable. Usually localhost; per-node
# ports are derived, so only the host goes here. Multi-person runs point at the
# master-owner's machine.
EDGELAKE_HOST = "127.0.0.1"

# EdgeLake master ports (the shared blockchain ledger). Operators are derived from the
# node index; only the master is fixed here.
MASTER_REST_PORT = 32049
MASTER_TCP_PORT = 32048

# Python interpreter for launching node_server / aggregator subprocesses. Needs the
# repo's requirements installed.
PYTHON_BIN = "python3"

# Docker socket / host. None uses the environment default, same as docker.from_env().
DOCKER_HOST = None

# How containers reach services published on THIS host (operator -> Postgres).
# Docker Desktop (mac/win): "host.docker.internal". Linux: usually "172.17.0.1".
# Only used by the edgelake backend (anylog uses host networking).
DOCKER_INTERNAL_HOST = "host.docker.internal"

# Which Docker stack runs the blockchain layer.
#   "anylog"   - PRIMARY on Apple Silicon: AnyLog-co/docker-compose repo (pre-develop),
#                host networking, needs ANYLOG_LICENSE exported. See AnyLog-Setup.md.
#   "edgelake" - open-source image via the in-repo EdgeLake Makefile, bridge networking.
INFRA_BACKEND = "anylog"

# anylog backend only: local checkout of github.com/AnyLog-co/docker-compose (pre-develop).
ANYLOG_COMPOSE_DIR = "/Users/myname/Code/docker-compose"

# anylog backend only: image tag passed to `make up` as TAG=. Omit to use the compose
# repo's Makefile default.
# ANYLOG_TAG = "2.1.2610"

# anylog backend only: operator database, "sqlite" or "psql". Omit to use the DB_TYPE in
# the compose repo's anylog-operator1 config. "psql" starts one Postgres shard per node.
# OPERATOR_DB_TYPE = "sqlite"

# Postgres superuser creds for (re)building operator data shards. Match the values in
# your EdgeLake operator env files (DB_USER / DB_PASSWD).
PSQL_USER = "demo"
PSQL_PASSWORD = "passwd"
