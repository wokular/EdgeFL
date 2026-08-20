"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse

from platform_components.EdgeLake_functions.blockchain_EL_functions import get_local_ip, \
    connect_to_db, get_all_databases, get_policies, fetch_data_from_db
from platform_components.benchmarking import get_benchmarker
from platform_components.node.node import Node
import asyncio
import logging
import pickle
import threading
import time
from dotenv import load_dotenv
import os
import argparse
import requests
import warnings

from uvicorn import run
from fastapi import FastAPI, HTTPException, status

from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional

from platform_components.node.rollback_manager import (
    RollbackConfig, load_rollback_config,
    get_accuracy_history, should_auto_rollback, select_rollback_round,
)

from platform_components.lib.logger.logger_config import configure_logging


warnings.filterwarnings("ignore")

load_dotenv()

edgelake_node_url = f'http://{os.getenv("EXTERNAL_IP")}'
edgelake_node_port = edgelake_node_url.split(":")[2]

configure_logging(f"node_server_{edgelake_node_port}")

logger = logging.getLogger(__name__)

# Initialize the Node instance
node_instance = None
listener_thread = None
stop_listening_thread = False

# Per-index pause flag toggled by /pause and /unpause. Separate from the
# DRIFT_HANDLING=pause auto-pause loop.
manual_pause_state = {}

# Latched so an invalid DRIFT_HANDLING only warns once per process.
_drift_handling_warning_logged = False

rollback_cfg: RollbackConfig = load_rollback_config()

# Guards the listener thread during rollback so aggregator weights can't
# overwrite the rolled-back model before it gets a chance to be used.
_node_ready = threading.Event()
_node_ready.set()  # starts in "ready" state; cleared only during rollback

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Self-init off the main thread so blockchain polling doesn't block startup.
    if os.getenv("SELF_START", "false").lower() == "true":
        threading.Thread(
            name="self-start",
            target=run_self_start,
            daemon=True
        ).start()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InitNodeRequest(BaseModel):
    replica_name: str
    replica_ip: str
    replica_port: str
    replica_index: str
    round_number: int


class RollbackRequest(BaseModel):
    round: int
    reason: Optional[str] = "manual"


class RollbackConfigUpdate(BaseModel):
    auto_enabled: Optional[bool] = None
    patience_rounds: Optional[int] = None
    min_delta: Optional[float] = None
    allow_manual: Optional[bool] = None
    log_events: Optional[bool] = None


def _initialize_node_for_index(replica_name, port, index, round_number):
    """Set up the Node for this index and start its listener thread.
    Used by /init-node and by self-start. Returns the mode label."""
    global node_instance, listener_thread

    ip = get_local_ip()
    module_name = os.getenv("MODULE_NAME")
    module_file = os.getenv("MODULE_FILE")
    db_name = os.getenv("LOGICAL_DATABASE")

    aggregation_mode = os.getenv("AGGREGATION_MODE", "centralized").lower()
    is_aggregator = aggregation_mode == "decentralized"
    min_params = int(os.getenv("MIN_PARAMS", "1"))

    if not node_instance:
        node_instance = Node(replica_name, ip, port, logger)

    if index not in node_instance.databases:
        node_instance.databases[index] = db_name

    node_instance.initialize_specific_node_on_index(index, module_name, module_file)
    node_instance.round_number[index] = round_number
    node_instance.is_aggregator[index] = is_aggregator
    node_instance.minParams[index] = min_params

    mode_label = "decentralized (DFL)" if is_aggregator else "centralized (CFL)"
    logger.info(f"{replica_name} initialized for ({index}) in {mode_label} mode at round {round_number}"
                + (f" with minParams={min_params}" if is_aggregator else ""))

    listener_thread = threading.Thread(
        name=f"{replica_name}--{index}",
        target=listen_for_start_round,
        args=(node_instance, index, lambda: stop_listening_thread)
    )
    listener_thread.daemon = True
    listener_thread.start()

    return mode_label


def _is_already_initialized_for_index(index):
    """True if the node has a data handler loaded for this index."""
    return (
        node_instance is not None
        and index in node_instance.indexes
        and index in node_instance.data_handlers
    )


@app.post('/init-node')
def init_node(request: InitNodeRequest):
    try:
        port = request.replica_port
        replica_name = request.replica_name
        index = request.replica_index
        most_recent_round = request.round_number

        # Already set up by self-start or a prior /init-node. The aggregator
        # ignores the response body so this is a no-op for it.
        if _is_already_initialized_for_index(index):
            logger.info(
                f"/init-node received for ({index}) but node is already initialized "
                f"(likely via SELF_START); ignoring request from aggregator."
            )
            return {
                'status': 'success',
                'message': f'Node already initialized for ({index}); ignoring redundant /init-node call'
            }

        mode_label = _initialize_node_for_index(replica_name, port, index, most_recent_round)

        return {
            'status': 'success',
            'message': f'Node initialized successfully in {mode_label} mode'
        }
    except ValueError as e:
        raise ValueError(
            f"No data found in the database: {os.getenv('LOGICAL_DATABASE')}"
        )
    except HTTPException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"/init-node - {str(e)}"
        )
    except ConnectionError as e:
        raise ConnectionError(
            f"Unable to access the database tables: {str(e)}"
        )


def _get_latest_round_start(index):
    """Return the RoundStart policy at this index with the highest
    round_number, or None."""
    try:
        headers = {
            'User-Agent': 'AnyLog/1.23',
            'command': f'blockchain get {index} where policy_type = RoundStart'
        }
        response = requests.get(edgelake_node_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json()
        if not data:
            return None

        latest = None
        latest_round = -1
        for item in data:
            policy = item.get(index)
            if not policy:
                continue
            r = int(policy.get('round_number', 0))
            if r > latest_round:
                latest_round = r
                latest = policy
        return latest
    except Exception as e:
        logger.error(f"[{index}] Error fetching RoundStart policies: {str(e)}")
        return None


def _publish_cold_start_round(replica_name, index):
    """Publish a round-1 RoundStart with empty initParams. The listener
    picks it up and trains from the data handler's initial weights
    (train_model_params falls back to those when paramsLink is empty at
    round 1). Tagged as dfl_aggregator since the bootstrapping node owns
    this round."""
    from platform_components.EdgeLake_functions.blockchain_EL_functions import (
        insert_policy, check_policy_inserted
    )
    edgelake_tcp_node_ip_port = os.getenv("EXTERNAL_TCP_IP_PORT")

    data = f'''<my_policy = {{"{index}" : {{
                                "index" : "{index}",
                                "policy_type": "RoundStart",
                                "node_type": "dfl_aggregator",
                                "round_number": 1,
                                "initParams": "",
                                "node_id": "{replica_name}",
                                "ip_port": "{edgelake_tcp_node_ip_port}",
                                "rest_ip_port": "{edgelake_node_url}"
                      }} }}>'''
    success = False
    while not success:
        response = insert_policy(edgelake_node_url, data)
        if response.status_code == 200:
            success = True
        else:
            time.sleep(3)
            if check_policy_inserted(edgelake_node_url, data):
                success = True
    logger.info(f"[{index}] COLD_START: published bootstrap RoundStart for round 1")


def run_self_start():
    """Autonomous init loop. Polls for an existing RoundStart at
    REPLICA_INDEX and joins it. If none exists and COLD_START=true,
    bootstraps round 1. Otherwise keeps polling."""
    replica_name = os.getenv("REPLICA_NAME")
    index = os.getenv("REPLICA_INDEX")

    if not replica_name or not index:
        logger.error("SELF_START=true but REPLICA_NAME/REPLICA_INDEX missing in env. Aborting self-start.")
        return

    cold_start = os.getenv("COLD_START", "false").lower() == "true"
    aggregation_mode = os.getenv("AGGREGATION_MODE", "centralized").lower()

    if cold_start and aggregation_mode != "decentralized":
        logger.warning(
            f"[{index}] COLD_START=true but AGGREGATION_MODE={aggregation_mode}. "
            "COLD_START only makes sense in decentralized mode; ignoring COLD_START."
        )
        cold_start = False

    port = os.getenv("EXTERNAL_IP", "").split(":")[-1] if os.getenv("EXTERNAL_IP") else ""

    logger.info(f"[{index}] SELF_START: polling for existing RoundStart at index '{index}'...")

    poll_attempts = 0
    while True:
        # Bail if /init-node beat us to initialization.
        if _is_already_initialized_for_index(index):
            logger.info(
                f"[{index}] SELF_START: aborting; node was already initialized "
                "via /init-node before self-start completed."
            )
            return

        latest = _get_latest_round_start(index)

        if latest:
            # Warm start at the latest network round.
            joined_round = int(latest.get('round_number', 1))
            logger.info(
                f"[{index}] SELF_START: found RoundStart at round {joined_round} "
                f"(node_id={latest.get('node_id')}); joining the network."
            )
            _initialize_node_for_index(replica_name, port, index, joined_round)
            return

        # No RoundStart found yet
        if cold_start:
            logger.info(f"[{index}] SELF_START + COLD_START: bootstrapping round 1.")
            _publish_cold_start_round(replica_name, index)

            # Brief settle, then warn if a peer also cold-started.
            time.sleep(2)
            try:
                headers = {
                    'User-Agent': 'AnyLog/1.23',
                    'command': f'blockchain get {index} where policy_type = RoundStart and round_number = 1'
                }
                response = requests.get(edgelake_node_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    bootstraps = response.json() or []
                    distinct_node_ids = {
                        item.get(index, {}).get('node_id')
                        for item in bootstraps
                        if index in item
                    }
                    if len(distinct_node_ids) > 1:
                        logger.warning(
                            f"[{index}] COLD_START RACE DETECTED: multiple bootstrap RoundStart "
                            f"policies at round 1 from node_ids {distinct_node_ids}. "
                            "See dflFeatures.txt OPEN QUESTIONS for fix plan."
                        )
            except Exception as e:
                logger.error(f"[{index}] Error during cold-start race check: {str(e)}")

            _initialize_node_for_index(replica_name, port, index, 1)
            return

        # No RoundStart and no COLD_START: keep polling
        poll_attempts += 1
        if poll_attempts % 30 == 0:  # log every ~60s
            logger.info(f"[{index}] SELF_START: still waiting for a RoundStart to appear...")
        time.sleep(2)


# Drift handling helpers.

def _get_drift_handling():
    """Validated DRIFT_HANDLING value. Falls back to 'none' on unset/invalid."""
    global _drift_handling_warning_logged
    val = os.getenv("DRIFT_HANDLING", "none").lower()
    if val not in ("none", "skip", "pause"):
        if not _drift_handling_warning_logged:
            logger.warning(
                f"Invalid DRIFT_HANDLING={val}; falling back to 'none'. "
                f"Valid values: none, skip, pause."
            )
            _drift_handling_warning_logged = True
        return "none"
    return val


def _get_highest_submodel_round(index):
    """Max round_number across all submodels at this index, or 0."""
    try:
        headers = {
            'User-Agent': 'AnyLog/1.23',
            'command': f'blockchain get {index} where node_type = training'
        }
        response = requests.get(edgelake_node_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return 0
        data = response.json() or []
        max_round = 0
        for item in data:
            policy = item.get(index)
            if not policy:
                continue
            r = int(policy.get('round_number', 0))
            if r > max_round:
                max_round = r
        return max_round
    except Exception as e:
        logger.error(f"[{index}] Error fetching highest submodel round: {str(e)}")
        return 0


def _get_network_lowest_round(index):
    """Slowest peer's progress: min over each node's max published round.
    None if no submodels exist."""
    try:
        headers = {
            'User-Agent': 'AnyLog/1.23',
            'command': f'blockchain get {index} where node_type = training'
        }
        response = requests.get(edgelake_node_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json() or []
        if not data:
            return None

        per_node_max = {}  # {node_name: max_round}
        for item in data:
            policy = item.get(index)
            if not policy:
                continue
            node_name = policy.get('node')
            if not node_name:
                continue
            r = int(policy.get('round_number', 0))
            per_node_max[node_name] = max(per_node_max.get(node_name, 0), r)

        if not per_node_max:
            return None
        return min(per_node_max.values())
    except Exception as e:
        logger.error(f"[{index}] Error fetching network lowest round: {str(e)}")
        return None


def _count_submodels_at_round(index, round_number):
    """Submodel count at the given round."""
    try:
        headers = {
            'User-Agent': 'AnyLog/1.23',
            'command': f'blockchain get {index} where node_type = training and round_number = {round_number}'
        }
        response = requests.get(edgelake_node_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return 0
        data = response.json() or []
        return len(data)
    except Exception as e:
        logger.error(f"[{index}] Error counting submodels at round {round_number}: {str(e)}")
        return 0


def _count_distinct_publishing_nodes(index):
    """Distinct nodes across all submodels at this index.

    TODO: counts ever-published, not currently-active. Swap for a
    recency-windowed count if churn becomes a problem.
    """
    try:
        headers = {
            'User-Agent': 'AnyLog/1.23',
            'command': f'blockchain get {index} where node_type = training'
        }
        response = requests.get(edgelake_node_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return 0
        data = response.json() or []
        distinct = set()
        for item in data:
            policy = item.get(index)
            if policy and policy.get('node'):
                distinct.add(policy['node'])
        return len(distinct)
    except Exception as e:
        logger.error(f"[{index}] Error counting distinct publishing nodes: {str(e)}")
        return 0


def _apply_skip_drift_if_needed(current_round, index):
    """Skip-mode check, called between training and publishing. Returns the
    round to publish to: latest_round if we're lagging by more than
    ROUND_LAG_THRESHOLD, else current_round unchanged. Caller should
    overwrite its local current_round with the return value."""
    if _get_drift_handling() != "skip":
        return current_round

    threshold = int(os.getenv("ROUND_LAG_THRESHOLD", "3"))
    latest_round = _get_highest_submodel_round(index)

    if latest_round - current_round > threshold:
        logger.info(
            f"[{index}] DRIFT skip: my_round={current_round}, network_latest={latest_round} "
            f"(threshold={threshold}); jumping ahead to round {latest_round}."
        )
        return latest_round
    return current_round


def _apply_pause_drift(nodeInstance, current_round, index):
    """Pause-mode check, called after publishing a submodel. Sleeps until
    peers catch up or PAUSE_MAX_SECONDS elapses. Refuses to pause when
    doing so would stall aggregation at the current round."""
    if _get_drift_handling() != "pause":
        return

    threshold = int(os.getenv("DRIFT_THRESHOLD", "3"))
    max_seconds = int(os.getenv("PAUSE_MAX_SECONDS", "60"))

    network_lowest = _get_network_lowest_round(index)
    if network_lowest is None:
        return  # no peer info yet; nothing to compare against

    if current_round - network_lowest <= threshold:
        return  # not too far ahead

    # Deadlock guard: skip the pause if our absence would stall aggregation.
    min_params = nodeInstance.minParams.get(index, 1)
    submodels_at_my_round = _count_submodels_at_round(index, current_round)
    distinct_active_nodes = _count_distinct_publishing_nodes(index)
    pausing_drops_below_min = (distinct_active_nodes - 1) < min_params

    if submodels_at_my_round < min_params and pausing_drops_below_min:
        logger.info(
            f"[{index}] DRIFT pause: would pause (my_round={current_round}, "
            f"network_lowest={network_lowest}) but round {current_round} has only "
            f"{submodels_at_my_round}/{min_params} submodels and active_nodes={distinct_active_nodes}; "
            f"skipping pause to avoid stalling aggregation."
        )
        return

    logger.info(
        f"[{index}] DRIFT pause: my_round={current_round}, network_lowest={network_lowest} "
        f"(threshold={threshold}); self-pausing for up to {max_seconds}s."
    )

    start_time = time.time()
    while time.time() - start_time < max_seconds:
        # Manual /pause takes precedence; bail and let it hold us.
        if manual_pause_state.get(index, False):
            logger.info(f"[{index}] DRIFT pause: manual /pause is active; deferring to manual control.")
            return
        time.sleep(2)
        latest_lowest = _get_network_lowest_round(index)
        if latest_lowest is None:
            continue
        if current_round - latest_lowest <= threshold:
            logger.info(
                f"[{index}] DRIFT pause: peers caught up (network_lowest={latest_lowest}); "
                f"resuming."
            )
            return

    logger.info(
        f"[{index}] DRIFT pause: PAUSE_MAX_SECONDS={max_seconds} reached; resuming anyway."
    )


def listen_for_start_round(nodeInstance, index, stop_event):
    current_round = nodeInstance.round_number[index]
    is_dfl = nodeInstance.is_aggregator.get(index, False)

    benchmarker = get_benchmarker()
    round_wait_started = time.time()

    logger.info(f"[{index}][Round {current_round}] Listening for start round {current_round}"
                + (" (DFL mode)" if is_dfl else ""))
    while True:
        try:
            # Honor manual /pause.
            if manual_pause_state.get(index, False):
                time.sleep(2)
                continue

            # DFL nodes listen for both aggregator and dfl_aggregator RoundStart policies
            if is_dfl:
                headers = {
                    'User-Agent': 'AnyLog/1.23',
                    'command': f'blockchain get {index} where round_number = {current_round} and policy_type = RoundStart'
                }
            else:
                headers = {
                    'User-Agent': 'AnyLog/1.23',
                    'command': f'blockchain get {index} where round_number = {current_round} and node_type = aggregator'
                }
            response = requests.get(edgelake_node_url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                if not data:
                    time.sleep(2)
                    continue
                round_data = data[0].get(index)

                if round_data:
                    # Block here if a rollback is in progress so its weights aren't
                    # overwritten the moment the next round fires.
                    _node_ready.wait()
                    # Resync round in case the rollback changed nodeInstance.round_number
                    current_round = nodeInstance.round_number[index]

                    logger.debug(f"[{index}] Round Data: {round_data}")
                    benchmarker.record_simple_metric(
                        index, current_round, nodeInstance.replica_name,
                        "polling_time_s", time.time() - round_wait_started)

                    paramsLink = round_data.get('initParams', '')
                    ip_port = round_data.get('ip_port', '')
                    rest_ip_port = round_data.get('rest_ip_port', '')

                    # If a rollback is pending, skip downloading initParams this round and
                    # train directly from the rolled-back weights already in the model.
                    # WARNING: this node's update will be stale relative to the current
                    # global model — W_agg_{current_round-1} — and may pull aggregation
                    # in an older direction. Use staleness-aware aggregation if this is
                    # a concern.
                    skip_download = nodeInstance._rollback_pending.get(index, False)
                    if skip_download:
                        nodeInstance._rollback_pending[index] = False
                        logger.warning(
                            f"[{index}] Round {current_round}: rollback active — "
                            f"training from W_agg_{nodeInstance._stale_round.get(index, '?')} "
                            f"instead of W_agg_{current_round - 1}. "
                            f"Stale gradient warning: aggregator will receive an update "
                            f"computed from an older global model."
                        )

                    training_started = time.time()
                    result = nodeInstance.train_model_params(paramsLink, current_round, ip_port, rest_ip_port, index, skip_download=skip_download)
                    benchmarker.record_simple_metric(
                        index, current_round, nodeInstance.replica_name,
                        "training_time_s", time.time() - training_started)

                    # Re-tag to the network's latest round if skip-mode says we're lagging.
                    publish_round = _apply_skip_drift_if_needed(current_round, index)
                    if publish_round != current_round:
                        current_round = publish_round

                    nodeInstance.add_node_params(current_round, result['model_path'], index)
                    logger.info(f"[{index}][Round {current_round}] Step 3 Complete: Model parameters published")
                    benchmarker.record_simple_metric(
                        index, current_round, nodeInstance.replica_name,
                        "total_round_time_s", time.time() - round_wait_started)

                    # Mirror this round's post-training accuracy into benchmarkfl so it
                    # lands in the harness CSV alongside the timing metrics. The
                    # node_accuracy table below stays the source of truth for rollback;
                    # this is the same number in the benchmarking long format.
                    benchmarker.record_simple_metric(
                        index, current_round, nodeInstance.replica_name,
                        "round_accuracy", result['final_accuracy'])

                    # Write initial_accuracy and final_accuracy for this round to AnyLog
                    # table "node_accuracy" — also the source auto-rollback reads from.
                    nodeInstance.push_accuracy(index, current_round, result['initial_accuracy'],
                                               result['final_accuracy'], result['model_path'])

                    # DFL: after training and publishing, aggregate from peers
                    if is_dfl:
                        logger.info(f"[{index}][Round {current_round}] DFL: Starting peer aggregation")
                        agg_thread = threading.Thread(
                            name=f"{nodeInstance.replica_name}--{index}--dfl-agg-r{current_round}",
                            target=dfl_aggregate_round,
                            args=(nodeInstance, current_round, index)
                        )
                        agg_thread.daemon = True
                        agg_thread.start()

                    # Throttle here if we're getting too far ahead of peers.
                    _apply_pause_drift(nodeInstance, current_round, index)

                    current_round += 1
                    # Publish the round back to shared state — /rollback and the resync
                    # above both read nodeInstance.round_number, not this local.
                    nodeInstance.round_number[index] = current_round
                    round_wait_started = time.time()
                    logger.info(f"[{index}][Round {current_round}] Listening for start round {current_round}")

                    # Auto-rollback: runs only when ROLLBACK_AUTO_ENABLED=true.
                    # CFL only for now — in DFL the peer-aggregation thread loads weights
                    # outside the _node_ready guard, so a rollback could be clobbered.
                    if rollback_cfg.auto_enabled and not is_dfl:
                        db_name = os.getenv("LOGICAL_DATABASE", "mnist_fl")
                        # Retry until AnyLog commits the streaming write for this round.
                        # push_accuracy() uses streaming mode so the row may not be
                        # queryable immediately — poll until it appears or give up.
                        completed_round = current_round - 1
                        history = []
                        for attempt in range(6):
                            history = get_accuracy_history(index, db_name, edgelake_node_url, nodeInstance.replica_name)
                            if history and int(history[-1].get("round_number", 0)) >= completed_round:
                                break
                            logger.debug(f"[{index}] Waiting for round {completed_round} accuracy to commit (attempt {attempt + 1}/6)")
                            time.sleep(5)
                        if should_auto_rollback(history, rollback_cfg):
                            target = select_rollback_round(history, rollback_cfg)
                            logger.info(f"[{index}] Auto-rollback triggered: target round={target}")
                            _node_ready.clear()
                            try:
                                nodeInstance.rollback_to_round(index, target, reason="automatic", trigger_type="automatic")
                                logger.info(f"[{index}] Auto-rollback complete, resuming from round {current_round}")
                            except Exception as rollback_err:
                                logger.error(f"[{index}] Auto-rollback failed: {rollback_err}")
                            finally:
                                _node_ready.set()

            time.sleep(5)
        except Exception as e:
            logger.error(f"[{index}] Error in listener thread: {str(e)}")
            time.sleep(2)


def dfl_aggregate_round(nodeInstance, round_number, index):
    """Wait for peer submodels, aggregate once minParams are in, update the
    local model, and publish RoundStart for the next round."""
    min_params = nodeInstance.minParams.get(index, 1)
    decoded_params = {}
    check_chances = 5

    logger.info(f"[{index}][Round {round_number}] DFL: Waiting for {min_params} peer submodels")

    while True:
        try:
            headers = {
                'User-Agent': 'AnyLog/1.23',
                'command': f'blockchain get {index} where round_number={round_number} and node_type=training'
            }
            response = requests.get(edgelake_node_url, headers=headers)
            response.raise_for_status()

            result = response.json()
            if result:
                node_params_links = [
                    item.get(index).get('trained_params_local_path')
                    for item in result
                    if index in item
                ]
                ip_ports = [
                    item.get(index).get('ip_port')
                    for item in result
                    if index in item
                ]
                rest_ip_ports = [
                    item.get(index).get('rest_ip_port')
                    for item in result
                    if index in item
                ]

                nodeInstance.fetch_decoded_params(
                    decoded_params_dict=decoded_params,
                    node_param_download_links=node_params_links,
                    ip_ports=ip_ports,
                    rest_ip_ports=rest_ip_ports,
                    index=index
                )

            if len(decoded_params) >= min_params or (decoded_params and not check_chances):
                # Aggregate
                aggregation_started = time.time()
                aggregated_params_link = nodeInstance.aggregate_model_params(
                    decoded_params=list(decoded_params.values()),
                    round_number=round_number,
                    index=index
                )
                get_benchmarker().record_simple_metric(
                    index, round_number, nodeInstance.replica_name,
                    "aggregation_time_s", time.time() - aggregation_started)
                logger.info(f"[{index}][Round {round_number}] DFL: Aggregated {len(decoded_params)} submodels")

                # Update local model with aggregated weights
                local_path = f"{nodeInstance.file_write_destination}/{index}/{round_number}-{nodeInstance.name}_update.json"
                with open(local_path, "rb") as f:
                    data = pickle.load(f)

                if data and 'newUpdates' in data:
                    weights = nodeInstance.decode_params(data['newUpdates'])
                else:
                    logger.error(f"[{index}] Invalid aggregated data")
                    return

                nodeInstance.data_handlers[index].update_model(weights)

                # Publish RoundStart for next round so peers can pick it up
                next_round = round_number + 1
                nodeInstance.start_round(aggregated_params_link, next_round, index, node_type="dfl_aggregator")
                logger.info(f"[{index}][Round {round_number}] DFL: Published RoundStart for round {next_round}")
                return

            if decoded_params and check_chances:
                check_chances -= 1

            if not decoded_params and not check_chances:
                check_chances = 5

        except Exception as e:
            logger.error(f"[{index}][Round {round_number}] DFL aggregation error: {str(e)}")

        time.sleep(2)


# Extracts initParams from the policy at the specified index
def get_most_recent_agg_params(index):
    policy_name = f"{index}-r"
    agg_params = None

    try:
        headers = {
            'User-Agent': 'AnyLog/1.23',
            'command': f'blockchain get {index}'
        }
        response = requests.get(edgelake_node_url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            if data:
                policy = data[0]
                policy_data = policy[policy_name]
                agg_params = policy_data["initParams"]

        return agg_params
    except Exception as e:
        logger.error(f"[{index}] Error in extracting round number: {str(e)}")


@app.post('/inference/{index}', response_class=PlainTextResponse)
def inference(index):
    """Inference on current model w/ data passed in."""
    try:
        logger.info(f"[{index}] received inference request")
        if not index:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Index must be specified."
            )
        results = node_instance.inference(index)
        response = {
                    'index': f'{index}',
                    'status': 'success',
                    'message': 'Inference completed successfully',
                    'model_accuracy': f'{str(results)}'
                    }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

class InferenceRequest(BaseModel):
    input: list
    index: str


class PauseRequest(BaseModel):
    index: str


@app.post('/pause')
def pause_index(request: PauseRequest):
    """Halt this index's listener until /unpause."""
    index = request.index
    if node_instance is None or index not in node_instance.indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index {index} not initialized on this node."
        )
    manual_pause_state[index] = True
    logger.info(f"[{index}] Manually paused via /pause endpoint.")
    return {
        'status': 'success',
        'message': f'Index {index} paused.'
    }


@app.post('/unpause')
def unpause_index(request: PauseRequest):
    """Resume this index's listener."""
    index = request.index
    if node_instance is None or index not in node_instance.indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index {index} not initialized on this node."
        )
    manual_pause_state[index] = False
    logger.info(f"[{index}] Manually unpaused via /unpause endpoint.")
    return {
        'status': 'success',
        'message': f'Index {index} unpaused.'
    }

@app.post('/infer')
def direct_inference(request: InferenceRequest):
    """Inference on current model w/ data passed in."""
    try:
        float_list = request.input
        index = request.index
        results = node_instance.direct_inference(index, float_list)
        response = {
            'prediction': str(results),
        }
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error executing inference on model. Check inference function in data handler"
        )

@app.get('/accuracy-report', response_class=PlainTextResponse)
def accuracy_report(index: str = None):
    """
    Query node_accuracy from AnyLog and print one table per index_name.

    Usage:
        curl http://localhost:8080/accuracy-report
        curl "http://localhost:8080/accuracy-report?index=mnist"
    """
    try:
        db_name = os.getenv("LOGICAL_DATABASE", "mnist_fl")
        where_clause = f"WHERE index_name = '{index}'" if index else ""
        sql = (
            f"SELECT node_name, index_name, round_number, initial_accuracy, final_accuracy "
            f"FROM node_accuracy {where_clause} ORDER BY index_name, round_number"
        )
        query = f'sql {db_name} format=json "{sql}"'

        operators = get_policies(edgelake_node_url, index='operator')
        rows = []
        for op in operators:
            rest_url = f"http://{op['ip']}:{op['rest_port']}"
            tcp_addr = f"{op['ip']}:{op['port']}"
            try:
                payload = fetch_data_from_db(rest_url, query, tcp_addr)
                rows.extend(payload.get("Query", []) if isinstance(payload, dict) else [])
            except Exception:
                pass

        if not rows:
            return "No accuracy data found in node_accuracy.\n"

        # Group by (index_name, node_name) so each node gets its own table
        groups: dict[tuple, list] = {}
        for row in rows:
            key = (row.get('index_name', 'unknown'), row.get('node_name', 'unknown'))
            groups.setdefault(key, []).append(row)

        lines = []
        # Column headers:
        #   round        — training round number
        #   global@node  — initial_accuracy: how well W_agg_{R-1} performs on THIS node's
        #                  local test set before any training (measures global model quality
        #                  from this node's perspective)
        #   after_train  — final_accuracy: accuracy after this node's local fine-tuning
        #   contributed  — improvement this node added (after_train - global@node)
        #   Δ_global     — change in global@node vs previous round (is the global model
        #                  improving for this node? negative = rollback candidate)
        header = f"  {'round':>5}  {'global@node':>11}  {'after_train':>11}  {'contributed':>11}  {'Δ_global':>8}"
        sep    = f"  {'-----':>5}  {'-----------':>11}  {'-----------':>11}  {'-----------':>11}  {'--------':>8}"
        for (idx_name, node_name) in sorted(groups):
            idx_rows = groups[(idx_name, node_name)]
            lines.append(f"\n{'=' * 60}")
            lines.append(f"  index: {idx_name}   node: {node_name}")
            lines.append(
                f"  global@node = accuracy of global model on this node's data (pre-train)\n"
                f"  Δ_global    = change vs previous round — negative means rollback candidate"
            )
            lines.append(f"{'=' * 60}")
            lines.append(header)
            lines.append(sep)
            prev_initial = None
            for row in idx_rows:
                init_acc  = float(row['initial_accuracy'])
                final_acc = float(row['final_accuracy'])
                contributed = final_acc - init_acc
                if prev_initial is None:
                    delta_str = f"{'--':>8}"
                else:
                    delta = init_acc - prev_initial
                    sign = "+" if delta >= 0 else ""
                    marker = "  !" if delta < -5 else ""
                    delta_str = f"{sign}{delta:>7.1f}{marker}"
                prev_initial = init_acc
                lines.append(
                    f"  {row['round_number']:>5}  "
                    f"{init_acc:>10.1f}%  "
                    f"{final_acc:>10.1f}%  "
                    f"{'+' if contributed >= 0 else ''}{contributed:>10.1f}%  "
                    f"{delta_str}"
                )
        return "\n".join(lines) + "\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/rollback')
def rollback(request: RollbackRequest):
    """
    Manual rollback to a specific round.
    Fetches the aggregator-published model weights for that round via the blockchain
    RoundStart policy and loads them as the active model.
    """
    if not rollback_cfg.enabled:
        raise HTTPException(status_code=403, detail="Rollback is disabled on this node")
    if not rollback_cfg.allow_manual:
        raise HTTPException(status_code=403, detail="Manual rollback is disabled on this node")
    if not node_instance:
        raise HTTPException(status_code=400, detail="Node is not initialized")

    index = next(iter(node_instance.indexes), None)
    if not index:
        raise HTTPException(status_code=400, detail="No index initialized on this node")

    # DFL peer aggregation loads weights from its own thread, outside the _node_ready
    # guard the listener honors — a rollback there can be silently overwritten.
    if node_instance.is_aggregator.get(index, False):
        raise HTTPException(
            status_code=501,
            detail="Rollback is not supported in decentralized (DFL) mode yet — CFL only"
        )

    _node_ready.clear()  # pause the listener thread before touching model weights
    try:
        result = node_instance.rollback_to_round(
            index, request.round,
            reason=request.reason or "manual",
            trigger_type="manual",
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _node_ready.set()  # always resume, even on error


@app.get('/rollback/config')
def get_rollback_config():
    """Return the current rollback configuration (env defaults + any runtime overrides)."""
    return {
        "rollback_enabled":  rollback_cfg.enabled,
        "auto_enabled":      rollback_cfg.auto_enabled,
        "patience_rounds":   rollback_cfg.patience_rounds,
        "min_delta":         rollback_cfg.min_delta,
        "allow_manual":      rollback_cfg.allow_manual,
        "log_events":        rollback_cfg.log_events,
    }


@app.put('/rollback/config')
def update_rollback_config(request: RollbackConfigUpdate):
    """Update rollback config at runtime (in-memory only, does not persist to .env)."""
    global rollback_cfg
    if request.auto_enabled is not None:
        rollback_cfg.auto_enabled = request.auto_enabled
    if request.patience_rounds is not None:
        rollback_cfg.patience_rounds = request.patience_rounds
    if request.min_delta is not None:
        rollback_cfg.min_delta = request.min_delta
    if request.allow_manual is not None:
        rollback_cfg.allow_manual = request.allow_manual
    if request.log_events is not None:
        rollback_cfg.log_events = request.log_events

    return {"status": "success", "config": {
        "rollback_enabled":  rollback_cfg.enabled,
        "auto_enabled":      rollback_cfg.auto_enabled,
        "patience_rounds":   rollback_cfg.patience_rounds,
        "min_delta":         rollback_cfg.min_delta,
        "allow_manual":      rollback_cfg.allow_manual,
        "log_events":        rollback_cfg.log_events,
    }}


@app.get('/rollback/history')
def rollback_history(index: str = None):
    """
    Query rollback_events from AnyLog via run client () — the distributed network path.
    Optional ?index= filter to scope by training index.
    """
    try:
        db_name = os.getenv("LOGICAL_DATABASE", "mnist_fl")
        where_clause = f"WHERE index_name = '{index}'" if index else ""
        sql = (
            f"SELECT node_name, index_name, trigger_type, from_round, to_round, reason, status "
            f"FROM rollback_events {where_clause} ORDER BY index_name, from_round"
        )
        tcp_addr = os.getenv("EXTERNAL_TCP_IP_PORT", "")
        try:
            payload = fetch_data_from_db(edgelake_node_url, f'sql {db_name} format=json "{sql}"', tcp_addr)
        except Exception:
            return {"events": []}
        rows = payload.get("Query", []) if isinstance(payload, dict) else payload
        return {"events": rows if rows else []}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    global port
    parser = argparse.ArgumentParser(description="Run the Node Server.")
    parser.add_argument('--port', type=int, default=8080, help="Port to run the server on.")
    args = parser.parse_args()

    run(
    "node_server:app",
        host="0.0.0.0",
        port=args.port,
        reload=False
    )
