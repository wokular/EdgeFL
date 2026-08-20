"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""
import json
import os
import pickle
import time
from asyncio import sleep

import requests
from dotenv import load_dotenv

from platform_components.EdgeLake_functions.blockchain_EL_functions import insert_policy, check_policy_inserted
from platform_components.EdgeLake_functions.mongo_file_store import copy_file_to_container, copy_file_from_container
from platform_components.EdgeLake_functions.mongo_file_store import read_file
from platform_components.base_fl_participant import BaseFLParticipant

load_dotenv()


class Node(BaseFLParticipant):
    def __init__(self, replica_name, ip, port, logger):
        super().__init__(replica_name, logger)

        self.replica_name = replica_name
        self.node_ip = ip
        self.node_port = port

        self.logger.debug("Node initializing")

        # ===== Node-specific state
        self.data_batches = {}

        # DFL state (per-index)
        self.is_aggregator = {}   # {index: True/False}
        self.minParams = {}       # {index: int}
        self.end_round = {}       # {index: int}

        # Rollback state: set by rollback_to_round(), consumed by train_model_params()
        self._rollback_pending = {}  # {index: bool} — skip initParams download next round
        self._stale_round = {}       # {index: int}  — which round was rolled back to
        # =====

    def initialize_specific_node_on_index(self, index, module_name, module_path):
        self.initialize_index(index)
        self.set_module_at_index(index, module_name, module_path)
        self.initialize_training_app_on_index(index)
        self.initialize_file_write_paths_on_index(index)

    '''
    add_data_batch(data)
        - Adds passed in data to local storage
        - Used for simulating data stream
        - Assumes data is in correct format for model / datahandler
    '''
    def add_data_batch(self, index, data):
        self.data_batches[index].append(data)

    '''
    add_node_params()
        - Returns current node model parameters to edgefl via event listener
    '''
    def add_node_params(self, round_number, model_metadata, index):
        self.logger.debug(f"[{index}] in add_node_params")
        try:
            # Stamped once, before the retry loop: a retry must re-send a byte-identical
            # policy or check_policy_inserted() below can't match it. This is the node's
            # own view of when it finished the round, which is what makes
            # first_to_last_arrival_s measure node lateness rather than aggregator
            # polling jitter.
            published_ts = time.time()
            data = f'''<my_policy = {{"{index}" : {{
                                "node" : "{self.replica_name}",
                                "round_number" : {round_number},
                                "policy_type": "submodel",
                                "index": "{index}",
                                "node_type": "training",
                                "published_ts": {published_ts},
                                "ip_port": "{self.edgelake_tcp_node_ip_port}",
                                "rest_ip_port": "{self.edgelake_node_url}",
                                "trained_params_local_path": "{model_metadata}"
            }} }}>'''

            success = False
            while not success:
                self.logger.debug(f"[{index}] Attempting insert")
                response = insert_policy(self.edgelake_node_url, data)
                if response.status_code == 200:
                    success = True
                else:
                    sleep(5)
                    if check_policy_inserted(self.edgelake_node_url, data):
                        success = True

            self.logger.debug(f"[{index}] Submitting results for round {round_number}")

            return {
                'status': 'success',
                'message': 'node model parameters added successfully'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    '''
    train_model_params(aggregator_model_params)
        - Uses updated aggregator model params and updates local model
        - Gets local data and runs training on updated model
    '''
    def train_model_params(self, aggregator_model_params_db_link, round_number, ip_ports, rest_ip_port, index, skip_download=False):
        self.logger.debug(f"[{index}] in train_model_params for round {round_number}")
        if index not in self.data_handlers:
            self.logger.warning(f"[{index}] data handler missing, initializing now")
            self.initialize_training_app_on_index(index)

        # First round initialization
        if round_number == 1 and not aggregator_model_params_db_link:
            weights = self.data_handlers[index].get_weights()
            self.data_handlers[index].update_model(weights)
            self.logger.info(f"[{index}] Round {round_number}: loaded initial random weights (no aggregated model yet)")
        elif skip_download:
            # Rollback active: model already holds the rolled-back weights — skip fetch and load.
            # This round's gradient will be stale relative to W_agg_{round_number - 1}.
            stale_from = self._stale_round.get(index, '?')
            self.logger.warning(
                f"[{index}] Round {round_number}: training from rolled-back weights "
                f"(W_agg_{stale_from}) instead of W_agg_{round_number - 1}. "
                f"This node's update will be stale. Consider staleness-aware aggregation."
            )
        else:
            try:
                # Extract the key from the URL
                filename = aggregator_model_params_db_link.split('/')[-1]
                if self.docker_running:
                    response = copy_file_from_container(os.path.join(self.tmp_dir, index), self.docker_container_name, rest_ip_port, aggregator_model_params_db_link, f'{self.file_write_destination}/{index}/{filename}', ip_ports)
                else:
                    response = read_file(rest_ip_port, aggregator_model_params_db_link, f'{self.file_write_destination}/{index}/{filename}', ip_ports)

                if response.status_code == 200:
                    sleep(1)
                    with open(
                            f'{self.file_write_destination}/{index}/{filename}',
                            'rb') as f:
                        data = pickle.load(f)

                # Ensure the data is valid and decode the parameters
                if data and 'newUpdates' in data:
                    weights = self.decode_params(data['newUpdates'])
                else:
                    self.logger.error(f"[{index}] Invalid data or 'newUpdates' missing in Firestore response: {data}")
                    raise ValueError(f"[{index}] Invalid data or 'newUpdates' missing in Firestore response: {data}")

                self.data_handlers[index].update_model(weights)
                self.logger.info(f"[{index}] Round {round_number}: loaded weights from '{filename}'")
            except Exception as e:
                self.logger.error(f"[{index}] Error getting weights: {str(e)}")
                raise

        # Accuracy of the global weights on this node's local test set (pre-training baseline)
        initial_accuracy = self.data_handlers[index].run_inference()

        # Train model
        model_params = self.data_handlers[index].train(round_number)
        self.logger.info(f"[{index}][Round {round_number}] Step 2 Complete: Model training done")

        # Accuracy after local training — change from initial_accuracy shows this node's contribution
        final_accuracy = self.data_handlers[index].run_inference()

        # Save and return new weights
        encoded_params = self.encode_params(model_params)
        file = f"{round_number}-replica-{self.replica_name}.pkl"
        os.makedirs(os.path.dirname(f"{self.file_write_destination}/{index}/"), exist_ok=True)
        file_name = f"{self.file_write_destination}/{index}/{file}"
        with open(f"{file_name}", "wb") as f:
            f.write(encoded_params)

        if self.docker_running:
            self.logger.debug(f'[{index}] written to container at {f"{self.docker_file_write_destination}/{index}/{file}"}')
            copy_file_to_container(os.path.join(self.tmp_dir, index), self.docker_container_name, self.edgelake_node_url, file_name, f"{self.docker_file_write_destination}/{index}/{file}")
            # Return dict so the caller gets the model path AND both accuracy snapshots for storage
            return {'model_path': f'{self.docker_file_write_destination}/{index}/{file}',
                    'initial_accuracy': initial_accuracy,
                    'final_accuracy': final_accuracy}
        # Return dict so the caller gets the model path AND both accuracy snapshots for storage
        return {'model_path': file_name,
                'initial_accuracy': initial_accuracy,
                'final_accuracy': final_accuracy}

    def rollback_to_round(self, index: str, round_num: int, reason: str = "manual", trigger_type: str = "manual") -> dict:
        """
        Find the aggregator-published model for round_num, fetch and load its weights,
        update round state, and log the event.
        Raises ValueError/RuntimeError on failure so the caller can return a clean error response.
        """
        from platform_components.node.rollback_manager import (
            find_roundstart_policy, fetch_and_load_weights, log_rollback_event
        )

        from_round = self.round_number.get(index, 0)

        # Round N's aggregated weights (W_agg_N) are published as the initParams
        # of round N+1's RoundStart policy — not round N's.  Fetching round N's
        # policy would give W_agg_{N-1}, which predates the round we want.
        policy = find_roundstart_policy(index, round_num + 1, self.edgelake_node_url)
        if not policy:
            log_rollback_event(self, index, trigger_type, from_round, round_num, reason, "error")
            raise ValueError(
                f"No RoundStart policy found for round {round_num + 1} "
                f"(required to restore aggregated weights from round {round_num})"
            )

        try:
            model_path = fetch_and_load_weights(self, index, policy)
        except Exception:
            log_rollback_event(self, index, trigger_type, from_round, round_num, reason, "error")
            raise

        # Signal the listener thread to skip initParams next round and train from these weights.
        self._rollback_pending[index] = True
        self._stale_round[index] = round_num

        log_rollback_event(self, index, trigger_type, from_round, round_num, reason, "success")
        self.logger.info(f"[{index}] Rolled back from round {from_round} to round {round_num}")

        return {
            "status": "success",
            "rolled_back_to_round": round_num,
            "model_path": model_path,
            "message": f"Model rolled back to round {round_num}",
        }

    def push_accuracy(self, index, round_number, initial_accuracy, final_accuracy, model_path):
        # Build the row that will be stored in AnyLog under table "node_accuracy"
        row = {
            "node_name":        self.replica_name,
            "index_name":       index,
            "round_number":     round_number,
            "initial_accuracy": round(initial_accuracy, 4),
            "final_accuracy":   round(final_accuracy, 4),
            "model_path":       model_path
        }
        # AnyLog expects data via PUT with these headers to route the row to the right db/table
        headers = {
            "type":         "json",
            "dbms":         self.databases[index],   # logical db name, "mnist_fl"
            "table":        "node_accuracy",
            "mode":         "streaming",
            "Content-Type": "text/plain"
        }
        try:
            self.logger.info(f"[{index}] Pushing accuracy row: {row}")
            response = requests.put(url=self.edgelake_node_url, data=json.dumps(row), headers=headers)
            # Log the status code so we can confirm AnyLog accepted the row
            self.logger.info(f"[{index}][Round {round_number}] Accuracy stored: initial={initial_accuracy:.2f}%  final={final_accuracy:.2f}%  status={response.status_code}")
        except Exception as e:
            self.logger.error(f"[{index}] Failed to push accuracy: {str(e)}")
