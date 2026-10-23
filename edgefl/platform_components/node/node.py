"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""
import os
import pickle
import time

from dotenv import load_dotenv

from platform_components.EdgeLake_functions.blockchain_EL_functions import insert_policy, check_policy_inserted
from platform_components.EdgeLake_functions.mongo_file_store import copy_file_to_container, copy_file_from_container
from platform_components.EdgeLake_functions.mongo_file_store import read_file
from platform_components.base_fl_participant import BaseFLParticipant
from platform_components.helpers.LoadClassFromFile import load_class_from_file

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
        # =====

    def initialize_specific_node_on_index(self, index, module_name, module_path):
        self.initialize_index(index)
        self.set_module_at_index(index, module_name, module_path)
        self.initialize_training_app_on_index(index)
        self.initialize_file_write_paths_on_index(index)

    def initialize_training_app_on_index(self, index):
        training_app_path = os.path.join(self.training_application_dir, self.module_paths[index])
        class_name = self.module_names[index]
        try:
            TrainingApp_class = load_class_from_file(training_app_path, class_name)
        except Exception as error:
            message = f"Could not load training class '{class_name}' from '{training_app_path}': {error}"
            self.logger.error(f"[{index}] {message}")
            raise RuntimeError(message) from error
        self.data_handlers[index] = TrainingApp_class(self.replica_name) # Create an instance at index

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
            data = f'''<my_policy = {{"{index}" : {{
                                "node" : "{self.replica_name}",
                                "round_number" : {round_number},
                                "policy_type": "submodel",
                                "index": "{index}",
                                "node_type": "training",
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
                    time.sleep(5)
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
    def train_model_params(self, aggregator_model_params_db_link, round_number, ip_ports, rest_ip_port, index):
        self.logger.debug(f"[{index}] in train_model_params for round {round_number}")

        # First round initialization
        if round_number == 1 and not aggregator_model_params_db_link:
            weights = self.data_handlers[index].get_weights()
        else:
            try:
                # Extract the key from the URL
                filename = aggregator_model_params_db_link.split('/')[-1]
                if self.docker_running:
                    response = copy_file_from_container(os.path.join(self.tmp_dir, index), self.docker_container_name, rest_ip_port, aggregator_model_params_db_link, f'{self.file_write_destination}/{index}/{filename}', ip_ports)
                else:
                    response = read_file(rest_ip_port, aggregator_model_params_db_link, f'{self.file_write_destination}/{index}/{filename}', ip_ports)

                if response and response.status_code == 200:
                    time.sleep(1)
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
            except Exception as e:
                self.logger.error(f"[{index}] Error getting weights: {str(e)}")
                raise

        # Update model with weights
        self.data_handlers[index].update_model(weights)

        # Train model
        model_params = self.data_handlers[index].train(round_number)
        self.logger.info(f"[{index}][Round {round_number}] Step 2 Complete: Model training done")

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
            return f'{self.docker_file_write_destination}/{index}/{file}'
        return file_name
