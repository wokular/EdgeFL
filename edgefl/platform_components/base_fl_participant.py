"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

import os
import pickle
import time

from dotenv import load_dotenv

from platform_components.EdgeLake_functions.blockchain_EL_functions import insert_policy, check_policy_inserted, \
    get_policies
from platform_components.EdgeLake_functions.mongo_file_store import (
    copy_file_to_container, create_directory_in_container,
    read_file, copy_file_from_container
)
from platform_components.helpers.LoadClassFromFile import load_class_from_file
from platform_components.lib.modules.local_model_update import LocalModelUpdate

load_dotenv()


def _require_env(name):
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class BaseFLParticipant:
    """
    Base class for federated learning participants (aggregator and nodes).
    Contains shared state and methods for index management, file I/O,
    blockchain interaction, model serialization, and training app lifecycle.
    """

    def __init__(self, name, logger):
        self.github_dir = _require_env('GITHUB_DIR')
        self.edgelake_node_url = f'http://{os.getenv("EXTERNAL_IP")}'
        self.edgelake_tcp_node_ip_port = f'{os.getenv("EXTERNAL_TCP_IP_PORT")}'

        self.name = name
        self.logger = logger

        # Index-specific data
        self.indexes = set()
        self.module_names = {}
        self.module_paths = {}
        self.data_handlers = {}
        self.databases = {}
        self.round_number = {}

        # File paths
        self.training_application_dir = os.path.join(self.github_dir, _require_env("TRAINING_APPLICATION_DIR"))
        self.file_write_destination = os.path.join(self.github_dir, _require_env("FILE_WRITE_DESTINATION"), self.name)
        self.tmp_dir = os.path.join(self.github_dir, _require_env("TMP_DIR"), self.name)
        self.docker_file_write_destination = None

        # Docker
        if _require_env("EDGELAKE_DOCKER_RUNNING").lower() == "false":
            self.docker_running = False
        else:
            self.docker_running = True

    def initialize_file_write_paths_on_index(self, index):
        if not os.path.exists(os.path.join(self.file_write_destination, index)):
            os.makedirs(os.path.dirname(
                f"{self.file_write_destination}/{index}/"),
                exist_ok=True)

        if not os.path.exists(os.path.join(self.tmp_dir, index)):
            os.makedirs(os.path.join(self.tmp_dir, index), exist_ok=True)

        if self.docker_running:
            self.docker_file_write_destination = os.path.join(_require_env("DOCKER_FILE_WRITE_DESTINATION"), self.name)
            self.docker_container_name = os.getenv("EDGELAKE_DOCKER_CONTAINER_NAME")
            create_directory_in_container(self.edgelake_node_url, self.docker_container_name,
                                          os.path.join(self.docker_file_write_destination, index))

    def initialize_training_app_on_index(self, index):
        try:
            training_app_path = os.path.join(self.training_application_dir, self.module_paths[index])
            TrainingApp_class = load_class_from_file(training_app_path, self.module_names[index])
            self.data_handlers[index] = TrainingApp_class(self.name)
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def initialize_index(self, index):
        self.indexes.add(index)

    def fetch_indexes_and_modules(self):
        policies = get_policies(self.edgelake_node_url, 'index')
        for policy in policies:
            index = policy['name']
            self.indexes.add(index)
            self.module_names[index] = policy['module_name']
            self.module_paths[index] = policy['module_path']

    def set_module_at_index(self, index, module_name, module_path):
        try:
            index_data = self.get_index_data_in_blockchain(index)
            if index in self.module_names:
                self.logger.info(f'Index "{index}" already has a module: "{self.module_names[index]}"')
                return {
                    'status': 'error',
                    'message': f'Index "{index}" already has a module: "{self.module_names[index]}"'
                }
            elif index_data:
                self.logger.info(
                    f'Index "{index}" already has a module in the blockchain: "{index_data["module_name"]}". Fetching now.')
                self.module_names[index] = index_data['module_name']
                self.module_paths[index] = index_data['module_path']
                return {
                    'status': 'success',
                    'message': f'Index "{index}" already has a module in the blockchain: "{index_data["module_name"]}". Fetching now.'
                }

            self.module_names[index] = module_name
            self.module_paths[index] = module_path
            self.logger.info(f'Added module "{module_name}" to index "{index}"')
            return {
                'status': 'success',
                'message': f'Added module "{module_name}" to index {index}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_index_data_in_blockchain(self, index):
        where_condition = f"where policy_type = init"
        policies = get_policies(self.edgelake_node_url, index, where_condition)
        if not policies:
            return None
        if len(policies) > 1:
            raise Exception(f"Multiple instances of index {index} found in the blockchain")
        return policies[0]

    # Serialization

    def encode_params(self, data):
        serialized_data = pickle.dumps(data)
        return serialized_data

    def decode_params(self, encoded_data):
        model_weights = pickle.loads(encoded_data)
        return model_weights

    # Aggregation methods (used by Aggregator and DFL nodes) 

    def fetch_decoded_params(self, decoded_params_dict, node_param_download_links, ip_ports, rest_ip_ports, index):
        for i, path in enumerate(node_param_download_links):
            if path in decoded_params_dict:
                continue

            try:
                filename = path.split('/')[-1]
                local_path = f'{self.file_write_destination}/{index}/{filename}'
                if self.docker_running:
                    docker_file_path = f'{self.docker_file_write_destination}/{index}/{filename}'
                    response = copy_file_from_container(os.path.join(self.tmp_dir, index), self.docker_container_name,
                                                        rest_ip_ports[i], node_param_download_links[i],
                                                        local_path, ip_ports[i])
                else:
                    response = read_file(rest_ip_ports[i], path,
                                          local_path, ip_ports[i])

                if response and response.status_code != 200:
                    raise ValueError(
                        f"Failed to retrieve node params from link: {filename}. HTTP Status: {response.status_code}"
                    )

                time.sleep(1)
                with open(local_path, 'rb') as f:
                    data = pickle.load(f)
                if not data:
                    raise ValueError(f"Missing model_weights in data from file: {filename}")
                decoded_params_dict[path] = LocalModelUpdate(weights=data)
            except Exception as e:
                self.logger.error(f"Error retrieving data from link {filename}: {str(e)}")
                continue

    def aggregate_model_params(self, decoded_params, round_number, index):
        aggregate_params_weights = self.data_handlers[index].aggregate_model_weights(decoded_params)
        aggregate_model_update = LocalModelUpdate(weights=aggregate_params_weights)
        encoded_params = self.encode_params(aggregate_model_update)

        data_entry = {
            'newUpdates': encoded_params
        }

        file_write_path = f'{self.file_write_destination}/{index}/{round_number}-{self.name}_update.json'

        with open(file_write_path, 'wb') as f:
            f.write(self.encode_params(data_entry))

        if self.docker_running:
            docker_file_write_path = f'{self.docker_file_write_destination}/{index}/{round_number}-{self.name}_update.json'
            copy_file_to_container(os.path.join(self.tmp_dir, index), self.docker_container_name,
                                    self.edgelake_node_url,
                                    file_write_path,
                                    docker_file_write_path)
            return docker_file_write_path

        return file_write_path

    def start_round(self, initParams_link, round_number, index, node_type="aggregator"):
        try:
            data = f'''<my_policy = {{"{index}" : {{
                                        "index" : "{index}",
                                        "policy_type": "RoundStart",
                                        "node_type": "{node_type}",
                                        "round_number": {round_number},
                                        "initParams": "{initParams_link}",
                                        "node_id": "{self.name}",
                                        "ip_port": "{self.edgelake_tcp_node_ip_port}",
                                        "rest_ip_port": "{self.edgelake_node_url}"
                              }} }}>'''
            success = False
            while not success:
                response = insert_policy(self.edgelake_node_url, data)
                if response and response.status_code == 200:
                    success = True
                else:
                    time.sleep(5)

                    if check_policy_inserted(self.edgelake_node_url, data):
                        success = True
            if success:
                return {
                    'status': 'success',
                    'message': 'initTraining called successfully'
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Request failed with status code: {response.status_code}'
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    # Inference

    def inference(self, index):
        return self.data_handlers[index].run_inference()

    def direct_inference(self, index, data, labels=None):
        if labels is not None:
            return self.data_handlers[index].direct_inference(data, labels)
        return self.data_handlers[index].direct_inference(data)
