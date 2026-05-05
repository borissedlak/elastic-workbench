import logging
import os
import random
import time
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

import utils
from HttpClient import HttpClient
from agent import agent_utils
from agent.components.PolicySolverRASK import solve_global
from agent.components.RASK import RASK
from agent.ScalingAgent import ScalingAgent, QR_DATA_QUALITY_DEFAULT, CV_M_SIZE_DEFAULT, CV_DATA_QUALITY_DEFAULT, \
    PC_DISTANCE_DEFAULT
from agent.agent_utils import export_experience_buffer
from agent.components.es_registry import ServiceID, ServiceType, ESType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multiscale")

ROOT = os.path.dirname(__file__)

##### Scaling Agent Hyperparameters #######

MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))
EXPLORE_ROUNDS = int(utils.get_env_param('EXPLORE_ROUNDS', 0))
GAUSSIAN_NOISE = float(utils.get_env_param('GAUSSIAN_NOISE', 0.0))
EVALUATION_CYCLE_DELAY = int(utils.get_env_param('EVALUATION_CYCLE_DELAY', 10))

########## Service Definitions ############

SERVICE_HOST = utils.get_env_param('SERVICE_HOST', "localhost")
REMOTE_VM = utils.get_env_param('REMOTE_VM', "128.131.172.182")
SERVICE_REPLICATION = int(utils.get_env_param('SERVICE_REPLICATION', 1))
PROMETHEUS = f"http://{SERVICE_HOST}:9090"  # "128.131.172.182"

QR_RPS = 80
CV_RPS = 5
PC_RPS = 50

qr_local_1 = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-1", port="8080")
qr_local_2 = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-2", port="8083")
qr_local_3 = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-3", port="8086")

cv_local_1 = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-1", port="8081")
cv_local_2 = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-2", port="8084")
cv_local_3 = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-3", port="8087")

pc_local_1 = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-1", port="8082")
pc_local_2 = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-2", port="8085")
pc_local_3 = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-3", port="8088")

services_3 = [qr_local_1, cv_local_1, pc_local_1]
services_6 = services_3 + [qr_local_2, cv_local_2, pc_local_2]
services_9 = services_6 + [qr_local_3, cv_local_3, pc_local_3]

services_convert = {1: services_3, 2: services_6, 3: services_9}


class RASK_Agent(ScalingAgent):

    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path=ROOT + "/../config/slo_config.json",
                 es_registry_path=ROOT + "/../config/es_registry.json",
                 log_experience=None, explore_rounds=25, gaussian_noise=0.05,
                 cache_last_assignment=True, replay_path=None):

        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path,
                         es_registry_path, log_experience)

        self.explore_count = 0
        self.explore_rounds = explore_rounds
        self.gaussian_noise = gaussian_noise
        self.cache_last_assignment = cache_last_assignment

        self.last_assignments = None
        # self.last_assignments = [{'data_quality': QR_DATA_QUALITY_DEFAULT, 'cores': MAX_CORES / 3},
        #                          {'model_size': CV_M_SIZE_DEFAULT, 'data_quality': CV_DATA_QUALITY_DEFAULT,
        #                           'cores': MAX_CORES / 3},
        #                          {'data_quality': PC_DISTANCE_DEFAULT, 'cores': MAX_CORES / 3}]

        self.rask = RASK(show_figures=False)

        # Replay Logic initialization
        self.replay_path = replay_path
        if self.replay_path:
            logger.info(f"Agent initialized in REPLAY mode using: {replay_path}")
            df_replay = pd.read_csv(replay_path)
            # Group by Generation so we can pop one generation at a time
            self.replay_steps = df_replay.to_dict('records')
            self.current_step = 0

    @utils.print_execution_time
    def orchestrate_services_optimally(self, services_m: list[ServiceID]):

        # Need to build the state in any case to evaluate the SLOs
        service_contexts = []
        for service_m in services_m:  # For all monitored services
            service_contexts.append(self.prepare_service_context(service_m))

        # 2. Decision Logic: Replay vs Explore vs Optimize
        if self.replay_path:
            self.execute_replay_step(services_m)

        elif self.explore_count < self.explore_rounds:
            logger.info("Agent is exploring.....")
            self.explore_count += 1
            self.call_all_ES_randomly(services_m)
        else:
            self.rask.init_models()  # Reloads the RASK model from the metrics.csv
            assignments = solve_global(service_contexts, MAX_CORES, self.rask, self.last_assignments)
            assignments = apply_gaussian_noise_to_asses(assignments, self.gaussian_noise)
            self.call_all_ES_deterministic(services_m, assignments)

            if self.cache_last_assignment:
                self.last_assignments = assignments


    def execute_replay_step(self, services_m: list[ServiceID]):
        """Helper to extract params from CSV and apply them"""
        if self.current_step >= len(self.replay_steps):
            logger.warning("Replay data exhausted. Terminating.")
            self.terminate_gracefully()
            return
        
        row = self.replay_steps[self.current_step]
        
        # We need to format the CSV row back into the 'assignments' list format
        # expected by our existing call_all_ES_deterministic method
        csv_assignments = []
        
        for service_m in services_m:
            
            # Map CSV columns back to parameter keys
            # Adjust these keys if your CSV column names are different!
            params = {
                'cores': row.get('cores'),
                'data_quality': row.get('data_quality')
            }
            if service_m.service_type == ServiceType.CV:
                params['model_size'] = row.get('model_size')
                
            csv_assignments.append(params)

        self.call_all_ES_deterministic(services_m, csv_assignments)
        self.current_step += 1

    def prepare_service_context(self, service_m: ServiceID) -> Tuple[ServiceType, Dict[ESType, Dict], Any, int]:
        assigned_clients = self.reddis_client.get_assignments_for_service(service_m)

        # TODO: Fix this ....
        if assigned_clients == {}:
            logging.warning("No clients found, but why?")
            time.sleep(0.01)
            return self.prepare_service_context(service_m)

        service_state = self.resolve_service_state(service_m, assigned_clients)
        es_parameter_bounds = self.es_registry.get_parameter_bounds_for_active_ES(service_m.service_type)
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service_m.service_type, assigned_clients)
        total_rps = utils.to_absolut_rps(assigned_clients)

        if self.log_experience is not None:
            self.evaluate_slos_and_buffer(service_m, service_state, all_client_slos)

        return service_m.service_type, es_parameter_bounds, all_client_slos, total_rps

    # @utils.print_execution_time
    def call_all_ES_deterministic(self, services_m: list[ServiceID], assignments):
        for i, service_m in enumerate(services_m):  # For all monitored services
            all_es = self.es_registry.get_active_ES_for_service(service_m.service_type)
            for target_ES in all_es:
                self.execute_ES(service_m, target_ES, assignments[i], respect_cooldown=False)

    def call_all_ES_randomly(self, services_m: list[ServiceID]):
        # Shuffle services to avoid the first always getting the most resources
        shuffled_services = services_m.copy()
        random.shuffle(shuffled_services)

        assigned_cores = 0

        for index, service_m in enumerate(shuffled_services):
            all_ES_active = self.es_registry.get_active_ES_for_service(service_m.service_type)
            max_available_cores = MAX_CORES - assigned_cores + (index - 2)  # 6 Cores for first service

            for es in all_ES_active:
                param_bounds = self.es_registry.get_parameter_bounds_for_active_ES(service_m.service_type,
                                                                                   max_available_cores).get(es, {})

                random_params = agent_utils.get_random_parameter_assignments(param_bounds)
                self.execute_ES(service_m, es, random_params, respect_cooldown=False)

                if es == ESType.RESOURCE_SCALE:
                    assigned_cores += random_params['cores']

    def set_last_assignments(self, assignments):
        self.last_assignments = assignments


def apply_gaussian_noise_to_asses(assignment, noise):
    for ass_group in assignment:
        for var in ass_group:
            value = ass_group[var]
            std_dev = noise * abs(value)  # 8% of the value as standard deviation
            ass_group[var] += np.random.normal(0, std_dev)

    return assignment


if __name__ == '__main__':

    agent = RASK_Agent(services_monitored=services_convert[SERVICE_REPLICATION], prom_server=PROMETHEUS,
                       evaluation_cycle=EVALUATION_CYCLE_DELAY, log_experience="#",
                       explore_rounds=EXPLORE_ROUNDS, gaussian_noise=GAUSSIAN_NOISE,)
    

    http_client = HttpClient()
    http_client.update_service_rps(qr_local_1, QR_RPS)
    http_client.update_service_rps(cv_local_1, CV_RPS)
    http_client.update_service_rps(pc_local_1, PC_RPS)

    if SERVICE_REPLICATION >= 2:
        http_client.update_service_rps(qr_local_2, QR_RPS)
        http_client.update_service_rps(cv_local_2, CV_RPS)
        http_client.update_service_rps(pc_local_2, PC_RPS)
    if SERVICE_REPLICATION >= 3:
        http_client.update_service_rps(qr_local_3, QR_RPS)
        http_client.update_service_rps(cv_local_3, CV_RPS)
        http_client.update_service_rps(pc_local_3, PC_RPS)

    agent.reset_services_states()
    time.sleep(EVALUATION_CYCLE_DELAY / 2) # Needs a couple of seconds after resetting services (i.e., calling ES)
    agent.start()

    while True:
        time.sleep(EVALUATION_CYCLE_DELAY)
        export_experience_buffer(agent.experience_buffer, ROOT + f"/agent_experience_RASK.csv")
