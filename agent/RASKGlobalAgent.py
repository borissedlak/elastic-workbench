import logging
import os
import random
import time
from typing import Dict, Tuple, Any

import numpy as np

import utils
from agent import agent_utils
from agent.RASK import RASK
from agent.agent_utils import export_experience_buffer
from agent.es_registry import ServiceID, ServiceType, ESType
from agent.PolicySolverRASK import solve_global
from agent.ScalingAgent import ScalingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multiscale")

MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))
EVALUATION_CYCLE_DELAY = int(utils.get_env_param('EVALUATION_CYCLE_DELAY', 5))
SERVICE_HOST = utils.get_env_param('SERVICE_HOST', "localhost")

ROOT = os.path.dirname(__file__)


class RASK_Global_Agent(ScalingAgent):

    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path=ROOT + "/../config/slo_config.json",
                 es_registry_path=ROOT + "/../config/es_registry.json",
                 log_experience=None, max_explore=10):
        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
                         log_experience)
        self.explore_count = 0
        self.max_explore = max_explore
        self.rask = RASK(show_figures=False)

    @utils.print_execution_time
    def orchestrate_services_optimally(self, services_m: list[ServiceID]):

        if self.explore_count < self.max_explore:
            logger.info("Agent is exploring.....")
            self.explore_count += 1
            self.call_all_ES_randomly(services_m)
        else:
            service_contexts = []
            for service_m in services_m:  # For all monitored services
                service_contexts.append(self.prepare_service_context(service_m))

            self.rask.init_models()  # Reloads the RASK model from the metrics.csv
            assignments = solve_global(service_contexts, MAX_CORES, self.rask)
            assignments = apply_gaussian_noise_to_asses(assignments)
            self.call_all_ES_deterministic(services_m, assignments)

    def prepare_service_context(self, service_m: ServiceID) -> Tuple[ServiceType, Dict[ESType, Dict], Any, int]:
        assigned_clients = self.reddis_client.get_assignments_for_service(service_m)

        service_state = self.resolve_service_state(service_m, assigned_clients)
        es_parameter_bounds = self.es_registry.get_parameter_bounds_for_active_ES(service_m.service_type)
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service_m.service_type, assigned_clients)
        total_rps = utils.to_absolut_rps(assigned_clients)

        if self.log_experience is not None:
            self.evaluate_slos_and_buffer(service_m, service_state, all_client_slos)

        return service_m.service_type, es_parameter_bounds, all_client_slos, total_rps

    # @utils.print_execution_time
    def call_all_ES_deterministic(self, services_m: list[ServiceID], assignments):
        # TODO: Ideally, this needs a mechanisms that avoids oscillating or changing the instance if it stays the same
        for i, service_m in enumerate(services_m):  # For all monitored services
            all_es = self.es_registry.get_active_ES_for_service(service_m.service_type)
            for target_ES in all_es:
                self.execute_ES(service_m, target_ES, assignments[i], respect_cooldown=False)

    def call_all_ES_randomly(self, services_m: list[ServiceID]):
        # Shuffle services to avoid the first always getting the most resources
        shuffled_services = services_m.copy()
        random.shuffle(shuffled_services)

        for service_m in shuffled_services:

            all_ES_active = self.es_registry.get_active_ES_for_service(service_m.service_type)
            for es in all_ES_active:
                max_available_cores = self.get_max_available_cores(service_m)
                param_bounds = self.es_registry.get_parameter_bounds_for_active_ES(service_m.service_type,
                                                                                   max_available_cores).get(es, {})

                random_params = agent_utils.get_random_parameter_assignments(param_bounds)
                self.execute_ES(service_m, es, random_params, respect_cooldown=False)


def apply_gaussian_noise_to_asses(assignment, noise=0.08):
    for ass_group in assignment:
        for var in ass_group:
            value = ass_group[var]
            std_dev = noise * abs(value)  # 8% of the value as standard deviation
            ass_group[var] += np.random.normal(0, std_dev)

    return assignment


if __name__ == '__main__':
    # remote_vm = "128.131.172.182"
    ps = f"http://{SERVICE_HOST}:9090"
    qr_local = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-1", port="8080")
    cv_local = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-1", port="8081")
    pc_local = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-1", port="8082")
    agent = RASK_Global_Agent(services_monitored=[cv_local, qr_local, pc_local], prom_server=ps,
                              evaluation_cycle=EVALUATION_CYCLE_DELAY, max_explore=20, log_experience="RRM")

    agent_utils.stream_remote_metrics_file()

    agent.reset_services_states()
    agent.start()

    while True:
        time.sleep(5)
        export_experience_buffer(agent.experience_buffer, ROOT + f"/agent_experience_RRM.csv")
