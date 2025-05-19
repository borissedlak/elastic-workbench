import logging
import os
from typing import Dict, Tuple, Any

import numpy as np

import utils
from agent.ES_Registry import ServiceID, ServiceType, EsType
from agent.PolicySolver_RRM import solve_global
from agent.ScalingAgent import ScalingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multiscale")

MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))
EVALUATION_CYCLE_DELAY = int(utils.get_env_param('EVALUATION_CYCLE_DELAY', 10))

ROOT = os.path.dirname(__file__)


class RRM_Global_Agent(ScalingAgent):

    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path=ROOT + "/../config/slo_config.json",
                 es_registry_path=ROOT + "/../config/es_registry.json",
                 log_experience=None):
        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
                         log_experience)

    def orchestrate_services_optimally(self, services_m: list[ServiceID]):
        service_contexts = []
        for service_m in services_m:  # For all monitored services
            service_contexts.append(self.prepare_service_context(service_m))

        if np.random.rand() > 0.15: # TODO: Start high and converge to a minimum of 0.05 (?)
            assignments = solve_global(service_contexts, MAX_CORES)
            assignments = apply_gaussian_noise_to_asses(assignments)
            self.orchestrate_all_ES_deterministic(services_m, assignments)
        else:
            logger.info("Agent is exploring.....")
            self.orchestrate_all_ES_randomly(services_m)



    def prepare_service_context(self, service_m: ServiceID) -> Tuple[ServiceType, Dict[EsType, Dict], Any, int]:
        assigned_clients = self.reddis_client.get_assignments_for_service(service_m)

        service_state = self.resolve_service_state(service_m, assigned_clients)
        if self.log_experience is not None:
            self.build_state_and_log(service_state, service_m, assigned_clients)

        ES_parameter_bounds = self.es_registry.get_parameter_bounds_for_active_ES(service_m.service_type)
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service_m.service_type, assigned_clients)
        total_rps = utils.to_absolut_rps(assigned_clients)
        return service_m.service_type, ES_parameter_bounds, all_client_slos, total_rps

    def orchestrate_all_ES_deterministic(self, services_m: list[ServiceID], assignments):
        # TODO: Ideally, this needs a mechanisms that avoids oscillating or changing the instance if it stays the same
        for i, service_m in enumerate(services_m):  # For all monitored services
            all_ES = self.es_registry.get_supported_ES_for_service(service_m.service_type)
            for target_ES in all_ES:
                self.execute_ES(service_m.host, service_m, target_ES, assignments[i], respect_cooldown=False)

    def orchestrate_all_ES_randomly(self, services_m: list[ServiceID]):
        for service_m in services_m:
            rand_ES, rand_params = self.es_registry.get_random_ES_and_params(service_m.service_type)
            self.execute_ES(service_m.host, service_m, rand_ES, rand_params)

def apply_gaussian_noise_to_asses(assignment, noise = 0.08):
    for ass_group in assignment:
        for var in ass_group:
            value = ass_group[var]
            std_dev = noise * abs(value)  # 5% of the value as standard deviation
            ass_group[var] += np.random.normal(0, std_dev)

    return assignment


if __name__ == '__main__':
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")
    agent = RRM_Global_Agent(services_monitored=[qr_local, cv_local], prom_server=ps,
                             evaluation_cycle=EVALUATION_CYCLE_DELAY, log_experience="RRM Agent")

    agent.reset_services_states()
    agent.start()
