import logging
import os
import platform
import random
from typing import Dict

import utils
from agent import PolicySolver
from agent.ES_Registry import ServiceID, ServiceType, EsType
from agent.LGBN import LGBN
from agent.ScalingAgent import ScalingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multiscale")

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))
EVALUATION_CYCLE_DELAY = int(utils.get_env_param('EVALUATION_CYCLE_DELAY', 30))

ROOT = os.path.dirname(__file__)


class LGBN_Local_Agent(ScalingAgent):

    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path=ROOT + "/../config/slo_config.json",
                 es_registry_path=ROOT + "/../config/es_registry.json",
                 log_experience=None):
        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
                         log_experience)

        self.lgbn = LGBN()

    def orchestrate_services_optimally(self, services_m):
        shuffled_services = self.services_monitored.copy()
        random.shuffle(shuffled_services)  # Shuffling the clients avoids that one can always pick cores first

        for service_m in shuffled_services:  # For all monitored services

            service_m: ServiceID = service_m
            assigned_clients = self.reddis_client.get_assignments_for_service(service_m)
            service_state = self.resolve_service_state(service_m, assigned_clients)

            if service_state == {}:
                logger.warning(f"Cannot find state for service {service_m}")
                continue

            logger.info(f"Current state for <{service_m.host},{service_m.container_id}>: {service_state}")
            all_client_SLO_F = self.get_clients_SLO_F(service_m, service_state, assigned_clients)
            print(all_client_SLO_F)

            host_fix = "localhost" if platform.system() == "Windows" else service_m.host

            # TODO: This will cause the service to oscillate.....
            ES_list, all_elastic_params_ass = self.get_optimal_local_ES(service_m, assigned_clients)
            if ES_list is None:
                logger.info("Agent decided to do nothing")
            else:
                for target_ES in ES_list:
                    self.execute_ES(host_fix, service_m, target_ES, all_elastic_params_ass, respect_cooldown=False)

            # rand_ES, rand_params = self.es_registry.get_random_ES_and_params(service_m.service_type)
            # self.execute_ES(host_fix, service_m.service_type, rand_ES, rand_params)

            if self.log_experience is not None:
                self.build_state_and_log(service_state, service_m, assigned_clients)

    def get_optimal_local_ES(self, service: ServiceID, assigned_clients: Dict[str, int]):

        max_available_c = self.get_max_available_cores(service)
        ES_parameter_bounds = self.es_registry.get_parameter_bounds_for_active_ES(service.service_type, max_available_c)

        # TODO: This is too much logic here
        if not ES_parameter_bounds:
            logger.warning("Cannot get optimal ES parameters because no ES configured")
            return [], {}
        if not assigned_clients:
            if EsType.RESOURCE_SCALE in [item['es_type'] for item in ES_parameter_bounds]:
                logger.info("No clients connected, releasing all cores except one")
                return [EsType.RESOURCE_SCALE], {'cores': 1}  # Free all cores, just in case. If this ES is active
            else:
                return [], {}

        linear_relations = self.lgbn.get_linear_relations(service.service_type)
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service.service_type, assigned_clients)
        total_rps = utils.to_absolut_rps(assigned_clients)

        all_ES = self.es_registry.get_supported_ES_for_service(service.service_type)
        return all_ES, PolicySolver.solve(ES_parameter_bounds, linear_relations, all_client_slos, total_rps)


if __name__ == '__main__':
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")
    agent = LGBN_Local_Agent(services_monitored=[qr_local, cv_local], prom_server=ps, evaluation_cycle=EVALUATION_CYCLE_DELAY)

    agent.reset_services_states()
    agent.start()
