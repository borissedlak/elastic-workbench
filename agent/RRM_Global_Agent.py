import logging
import os
from typing import Dict, Tuple, Any

import utils
from agent.ES_Registry import ServiceID, ServiceType, EsType
from agent.PolicySolver_RRM import solve_global
from agent.ScalingAgent import ScalingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multiscale")

MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))
EVALUATION_CYCLE_DELAY = int(utils.get_env_param('EVALUATION_CYCLE_DELAY', 30))

ROOT = os.path.dirname(__file__)


class RRM_Global_Agent(ScalingAgent):

    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path=ROOT + "/../config/slo_config.json",
                 es_registry_path=ROOT + "/../config/es_registry.json",
                 log_experience=None):
        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
                         log_experience)

        # self.rrm = RRM()

    def orchestrate_services_optimally(self, services_m):
        service_contexts = []
        for service_m in services_m:  # For all monitored services
            service_contexts.append(self.prepare_service_context(service_m))

            # if self.log_experience is not None:
            #     self.build_state_and_log(service_state, service_m, assigned_clients)

        solve_global(service_contexts, MAX_CORES)

    def prepare_service_context(self, service_m: ServiceID) -> Tuple[ServiceType, Dict[EsType, Dict], Any, int]:
        assigned_clients = self.reddis_client.get_assignments_for_service(service_m)
        # service_state = self.resolve_service_state(service_m, assigned_clients)

        max_available_c = self.get_max_available_cores(service_m)
        ES_parameter_bounds = self.es_registry.get_parameter_bounds_for_active_ES(service_m.service_type,
                                                                                  max_available_c)
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service_m.service_type, assigned_clients)
        total_rps = utils.to_absolut_rps(assigned_clients)
        return service_m.service_type, ES_parameter_bounds, all_client_slos, total_rps

    def get_optimal_local_ES(self, service: ServiceID, assigned_clients: Dict[str, int]):
        pass


if __name__ == '__main__':
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")
    RRM_Global_Agent(services_monitored=[qr_local, cv_local], prom_server=ps,
                     evaluation_cycle=EVALUATION_CYCLE_DELAY).start()
