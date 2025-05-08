import logging
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


class LGBNAgent(ScalingAgent):
    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle):
        super().__init__(prom_server, services_monitored, evaluation_cycle)

        self.lgbn = LGBN()

    def get_optimal_local_ES(self, service: ServiceID, service_state, assigned_clients: Dict[str, int]):

        max_available_c = self.get_max_available_cores(service)
        # TODO: The problem is that these parameter bounds are different from the policy solver test
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
    LGBNAgent(services_monitored=[qr_local], prom_server=ps, evaluation_cycle=EVALUATION_CYCLE_DELAY).start()
