import logging
from typing import Dict

import utils
from agent.ES_Registry import ServiceID, ServiceType
from agent.ScalingAgent import ScalingAgent, EVALUATION_CYCLE_DELAY
import pymdp

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)


class pymdp_Agent(ScalingAgent):
    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path = "../config/slo_config.json", es_registry_path = "../config/es_registry.json",
                 log_experience = None):

        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
                         log_experience)


    def get_optimal_local_ES(self, service: ServiceID, service_state, assigned_clients: Dict[str, int]):
        return None, {}


if __name__ == '__main__':
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")
    pymdp_Agent(services_monitored=[qr_local], prom_server=ps,
              evaluation_cycle=EVALUATION_CYCLE_DELAY).start()
