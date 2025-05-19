import logging
import os
from typing import Dict

import utils
from agent.ES_Registry import ServiceID, ServiceType
from agent.LGBN import LGBN
from agent.ScalingAgent import ScalingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multiscale")

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))
EVALUATION_CYCLE_DELAY = int(utils.get_env_param('EVALUATION_CYCLE_DELAY', 30))

ROOT = os.path.dirname(__file__)


class RRM_Global_Agent(ScalingAgent):

    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path=ROOT + "/../config/slo_config.json",
                 es_registry_path=ROOT + "/../config/es_registry.json",
                 log_experience=None):
        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
                         log_experience)

        self.lgbn = LGBN()

    def orchestrate_services_optimally(self, services_m):
        pass

    def get_optimal_local_ES(self, service: ServiceID, assigned_clients: Dict[str, int]):
        pass


if __name__ == '__main__':
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")
    RRM_Global_Agent(services_monitored=[qr_local, cv_local], prom_server=ps,
                     evaluation_cycle=EVALUATION_CYCLE_DELAY).start()
