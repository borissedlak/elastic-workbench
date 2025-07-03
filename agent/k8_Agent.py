import logging
import os
import time
from typing import List, Tuple, Dict

import utils
from HttpClient import HttpClient
from agent.ScalingAgent import ScalingAgent
from agent.agent_utils import export_experience_buffer
from agent.components.es_registry import ServiceID, ServiceType, ESType

MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))
EVALUATION_CYCLE_DELAY = int(utils.get_env_param('EVALUATION_CYCLE_DELAY', 10))
SERVICE_HOST = utils.get_env_param('SERVICE_HOST', "localhost")
REMOTE_VM = utils.get_env_param('REMOTE_VM', "128.131.172.182")

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)


class k8_Agent(ScalingAgent):

    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path=ROOT + "/../config/slo_config.json",
                 es_registry_path=ROOT + "/../config/es_registry.json",
                 log_experience=None):

        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path,
                         es_registry_path, log_experience)


    def prepare_service_context(self, service_m: ServiceID) -> Tuple[ServiceType, Dict[ESType, Dict]]:
        assigned_clients = self.reddis_client.get_assignments_for_service(service_m)

        service_state = self.resolve_service_state(service_m, assigned_clients)
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service_m.service_type, assigned_clients)

        if self.log_experience is not None:
            self.evaluate_slos_and_buffer(service_m, service_state, all_client_slos)

        return service_m.service_type, service_state

    def orchestrate_services_optimally(self, services_m: List[ServiceID]):
        # TODO: (1) See which containers currently don't have enough CPU
        #       (2) Increase their CPU (if possible)
        #       (?) Get free cores
        pass


if __name__ == '__main__':
    ps = f"http://{SERVICE_HOST}:9090"  # "128.131.172.182"
    qr_local = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-1", port="8080")
    cv_local = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-1", port="8081")
    pc_local = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-1", port="8082")

    agent = k8_Agent(services_monitored=[cv_local, qr_local, pc_local], prom_server=ps,
                     evaluation_cycle=EVALUATION_CYCLE_DELAY, log_experience="#")

    # agent_utils.delete_file_if_exists(ROOT + "/../share/metrics/metrics.csv")
    # agent_utils.stream_remote_metrics_file(REMOTE_VM, EVALUATION_CYCLE_DELAY)

    http_client = HttpClient()
    http_client.update_service_rps(qr_local, 80)
    http_client.update_service_rps(cv_local, 5)
    http_client.update_service_rps(pc_local, 50)

    agent.reset_services_states()
    agent.start()

    while True:
        time.sleep(5)
        export_experience_buffer(agent.experience_buffer, ROOT + f"/agent_experience_RASK.csv")
