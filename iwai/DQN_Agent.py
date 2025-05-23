import logging
import os
import random
from typing import Dict

import numpy as np

import utils
from agent.ES_Registry import ServiceID, ServiceType, EsType
from agent.ScalingAgent import ScalingAgent, EVALUATION_CYCLE_DELAY
from agent.agent_utils import Full_State_DQN
from iwai.DQN_Trainer import DQN, ACTION_DIM_QR
from iwai.DQN_Trainer import STATE_DIM

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)

ROOT = os.path.dirname(__file__)


class DQN_Agent(ScalingAgent):
    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle, dqn=None,
                 slo_registry_path=ROOT + "/../config/slo_config.json",
                 es_registry_path=ROOT + "/../config/es_registry.json",
                 log_experience=None):

        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
                         log_experience)

        self.dqn = dqn if dqn is not None else DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_QR)

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

            ES_list, all_elastic_params_ass = self.get_optimal_local_ES(service_m, service_state, assigned_clients)
            if ES_list is None:
                logger.info("Agent decided to do nothing")
            else:
                for target_ES in ES_list:
                    self.execute_ES(service_m.host, service_m, target_ES, all_elastic_params_ass,
                                    respect_cooldown=False)

            if self.log_experience is not None:
                self.build_state_and_log(service_state, service_m, assigned_clients)

    def get_optimal_local_ES(self, service: ServiceID, service_state, assigned_clients: Dict[str, int]):
        free_cores = self.get_free_cores()
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service.service_type, assigned_clients)

        quality_t, throughput_t = all_client_slos[0]['quality'].thresh, all_client_slos[0]['throughput'].thresh
        state_pw = Full_State_DQN(service_state['quality'], quality_t, service_state['throughput'], throughput_t,
                                  service_state['cores'], free_cores)
        action_pw = self.dqn.choose_action(np.array(state_pw.for_tensor()), rand=0.0)

        if 1 <= action_pw <= 2:
            delta_quality = -100 if action_pw == 1 else 100
            return [EsType.QUALITY_SCALE], {'quality': int(state_pw.quality + delta_quality)}
        if 3 <= action_pw <= 4:
            delta_cores = -1 if action_pw == 3 else 1
            return [EsType.RESOURCE_SCALE], {'cores': int(state_pw.cores + delta_cores)}

        return None, None


if __name__ == '__main__':
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")
    DQN_Agent(services_monitored=[qr_local], prom_server=ps, evaluation_cycle=EVALUATION_CYCLE_DELAY).start()
