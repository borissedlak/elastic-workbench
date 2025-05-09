import logging
from typing import Dict

import utils
from agent.ES_Registry import ServiceID, ServiceType
from agent.ScalingAgent import ScalingAgent, EVALUATION_CYCLE_DELAY
from iwai.DQN_Trainer import DQN
from iwai.DQN_Trainer import STATE_DIM

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)


class DAI_Agent(ScalingAgent):
    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle):
        super().__init__(prom_server, services_monitored, evaluation_cycle)

        self.dqn = DQN(state_dim=STATE_DIM, action_dim=5)

    def get_optimal_local_ES(self, service: ServiceID, service_state, assigned_clients: Dict[str, int]):
        pass


if __name__ == '__main__':
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")
    DQN_Agent(services_monitored=[qr_local], prom_server=ps,
              evaluation_cycle=EVALUATION_CYCLE_DELAY).start()
