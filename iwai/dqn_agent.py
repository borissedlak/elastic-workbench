import logging
import os
import time
from typing import Dict, List

import numpy as np

import utils
from agent.ScalingAgent import ScalingAgent, EVALUATION_CYCLE_DELAY, convert_action_to_real_ES
from agent.agent_utils import FullStateDQN
from agent.components.es_registry import ServiceID, ServiceType, ESType
from iwai.dqn_trainer import (
    DQN,
    ACTION_DIM_QR,
    ACTION_DIM_CV,
)
from iwai.dqn_trainer import STATE_DIM

MAX_CORES = int(utils.get_env_param("MAX_CORES", 8))

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)

ROOT = os.path.dirname(__file__)


class DQNAgent(ScalingAgent):
    def __init__(
        self,
        prom_server,
        services_monitored: list[ServiceID],
        dqn_for_services: list[DQN],
        evaluation_cycle,
        slo_registry_path=ROOT + "/../config/slo_config.json",
        es_registry_path=ROOT + "/../config/es_registry.json",
        log_experience=None,
    ):

        super().__init__(
            prom_server,
            services_monitored,
            evaluation_cycle,
            slo_registry_path,
            es_registry_path,
            log_experience,
        )

        self.dqn = {}
        for service, dqn in zip(services_monitored, dqn_for_services):
            self.dqn[service.service_type] = dqn

    def orchestrate_services_optimally(self, services_m: List[ServiceID]):
        # shuffled_services = self.services_monitored.copy()
        # random.shuffle(shuffled_services)

        for service_m in services_m:
            service_m: ServiceID = service_m
            assigned_clients = self.reddis_client.get_assignments_for_service(service_m)
            service_state = self.resolve_service_state(service_m, assigned_clients)

            if service_state == {}:
                logger.warning(f"Cannot find state for service {service_m}")
                continue

            es, elastic_params_ass = self.get_optimal_local_es(
                service_m, service_state, assigned_clients
            )
            if es == ESType.IDLE:
                logger.info("Agent decided to do nothing for service %s", service_m)
            else:
                self.execute_ES(
                    service_m.host,
                    service_m,
                    es,
                    elastic_params_ass,
                    respect_cooldown=False,
                )

    def get_optimal_local_es(
        self, service: ServiceID, service_state, assigned_clients: Dict[str, int]
    ) -> tuple[ESType, Dict]:
        free_cores = self.get_free_cores()
        boundaries = self.es_registry.get_boundaries_minimalistic(
            service.service_type, MAX_CORES
        )
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(
            service.service_type, assigned_clients
        )

        if self.log_experience is not None:
            self.evaluate_slos_and_buffer(service, service_state, all_client_slos)

        model_size, model_size_t = 1, 1
        if "model_size" in service_state or "model_size" in all_client_slos[0]:
            model_size, model_size_t = (
                service_state["model_size"],
                all_client_slos[0]["model_size"].target,
            )

        data_quality_t, throughput_t = (
            all_client_slos[0]["data_quality"].target,
            all_client_slos[0]["throughput"].target,
        )
        state_pw = FullStateDQN(
            service_state["data_quality"],
            data_quality_t,
            service_state["throughput"],
            throughput_t,
            model_size,
            model_size_t,
            service_state["cores"],
            free_cores,
            boundaries,
        )

        action_pw = self.dqn[service.service_type].choose_action(
            np.array(state_pw.to_np_ndarray(True)), rand=0.0
        )

        return convert_action_to_real_ES(service, state_pw, action_pw, free_cores)


if __name__ == "__main__":
    ps = "http://localhost:9090"

    qr_local = ServiceID(
        "172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1"
    )
    cv_local = ServiceID(
        "172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1"
    )

    # Load the trained DQNs
    dqn_qr = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_QR)
    dqn_qr.load("Q_QR_joint.pt")

    dqn_cv = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_CV)
    dqn_cv.load("Q_CV_joint.pt")

    # Start the agent with both services
    agent = DQNAgent(
        prom_server=ps,
        services_monitored=[qr_local, cv_local],
        dqn_for_services=[dqn_qr, dqn_cv],
        evaluation_cycle=EVALUATION_CYCLE_DELAY,
    )

    agent.reset_services_states()
    time.sleep(3)
    agent.start()
