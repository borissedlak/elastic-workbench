import logging
import os
import random
import time

import numpy as np
import torch

import utils
from agent.ScalingAgent import ScalingAgent, convert_action_to_real_ES
from agent.agent_utils import FullStateDQN, export_experience_buffer
from agent.daci.mcts_utils import MCTS
from agent.es_registry import ServiceID, ServiceType, ESType

def scale_joint(raw: torch.Tensor, vec_env) -> torch.Tensor:
    """Scale an 8‑D raw joint observation to 0‑1 using the env’s helper."""
    return vec_env.min_max_scale(raw)


PHYSICAL_CORES = int(utils.get_env_param("MAX_CORES", 8))

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)

ROOT = os.path.dirname(__file__)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class DAIAgent(ScalingAgent):

    def __init__(
            self,
            prom_server,
            services_monitored: list[ServiceID],
            evaluation_cycle,
            slo_registry_path=ROOT + "/../../config/slo_config.json",
            es_registry_path=ROOT + "/../../config/es_registry.json",
            log_experience=None,
            depth = 5,
            max_length_trajectory = 30,
            iterations = 500,
            c = 0.5,
            eh = True
    ):
        super().__init__(
            prom_server,
            services_monitored,
            evaluation_cycle,
            slo_registry_path,
            es_registry_path,
            log_experience,
        )
        self.device = "cpu"
        self.depth: int = depth
        self.max_length_trajectory = max_length_trajectory
        self.iterations: int = iterations
        self.c: float = c
        self.eh = eh

        if self.eh:
            agent_file = ROOT + "/hybrid_agent_checkpoint__hybrid_adaptive_ehv2.pth"
        else:
            agent_file = ROOT + "/hybrid_agent_checkpoint__hybrid_adaptive.pth"

        self.agent = torch.load(agent_file, weights_only=False, map_location=self.device)["agent"]
        self.agent.device = self.device

        self.mcts = MCTS(
            action_dim_qr=self.agent.action_dim_qr,
            action_dim_cv=self.agent.action_dim_cv,
            agent=self.agent,
            depth=self.depth,
            iterations=self.iterations,
            max_len=self.max_length_trajectory,
            c=self.c,
            use_eh=self.eh,
        )

    def get_raw_state_for_service(self, service: ServiceID):

        assigned_clients = self.reddis_client.get_assignments_for_service(service)
        service_state = self.resolve_service_state(service, assigned_clients)
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service.service_type, assigned_clients)

        if self.log_experience is not None:
            self.evaluate_slos_and_buffer(service, service_state, all_client_slos)

            # TODO: Remove for real experiment
            # export_experience_buffer(self.experience_buffer, ROOT + f"/agent_experience_DACI.csv")
            # self.experience_buffer = []

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

        free_cores = self.get_free_cores()
        start_state_raw = torch.tensor([service_state["data_quality"],
                                        data_quality_t,
                                        service_state["throughput"],
                                        throughput_t,
                                        model_size,
                                        model_size_t,
                                        service_state["cores"],
                                        free_cores])

        start_state_full = FullStateDQN(
            service_state["data_quality"],
            data_quality_t,
            service_state["throughput"],
            throughput_t,
            model_size,
            model_size_t,
            service_state["cores"],
            free_cores,
            {}
        )

        # Initialize start state
        # start_state_raw_cv = torch.tensor([256, 288, 2, 5, 2, 3, 2, 4])
        # start_state_raw_qr = torch.tensor([700, 900, 2, 60, 1, 1, 2, 4])
        return start_state_raw, start_state_full

    def convert_to_joint_state(self, start_state_raw_qr, start_state_raw_cv):
        joint_state = torch.cat([start_state_raw_cv, start_state_raw_qr], dim=0).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        joint_state = scale_joint(joint_state, self.agent.vec_env)
        return joint_state

    def orchestrate_services_optimally(self, services_m):
        try:
            start_state_raw_qr, start_state_full_qr = self.get_raw_state_for_service(services_m[0])
            start_state_raw_cv, start_state_full_cv = self.get_raw_state_for_service(services_m[1])
            joint_state = self.convert_to_joint_state(start_state_raw_qr, start_state_raw_cv)
        except KeyError as e:
            logger.error("Could not resolve state, need to retry due to ", e)
            time.sleep(0.5)
            self.orchestrate_services_optimally(services_m)
            return

        print(joint_state)
        print(start_state_full_qr)
        print(start_state_full_cv)

        trajectory, stats, root = self.mcts.run_mcts(joint_state)
        action_cv, action_qr = trajectory[0]

        free_cores = self.get_free_cores()
        es_qr, params_qr = convert_action_to_real_ES(services_m[0], start_state_full_qr, action_qr, free_cores)
        wants_to_die = (start_state_full_qr.cores <= 4 and action_qr == 3 ) or \
                       (start_state_full_qr.data_quality >= 900 and action_qr == 2) or \
                       (start_state_full_qr.data_quality <= 500 and action_qr == 1)

        if es_qr == ESType.IDLE:
            logger.info("Agent decided to do nothing for service %s", services_m[0])
        elif wants_to_die:
            logger.info("Preventing the agent to act stupid for service %s", services_m[0])
        else:
            self.execute_ES(services_m[0].host, services_m[0], es_qr, params_qr, respect_cooldown=False,)

        time.sleep(0.01)

        free_cores = self.get_free_cores()
        es_cv, params_qr = convert_action_to_real_ES(services_m[1], start_state_full_cv, action_cv, free_cores)
        wants_to_die = (start_state_full_cv.cores <= 4 and action_cv == 3 ) or \
                       (start_state_full_cv.model_size >= 3 and action_cv == 6) or \
                       (start_state_full_cv.data_quality >= 288 and action_cv == 2)
        if es_cv == ESType.IDLE:
            logger.info("Agent decided to do nothing for service %s", services_m[1])
        elif wants_to_die:
            logger.info("Preventing the agent to act stupid for service %s", services_m[1])
        else:
            self.execute_ES(services_m[1].host, services_m[1], es_cv, params_qr, respect_cooldown=False,)
        print(action_cv, action_qr)


if __name__ == "__main__":
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")

    dai_agent = DAIAgent(
        services_monitored=[qr_local, cv_local],
        prom_server=ps,
        evaluation_cycle=10,
        iterations=70,
        depth=5,
        eh = True
    )
    dai_agent.reset_services_states()
    dai_agent.start()
