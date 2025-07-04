import logging
import os

import gymnasium
import pandas as pd

import utils
from agent.ScalingAgent import CV_DATA_QUALITY_DEFAULT, CV_M_SIZE_DEFAULT, QR_DATA_QUALITY_DEFAULT, PC_DISTANCE_DEFAULT
from agent.agent_utils import FullStateDQN
from agent.components.LGBN import LGBN
from agent.components.SLORegistry import (
    calculate_slo_fulfillment,
    to_normalized_slo_f,
    SLO_Registry,
)
from agent.components.es_registry import ServiceType, ESRegistry
# from experiments.tsc.E1.E1 import QR_RPS, CV_RPS, PC_RPS
from iwai.proj_types import ESServiceAction

logger = logging.getLogger("multiscale")

ROOT = os.path.dirname(__file__)
MAX_CORES = int(utils.get_env_param("MAX_CORES", 8))
INVALID_ACTION_PUNISHMENT = -5

DEFAULT_CLIENTS = {ServiceType.QR: 80, ServiceType.CV: 5, ServiceType.PC: 50}

class LGBNTrainingEnv(gymnasium.Env):
    def __init__(self, service_type, step_data_quality, step_cores=1, step_model_size=1):
        super().__init__()
        self.state: FullStateDQN = None
        self.lgbn: LGBN = None
        self.service_type: ServiceType = service_type
        self.es_registry = ESRegistry(ROOT + "/../config/es_registry.json")
        self.slo_registry = SLO_Registry(ROOT + "/../config/slo_config.json")

        self.step_data_quality = step_data_quality
        self.step_cores = step_cores
        self.step_model_size = step_model_size

        self.boundaries = self.es_registry.get_boundaries_minimalistic(
            self.service_type, MAX_CORES
        )
        self.client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(
            self.service_type, {"C_1": DEFAULT_CLIENTS[service_type]}
        )[0]

    def step(self, action: ESServiceAction):
        additional_reward_punishment = 0
        new_state = self.state._asdict()
        done = False

        # Do nothing at 0
        if action.value == 0:
            additional_reward_punishment += 0.1  # Encourage the client not to oscillate

        if 1 <= action.value <= 2:
            delta_data_quality = -self.step_data_quality if action.value == 1 else self.step_data_quality
            new_data_quality = self.state.data_quality + delta_data_quality

            if (
                    new_data_quality < self.boundaries["data_quality"]["min"]
                    or new_data_quality > self.boundaries["data_quality"]["max"]
            ):
                # behavioral_punishment = INVALID_ACTION_PUNISHMENT
                # done = True
                pass
            else:
                new_state["data_quality"] = new_data_quality

        elif 3 <= action.value <= 4:
            # delta is always 1
            delta_cores = -self.step_cores if action.value == 3 else self.step_cores
            new_core = self.state.cores + delta_cores

            if new_core <= 0:  # Wants to go lower than 0 core
                # behavioral_punishment = INVALID_ACTION_PUNISHMENT
                # done = True
                pass
            elif (
                    delta_cores > self.state.free_cores
            ):  # Want to consume resources that are not free
                # behavioral_punishment = INVALID_ACTION_PUNISHMENT
                # done = True
                pass
            else:
                new_state["cores"] = self.state.cores + delta_cores
                new_state["free_cores"] = new_state["free_cores"] - delta_cores

        elif 5 <= action.value <= 6:
            # step size is always 1
            delta_model = -self.step_model_size if action.value == 5 else self.step_model_size
            new_model_s = self.state.model_size + delta_model

            if (
                    new_model_s < self.boundaries["model_size"]["min"]
                    or new_model_s > self.boundaries["model_size"]["max"]
            ):
                # behavioral_punishment = INVALID_ACTION_PUNISHMENT
                # done = True
                pass
            else:
                new_state["model_size"] = new_model_s

        new_state["throughput"] = self.sample_throughput_from_lgbn(
            new_state["data_quality"], new_state["cores"], new_state["model_size"]
        )
        self.state = FullStateDQN(**new_state)

        reward = to_normalized_slo_f(
            calculate_slo_fulfillment(self.state.to_normalized_dict(), self.client_slos),
            self.client_slos,
        ) + additional_reward_punishment

        return self.state, reward, done, False, {}

    def sample_throughput_from_lgbn(self, data_quality, cores, model_size):
        partial_state = {"data_quality": data_quality, "cores": cores, "model_size": model_size}
        full_state = self.lgbn.predict_lgbn_vars(partial_state, self.service_type)
        return full_state['max_tp']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO: Change to default assignments

        data_quality_target = self.client_slos["data_quality"].target

        model_size = 1
        model_size_target = 1

        if self.service_type == ServiceType.QR:
            data_quality = QR_DATA_QUALITY_DEFAULT
        elif self.service_type == ServiceType.CV:
            model_size = CV_M_SIZE_DEFAULT
            model_size_target = self.client_slos["model_size"].target
            data_quality = CV_DATA_QUALITY_DEFAULT
        elif self.service_type == ServiceType.PC:
            data_quality = PC_DISTANCE_DEFAULT
        else:
            raise RuntimeWarning("Unknown service type")

        ass_cores = MAX_CORES / 3
        free_cores = MAX_CORES - ass_cores

        throughput = self.sample_throughput_from_lgbn(data_quality, ass_cores, model_size)
        completion_rate = throughput / DEFAULT_CLIENTS[self.service_type]  # TODO: Needs current rps
        completion_target = self.client_slos["completion_rate"].target

        self.state = FullStateDQN(
            data_quality,
            data_quality_target,
            completion_rate,
            completion_target,
            model_size,
            model_size_target,
            ass_cores,
            free_cores,
            self.boundaries,
        )
        return self.state, {}

    def reload_lgbn_model(self, df):
        self.lgbn = LGBN(show_figures=False, structural_training=False, df=df)
        logger.info("Retrained LGBN model for Env")


if __name__ == "__main__":
    env = LGBNTrainingEnv(
        ServiceType.CV, step_data_quality=32, step_cores=1, step_model_size=1
    )

    df_t = pd.read_csv("../share/metrics/LGBN.csv")
    env.reload_lgbn_model(df_t)
    env.reset()

    boundaries = env.es_registry.get_boundaries_minimalistic(ServiceType.CV, MAX_CORES)
    env.state = FullStateDQN(256, 288, 0, 2, 3, 5, 2, 6, boundaries)
    for i in range(1, 100):
        print(env.step(ESServiceAction.DILLY_DALLY))
        # print(env.state.discretize(ServiceType.CV))
        # print(env.state.discretize(ServiceType.QR))
