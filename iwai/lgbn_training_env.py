import logging
import os
from random import randint

import gymnasium
import numpy as np
import pandas as pd

import utils
from agent.es_registry import ServiceType, ESRegistry
from agent.LGBN import LGBN
from agent.SLORegistry import (
    calculate_slo_fulfillment,
    to_normalized_slo_f,
    SLO_Registry,
)
from agent.agent_utils import FullStateDQN
from proj_types import ESServiceAction

logger = logging.getLogger("multiscale")

ROOT = os.path.dirname(__file__)
MAX_CORES = int(utils.get_env_param("MAX_CORES", 8))
INVALID_ACTION_PUNISHMENT = -5


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
            self.service_type, {"C_1": 100}
        )[0]

    def step(self, action: ESServiceAction):
        behavioral_punishment = 0
        new_state = self.state._asdict()
        done = False

        # Do nothing at 0
        if action.value == 0:
            pass
            # behavioral_punishment += 0.1  # Encourage the client not to oscillate

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
        )
        reward += 0.1 if action == 0 and reward > 0.8 else 0.0

        return self.state, reward, done, False, {}

    def sample_throughput_from_lgbn(self, data_quality, cores, model_size):
        partial_state = {"data_quality": data_quality, "cores": cores, "model_size": model_size}
        full_state = self.lgbn.predict_lgbn_vars(partial_state, self.service_type)
        return full_state['throughput']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # data_quality = randint(
        #     self.boundaries["data_quality"]["min"], self.boundaries["data_quality"]["max"]
        # )
        # data_quality = round(data_quality / self.step_data_quality) * self.step_data_quality
        data_quality = 256 if self.service_type == ServiceType.CV else 700
        data_quality_target = self.client_slos["data_quality"].target

        if self.service_type == ServiceType.CV:
            # model_size = randint(
            #     self.boundaries["model_size"]["min"],
            #     self.boundaries["model_size"]["max"],
            # )
            model_size = 3
            model_size_target = self.client_slos["model_size"].target
        else:
            model_size = 1
            model_size_target = 1

        ass_cores = 2  # randint(1, int(MAX_CORES / 2))
        free_cores = MAX_CORES - ass_cores

        throughput = self.sample_throughput_from_lgbn(data_quality, ass_cores, model_size)
        tp_target = self.client_slos["throughput"].target

        self.state = FullStateDQN(
            data_quality,
            data_quality_target,
            throughput,
            tp_target,
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
