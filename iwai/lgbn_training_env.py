import logging
import os
from random import randint

import gymnasium
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
    def __init__(self, service_type, step_quality, step_cores=1, step_model_size=1):
        super().__init__()
        self.state: FullStateDQN = None
        self.lgbn: LGBN = None
        self.service_type: ServiceType = service_type
        self.es_registry = ESRegistry(ROOT + "/../config/es_registry.json")
        self.slo_registry = SLO_Registry(ROOT + "/../config/slo_config.json")

        self.step_quality = step_quality
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
        if action == 0:
            behavioral_punishment += 0.1  # Encourage the client not to oscillate

        if 1 <= action <= 2:
            delta_quality = -self.step_quality if action == 1 else self.step_quality
            new_quality = self.state.data_quality + delta_quality

            if (
                new_quality < self.boundaries["quality"]["min"]
                or new_quality > self.boundaries["quality"]["max"]
            ):
                behavioral_punishment = INVALID_ACTION_PUNISHMENT
                done = True
            else:
                new_state["quality"] = new_quality

        elif 3 <= action <= 4:
            # delta is always 1
            delta_cores = -self.step_cores if action == 3 else self.step_cores
            new_core = self.state.cores + delta_cores

            if new_core <= 0:  # Wants to go lower than 0 core
                behavioral_punishment = INVALID_ACTION_PUNISHMENT
                done = True
            elif (
                delta_cores > self.state.free_cores
            ):  # Want to consume resources that are not free
                behavioral_punishment = INVALID_ACTION_PUNISHMENT
                done = True
            else:
                new_state["cores"] = self.state.cores + delta_cores
                new_state["free_cores"] = new_state["free_cores"] - delta_cores

        elif 5 <= action <= 6:
            # step size is always 1
            delta_model = -self.step_model_size if action == 5 else self.step_model_size
            new_model_s = self.state.model_size + delta_model

            if (
                new_model_s < self.boundaries["model_size"]["min"]
                or new_model_s > self.boundaries["model_size"]["max"]
            ):
                behavioral_punishment = INVALID_ACTION_PUNISHMENT
                done = True
            else:
                new_state["model_size"] = new_model_s

        new_state["throughput"] = self.sample_from_lgbn(
            new_state["quality"], new_state["cores"], new_state["model_size"]
        )["throughput"]
        self.state = FullStateDQN(**new_state)

        reward = (
                to_normalized_slo_f(
                calculate_slo_fulfillment(self.state._asdict(), self.client_slos),
                self.client_slos,
            )
                + behavioral_punishment
        )
        return self.state, reward, done, False, {}

    def sample_from_lgbn(self, quality, cores, model_size):
        partial_state = {"quality": quality, "cores": cores, "model_size": model_size}
        full_state = self.lgbn.predict_lgbn_vars(partial_state, self.service_type)
        return full_state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        quality = randint(
            self.boundaries["quality"]["min"], self.boundaries["quality"]["max"]
        )
        quality = round(quality / self.step_quality) * self.step_quality
        quality_thresh = self.client_slos["quality"].thresh

        if self.service_type == ServiceType.CV:
            model_size = randint(
                self.boundaries["model_size"]["min"],
                self.boundaries["model_size"]["max"],
            )
            model_size_thresh = self.client_slos["model_size"].thresh
        else:
            model_size = 1
            model_size_thresh = 1

        ass_cores = randint(1, int(MAX_CORES / 2))
        free_cores = MAX_CORES - ass_cores

        throughput = self.sample_from_lgbn(quality, ass_cores, model_size)["throughput"]
        tp_thresh = self.client_slos["throughput"].thresh

        self.state = FullStateDQN(
            quality,
            quality_thresh,
            throughput,
            tp_thresh,
            model_size,
            model_size_thresh,
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
        ServiceType.CV, step_quality=32, step_cores=1, step_model_size=1
    )

    df_t = pd.read_csv("../share/metrics/LGBN.csv")
    env.reload_lgbn_model(df_t)
    env.reset()

    boundaries = env.es_registry.get_boundaries_minimalistic(ServiceType.CV, MAX_CORES)
    env.state = FullStateDQN(1000, 100, 0, 100, 5, 5, 1, 7, boundaries)
    for i in range(1, 100):
        # print(env.step(0))
        print(env.state.discretize(ServiceType.CV))
        print(env.state.discretize(ServiceType.QR))
