import logging
from random import randint

import gymnasium
import numpy as np
import pandas as pd
from pgmpy.models import LinearGaussianBayesianNetwork

from agent import agent_utils
from slo_config import calculate_slo_reward, PW_MAX_CORES

logger = logging.getLogger("multiscale")


class LGBN_Env(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.state = None
        self.lgbn: LinearGaussianBayesianNetwork = None

    def step(self, action):
        punishment_off = 0

        # Do nothing at 0
        if 1 <= action <= 2:
            delta_pixel = -100 if action == 1 else 100
            self.state[0] += delta_pixel
            if self.state[0] < 100 or self.state[0] > 2000:
                self.state[0] = np.clip(self.state[0], 100, 2000)
                punishment_off = - 10

        elif 3 <= action <= 4:
            delta_cores = -1 if action == 3 else 1
            self.state[2] += delta_cores
            if self.state[2] < 1 or self.state[2] > PW_MAX_CORES:
                self.state[2] = np.clip(self.state[2], 1, PW_MAX_CORES)
                punishment_off = - 10

        self.state[1], self.state[3] = self.sample_values_from_lgbn(self.state[0], self.state[2])

        reward = np.sum(calculate_slo_reward(self.state)) + punishment_off
        return self.state, reward, False, False, {}

    # @utils.print_execution_time
    # NTH: Make this more modular
    def sample_values_from_lgbn(self, pixel, cores):
        var, mean, vari = self.lgbn.predict(pd.DataFrame({'pixel': [pixel], 'cores': [cores]}))

        samples = {}
        for index, v in enumerate(var):
            mu, sigma = mean[0][index], np.sqrt(vari[index][index]) # [[  255.21202708 27200.09321573], [  788.21188594 19991.6047412 ]]
            sample_val = np.random.normal(mu, sigma, 1)[0]
            samples = samples | {v: sample_val}

        return samples['fps'], samples['energy']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pixel = randint(1, 20) * 100
        cores = randint(1, PW_MAX_CORES)
        avail_cores = PW_MAX_CORES - cores - randint(0, PW_MAX_CORES - cores)
        fps, energy = self.sample_values_from_lgbn(pixel, cores)
        self.state = [pixel, fps, cores, energy]

        return self.state, {}

    def reload_lgbn_model(self):
        self.lgbn = agent_utils.train_lgbn_model(show_result=False)
        logger.info("Retrained LGBN model for Env")
