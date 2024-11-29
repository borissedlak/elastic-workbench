import logging
from random import randint

import gymnasium
import numpy as np
import pandas as pd
from pgmpy.models import LinearGaussianBayesianNetwork

from agent import agent_utils
from slo_config import calculate_slo_reward, PW_MAX_CORES, Full_State

logger = logging.getLogger("multiscale")


class LGBN_Env(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.state: Full_State = None
        self.lgbn: LinearGaussianBayesianNetwork = None

    def step(self, action):
        punishment_off = 0
        new_state = self.state._asdict()

        # Do nothing at 0
        if 1 <= action <= 2:
            delta_pixel = -100 if action == 1 else 100
            if new_state['pixel'] == 100 or new_state['pixel'] >= 2000:
                punishment_off = - 5
            else:
                new_state['pixel'] = new_state['pixel'] + delta_pixel

        elif 3 <= action <= 4:
            delta_cores = -1 if action == 3 else 1

            if delta_cores == -1 and new_state['cores'] == 1:  # Want to go lower
                punishment_off = - 10
            elif delta_cores == +1 and new_state['free_cores'] <= 0:  # Want to consume what does not exist
                punishment_off = - 10
            else:
                new_state['cores'] = new_state['cores'] + delta_cores
                new_state['free_cores'] = new_state['free_cores'] - delta_cores

        new_state['fps'], new_state['energy'] = self.sample_values_from_lgbn(new_state['pixel'], new_state['cores'])
        self.state = Full_State(**new_state)

        reward = np.sum(calculate_slo_reward(self.state.for_tensor())) + punishment_off
        return self.state, reward, False, False, {}

    # @utils.print_execution_time
    # NTH: Make this more modular
    def sample_values_from_lgbn(self, pixel, cores):
        var, mean, vari = self.lgbn.predict(pd.DataFrame({'pixel': [pixel], 'cores': [cores]}))

        samples = {}
        for index, v in enumerate(var):
            mu, sigma = mean[0][index], np.sqrt(
                vari[index][index])  # [[  255.21202708 27200.09321573], [  788.21188594 19991.6047412 ]]
            sample_val = np.random.normal(mu, sigma, 1)[0]
            samples = samples | {v: sample_val}

        return float(samples['fps']), int(samples['energy'])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pixel = randint(1, 20) * 100
        cores = randint(1, PW_MAX_CORES)
        avail_cores = PW_MAX_CORES - cores - randint(0, PW_MAX_CORES - cores)
        fps, energy = self.sample_values_from_lgbn(pixel, cores)
        pixel_thresh = randint(6, 10) * 100
        fps_thresh = randint(15, 45)

        self.state = Full_State(pixel, pixel_thresh, fps, fps_thresh, energy, cores, avail_cores)
        return self.state, {}

    def reload_lgbn_model(self):
        self.lgbn = agent_utils.train_lgbn_model(show_result=False)
        logger.info("Retrained LGBN model for Env")
