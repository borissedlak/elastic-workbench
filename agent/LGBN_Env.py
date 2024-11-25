from random import randint

import gymnasium
import numpy as np
import pandas as pd

from agent import agent_utils
from slo_config import MB


class LGBN_Env(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.state = None
        self.lgbn = None
        self.reset()
        self.reload_lgbn_model()
        self.done = False  # TODO: How can I optimize rounds with done?

    # def get_current_state(self):
    #     fps_infer = int(self.regression_model.predict([[self.pixel, 2.0]])[0])
    #     return np.array([self.pixel, fps_infer])

    def step(self, action):
        punishment_off = 0

        if 0 <= action < 3:
            delta_pixel = int((action - 1) * 100)
            self.state[0] += delta_pixel
            if self.state[0] < 100 or self.state[0] > 2000:
                self.state[0] = np.clip(self.state[0], 100, 2000)
                punishment_off = - 10
        elif 3 <= action < 6:
            punishment_off = - 10
        elif 6 <= action < 9:
            punishment_off = - 10

        #TODO: Finish this; get sample from std and mean
        self.lgbn.predict(pd.DataFrame({'pixel': [self.state[0]]}))
        self.state[1] = 0

        reward = np.sum(calculate_slo_reward(self.state)) + punishment_off
        return self.state, reward, self.done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = [randint(1, 20) * 100]

        return self.state, {}

    def reload_lgbn_model(self):
        self.lgbn = agent_utils.train_lgbn_model()


def calculate_slo_reward(state, slos=MB['slos']):
    fuzzy_slof = []

    for index, value in enumerate(state):
        func, k, c, boost = slos[index]
        slo_f = boost * func(value, k, c)

        slo_f = np.clip(slo_f, 0.0, 1.0)
        # if slo_f > 1:
        #     slo_f = 2 - slo_f

        fuzzy_slof.append(slo_f)

    return fuzzy_slof
