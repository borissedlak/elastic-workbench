from random import randint

import gymnasium
import numpy as np
import pandas as pd

from agent import agent_utils
from slo_config import MB


class ScalingEnv(gymnasium.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.regression_model = agent_utils.get_regression_model(pd.read_csv("../../metrics/regression_data.csv"))
        self.pixel = None
        self.reset()

        # Define the action space (e.g., discrete actions: 0, 1, 2)
        # self.action_space = spaces.Discrete(9)
        #
        # # Define the observation space (e.g., a continuous vector with 2 elements)
        # self.observation_space = spaces.Box(low=np.array([100, 0]),
        #                                     high=np.array([2000, 999]),
        #                                     dtype=np.int64)

        # Initialize the state
        self.state = None
        self.done = False

    def get_current_state(self):
        fps_infer = int(self.regression_model.predict([[self.pixel, 2.0]])[0])
        return np.array([self.pixel, fps_infer])

    def step(self, action):
        punishment_off = 0
        if 0 <= action < 3:
            self.pixel = int(self.pixel + ((action - 1) * 100))

            if self.pixel < 100 or self.pixel > 2000:
                self.pixel = np.clip(self.pixel, 100, 2000)
                punishment_off = - 10
        elif 3 <= action < 6:

            punishment_off = - 10
        elif 6 <= action < 9:

            punishment_off = - 10

        reward = np.sum(calculate_slo_reward(self.get_current_state())) + punishment_off
        return self.get_current_state(), reward, self.done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pixel = randint(1, 20) * 100

        return self.get_current_state(), {}


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
