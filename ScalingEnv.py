from random import randint

import gymnasium
import numpy as np
import pandas as pd
from gymnasium import spaces

import utils

MB = {'variables': ['fps', 'pixel', 'energy', 'cores'],
      'parameter': ['pixel', 'cores'],
      'slos': [('pixel', utils.sigmoid, 0.015, 450, 0.8),
               ('fps', utils.sigmoid, 0.35, 25, 1.6)]}


class ScalingEnv(gymnasium.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.regression_model = utils.get_regression_model(pd.read_csv("regression_data.csv"))
        self.pixel = None
        self.reset()

        # Define the action space (e.g., discrete actions: 0, 1, 2)
        self.action_space = spaces.Discrete(3)

        # Define the observation space (e.g., a continuous vector with 2 elements)
        self.observation_space = spaces.Box(low=np.array([100, 0]),
                                            high=np.array([2000, 999]),
                                            dtype=np.int64)

        # Initialize the state
        self.state = None
        self.done = False

    def get_current_state(self):
        try:
            fps_infer = int(self.regression_model.predict([[self.pixel, 2.0]])[0])
            return np.array([self.pixel, fps_infer])
        except ValueError:
            print("Error")
            self.pixel = randint(100, 2000)
            return self.get_current_state()

    def step(self, action):
        self.pixel = int(self.pixel + action)

        punishment_off = 0
        if self.pixel < 100 or self.pixel > 2000:
            self.pixel = np.clip(self.pixel, 100, 2000)
            punishment_off = - 5

        updated_state = {'pixel': self.get_current_state()[0],
                         'fps': self.get_current_state()[1]}
        return self.get_current_state(), np.sum(
            calculate_value_slo(updated_state)) + punishment_off, self.done, False, None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pixel = randint(1, 20) * 100

        return self.get_current_state(), {}


def calculate_value_slo(state, slos=MB['slos']):
    fuzzy_slof = []

    for var_name, value in state.items():
        if var_name not in [v[0] for v in slos]:
            continue

        # var, func, k, c, boost = utils.filter_tuple(slos, var_name, 0)
        # fuzzy_slof.append(boost * func(value, k, c))

        if var_name == "pixel":
            fuzzy_slof.append(value >= 800)
        elif var_name == "fps":
            fuzzy_slof.append(value >= 20)
        else:
            raise RuntimeError("WHY??")

    return fuzzy_slof


if __name__ == '__main__':
    env = ScalingEnv()
    initial_state = env.get_current_state()
    print(f"Initial State: {initial_state}")
    print(
        f"Reward for current state: {np.sum(calculate_value_slo({"pixel": initial_state[0], "fps": initial_state[1]}))}")
    env.act_on_env(1200)
    updated_state = env.get_current_state()
    print(f"Updated State: {updated_state}")
    print(
        f"Reward for updated state: {np.sum(calculate_value_slo({"pixel": updated_state[0], "fps": updated_state[1]}))}")
    env.act_on_env(700)
    updated_state = env.get_current_state()
    print(f"Updated State: {updated_state}")
    print(
        f"Reward for updated state: {np.sum(calculate_value_slo({"pixel": updated_state[0], "fps": updated_state[1]}))}")
