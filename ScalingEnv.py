from random import randint

import numpy as np
import pandas as pd

import utils


MB = {'variables': ['fps', 'pixel', 'energy', 'cores'],
      'parameter': ['pixel', 'cores'],
      'slos': [('pixel', utils.sigmoid, 0.015, 450, 0.8),
               ('fps', utils.sigmoid, 0.35, 25, 1.6)]}

class ScalingEnv:
    def __init__(self):
        self.regression_model = utils.get_regression_model(pd.read_csv("regression_data.csv"))
        self.pixel = randint(100, 1200)

    def get_current_state(self):
        try:
           return [self.pixel, int(self.regression_model.predict([[self.pixel, 2.0]])[0])]
        except ValueError:
            print("Error")
            self.pixel = randint(100, 2000)
            return self.get_current_state()

    def act_on_env(self, pixel):
        self.pixel = pixel


def calculate_value_slo(state, slos=MB['slos']):
    fuzzy_slof = []

    for var_name, value in state.items():
        if var_name not in [v[0] for v in slos]:
            continue

        var, func, k, c, boost = utils.filter_tuple(slos, var_name, 0)
        fuzzy_slof.append(boost * func(value, k, c))

    return fuzzy_slof


if __name__ == '__main__':
    env = ScalingEnv()
    initial_state = env.get_current_state()
    print(f"Initial State: {initial_state}")
    print(f"Reward for current state: {np.sum(calculate_value_slo({"pixel": initial_state[0], "fps": initial_state[1]}))}")
    env.act_on_env(1200)
    updated_state = env.get_current_state()
    print(f"Updated State: {updated_state}")
    print(f"Reward for updated state: {np.sum(calculate_value_slo({"pixel": updated_state[0], "fps": updated_state[1]}))}")
    env.act_on_env(700)
    updated_state = env.get_current_state()
    print(f"Updated State: {updated_state}")
    print(f"Reward for updated state: {np.sum(calculate_value_slo({"pixel": updated_state[0], "fps": updated_state[1]}))}")