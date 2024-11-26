import numpy as np

import utils

MB = {'variables': ['fps', 'pixel', 'energy', 'cores'],
      'parameter': ['pixel', 'cores'],
      'slos': [(utils.sigmoid, 0.015, 450, 1.0),
               (utils.sigmoid, 0.35, 25, 1.0)]}


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
