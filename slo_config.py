import numpy as np

import utils

MB = {'variables': ['fps', 'pixel', 'energy', 'cores'],
      'parameter': ['pixel', 'cores'],
      'slos': [(800, utils.sigmoid, 0.015, 450, 1.0),
               (30, utils.sigmoid, 0.35, 25, 1.0),
               (1, utils.sigmoid, 0.35, 25, 1.0)]}

PW_MAX_CORES = 10


def calculate_slo_reward(state, slos=MB['slos']):
    fuzzy_slof = []

    for index, value in enumerate(state):
        t, func, k, c, boost = slos[index]
        # slo_f = boost * func(value, k, c)
        slo_f = (value / t)

        # TODO: Right now this only punished high pixel, but it also needs an explicit energy SLO
        punishment = 0
        if index == 0 and slo_f > 1.15:
            punishment = (slo_f - 1.15) / 2 # Might need to scale this

        slo_f = np.clip(slo_f, 0.0, 1.15)
        slo_f -= punishment

        fuzzy_slof.append(slo_f)

    return fuzzy_slof
