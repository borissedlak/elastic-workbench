import numpy as np

MB = {'variables': ['pixel', 'fps', 'cores', 'energy'],
      'parameter': ['pixel', 'cores'],
      'slos': [(800, False, 1.0),
               (30, False, 1.0),
               (10, True, 0.5),
               (100, True, 0.0),
               (1, False, 0.0)]}

PW_MAX_CORES = 10


def calculate_slo_reward(state, slos=MB['slos']):
    fuzzy_slof = []

    for index, value in enumerate(state):
        t, neg, boost = slos[index]

        if neg:
            slo_f = 1 - (value / t)
        else:
            slo_f = (value / t)

        slo_f = np.clip(slo_f, 0.0, 1.10) * boost
        fuzzy_slof.append(slo_f)

    return fuzzy_slof
