import numpy as np

# Define sigmoid using numpy
# def sigmoid(x, k=1, c=0):
#     return 1 / (1 + np.exp(-k * (x - c)))
#
#
# def linear(x):
#     max_val = 100 * PW_MAX_CORES
#     return np.clip(max_val - (x * PW_MAX_CORES), 0, 100)


MB = {'variables': ['pixel', 'fps', 'cores', 'energy'],
      'parameter': ['pixel', 'cores'],
      'slos': [(800, False, 1.0),
               (30, False, 1.0),
               (10, True, 0.5),  # No preference for # of cores
               (100, True, 0.0)]}

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

        # punishment = 0
        # if index == 0 and slo_f > 1.15:
        #     punishment = (slo_f - 1.15) / 2  # Might need to scale this

        # slo_f -= punishment

        fuzzy_slof.append(slo_f)

    return fuzzy_slof
