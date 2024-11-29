from typing import NamedTuple

import numpy as np


class Full_State(NamedTuple):
    pixel: int
    pixel_thresh: int
    fps: float
    fps_thresh: int
    energy: int
    cores: int
    free_cores: int

    def for_tensor(self):
        return [self.pixel / self.pixel_thresh, self.fps / self.fps_thresh, self.cores,
                self.can_change_pixel(), self.free_cores]

    # return +1 if I can only change up, and -1 if I can only decrease, 0 if all possible
    def can_change_pixel(self):
        if self.pixel == 100:
            return 1.0
        elif self.pixel == 2000:
            return -1
        else:
            return 0

    # # return +1 if I can only change up, and -1 if I can only decrease, 0 if all possible
    # def can_change_cores(self):
    #     if self.free_cores > 0:
    #         return 1.0
    #     elif self.pixel == 2000:
    #         return -1
    #     else:
    #         return 0


MB = {'variables': ['pixel', 'fps', 'cores', 'energy'],
      'parameter': ['pixel', 'cores'],
      'slos': [(1.0, False, 1.0),
               (1.0, False, 1.0),
               (10, True, 0.5),
               (1, False, 0.0),
               (1, False, 0.0)]}

PW_MAX_CORES = 10


def calculate_slo_reward(state, slos=MB['slos']):
    fuzzy_slof = []

    # slos[0] = (state[4], True, 1.0)
    # slos[1] = (state[5], True, 1.0)

    for index, value in enumerate(state):
        t, neg, boost = slos[index]

        if neg:
            slo_f = 1 - (value / t)
        else:
            slo_f = (value / t)

        slo_f = np.clip(slo_f, 0.0, 1.10) * boost
        fuzzy_slof.append(slo_f)

    return fuzzy_slof
