import logging
from random import randint
from typing import NamedTuple

import gymnasium

import utils
from agent.ES_Registry import ServiceType
from agent.LGBN import LGBN
from agent.SLO_Registry import calculate_slo_fulfillment, SLO_Registry, SLO, to_avg_SLO_F

logger = logging.getLogger("multiscale")

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))


class LGBN_Env(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.state: Full_State = None
        self.lgbn: LGBN = None
        self.slo_registry = SLO_Registry("../config/slo_config.json")

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

        new_state['fps'] = self.sample_values_from_lgbn(new_state['pixel'], new_state['cores'])['fps']
        self.state = Full_State(**new_state)

        # client_SLOs = self.slo_registry.get_SLOs_for_client("LGBN", ServiceType.QR_DEPRECATED)
        client_SLOs = {
            'pixel': SLO(**{'var': 'pixel', 'larger': True, 'thresh': self.state.pixel_thresh, 'weight': 1.0}),
            'fps': SLO(**{'var': 'fps', 'larger': False, 'thresh': self.state.fps_thresh, 'weight': 1.0})}
        reward = to_avg_SLO_F(calculate_slo_fulfillment(self.state._asdict(), client_SLOs)) + punishment_off
        return self.state, reward, False, False, {}

    # @utils.print_execution_time
    def sample_values_from_lgbn(self, pixel, cores):
        full_state = self.lgbn.predict_lgbn_vars({'pixel': pixel, 'cores': cores}, ServiceType.QR_DEPRECATED)
        return full_state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pixel = randint(1, 20) * 100
        cores = randint(1, PHYSICAL_CORES)
        avail_cores = PHYSICAL_CORES - cores - randint(0, PHYSICAL_CORES - cores)
        fps = self.sample_values_from_lgbn(pixel, cores)['fps']
        pixel_thresh = randint(5, 12) * 100
        fps_thresh = randint(20, 40)

        self.state = Full_State(pixel, pixel_thresh, fps, fps_thresh, cores, avail_cores)
        return self.state, {}

    def reload_lgbn_model(self, df):
        df['service_type'] = ServiceType.QR_DEPRECATED.value
        self.lgbn = LGBN(show_figures=False, structural_training=False, df=df)
        logger.info("Retrained LGBN model for Env")


class Full_State(NamedTuple):
    pixel: int
    pixel_thresh: int
    fps: float
    fps_thresh: int
    cores: int
    free_cores: int

    def for_tensor(self):
        return [self.pixel / self.pixel_thresh, self.fps / self.fps_thresh, self.cores,
                self.pixel > 100, self.pixel < 2000, self.free_cores > 0]
