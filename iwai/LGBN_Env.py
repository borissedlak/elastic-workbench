import logging
from random import randint
from typing import NamedTuple

import gymnasium

import utils
from agent.ES_Registry import ServiceType
from agent.LGBN import LGBN
from agent.SLO_Registry import calculate_slo_fulfillment, SLO, to_avg_SLO_F
from agent.agent_utils import Full_State

logger = logging.getLogger("multiscale")

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))


class LGBN_Env(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.state: Full_State = None
        self.lgbn: LGBN = None

    def step(self, action):
        behavioral_punishment = 0
        new_state = self.state._asdict()
        done = False

        # Do nothing at 0
        if action == 0:
            behavioral_punishment = 0.1  # Encourage the client not to oscillate

        # Do nothing at 0
        if 1 <= action <= 2:
            delta_quality = -100 if action == 1 else 100
            if (self.state.quality + delta_quality < 300) or (self.state.quality + delta_quality > 1100):
                behavioral_punishment = - 50
                done = True
            else:
                new_state['quality'] = new_state['quality'] + delta_quality

        elif 3 <= action <= 4:
            delta_cores = -1 if action == 3 else 1

            if delta_cores == -1 and self.state.cores == 1:  # Want to go lower
                behavioral_punishment = - 50
                done = True
            elif delta_cores == +1 and self.state.free_cores <= 0:  # Want to consume what does not exist
                behavioral_punishment = - 50
                done = True
            else:
                new_state['cores'] = new_state['cores'] + delta_cores
                new_state['free_cores'] = new_state['free_cores'] - delta_cores

        new_state['throughput'] = self.sample_values_from_lgbn(new_state['quality'], new_state['cores'])['throughput']
        self.state = Full_State(**new_state)

        client_SLOs = {
            'quality': SLO(**{'var': 'quality', 'larger': True, 'thresh': self.state.quality_thresh, 'weight': 1.0}),
            'throughput': SLO(**{'var': 'throughput', 'larger': True, 'thresh': self.state.tp_thresh, 'weight': 1.0})}

        reward = to_avg_SLO_F(calculate_slo_fulfillment(self.state._asdict(), client_SLOs)) + behavioral_punishment
        return self.state, reward, done, False, {}

    def sample_values_from_lgbn(self, quality, cores):
        full_state = self.lgbn.predict_lgbn_vars({'quality': quality, 'cores': cores}, ServiceType.QR_DEPRECATED)
        return full_state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        quality = randint(3, 11) * 100
        quality_thresh = randint(3, 10) * 100
        ass_cores = randint(1, PHYSICAL_CORES)
        free_cores = PHYSICAL_CORES - ass_cores
        throughput = self.sample_values_from_lgbn(quality, ass_cores)['throughput']
        tp_thresh = randint(10, 100)

        self.state = Full_State(quality, quality_thresh, throughput, tp_thresh, ass_cores, free_cores)
        return self.state, {}

    def reload_lgbn_model(self, df):
        df['service_type'] = ServiceType.QR_DEPRECATED.value
        self.lgbn = LGBN(show_figures=False, structural_training=False, df=df)
        logger.info("Retrained LGBN model for Env")
