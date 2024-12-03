import numpy as np
import pandas as pd

from agent import agent_utils
from agent.ScalingAgent_v2 import ScalingAgent
from slo_config import Full_State


class Global_Service_Optimizer:
    def __init__(self, agents: [ScalingAgent]):
        self.s_agents = agents
        self.lgbn = agent_utils.train_lgbn_model(pd.read_csv("./LGBN.csv"), show_result=False)

    def evaluate_slof(self):
        pass

    def estimate_swapping(self):
        state_1: Full_State = self.s_agents[0].get_state_PW()
        state_2: Full_State = self.s_agents[1].get_state_PW()

        c_1, c_2, f_1, f_2 = state_1.cores, state_2.cores, state_1.free_cores, state_2.free_cores

        if f_1 != f_2:
            raise RuntimeError("Should not happen!!")
        elif f_1 > 0:
            print("No need to swap, not exhausted")
            return -1

    def swap_core(self):
        pass


def sample_values_from_lgbn(lgbn, pixel, cores):
    var, mean, vari = lgbn.predict(pd.DataFrame({'pixel': [pixel], 'cores': [cores]}))

    samples = {}
    for index, v in enumerate(var):
        mu, sigma = mean[0][index], np.sqrt(vari[index][index])
        sample_val = np.random.normal(mu, sigma, 1)[0]
        samples = samples | {v: sample_val}

    return float(samples['fps'])


if __name__ == "__main__":
    lgbn = agent_utils.train_lgbn_model(pd.read_csv("./LGBN.csv"), show_result=False)

    sample_values_from_lgbn(lgbn, 1800, 5)
    sample_values_from_lgbn(lgbn, 1800, 3)
