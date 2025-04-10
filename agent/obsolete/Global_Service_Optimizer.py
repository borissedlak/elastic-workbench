import logging

import numpy as np
import pandas as pd

from HttpClient import HttpClient
from agent import agent_utils
from agent.ScalingAgent_v2 import ScalingAgent
from agent.slo_config import Full_State, calculate_slo_reward

http_client = HttpClient()

logger = logging.getLogger("multiscale")


class Global_Service_Optimizer:
    def __init__(self, agents: [ScalingAgent]):
        self.s_agents = agents
        self.lgbn = agent_utils.train_lgbn_model(pd.read_csv("./LGBN.csv"), show_result=False)

    def estimate_swapping(self):
        state_1: Full_State = self.s_agents[0].get_state_PW()
        state_2: Full_State = self.s_agents[1].get_state_PW()

        c_1, c_2, f_1, f_2 = state_1.cores, state_2.cores, state_1.free_cores, state_2.free_cores

        if f_1 != f_2:
            raise RuntimeError("Should not happen!!")

        slo_f_1 = np.sum(calculate_slo_reward(state_1.for_tensor()))
        slo_f_2 = np.sum(calculate_slo_reward(state_2.for_tensor()))
        options = [(slo_f_1 + slo_f_2, slo_f_1, slo_f_2)]

        for combi in [(c_1 + 1, c_2 - 1), (c_1 - 1, c_2 + 1)]:
            if combi[0] < 1 or combi[1] < 1:
                options.append((0, 0, 0))
                continue

            fps_a = sample_values_from_lgbn(self.lgbn, state_1.pixel, combi[0])
            fps_b = sample_values_from_lgbn(self.lgbn, state_2.pixel, combi[1])

            s_new_1 = state_1._replace(fps=fps_a, cores=combi[0])
            s_new_2 = state_2._replace(fps=fps_b, cores=combi[1])

            slo_f_1 = np.sum(calculate_slo_reward(s_new_1.for_tensor()))
            slo_f_2 = np.sum(calculate_slo_reward(s_new_2.for_tensor()))
            options.append((slo_f_1 + slo_f_2, slo_f_1, slo_f_2))

        return options

    def swap_core(self, estimates):

        # state_1: Full_State = self.s_agents[0].get_state_PW()
        # state_2: Full_State = self.s_agents[1].get_state_PW()

        if estimates[1][0] > estimates[0][0] and estimates[1][0] > estimates[2][0]:
            print("Recommended to swap core from A <-- B")
            self.s_agents[1].act_on_env(3, self.s_agents[1].get_state_PW())
            self.s_agents[0].act_on_env(4, self.s_agents[0].get_state_PW())  # TODO: Should already be allowed

        elif estimates[2][0] > estimates[0][0] and estimates[2][0] > estimates[1][0]:
            print("Recommended to swap core from A --> B")
            self.s_agents[0].act_on_env(3, self.s_agents[0].get_state_PW())
            self.s_agents[1].act_on_env(4, self.s_agents[1].get_state_PW())  # TODO: Should already be allowed


def sample_values_from_lgbn(lgbn, pixel, cores):
    var, mean, vari = lgbn.predict(pd.DataFrame({'pixel': [pixel], 'cores': [cores]}))

    samples = {}
    for index, v in enumerate(var):
        mu, sigma = mean[0][index], np.sqrt(vari[index][index])
        sample_val = np.random.normal(mu, 0, 1)[0]
        samples = samples | {v: sample_val}

    return float(samples['fps'])

# if __name__ == "__main__":
#     lgbn = agent_utils.train_lgbn_model(pd.read_csv("../results/E2/LGBN.csv"), show_result=False)
#
#     print(sample_values_from_lgbn(lgbn, 1400, 5))
#     print(sample_values_from_lgbn(lgbn, 1300, 3))
