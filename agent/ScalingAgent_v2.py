import os
from threading import Thread

import numpy as np
import torch
from matplotlib import pyplot as plt

import utils
from DQN import DQNAgent
from DockerClient import DockerClient
from HttpClient import HttpClient
from PrometheusClient import INTERVAL, PrometheusClient
from agent import agent_utils
from agent.LGBN_Env import LGBN_Env, calculate_slo_reward
from slo_config import MB

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")


# TODO: So what the agent must do on a high level is:
#  1) Collect sensory state from Prometheus --> Easy ✓
#  2) Evaluate if SLOs are fulfilled --> Easy ✓
#  3) Retrain its interpretation model --> Medium
#  4) Act so that SLO-F is optimized --> Hard
#  _
#  However, I assume that the agent has the variable DAG and the SLO thresholds
#  And I dont have to resolve variable names dynamically, but keep them hardcoded


class AIFAgent(Thread):
    def __init__(self):
        super().__init__()

        self.prom_client = PrometheusClient()
        self.docker_client = DockerClient(DOCKER_SOCKET)
        self.http_client = HttpClient()
        self.round_counter = 0
        self.dqn = DQNAgent(state_dim=2, action_dim=3)
        self.env = LGBN_Env()

    def run(self):
        score = 0.0
        score_list = []

        while self.round_counter < 40 * 500:
            # while True:

            # 1) Get initial state s ################################

            initial_state = self.env.get_current_state()

            # 2) Get action from policy #############################                                                                                                                                                                                                                                                                                       #######################

            # random = self.round_counter % 10 == 0 or self.round_counter < 1000  # e - greedy with 0.1
            action = self.dqn.choose_action(torch.FloatTensor(np.array(initial_state)))

            # 3) Enact on environment ###############################

            next_state, reward, _, _, _ = self.env.step(action)

            # 4) Get updated state s' ###############################

            # time.sleep(5.0)
            next_state = self.env.get_current_state()
            # print(f"State transition {initial_state_f} --> {updated_state_f}")

            # 5) Calculate reward for s' ############################

            self.dqn.memory.put((initial_state, action, reward, next_state))
            score += reward

            # 6) Retrain the agents networks ########################

            if self.dqn.memory.size() > self.dqn.batch_size:
                self.dqn.train_agent()  # Probably due to buffer that is trained

            self.round_counter += 1

            if self.round_counter % 500 == 0:
                self.env.reset()
                self.dqn.epsilon *= self.dqn.epsilon_decay

                print(
                    "EP:{}, Abs_Score:{:.1f}, Epsilon:{:.3f}, SLO-F:{:.2f}, State:{}".format(
                        self.round_counter, score, self.dqn.epsilon,
                        np.sum(calculate_slo_reward(self.env.get_current_state())),
                        self.env.get_current_state()))
                score_list.append(score)
                score = 0.0

        # TODO: DO this through an animation or interactive plot
        plt.plot(score_list)
        plt.show()

    def get_current_state(self):
        metric_vars = list(set(MB['variables']) - set(MB['parameter']))
        prom_metric_states = self.prom_client.get_metric_values("|".join(metric_vars), period=INTERVAL)
        prom_parameter_states = self.prom_client.get_metric_values("|".join(MB['parameter']))

        cpu_cores = os.cpu_count()
        return prom_metric_states | prom_parameter_states | {"max_cores": cpu_cores}

    def act_on_env(self, a_pixel):
        self.http_client.change_config("localhost", {'pixel': int(a_pixel)})


if __name__ == '__main__':
    agent = AIFAgent()
    agent.run()
