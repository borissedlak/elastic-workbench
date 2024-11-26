import logging
import os
import time
from threading import Thread

import numpy as np
from matplotlib import pyplot as plt

import utils
from DQN import DQNAgent
from DockerClient import DockerClient
from HttpClient import HttpClient
from PrometheusClient import INTERVAL, PrometheusClient
from agent.LGBN_Env import LGBN_Env, calculate_slo_reward
from slo_config import MB

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")
logging.getLogger("multiscale").setLevel(logging.INFO)


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
        # self.round_counter = 0
        self.dqn = DQNAgent(state_dim=2, action_dim=3)
        self.env = LGBN_Env()

    def run(self):
        while True:
            self.dqn.reset_networks()  # TODO: Resume training from intermediary point
            self.train()
            time.sleep(10)
            self.env.reload_lgbn_model()

    @utils.print_execution_time
    def train(self):
        score = 0.0
        score_list = []
        round_counter = 0

        while round_counter < 20 * 500:

            initial_state = self.env.state.copy()
            action = self.dqn.choose_action(np.array(self.env.state))
            next_state, reward, done, _, _ = self.env.step(action)
            # print(f"State transition {initial_state}, {action} --> {next_state}")

            self.dqn.memory.put((initial_state, action, reward, next_state, done))
            score += reward

            if self.dqn.memory.size() > self.dqn.batch_size:
                self.dqn.train_agent()

            round_counter += 1

            if round_counter % 500 == 0:
                self.env.reset()
                self.dqn.epsilon *= self.dqn.epsilon_decay

                print(
                    "EP:{}, Abs_Score:{:.1f}, Epsilon:{:.3f}, SLO-F:{:.2f}, State:{}".format(
                        round_counter, score, self.dqn.epsilon,
                        np.sum(calculate_slo_reward(self.env.state)),
                        self.env.state))
                score_list.append(score)
                score = 0.0

        # TODO: DO this through an animation or interactive plot
        plt.plot(score_list)
        plt.show()

    def get_state_PW(self):
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
