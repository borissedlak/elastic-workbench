import logging
import os
import time
from threading import Thread

import numpy as np
from matplotlib import pyplot as plt

import utils
from DQN import DQN
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
        self.dqn = DQN(state_dim=2, action_dim=3)

    def run(self):
        while True:

            # REAL INFERENCE ############

            # TRAINING OCCASIONALLY #####

            Thread(target=self.dqn.train_dqn_from_env, args=(), daemon=True).run()

            time.sleep(60)

    # def retrain_Q_network(self):
    #     self.train()


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
