import logging
import os
import time
from datetime import datetime, timedelta
from threading import Thread

import numpy as np

import utils
from DQN import DQN
from DockerClient import DockerClient
from HttpClient import HttpClient
from PrometheusClient import INTERVAL, PrometheusClient
from slo_config import MB

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")

logger = logging.getLogger("multiscale")
logger.setLevel(logging.INFO)


# TODO: So what the agent must do on a high level is:
#  1) Collect sensory state from Prometheus --> Easy ✓
#  2) Evaluate if SLOs are fulfilled --> Easy ✓
#  3) Retrain its interpretation model --> Medium ✓
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
        self.dqn = DQN(state_dim=2, action_dim=3)

    def run(self):
        while True:

            #### TRAINING OCCASIONALLY #####
            if not self.dqn.currently_training and datetime.now() - self.dqn.last_time_trained > timedelta(seconds=1):
                Thread(target=self.dqn.train_dqn_from_env, args=(), daemon=True).start()

            #### REAL INFERENCE ############
            state_pw = self.get_state_PW()
            state_pw_f = [state_pw['pixel'], state_pw['fps']]

            # TODO: This should have an own exploration factor which is a consequence of the model quality
            action_pw = self.dqn.choose_action(np.array(state_pw_f))
            # self.act_on_env(action_pw, state_pw_f)

            time.sleep(1)  # TODO: Should be 5s

    # def retrain_Q_network(self):
    #     self.train()

    def get_state_PW(self):
        metric_vars = list(set(MB['variables']) - set(MB['parameter']))
        prom_metric_states = self.prom_client.get_metric_values("|".join(metric_vars), period=INTERVAL)
        prom_parameter_states = self.prom_client.get_metric_values("|".join(MB['parameter']))

        cpu_cores = os.cpu_count()
        return prom_metric_states | prom_parameter_states | {"max_cores": cpu_cores}

    def act_on_env(self, a_pixel, state_f):

        if 0 <= a_pixel < 3:
            delta_pixel = int((a_pixel - 1) * 100)
            pixel_abs = state_f[0] + delta_pixel

            logger.info(f"Request pixel to change to {pixel_abs}")
            self.http_client.change_config("localhost", {'pixel': int(pixel_abs)})


if __name__ == '__main__':
    agent = AIFAgent()
    agent.run()
