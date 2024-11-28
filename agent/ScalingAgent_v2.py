import itertools
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from threading import Thread

import numpy as np

import utils
from DQN import DQN
from DockerClient import DockerClient
from HttpClient import HttpClient
from PrometheusClient import PrometheusClient
from agent.agent_utils import get_free_cores
from slo_config import MB, calculate_slo_reward

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")

logger = logging.getLogger("multiscale")
logger.setLevel(logging.INFO)

core_state = {}
access_state = threading.Lock()


class AIFAgent(Thread):
    def __init__(self, observed_container):
        super().__init__()

        self.prom_client = PrometheusClient()
        self.docker_client = DockerClient(DOCKER_SOCKET)
        self.observed_container = observed_container
        self.http_client = HttpClient()
        self.dqn = DQN(state_dim=5, action_dim=5)
        # Explore 4 combinations of Pixel / Cores if the model was not trained before
        self.explore_initial =  list(itertools.product([500, 1200], [3, 7])) if self.dqn.training_rounds != 0.5 else []
        self.unchanged_iterations = 0

    def run(self):
        while True:

            # TRAINING OCCASIONALLY #####
            if not self.dqn.currently_training and datetime.now() - self.dqn.last_time_trained > timedelta(seconds=60):
                Thread(target=self.dqn.train_dqn_from_env, args=(), daemon=True).start()

            # REAL INFERENCE ############
            state_pw = self.get_state_PW()
            state_pw_f = [state_pw['pixel'], state_pw['fps'], state_pw['cores'],
                          state_pw['free_cores'], state_pw['free_cores']]
            logger.info(f"Current SLO-F before change is {calculate_slo_reward(state_pw_f)}")

            if len(self.explore_initial) > 0:
                action_pw = 5  # Indicate exploration path
            else:
                action_pw = self.dqn.choose_action(np.array(state_pw_f), rand=0.15)
            self.act_on_env(action_pw, state_pw_f)

            time.sleep(4.5)

    def get_state_PW(self):
        metric_vars = list(set(MB['variables']) - set(MB['parameter']))

        obs_period = (self.unchanged_iterations + 1) * 2
        prom_metric_states = {}
        while len(prom_metric_states) < 2:
            prom_metric_states = self.prom_client.get_metric_values("|".join(metric_vars), period=f"{obs_period}s")
            if len(prom_metric_states) < 2:
                logger.warning("Need to query metrics again, result was incomplete")
                time.sleep(0.1)


        prom_parameter_states = {}
        while len(prom_parameter_states) < 2:
            prom_parameter_states = self.prom_client.get_metric_values("|".join(MB['parameter']))
            if len(prom_parameter_states) < 2:
                logger.warning("Need to query parameters again, result was incomplete")
                time.sleep(0.1)

        with access_state:
            free_cores = get_free_cores(core_state)

        return prom_metric_states | prom_parameter_states | {"free_cores": free_cores}

    def act_on_env(self, action, state_f):
        global core_state
        if action == 0:
            logger.info(f"No change requested, system conf stays at {state_f[0], state_f[2]}")
            self.unchanged_iterations += 1
        else:
            self.unchanged_iterations = 0

        if 1 <= action <= 2:
            delta_pixel = -100 if action == 1 else 100
            pixel_abs = state_f[0] + delta_pixel
            pixel_abs = np.clip(pixel_abs, 100, 2000)

            logger.info(f"Change pixel to {pixel_abs, state_f[2]}")
            self.http_client.change_config("localhost", {'pixel': int(pixel_abs)})

        elif 3 <= action <= 4:
            delta_cores = -1 if action == 3 else 1
            if delta_cores == +1 and state_f[4] == 0:
                logger.warning(f"Agent tried to scale a core, but none available")
                return

            cores_abs = state_f[2] + delta_cores
            logger.info(f"Change cores to {state_f[0], cores_abs}")
            # TODO: This is very error prone to forgetting one part + slow due to 2 requests
            self.docker_client.update_cpu(self.observed_container, int(cores_abs))
            self.http_client.change_threads("localhost", int(cores_abs))

            with access_state:
                core_state = core_state | {self.observed_container: cores_abs}

        elif action == 5:
            pixel, cores = self.explore_initial.pop()
            logger.info(f"Setting up interpolation, moving config to {pixel, cores}")

            self.http_client.change_config("localhost", {'pixel': int(pixel)})
            self.docker_client.update_cpu(self.observed_container, int(cores))
            self.http_client.change_threads("localhost", int(cores))
            with access_state:
                core_state = core_state | {self.observed_container: cores}


if __name__ == '__main__':
    agent = AIFAgent(observed_container="multiscaler-video-processing-a-1")
    agent.start()
