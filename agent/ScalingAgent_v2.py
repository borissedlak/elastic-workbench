import itertools
import logging
import threading
import time
from threading import Thread

import numpy as np

import utils
from DockerClient import DockerClient, DockerInfo
from HttpClient import HttpClient
from PrometheusClient import PrometheusClient
from agent.DQN import DQN
from agent.DQN import STATE_DIM
from agent.agent_utils import get_free_cores, log_agent_experience
from slo_config import MB, calculate_slo_reward, Full_State, PW_MAX_CORES

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")
# MAX_CORES = utils.get_env_param('MAX_CORES', 10)

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)

core_state = {}
access_state = threading.Lock()


class ScalingAgent(Thread):
    def __init__(self, container: DockerInfo, prom_server, thresholds, dqn=None, log=None, max_cores=PW_MAX_CORES):
        super().__init__()

        self.container = container
        self.prom_client = PrometheusClient(prom_server)
        self.docker_client = DockerClient(DOCKER_SOCKET)
        self.http_client = HttpClient()
        self.log = log
        self.max_cores = max_cores

        self.dqn = dqn
        if self.dqn is None:
            self.dqn = DQN(state_dim=STATE_DIM, action_dim=5)

        # Explore 4 combinations of Pixel / Cores if the model was not trained before
        self.explore_initial = list(itertools.product([500, 1200], [3, 7])) if self.dqn.training_rounds != 0.5 else []
        self.unchanged_iterations = 0
        self.thresholds = thresholds
        self._running = True
        self._idle = False

    def run(self):
        global core_state

        initial_state = self.get_state_PW()
        with access_state:
            core_state = core_state | {self.container.container_id: initial_state.cores}
            logger.info(core_state)

        while self._running:
            # TRAINING OCCASIONALLY ##### # TODO: Pause this for now, it only destroys the result
            # if not self.dqn.currently_training and datetime.now() - self.dqn.last_time_trained > timedelta(seconds=60):
            #     Thread(target=self.dqn.train_dqn_from_env, args=(), daemon=True).start()

            # REAL INFERENCE ############
            state_pw = self.get_state_PW()
            logger.debug(f"Current state before change is {state_pw}")
            logger.debug(f"Current SLO-F before change is {calculate_slo_reward(state_pw.for_tensor())}")
            log_agent_experience(state_pw, self.log) if self.log else None

            if len(self.explore_initial) > 0:
                action_pw = 5  # Indicate exploration path
            else:
                action_pw = self.dqn.choose_action(np.array(state_pw.for_tensor()), rand=0.15)

            if not self._idle:
                self.act_on_env(action_pw, state_pw)

            time.sleep(5)

    def stop(self):
        self._running = False

    def set_idle(self, idle):
        self._idle = idle

    def has_free_cores(self):
        with access_state:
            free_cores = get_free_cores(core_state, self.max_cores)

        return free_cores > 0


    def get_state_PW(self) -> Full_State:
        metric_str = "|".join(list(set(MB['variables']) - set(MB['parameter'])))

        period = (self.unchanged_iterations + 1) * 2
        prom_metrics = {}
        while len(prom_metrics) < 1:
            prom_metrics = self.prom_client.get_metrics(metric_str, period=f"{period}s", instance=self.container.ip_a)
            if len(prom_metrics) < 1:
                logger.warning("Need to query metrics again, result was incomplete")
                time.sleep(0.1)

        prom_parameters = {}
        while len(prom_parameters) < 2:
            prom_parameters = self.prom_client.get_metrics("|".join(MB['parameter']), instance=self.container.ip_a)
            if len(prom_parameters) < 2:
                logger.warning("Need to query parameters again, result was incomplete")
                time.sleep(0.1)

        with access_state:
            free_cores = get_free_cores(core_state, self.max_cores)

        state_dict = prom_metrics | prom_parameters | {"free_cores": free_cores}
        state_pw_f = Full_State(state_dict['pixel'], self.thresholds[0], state_dict['fps'], self.thresholds[1],
                                0, state_dict['cores'], state_dict['free_cores'])
        return state_pw_f

    def act_on_env(self, action, state_f: Full_State):
        global core_state
        if action == 0:
            logger.info(f"{self.container.alias}| No change requested, system stays at {state_f.pixel, state_f.cores}")
            self.unchanged_iterations += 1
        else:
            self.unchanged_iterations = 0

        if 1 <= action <= 2:
            delta_pixel = -100 if action == 1 else 100
            pixel_abs = state_f.pixel + delta_pixel
            pixel_abs = np.clip(pixel_abs, 100, 2000)

            logger.info(f"{self.container.alias}| Change pixel to {pixel_abs, state_f.cores}")
            self.http_client.change_config(self.container.ip_a, {'pixel': int(pixel_abs)})

        elif 3 <= action <= 4:
            delta_cores = -1 if action == 3 else 1
            if delta_cores == +1 and state_f.free_cores == 0:
                logger.warning(f"{self.container.alias}| Agent tried to scale up a core, but none available")
                return

            if delta_cores == -1 and state_f.cores == 1:
                logger.warning(f"{self.container.alias}| Agent tried to scale down a core, but only one using")
                return

            cores_abs = state_f.cores + delta_cores
            logger.info(f"{self.container.alias}| Change cores to {state_f.pixel, cores_abs}")
            self.http_client.change_threads(self.container.ip_a, int(cores_abs))

            with access_state:
                core_state = core_state | {self.container.container_id: cores_abs}
                logger.info(core_state)

        elif action == 5:
            pixel, cores = self.explore_initial.pop()
            logger.info(f"{self.container.alias}| Setting up interpolation, moving config to {pixel, cores}")

            self.http_client.change_config(self.container.ip_a, {'pixel': int(pixel)})
            self.http_client.change_threads(self.container.ip_a, int(cores))
            with access_state:
                core_state = core_state | {self.container.container_id: cores}
                logger.info(core_state)


def reset_core_states(container: DockerInfo, core):
    global core_state
    with access_state:
        core_state = {container.container_id: core}

def reset_core_states_2(container: [DockerInfo], core):
    global core_state
    with access_state:
        core_state = {container[0].container_id: core[0], container[1].container_id: core[1]}


if __name__ == '__main__':
    ps = "http://localhost:9090"
    ScalingAgent(container=DockerInfo("multiscaler-video-processing-a-1", "172.18.0.4", "Alice"), prom_server=ps,
                 thresholds=(600, 25)).start()
    # ScalingAgent(container=DockerInfo("multiscaler-video-processing-b-1", "172.18.0.5", "Bob"), prom_server=ps,
    #              thresholds=(1400, 25)).start()
