import logging
import threading
import time

import utils
from DockerClient import DockerInfo
from agent.ScalingAgent_v2 import ScalingAgent
from agent.agent_utils import log_agent_experience
from agent.obsolete.slo_config import calculate_slo_reward, PW_MAX_CORES, Full_State

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")
# MAX_CORES = utils.get_env_param('MAX_CORES', 10)

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)

core_state = {}
access_state = threading.Lock()


class BaseAgent(ScalingAgent):
    def __init__(self, container: DockerInfo, prom_server, thresholds, log=None, max_cores=PW_MAX_CORES):
        super().__init__(container, prom_server, thresholds, None, log, max_cores)

    def run(self):
        global core_state

        initial_state = self.get_state_PW()
        with access_state:
            core_state = core_state | {self.container.container_id: initial_state.cores}
            logger.info(core_state)

        while self._running:
            state_pw = self.get_state_PW()
            logger.debug(f"Current state before change is {state_pw}")
            logger.debug(f"Current SLO-F before change is {calculate_slo_reward(state_pw.for_tensor())}")
            log_agent_experience(state_pw, self.log) if self.log else None

            action_pw = self.choose_action(state_pw)
            self.act_on_env(action_pw, state_pw)

            time.sleep(5)

    def choose_action(self, state: Full_State):

        rate = state.fps / state.fps_thresh

        if rate < 1.0 and state.free_cores > 0:
            return 4
        elif rate > 1.1 and state.cores > 0:
            return 3

        return 0


if __name__ == '__main__':
    ps = "http://172.18.0.2:9090"
    ScalingAgent(container=DockerInfo("multiscaler-video-processing-a-1", "172.18.0.4", "Alice"), prom_server=ps,
                 thresholds=(1400, 25)).start()
    ScalingAgent(container=DockerInfo("multiscaler-video-processing-b-1", "172.18.0.5", "Bob"), prom_server=ps,
                 thresholds=(1400, 25)).start()
