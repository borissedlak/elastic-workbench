# TODO: 1) This will have to start two agents that greedily consume all resources
import time

import pandas as pd

from DockerClient import DockerInfo
from agent.DQN import DQN, STATE_DIM
from agent.ScalingAgent_v2 import AIFAgent
from slo_config import PW_MAX_CORES

container = DockerInfo("multiscaler-video-processing-a-1", "172.18.0.4", "Alice")
p_s = "http://172.18.0.2:9090"
nn = "./networks"
df = pd.read_csv("LGBN.csv")

max_cores = PW_MAX_CORES
pixel_t, fps_t = 900, 25

dqn = DQN(state_dim=STATE_DIM, action_dim=5, force_restart=True, nn_folder=nn)
dqn.train_dqn_from_env(df=df, suffix=f"5")


def start_agents():
    agent_1 = AIFAgent(container=container, prom_server=p_s, thresholds=(pixel_t, fps_t),
                       dqn=dqn, log=f"S1.1", max_cores=max_cores)
    agent_1.start()
    agent_2 = AIFAgent(container=container, prom_server=p_s, thresholds=(pixel_t, fps_t),
                       dqn=dqn, log=f"S1.2", max_cores=max_cores)
    agent_2.start()
    time.sleep(50)  # May extend when things work

    agent_1.stop()
    agent_2.stop()


# TODO: 2) We will let the main class analyze how the SLO-F can be improved
def calculate_improvement():


# TODO: 3) This is orchestrated and we measure the improvement

if __name__ == '__main__':
    start_agents()
    calculate_improvement()
