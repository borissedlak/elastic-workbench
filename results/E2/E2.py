# TODO: 1) This will have to start two agents that greedily consume all resources
import time

import pandas as pd

from DockerClient import DockerInfo
from agent.DQN import DQN, STATE_DIM
from agent.Global_Service_Optimizer import Global_Service_Optimizer
from agent.ScalingAgent_v2 import ScalingAgent
from slo_config import PW_MAX_CORES

container = DockerInfo("multiscaler-video-processing-a-1", "172.18.0.4", "Alice")
p_s = "http://172.18.0.2:9090"
nn = "./networks"
df = pd.read_csv("LGBN.csv")
reps = 5
changes = 10

max_cores = PW_MAX_CORES
pixel_t, fps_t = 900, 25

dqn = DQN(state_dim=STATE_DIM, action_dim=5, force_restart=True, nn_folder=nn)
dqn.train_dqn_from_env(df=df, suffix=f"5")

agent_1 = ScalingAgent(container=container, prom_server=p_s, thresholds=(pixel_t, fps_t),
                       dqn=dqn, log=f"S1.1", max_cores=max_cores)
agent_2 = ScalingAgent(container=container, prom_server=p_s, thresholds=(pixel_t, fps_t),
                       dqn=dqn, log=f"S1.2", max_cores=max_cores)

glo = Global_Service_Optimizer(agents=[agent_1, agent_2])


def start_greedy_agents():
    agent_1.start()
    agent_2.start()
    time.sleep(15)

    # agent_1.stop()
    # agent_2.stop()


# TODO: 2) We will let the main class analyze how the SLO-F can be improved
# TODO: 3) This is orchestrated and we measure the improvement
def improve_global_slof():
    for i in range(1, changes + 1):
        glo.estimate_swapping()
        glo.swap_core()

        time.sleep(10)


if __name__ == '__main__':
    start_greedy_agents()
    improve_global_slof()
