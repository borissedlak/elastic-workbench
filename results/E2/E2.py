import time

import numpy as np
from matplotlib import pyplot as plt

from DockerClient import DockerInfo
from HttpClient import HttpClient
from agent.DQN import DQN, STATE_DIM
from agent.Global_Service_Optimizer import Global_Service_Optimizer
from agent.ScalingAgent_v2 import ScalingAgent, reset_core_states_2

container_1 = DockerInfo("multiscaler-video-processing-a-1", "172.18.0.4", "Alice")
container_2 = DockerInfo("multiscaler-video-processing-b-1", "172.18.0.5", "Bob")
p_s = "http://172.18.0.2:9090"
http_client = HttpClient()

nn = "./networks"
changes = 10

max_cores = 8
starting_pixel, starting_cores = 1600, 2

dqn = DQN(state_dim=STATE_DIM, action_dim=5, nn_folder=nn, suffix="5")
agent_1 = ScalingAgent(container=container_1, prom_server=p_s, thresholds=(1500, 30),
                       dqn=dqn, log=f"S1.1", max_cores=max_cores)
agent_2 = ScalingAgent(container=container_2, prom_server=p_s, thresholds=(1500, 20),
                       dqn=dqn, log=f"S1.2", max_cores=max_cores)

glo = Global_Service_Optimizer(agents=[agent_1, agent_2])


def reset_container_params(c, pixel, cores):
    http_client.change_config(c.ip_a, {'pixel': int(pixel)})
    http_client.change_threads(c.ip_a, int(cores))


def start_greedy_agents():
    reset_container_params(container_1, starting_pixel, starting_cores)
    reset_container_params(container_2, starting_pixel, starting_cores)
    reset_core_states_2([container_1, container_2], [starting_cores, starting_cores])
    time.sleep(1)

    agent_1.start()
    agent_2.start()
    time.sleep(10)

    while agent_1.has_free_cores():
        print("Still free cores available, waiting...")
        time.sleep(5)

    # agent_1.stop()
    # agent_2.stop()


def improve_global_slof():
    for i in range(0, changes):
        if not agent_1.has_free_cores():
            estimates = glo.estimate_swapping()
            glo.swap_core(estimates)

        time.sleep(5)


def visualize_data():
    x = np.arange(len(m_meth))
    lower_bound = np.array(m_meth) - np.array(std_meth)
    upper_bound = np.array(m_meth) + np.array(std_meth)

    plt.figure(figsize=(6.0, 3.8))
    # plt.plot(x, m_base, label='Baseline', color='red', linewidth=1)
    plt.plot(x, m_meth, label='Mean SLO Fulfillment', color='blue', linewidth=2)
    plt.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.2, label='Standard Deviation')
    plt.plot(x, m_base, label='Baseline Scaler', color='red', linewidth=1)
    plt.vlines([10, 20, 30, 40], ymin=1.25, ymax=2.75, label='Adjust Thresholds', linestyles="--")

    plt.xlim(-0.1, 49.1)
    plt.ylim(1.4, 2.4)

    plt.xlabel('Scaling Agent Iterations (50 cycles = 250 seconds)')
    plt.ylabel('SLO Fulfillment')
    # plt.title('Mean SLO Fulfillment')
    plt.legend()
    plt.savefig("./abcd.png", dpi=600, bbox_inches="tight", format="png")
    plt.show()


if __name__ == '__main__':
    start_greedy_agents()
    improve_global_slof()
    # visualize_data()
