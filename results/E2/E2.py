import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from DockerClient import DockerInfo
from HttpClient import HttpClient
from agent.obsolete.DQN import DQN, STATE_DIM
from agent.obsolete.Global_Service_Optimizer import Global_Service_Optimizer
from agent.ScalingAgent_v2 import ScalingAgent, reset_core_states_2
from agent.slo_config import Full_State, calculate_slo_reward

plt.rcParams.update({'font.size': 12})

container_1 = DockerInfo("multiscaler-video-processing-a-1", "172.18.0.4", "Alice")
container_2 = DockerInfo("multiscaler-video-processing-b-1", "172.18.0.5", "Bob")
p_s = "http://172.18.0.2:9090"
http_client = HttpClient()

nn = "./networks"
changes = 10

max_cores = 8
starting_pixel, starting_cores = 1400, 2

dqn = DQN(state_dim=STATE_DIM, action_dim=5, nn_folder=nn, suffix="5")
agent_1 = ScalingAgent(container=container_1, prom_server=p_s, thresholds=(1300, 30),
                       dqn=dqn, log=("Alice", None), max_cores=max_cores)
agent_2 = ScalingAgent(container=container_2, prom_server=p_s, thresholds=(1300, 10),
                       dqn=dqn, log=("Bob", None), max_cores=max_cores)

glo = Global_Service_Optimizer(agents=[agent_1, agent_2])


def reset_container_params(c, pixel, cores):
    http_client.change_config(c.ip_a, {'pixel': int(pixel)})
    http_client.change_threads(c.ip_a, int(cores))


def start_greedy_agents():
    reset_container_params(container_1, starting_pixel, starting_cores)
    reset_container_params(container_2, starting_pixel, starting_cores)
    reset_core_states_2([container_1, container_2], [starting_cores, starting_cores])
    time.sleep(3)

    agent_1.start()
    agent_2.start()
    time.sleep(10)

    while agent_1.has_free_cores():
        print("Still free cores available, waiting...")
        time.sleep(5)

    agent_1.set_idle(True)
    agent_2.set_idle(True)


def improve_global_slof():
    for i in range(0, changes):
        if not agent_1.has_free_cores():
            estimates = glo.estimate_swapping()
            glo.swap_core(estimates)

        print(f"Finish iteration {i + 1} / {changes}")
        time.sleep(5)

    agent_1.stop()
    agent_2.stop()


def visualize_data():
    df = pd.read_csv("./slo_f.csv")
    del df['timestamp']

    alice = df[df['index'] == "Alice"]
    bob = df[df['index'] == "Bob"]

    states_alice = [Full_State(*row[2:]) for row in alice.itertuples(index=False, name=None)]
    states_bob = [Full_State(*row[2:]) for row in bob.itertuples(index=False, name=None)]

    slo_f_alice = [np.sum(calculate_slo_reward(s.for_tensor())) for s in states_alice]
    slo_f_bob = [np.sum(calculate_slo_reward(s.for_tensor())) for s in states_bob]

    x = np.arange(len(slo_f_alice))

    plt.figure(figsize=(6.0, 2.8))
    # plt.plot(x, m_base, label='Baseline', color='red', linewidth=1)
    plt.plot(x, slo_f_alice, label='Alice: SLO Fulfillment', color='red', linewidth=2)
    plt.plot(x, slo_f_bob, label='Bob: SLO Fulfillment', color='blue', linewidth=2)
    # plt.vlines([10, 20, 30, 40], ymin=1.25, ymax=2.75, label='Adjust Thresholds', linestyles="--")

    plt.xlim(-0.1, 12.1)
    plt.ylim(1.6, 2.4)

    plt.vlines([2, 3, 5], ymin=1.25, ymax=2.75, label='Alice +1; Bob -1', color='red', alpha=0.5, linestyles="--")
    plt.vlines([7], ymin=1.25, ymax=2.75, label='Alice -1; Bob +1', color='blue', alpha=0.5, linestyles="--")

    plt.xlabel('Scaling Agent Iterations (12 cycles = 60 seconds)')
    plt.ylabel('SLO Fulfillment')
    plt.legend()
    plt.savefig("./global_optimizer.png", dpi=600, bbox_inches="tight", format="png")
    plt.show()


if __name__ == '__main__':
    # start_greedy_agents()
    # improve_global_slof()
    visualize_data()
