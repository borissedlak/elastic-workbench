import time
from random import randint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from DockerClient import DockerInfo
from HttpClient import HttpClient
from agent.DQN import DQN, STATE_DIM
from agent.ScalingAgent_v2 import AIFAgent
from slo_config import PW_MAX_CORES, Full_State, calculate_slo_reward

nn = "./networks"
reps = 10
partitions = 5
container = DockerInfo("multiscaler-video-processing-a-1", "172.18.0.4", "Alice")
http_client = HttpClient()
p_s = "http://172.18.0.2:9090"


def train_networks():
    file_path = "LGBN.csv"
    df = pd.read_csv(file_path)

    for i in range(1, partitions + 1):
        partition_size = int(len(df) * (i / partitions))
        partition_df = df.iloc[:partition_size]

        dqn = DQN(state_dim=STATE_DIM, action_dim=5, force_restart=True, nn_folder=nn)
        dqn.train_dqn_from_env(df=partition_df, suffix=f"{i}")

        print(f"{(i / partitions) * 100}% finished")


def eval_networks():
    for i in range(1, partitions + 1):
        pixel_t = randint(7, 10) * 100
        fps_t = randint(25, 35)

        for j in range(1, reps + 1):
            reset_container_params()

            dqn = DQN(state_dim=STATE_DIM, action_dim=5, nn_folder=nn, suffix=f"{i}")
            agent = AIFAgent(container=container, prom_server=p_s, thresholds=(pixel_t, fps_t), dqn=dqn, log=f"{i}_{j}")
            agent.start()
            time.sleep(60)
            agent.stop()

        print(f"{(i / partitions) * 100}% finished")


def visualize_data():
    df = pd.read_csv("slo_f.csv")
    del df['timestamp']
    del df['id']

    states = [Full_State(*row) for row in df.itertuples(index=False, name=None)]
    slo_fs = [np.sum(calculate_slo_reward(state.for_tensor())) for state in states]
    plt.plot(slo_fs)
    plt.show()


def reset_container_params():
    pixel = randint(1, 20) * 100
    cores = randint(1, PW_MAX_CORES)
    http_client.change_config(container.ip_a, {'pixel': int(pixel)})
    http_client.change_threads(container.ip_a, int(cores))


if __name__ == '__main__':
    # train_networks()
    # eval_networks()
    visualize_data()
