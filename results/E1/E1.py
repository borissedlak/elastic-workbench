import time
from random import randint

import pandas as pd

from DockerClient import DockerInfo
from HttpClient import HttpClient
from agent.DQN import DQN, STATE_DIM
from agent.ScalingAgent_v2 import AIFAgent
from slo_config import PW_MAX_CORES

nn = "./networks"
reps = 1  # TODO: At most 5 I think
container = DockerInfo("multiscaler-video-processing-a-1", "172.18.0.4", "Alice")
http_client = HttpClient()
p_s = "http://172.18.0.2:9090"


def train_networks():
    file_path = "LGBN.csv"
    df = pd.read_csv(file_path)

    for i in range(1, 11):
        partition_size = int(len(df) * (i / 10))
        partition_df = df.iloc[:partition_size]

        # for j in range(1, reps + 1):
        dqn = DQN(state_dim=STATE_DIM, action_dim=5, force_restart=True, nn_folder=nn)
        dqn.train_dqn_from_env(df=partition_df, suffix=f"{i}")

        print(f"{i * 10}% finished")


def eval_networks():
    for i in range(1, 11):
        for j in range(1, reps + 1):
            reset_agent()
            pixel_t = randint(6, 10) * 100
            fps_t = randint(15, 45)

            dqn = DQN(state_dim=STATE_DIM, action_dim=5, nn_folder=nn, suffix=f"{i}")
            agent = AIFAgent(container=container, prom_server=p_s, thresholds=(pixel_t, fps_t), dqn=dqn, log=f"{i}_{j}")
            agent.start()
            time.sleep(50)
            agent.stop()

        print(f"{i * 10}% finished")


def reset_agent():
    pixel = randint(1, 20) * 100
    cores = randint(1, PW_MAX_CORES)
    http_client.change_config(container.ip_a, {'pixel': int(pixel)})
    http_client.change_threads(container.ip_a, int(cores))


if __name__ == '__main__':
    # train_networks()
    eval_networks()
