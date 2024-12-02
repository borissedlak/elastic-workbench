import csv
import time
from random import randint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from DockerClient import DockerInfo
from HttpClient import HttpClient
from agent.DQN import DQN, STATE_DIM
from agent.ScalingAgent_v2 import ScalingAgent
from slo_config import PW_MAX_CORES, Full_State, calculate_slo_reward

nn = "./networks"
routine_file = "test_routine.csv"
reps = 5
partitions = 5
container = DockerInfo("multiscaler-video-processing-a-1", "172.18.0.4", "Alice")
http_client = HttpClient()
p_s = "http://172.18.0.2:9090"


def train_networks():
    file_path = "LGBN.csv"
    df = pd.read_csv(file_path)

    df_size = len(df)
    end_indices = [166, 262, int(df_size * 3 / 5), int(df_size * 4 / 5), df_size]

    for i, val in enumerate(end_indices):
        partition_df = df.iloc[:val]

        dqn = DQN(state_dim=STATE_DIM, action_dim=5, force_restart=True, nn_folder=nn)
        dqn.train_dqn_from_env(df=partition_df, suffix=f"{i + 1}")

        print(f"{((i + 1) / partitions) * 100}% finished")


def eval_networks():
    with open(routine_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            i, j, pixel, cores, pixel_t, fps_t, max_cores = tuple(map(int, row))
            reset_container_params(pixel, cores)

            dqn = DQN(state_dim=STATE_DIM, action_dim=5, nn_folder=nn, suffix=f"{i}")
            agent = ScalingAgent(container=container, prom_server=p_s, thresholds=(pixel_t, fps_t),
                                 dqn=dqn, log=f"{i}_{j}", max_cores=max_cores)
            agent.start()
            time.sleep(50)
            agent.stop()

            print(f"{((i * reps) / (reps * partitions)) * 100}% finished")


def create_test_routine():
    runs = [["index", "rep", "pixel", "cores", "pixel_t", "fps_t", "max_cores"]]
    for i in range(1, partitions + 1):
        for j in range(1, reps + 1):
            pixel = randint(1, 20) * 100
            max_cores = randint(1, PW_MAX_CORES)
            cores = randint(1, max_cores)
            pixel_t = randint(7, 10) * 100
            fps_t = randint(25, 35)

            runs.append([i, j, pixel, cores, pixel_t, fps_t, max_cores])

    with open(routine_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(runs)


def reset_container_params(pixel, cores):
    http_client.change_config(container.ip_a, {'pixel': int(pixel)})
    http_client.change_threads(container.ip_a, int(cores))


def visualize_data():
    df = pd.read_csv("slo_f.csv")
    del df['timestamp']
    del df['id']

    # TODO: Need to put the ID here, group by the run, and calculate the mean and std
    states = [Full_State(*row) for row in df.itertuples(index=False, name=None)]
    slo_fs = [np.sum(calculate_slo_reward(s.for_tensor())) for s in states]
    plt.plot(slo_fs)
    plt.show()


if __name__ == '__main__':
    # train_networks()
    # create_test_routine()
    # eval_networks()
    visualize_data()
