import csv
import time
from random import randint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from DockerClient import DockerInfo
from HttpClient import HttpClient
from agent.BaseAgent import BaseAgent
from agent.DQN import DQN, STATE_DIM
from agent.ScalingAgent_v2 import ScalingAgent, reset_core_states
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
            reset_container_params(container, pixel, cores)
            reset_core_states(container, cores)
            time.sleep(1)

            dqn = DQN(state_dim=STATE_DIM, action_dim=5, nn_folder=nn, suffix=f"{i}")
            agent = ScalingAgent(container=container, prom_server=p_s, thresholds=(pixel_t, fps_t),
                                 dqn=dqn, log=(i, j), max_cores=max_cores)
            agent.start()
            time.sleep(50)
            agent.stop()

            print(f"{(((i * partitions) + j) / (reps * partitions)) * 100}% finished")


def eval_baseline():
    with open(routine_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            i, j, _, cores, pixel_t, fps_t, max_cores = tuple(map(int, row))
            reset_container_params(container, pixel_t, cores)
            reset_core_states(container, cores)
            time.sleep(1)

            agent = BaseAgent(container=container, prom_server=p_s, thresholds=(pixel_t, fps_t),
                              log=(i, j), max_cores=max_cores)
            agent.start()
            time.sleep(50)
            agent.stop()

            print(f"{(((i * partitions) + j) / (reps * partitions)) * 100}% finished")


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

    print("Created new test routine")


def reset_container_params(c, pixel, cores):
    http_client.change_config(c.ip_a, {'pixel': int(pixel)})
    http_client.change_threads(c.ip_a, int(cores))


def visualize_data(slof_file, output_file):
    df = pd.read_csv(slof_file)
    del df['timestamp']

    means = []
    stds = []

    for i in range(1, partitions + 1):
        partition_df = df[df['index'] == i]

        parts = np.array_split(partition_df, reps)
        slo_fs_index = []

        # Step 2: Reindex each part
        for j in range(len(parts)):
            # parts[j] = parts[j].reset_index(drop=True)
            states = [Full_State(*row[2:]) for row in parts[j].itertuples(index=False, name=None)]
            slo_f_run = [np.sum(calculate_slo_reward(s.for_tensor())) for s in states]
            slo_fs_index.append(slo_f_run)

        array = np.array(slo_fs_index)
        mean_per_field = np.mean(array, axis=0)
        std_per_field = np.std(array, axis=0)

        means.extend(mean_per_field)
        stds.extend(std_per_field)

    x = np.arange(len(means))
    lower_bound = np.array(means) - np.array(stds)
    upper_bound = np.array(means) + np.array(stds)

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, means, label='Mean SLO Fulfillment', color='blue', linewidth=2)
    plt.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.2, label='Standard Deviation')
    plt.vlines([10, 20, 30, 40], ymin=1.25, ymax=2.75, label='Adjust Thresholds', linestyles="--")

    plt.xlim(-0.1, 49.1)
    plt.ylim(1.4, 2.6)

    plt.xlabel('Scaling Agent Iterations (50 cycles = 250 seconds)')
    plt.ylabel('SLO Fulfillment')
    # plt.title('Mean SLO Fulfillment')
    plt.legend()
    plt.savefig(output_file, dpi=600, bbox_inches="tight", format="png")
    plt.show()


if __name__ == '__main__':
    # train_networks()
    # create_test_routine()
    eval_networks()
    eval_baseline()
    # visualize_data("slo_f.csv", "./plots/.png")
    # visualize_data("slo_f_b.csv", "./plots/baseline_agent.png")
