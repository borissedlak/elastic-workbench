import itertools
import logging
import os
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

import utils
from HttpClient import HttpClient
from agent import agent_utils
from agent.RASKGlobalAgent import RASK_Global_Agent
from agent.agent_utils import export_experience_buffer, delete_file_if_exists
from agent.es_registry import ServiceID, ServiceType

ROOT = os.path.dirname(__file__)
plt.rcParams.update({'font.size': 12})

http_client = HttpClient()
logging.getLogger('multiscale').setLevel(logging.INFO)
nn_folder = "./networks"

######## Experimental Parameters ##########

EXPERIMENT_REPETITIONS = 5
EXPERIMENT_DURATION = 600  # seconds, so 600 = 10min

##### Scaling Agent Hyperparameters #######

MAX_EXPLORE = [0, 10, 20]  # [0, 10, 20]
GAUSSIAN_NOISE = [0, 0.10]  # [0, 0.05, 0.10]
EVALUATION_FREQUENCY = 10

########## Service Definitions ############

SERVICE_HOST = utils.get_env_param('SERVICE_HOST', "localhost")
REMOTE_VM = utils.get_env_param('REMOTE_VM', "128.131.172.182")
PROMETHEUS = f"http://{SERVICE_HOST}:9090"  # "128.131.172.182"

qr_local = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-1", port="8080")
cv_local = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-1", port="8081")
pc_local = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-1", port="8082")

QR_RPS = 80
CV_RPS = 5
PC_RPS = 50


def eval_scaling_agent(agent_factory, agent_suffix):
    print(f"Starting experiment for {agent_suffix} agent")

    http_client.update_service_rps(qr_local, QR_RPS)
    http_client.update_service_rps(cv_local, CV_RPS)
    http_client.update_service_rps(pc_local, PC_RPS)

    for rep in range(1, EXPERIMENT_REPETITIONS + 1):
        agent = agent_factory(rep)
        agent.reset_services_states()
        time.sleep(EVALUATION_FREQUENCY)  # Needs a couple of seconds after resetting services (i.e., calling ES)
        delete_file_if_exists(ROOT + "/../../../share/metrics/metrics.csv")
        time.sleep(EVALUATION_FREQUENCY)  # Needs a couple of seconds after resetting services (i.e., calling ES)

        agent.start()
        time.sleep(EXPERIMENT_DURATION)
        agent.terminate_gracefully()
        export_experience_buffer(agent.experience_buffer, ROOT + f"/agent_experience_{agent_suffix}.csv")
        print(f"{agent_suffix} agent finished evaluation round #{rep} after {EXPERIMENT_DURATION * rep} seconds")


COLOR_DICT = {"elastic-workbench-qr-detector-1": "red", "elastic-workbench-cv-analyzer-1": "green"}
COLOR_DICT_AGENT = {"DQN": "red", "ASK": "green", "AIF": "blue", "DACI": "grey"}
LINE_STYLE_DICT = {"DQN": "--", "ASK": "-", "AIF": "-.", "DACI": ':'}

FILE_COLOR_MAP = {
    'agent_experience_RASK_0_0.1.csv': '#ff9999',  # light red
    'agent_experience_RASK_0_0.05.csv': '#ff6666',  # medium red
    'agent_experience_RASK_0_0.csv': '#cc0000',  # dark red

    'agent_experience_RASK_10_0.1.csv': '#99ff99',  # light green
    'agent_experience_RASK_10_0.05.csv': '#66cc66',  # medium green
    'agent_experience_RASK_10_0.csv': '#009900',  # dark green

    'agent_experience_RASK_20_0.1.csv': '#9999ff',  # light blue
    'agent_experience_RASK_20_0.05.csv': '#6666cc',  # medium blue
    'agent_experience_RASK_20_0.csv': '#0000cc',  # dark blue
}

LINE_STYLE_DICT = {
    'agent_experience_RASK_0_0.1.csv': "--",
    'agent_experience_RASK_0_0.csv': "--",

    'agent_experience_RASK_10_0.1.csv': "-",
    'agent_experience_RASK_10_0.csv': "-",

    'agent_experience_RASK_20_0.1.csv': "-.",
    'agent_experience_RASK_20_0.csv': "-.",
}

def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def visualize_data(rask_configs: list[Tuple[str,str]], output_file: str):
    x = np.arange(1, (EXPERIMENT_DURATION / EVALUATION_FREQUENCY) + 1)
    # plt.figure(figsize=(6.0, 3.8))
    plt.figure(figsize=(18.0, 4.8))

    for file, name in rask_configs:
        df = pd.read_csv(ROOT + f"/run_6/{file}")

        paired_df = df.groupby(df.index // 3).agg({
            'rep': 'first',
            'timestamp': 'first',
            'slo_f': 'mean'
        })

        paired_df['slo_f'] = moving_average(paired_df['slo_f'], window_size=2)
        s_mean, s_std = calculate_mean_and_std(paired_df)
        lower_bound = np.array(s_mean) - np.array(s_std)
        upper_bound = np.array(s_mean) + np.array(s_std)
        plt.plot(x, s_mean, color=FILE_COLOR_MAP[file], label=f"{name}", linewidth=2,
                 linestyle=LINE_STYLE_DICT[file])
        plt.fill_between(x, lower_bound, upper_bound, color=FILE_COLOR_MAP[file], alpha=0.1)

    plt.xlim(1, (EXPERIMENT_DURATION / EVALUATION_FREQUENCY))
    plt.xticks([1, 10, 20, 30, 40, 50, 60])
    plt.ylim(0.5, 1.0)

    plt.xlabel('Scaling Agent Iterations')
    plt.ylabel('Global SLO Fulfillment')
    plt.legend()
    plt.savefig(output_file, dpi=600, bbox_inches="tight", format="png")
    plt.show()


def calculate_mean_and_std(df: DataFrame):
    # del df['timestamp']

    slo_fs_index = []

    # Step 2: Reindex each part
    for j in range(1, EXPERIMENT_REPETITIONS + 1):
        slo_f_run = df[df['rep'] == j]['slo_f']
        slo_fs_index.append(slo_f_run.to_list())

    array = np.array(slo_fs_index)
    mean_over_time = np.mean(array, axis=0)
    std_over_time = np.std(array, axis=0)

    return mean_over_time, std_over_time


if __name__ == '__main__':
    files = [
        ('agent_experience_RASK_0_0.csv', 'RASK, no exploration, no noise'),
        ('agent_experience_RASK_0_0.1.csv', 'RASK, no exploration, 10% noise'),
        # ('agent_experience_RASK_0_0.05.csv', 'RASK, no exploration, 5% noise'),
        ('agent_experience_RASK_10_0.csv', 'RASK, exploration=0, no noise'),
        ('agent_experience_RASK_10_0.1.csv', 'RASK, exploration=10, 10% noise'),
        # ('agent_experience_RASK_10_0.05.csv', 'RASK, 10 exploration steps, 5% noise'),
        ('agent_experience_RASK_20_0.csv', 'RASK, exploration=20, no noise'),
        ('agent_experience_RASK_20_0.1.csv', 'RASK, exploration=20, 10% noise'),
        # ('agent_experience_RASK_20_0.05.csv', 'RASK, 20 exploration steps, 5% noise'),
    ]

    # agent_utils.stream_remote_metrics_file(REMOTE_VM, EVALUATION_FREQUENCY)

    # for max_exploration, noise in itertools.product(MAX_EXPLORE, GAUSSIAN_NOISE):
    #     agent_fact_rask = lambda repetition: RASK_Global_Agent(
    #         prom_server=PROMETHEUS,
    #         services_monitored=[qr_local, cv_local, pc_local],
    #         evaluation_cycle=EVALUATION_FREQUENCY,
    #         log_experience=repetition,
    #         max_explore=max_exploration,
    #         gaussian_noise=noise
    #     )
    #
    # eval_scaling_agent(agent_fact_rask, f"RASK_{max_exploration}_{noise}")

    visualize_data(files, ROOT + "/plots/slo_f_run6.png")
