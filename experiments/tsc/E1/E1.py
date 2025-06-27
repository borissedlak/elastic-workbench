import itertools
import logging
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

import utils
from agent import agent_utils
from agent.RASKGlobalAgent import RASK_Global_Agent
from agent.agent_utils import export_experience_buffer, delete_file_if_exists
from agent.es_registry import ServiceID, ServiceType

ROOT = os.path.dirname(__file__)
plt.rcParams.update({'font.size': 12})

logging.getLogger('multiscale').setLevel(logging.INFO)
nn_folder = "./networks"

######## Experimental Parameters ##########

EXPERIMENT_REPETITIONS = 5
EXPERIMENT_DURATION = 350

##### Scaling Agent Hyperparameters #######

MAX_EXPLORE = [0, 25, 50]
GAUSSIAN_NOISE = [0, 0.05, 0.10]
EVALUATION_FREQUENCY = 5

########## Service Definitions ############

SERVICE_HOST = utils.get_env_param('SERVICE_HOST', "localhost")
REMOTE_VM = utils.get_env_param('REMOTE_VM', "128.131.172.182")
PROMETHEUS = f"http://{SERVICE_HOST}:9090"  # "128.131.172.182"

qr_local = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-1", port="8080")
cv_local = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-1", port="8081")
pc_local = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-1", port="8082")


def eval_scaling_agent(agent_factory, agent_suffix):
    # delete_file_if_exists(ROOT + f"/agent_experience_{agent_type}.csv")

    print(f"Starting experiment for {agent_suffix} agent")

    for rep in range(1, EXPERIMENT_REPETITIONS + 1):
        delete_file_if_exists(ROOT + "/../../../share/metrics/metrics.csv")

        agent = agent_factory(rep)
        agent.reset_services_states()
        time.sleep(EVALUATION_FREQUENCY * 2)  # Needs a couple of seconds after resetting the services (i.e., calling ES)

        agent.start()
        time.sleep(EXPERIMENT_DURATION)
        agent.terminate_gracefully()
        export_experience_buffer(agent.experience_buffer, ROOT + f"/agent_experience_{agent_suffix}.csv")
        print(f"{agent_suffix} agent finished evaluation round #{rep} after {EXPERIMENT_DURATION * rep} seconds")


COLOR_DICT = {"elastic-workbench-qr-detector-1": "red", "elastic-workbench-cv-analyzer-1": "green"}
COLOR_DICT_AGENT = {"DQN": "red", "ASK": "green", "AIF": "blue", "DACI": "grey"}
LINE_STYLE_DICT = {"DQN": "--", "ASK": "-", "AIF": "-.", "DACI": ':'}


def visualize_data(agent_types: list[str], output_file: str):

    df_layout = pd.read_csv(ROOT + f"/agent_experience_{agent_types[0]}.csv")
    x = np.arange(1, len(df_layout.index) / (EXPERIMENT_REPETITIONS * 2) + 1)
    plt.figure(figsize=(6.0, 3.8))

    for agent in agent_types:
        df = pd.read_csv(ROOT + f"/agent_experience_{agent}.csv")

        paired_df = df.groupby(df.index // 2).agg({
            'rep': 'first',
            'timestamp': 'first',
            'slo_f': 'mean'
        })

        s_mean, s_std = calculate_mean_and_std(paired_df)
        lower_bound = np.array(s_mean) - np.array(s_std)
        upper_bound = np.array(s_mean) + np.array(s_std)
        plt.plot(x, s_mean, label=f"{agent}", color=COLOR_DICT_AGENT[agent], linewidth=2,
                 linestyle=LINE_STYLE_DICT[agent])
        plt.fill_between(x, lower_bound, upper_bound, color=COLOR_DICT_AGENT[agent], alpha=0.1)

    plt.xlim(1.0, len(df_layout.index) / (EXPERIMENT_REPETITIONS * 2))
    plt.xticks([1, 10, 20, 30, 40, 50])
    plt.ylim(0.0, 1.0)

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

    # delete_file_if_exists(ROOT + "/../../../share/metrics/metrics.csv")
    agent_utils.stream_remote_metrics_file(REMOTE_VM, EVALUATION_FREQUENCY)

    for max_exploration, noise in itertools.product(MAX_EXPLORE, GAUSSIAN_NOISE):

        agent_fact_rask = lambda repetition: RASK_Global_Agent(
            prom_server=PROMETHEUS,
            services_monitored=[qr_local, cv_local, pc_local],
            evaluation_cycle=EVALUATION_FREQUENCY,
            log_experience=repetition,
            max_explore=max_exploration,
            gaussian_noise=noise
        )

        eval_scaling_agent(agent_fact_rask, f"RASK_{max_exploration}_{noise}")

    # visualize_data(["DQN", "ASK", "AIF", "DACI"], ROOT + "/plots/slo_f.png")
