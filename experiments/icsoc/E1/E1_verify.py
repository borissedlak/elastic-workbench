import itertools
import logging
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from agent.RASK_Agent import RASK_Agent
import utils
from HttpClient import HttpClient
from agent.agent_utils import export_experience_buffer, delete_file_if_exists
from agent.components.es_registry import ServiceID, ServiceType

ROOT = os.path.dirname(__file__)
plt.rcParams.update({'font.size': 12})

http_client = HttpClient()
logging.getLogger('multiscale').setLevel(logging.INFO)
nn_folder = "./networks"

import pandas as pd

candidate_path = ROOT + "/run_1/candidate_solutions.csv"
evaluation_script = pd.read_csv(candidate_path)

##### Scaling Agent Hyperparameters #######

EVALUATION_FREQUENCY = 10

#### Special Configs ######################

EXPERIMENT_REPETITIONS = 1 # might also be more!
EXPERIMENT_DURATION = len(evaluation_script) * EVALUATION_FREQUENCY

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
        time.sleep(EVALUATION_FREQUENCY / 2)  # Needs a couple of seconds after resetting services (i.e., calling ES)
        delete_file_if_exists(ROOT + "/../../../share/metrics/metrics.csv")
        time.sleep(EVALUATION_FREQUENCY / 2)  # Needs a couple of seconds after resetting services (i.e., calling ES)

        agent.start()
        time.sleep(EXPERIMENT_DURATION)
        agent.terminate_gracefully()
        export_experience_buffer(agent.experience_buffer, ROOT + f"/agent_experience_{agent_suffix}.csv")
        print(f"{agent_suffix} agent finished evaluation round #{rep} after {EXPERIMENT_DURATION * rep} seconds")




def calculate_mean_and_std(df: DataFrame, experiment_repetitions: int):
    slo_fs_index = []

    # Step 2: Reindex each part
    for j in range(1, experiment_repetitions + 1):
        slo_f_run = df[df['rep'] == j]['slo_f']
        slo_fs_index.append(slo_f_run.to_list())

    array = np.array(slo_fs_index)
    mean_over_time = np.mean(array, axis=0)
    std_over_time = np.std(array, axis=0)

    return mean_over_time, std_over_time

# TODO: (1) Currently, only one service is adjusted; this is ok, but ideally we should parallelize this.

if __name__ == '__main__':

    agent_fact_rask = lambda repetition: RASK_Agent(
        prom_server=PROMETHEUS,
        evaluation_cycle=EVALUATION_FREQUENCY,
        services_monitored=[qr_local],
        log_experience=repetition,
        replay_path=candidate_path
    )

    eval_scaling_agent(agent_fact_rask, f"RASK_verify")

    # visualize_data(files, ROOT + "/plots/E1_SLO_F.pdf")
