import itertools
import logging
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils
from HttpClient import HttpClient
from agent import agent_utils
from agent.RASKGlobalAgent import RASK_Global_Agent
from agent.agent_utils import export_experience_buffer, delete_file_if_exists
from agent.components.es_registry import ServiceID, ServiceType
from experiments.tsc.E1.E1 import PC_RPS, QR_RPS, CV_RPS, calculate_mean_and_std
from experiments.tsc.E1.E1 import moving_average
from experiments.tsc.E2.pattern.PatternRPS import PatternRPS, RequestPattern

ROOT = os.path.dirname(__file__)
plt.rcParams.update({'font.size': 12})
METRICS_FILE_PATH = ROOT + "/../../../share/metrics/metrics.csv"

http_client = HttpClient()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multiscale")

######## Experimental Parameters ##########

EXPERIMENT_REPETITIONS = 5
EXPERIMENT_DURATION = 3600  # seconds, so its 1 hour

##### Scaling Agent Hyperparameters #######

MAX_EXPLORE = 0
GAUSSIAN_NOISE = 0
EVALUATION_FREQUENCY = 10
REQUEST_PATTERN = RequestPattern.DIURNAL

########## Service Definitions ############

SERVICE_HOST = utils.get_env_param('SERVICE_HOST', "localhost")
REMOTE_VM = utils.get_env_param('REMOTE_VM', "128.131.172.182")
PROMETHEUS = f"http://{SERVICE_HOST}:9090"  # "128.131.172.182"

qr_local = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-1", port="8080")
cv_local = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-1", port="8081")
pc_local = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-1", port="8082")

MAX_RPS_QR = 100
MAX_RPS_CV = 10


def ingest_metrics_data(source):
    with open(source, "r") as f:
        lines = f.readlines()

    data_lines = lines[1:]
    utils.write_metrics_to_csv(data_lines, pure_string=True)
    print(f"Ingested metrics from {source}")


def eval_scaling_agent(agent_factory, agent_suffix, request_pattern: RequestPattern):
    pattern_rps = PatternRPS()
    print(f"Starting experiment for {agent_suffix} agent")

    http_client.update_service_rps(qr_local, QR_RPS)
    http_client.update_service_rps(cv_local, CV_RPS)
    http_client.update_service_rps(pc_local, PC_RPS)

    experience_file = ROOT + f"/agent_experience_{agent_suffix}_{request_pattern.value}.csv"

    for rep in range(1, EXPERIMENT_REPETITIONS + 1):

        runtime_sec = 0
        delete_file_if_exists(METRICS_FILE_PATH)
        ingest_metrics_data(ROOT + "/../E1/run_6/metrics_20_0.csv")

        agent = agent_factory(rep)
        if agent.update_last_assignment:
            last_assignments = agent_utils.get_last_assignment_from_metrics(METRICS_FILE_PATH)
            agent.set_last_assignments(last_assignments)

        agent.reset_services_states()
        time.sleep(EVALUATION_FREQUENCY / 2)  # Needs a couple of seconds after resetting services (i.e., calling ES)
        agent.start()

        while runtime_sec < EXPERIMENT_DURATION:
            pattern_rps.reconfigure_rps(request_pattern, qr_local, MAX_RPS_QR, runtime_sec)
            pattern_rps.reconfigure_rps(request_pattern, cv_local, MAX_RPS_CV, runtime_sec)
            time.sleep(EVALUATION_FREQUENCY)

            runtime_sec += EVALUATION_FREQUENCY
            export_experience_buffer(agent.experience_buffer, experience_file)
            agent.experience_buffer.clear()

        agent.terminate_gracefully()
        print(
            f"Run #{rep}| {agent_suffix} agent finished {request_pattern.value} evaluation after {runtime_sec} seconds")


def visualize_data(agent_types: list[str], output_file: str):
    x = np.arange(1, (EXPERIMENT_DURATION / EVALUATION_FREQUENCY) + 2)
    # plt.figure(figsize=(6.0, 3.8))
    plt.figure(figsize=(18.0, 4.8))

    for agent in agent_types:
        df = pd.read_csv(ROOT + f"/{agent}")
        paired_df = df.groupby(df.index // 3).agg({
            'rep': 'first',
            'timestamp': 'first',
            'slo_f': 'mean'
        })
        paired_df['slo_f'] = moving_average(paired_df['slo_f'], window_size=2)
        s_mean, s_std = calculate_mean_and_std(paired_df, EXPERIMENT_REPETITIONS)
        lower_bound = np.array(s_mean) - np.array(s_std)
        upper_bound = np.array(s_mean) + np.array(s_std)
        plt.plot(x, s_mean, label=f"{agent}", linewidth=2)
        plt.fill_between(x, lower_bound, upper_bound, alpha=0.1)

    plt.xlim(1, (EXPERIMENT_DURATION / EVALUATION_FREQUENCY) + 2)
    # plt.ylim(0.5, 0.95)

    plt.xlabel('Scaling Agent Iterations')
    plt.ylabel('Global SLO Fulfillment')
    plt.legend()
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    # agent_utils.stream_remote_metrics_file(REMOTE_VM, EVALUATION_FREQUENCY)

    for es_file_ext, update_last_ass in itertools.product(['_lim_1', '_lim_2', ''], [True, False]):
        agent_fact_rask = lambda repetition: RASK_Global_Agent(
            prom_server=PROMETHEUS,
            services_monitored=[qr_local, cv_local, pc_local],
            evaluation_cycle=EVALUATION_FREQUENCY,
            log_experience=repetition,
            max_explore=MAX_EXPLORE,
            gaussian_noise=GAUSSIAN_NOISE,
            es_registry_path=ROOT + f"/../../../config/es_registry{es_file_ext}.json",
            update_last_assignment=update_last_ass
        )

        eval_scaling_agent(agent_fact_rask, f"RASK{es_file_ext}_{update_last_ass}", REQUEST_PATTERN)

    # The run_2 are also nice ...
    # visualize_data(["run_2/agent_experience_RASK_0_bursty.csv","run_3/agent_experience_k8_0_bursty.csv","agent_experience_dqn_0_bursty.csv"], ROOT + "/plots/slo_f_bursty.eps")
    # visualize_data(["agent_experience_RASK_False_bursty.csv", "agent_experience_RASK_lim_1_False_bursty.csv","agent_experience_RASK_lim_2_False_bursty.csv"], ROOT + "/plots/slo_f_diurnal.eps")
    # "agent_experience_RASK_True_bursty.csv","agent_experience_RASK_lim_1_True_bursty.csv","agent_experience_RASK_lim_2_True_bursty.csv",
