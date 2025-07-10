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

qr_local_1 = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-1", port="8080")
qr_local_2 = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-2", port="8083")
qr_local_3 = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-3", port="8086")
qr_local_4 = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-4", port="8089")
cv_local_1 = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-1", port="8081")
cv_local_2 = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-2", port="8084")
cv_local_3 = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-3", port="8087")
cv_local_4 = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-4", port="8090")
pc_local_1 = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-1", port="8082")
pc_local_2 = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-2", port="8085")
pc_local_3 = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-3", port="8088")
pc_local_4 = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-4", port="8091")

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
    service_replication = int(len(agent_factory(0).services_monitored) / 3)
    print(f"Starting experiment for {agent_suffix} agent with replication {service_replication}")

    http_client.update_service_rps(qr_local_1, QR_RPS)
    http_client.update_service_rps(cv_local_1, CV_RPS)
    http_client.update_service_rps(pc_local_1, PC_RPS) # RESET

    if service_replication >= 2:
        http_client.update_service_rps(qr_local_2, QR_RPS)
        http_client.update_service_rps(cv_local_2, CV_RPS)
        http_client.update_service_rps(pc_local_2, PC_RPS)
    if service_replication >= 3:
        http_client.update_service_rps(qr_local_3, QR_RPS)
        http_client.update_service_rps(cv_local_3, CV_RPS)
        http_client.update_service_rps(pc_local_3, PC_RPS)

    experience_file = ROOT + f"/agent_experience_{agent_suffix}_{request_pattern.value}.csv"

    for rep in range(1, EXPERIMENT_REPETITIONS + 1):

        runtime_sec = 0
        delete_file_if_exists(METRICS_FILE_PATH)
        ingest_metrics_data(ROOT + "/../E1/run_6/metrics_20_0.csv")

        agent = agent_factory(rep)

        agent.reset_services_states()
        time.sleep(EVALUATION_FREQUENCY / 2)  # Needs a couple of seconds after resetting services (i.e., calling ES)
        agent.start()
        agent.last_assignments = None

        while runtime_sec < EXPERIMENT_DURATION:
            pattern_rps.reconfigure_rps(request_pattern, qr_local_1, MAX_RPS_QR, runtime_sec)
            pattern_rps.reconfigure_rps(request_pattern, cv_local_1, MAX_RPS_CV, runtime_sec)

            if service_replication >= 2:
                pattern_rps.reconfigure_rps(request_pattern, qr_local_2, MAX_RPS_QR, runtime_sec)
                pattern_rps.reconfigure_rps(request_pattern, cv_local_2, MAX_RPS_CV, runtime_sec)
            if service_replication >= 3:
                pattern_rps.reconfigure_rps(request_pattern, qr_local_3, MAX_RPS_QR, runtime_sec)
                pattern_rps.reconfigure_rps(request_pattern, cv_local_3, MAX_RPS_CV, runtime_sec)

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

    services_3 = [qr_local_1, cv_local_1, pc_local_1]
    services_6 = services_3 + [qr_local_2, cv_local_2, pc_local_2]
    services_9 = services_6 + [qr_local_3, cv_local_3, pc_local_3]
    # services_12 = services_9 + [qr_local_4, cv_local_4, pc_local_4]

    for agent_list in [services_9]:
        agent_fact_rask = lambda repetition: RASK_Global_Agent(
            prom_server=PROMETHEUS,
            services_monitored=agent_list,
            evaluation_cycle=EVALUATION_FREQUENCY,
            log_experience=repetition,
            max_explore=MAX_EXPLORE,
            gaussian_noise=GAUSSIAN_NOISE,
            update_last_assignment=True
        )

        eval_scaling_agent(agent_fact_rask, f"RASK_{len(agent_list)}", REQUEST_PATTERN)

    # The run_2 are also nice ...
    # visualize_data(["run_2/agent_experience_RASK_0_bursty.csv","run_3/agent_experience_k8_0_bursty.csv","agent_experience_dqn_0_bursty.csv"], ROOT + "/plots/slo_f_bursty.eps")
    # visualize_data(["agent_experience_RASK_False_bursty.csv", "agent_experience_RASK_lim_1_False_bursty.csv","agent_experience_RASK_lim_2_False_bursty.csv"], ROOT + "/plots/slo_f_diurnal.eps")
    # "agent_experience_RASK_True_bursty.csv","agent_experience_RASK_lim_1_True_bursty.csv","agent_experience_RASK_lim_2_True_bursty.csv",
