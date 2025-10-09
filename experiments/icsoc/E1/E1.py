import itertools
import logging
import os
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
from agent.components.es_registry import ServiceID, ServiceType
from experiments.tsc.E2.E2 import ingest_metrics_data, MAX_RPS_QR, MAX_RPS_CV
from experiments.tsc.E2.pattern.PatternRPS import PatternRPS, RequestPattern

ROOT = os.path.dirname(__file__)
plt.rcParams.update({'font.size': 12})

http_client = HttpClient()
logging.getLogger('multiscale').setLevel(logging.INFO)
nn_folder = "./networks"

######## Experimental Parameters ##########

DURATION_EXPLORE = 5 * 60  # seconds, so 300 = 5min
DURATION_OPERATE = 10 * 60  # seconds, so 600 = 10min

##### Scaling Agent Hyperparameters #######

MAX_EXPLORE = 20
GAUSSIAN_NOISE = 0.05
EVALUATION_FREQUENCY = 10

########## Service Definitions ############

SERVICE_HOST = utils.get_env_param('SERVICE_HOST', "localhost")
REMOTE_VM = utils.get_env_param('REMOTE_VM', "128.131.172.182")
PROMETHEUS = f"http://{SERVICE_HOST}:9090"  # "128.131.172.182"

qr_local = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-1", port="8080")
cv_local = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-1", port="8081")
pc_local = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-1", port="8082")

QR_RPS_DEFAULT = 80
CV_RPS_DEFAULT = 5
PC_RPS_DEFAULT = 50


def reset_services_default_rps():
    http_client.update_service_rps(qr_local, QR_RPS_DEFAULT)
    http_client.update_service_rps(cv_local, CV_RPS_DEFAULT)
    http_client.update_service_rps(pc_local, PC_RPS_DEFAULT)


def train_scaling_agent(rask_agent, agent_suffix):
    print(f"Starting exploration for {agent_suffix} agent")

    reset_services_default_rps()
    rask_agent.reset_services_states()
    delete_file_if_exists(ROOT + "/../../../share/metrics/metrics.csv")
    delete_file_if_exists(ROOT + f"/agent_experience_{agent_suffix}.csv")
    time.sleep(EVALUATION_FREQUENCY / 2)  # Needs a couple of seconds after resetting services

    rask_agent.start()
    time.sleep(DURATION_EXPLORE)
    rask_agent.terminate_gracefully()
    export_experience_buffer(rask_agent.experience_buffer, ROOT + f"/agent_experience_{agent_suffix}.csv")
    print(f"{agent_suffix} agent finished evaluation after {DURATION_EXPLORE} seconds")


def operate_scaling_agent(rask_agent, agent_suffix, request_pattern: RequestPattern):
    print(f"Starting operation for {agent_suffix} agent")
    pattern_rps = PatternRPS()

    reset_services_default_rps()
    experience_file = ROOT + f"/agent_experience_{agent_suffix}_{request_pattern.value}.csv"
    delete_file_if_exists(ROOT + "/../../../share/metrics/metrics.csv")
    ingest_metrics_data(ROOT + "/../E1/run_6/metrics_20_0.csv")
    time.sleep(EVALUATION_FREQUENCY / 2)  # Needs a couple of seconds after resetting services

    # Don't know if that's actually needed anymore
    last_assignments = agent_utils.get_last_assignment_from_metrics(ROOT + "/../../../share/metrics/metrics.csv")
    rask_agent.set_last_assignments(last_assignments)

    rask_agent.start()
    runtime_sec = 0
    while runtime_sec < DURATION_OPERATE:
        pattern_rps.reconfigure_rps(request_pattern, qr_local, MAX_RPS_QR, runtime_sec)
        pattern_rps.reconfigure_rps(request_pattern, cv_local, MAX_RPS_CV, runtime_sec)
        time.sleep(EVALUATION_FREQUENCY)

        runtime_sec += EVALUATION_FREQUENCY
        export_experience_buffer(rask_agent.experience_buffer, experience_file)
        rask_agent.experience_buffer.clear()

    rask_agent.terminate_gracefully()
    # export_experience_buffer(rask_agent.experience_buffer, ROOT + f"/agent_experience_{agent_suffix}.csv")
    print(f"{agent_suffix} agent finished evaluation after {DURATION_EXPLORE} seconds")


if __name__ == '__main__':
    agent_utils.stream_remote_metrics_file(REMOTE_VM, EVALUATION_FREQUENCY)

    exploring_agent = RASK_Global_Agent(
        prom_server=PROMETHEUS,
        services_monitored=[qr_local, cv_local, pc_local],
        evaluation_cycle=EVALUATION_FREQUENCY,
        log_experience=1,
        max_explore=MAX_EXPLORE,
        gaussian_noise=GAUSSIAN_NOISE
    )

    train_scaling_agent(exploring_agent, f"RASK_{MAX_EXPLORE}_{GAUSSIAN_NOISE}")

    operating_agent = RASK_Global_Agent(
        prom_server=PROMETHEUS,
        services_monitored=[qr_local, cv_local, pc_local],
        evaluation_cycle=EVALUATION_FREQUENCY,
        log_experience=1
    )

    operate_scaling_agent(operating_agent, "RASK_OPERATE", RequestPattern.BURSTY)

