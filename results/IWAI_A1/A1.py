import logging
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

import utils
from agent.ES_Registry import ServiceID, ServiceType
from agent.RRM_Global_Agent import RRM_Global_Agent
from agent.SLO_Registry import SLO_Registry
from agent.agent_utils import delete_file_if_exists
from iwai.DQN_Agent import DQN_Agent
from iwai.DQN_Trainer import ACTION_DIM, DQN, STATE_DIM

plt.rcParams.update({'font.size': 12})

nn_folder = "./networks"
EXPERIMENT_REPETITIONS = 5
EXPERIMENT_DURATION = 60

ps = "http://172.20.0.2:9090"

qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")

EVALUATION_FREQUENCY = 5
MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))

slo_registry = SLO_Registry("./config/slo_config.json")
client_SLOs = slo_registry.get_SLOs_for_client("C_1", qr_local.service_type)

logging.getLogger('multiscale').setLevel(logging.INFO)


def train_q_network():
    file_path = "LGBN.csv"
    df = pd.read_csv(file_path)

    dqn = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM, force_restart=True, nn_folder=nn_folder)
    dqn.train_dqn_from_env(df)

    print(f"Finished Q-Network Training")


# TODO: Generalize for different agent types
def eval_DQN_agent():
    delete_file_if_exists()

    print(f"Starting experiment for Agent")
    dqn = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM, nn_folder=nn_folder)

    for rep in range(1, EXPERIMENT_REPETITIONS + 1):
        agent = DQN_Agent(services_monitored=[qr_local], prom_server=ps, evaluation_cycle=EVALUATION_FREQUENCY, dqn=dqn,
                          slo_registry_path="./config/slo_config.json", es_registry_path="./config/es_registry.json",
                          log_experience=rep)
        agent.reset_services_states()
        time.sleep(2)

        agent.start()
        time.sleep(EXPERIMENT_DURATION)
        agent.terminate_gracefully()
        print(f"Agent finished evaluation round #{rep} after {EXPERIMENT_DURATION * rep} seconds")


def eval_RRM_agent():
    delete_file_if_exists("./agent_experience.csv")
    delete_file_if_exists("../../share/metrics/metrics.csv")

    print(f"Starting experiment for Agent")

    for rep in range(1, EXPERIMENT_REPETITIONS + 1):
        agent = RRM_Global_Agent(services_monitored=[qr_local, cv_local], prom_server=ps,
                                 evaluation_cycle=EVALUATION_FREQUENCY, slo_registry_path="./config/slo_config.json",
                                 es_registry_path="./config/es_registry.json",
                                 log_experience=rep)
        agent.reset_services_states()
        time.sleep(5)  # Needs a couple of seconds after resetting the services (i.e., calling ES)

        agent.start()
        time.sleep(EXPERIMENT_DURATION)
        agent.terminate_gracefully()
        print(f"Agent finished evaluation round #{rep} after {EXPERIMENT_DURATION * rep} seconds")


def color_for_s(service_type):
    if service_type == "elastic-workbench-qr-detector-1":
        return 'red'
    elif service_type == "elastic-workbench-cv-analyzer-1":
        return 'green'
    else:
        raise Exception(f"Unknown service_type: {service_type}")


def visualize_data(slof_files, output_file):
    # changes_meth, changes_base = get_changed_lines(slof_files[0]), get_changed_lines(slof_files[1])
    df = pd.read_csv(slof_files[0])

    # TODO: Maybe I can switch to timesteps on the x axis? Also is more intuitive to read
    x = np.arange(len(df.index) / (EXPERIMENT_REPETITIONS * 2))  # len(m_meth))

    plt.figure(figsize=(6.0, 3.8))
    # plt.plot(x, m_base, label='Baseline', color='red', linewidth=1)

    for service in df['service'].unique():
        df_filtered = df[df['service'] == service]
        s_mean, s_std = calculate_mean_std(df_filtered)
        lower_bound = np.array(s_mean) - np.array(s_std)
        upper_bound = np.array(s_mean) + np.array(s_std)
        plt.plot(x, s_mean, label=service, color=color_for_s(service), linewidth=2)  # label = ''
        plt.fill_between(x, lower_bound, upper_bound, color=color_for_s(service), alpha=0.2)

    # plt.plot(x, m_base, label='Baseline VPA', color='black', linewidth=1.5)
    # plt.vlines([0.1, 10, 20, 30, 40], ymin=1.25, ymax=2.75, label='Adjust Thresholds', linestyles="--")

    # plt.xlim(-0.1, 49.1)
    plt.ylim(0.0, 1.0)

    plt.xlabel('Scaling Agent Iterations')
    plt.ylabel('SLO Fulfillment')
    plt.legend()
    plt.savefig(output_file, dpi=600, bbox_inches="tight", format="png")
    plt.show()


def get_changed_lines(slof_file):
    df = pd.read_csv(slof_file)

    df['cores_change'] = df['cores'].ne(df['cores'].shift())
    df['quality_change'] = df['quality'].ne(df['quality'].shift())

    # Combine the change indicators
    df['change'] = df['cores_change'] | df['quality_change']
    return df['change']


def calculate_mean_std(df: DataFrame):
    del df['timestamp']

    # parts = np.array_split(df, EXPERIMENT_REPETITIONS)
    slo_fs_index = []

    # Step 2: Reindex each part
    for j in range(1, EXPERIMENT_REPETITIONS + 1):
        slo_f_run = df[df['rep'] == j]['slo_f']
        # slo_f_run = [r['slo_f'] for r in rows]
        slo_fs_index.append(slo_f_run.to_list())

    array = np.array(slo_fs_index)
    mean_over_time = np.mean(array, axis=0)
    std_over_time = np.std(array, axis=0)

    return mean_over_time, std_over_time


if __name__ == '__main__':
    # delete_experience_file()
    # agent = RRM_Global_Agent(services_monitored=[qr_local, cv_local], prom_server=ps,
    #                          evaluation_cycle=EVALUATION_FREQUENCY, slo_registry_path="./config/slo_config.json",
    #                          es_registry_path="./config/es_registry.json")
    # agent.reset_services_states()
    # sys.exit()

    # train_q_network()
    # eval_DQN_agent()
    eval_RRM_agent()
    visualize_data(["agent_experience.csv"], "./plots/slo_f.png")
