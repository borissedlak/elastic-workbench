import logging
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils
from agent.ES_Registry import ServiceID, ServiceType
from agent.RRM_Global_Agent import RRM_Global_Agent
from agent.SLO_Registry import calculate_slo_fulfillment, SLO_Registry, to_normalized_SLO_F
from agent.agent_utils import delete_experience_file
from iwai.DQN_Agent import DQN_Agent
from iwai.DQN_Trainer import ACTION_DIM, DQN, STATE_DIM

plt.rcParams.update({'font.size': 12})

nn_folder = "./networks"
EXPERIMENT_REPETITIONS = 3
EXPERIMENT_DURATION = 200

ps = "http://172.20.0.2:9090"

qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")

EVALUATION_FREQUENCY = 4
MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))

slo_registry = SLO_Registry("./config/slo_config.json")
client_SLOs = slo_registry.get_SLOs_for_client("C_1", qr_local.service_type)

logging.getLogger('multiscale').setLevel(logging.WARNING)


def train_q_network():
    file_path = "LGBN.csv"
    df = pd.read_csv(file_path)

    dqn = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM, force_restart=True, nn_folder=nn_folder)
    dqn.train_dqn_from_env(df)

    print(f"Finished Q-Network Training")


# TODO: Generalize for different agent types
def eval_DQN_agent():
    delete_experience_file()

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
    delete_experience_file()

    print(f"Starting experiment for Agent")

    for rep in range(1, EXPERIMENT_REPETITIONS + 1):
        agent = RRM_Global_Agent(services_monitored=[qr_local, cv_local], prom_server=ps,
                                 evaluation_cycle=EVALUATION_FREQUENCY, slo_registry_path="./config/slo_config.json",
                                 es_registry_path="./config/es_registry.json",
                                 log_experience=rep)
        agent.reset_services_states()
        time.sleep(2)

        agent.start()
        time.sleep(EXPERIMENT_DURATION)
        agent.terminate_gracefully()
        print(f"Agent finished evaluation round #{rep} after {EXPERIMENT_DURATION * rep} seconds")


def visualize_data(slof_files, output_file):
    m_meth, std_meth = calculate_mean_std(slof_files[0])
    # m_base, _ = calculate_mean_std(slof_files[1])
    # changes_meth, changes_base = get_changed_lines(slof_files[0]), get_changed_lines(slof_files[1])

    # TODO: Maybe I can switch to timesteps on the x axis? Also is more intuitive to read
    x = np.arange(len(m_meth))
    lower_bound = np.array(m_meth) - np.array(std_meth)
    upper_bound = np.array(m_meth) + np.array(std_meth)

    plt.figure(figsize=(6.0, 3.8))
    # plt.plot(x, m_base, label='Baseline', color='red', linewidth=1)
    plt.plot(x, m_meth, label='Mean SLO Fulfillment', color='red', linewidth=2)
    plt.fill_between(x, lower_bound, upper_bound, color='red', alpha=0.2)  # , label='Standard Deviation')
    # plt.plot(x, m_base, label='Baseline VPA', color='black', linewidth=1.5)
    # plt.vlines([0.1, 10, 20, 30, 40], ymin=1.25, ymax=2.75, label='Adjust Thresholds', linestyles="--")

    # plt.xlim(-0.1, 49.1)
    plt.ylim(0.0, 1.0)

    plt.xlabel('Scaling Agent Iterations')
    plt.ylabel('SLO Fulfillment')
    # plt.title('Mean SLO Fulfillment')
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


def calculate_mean_std(slof_file):
    df = pd.read_csv(slof_file)
    del df['timestamp']

    # parts = np.array_split(df, EXPERIMENT_REPETITIONS)
    slo_fs_index = []

    # Step 2: Reindex each part
    for j in range(1, EXPERIMENT_REPETITIONS + 1):
        states = df[df['rep'] == j].to_dict(orient='records')
        slo_f_run = [to_normalized_SLO_F(calculate_slo_fulfillment(s, client_SLOs)) for s in states]
        slo_fs_index.append(slo_f_run)

    array = np.array(slo_fs_index)
    mean_over_time = np.mean(array, axis=0)
    std_over_time = np.std(array, axis=0)

    return mean_over_time, std_over_time


if __name__ == '__main__':
    # train_q_network()
    # eval_DQN_agent()
    eval_RRM_agent()
    visualize_data(["agent_experience.csv"], "./plots/slo_f.png")
