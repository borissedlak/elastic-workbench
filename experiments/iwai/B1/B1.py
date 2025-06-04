import logging
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from agent.RRMGlobalAgent import RRM_Global_Agent
from agent.agent_utils import delete_file_if_exists, export_experience_buffer
from agent.es_registry import ServiceID, ServiceType
from experiments.iwai.transform_pymdp_logs import import_pymdp_logs
from iwai.dqn_agent import DQNAgent
from iwai.dqn_trainer import ACTION_DIM_QR, DQN, STATE_DIM, ACTION_DIM_CV

ROOT = os.path.dirname(__file__)
plt.rcParams.update({'font.size': 12})

nn_folder = "./networks"
EXPERIMENT_REPETITIONS = 3
EXPERIMENT_DURATION = 200
MAX_EXPLORE = 15

ps = "http://172.20.0.2:9090"

qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")

EVALUATION_FREQUENCY = 5

logging.getLogger('multiscale').setLevel(logging.INFO)



def eval_scaling_agent(agent_factory, agent_type):
    delete_file_if_exists(ROOT + f"/agent_experience_{agent_type}.csv")
    delete_file_if_exists(ROOT + "/../../../share/metrics/metrics.csv")

    print(f"Starting experiment for {agent_type} agent")

    for rep in range(1, EXPERIMENT_REPETITIONS + 1):
        # Start the agent with both services
        agent = agent_factory(rep)
        agent.reset_services_states()
        time.sleep(4)  # Needs a couple of seconds after resetting the services (i.e., calling ES)

        agent.start()
        time.sleep(EXPERIMENT_DURATION)
        agent.terminate_gracefully()
        export_experience_buffer(agent.experience_buffer, ROOT + f"/agent_experience_{agent_type}.csv")
        print(f"{agent_type} agent finished evaluation round #{rep} after {EXPERIMENT_DURATION * rep} seconds")


color_dict = {"elastic-workbench-qr-detector-1": "red", "elastic-workbench-cv-analyzer-1": "green"}
color_dict_agent = {"DQN": "red", "RRM": "green", "AIF": "blue"}
line_style_dict = {"DQN": "--", "RRM": "-", "AIF": "-."}


def visualize_data(agent_types: list[str], output_file: str):

    # changes_meth, changes_base = get_changed_lines(slof_files[0]), get_changed_lines(slof_files[1])
    df_layout = pd.read_csv(ROOT + f"/agent_experience_{agent_types[0]}.csv")
    x = np.arange(len(df_layout.index) / (EXPERIMENT_REPETITIONS * 2))  # len(m_meth))
    plt.figure(figsize=(6.0, 3.8))


    for agent in agent_types:
        df = pd.read_csv(ROOT + f"/agent_experience_{agent}.csv")
        # for service in df['service'].unique():
        #     df_filtered = df[df['service'] == service]
        #     s_mean, s_std = calculate_mean_std(df_filtered)
        #     lower_bound = np.array(s_mean) - np.array(s_std)
        #     upper_bound = np.array(s_mean) + np.array(s_std)
        #     plt.plot(x, s_mean, label=service+ f", {agent}", color=color_dict[service], linewidth=2,
        #              linestyle=line_style_dict[agent])
        #     plt.fill_between(x, lower_bound, upper_bound, color=color_dict[service], alpha=0.2)

        paired_df = df.groupby(df.index // 2).agg({
            'rep': 'first',
            'timestamp': 'first',
            'slo_f': 'mean'
        })

        s_mean, s_std = calculate_mean_std(paired_df) if agent != "AIF" else (paired_df['slo_f'].values, 0)
        lower_bound = np.array(s_mean) - np.array(s_std)
        upper_bound = np.array(s_mean) + np.array(s_std)
        plt.plot(x, s_mean, label=f"{agent}", color=color_dict_agent[agent], linewidth=2,
                 linestyle=line_style_dict[agent])
        plt.fill_between(x, lower_bound, upper_bound, color=color_dict_agent[agent], alpha=0.2)

    # plt.plot(x, m_base, label='Baseline VPA', color='black', linewidth=1.5)
    # plt.vlines([0.1, 10, 20, 30, 40], ymin=1.25, ymax=2.75, label='Adjust Thresholds', linestyles="--")

    plt.xlim(0.0, len(df_layout.index) / (EXPERIMENT_REPETITIONS * 2) - 1)
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
    #train_joint_q_networks(nn_folder=ROOT + "/networks")

    # Load the trained DQNs
    dqn_qr = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_QR)
    dqn_qr.load("Q_QR_joint.pt")
    dqn_cv = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_CV)
    dqn_cv.load("Q_CV_joint.pt")

    agent_fact_dqn = lambda repetition: DQNAgent(
        prom_server=ps,
        services_monitored=[qr_local, cv_local],
        dqn_for_services=[dqn_qr, dqn_cv],
        evaluation_cycle=EVALUATION_FREQUENCY,
        log_experience=repetition
    )

    agent_fact_rrm = lambda repetition: RRM_Global_Agent(
        prom_server=ps,
        services_monitored=[qr_local, cv_local],
        evaluation_cycle=EVALUATION_FREQUENCY,
        log_experience=repetition,
        max_explore=MAX_EXPLORE
    )

    #eval_scaling_agent(agent_fact_dqn, "DQN")
    #eval_scaling_agent(agent_fact_rrm, "RRM")
    import_pymdp_logs(filename = ROOT+ "/../20250604_175657_pymdp_service_log.csv")
    visualize_data(["RRM", "DQN", "AIF"], ROOT + "/plots/slo_f.png")
