import logging
import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

import utils
from agent.PolicySolverRRM import solve_global
from agent.RRMGlobalAgent import RRM_Global_Agent, apply_gaussian_noise_to_asses
from agent.es_registry import ServiceID, ServiceType, ESType
from iwai.dqn_trainer import ACTION_DIM_QR, DQN, STATE_DIM, ACTION_DIM_CV, QR_DATA_QUALITY_STEP, CV_DATA_QUALITY_STEP, \
    NO_EPISODES, EPISODE_LENGTH
from iwai.global_dqn_trainer import JointDQNTrainer
from iwai.global_training_env import GlobalTrainingEnv
from iwai.lgbn_training_env import LGBNTrainingEnv

ROOT = os.path.dirname(__file__)
plt.rcParams.update({'font.size': 12})

nn_folder = "./networks"
EXPERIMENT_REPETITIONS = 5
EXPERIMENT_DURATION = 100
MAX_EXPLORE = 15
MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))

ps = "http://172.20.0.2:9090"

qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")

EVALUATION_FREQUENCY = 5

# slo_path = "../../config/slo_config.json"
# es_path = "../../config/es_registry.json"

logger = logging.getLogger("multiscale")
logging.getLogger('multiscale').setLevel(logging.INFO)

df = pd.read_csv(ROOT + "/../../../share/metrics/LGBN.csv")

env_qr = LGBNTrainingEnv(ServiceType.QR, step_data_quality=QR_DATA_QUALITY_STEP)
env_qr.reload_lgbn_model(df)

env_cv = LGBNTrainingEnv(ServiceType.CV, step_data_quality=CV_DATA_QUALITY_STEP)
env_cv.reload_lgbn_model(df)

# Wrap in joint environment
joint_env = GlobalTrainingEnv(env_qr, env_cv, max_cores=8)


def train_q_network():
    joint_env.reset()

    # Create DQNs
    dqn_qr = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_QR)
    dqn_cv = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_CV)

    # Train jointly
    trainer = JointDQNTrainer(dqn_qr, dqn_cv, joint_env)
    trainer.train()

    print(f"Finished Q-Network Training")


class TestableRRMAgent(RRM_Global_Agent):
    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle, testing_env,
                 slo_registry_path=ROOT + "/../../../config/slo_config.json",
                 es_registry_path=ROOT + "/../../../config/es_registry.json",
                 log_experience=None, max_explore=10):
        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
                         log_experience, max_explore)
        self.testing_env: GlobalTrainingEnv = testing_env

    def execute_ES(self, host, service: ServiceID, es_type: ESType, params, respect_cooldown=True):
        print(es_type)

    def prepare_service_context(self, service_m: ServiceID) -> Tuple[ServiceType, Dict[ESType, Dict], Any, int]:
        assigned_clients = self.reddis_client.get_assignments_for_service(service_m)

        service_state = self.testing_env.env_cv.state if service_m == ServiceType.CV else self.testing_env.env_qr.state
        es_parameter_bounds = self.es_registry.get_parameter_bounds_for_active_ES(service_m.service_type)
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service_m.service_type, assigned_clients)
        total_rps = utils.to_absolut_rps(assigned_clients)

        # if self.log_experience is not None:
        #     self.evaluate_slos_and_buffer(service_m, service_state, all_client_slos)

        return service_m.service_type, es_parameter_bounds, all_client_slos, total_rps

    def orchestrate_services_optimally(self, services_m: list[ServiceID]):

        service_contexts = []
        for service_m in services_m:  # For all monitored services
            service_contexts.append(self.prepare_service_context(service_m))

        if self.explore_count < self.max_explore:
            logger.info("Agent is exploring.....")
            self.explore_count += 1
            # self.epsilon *= self.epsilon_decay
            self.call_all_ES_randomly(services_m)
        else:
            # TODO: Fix loading of experience here
            self.rrm.init_models()  # Reloads the RRM model from the metrics.csv
            assignments = solve_global(service_contexts, MAX_CORES, self.rrm)
            assignments = apply_gaussian_noise_to_asses(assignments)
            self.call_all_ES_deterministic(services_m, assignments)


# TODO: This will be a nice experiment, but the problem is that I don't have time. For the extension I need to do:
#  1) Bypass somehow the metrics from the services and ingest to the RRM init_model()
#  2) Act on the LGBN env, this might require another env
#  3) Calculate the reward and visualize it
def train_rrm_model():
    # joint_env.reset()
    agent = TestableRRMAgent(
        prom_server=ps,
        services_monitored=[qr_local, cv_local],
        evaluation_cycle=EVALUATION_FREQUENCY,
        testing_env=joint_env,
        log_experience=1,
        max_explore=MAX_EXPLORE,
    )

    max_episodes = NO_EPISODES
    episode_length = EPISODE_LENGTH
    score_list = []
    for ep in range(max_episodes):
        state_qr, state_cv = joint_env.reset()
        ep_score = 0

        for t in range(episode_length):
            agent.orchestrate_services_optimally([qr_local, cv_local])

    print(f"Finished RRM Training")


color_dict = {"elastic-workbench-qr-detector-1": "red", "elastic-workbench-cv-analyzer-1": "green"}
color_dict_agent = {"DQN": "red", "RRM": "green"}
line_style_dict = {"DQN": "--", "RRM": "-"}


def visualize_data(agent_types: list[str], output_file: str):
    # changes_meth, changes_base = get_changed_lines(slof_files[0]), get_changed_lines(slof_files[1])
    df = pd.read_csv(ROOT + f"/agent_experience_{agent_types[0]}.csv")
    x = np.arange(len(df.index) / (EXPERIMENT_REPETITIONS * 2))  # len(m_meth))
    plt.figure(figsize=(6.0, 3.8))

    # TODO: Ideally, I get the overall SLO-F per agent and show it with mean and std
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

        s_mean, s_std = calculate_mean_std(paired_df)
        lower_bound = np.array(s_mean) - np.array(s_std)
        upper_bound = np.array(s_mean) + np.array(s_std)
        plt.plot(x, s_mean, label=f"{agent}", color=color_dict_agent[agent], linewidth=2,
                 linestyle=line_style_dict[agent])
        plt.fill_between(x, lower_bound, upper_bound, color=color_dict_agent[agent], alpha=0.2)

    # plt.plot(x, m_base, label='Baseline VPA', color='black', linewidth=1.5)
    # plt.vlines([0.1, 10, 20, 30, 40], ymin=1.25, ymax=2.75, label='Adjust Thresholds', linestyles="--")

    plt.xlim(0.0, len(df.index) / (EXPERIMENT_REPETITIONS * 2) - 1)
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
    # train_q_network()
    train_rrm_model()

    # visualize_data(["RRM", "DQN"], ROOT + "/plots/slo_f.png")
    # sys.exit()
