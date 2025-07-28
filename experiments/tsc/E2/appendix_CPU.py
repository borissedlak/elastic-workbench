import ast  # for safe literal_eval of state dicts
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from agent.components.SLORegistry import smoothstep
from experiments.tsc.E1.E1 import moving_average
from experiments.tsc.E2.E2 import EXPERIMENT_DURATION, EVALUATION_FREQUENCY, EXPERIMENT_REPETITIONS, LINE_STYLE_DICT

ROOT = os.path.dirname(__file__)

# Custom SLO thresholds
THRESHOLDS = {
    "elastic-workbench-qr-detector-1": {
        "data_quality": 800,
        "model_size": 1,
        "completion_rate": 1,
    },
    "elastic-workbench-cv-analyzer-1": {
        "data_quality": 288,
        "model_size": 3,
        "completion_rate": 1
    },
    "elastic-workbench-pc-visualizer-1": {
        "data_quality": 40,
        "model_size": 1,
        "completion_rate": 1,
    }
}


def calculate_mean_and_std(df: pd.DataFrame, experiment_repetitions: int, metric_name: str = 'slo_f'):
    """
    Compute the mean and standard deviation over time for the given SLO metric across experiment repetitions.

    Args:
        df (pd.DataFrame): The input DataFrame with at least ['rep', metric_name] columns.
        experiment_repetitions (int): Number of repetitions to group by.
        metric_name (str): Column name of the SLO metric to compute stats for (default: 'slo_f').

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean and standard deviation arrays over time.
    """
    slo_fs_index = []

    for j in range(1, experiment_repetitions + 1):
        slo_f_run = df[df['rep'] == j][metric_name]
        slo_fs_index.append(slo_f_run.to_list())

    array = np.array(slo_fs_index)
    mean_over_time = np.mean(array, axis=0)
    std_over_time = np.std(array, axis=0)

    return mean_over_time, std_over_time


def calculate_cores(row):
    state = ast.literal_eval(row['state'])

    cores = float(state.get('cores', 0))
    return pd.Series([cores], index=['cores'])


num_points = int(EXPERIMENT_DURATION / EVALUATION_FREQUENCY) + 1
x = np.arange(0, (num_points + 1) * EVALUATION_FREQUENCY, EVALUATION_FREQUENCY)

def visualize_data(data, pattern):

    for file, agent, color in data:
        df = pd.read_csv(ROOT + f"/{file}")

        # Add custom SLO columns
        df[['cores']] = df.apply(calculate_cores, axis=1)

        plt.figure(figsize=(5.4, 3.2))
        for service, alias, line in [("elastic-workbench-qr-detector-1", "QR", "-"), ("elastic-workbench-cv-analyzer-1", "CV", ":"), ("elastic-workbench-pc-visualizer-1", "PC", "--")]:

            subset_df = df[df['service'] == service]

            for metric, label, linestyle in [('cores', "CPU Cores", line)]:
                # Group every 3 rows (assumes they are time-step related)

                subset_df[metric] = moving_average(subset_df[metric], window_size=20)
                s_mean, _ = calculate_mean_and_std(subset_df, EXPERIMENT_REPETITIONS, "cores")

            plt.plot(x[:len(s_mean)], s_mean, label=f"{alias}", linewidth=2, linestyle=linestyle)

        plt.xlim(0, x[len(s_mean) - 1])
        plt.ylim(0, 6.1)
        plt.xlabel("Time in Experiment (s)")
        plt.ylabel(f"CPU Core Allocation")
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(ROOT + f"/plots/appendix/E2_CPU_{pattern}_{agent}.pdf", dpi=600, bbox_inches="tight")
        plt.show()

bursty_runs_2 = [
    ("run_3/agent_experience_RASK_0_bursty.csv", "RASK", "blue"),
    ("run_4/agent_experience_k8_0_bursty.csv", "VPA", "orange"),
    ("run_4/agent_experience_dqn_0_bursty.csv", "DQN", "green"),
]
visualize_data(bursty_runs_2, "bursty")

diurnal_runs = [
    ("run_3/agent_experience_RASK_0_diurnal.csv", "RASK", "blue"),
    ("run_3/agent_experience_k8_0_diurnal.csv", "VPA", "orange"),
    ("run_3/agent_experience_dqn_0_diurnal.csv", "DQN", "green")
]

visualize_data(diurnal_runs, "diurnal")