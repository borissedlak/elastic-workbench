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


def calculate_custom_slo_fulfillment(row):
    try:
        state = ast.literal_eval(row['state'])
    except Exception:
        return pd.Series([np.nan, np.nan, np.nan])

    # Apply custom SLO logic
    f_quality = smoothstep(float(state.get('data_quality', 0)) / THRESHOLDS[row['service']]['data_quality'])
    f_model_size = smoothstep(float(state.get('model_size', 0)) / THRESHOLDS[row['service']]['model_size'])
    f_completion_rate = smoothstep(float(state.get('completion_rate', 0)) / THRESHOLDS[row['service']]['completion_rate'])
    return pd.Series([f_quality, f_model_size, f_completion_rate],
                     index=['slo_f_quality', 'slo_f_model_size', 'slo_f_completion_rate'])


num_points = int(EXPERIMENT_DURATION / EVALUATION_FREQUENCY) + 1
x = np.arange(0, (num_points + 1) * EVALUATION_FREQUENCY, EVALUATION_FREQUENCY)

def visualize_data(data, pattern):
    for file, agent, color in data:
        df = pd.read_csv(ROOT + f"/{file}")

        # Add custom SLO columns
        df[['slo_f_quality', 'slo_f_model_size', 'slo_f_completion_rate']] = df.apply(calculate_custom_slo_fulfillment,
                                                                                      axis=1)

        plt.figure(figsize=(5.4, 3.2))
        for service, alias in [("elastic-workbench-qr-detector-1", "QR"), ("elastic-workbench-cv-analyzer-1", "CV"),
                        ("elastic-workbench-pc-visualizer-1", "PC")]:

            subset_df = df[df['service'] == service]
            for slo_metric, label, linestyle in [('slo_f', "global weighted", "-"), ('slo_f_quality', 'data quality', ":"),
                                      ('slo_f_completion_rate', 'completion rate', "--"), ('slo_f_model_size', 'model size', '-.')]:
                # Group every 3 rows (assumes they are time-step related)

                if service != "elastic-workbench-cv-analyzer-1" and slo_metric == 'slo_f_model_size':
                    continue

                subset_df[slo_metric] = moving_average(subset_df[slo_metric], window_size=20)
                s_mean, _ = calculate_mean_and_std(subset_df, EXPERIMENT_REPETITIONS, slo_metric)
                plt.plot(x[:len(s_mean)], s_mean, label=f"{label}", linewidth=2, linestyle=linestyle)

            plt.xlim(0, x[len(s_mean) - 1])
            plt.ylim(0.0, 1.02)
            plt.xlabel("Time in Experiment (s)")
            plt.ylabel(f"{alias}: SLO Fulfillment")
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(ROOT + f"/plots/appendix/E2_SLO_F_{pattern}_{agent}_{alias}.pdf", dpi=600, bbox_inches="tight")
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