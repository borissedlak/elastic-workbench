import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.iwai.B1.B1 import COLOR_DICT_AGENT

ROOT = os.path.dirname(__file__)
AGENT_TYPES = ["DQN", "RASK", "AIF", "DACI"]
plt.rcParams.update({'font.size': 12})


def load_and_process_experience(agent_type):
    file_path = os.path.join(ROOT, f"agent_experience_{agent_type}.csv")
    df = pd.read_csv(file_path)

    # Filter out invalid iteration lengths
    df = df[df['last_iteration_length'] != -1]

    paired_df = df.groupby(df.index // 2).agg({
        'rep': 'first',
        'timestamp': 'first',
        'last_iteration_length': 'sum'
    })

    if agent_type != "RASK":
        return {agent_type: paired_df['last_iteration_length'].values}

    # Special handling for RASK: split into warmup and main
    warmup, main = [], []
    for rep in paired_df['rep'].unique():
        rep_data = paired_df[paired_df['rep'] == rep]['last_iteration_length'].values
        warmup.extend(rep_data[:40])
        main.extend(rep_data[40:])

    return {
        "RASK (expl)": warmup,
        "RASK (inf)": main
    }


def main():
    data = {}
    for agent in AGENT_TYPES:
        results = load_and_process_experience(agent)
        data.update(results)
    # With log10 data - so we do not need to have a log-axis, and we show the variance properly
    # data = {agent: np.log10(values) for agent, values in data.items()}

    # Define plotting order
    plot_labels = ["DQN", "RASK (expl)", "RASK (inf)", "AIF", "DACI"]
    plot_data = [data[label] for label in plot_labels]

    # Create the boxplot
    plt.figure(figsize=(6.0, 3.8))
    box = plt.boxplot(
        plot_data,
        labels=plot_labels,
        patch_artist=True
    )

    # Custom coloring
    for patch, label in zip(box['boxes'], plot_labels):
        base_label = label.split(' ')[0]  # e.g., "RASK" from "RASK_main"
        patch.set_facecolor(COLOR_DICT_AGENT[base_label])

    plt.ylabel("Agent Cycle Duration (ms)")
    # plt.title("Iteration Time Distribution per Agent")
    plt.yscale("log")  # <- Logarithmic y-axis
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.yticks([10, 100, 1000, 10000])

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "plots", "iteration_time.png"), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
