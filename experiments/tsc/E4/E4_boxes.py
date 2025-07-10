import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(__file__)
plt.rcParams.update({'font.size': 12})

# TODO: If I really want, I can rerun the RASK 3, but I dont think it should be necessary
files = [
    "agent_experience_RASK_3_diurnal.csv",
    "agent_experience_RASK_6_diurnal.csv",
    "agent_experience_RASK_9_diurnal_temp.csv",
]

# Clean aliases for nicer labeling
aliases = {
    "agent_experience_RASK_3_diurnal.csv": "3 Services\n8 Cores",
    "agent_experience_RASK_6_diurnal.csv": "6 Services\n16 Cores",
    "agent_experience_RASK_9_diurnal_temp.csv": "9 Services\n24 Cores",
}


def load_and_process_experience(metrics_file):
    file_path = os.path.join(ROOT, metrics_file)
    df = pd.read_csv(file_path)

    # Filter out invalid iteration lengths
    df = df[df['last_iteration_length'] != -1]

    # Group every 3 rows to form a "cycle"
    paired_df = df.groupby(df.index // 3).agg({
        'last_iteration_length': 'mean',
        'slo_f': 'mean'  # average SLO per cycle
    })

    return paired_df['last_iteration_length'].values, paired_df['slo_f'].values


def main():
    iteration_data = []
    slo_data = []
    labels = []

    for f in files:
        iteration_values, slo_values = load_and_process_experience(f)
        iteration_data.append(iteration_values)
        slo_data.append(slo_values)
        labels.append(aliases[f])

    # Plot 1: Iteration Time
    fig1, ax1 = plt.subplots(figsize=(4.2, 5))
    box = ax1.boxplot(iteration_data, tick_labels=labels)
    medians = [line.get_ydata()[0] for line in box['medians']]
    x_positions = list(range(1, len(medians) + 1))
    ax1.plot(x_positions, medians, linestyle='--', color='green', label='Caching last scaling action')
    ax1.hlines(y=10000, xmin=0.5, xmax=3.5, color='red', label='Maximum cycle length', linewidth=2)
    ax1.legend(loc='upper left')

    ax1.set_ylabel("Total Iteration Time per Cycle (ms)")
    ax1.set_ylim(0, 12500)
    ax1.tick_params(axis='x')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    fig1.tight_layout()
    fig1.savefig("plots/E3_runtime.pdf")
    plt.show()
    plt.close(fig1)

    # # Plot 2: SLO Fulfillment
    fig2, ax2 = plt.subplots(figsize=(4.0, 5))
    box2 = ax2.boxplot(slo_data, tick_labels=labels)
    medians2 = [line.get_ydata()[0] for line in box2['medians']]
    x_positions2 = list(range(1, len(medians2) + 1))
    ax2.plot(x_positions2, medians2, linestyle='--', color='green', label='Caching last scaling action')

    ax2.set_ylabel("Global SLO Fulfillment during Experiment")
    ax2.set_ylim(0.54, 1.025)
    ax2.tick_params(axis='x')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left')
    fig2.tight_layout()
    fig2.savefig("plots/E3_SLO_F.pdf")
    plt.show()
    plt.close(fig2)


if __name__ == '__main__':
    main()
