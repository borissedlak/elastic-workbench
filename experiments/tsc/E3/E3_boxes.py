import os
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(__file__)
plt.rcParams.update({'font.size': 12})

files = [
    "agent_experience_RASK_lim_1_True_bursty.csv",
    "agent_experience_RASK_lim_1_False_bursty.csv",
    "agent_experience_RASK_lim_2_True_bursty.csv",
    "agent_experience_RASK_lim_2_False_bursty.csv",
    "agent_experience_RASK_True_bursty.csv",
    "agent_experience_RASK_False_bursty.csv",
]

# Clean aliases for nicer labeling
aliases = {
    "agent_experience_RASK_False_bursty.csv":       "3 Dim.\nNo Cache",
    "agent_experience_RASK_True_bursty.csv":        "3 Dim.\nCache",
    "agent_experience_RASK_lim_2_False_bursty.csv": "2 Dim.\nNo Cache",
    "agent_experience_RASK_lim_2_True_bursty.csv":  "2 Dim.\nCache",
    "agent_experience_RASK_lim_1_False_bursty.csv": "1 Dim.\nNo Cache",
    "agent_experience_RASK_lim_1_True_bursty.csv":  "1 Dim.\nCache",
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
    fig1, ax1 = plt.subplots(figsize=(5.5, 5))
    # ax1.boxplot(iteration_data, labels=labels)
    box = ax1.boxplot(iteration_data, labels=labels)

    # Draw the line between the medians
    medians = [line.get_ydata()[0] for line in box['medians']]
    # Define x positions for the boxplots (default is 1-indexed)
    x_positions = list(range(1, len(medians) + 1))
    # Draw line between medians of box 1, 3, 5
    odd_x = [x_positions[i] for i in [0, 2, 4]]
    odd_medians = [medians[i] for i in [0, 2, 4]]
    ax1.plot(odd_x, odd_medians, linestyle='--', color='blue', label='Without caching last actions')
    # Draw line between medians of box 2, 4, 6
    even_x = [x_positions[i] for i in [1, 3, 5]]
    even_medians = [medians[i] for i in [1, 3, 5]]
    ax1.plot(even_x, even_medians, linestyle='--', color='green', label='Caching last scaling action')
    ax1.legend(loc='upper left')

    ax1.set_ylabel("Total Iteration Time per Cycle (ms)")
    ax1.set_ylim(0, 3000)
    ax1.tick_params(axis='x')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    fig1.tight_layout()
    fig1.savefig("plots/E3_runtime.pdf")
    plt.show()
    plt.close(fig1)

    # Plot 2: SLO Fulfillment
    fig2, ax2 = plt.subplots(figsize=(5.5, 5))
    ax2.boxplot(slo_data, labels=labels)
    ax2.set_ylabel("Global SLO Fulfillment during Experiment")
    ax2.set_ylim(0.5, 1.05)
    ax2.tick_params(axis='x')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    fig2.tight_layout()
    fig2.savefig("plots/E3_SLO_F.pdf")
    plt.show()
    plt.close(fig2)

if __name__ == '__main__':
    main()
