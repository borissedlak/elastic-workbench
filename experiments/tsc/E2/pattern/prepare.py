import csv

import numpy as np
from matplotlib import pyplot as plt

from agent import agent_utils

plt.rcParams.update({'font.size': 12})

for pattern in ['bursty', 'diurnal']:

    # Read, apply, and collect results
    with open(f"{pattern}.txt", "r") as file:

        values = [int(line.strip()) for line in file]
        max_value = max(values)

        values = [(x, agent_utils.min_max_scale(x, 0, max_value)) for x in values]
        print(max_value)

        # Write results back to the same file
        with open(f"{pattern}.csv", "w") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["rps_absolute", "rps_normalized"])

            for abs, rel in values:
                file.write(f"{abs},{rel}\n")

        x = np.arange(1, 3601)
        y = [v[1] for v in values]
        plt.figure(figsize=(4.5, 3.2))
        plt.plot(x, y, linewidth=2)
        plt.ylabel("% Request Load")
        plt.xlabel("Time in Experiment (s)")
        plt.savefig(f"{pattern}.eps", bbox_inches='tight')
        plt.show()
