import csv
import os

ROOT = os.path.dirname(__file__)

for f in ["0ES", "1ES", "2ES", "3ES"]:

    total = 0
    count = 0
    
    with open(ROOT+ f'/cpu_log_{f}.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total += float(row['CPU_Usage_Percent'])
            count += 1

    mean_usage = total / count
    print(f"Mean CPU usage for {f}: {mean_usage:.2f}%")