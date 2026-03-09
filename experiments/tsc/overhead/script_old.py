import subprocess
import csv
import time
from datetime import datetime

CSV_FILENAME = "cpu_log.csv"
LOG_INTERVAL_SECONDS = 1

def get_cpu_usage_linux():
    """Returns CPU usage as a float on Linux/macOS using top command."""
    try:
        output = subprocess.check_output(
            ["top", "-bn1"], stderr=subprocess.DEVNULL, universal_newlines=True
        )
        for line in output.split("\n"):
            if "Cpu(s):" in line:
                usage = line.split(":")[1].split(",")[0]
                return float(usage.strip().split()[0])
    except Exception as e:
        print("Error reading CPU usage:", e)
        return None

def log_cpu_usage():
    # Create CSV file with header if it doesn't exist
    try:
        with open(CSV_FILENAME, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "CPU_Usage_Percent"])
    except FileExistsError:
        pass

    print(f"Logging CPU usage to {CSV_FILENAME} every {LOG_INTERVAL_SECONDS} seconds. Press Ctrl+C to stop.")

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_usage = get_cpu_usage_linux()
            if cpu_usage is not None:
                with open(CSV_FILENAME, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, cpu_usage])
            time.sleep(LOG_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nLogging stopped.")

if __name__ == "__main__":
    log_cpu_usage()
