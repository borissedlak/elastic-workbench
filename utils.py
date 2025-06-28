import csv
import logging
import os
import time
from typing import Dict

import numpy as np

logger = logging.getLogger('multiscale')
ROOT = os.path.dirname(__file__)

def get_env_param(var, default) -> str:
    env = os.environ.get(var)
    if env:
        logger.info(f'Found ENV value for {var}: {env}')
    else:
        env = default
        logger.warning(f"Didn't find ENV value for {var}, default to: {default}")
    return env


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000.0
        logger.info(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        # print(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        return result

    return wrapper


class FPS_:
    def __init__(self, max_fps=300):
        self.prev_time = 0
        self.new_time = 0

        self.time_store = Cyclical_Array(max_fps)

    def tick(self) -> None:
        self.time_store.put(time.time())

    # @print_execution_time
    def get_current_fps(self) -> int:
        current_time = time.time()
        recent_timestamps = [t for t in self.time_store.data if current_time - t <= 1]
        return len(recent_timestamps)


class Cyclical_Array:
    def __init__(self, size):
        self.data = np.zeros(size, dtype=object)
        self.index = 0
        self.size = size

    def put(self, item):
        self.data[self.index % self.size] = item
        self.index = self.index + 1

    def get_average(self):
        return np.mean(self.data, dtype=np.float64)


def convert_prom_multi(raw_result, decimal=False, avg=False):
    return {
        item['metric']["metric_id"]: (float if decimal else int)(item['value'][1])
        for item in raw_result
    }


def filter_tuple(t, name, index):
    return next((item for item in t if item[index] == name), None)


# @print_execution_time
def write_metrics_to_csv(lines, pure_string=False):
    # Define the directory and file name
    directory = ROOT + "/share/metrics"
    file_name = "metrics.csv"
    file_path = os.path.join(directory, file_name)

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)

        if not file_exists or os.path.getsize(file_path) == 0:
            csv_writer.writerow(["timestamp", "service_type", "container_id", "avg_p_latency", "s_config", "cores",
                             "rps", "throughput", "cooldown"])

        if pure_string:
            file.writelines(lines)
        else:
            csv_writer.writerows(lines)
        # print("Wrote lines")


def to_absolut_rps(client_arrivals: Dict[str, int]) -> int:
    return sum(i for i in client_arrivals.values())


def cores_to_threads(cores_reserved):
    return max(1, round(cores_reserved))
