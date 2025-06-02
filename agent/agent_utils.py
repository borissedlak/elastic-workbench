import csv
import datetime
import logging
import os
import random
import time
from typing import NamedTuple, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger('multiscale')


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000.0
        logger.info(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        return result

    return wrapper


# @utils.print_execution_time # Recently almost 500ms
def filter_rows_during_cooldown(df: pd.DataFrame):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Identify timestamps where the flag is True
    flagged_rows = df.loc[df['cooldown'] > 0]

    # Add a 3-second range for each flagged timestamp
    mask = pd.Series(False, index=df.index)

    for index, row in flagged_rows.iterrows():
        mask |= ((df['timestamp'] >= row['timestamp']) &
                 (df['timestamp'] <= row['timestamp'] + pd.Timedelta(seconds=row['cooldown'] / 1000)))  # FUll sec

    filtered_df = df[~mask]
    return filtered_df


def get_random_parameter_assignments(parameters):
    random_params = {}

    for param, bounds in parameters.items():
        random_ass = random.randint(bounds['min'], bounds['max'])
        random_params[param] = random_ass

    return random_params


def to_partial(full_state):
    partial_state = full_state.copy()
    del partial_state["avg_p_latency"]
    del partial_state["throughput"]
    del partial_state["completion_rate"]

    return partial_state


def normalize_in_bounds(vector, min_val, max_val):
    normalized = (vector - min_val) / (max_val - min_val)
    return np.clip(normalized, min_val, max_val)


class Full_State_DQN(NamedTuple):
    quality: int
    quality_thresh: int
    throughput: int
    tp_thresh: int
    model_size: int
    model_size_thresh: int
    cores: int
    free_cores: int
    bounds: Dict[str, Dict]

    def discretize(self, service_type):
        return [
            np.digitize(self.quality, np.arange(300, 1000, 100)) if service_type.value == "CV" else np.digitize(self.quality, np.arange(128, 320, 32)),
            np.digitize(self.throughput, np.arange(0, 100, 5))]

    def for_tensor(self):
        return [
            self.quality <= self.bounds['quality']['min'],
            self.quality >= self.bounds['quality']['max'],
            (self.model_size <= self.bounds['model_size']['min']) if 'model_size' in self.bounds else True,
            (self.model_size >= self.bounds['model_size']['max']) if 'model_size' in self.bounds else True,
            self.cores <= self.bounds['cores']['min'],
            self.free_cores > 0,
            self.quality / self.quality_thresh,
            self.model_size / self.model_size_thresh,
            self.throughput / self.tp_thresh]


# @utils.print_execution_time
def export_experience_buffer(rows: tuple, file_name):

    file_exists = os.path.isfile(file_name)
    is_empty = not file_exists or os.path.getsize(file_name) == 0

    data = []

    if is_empty:
        data.append(["rep", "timestamp", "service", "slo_f", "state"])

    data.extend([
        [prefix, timestamp, service.container_id, slo_f, service_state]
        for service, timestamp, slo_f, prefix, service_state in rows
    ])

    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def log_service_state(state: Full_State_DQN, prefix):
    # Define the directory and file name
    directory = "./"
    file_name = "agent_experience.csv"
    file_path = os.path.join(directory, file_name)

    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writerow(
                ["rep", "timestamp", "quality", "quality_thresh", "throughput", "throughput_thresh", "cores",
                 "free_cores"])

        writer.writerow([prefix, datetime.datetime.now()] + list(state))


def wait_for_remaining_interval(interval_length: int, start_time: float):
    interval_ms = 1000 * interval_length
    time_elapsed = int((time.perf_counter() - start_time) * 1000)
    if time_elapsed < interval_ms:
        time.sleep((interval_ms - time_elapsed) / 1000)


def delete_file_if_exists(file_path="./agent_experience.csv"):
    file_path = os.path.join(file_path)

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} deleted.")
    else:
        print(f"{file_path} does not exist.")
