import csv
import logging
import os
import random
import subprocess
import threading
import time
from typing import NamedTuple, Dict

import numpy as np
import pandas as pd

import utils

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
        random_ass = random.randint(round(bounds['min']), round(bounds['max']))
        random_params[param] = random_ass

    return random_params


def min_max_scale(value: float | int, min_val: float, max_val: float) -> float:
    """Min max scale to 0 and 1"""
    if min_val == max_val:
        return 1.0  # Should only happen for QR service model_size

    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_value = np.clip(scaled_value, 0, 1)

    return scaled_value


def to_partial(full_state):
    partial_state = full_state.copy()
    del partial_state["avg_p_latency"]
    del partial_state["throughput"]
    del partial_state["completion_rate"]

    return partial_state


def normalize_in_bounds(vector, min_val, max_val):
    normalized = (vector - min_val) / (max_val - min_val)
    return np.clip(normalized, min_val, max_val)


class FullStateDQN(NamedTuple):
    data_quality: int
    data_quality_target: int
    throughput: int
    throughput_target: int
    model_size: int  # only for CV!
    model_size_target: int  # only for CV!
    cores: int
    free_cores: int
    bounds: Dict[str, Dict]

    def for_pymdp(self, env_type):
        def discretize_value(value, bins):
            index = np.digitize(value, bins, right=False)
            # Cap to highest valid bin index
            return min(index, len(bins) - 1)
        if env_type == 'qr':
            throughput_qr_bins = np.arange(0, 101, 20)
            aif_throughput = discretize_value(self.throughput, throughput_qr_bins)

            base_quality = np.arange(300, 1100, 100)
            aif_quality = np.argmin(np.abs(base_quality - self.data_quality))

            aif_cores = self.cores - 1

            return [aif_throughput, aif_quality, aif_cores]

        elif env_type == 'cv':
            throughput_cv_bins = np.arange(0, 6, 1)
            aif_throughput = discretize_value(self.throughput, throughput_cv_bins)

            base_quality = np.arange(128, 352, 32)
            aif_quality = np.argmin(np.abs(base_quality - self.data_quality))

            aif_model_size = self.model_size - 1

            aif_cores = self.cores - 1

            return [aif_throughput, aif_quality, aif_model_size, aif_cores]

    def for_tensor(self):
        return [
            self.data_quality <= self.bounds['data_quality']['min'],
            self.data_quality >= self.bounds['data_quality']['max'],
            (self.model_size <= self.bounds['model_size']['min']) if 'model_size' in self.bounds else True,
            (self.model_size >= self.bounds['model_size']['max']) if 'model_size' in self.bounds else True,
            self.cores <= self.bounds['cores']['min'],
            self.free_cores > 0,
            self.data_quality / self.data_quality_target,
            self.model_size / self.model_size_target,
            self.throughput / self.throughput_target]

    def to_normalized_dict(self):
        state_array = self.to_np_ndarray(True)
        state_dict = {
            "data_quality": state_array[0],
            "data_quality_target": state_array[1],
            "throughput": state_array[2],
            "throughput_target": state_array[3],
            "model_size": state_array[4],
            "model_size_target": state_array[5],
            "cores": state_array[6],
            "free_cores": state_array[7],
        }
        return state_dict

    def to_np_ndarray(self, normalized: bool = True) -> np.ndarray[float]:
        if normalized:
            return np.asarray([
                min_max_scale(self.data_quality, min_val=self.bounds["data_quality"]["min"],
                              max_val=self.bounds["data_quality"]["max"]),
                min_max_scale(self.data_quality_target, min_val=self.bounds["data_quality"]["min"],
                              max_val=self.bounds["data_quality"]["max"]),
                min_max_scale(self.throughput, min_val=0, max_val=100),
                min_max_scale(self.throughput_target, min_val=0, max_val=100),
                min_max_scale(self.model_size, min_val=self.bounds["model_size"]["min"],
                              max_val=self.bounds["model_size"]["max"]) if 'model_size' in self.bounds else 1.0,
                min_max_scale(self.model_size_target, min_val=self.bounds["model_size"]["min"],
                              max_val=self.bounds["model_size"]["max"]) if 'model_size' in self.bounds else 1.0,
                min_max_scale(self.cores, min_val=self.bounds["cores"]["min"], max_val=self.bounds["cores"]["max"]),
                min_max_scale(self.free_cores, min_val=self.bounds["cores"]["min"],
                              max_val=self.bounds["cores"]["max"]),
            ])
        else:
            return np.asarray([
                self.data_quality,
                self.data_quality_target,
                self.throughput,
                self.throughput_target,
                self.model_size if 'model_size' in self.bounds else 1.0,
                self.model_size_target if 'model_size' in self.bounds else 1.0,
                self.cores,
                self.free_cores,
            ])


# @utils.print_execution_time
def export_experience_buffer(rows: tuple, file_name):
    file_exists = os.path.isfile(file_name)
    is_empty = not file_exists or os.path.getsize(file_name) == 0

    data = []

    if is_empty:
        data.append(["rep", "timestamp", "service", "slo_f", "state", "last_iteration_length"])

    data.extend([
        [prefix, timestamp, service.container_id, slo_f, service_state, last_iteration]
        for service, timestamp, slo_f, prefix, service_state, last_iteration in rows
    ])

    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


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


def stream_remote_metrics_file(remote_server: str, cycle_delay_seconds: int):
    def stream_csv():
        csv_buffer = []
        last_flush_time = time.monotonic()

        cmd = ["ssh", f"root@{remote_server}", "tail", "-f", "~/development/elastic-workbench/share/metrics/metrics.csv"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

        for line in process.stdout:
            csv_buffer.append(line)

            # Check if the cycle delay has passed
            if time.monotonic() - last_flush_time >= cycle_delay_seconds:
                if csv_buffer:  # Only write if there’s something to write
                    utils.write_metrics_to_csv(csv_buffer, pure_string=True)
                    csv_buffer = []
                last_flush_time = time.monotonic()  # Reset timer

    # Start the background thread
    thread = threading.Thread(target=stream_csv, daemon=True)
    thread.start()