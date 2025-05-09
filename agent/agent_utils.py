import csv
import datetime
import logging
import os
import random
import time
from typing import NamedTuple

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


class Full_State(NamedTuple):
    quality: int
    quality_thresh: int
    throughput: int
    tp_thresh: int
    cores: int
    free_cores: int

    def for_tensor(self):
        return [self.quality / self.quality_thresh, self.throughput / self.tp_thresh, self.cores,
                self.quality > 300, self.quality < 1100, self.free_cores > 0]

        # return [self.quality, self.quality / self.quality_thresh, self.throughput, self.throughput / self.tp_thresh,
        #     self.cores, self.free_cores > 0]

        # return [self.quality, self.quality_thresh, self.throughput, self.tp_thresh,
        #     self.cores, self.free_cores > 0]


def log_agent_experience(state: Full_State, prefix):
    # Define the directory and file name
    directory = "./"
    file_name = "slo_f.csv"
    file_path = os.path.join(directory, file_name)

    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writerow(
                ["rep", "timestamp", "pixel", "pixel_thresh", "fps", "fps_thresh", "energy", "cores", "free_cores"])

        writer.writerow([prefix, datetime.datetime.now()] + list(state))
