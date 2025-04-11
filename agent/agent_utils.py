import csv
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd

from agent.obsolete.slo_config import Full_State

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
                ["index", "rep", "timestamp", "pixel", "pixel_thresh", "fps", "fps_thresh", "energy", "cores",
                 "free_cores"])

        writer.writerow([prefix[0], prefix[1], datetime.now()] + list(state))


def get_random_parameter_assignments(parameters):
    random_ass = {}

    for param in parameters:
        value = random.randint(param['min'], param['max'])
        random_ass[param['name']] = value

    return random_ass

