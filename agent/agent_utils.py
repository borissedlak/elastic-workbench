import csv
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.readwrite import XMLBIFWriter

from slo_config import PW_MAX_CORES, Full_State

logger = logging.getLogger('multiscale')


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


# @print_execution_time # Roughly 1 to 1.5s
def train_lgbn_model(df, show_result=False):
    df_filtered = filter_3s_after_change(df.copy())

    model = LinearGaussianBayesianNetwork([('pixel', 'fps'), ('cores', 'fps'), ('cores', 'energy'), ('pixel', 'energy')])
    # XMLBIFWriter(model).write_xmlbif("../model.xml")
    model.fit(df_filtered)

    if show_result:
        for cpd in model.get_cpds():
            print(cpd)

        for states in [["pixel", "fps"], ["cores", "fps"], ["pixel", "energy"], ["cores", "energy"]]:
            X_samples = model.simulate(1500, 35)
            X_df = pd.DataFrame(X_samples, columns=states)

            sns.jointplot(x=X_df[states[0]], y=X_df[states[1]], kind="kde", height=10, space=0, cmap="viridis")
            plt.show()

    return model


# @utils.print_execution_time # Recently almost 500ms
def filter_3s_after_change(df: pd.DataFrame):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Identify timestamps where the flag is True
    flagged_times = df.loc[df['change_flag'], 'timestamp']

    # Add a 3-second range for each flagged timestamp
    mask = pd.Series(False, index=df.index)

    for t in flagged_times:
        mask |= (df['timestamp'] >= t) & (df['timestamp'] <= t + pd.Timedelta(seconds=3))

    filtered_df = df[~mask]
    return filtered_df


def get_free_cores(core_dict, max_cores):
    if len(core_dict) == 0:
        return max_cores

    free_cores = max_cores - np.sum([item[1] for item in core_dict.items()])
    return free_cores


def was_qn_ever_trained(qn):
    return qn.last_time_trained != datetime(1970, 1, 1, 0, 0, 0)


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
            writer.writerow(["id", "timestamp", "pixel", "pixel_thresh", "fps", "fps_thresh", "energy", "cores", "free_cores"])

        writer.writerow([prefix, datetime.now()] +  list(state))
