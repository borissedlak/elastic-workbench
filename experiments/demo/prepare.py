#!/usr/bin/env python3
"""
gui_levers_maximize.py
Tkinter GUI with visible sliders (levers), live text preview, a canvas preview,
and a top control bar containing a Maximize/Restore toggle button.
"""
import logging
import os
import tkinter as tk
from datetime import timedelta
from tkinter import ttk

import pandas as pd

from agent import agent_utils
from agent.components.RASK import RASK

ROOT = os.path.dirname(__file__)
logger = logging.getLogger("multiscale")

def create_rask_model_renderings(demo_part):
    df = pd.read_csv(ROOT + f"/metrics_{demo_part}.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    start = df['timestamp'].min()
    end = df['timestamp'].max()
    logger.info(f"Iterating from {start} to {end} by 1s steps")

    current = start
    iteration = 1 if demo_part == "EXPLORE" else 31

    df_train = pd.read_csv(ROOT + "/metrics_EXPLORE.csv") if demo_part == "OPERATE" else pd.DataFrame([])
    df_train['timestamp'] = pd.to_datetime(df['timestamp']) if demo_part == "OPERATE" else None

    rask = RASK(show_figures=True)
    while current <= end:
        filter_df = df[df['timestamp'] <= current].copy()
        merge_df = pd.concat([df_train, filter_df], axis=0, ignore_index=True)

        # if len(merge_df) >= 2:
        try:
            rask.init_models(merge_df, f"{iteration}")
        except Exception as e:
            logger.exception(f"Training failed at {current}: {e}")
        # else:
        #     logger.debug(f"At time {current} not enough rows ({len(merge_df)}) to train")

        iteration += 1
        logger.debug(f"Finished iteration {iteration} after {current} time in df")
        current = current + timedelta(seconds=10)

    logger.info(f"Completed {iteration} iterations from {start} to {end}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # show DEBUG and above
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # agent_utils.delete_folder_if_exists(ROOT + "/rask_plots")
    create_rask_model_renderings("EXPLORE")
    create_rask_model_renderings("OPERATE")

