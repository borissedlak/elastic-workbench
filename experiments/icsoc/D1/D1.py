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

class LeverGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Levers & Text — example GUI")
        self.geometry("760x420")        # initial size
        self.minsize(640, 360)
        self._is_fullscreen = False    # for our maximize/restore
        self._create_widgets()
        self._layout_widgets()
        self._update_text()
        self._update_preview()

    def _create_widgets(self):
        # Top control bar (buttons in the "control bar on top")
        self.top_bar = ttk.Frame(self, padding=(6,4))
        self.max_btn = ttk.Button(self.top_bar, text="🔳 Maximize", width=10, command=self._toggle_maximize)
        self.reset_btn = ttk.Button(self.top_bar, text="Reset", command=self._reset)
        self.copy_btn = ttk.Button(self.top_bar, text="Copy values", command=self._copy_values_to_clipboard)

        # Main content frame
        self.main = ttk.Frame(self, padding=8)

        # Left: sliders
        self.controls = ttk.LabelFrame(self.main, text="Levers", padding=10)

        self.lev1 = tk.DoubleVar(value=50.0)   # 0..100
        self.lev2 = tk.DoubleVar(value=128.0)  # 0..255
        self.lev3 = tk.DoubleVar(value=0.5)    # 0..1

        # Use explicit length so they're obvious
        self.s1_label = ttk.Label(self.controls, text="Power (0–100)")
        self.s1 = ttk.Scale(self.controls, from_=0, to=100, orient="horizontal",
                            variable=self.lev1, command=lambda e: self._on_change(), length=320)

        self.s2_label = ttk.Label(self.controls, text="Color (0–255)")
        self.s2 = ttk.Scale(self.controls, from_=0, to=255, orient="horizontal",
                            variable=self.lev2, command=lambda e: self._on_change(), length=320)

        self.s3_label = ttk.Label(self.controls, text="Transparency (0.0–1.0)")
        self.s3 = ttk.Scale(self.controls, from_=0.0, to=1.0, orient="horizontal",
                            variable=self.lev3, command=lambda e: self._on_change(), length=320)

        # Right: text and preview
        self.preview_frame = ttk.LabelFrame(self.main, text="Preview & Values", padding=10)
        self.text = tk.Text(self.preview_frame, width=36, height=12, wrap="word")
        self.text.configure(state="disabled")
        self.canvas = tk.Canvas(self.preview_frame, width=300, height=300, bd=1, relief="solid")

    def _layout_widgets(self):
        # Top bar layout
        self.top_bar.pack(fill="x")
        self.max_btn.pack(side="left", padx=(2,6))
        self.reset_btn.pack(side="left")
        self.copy_btn.pack(side="left", padx=(6,2))

        # Main area
        self.main.pack(fill="both", expand=True)

        # Left controls and right preview use grid for robust resizing
        self.controls.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        self.preview_frame.grid(row=0, column=1, sticky="nsew")

        # Configure grid weights so preview expands
        self.main.columnconfigure(0, weight=0)
        self.main.columnconfigure(1, weight=1)
        self.main.rowconfigure(0, weight=1)

        # Pack sliders inside controls using grid
        pad_y = 6
        self.s1_label.grid(row=0, column=0, sticky="w")
        self.s1.grid(row=1, column=0, sticky="ew", pady=(2, pad_y))
        self.s2_label.grid(row=2, column=0, sticky="w")
        self.s2.grid(row=3, column=0, sticky="ew", pady=(2, pad_y))
        self.s3_label.grid(row=4, column=0, sticky="w")
        self.s3.grid(row=5, column=0, sticky="ew", pady=(2, pad_y))

        # Expand scales horizontally
        self.controls.columnconfigure(0, weight=1)

        # Preview layout inside preview_frame
        self.text.grid(row=0, column=0, sticky="n", padx=(0,8))
        self.canvas.grid(row=0, column=1, sticky="n")
        self.preview_frame.columnconfigure(1, weight=1)

    def _on_change(self):
        self._update_text()
        self._update_preview()

    def _update_text(self):
        power = self.lev1.get()
        color_val = int(self.lev2.get())
        alpha = round(self.lev3.get(), 2)
        text_lines = [
            f"Power: {power:.1f}",
            f"Color value: {color_val}",
            f"Transparency: {alpha:.2f}",
            "",
            "Log:",
            f"- Updated sliders (Power={power:.1f}, Color={color_val}, Trans={alpha:.2f})"
        ]
        content = "\n".join(text_lines)
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")

    def _update_preview(self):
        self.canvas.delete("all")
        # Map color_val (0..255) to RGB gradient blue->red
        color_val = int(self.lev2.get())
        r = int((color_val / 255.0) * 255)
        g = 50
        b = 255 - r
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        power = self.lev1.get() / 100.0  # normalize
        alpha = self.lev3.get()

        base = 260
        w = int(base * (0.35 + 0.65 * power))
        h = int(base * (0.35 + 0.65 * power))
        canvas_w = int(self.canvas["width"])
        canvas_h = int(self.canvas["height"])
        x0 = (canvas_w - w) // 2
        y0 = (canvas_h - h) // 2
        x1 = x0 + w
        y1 = y0 + h

        # background
        self.canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill="#f2f2f2", outline="")

        # main rectangle
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=hex_color,
                                     outline="#000000", width=max(1, int(alpha * 8)))

        # simulate transparency with a white overlay rectangle when alpha < 1
        if alpha < 1.0:
            overlay_int = int((1.0 - alpha) * 200)  # 0..200
            ov_color = f"#{overlay_int:02x}{overlay_int:02x}{overlay_int:02x}"
            self.canvas.create_rectangle(x0+8, y0+8, x1-8, y1-8, fill=ov_color, outline="")

    def _reset(self):
        self.lev1.set(50.0)
        self.lev2.set(128.0)
        self.lev3.set(0.5)
        self._on_change()

    def _copy_values_to_clipboard(self):
        values = f"Power={self.lev1.get():.1f}, Color={int(self.lev2.get())}, Trans={self.lev3.get():.2f}"
        try:
            self.clipboard_clear()
            self.clipboard_append(values)
            # feedback in text
            self.text.configure(state="normal")
            self.text.insert("end", f"\n- Copied to clipboard: {values}")
            self.text.configure(state="disabled")
        except tk.TclError:
            # clipboard may not be available in some headless environments
            pass

    def _toggle_maximize(self):
        """
        Toggle fullscreen as a reliable cross-platform 'maximize'.
        On some platforms, state('zoomed') works; we try both behaviors.
        """
        if not self._is_fullscreen:
            # Try fullscreen attribute
            try:
                self.attributes("-fullscreen", True)
            except Exception:
                # fallback to maximized state (Windows)
                try:
                    self.state("zoomed")
                except Exception:
                    pass
            self.max_btn.config(text="🗗 Restore")
            self._is_fullscreen = True
        else:
            # restore
            try:
                self.attributes("-fullscreen", False)
            except Exception:
                try:
                    self.state("normal")
                except Exception:
                    pass
            self.max_btn.config(text="🔳 Maximize")
            self._is_fullscreen = False

def create_rask_model_renderings(demo_part):
    df = pd.read_csv(ROOT + f"/../E1/metrics_{demo_part}.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    start = df['timestamp'].min()
    end = df['timestamp'].max()
    logger.info(f"Iterating from {start} to {end} by 1s steps")

    current = start
    iteration = 0
    rask_models_store = {}  # if you want to capture models per step

    df_train = pd.read_csv(ROOT + "/../E1/metrics_TRAIN.csv") if demo_part == "OPERATE" else pd.DataFrame()
    df_train['timestamp'] = pd.to_datetime(df['timestamp'])

    rask = RASK(show_figures=True)
    while current <= end:
        filter_df = df[df['timestamp'] <= current].copy()
        merge_df = pd.concat([df_train, filter_df], axis=0, ignore_index=True)

        if len(merge_df) >= 2:
            try:
                rask.init_models(merge_df, f"{demo_part}_{iteration}")
            except Exception as e:
                logger.exception(f"Training failed at {current}: {e}")
        else:
            logger.debug(f"At time {current} not enough rows ({len(merge_df)}) to train")

        iteration += 1
        logger.debug(f"Finished iteration {iteration} after {current} time in df")
        current = current + timedelta(seconds=10)

    logger.info(f"Completed {iteration} iterations from {start} to {end}")
    return rask_models_store

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # show DEBUG and above
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # agent_utils.delete_folder_if_exists(ROOT + "/rask_plots")
    # create_rask_model_renderings("TRAIN")
    create_rask_model_renderings("OPERATE")

    # app = LeverGUI()
    # app.mainloop()
