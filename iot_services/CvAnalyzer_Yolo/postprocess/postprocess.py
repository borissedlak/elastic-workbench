import os

import cv2
import numpy as np
import pandas as pd
import ast

from iot_services.VideoReader import VideoReader
from video_utils import yolo_model_sizes, draw_detections

ROOT = os.path.dirname(__file__)

from iot_services.CvAnalyzer_Yolo.YOLOv10_ONNX import YOLOv10

# Load your csv file
# 1. Load both CSV files
df_explore = pd.read_csv("agent_experience_EXPLORE.csv")
df_exploit = pd.read_csv("agent_experience_OPERATE.csv")

# 2. Concatenate them into one DataFrame
df = pd.concat([df_explore, df_exploit], ignore_index=True)

cv_df = df[df['service'].str.contains('cv-analyzer')].copy()
cv_df['model_size'] = cv_df['state'].apply(lambda x: int(ast.literal_eval(x).get('model_size')))
cv_df['data_quality'] = cv_df['state'].apply(lambda x: int(ast.literal_eval(x).get('data_quality')))
cv_df['throughput'] = cv_df['state'].apply(lambda x: int(ast.literal_eval(x).get('throughput')))

# Optional: Reset index so your first 'cv' row starts at 0
cv_df = cv_df.reset_index(drop=True)

# The first value is only present for a single frame, so I have to shift all to the front by one
cv_df_backup = cv_df.copy()
for i in range(0, 30):
    cv_df.iloc[i] = cv_df.iloc[i + 1]

print(cv_df[['timestamp', 'service', 'model_size', 'data_quality']].head())

print(len(cv_df))

detectors: dict[int, YOLOv10] = {}

for i in range(1, 6):
    model_path = f"../models/yolov10{yolo_model_sizes[i]}.onnx"
    detector = YOLOv10(model_path, conf_threshold=0.7)
    detectors[i] = detector

ACCELERATION_FACTOR = 2
data_stream = VideoReader("../data/CV_Video_race.mp4")
video_batch = data_stream.get_batch(600 * ACCELERATION_FACTOR)

for timestep in range(0, 600):

    agent_cycle = int(np.floor(timestep / 10))
    agent_experience = cv_df.iloc[agent_cycle]
    model_size_index = agent_experience['model_size']

    frame = video_batch[timestep * ACCELERATION_FACTOR]

    target_height = int(agent_experience['data_quality'])
    original_width, original_height = frame.shape[1], frame.shape[0]
    ratio = original_height / target_height
    resized_frame = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))

    class_ids, boxes, confidences = detectors[model_size_index](resized_frame)
    combined_img = draw_detections(resized_frame, boxes, confidences, class_ids)

    print(boxes)

    if combined_img is not None:
        filename = f"../../../../website-new/static/percom-demo-2026/service_output/elastic-workbench-cv-analyzer/{timestep}.jpg"
        cv2.imwrite(filename, combined_img)
        print(f"Write image {filename}")
        pass
