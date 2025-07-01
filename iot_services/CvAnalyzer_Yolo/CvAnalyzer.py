import logging
import os
import time
from typing import Any

import cv2

from agent.es_registry import ServiceType
from iot_services.CvAnalyzer_Yolo.YOLOv8_ONNX import YOLOv8
from iot_services.IoTService import IoTService
from iot_services.VideoReader import VideoReader
from video_utils import yolo_model_sizes, draw_detections

logger = logging.getLogger("multiscale")

ROOT = os.path.dirname(__file__)
CV_DATA_QUALITY_DEFAULT = 224
CV_M_SIZE_DEFAULT = 3


# WRITE: Show the impact of resources on throughput and that this is heavily penalized
class CvAnalyzer(IoTService):
    def __init__(self, store_to_csv=True):
        super().__init__(store_to_csv)
        self.service_conf = {'data_quality': CV_DATA_QUALITY_DEFAULT, 'model_size': CV_M_SIZE_DEFAULT}
        self.service_type = ServiceType.CV
        self.data_stream = VideoReader(ROOT + "/data/CV_Video.mp4")

        self.detectors: dict[int, YOLOv8] = {}
        self.load_models()
        self.metric_buffer = []

    # @utils.print_execution_time
    def load_models(self):
        for i in range(1, 6):
            # logger.info(f"Loading Detector with Yolov8{yolo_model_sizes[self.service_conf['model_size']]}")
            model_path = ROOT + f"/models/yolov8{yolo_model_sizes[i]}.onnx"
            detector = YOLOv8(model_path, conf_threshold=0.3)
            self.detectors[i] = detector

    def get_service_parallelism(self) -> int:
        return 1

    def process_one_iteration(self, frame) -> (Any, int):
        start = time.perf_counter()

        target_height = int(self.service_conf['data_quality'])
        original_width, original_height = frame.shape[1], frame.shape[0]
        ratio = original_height / target_height
        resized_frame = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))

        # detector_index = int(threading.current_thread().name.split("_")[1])
        class_ids, boxes, confidences = self.detectors[self.service_conf['model_size']](resized_frame)
        combined_img = draw_detections(resized_frame, boxes, confidences, class_ids)

        # Resulting image and total processing time
        duration = (time.perf_counter() - start) * 1000
        return combined_img, duration


if __name__ == '__main__':
    qd = CvAnalyzer(store_to_csv=False)
    qd.client_arrivals = {'C3': 10}
    qd.start_process()

    while qd.is_running():
        time.sleep(0.1)
