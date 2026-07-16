import logging
import os
import time
from typing import Any, Tuple

import cv2

from agent.components.es_registry import ServiceType
from iot_services.CvAnalyzer_Yolo.YOLOv10_ONNX import YOLOv10
from iot_services.IoTService import IoTService
from iot_services.VideoReader import VideoReader
from video_utils import draw_detections
from video_utils import yolo_model_sizes

logger = logging.getLogger("multiscale")

ROOT = os.path.dirname(__file__)
CV_DATA_QUALITY_DEFAULT = 224
CV_M_SIZE_DEFAULT = 3
CV_PARALLELISM_DEFAULT = 1

# ==========================================
# WORKER INITIALIZER (Executes once per process startup)
# ==========================================
# Keeps a single process-local global model reference
_active_detector: YOLOv10 | None = None


def init_single_model(root_dir: str, model_size_idx: int):
    """Executes on worker processes to load the ONNX model."""
    global _active_detector

    model_path = os.path.join(root_dir, f"models/yolov10{yolo_model_sizes[model_size_idx]}.onnx")
    logger.info(f"[Worker PID {os.getpid()}] Loading model: yolov10{yolo_model_sizes[model_size_idx]}")
    _active_detector = YOLOv10(model_path, conf_threshold=0.3)


class CvAnalyzer(IoTService):
    def __init__(self, store_to_csv=True):
        super().__init__(store_to_csv)
        self.service_conf = {'data_quality': CV_DATA_QUALITY_DEFAULT, 'model_size': CV_M_SIZE_DEFAULT,
                             'parallelism': CV_PARALLELISM_DEFAULT}
        self.service_type = ServiceType.CV
        self.data_stream = VideoReader(ROOT + "/data/CV_Video.mp4")

        self.detectors: dict[int, YOLOv10] = {}
        self.metric_buffer = []

    def get_model_size(self) -> int:
        return self.service_conf['model_size']

    def get_service_parallelism(self) -> int:
        return self.service_conf['parallelism']

    def get_executor_initializer(self) -> tuple:
        # Pass the global setup function and its dynamic args to the base loop
        return init_single_model, (ROOT, self.get_model_size())

    def preprocess_buffer_items(self, buffer, data_quality):
        # Scale frames on the parent process before writing them to OS Pipes
        target_height = int(data_quality)
        preprocessed = []
        for frame in buffer:
            original_width, original_height = frame.shape[1], frame.shape[0]
            ratio = original_height / target_height
            resized = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))
            preprocessed.append(resized)
        return preprocessed

    @staticmethod
    def process_one_iteration(frame) -> Tuple[Any, float]:
        """Processes the frame using the single global detector inside this worker process."""
        start = time.perf_counter()
        global _active_detector

        if _active_detector is None:
            raise RuntimeError("Process model was not initialized correctly!")

        # Perform inference on the pre-resized frame
        class_ids, boxes, confidences = _active_detector(frame)
        combined_img = draw_detections(frame, boxes, confidences, class_ids)

        duration = (time.perf_counter() - start) * 1000
        return combined_img, duration

    def write_result_to_sink(self, result, timestep):
        directory = ROOT + f"/../../share/service_output/{self.service_type.value}"

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        if result is not None:
            filename = f"{directory}/{timestep}.jpg"
            cv2.imwrite(filename, result)
            print(f"Write image {filename}")
            pass


if __name__ == '__main__':
    qd = CvAnalyzer(store_to_csv=False)
    qd.client_arrivals = {'C3': 100}
    qd.start_process()

    while qd.is_running():
        time.sleep(0.1)
