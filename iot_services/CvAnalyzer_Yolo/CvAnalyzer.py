import concurrent.futures
import logging
import os
import time
from typing import Any

import cv2
import numpy as np

import utils
from agent.es_registry import ServiceType
from iot_services.CvAnalyzer_Yolo.YOLOv8_ONNX import YOLOv8
from iot_services.IoTService import IoTService
from iot_services.VideoReader import VideoReader
from video_utils import yolo_model_sizes, draw_detections

logger = logging.getLogger("multiscale")

CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")
ROOT = os.path.dirname(__file__)
CV_QUALITY_DEFAULT = 256
CV_M_SIZE_DEFAULT = 3

# WRITE: Show the impact of resources on throughput and that this is heavily penalized
class CvAnalyzer(IoTService):
    def __init__(self, store_to_csv=True):
        super().__init__(store_to_csv)
        self.service_conf = {'quality': CV_QUALITY_DEFAULT, 'model_size': CV_M_SIZE_DEFAULT}
        self.service_type = ServiceType.CV
        self.video_stream = VideoReader(ROOT + "/data/CV_Video.mp4")

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

    def process_one_iteration(self, frame) -> (Any, int):
        start = time.perf_counter()

        target_height = int(self.service_conf['quality'])
        original_width, original_height = frame.shape[1], frame.shape[0]
        ratio = original_height / target_height
        resized_frame = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))

        # detector_index = int(threading.current_thread().name.split("_")[1])
        class_ids, boxes, confidences = self.detectors[self.service_conf['model_size']](resized_frame)
        combined_img = draw_detections(resized_frame, boxes, confidences, class_ids)

        # Resulting image and total processing time
        duration = (time.perf_counter() - start) * 1000
        return combined_img, duration

    def process_loop(self):
        while self._running:

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            start_time = time.perf_counter()
            processed_item_counter = 0
            processed_item_durations = []

            try:
                buffer = self.video_stream.get_batch(utils.to_absolut_rps(self.client_arrivals))
                future_dict = {executor.submit(self.process_one_iteration, frame): frame for frame in buffer}

                while future_dict:
                    done, _ = concurrent.futures.wait(
                        future_dict,
                        timeout=0.015,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        result = future.result()
                        processed_item_durations.append(np.abs(result[1]))
                        processed_item_counter += 1
                        del future_dict[future]

                        # Optionally display or process result
                        # cv2.imshow("Detected Objects", result[0])
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     self.terminate()

                    if self.has_processing_timeout(start_time):
                        # LL: When I run the shutdown in a "with Executor" section, it's still blocking
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
            finally:
                self.export_processing_metrics(processed_item_counter, processed_item_durations)
                if self.simulate_arrival_interval:
                    self.simulate_interval(start_time)

        self._terminated = True
        logger.info(f"{self.service_type.value} stopped")


if __name__ == '__main__':
    qd = CvAnalyzer(store_to_csv=False)
    qd.client_arrivals = {'C3': 10}
    qd.start_process()

    while qd.is_running():
        time.sleep(0.1)
