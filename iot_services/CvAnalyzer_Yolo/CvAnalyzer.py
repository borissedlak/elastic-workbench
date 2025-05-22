import concurrent.futures
import logging
import os
import time
from typing import Any

import cv2
import numpy as np

import utils
from agent.ES_Registry import ServiceType
from iot_services.CvAnalyzer_Yolo.YOLOv8_ONNX import YOLOv8
from iot_services.IoTService import IoTService
from iot_services.VideoReader import VideoReader
from video_utils import yolo_model_sizes, draw_detections

logger = logging.getLogger("multiscale")

CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")
ROOT = os.path.dirname(__file__)


# WRITE: Show the impact of resources on throughput and that this is heavily penalized
class CvAnalyzer(IoTService):
    def __init__(self, store_to_csv=True):
        super().__init__(store_to_csv)
        self.service_conf = {'quality': 720, 'model_size': 1}
        self.service_type = ServiceType.CV
        self.video_stream = VideoReader(ROOT + "/data/CV_Video.mp4")

        self.detector: YOLOv8 = None
        self.metric_buffer = []

    def reinitialize_models(self):  # Assumes that service_conf changed in the background
        model_path = ROOT + f"/models/yolov8{yolo_model_sizes[self.service_conf['model_size']]}.onnx"
        self.detector = YOLOv8(model_path, conf_threshold=0.3)

    def process_one_iteration(self, frame) -> (Any, int):
        start = time.perf_counter()

        target_height = int(self.service_conf['quality'])
        original_width, original_height = frame.shape[1], frame.shape[0]
        ratio = original_height / target_height
        resized_frame = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))

        # detector_index = int(threading.current_thread().name.split("_")[1])
        class_ids, boxes, confidences = self.detector(resized_frame)
        combined_img = draw_detections(resized_frame, boxes, confidences, class_ids)

        # Resulting image and total processing time
        duration = (time.perf_counter() - start) * 1000
        return combined_img, duration

    def process_loop(self):
        self.reinitialize_models()  # Place here so that it reloads when model size is changed

        while self._running:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                start_time = time.perf_counter()
                buffer = self.video_stream.get_batch(utils.to_absolut_rps(self.client_arrivals))
                future_dict = {executor.submit(self.process_one_iteration, frame): frame for frame in buffer}

                processed_item_counter = 0
                processed_item_durations = []
                for future in concurrent.futures.as_completed(future_dict):
                    processed_item_durations.append(np.abs(future.result()[1]))
                    processed_item_counter += 1

                    # cv2.imshow("Detected Objects", future.result()[0])
                    # # Press key q to stop
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     self.terminate()

                    if self.has_processing_timeout(start_time):
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

            # This is only executed once after the batch is processed
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
