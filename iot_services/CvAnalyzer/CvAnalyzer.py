import concurrent.futures
import datetime
import logging
import os
import threading
import time
from typing import Any

import numpy as np

import utils
from agent.ES_Registry import ServiceType
from iot_services.CvAnalyzer.VideoReader import VideoReader
from iot_services.CvAnalyzer.YOLOv10 import YOLOv10
from iot_services.IoTService import IoTService
from video_utils import draw_detections, yolo_model_sizes

logger = logging.getLogger("multiscale")

CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")
ROOT = os.path.dirname(__file__)


class CvAnalyzer(IoTService):
    def __init__(self, store_to_csv=True):
        super().__init__()
        self.service_conf = {'quality': 800, 'model_size': 1}
        self.store_to_csv = store_to_csv
        self.service_type = ServiceType.CV
        self.video_stream = VideoReader()

        self.detectors = {}
        self.metric_buffer = []

    def reinitialize_models(self):  # Assumes that service_conf changed in the background
        model_path = ROOT + f"/models/yolov10{yolo_model_sizes[self.service_conf['model_size']]}.onnx"
        for i in range(0, self.cores_reserved):
            self.detectors[i] = YOLOv10(model_path, conf_thres=0.3)

    def process_one_iteration(self, config_params, frame) -> (Any, int):
        start = time.time()

        detector_index = int(threading.current_thread().name.split("_")[1])
        class_ids, boxes, confidences = self.detectors[detector_index](frame)
        combined_img = draw_detections(frame, boxes, confidences, class_ids)

        # model_path = ROOT + f"/models/yolov10{yolo_model_sizes[self.service_conf['model_size']]}.onnx"
        # detector = YOLOv10(model_path, conf_thres=0.3)
        # detector(frame)
        # combined_img = None

        # Resulting image and total processing time --> unused
        duration = (time.time() - start) * 1000
        return combined_img, duration

    def process_loop(self):
        self.reinitialize_models()  # Place here so that it reloads when cores are changed

        while self._running:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.cores_reserved) as executor:
                start_time = time.perf_counter()
                buffer = self.video_stream.get_batch(utils.to_absolut_rps(self.client_arrivals))
                future_dict = {executor.submit(self.process_one_iteration, self.service_conf, frame): frame
                               for frame in buffer}

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

            logger.info(processed_item_durations)
            logger.info(processed_item_counter)

            # This is only executed once after the batch is processed
            self.prom_throughput.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                                        metric_id="throughput").set(processed_item_counter)
            avg_p_latency_v = int(np.mean(processed_item_durations)) if processed_item_counter > 0 else -1
            self.prom_avg_p_latency.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                                           metric_id="avg_p_latency").set(avg_p_latency_v)
            self.prom_quality.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                                     metric_id="quality").set(self.service_conf['quality'])
            self.prom_cores.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                                   metric_id="cores").set(self.cores_reserved)

            if self.store_to_csv:
                self.metric_buffer.append(
                    (datetime.datetime.now(), self.service_type.value, CONTAINER_REF, avg_p_latency_v,
                     self.service_conf, self.cores_reserved, self.flag_metric_cooldown))
                self.flag_metric_cooldown = 0
                utils.write_metrics_to_csv(self.metric_buffer)
                self.metric_buffer.clear()

            if self.simulate_arrival_interval:
                self.simulate_interval(start_time)

        self._terminated = True
        logger.info(f"{self.service_type.value} stopped")


if __name__ == '__main__':
    qd = CvAnalyzer(store_to_csv=True)
    qd.client_arrivals = {'C3': 10}
    qd.start_process()

    while qd.is_running():
        time.sleep(0.1)
