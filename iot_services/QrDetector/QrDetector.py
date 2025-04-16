import concurrent.futures
import datetime
import logging
import time
from typing import Any

import cv2
import numpy as np
from prometheus_client import start_http_server, Gauge
from pyzbar.pyzbar import decode

import utils
from agent.ES_Registry import ServiceType
from iot_services.IoTService import IoTService
from iot_services.QrDetector.VideoReader import VideoReader

logger = logging.getLogger("multiscale")

CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")

start_http_server(8000)
throughput = Gauge('throughput', 'Actual throughput', ['service_type', 'container_id', 'metric_id'])
avg_p_latency = Gauge('avg_p_latency', 'Processing latency / item',
                      ['service_type', 'container_id', 'metric_id'])
quality = Gauge('quality', 'Current configured quality', ['service_type', 'container_id', 'metric_id'])
# energy = Gauge('energy', 'Current processing energy', ['service_id', 'container_id', 'metric_id'])
cores = Gauge('cores', 'Current configured cores', ['service_type', 'container_id', 'metric_id'])


class QrDetector(IoTService):
    def __init__(self, store_to_csv=True):
        super().__init__()
        self.service_conf = {'quality': 800}
        self.store_to_csv = store_to_csv
        self.service_type = ServiceType.QR
        self.video_stream = VideoReader()

    def process_one_iteration(self, config_params, frame) -> (Any, int):
        start = datetime.datetime.now()

        target_height = int(config_params['quality'])
        original_width, original_height = frame.shape[1], frame.shape[0]
        ratio = original_height / target_height
        frame = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray)

        # Resulting image and total processing time --> unused
        combined_img = utils.highlight_qr_codes(frame, decoded_objects)
        duration = (datetime.datetime.now() - start).total_seconds() * 1000
        return combined_img, duration

    def process_loop(self):
        metric_buffer = []

        while self._running:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.cores_reserved) as executor:
                start_time = datetime.datetime.now()
                buffer = self.video_stream.get_batch(utils.to_absolut_rps(self.client_arrivals))
                future_dict = {executor.submit(self.process_one_iteration, self.service_conf, frame): frame
                               for frame in buffer}

                processed_item_counter = 0
                processed_item_durations = []
                for future in concurrent.futures.as_completed(future_dict):
                    processed_item_durations.append(future.result()[1])
                    processed_item_counter += 1

                    if self.has_processing_timeout(start_time):
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

            # This is only executed once after the batch is processed
            throughput.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                              metric_id="throughput").set(processed_item_counter)
            avg_p_latency_v = int(np.mean(processed_item_durations)) if processed_item_counter > 0 else -1
            avg_p_latency.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                                 metric_id="avg_p_latency").set(avg_p_latency_v)
            quality.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                           metric_id="quality").set(self.service_conf['quality'])
            cores.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                         metric_id="cores").set(self.cores_reserved)

            if self.store_to_csv:
                metric_buffer.append((datetime.datetime.now(), self.service_type.value, CONTAINER_REF, avg_p_latency_v,
                                      self.service_conf, self.cores_reserved, self.flag_metric_cooldown))
                self.flag_metric_cooldown = 0
                utils.write_metrics_to_csv(metric_buffer)
                metric_buffer.clear()

            if self.simulate_arrival_interval:
                self.simulate_interval(start_time)

        self._terminated = True
        logger.info(f"{self.service_type.value} stopped")


if __name__ == '__main__':
    qd = QrDetector(store_to_csv=False)
    qd.client_arrivals = {'C1': 20}
    qd.start_process()

    while True:
        time.sleep(1000)
