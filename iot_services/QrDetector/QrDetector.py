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
from iot_services.IoTService import IoTService, to_absolut_rps
from iot_services.QrDetector.VideoReader import VideoReader

logger = logging.getLogger("multiscale")

# TODO: Maybe I can somehow abstract the SLOs also for all service types
#  One step for this could be to call it simply throughput instead of fps
start_http_server(8000)
throughput = Gauge('throughput', 'Actual throughput', ['service_type', 'container_id', 'metric_id'])
avg_proc_latency = Gauge('avg_proc_latency', 'Processing latency / item',
                         ['service_type', 'container_id', 'metric_id'])
pixel = Gauge('pixel', 'Current configured pixel', ['service_type', 'container_id', 'metric_id'])
# energy = Gauge('energy', 'Current processing energy', ['service_id', 'container_id', 'metric_id'])
cores = Gauge('cores', 'Current configured cores', ['service_type', 'container_id', 'metric_id'])


class QrDetector(IoTService):
    def __init__(self, store_to_csv=True):
        super().__init__()
        self.service_conf = {'pixel': 800}
        self.store_to_csv = store_to_csv
        self.service_type = ServiceType.QR
        self.video_stream = VideoReader()

    def process_one_iteration(self, config_params, frame) -> (Any, int):
        start = datetime.datetime.now()

        target_height = int(config_params['pixel'])
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
                buffer = self.video_stream.get_batch(to_absolut_rps(self.client_arrivals))
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
            avg_proc_latency_num = int(np.mean(processed_item_durations)) if processed_item_counter > 0 else -1
            avg_proc_latency.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                                    metric_id="avg_proc_latency").set(avg_proc_latency_num)
            pixel.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                         metric_id="pixel").set(self.service_conf['pixel'])
            cores.labels(container_id=self.docker_container_ref, service_type=self.service_type.value,
                         metric_id="cores").set(self.cores_reserved)

            if self.store_to_csv:
                ES_cooldown = self.es_registry.get_ES_cooldown(self.service_type, self.flag_next_metrics) \
                    if self.flag_next_metrics else 0
                metric_buffer.append((datetime.datetime.now(), self.service_type.value, avg_proc_latency_num,
                                      self.service_conf, self.cores_reserved, ES_cooldown))
                self.flag_next_metrics = None
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
