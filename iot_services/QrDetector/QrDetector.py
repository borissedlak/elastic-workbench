import concurrent.futures
import datetime
import logging
import time

import cv2
from prometheus_client import start_http_server, Gauge
from pyzbar.pyzbar import decode

import utils
from VideoReader import VideoReader
from iot_services.IoTService import IoTService

logger = logging.getLogger("multiscale")

# TODO: Maybe I can somehow abstract the SLOs also for all service types
start_http_server(8000)
fps = Gauge('fps', 'Current processing FPS', ['service_id', 'metric_id'])
pixel = Gauge('pixel', 'Current configured pixel', ['service_id', 'metric_id'])
energy = Gauge('energy', 'Current processing energy', ['service_id', 'metric_id'])
cores = Gauge('cores', 'Current configured cores', ['service_id', 'metric_id'])


class QrDetector(IoTService):
    def __init__(self, store_to_csv=True):
        super().__init__()
        self.service_conf = {'pixel': 800}
        self.store_to_csv = store_to_csv
        self.service_type = "elastic-workbench-video-processing"

        self.simulate_arrival_interval = True
        self.available_timeframe = 1000  # ms
        self.batch_size = 200
        self.video_stream = VideoReader()

    def process_one_iteration(self, config_params, frame) -> None:

        target_height = int(config_params['pixel'])
        original_width, original_height = frame.shape[1], frame.shape[0]
        ratio = original_height / target_height
        frame = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray)

        # Resulting image and total processing time --> unused
        combined_img = utils.highlight_qr_codes(frame, decoded_objects)

    def process_loop(self):
        metric_buffer = []

        while self._running:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.cores_reserved) as executor:
                start_time = datetime.datetime.now()
                buffer = self.video_stream.get_batch(self.batch_size)
                future_dict = {executor.submit(self.process_one_iteration, self.service_conf, frame): frame
                               for frame in buffer}

                processed_item_counter = 0
                for future in concurrent.futures.as_completed(future_dict):
                    if self.has_processing_timeout(start_time):
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    else:
                        processed_item_counter += 1

            # This is only executed once after the batch is processed
            fps.labels(service_id=self.docker_container_ref, metric_id="fps").set(processed_item_counter)
            pixel.labels(service_id=self.docker_container_ref, metric_id="pixel").set(self.service_conf['pixel'])
            cores.labels(service_id=self.docker_container_ref, metric_id="cores").set(self.cores_reserved)

            # if self.store_to_csv:
            #     metric_buffer.append(
            #         (datetime.datetime.now(), processed_item_counter, self.service_conf['pixel'], self.cores_reserved,
            #          0, self.flag_next_metrics))
            #     self.flag_next_metrics = False
            #     if len(metric_buffer) >= 15:
            #         utils.write_metrics_to_csv(metric_buffer)
            #         metric_buffer.clear()

            if self.simulate_arrival_interval:
                self.simulate_interval(start_time)

        self._terminated = True
        logger.info(f"{self.service_type} stopped")

    # TODO: Maybe I can also move this to IoTService.class ?
    def has_processing_timeout(self, start_time):
        time_elapsed = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        return time_elapsed >= self.available_timeframe

    def simulate_interval(self, start_time):
        time_elapsed = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        if time_elapsed < self.available_timeframe:
            time.sleep((self.available_timeframe - time_elapsed) / 1000)


if __name__ == '__main__':
    qd = QrDetector(store_to_csv=False)
    qd.start_process()

    while True:
        time.sleep(1000)
