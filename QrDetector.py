import concurrent.futures
import logging
import os
import threading
import time
from multiprocessing import Lock, Process

import cv2
from prometheus_client import start_http_server, Gauge
from pyzbar.pyzbar import decode
from torch.distributed.elastic.multiprocessing import start_processes

import utils
from VehicleService import VehicleService
from VideoReader import VideoReader

DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")
logger = logging.getLogger("multiscale")

start_http_server(8000)
fps_average = Gauge('fps', 'Current processing FPS', ['service_id'])
in_time_fuzzy = Gauge('in_time_fuzzy', 'Fuzzy SLO fulfillment', ['service_id'])

frame_count = 0

class QrDetector(VehicleService):
    def __init__(self, show_results=False):
        super().__init__()
        self._running = False
        self.service_conf = {'pixel': 800, 'fps': 20}
        self.NUMBER_THREADS = 2
        self.fps = utils.FPS_(calculate_avg=5)

        self.webcam_stream = VideoReader(stream_id=0)  # stream_id = 0 is for primary camera
        self.webcam_stream.start()

        # self.device_metric_reporter = DeviceMetricReporter(leader_ip, gpu_available=False)
        # self.service_metric_reporter = ServiceMetricReporter("QR")

        self.show_result = show_results

    def process_one_iteration(self, params) -> None:
        target_height, source_fps = int(params['pixel']), int(params['fps'])

        # available_time_frame = (1000 / source_fps)
        original_frame = self.webcam_stream.read()

        if original_frame is None:
            pass

        original_width, original_height = original_frame.shape[1], original_frame.shape[0]
        ratio = original_height / target_height

        frame = cv2.resize(original_frame, (int(original_width / ratio), int(original_height / ratio)))

        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray)
        combined_img = utils.highlight_qr_codes(frame, decoded_objects)

        if self.show_result:
            cv2.imshow("Detected Objects", combined_img)

        self.fps.tick()
        processing_time = (time.time() - start_time) * 1000.0
        pixel = combined_img.shape[0]

        # service_blanket = self.service_metric_reporter.create_metrics(processing_time, source_fps, pixel=pixel)
        # device_blanket = self.device_metric_reporter.create_metrics()
        # merged_metrics = utils.merge_single_dicts(service_blanket["metrics"], device_blanket["metrics"])

        current_fps = self.fps.get_average()
        fps_average.labels(service_id="video").set(current_fps)
        in_time_fuzzy.labels(service_id="video").set(max(1, current_fps) / source_fps)

        # return processing_time

    def process_loop(self):
        global frame_count

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.NUMBER_THREADS) as executor:
            while self._running:
                print(f"{frame_count}| Started Frame ")
                # executor.map(self.process_one_iteration, self.service_conf)
                # TODO: The problem is somewhere here, because it does only read one frame at once
                future = executor.map(self.wait_task, range(1))
                # result = future.result()
                print(f"{frame_count}| Stopped Frame ")
                frame_count += 1
        logger.info("QR Detector stopped")

    def wait_task(self):
        time.sleep(0.5)

    def start_process(self):
        self._running = True
        processing_thread = threading.Thread(target=self.process_loop, daemon=True)
        processing_thread.start()
        logger.info("QR Detector started")

    def terminate(self):
        self._running = False

    def change_config(self, config):
        self.service_conf = config
        logger.info(f"QR Detector changed to {config}")


if __name__ == '__main__':
    qd = QrDetector(show_results=False)
    qd._running = True
    qd.process_loop()