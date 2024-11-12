import concurrent.futures
import logging
import multiprocessing
import threading
import time

import cv2
from prometheus_client import start_http_server, Gauge
from pyzbar.pyzbar import decode

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
        self.NUMBER_THREADS = 8
        self.fps = utils.FPS_()

        self.webcam_stream = VideoReader(stream_id=0)  # stream_id = 0 is for primary camera
        self.webcam_stream.start()

        # self.device_metric_reporter = DeviceMetricReporter(leader_ip, gpu_available=False)
        # self.service_metric_reporter = ServiceMetricReporter("QR")

        self.show_result = show_results

    def process_one_iteration(self, params, frame) -> None:

        global frame_count
        target_height, source_fps = int(params['pixel']), int(params['fps'])

        original_width, original_height = frame.shape[1], frame.shape[0]
        ratio = original_height / target_height

        frame = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))

        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray)
        combined_img = utils.highlight_qr_codes(frame, decoded_objects)

        if self.show_result:
            cv2.imshow("Detected Objects", combined_img)

        processing_time = (time.time() - start_time) * 1000.0
        # pixel = combined_img.shape[0]

        # service_blanket = self.service_metric_reporter.create_metrics(processing_time, source_fps, pixel=pixel)
        # device_blanket = self.device_metric_reporter.create_metrics()
        # merged_metrics = utils.merge_single_dicts(service_blanket["metrics"], device_blanket["metrics"])

        # frame_count += 1
        # print(frame_count, processing_time)

    def process_loop(self):

        buffer = []

        while len(buffer) < self.NUMBER_THREADS:
            frame = self.webcam_stream.read()
            buffer.append(frame)

        multiprocessing.set_start_method('fork')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.NUMBER_THREADS) as executor:
            while self._running:

                # buffer = []
                #
                # while len(buffer) < self.NUMBER_THREADS:
                #     buffer.append(self.webcam_stream.read())

                future_to_url = {executor.submit(self.process_one_iteration, self.service_conf, frame): frame for frame
                                 in buffer}

                # Process the results as they complete
                for future in concurrent.futures.as_completed(future_to_url):
                    number = future_to_url[future]
                    try:
                        result = future.result()
                        self.fps.tick()
                    except Exception as e:
                        print(f"Error occurred while fetching {number}: {e}")

                current_fps = self.fps.get_fps()
                fps_average.labels(service_id="video").set(current_fps)
                in_time_fuzzy.labels(service_id="video").set(max(1, current_fps) / self.service_conf['fps'])

        logger.info("QR Detector stopped")

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
    qd.start_process()

    # Needed to keep the daemon alive
    while True:
        time.sleep(1000)
    # qd._running = True
    # qd.process_loop()
