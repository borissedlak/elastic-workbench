import concurrent.futures
import logging
import os
import threading
import time
from multiprocessing import Lock, Process

import cv2
from prometheus_client import start_http_server, Gauge
from pyzbar.pyzbar import decode

import utils
from VehicleService import VehicleService
from multi import WebcamStream

DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")
logger = logging.getLogger("multiscale")
# ROOT = os.path.dirname(__file__)

start_http_server(8000)
fps_average = Gauge('fps', 'Current processing FPS', ['service_id'])
in_time_fuzzy = Gauge('in_time_fuzzy', 'Fuzzy SLO fulfillment', ['service_id'])

frame_count = 0

class QrDetector(VehicleService):
    def __init__(self, show_results=False):
        super().__init__()
        self._running = False
        # self.video_path = ROOT + "/data/QR_Video.mp4"
        self.simulate_fps = True
        self.service_conf = {'pixel': 800, 'fps': 20}
        self.number_threads = 1
        self.fps = utils.FPS_(calculate_avg=5)


        self.webcam_stream = WebcamStream(stream_id=0)  # stream_id = 0 is for primary camera
        self.webcam_stream.start()

        # self.device_metric_reporter = DeviceMetricReporter(leader_ip, gpu_available=False)
        # self.service_metric_reporter = ServiceMetricReporter("QR")

        self.show_result = show_results
        # self.initialize_video()

        # if not self.cap.isOpened():
        #     print("Error opening video ...")
        #     return

    def process_one_iteration(self, params) -> None:
        target_height, source_fps = int(params['pixel']), int(params['fps'])

        # print(f"Now processing: {params.source_pixel} p, {params.source_fps} FPS")
        available_time_frame = (1000 / source_fps)
        # with self.mutex:
        # print(time.time())
        original_frame = self.webcam_stream.read()
        # if not ret:
        #     self.webcam_stream.reset()
            # output_video.release()
            # sys.exit()

        if params is None:
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
        # output_video.writ(combined_img)

        self.fps.tick()
        print(self.fps.get_average())
        processing_time = (time.time() - start_time) * 1000.0
        pixel = combined_img.shape[0]

        # service_blanket = self.service_metric_reporter.create_metrics(processing_time, source_fps, pixel=pixel)
        # device_blanket = self.device_metric_reporter.create_metrics()
        # merged_metrics = utils.merge_single_dicts(service_blanket["metrics"], device_blanket["metrics"])

        # if self.simulate_fps:
        #     if processing_time < available_time_frame:
        #         time.sleep((available_time_frame - processing_time) / 1000)

        current_fps = self.fps.get_average()
        fps_average.labels(service_id="video").set(current_fps)
        in_time_fuzzy.labels(service_id="video").set(max(1, current_fps) / source_fps)

        # return processing_time

    def process_loop(self):

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.number_threads) as executor:
            while self._running:
                future = executor.submit(self.process_one_iteration, self.service_conf)
                result = future.result()
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
