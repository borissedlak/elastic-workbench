import concurrent.futures
import datetime
import logging
import threading
import time

import cv2
from prometheus_client import start_http_server, Gauge
from pyzbar.pyzbar import decode

import utils
from VideoReader import VideoReader
from iot_services.IoTService import IoTService

logger = logging.getLogger("multiscale")

start_http_server(8000)
fps = Gauge('fps', 'Current processing FPS', ['service_id', 'metric_id'])
pixel = Gauge('pixel', 'Current configured pixel', ['service_id', 'metric_id'])
energy = Gauge('energy', 'Current processing energy', ['service_id', 'metric_id'])
cores = Gauge('cores', 'Current configured cores', ['service_id', 'metric_id'])

docker_stats = None


class QrDetector(IoTService):
    def __init__(self, store_to_csv=True):
        super().__init__()
        self.service_conf = {'pixel': 800}
        self.cores = 10
        # self.thread_multiplier = 4
        self.number_threads = self.cores  # * self.thread_multiplier
        self.fps = utils.FPS_()
        self.store_to_csv = store_to_csv

        self.simulate_arrival_interval = True
        self.available_timeframe = 1000  # ms
        self.batch_size = 200
        self.video_stream = VideoReader()
        # self.webcam_stream.start()
        self.flag_next_metrics = False

    def process_one_iteration(self, config_params, frame) -> None:

        target_height = int(config_params['pixel'])

        original_width, original_height = frame.shape[1], frame.shape[0]
        ratio = original_height / target_height
        frame = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))

        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray)

        # Resulting image and total processing time --> unused
        combined_img = utils.highlight_qr_codes(frame, decoded_objects)
        processing_time = (time.time() - start_time) * 1000.0

    def process_loop(self):
        metric_buffer = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.number_threads) as executor:
            while self._running:

                start_time = datetime.datetime.now()
                buffer = self.video_stream.get_batch(self.batch_size)
                future_dict = {executor.submit(self.process_one_iteration, self.service_conf, frame): frame
                               for frame in buffer}

                for future in concurrent.futures.as_completed(future_dict):
                    number = future_dict[future]
                    try:
                        result = future.result()  # Does not even return anything!
                        self.fps.tick()
                    except Exception as e:
                        print(f"Error occurred while fetching {number}: {e}")

                # This is only executed once, and not for every frame
                processing_fps = self.fps.get_current_fps()
                fps.labels(service_id=self.service_id, metric_id="fps").set(processing_fps)
                pixel.labels(service_id=self.service_id, metric_id="pixel").set(self.service_conf['pixel'])
                cores.labels(service_id=self.service_id, metric_id="cores").set(self.cores)

                cpu_load = 0
                # cpu_load = utils.calculate_cpu_percentage(docker_stats)
                energy.labels(service_id=self.service_id, metric_id="energy").set(cpu_load)

                if self.store_to_csv:
                    metric_buffer.append(
                        (datetime.datetime.now(), processing_fps, self.service_conf['pixel'], self.cores,
                         cpu_load, self.flag_next_metrics))
                    # self.flag_next_metrics = False
                    if len(metric_buffer) >= 15:
                        utils.write_metrics_to_csv(metric_buffer)
                        metric_buffer.clear()

                if self.simulate_arrival_interval:
                    self.simulate_interval(start_time)

        self._terminated = True
        logger.info("QR Detector stopped")

    def start_process(self):
        self._terminated = False
        self._running = True

        processing_thread = threading.Thread(target=self.process_loop, daemon=True)
        processing_thread.start()
        logger.info("QR Detector started")

    def terminate(self):
        self._running = False

    def change_config(self, config):
        self.service_conf = config
        self.flag_next_metrics = True
        logger.info(f"QR Detector changed to {config}")

    # TODO: Takes too long with 106ms
    @utils.print_execution_time
    def vertical_scaling(self, c_threads):
        self.terminate()
        # Wait until it is really terminated and then start new
        while not self._terminated:
            time.sleep(0.01)

        self.cores = c_threads
        logger.info(f"QR Detector set to {c_threads} threads")
        self.start_process()

    def simulate_interval(self, start_time):
        overall_time = int((datetime.datetime.now() - start_time).microseconds / 1000)
        if overall_time < self.available_timeframe:
            # print(overall_time, self.available_timeframe)
            time.sleep((self.available_timeframe - overall_time) / 1000)


if __name__ == '__main__':
    qd = QrDetector(store_to_csv=False)
    qd.start_process()

    while True:
        time.sleep(1000)
