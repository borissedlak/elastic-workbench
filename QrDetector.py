import concurrent.futures
import datetime
import logging
import threading
import time

import cv2
from prometheus_client import start_http_server, Gauge
from pyzbar.pyzbar import decode

import utils
from DockerClient import DockerClient
from VehicleService import VehicleService
from VideoReader import VideoReader

DEVICE_NAME = utils.get_env_param("DEVICE_NAME", "Unknown")
DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")
logger = logging.getLogger("multiscale")

start_http_server(8000)
fps = Gauge('fps', 'Current processing FPS', ['service_id', 'metric_id'])
pixel = Gauge('pixel', 'Current configured pixel', ['service_id', 'metric_id'])
energy = Gauge('energy', 'Current processing energy', ['service_id', 'metric_id'])
cores = Gauge('cores', 'Current configured cores', ['service_id', 'metric_id'])

docker_stats = None


class QrDetector(VehicleService):
    def __init__(self):
        super().__init__()
        self._terminated = True
        self._running = False
        self.service_conf = {'pixel': 800}
        self.cores = 2
        self.thread_multiplier = 4
        self.number_threads = self.cores * self.thread_multiplier
        self.fps = utils.FPS_()

        self.webcam_stream = VideoReader()
        self.webcam_stream.start()
        self.flag_next_metrics = False
        self.docker_client = DockerClient(DOCKER_SOCKET)
        self.stats_stream = self.docker_client.get_container_stats("multiscaler-video-processing-1", stream=True)

        threading.Thread(target=resolve_docker_load, args=(self.stats_stream,), daemon=True).start()

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

                buffer = self.webcam_stream.get_buffer_size_n(self.number_threads)
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
                fps.labels(service_id="video", metric_id="fps").set(processing_fps)
                pixel.labels(service_id="video", metric_id="pixel").set(self.service_conf['pixel'])
                cores.labels(service_id="video", metric_id="cores").set(self.cores)

                try:
                    cpu_load = utils.calculate_cpu_percentage(docker_stats)
                except KeyError as e:
                    logger.warning(f"Cannot load in first iteration, setting to 0 for now; {e.args}")
                    cpu_load = 0
                energy.labels(service_id="video", metric_id="energy").set(cpu_load)


                metric_buffer.append((datetime.datetime.now(), processing_fps, self.service_conf['pixel'], self.cores,
                                      self.flag_next_metrics))
                self.flag_next_metrics = False
                if len(metric_buffer) >= 15:
                    utils.write_metrics_to_csv(metric_buffer)
                    metric_buffer.clear()

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
    # @utils.print_execution_time
    def change_threads(self, c_threads):
        self.terminate()
        # Wait until it is really terminated and then start new
        while not self._terminated:
            time.sleep(0.01)

        self.cores = c_threads
        logger.info(f"QR Detector set to {c_threads} threads")
        self.start_process()

# TODO: Must check the additional load to the thread
def resolve_docker_load(stream_object):
    global docker_stats
    for stats in stream_object:
        docker_stats = stats


if __name__ == '__main__':
    qd = QrDetector()
    qd.start_process()

    while True:
        time.sleep(1000)
