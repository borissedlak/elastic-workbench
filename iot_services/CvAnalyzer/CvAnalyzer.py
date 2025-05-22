import datetime
import logging
import os
import time
from typing import Any

import numpy as np

import utils
from agent.ES_Registry import ServiceType
from iot_services.CvAnalyzer.FaceDetector import FaceDetector
from iot_services.IoTService import IoTService
from iot_services.VideoReader import VideoReader
from video_utils import fd_model_sizes

logger = logging.getLogger("multiscale")

CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")
ROOT = os.path.dirname(__file__)


class CvAnalyzer(IoTService):
    def __init__(self, store_to_csv=True):
        super().__init__()
        self.service_conf = {'model_size': 1}
        self.store_to_csv = store_to_csv
        self.service_type = ServiceType.CV
        self.video_stream = VideoReader(ROOT + "/data/CV_Video.mp4")

        self.detector = None
        self.metric_buffer = []

    def reinitialize_models(self):  # Assumes that service_conf changed in the background
        model_path = ROOT + f"/models/version-RFB-{fd_model_sizes[self.service_conf['model_size']]}.onnx"
        self.detector = FaceDetector(model_path)

    def process_one_iteration(self, config_params, frame) -> (Any, int):
        start = time.perf_counter()

        # detector_index = int(threading.current_thread().name.split("_")[1])
        processed_frame = self.detector.detect_faces(frame)
        # self.model.detect_faces(frame)

        # Resulting image and total processing time --> unused
        duration = (time.perf_counter() - start) * 1000
        return processed_frame, duration

    def process_loop(self):
        self.reinitialize_models()  # Place here so that it reloads when cores are changed

        while self._running:
            # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            start_time = time.perf_counter()
            buffer = self.video_stream.get_batch(utils.to_absolut_rps(self.client_arrivals))
            # future_dict = {executor.submit(self.process_one_iteration, self.service_conf, frame): frame
            #                for frame in buffer}

            processed_item_counter = 0
            processed_item_durations = []
            for frame in buffer:
                result = self.process_one_iteration(self.service_conf, frame)
                processed_item_durations.append(np.abs(result[1]))
                processed_item_counter += 1

                # cv2.imshow("Detected Objects", future.result()[0])
                # # Press key q to stop
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     self.terminate()

                if self.has_processing_timeout(start_time):
                    break

            # This is only executed once after the batch is processed
            self.export_processing_metrics(processed_item_counter, processed_item_durations)

            if self.simulate_arrival_interval:
                self.simulate_interval(start_time)

        self._terminated = True
        logger.info(f"{self.service_type.value} stopped")

    # Since this has a static threadpool, no need to restart. Depending on the 3rd service I might move the method
    # def vertical_scaling(self, c_cores):
    #     self.cores_reserved = c_cores


if __name__ == '__main__':
    qd = CvAnalyzer(store_to_csv=False)
    qd.client_arrivals = {'C3': 100}
    qd.start_process()

    while qd.is_running():
        time.sleep(0.1)
