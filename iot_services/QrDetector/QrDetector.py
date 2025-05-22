import concurrent.futures
import logging
import os
import time
from typing import Any

import cv2
import numpy as np
from pyzbar.pyzbar import decode

import utils
from agent.ES_Registry import ServiceType
from iot_services.IoTService import IoTService
from iot_services.VideoReader import VideoReader

logger = logging.getLogger("multiscale")

ROOT = os.path.dirname(__file__)
CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")


class QrDetector(IoTService):

    def __init__(self, store_to_csv=True):
        super().__init__()
        self.service_conf = {'quality': 720}
        self.store_to_csv = store_to_csv
        self.service_type = ServiceType.QR
        self.video_stream = VideoReader(ROOT + "/data/QR_Video.mp4")

    def process_one_iteration(self, frame) -> (Any, int):
        start = time.perf_counter()

        target_height = int(self.service_conf['quality'])
        original_width, original_height = frame.shape[1], frame.shape[0]
        ratio = original_height / target_height
        frame = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray)

        # Resulting image and total processing time --> unused
        combined_img = utils.highlight_qr_codes(frame, decoded_objects)
        duration = (time.perf_counter() - start) * 1000
        return combined_img, duration

    def process_loop(self):

        while self._running:
            with concurrent.futures.ThreadPoolExecutor(max_workers=utils.cores_to_threads(self.cores_reserved)) as tpex:
                start_time = time.perf_counter()
                buffer = self.video_stream.get_batch(utils.to_absolut_rps(self.client_arrivals))
                future_dict = {tpex.submit(self.process_one_iteration, frame): frame for frame in buffer}

                processed_item_counter = 0
                processed_item_durations = []

                while future_dict:
                    # Poll with a short timeout to allow frequent timeout checks
                    done, _ = concurrent.futures.wait(
                        future_dict,
                        timeout=0.05,  # 50ms polling interval
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        result = future.result()
                        processed_item_durations.append(np.abs(result[1]))
                        processed_item_counter += 1
                        del future_dict[future] # Remove completed futures

                    if self.has_processing_timeout(start_time):
                        tpex.shutdown(wait=False, cancel_futures=True)
                        break

                # for future in concurrent.futures.as_completed(future_dict):
                #     processed_item_durations.append(future.result()[1])
                #     processed_item_counter += 1
                #
                #     if self.has_processing_timeout(start_time):
                #         tpex.shutdown(wait=False, cancel_futures=True)
                #         break

            # This is only executed once after the batch is processed
            self.export_processing_metrics(processed_item_counter, processed_item_durations)

            if self.simulate_arrival_interval:
                self.simulate_interval(start_time)

        self._terminated = True
        logger.info(f"{self.service_type.value} stopped")


if __name__ == '__main__':
    qd = QrDetector(store_to_csv=True)
    qd.client_arrivals = {'C1': 40, 'C2': 30}
    qd.start_process()

    while True:
        time.sleep(1000)
