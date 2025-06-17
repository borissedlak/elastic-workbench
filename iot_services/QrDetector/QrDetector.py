import logging
import os
import time
from typing import Any

import cv2
from pyzbar.pyzbar import decode

import utils
from agent.es_registry import ServiceType
from iot_services.IoTService import IoTService
from iot_services.VideoReader import VideoReader

logger = logging.getLogger("multiscale")

ROOT = os.path.dirname(__file__)
QR_DATA_QUALITY_DEFAULT = 700


class QrDetector(IoTService):

    def __init__(self, store_to_csv=True):
        super().__init__()
        self.service_conf = {'data_quality': QR_DATA_QUALITY_DEFAULT}
        self.store_to_csv = store_to_csv
        self.service_type = ServiceType.QR
        self.data_stream = VideoReader(ROOT + "/data/QR_Video.mp4")

    def get_service_parallelism(self) -> int:
        return utils.cores_to_threads(self.cores_reserved)

    def process_one_iteration(self, frame) -> (Any, int):
        start = time.perf_counter()

        target_height = int(self.service_conf['data_quality'])
        original_width, original_height = frame.shape[1], frame.shape[0]
        ratio = original_height / target_height
        frame = cv2.resize(frame, (int(original_width / ratio), int(original_height / ratio)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray)

        # Resulting image and total processing time --> unused
        combined_img = utils.highlight_qr_codes(frame, decoded_objects)
        duration = (time.perf_counter() - start) * 1000
        return combined_img, duration


if __name__ == '__main__':
    qd = QrDetector(store_to_csv=True)
    qd.client_arrivals = {'C1': 40, 'C2': 30}
    qd.start_process()

    while True:
        time.sleep(1000)
