import logging
import os
import time
from typing import Any

from agent.es_registry import ServiceType
from iot_services.IoTService import IoTService
from iot_services.KittiReader import KittiReader
from lidar_utils import parse_tracklets, fuse_pointclouds, point_cloud_to_bev, draw_bev_box

logger = logging.getLogger("multiscale")

ROOT = os.path.dirname(__file__)
PC_DISTANCE_DEFAULT = 30
PC_FUSION_DEFAULT = 1
PC_PARALLELISM_DEFAULT = 1

class PcVisualizer(IoTService):

    def __init__(self, store_to_csv=True):
        super().__init__()
        self.service_conf = {'data_quality': PC_DISTANCE_DEFAULT, 'model_size': PC_FUSION_DEFAULT,
                             'parallelism': PC_PARALLELISM_DEFAULT}
        self.store_to_csv = store_to_csv
        self.service_type = ServiceType.PC
        self.data_stream = KittiReader(ROOT + "/data", "2011_09_26", "0001")
        self.fusion_buffer = []

    def get_service_parallelism(self) -> int:
        return self.service_conf['parallelism']

    def process_one_iteration(self, frame) -> (Any, int):
        start = time.perf_counter()

        tracklets = parse_tracklets(ROOT + "/data/2011_09_26/tracklet_labels.xml")

        self.fusion_buffer.append(frame)
        if len(self.fusion_buffer) > self.service_conf['model_size']:
            self.fusion_buffer.pop(0)

        fused_points = fuse_pointclouds(self.fusion_buffer)
        bev = point_cloud_to_bev(fused_points, self.service_conf['data_quality'])

        # Overlay 3D boxes
        i = self.data_stream.get_current_index()
        for obj in tracklets:
            if i < obj["first_frame"] or i - obj["first_frame"] >= len(obj["poses"]):
                continue
            pose = obj["poses"][i - obj["first_frame"]]
            draw_bev_box(bev, pose, obj["size"], color=(0, 0, 255), max_dist=self.service_conf['data_quality'])

        # cv2.imshow("LIDAR BEV with Fused Frames", bev)
        # if cv2.waitKey(10) == 27:
        #     pass
        # time.sleep(0.1)

        duration = (time.perf_counter() - start) * 1000
        return bev, duration


if __name__ == '__main__':
    qd = PcVisualizer(store_to_csv=True)
    qd.client_arrivals = {'C1': 40, 'C2': 30}
    qd.start_process()

    while True:
        time.sleep(1000)
