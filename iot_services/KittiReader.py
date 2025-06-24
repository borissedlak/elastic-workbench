import itertools

import pykitti

from iot_services.IoTService import DataReader


class KittiReader(DataReader):
    def __init__(self, base_path='data', date='2011_09_26', drive='0001', buffer_size=100):
        super().__init__(buffer_size)
        self.kitti_data = pykitti.raw(base_path, date, drive)
        self.init_buffer()

    def init_buffer(self):
        self.buffer = list(itertools.islice(self.kitti_data.velo, self.buffer_size))

    # TODO: Get index correctly
    def get_current_index(self):
        return 0
