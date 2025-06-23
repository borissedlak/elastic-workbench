import itertools

import cv2
import pykitti

from iot_services.IoTService import DataReader


class KittiReader(DataReader):
    def __init__(self, base_path='data', date='2011_09_26', drive='0001', buffer_size=100):

        self.kitti_data = pykitti.raw(base_path, date, drive)
        self.buffer_size = buffer_size
        self.buffer = []
        self.init_buffer()

    def init_buffer(self):
        self.buffer = list(itertools.islice(self.kitti_data.velo, 100))

    # @utils.print_execution_time
    def get_batch(self, batch_size):
        full_repeats = batch_size // self.buffer_size
        remainder = batch_size % self.buffer_size

        return (self.buffer * full_repeats) + self.buffer[:remainder]

    # @utils.print_execution_time
    def get_current_index(self):
        return 0
