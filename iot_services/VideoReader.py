import cv2

from iot_services.IoTService import DataReader


class VideoReader(DataReader):
    def __init__(self, path, buffer_size=200):

        self.video_path = path
        self.buffer_size = buffer_size
        self.buffer = []

        self.vcap = cv2.VideoCapture(self.video_path)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)  # self.stopped is set to False when frames are being read from self.vcap stream

        self.init_buffer()

    def init_buffer(self):
        self.buffer = []
        for _ in range(self.buffer_size):
            self.grabbed, self.frame = self.vcap.read()
            self.buffer.append(self.frame)

    # @utils.print_execution_time
    def get_batch(self, batch_size):
        full_repeats = batch_size // self.buffer_size
        remainder = batch_size % self.buffer_size

        return (self.buffer * full_repeats) + self.buffer[:remainder]
