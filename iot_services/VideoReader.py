import cv2

from iot_services.IoTService import DataReader


class VideoReader(DataReader):
    def __init__(self, path, buffer_size=200):
        super().__init__(buffer_size)
        self.video_path = path

        self.vcap = cv2.VideoCapture(self.video_path)
        if not self.vcap.isOpened():
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if not self.grabbed:
            print('[Exiting] No more frames to read')
            exit(0)  # self.stopped is set to False when frames are being read from self.vcap stream

        self.init_buffer()

    def init_buffer(self):
        self.buffer = []
        for _ in range(self.buffer_size):
            self.grabbed, self.frame = self.vcap.read()
            self.buffer.append(self.frame)

