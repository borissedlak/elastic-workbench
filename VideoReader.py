
import os
import cv2
from threading import Thread

import utils


class VideoReader:
    def __init__(self, buffer_size=200):
        # self.stream_id = stream_id  # default is 0 for primary camera

        # opening video capture stream
        # self.vcap = cv2.VideoCapture(self.stream_id)

        ROOT = os.path.dirname(__file__)
        self.video_path = ROOT + "/data/QR_Video.mp4"
        self.buffer_size = buffer_size
        self.buffer = []
        # self.frame_count = 1
        # self.last_frame_read = False

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
        return self.buffer[:batch_size]

        # self.stopped = True  # reference to the thread for reading next available frame from input stream
        # self.t = Thread(target=self.update, args=())
        # self.t.daemon = True  # daemon threads keep running in the background while the program is executing

    # method for starting the thread for grabbing next available frame in input stream
    # def start(self):
    #     self.stopped = False
    #     self.t.start()  # method for reading next frame

    # def update(self):
    #     while not self.stopped:
    #         if self.last_frame_read:
    #
    #             if self.frame_count >= self.vcap.get(cv2.CAP_PROP_FRAME_COUNT):
    #                 self.frame_count = 1
    #                 self.vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #
    #             self.grabbed, self.frame = self.vcap.read()
    #             self.frame_count += 1
    #
    #             self.last_frame_read = False
    #
    #     self.vcap.release()  # method for returning latest read frame



    # def read(self):
    #     self.last_frame_read = True
    #     return self.frame

    # @utils.print_execution_time # Takes < 1ms, which I can hardly imagine
    # def get_buffer_size_n(self, size):
    #     buffer = []
    #     while len(buffer) < size:
    #         frame = self.read()
    #         buffer.append(frame)
    #
    #     return buffer

    # def stop(self):
    #     self.stopped = True

