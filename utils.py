import logging
import os
import sys
import time

import cv2
import numpy as np


def get_ENV_PARAM(var, DEFAULT) -> str:
    ENV = os.environ.get(var)
    if ENV:
        logging.info(f'Found ENV value for {var}: {ENV}')
    else:
        ENV = DEFAULT
        logging.warning(f"Didn't find ENV value for {var}, default to: {DEFAULT}")
    return ENV


# def get_local_ip():
#     interfaces = netifaces.interfaces()
#     for interface in interfaces:
#         ifaddresses = netifaces.ifaddresses(interface)
#         if netifaces.AF_INET in ifaddresses:
#             for addr in ifaddresses[netifaces.AF_INET]:
#                 ip = addr.get('addr')
#                 if ip and ip.startswith('192.168'):
#                     return ip
#     return None

def highlight_qr_codes(frame, decoded_objects):
    for obj in decoded_objects:
        points = obj.polygon
        if len(points) == 4:
            pts = np.array(points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 4)

        qr_data = obj.data.decode('utf-8')
        qr_type = obj.type
        text = f"{qr_type}: {qr_data}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000.0
        print(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        return result

    return wrapper


class FPS_:
    def __init__(self, calculate_avg=1, display_total=True):
        # self.output_string = output_string
        self.display_total = display_total
        self.prev_time = 0
        self.new_time = 0

        if calculate_avg < 1:
            raise ValueError("Average must be calculated over value 1 at least")
        self.store = Cyclical_Array(calculate_avg)

    def tick(self) -> None:
        self.new_time = time.time()
        dif = self.new_time - self.prev_time

        if dif != 0:
            self.store.put(1 / dif)
            self.prev_time = self.new_time
            # msg = "\r" + self.output_string + "%d" % self.store.get_average()
            # if self.display_total:
            #     msg += ", last total %.3fs" % dif
            # print(msg)
            # sys.stdout.write(msg)
            # sys.stdout.flush()

        # return dif

    def get_average(self) -> int:
        return self.store.get_average()


class Cyclical_Array:
    def __init__(self, size):
        self.data = np.zeros(size, dtype=object)
        self.index = 0
        self.size = size

    def put(self, item):
        self.data[self.index % self.size] = item
        self.index = self.index + 1

    def get_average(self):
        return np.mean(self.data)
