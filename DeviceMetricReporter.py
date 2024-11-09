import threading
from datetime import datetime

import GPUtil
import numpy as np
import psutil
import pymongo

from consumption.ConsRegression import ConsRegression
from utils import get_ENV_PARAM, DB_NAME, is_jetson_host

DEVICE_NAME = get_ENV_PARAM('DEVICE_NAME', "Unknown")
# LEADER_HOST = get_ENV_PARAM('LEADER_HOST', "localhost")

GPU_AVG_HISTORY_LENGTH = 50  # This is not really a hyperparameter but rather a dirty fix that is needed


class CyclicArray:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def append(self, value):
        if len(self.buffer) < self.max_size:
            self.buffer.append(value)
        else:
            self.buffer.pop(0)
            self.buffer.append(value)

    def already_x_values(self, x=None):
        if x is None:
            x = self.max_size
        return len(self.buffer) >= x

    def get_percentage_filled(self):
        return round(len(self.buffer) / self.max_size, 2)

    def get_number_items(self):
        return len(self.buffer)

    def is_empty(self):
        return len(self.buffer) == 0

    def get(self):
        return self.buffer

    def average(self):
        if not self.buffer:
            return 0
        return np.average(self.buffer)

    def arithmetic_mean(self):
        if not self.buffer:
            return 0
        return np.median(self.buffer)

    def clear(self):
        self.buffer = []


class DeviceMetricReporter:
    def __init__(self, mongo_host, gpu_available, gpu_avg_history_n=GPU_AVG_HISTORY_LENGTH):
        self.consumption_regression = None
        self.mongo_client = None
        self.gpu_available = gpu_available
        self.gpu_avg_history = None
        if mongo_host is not None:
            self.mongo_client = pymongo.MongoClient(mongo_host)[DB_NAME]

        if gpu_avg_history_n > 0:
            self.gpu_avg_history = CyclicArray(gpu_avg_history_n)

        if is_jetson_host(DEVICE_NAME):
            from jtop.jtop import jtop
            self.jetson_metrics = jtop()
            self.jetson_metrics.start_video_processing()
        else:
            self.consumption_regression = ConsRegression(DEVICE_NAME)

    def create_metrics(self):
        mem_buffer = psutil.virtual_memory()
        mem = (mem_buffer.total - mem_buffer.available) / mem_buffer.total * 100
        cpu = psutil.cpu_percent()

        gpu = 0
        if is_jetson_host(DEVICE_NAME):  # Has Jetson lib defined
            gpu = self.jetson_metrics.stats['GPU']
            cons = self.jetson_metrics.stats['Power TOT'] / 1000
            jetson_mode = self.jetson_metrics.stats['nvp model']

            if self.gpu_avg_history is not None:
                self.gpu_avg_history.append(gpu)
                gpu = self.gpu_avg_history.average()
        elif self.gpu_available:
            cons = self.consumption_regression.predict(cpu, self.gpu_available)
            if len(GPUtil.getGPUs()) > 0:  # Has Nvidia GPU but is no Jetson
                gpu = int(GPUtil.getGPUs()[0].load * 100)
            else:
                raise RuntimeError("How come?")
        else:
            cons = self.consumption_regression.predict(cpu, self.gpu_available)

        return {"target": DEVICE_NAME,
                "metrics": {"device_type": DEVICE_NAME, "cpu": cpu, "memory": mem, "consumption": cons,
                            "timestamp": datetime.now(), "gpu": gpu}}

    # @utils.print_execution_time
    def report_metrics(self, target, record):
        insert_thread = threading.Thread(target=self.run_detached, args=(target, record))
        insert_thread.start()

    def run_detached(self, target, record):
        self.mongo_client[target].insert_one(record)
