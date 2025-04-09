import datetime
import logging
import threading
import time

import utils
from DockerClient import DockerClient
from agent.ES_Registry import EsType, ES_Registry

logger = logging.getLogger("multiscale")

# DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")
CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")


class IoTService:
    def __init__(self):
        self.docker_container_ref = CONTAINER_REF
        self.service_type = "Empty"
        self._terminated = True
        self._running = False
        self.service_conf = {}
        self.cores_reserved = 2
        self.es_registry = ES_Registry()

        self.simulate_arrival_interval = True
        self.processing_timeframe = 1000  # ms
        self.batch_size = 200

        self.docker_client = DockerClient()
        self.flag_next_metrics = EsType.STARTUP  # Start with flag

    def process_one_iteration(self, params, frame) -> None:
        pass

    def start_process(self):
        self._terminated = False
        self._running = True

        processing_thread = threading.Thread(target=self.process_loop, daemon=True)
        processing_thread.start()
        logger.info(f"{self.service_type} started")

    def terminate(self):
        self._running = False

    def process_loop(self):
        pass

    def change_config(self, config):
        self.service_conf = config
        self.flag_next_metrics = EsType.QUALITY_S # TODO: Actually this is not 100% accurate here
        logger.info(f"{self.service_type} changed to {config}")

    # I'm always between calling this threads and cores, but it's the number of cores and I choose the threads
    # according to that. I think this is best to keep the abstract structure of the services
    def vertical_scaling(self, c_cores):
        self.terminate()
        # Wait until it is really terminated and then start new
        while not self._terminated:
            time.sleep(0.01)

        self.cores_reserved = c_cores
        self.start_process()
        self.flag_next_metrics = EsType.RESOURCE_S
        logger.info(f"{self.service_type} set to {c_cores} cores")

    def change_request_arrival(self, rps_arriving):
        self.batch_size = rps_arriving
        logger.info(f"{self.service_type} changed RPS to {rps_arriving}")

    def has_processing_timeout(self, start_time):
        time_elapsed = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        return time_elapsed >= self.processing_timeframe

    def simulate_interval(self, start_time):
        time_elapsed = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        if time_elapsed < self.processing_timeframe:
            time.sleep((self.processing_timeframe - time_elapsed) / 1000)
