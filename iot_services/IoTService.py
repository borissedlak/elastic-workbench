import logging
import threading
import time

import utils
from DockerClient import DockerClient

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

        self.docker_client = DockerClient()
        self.flag_next_metrics = False

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
        self.flag_next_metrics = True
        logger.info(f"{self.service_type} changed to {config}")

    # I'm always between calling this threads and cores, but it's the number of cores and I choose the threads
    # according to that. I think this is best to keep the abstract structure of the services
    def vertical_scaling(self, c_cores):
        self.terminate()
        # Wait until it is really terminated and then start new
        while not self._terminated:
            time.sleep(0.01)

        self.cores_reserved = c_cores
        logger.info(f"{self.service_type} set to {c_cores} cores")
        self.start_process()

    def change_request_arrival(self, arrival_number):
        pass
