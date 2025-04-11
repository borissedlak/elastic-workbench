import datetime
import logging
import threading
import time
from typing import Dict

import utils
from DockerClient import DockerClient
from RedisClient import RedisClient
from agent.ES_Registry import EsType, ES_Registry, ServiceID

logger = logging.getLogger("multiscale")

# DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")
CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")
REDIS_INSTANCE = utils.get_env_param("REDIS_INSTANCE", "localhost")


def to_absolut_rps(client_arrivals: Dict[str, int]) -> int:
    return sum(i for i in client_arrivals.values())


class IoTService:
    def __init__(self):
        self.docker_container_ref = CONTAINER_REF
        self.service_type = None
        self._terminated = True
        self._running = False
        self.service_conf = {}
        self.cores_reserved = 2
        self.es_registry = ES_Registry()

        self.simulate_arrival_interval = True
        self.processing_timeframe = 1000  # ms
        self.client_arrivals: Dict[str, int] = {}

        self.redis_client = RedisClient(host=REDIS_INSTANCE)
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
        self.flag_next_metrics = EsType.QUALITY_S  # TODO: Actually this is not 100% accurate here
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

    def change_request_arrival(self, client_id: str, client_rps: int):
        if client_rps <= 0:
            self.client_arrivals[client_id] = 0
            del self.client_arrivals[client_id]
            logger.info(f"Removed client {client_id} from service {self.service_type}")
        else:
            self.client_arrivals[client_id] = client_rps
            logger.info(f"Client {client_id} changed RPS to {client_rps}")

        container_ip = self.docker_client.get_container_ip(self.docker_container_ref)
        service_id = ServiceID(container_ip, self.service_type, self.docker_container_ref)
        self.redis_client.store_assignment(service_id, self.client_arrivals)
        logger.info(f"Total RPS is now {to_absolut_rps(self.client_arrivals)}")

    def has_processing_timeout(self, start_time):
        time_elapsed = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        return time_elapsed >= self.processing_timeframe

    def simulate_interval(self, start_time):
        time_elapsed = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        if time_elapsed < self.processing_timeframe:
            time.sleep((self.processing_timeframe - time_elapsed) / 1000)
