import datetime
import logging
import threading
import time
from typing import Dict

from prometheus_client import start_http_server, Gauge

import utils
from DockerClient import DockerClient
from RedisClient import RedisClient
from agent.ES_Registry import EsType, ES_Registry, ServiceID

logger = logging.getLogger("multiscale")

CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")
REDIS_INSTANCE = utils.get_env_param("REDIS_INSTANCE", "localhost")


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
        self.container_ip = self.docker_client.get_container_ip(self.docker_container_ref)
        self.flag_metric_cooldown = EsType.STARTUP  # Start with flag

        start_http_server(8000)  # Last time I tried to get rid of the metric_id I had problems when querying the data
        self.prom_throughput = Gauge('throughput', 'Actual throughput', ['service_type', 'container_id', 'metric_id'])
        self.prom_avg_p_latency = Gauge('avg_p_latency', 'Processing latency / item',
                                        ['service_type', 'container_id', 'metric_id'])
        self.prom_quality = Gauge('quality', 'Current configured quality',
                                  ['service_type', 'container_id', 'metric_id'])
        self.prom_cores = Gauge('cores', 'Current configured cores', ['service_type', 'container_id', 'metric_id'])
        self.prom_model_size = Gauge('model_size', 'Current model size', ['service_type', 'container_id', 'metric_id'])

    def process_one_iteration(self, params, frame) -> None:
        pass

    def start_process(self):
        self._terminated = False
        self._running = True

        processing_thread = threading.Thread(target=self.process_loop, daemon=True)
        processing_thread.start()
        logger.info(f"{self.service_type} started with {self.service_conf}")

    def terminate(self):
        self._running = False

    def is_running(self):
        return self._running

    def process_loop(self):
        pass

    def change_config(self, config):
        self.service_conf = config
        self.reinitialize_models()
        logger.info(f"{self.service_type} changed to {config}")

    def reinitialize_models(self):
        pass

    # I'm always between calling this threads and cores, but it's the number of cores and I choose the threads
    # according to that. I think this is best to keep the abstract structure of the services
    def vertical_scaling(self, c_cores):
        self.set_flag_and_cooldown(EsType.RESOURCE_SCALE)
        self.terminate()
        # Wait until it is really terminated and then start new
        while not self._terminated:
            time.sleep(0.01)

        self.cores_reserved = c_cores
        self.start_process()
        logger.info(f"{self.service_type} set to {c_cores} cores")

    def change_request_arrival(self, client_id: str, client_rps: int):
        if client_rps <= 0:
            self.client_arrivals[client_id] = 0  # Should be able to delete this??
            del self.client_arrivals[client_id]
            logger.info(f"Removed client {client_id} from service {self.service_type}")
        else:
            self.client_arrivals[client_id] = client_rps
            logger.info(f"Client {client_id} changed RPS to {client_rps}")

        self.redis_client.store_assignment(self.get_service_id(), self.client_arrivals)
        logger.info(f"Total RPS is now {utils.to_absolut_rps(self.client_arrivals)}")

    def has_processing_timeout(self, start_time):
        time_elapsed = int((time.perf_counter() - start_time) * 1000)
        return time_elapsed >= self.processing_timeframe

    def simulate_interval(self, start_time):
        time_elapsed = int((time.perf_counter() - start_time) * 1000)
        if time_elapsed < self.processing_timeframe:
            time.sleep((self.processing_timeframe - time_elapsed) / 1000)

    def get_service_id(self):
        return ServiceID(self.container_ip, self.service_type, self.docker_container_ref)

    def set_flag_and_cooldown(self, es_type: EsType):
        self.flag_metric_cooldown = self.es_registry.get_ES_cooldown(self.service_type, es_type)
        self.redis_client.store_cooldown(self.get_service_id(), es_type, self.flag_metric_cooldown)
