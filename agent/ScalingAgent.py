import datetime
import logging
import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import Dict, List

import utils
from HttpClient import HttpClient
from PrometheusClient import PrometheusClient
from RedisClient import RedisClient
from agent.agent_utils import wait_for_remaining_interval
from agent.components.RASK import calculate_missing_vars
from agent.components.SLORegistry import SLO_Registry, calculate_SLO_F_clients
from agent.components.es_registry import ESRegistry, ServiceID, ServiceType, ESType

CV_DATA_QUALITY_DEFAULT = 224
CV_M_SIZE_DEFAULT = 3
QR_DATA_QUALITY_DEFAULT = 550
PC_DISTANCE_DEFAULT= 30

logger = logging.getLogger("multiscale")

MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))
EVALUATION_CYCLE_DELAY = int(utils.get_env_param('EVALUATION_CYCLE_DELAY', 5))
SERVICE_HOST = utils.get_env_param('SERVICE_HOST', "localhost")



class ScalingAgent(Thread, ABC):
    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path, es_registry_path, log_experience):

        super().__init__()
        self._running = True
        self._idle = False
        self.evaluation_cycle = evaluation_cycle
        self.log_experience = log_experience

        self.services_monitored: list[ServiceID] = services_monitored
        self.prom_client = PrometheusClient(prom_server)
        # self.docker_client = DockerClient()
        self.http_client = HttpClient()
        self.reddis_client = RedisClient(SERVICE_HOST)
        self.slo_registry = SLO_Registry(slo_registry_path)
        self.es_registry = ESRegistry(es_registry_path)
        self.experience_buffer = []
        # This is needed because the Prom becomes unavailable (i.e., deletes metrics) while we load a new model
        # It's also not possible to load the model in a new Process because the onnxruntime cannot be pickled
        self.last_known_state = {}
        self.last_iteration_length = -1

    def resolve_service_state(self, service_id: ServiceID, assigned_clients: Dict[str, int]):
        """
        Queries the basic state from Prometheus and then extends it with the calculated metrics.

        :param service_id:
        :param assigned_clients:
        :return: Full service state; otherwise empty dict
        """
        metric_values = self.prom_client.get_metrics(["avg_p_latency", "throughput"], service_id, period="5s")
        parameter_ass = self.prom_client.get_metrics(["data_quality", "cores", "model_size"], service_id)

        if parameter_ass == {} or metric_values == {}:
            logger.warning(f"No metrics found for service {service_id}")  # Remove if never happens
            return self.last_known_state[service_id.service_type.value] if service_id.service_type.value in self.last_known_state else {}

        missing_vars = calculate_missing_vars(metric_values, utils.to_absolut_rps(assigned_clients))
        full_state = metric_values | parameter_ass | missing_vars

        self.last_known_state[service_id.service_type.value] = full_state
        return full_state

    def run(self):
        while self._running:
            start_time = time.perf_counter()
            self.orchestrate_services_optimally(self.services_monitored)

            agent_iteration_time = int((time.perf_counter() - start_time) * 1000)
            self.last_iteration_length = agent_iteration_time

            wait_for_remaining_interval(self.evaluation_cycle, start_time)

    def execute_ES(self, service: ServiceID, es_type: ESType, params, respect_cooldown=True):

        if respect_cooldown and self.reddis_client.is_under_cooldown(service):
            warning_msg = f"Service <{service.host, service.container_id}> is under cooldown, cannot call ES"
            logger.warning(warning_msg)
            return

        if not self.es_registry.is_es_supported(service.service_type, es_type):
            logger.warning(f"Trying to call unsupported ES for {service.service_type}, {es_type}")
            return

        ES_endpoint = self.es_registry.get_es_information(service.service_type, es_type)['endpoint']

        self.http_client.call_ES_endpoint(service, ES_endpoint, params)
        logger.info(f"Calling ES <{service.service_type},{es_type}> with {params}")

    @abstractmethod
    def orchestrate_services_optimally(self, services_m: List[ServiceID]):
        """Primary hook of an Agent's gameloop

        Most logic is coupled and tailored for the specific use cases. Subclasses may only implement logic to interct
        with the hardcoded services (QR and CV)

        :param services_m:
        :return:
        """
        pass

    # def get_free_cores(self) -> int:
    #     cores_ass = self.get_core_assignment(self.services_monitored)
    #     free_cores = MAX_CORES - sum(cores_ass.values())
    #     return free_cores

    # def get_max_available_cores(self, service: ServiceID) -> int:
    #     cores_ass = self.get_core_assignment(self.services_monitored)
    #     free_cores = MAX_CORES - sum(cores_ass.values())
    #     max_available_c = free_cores + cores_ass[service.container_id]
    #     return max_available_c

    # def get_core_assignment(self, service_list: list[ServiceID]) -> Dict[str, int]:
    #     cores_per_service = {}
    #
    #     for service_id in service_list:
    #         s_cores = self.docker_client.get_container_cores(service_id.container_id)
    #         cores_per_service[service_id.container_id] = s_cores
    #     return cores_per_service

    # Between the experiments, we need to reset the processing environment to a default state
    def reset_services_states(self, mute_first=None):

        if mute_first:
            for service_m in mute_first:
                self.execute_ES(service_m, ESType.RESOURCE_SCALE, {'cores': 0}, respect_cooldown=False)
                self.http_client.call_ES_endpoint(service_m, "/stop_processing", {})

        for service_m in self.services_monitored:  # For all monitored services

            if mute_first:
                self.http_client.call_ES_endpoint(service_m, "/start_processing", {})

            if service_m.service_type == ServiceType.QR:
                self.execute_ES(service_m, ESType.RESOURCE_SCALE, {'cores': MAX_CORES / 3}, respect_cooldown=False)
                self.execute_ES(service_m, ESType.QUALITY_SCALE, {'data_quality': QR_DATA_QUALITY_DEFAULT}, respect_cooldown=False)
            elif service_m.service_type == ServiceType.CV:
                self.execute_ES(service_m, ESType.RESOURCE_SCALE, {'cores': MAX_CORES / 3}, respect_cooldown=False)
                self.execute_ES(service_m, ESType.QUALITY_SCALE, {'data_quality': CV_DATA_QUALITY_DEFAULT}, respect_cooldown=False)
                self.execute_ES(service_m, ESType.MODEL_SCALE, {'model_size': CV_M_SIZE_DEFAULT}, respect_cooldown=False)
            elif service_m.service_type == ServiceType.PC:
                self.execute_ES(service_m, ESType.RESOURCE_SCALE, {'cores': MAX_CORES / 3}, respect_cooldown=False)
                self.execute_ES(service_m, ESType.QUALITY_SCALE, {'data_quality': PC_DISTANCE_DEFAULT}, respect_cooldown=False)
            else:
                raise RuntimeError("Not supported yet")

    def terminate_gracefully(self):
        self._running = False

    def evaluate_slos_and_buffer(self, service_m: ServiceID, service_state, slos_all_clients):
        slo_f = calculate_SLO_F_clients(service_m.service_type, service_state, slos_all_clients)
        self.experience_buffer.append((service_m, datetime.datetime.now(), slo_f, self.log_experience,
                                       service_state, self.last_iteration_length))
