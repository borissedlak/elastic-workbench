import logging
import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import Dict

import utils
from DockerClient import DockerClient
from HttpClient import HttpClient
from PrometheusClient import PrometheusClient
from RedisClient import RedisClient
from agent.ES_Registry import ES_Registry, ServiceID, ServiceType, EsType
from agent.LGBN import calculate_missing_vars
from agent.SLO_Registry import SLO_Registry, calculate_slo_fulfillment
from agent.agent_utils import log_agent_experience, Full_State

logger = logging.getLogger("multiscale")

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))
EVALUATION_CYCLE_DELAY = int(utils.get_env_param('EVALUATION_CYCLE_DELAY', 5))


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
        self.docker_client = DockerClient()
        self.http_client = HttpClient()
        self.reddis_client = RedisClient()
        self.slo_registry = SLO_Registry(slo_registry_path)
        self.es_registry = ES_Registry(es_registry_path)

    def resolve_service_state(self, service_id: ServiceID, assigned_clients: Dict[str, int]):
        """
        Queries the basic state from Prometheus and then extends it with the calculated metrics.

        :param service_id:
        :param assigned_clients:
        :return: Full service state; otherwise empty dict
        """
        metric_values = self.prom_client.get_metrics(["avg_p_latency", "throughput"], service_id, period="10s")
        parameter_ass = self.prom_client.get_metrics(["quality", "cores", "model_size"], service_id)

        if metric_values == {} and parameter_ass == {}:
            return {}

        missing_vars = calculate_missing_vars(metric_values, utils.to_absolut_rps(assigned_clients))
        return metric_values | parameter_ass | missing_vars

    # WRITE: Add a high-level algorithm of this to the paper
    def run(self):
        while self._running:
            self.orchestrate_services_optimally(self.services_monitored)
            time.sleep(self.evaluation_cycle)

    def get_clients_SLO_F(self, service_m: ServiceID, service_state, assigned_clients):

        all_client_SLO_F = {}
        for client_id, client_rps in assigned_clients.items():  # Check the SLO-F of their clients

            client_SLOs = self.slo_registry.get_SLOs_for_client(client_id, service_m.service_type)
            if client_SLOs == {}:
                logger.warning(f"Cannot find SLOs for service {service_m}, client {client_id}")
                continue

            client_SLO_F_emp = calculate_slo_fulfillment(service_state, client_SLOs)
            all_client_SLO_F[client_id] = client_SLO_F_emp

        # print("Actual SLO-F", all_client_SLO_F)
        return all_client_SLO_F

    # TODO: Remove this host fix as soon as possible
    def execute_ES(self, host, service: ServiceID, es_type: EsType, params, respect_cooldown=True):

        if respect_cooldown and self.reddis_client.is_under_cooldown(service):
            warning_msg = f"Service <{service.host, service.container_id}> is under cooldown, cannot call ES"
            logger.warning(warning_msg)
            return

        if not self.es_registry.is_ES_supported(service.service_type, es_type):
            logger.warning(f"Trying to call unsupported ES for {service.service_type}, {es_type}")
            return

        ES_endpoint = self.es_registry.get_ES_information(service.service_type, es_type)['endpoint']

        self.http_client.call_ES_endpoint(host, ES_endpoint, params)
        logger.info(f"Calling ES <{service.service_type},{es_type}> with {params}")

    @abstractmethod
    def orchestrate_services_optimally(self, services_m):
        pass

    # @abstractmethod
    # def get_optimal_local_ES(self, service: ServiceID, service_state, assigned_clients: Dict[str, int]):
    #     pass

    def get_free_cores(self):
        cores_ass = self.get_core_assignment(self.services_monitored)
        free_cores = PHYSICAL_CORES - sum(cores_ass.values())
        return free_cores

    def get_max_available_cores(self, service: ServiceID):
        cores_ass = self.get_core_assignment(self.services_monitored)
        free_cores = PHYSICAL_CORES - sum(cores_ass.values())
        max_available_c = free_cores + cores_ass[service.container_id]
        return max_available_c

    # @utils.print_execution_time
    def get_core_assignment(self, service_list: list[ServiceID]) -> Dict[str, int]:
        cores_per_service = {}

        for service_id in service_list:
            s_cores = self.docker_client.get_container_cores(service_id.container_id)
            cores_per_service[service_id.container_id] = s_cores
        return cores_per_service

    # Between the experiments, we need to reset the processing environment to a default state
    def reset_services_states(self):
        for service_m in self.services_monitored:  # For all monitored services
            if service_m.service_type == ServiceType.QR:
                self.execute_ES(service_m.host, service_m, EsType.RESOURCE_SCALE, {'cores': 2}, respect_cooldown=False)
                self.execute_ES(service_m.host, service_m, EsType.QUALITY_SCALE, {'quality': 800},
                                respect_cooldown=False)
            elif service_m.service_type == ServiceType.CV:
                self.execute_ES(service_m.host, service_m, EsType.RESOURCE_SCALE, {'cores': 2}, respect_cooldown=False)
                # self.execute_ES(service_m.host, service_m, EsType.QUALITY_SCALE, {'quality': 800}, respect_cooldown=False)
                self.execute_ES(service_m.host, service_m, EsType.MODEL_SCALE, {'model_size': 1},
                                respect_cooldown=False)
            else:
                raise RuntimeError("Not supported yet")

    def terminate_gracefully(self):
        self._running = False

    def build_state_and_log(self, service_state, service_m, assigned_clients):
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service_m.service_type, assigned_clients)
        free_cores = self.get_free_cores()

        # TODO: A bit too cheep here.....
        extra_var = 'quality' if service_m.service_type == ServiceType.QR else 'model_size'

        quality_t, tp_t = all_client_slos[0][extra_var].thresh, all_client_slos[0]['throughput'].thresh
        state_pw = Full_State(service_state[extra_var], quality_t, service_state['throughput'],
                              tp_t, service_state['cores'], free_cores)

        log_agent_experience(state_pw, self.log_experience)
