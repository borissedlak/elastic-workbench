import logging
import platform
import random
import time
from threading import Thread
from typing import Dict

import utils
from DockerClient import DockerClient
from HttpClient import HttpClient
from PrometheusClient import PrometheusClient
from RedisClient import RedisClient
from agent import agent_utils
from agent.ES_Registry import ES_Registry, ServiceID, ServiceType
from agent.SLO_Registry import SLO_Registry

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)


class ScalingAgent(Thread):
    def __init__(self, prom_server, services_monitored: [ServiceID], evaluation_cycle):
        super().__init__()
        self._running = True
        self._idle = False
        self.evaluation_cycle = evaluation_cycle

        self.services_monitored = services_monitored
        self.prom_client = PrometheusClient(prom_server)
        self.docker_client = DockerClient()
        self.http_client = HttpClient()
        self.es_registry = ES_Registry()
        self.reddis_client = RedisClient()
        self.slo_registry = SLO_Registry()

    def resolve_service_state(self, service_id: ServiceID, assigned_clients: Dict[str, int]):
        metric_values = self.prom_client.get_metrics(["avg_p_latency", "throughput"], service_id, period="10s")
        parameter_ass = self.prom_client.get_metrics(["pixel", "cores"], service_id)

        target_throughput = utils.to_absolut_rps(assigned_clients)
        completion_rate = metric_values['throughput'] / target_throughput * 100
        return metric_values | parameter_ass | {"completion_rate": completion_rate}

    def run(self):

        while self._running:
            for service_m in self.services_monitored:  # For all monitored services
                service_m: ServiceID = service_m
                assigned_clients = self.reddis_client.get_assignments_for_service(service_m)
                service_state = self.resolve_service_state(service_m, assigned_clients)

                if service_state == {}:
                    logger.warning(f"Cannot find state for service {service_m}")
                    continue

                logger.info(f"Current state for <{service_m.host},{service_m.container_id}>: {service_state}")
                self.get_clients_SLO_F(service_m, service_state, assigned_clients)

                host_fix = "localhost" if platform.system() == "Windows" else service_m.host
                if random.randrange(5) == 3:
                    self.execute_random_ES(host_fix, service_m.service_type)

            time.sleep(self.evaluation_cycle)

    def get_clients_SLO_F(self, service_m: ServiceID, service_state, assigned_clients):
        for client_id, client_rps in assigned_clients.items():  # Check the SLO-F of their clients

            # TODO: Calculate overall streaming latency and place into state
            #  Ideally I do this in a function that can also be reused for the expected SLO_F
            client_SLOs = self.slo_registry.get_SLOs_for_client(client_id, service_m.service_type)
            if client_SLOs == {}:
                logger.warning(f"Cannot find SLOs for service {service_m}, client {client_id}")
                continue

            client_SLO_F = self.slo_registry.calculate_slo_reward(service_state, client_SLOs)
            print(client_SLO_F)

    # TODO: This makes the assumption that only the desired container is running at the ip
    def execute_random_ES(self, host, service_type: ServiceType):
        rand_ES = self.es_registry.get_random_ES_for_service(service_type)

        if not self.es_registry.is_ES_supported(service_type, rand_ES):
            return

        ES_endpoints = self.es_registry.get_ES_information(service_type, rand_ES)['endpoints']
        for endpoint in ES_endpoints:
            random_params = agent_utils.get_random_parameter_assignments(endpoint['parameters'])
            self.http_client.call_ES_endpoint(host, endpoint['target'], random_params)

            logger.info(f"Calling random ES <{service_type},{rand_ES}> with {random_params}")
