import logging
import platform
import time
from threading import Thread

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
    def __init__(self, prom_server, services_monitored: [ServiceID]):
        super().__init__()
        self._running = True
        self._idle = False

        self.services_monitored = services_monitored
        self.prom_client = PrometheusClient(prom_server)
        self.docker_client = DockerClient()
        self.http_client = HttpClient()
        self.es_registry = ES_Registry()
        self.reddis_client = RedisClient()
        self.slo_registry = SLO_Registry()

    def resolve_service_state(self, service_id: ServiceID):
        metric_values = self.prom_client.get_metrics(["avg_p_latency", "throughput"], service_id, period="10s")
        parameter_ass = self.prom_client.get_metrics(["pixel", "cores"], service_id)
        return metric_values | parameter_ass

    def run(self):

        while self._running:
            for service_m in self.services_monitored:
                service_m: ServiceID = service_m
                current_state = self.resolve_service_state(service_m)

                if current_state == {}:
                    logger.warning(f"Cannot find state for service {service_m}")
                    continue

                logger.info(f"Current state for {service_m}: {current_state}")
                # TODO: Maybe I can create a function from that?
                assigned_clients = self.reddis_client.get_assignments_for_service(service_m)
                for client_id, client_rps in assigned_clients.items():

                    client_SLOs = self.slo_registry.get_SLOs_for_client(client_id, service_m.service_type)
                    if client_SLOs == {}:
                        logger.warning(f"Cannot find SLOs for service {service_m}, client {client_id}")
                        continue

                    client_SLO_F = self.slo_registry.calculate_slo_reward(current_state, client_SLOs)
                    print(client_SLO_F)

                host_fix = "localhost" if platform.system() == "Windows" else service_m.host
                self.execute_random_ES(host_fix, service_m.service_type)

            time.sleep(30)

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
