import logging
import time
from threading import Thread

from DockerClient import DockerClient
from HttpClient import HttpClient
from PrometheusClient import PrometheusClient
from agent import agent_utils
from agent.ES_Registry import ES_Registry

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)


class ScalingAgent(Thread):
    def __init__(self, prom_server, services_local):
        super().__init__()

        self._running = True
        self._idle = False

        self.services_local = services_local
        self.prom_client = PrometheusClient(prom_server)
        self.docker_client = DockerClient()
        self.http_client = HttpClient()
        self.es_registry = ES_Registry()

    def run(self):

        while self._running:

            # TODO: This will need a better wrapper most likely
            for s_local in self.services_local:
                host_address = self.docker_client.get_container_ip(s_local)  # TODO: Fix only for windows
                self.execute_random_ES(host_address, 'elastic-workbench-video-processing')

            time.sleep(30)

    def execute_random_ES(self, host, service_type):
        rand_ES_name = self.es_registry.get_random_ES_for_service(service_type)

        if not self.es_registry.is_ES_supported(service_type, rand_ES_name):
            return

        ES_endpoints = self.es_registry.get_ES_information(service_type, rand_ES_name)
        for endpoint in ES_endpoints:
            random_params = agent_utils.get_random_parameter_assignments(endpoint['parameters'])
            self.http_client.call_ES_endpoint(host, endpoint['target'], random_params)

            logger.info(f"Calling random ES <{service_type},{rand_ES_name}> with {random_params}")


if __name__ == '__main__':
    # TODO: This should not crash down only because the service is not available
    ps = "http://172.20.0.5:9090"
    ScalingAgent(services_local=["elastic-workbench-video-processing-1"], prom_server=ps).start()
