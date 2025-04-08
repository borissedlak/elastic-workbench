import logging
import time
from threading import Thread

from DockerClient import DockerClient
from HttpClient import HttpClient
from PrometheusClient import PrometheusClient
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
                self.es_registry.ES_random_execution("localhost", 'elastic-workbench-video-processing',
                                                     'resource_scaling')

            time.sleep(30)


if __name__ == '__main__':
    ps = "http://172.19.0.1:9090"
    ScalingAgent(services_local=["elastic-workbench-video-processing-1"], prom_server=ps).start()
