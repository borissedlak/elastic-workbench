import logging

import docker

import utils

logger = logging.getLogger("multiscale")

DOCKER_PREFIX = utils.get_ENV_PARAM('DOCKER_PREFIX', "unix://")
DOCKER_SOCKET_PATH = utils.get_ENV_PARAM('DOCKER_SOCKET_PATH', "/var/run/docker.sock")

class DockerClient:
    def __init__(self, url):
        self.client = docker.DockerClient(base_url=url)

    @utils.print_execution_time
    def update_cpu(self, container_id, cpus):
        try:
            container = self.client.containers.get(container_id)
            container.update(cpu_quota=cpus * 100000)
            logger.info(f"Container set to work with {cpus} cores")
        except Exception as e:
            logger.error("Could not connect to docker container", e)

    # @utils.print_execution_time # Takes 6 ms
    def get_container_id(self, container_name="multiscaler-video-processing-1"):
        container = self.client.containers.list(filters={'name': container_name})

        if container:
            return str(container[0].id)[:12]
        else:
            return "Unknown"


if __name__ == "__main__":
    client = DockerClient(DOCKER_PREFIX + DOCKER_SOCKET_PATH)
    # client.update_cpu("67959d3ff81a", 5)
    print(client.get_container_id())