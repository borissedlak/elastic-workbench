import logging

import docker

import utils

logger = logging.getLogger("multiscale")

DOCKER_PREFIX = utils.get_ENV_PARAM('DOCKER_PREFIX', "unix://")
DOCKER_SOCKET_PATH = utils.get_ENV_PARAM('DOCKER_SOCKET_PATH', "/var/run/docker.sock")

class DockerClient:
    def __init__(self, url):
        print(url)
        self.client = docker.DockerClient(base_url=url)

    def update_cpu(self, container_id, cpus):
        try:
            container = self.client.containers.get(container_id)
            container.update(cpu_quota=cpus * 100000)
        except Exception as e:
            logger.error("Could not connect to docker container", e)
            # print(e)


if __name__ == "__main__":
    client = DockerClient(DOCKER_PREFIX + DOCKER_SOCKET_PATH)
    client.update_cpu("67959d3ff81a", 5)
