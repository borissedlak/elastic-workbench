import logging

import docker
from typing import NamedTuple

import utils


logger = logging.getLogger("multiscale")
logging.getLogger("multiscale").setLevel(logging.INFO)

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")


class DockerClient:
    def __init__(self):
        self.client = docker.DockerClient(base_url=DOCKER_SOCKET)

    # TODO: Takes too long with 90ms
    @utils.print_execution_time
    def update_cpu(self, container_ref, cpus):
        try:
            container = self.client.containers.get(container_ref)
            container.update(cpu_quota=cpus * 100000)
            logger.info(f"Container set to work with {cpus} cores")
        except Exception as e:
            logger.error("Could not connect to docker container", e)

    @utils.print_execution_time
    def get_container_stats(self, container_ref, stream_p=False):
        try:
            container = self.client.containers.get(container_ref)
            stats = container.stats(stream=stream_p, decode=stream_p)
            return stats
        except Exception as e:
            logger.error("Could not connect to docker container", e)


class DockerInfo(NamedTuple):
    container_id: str
    ip_a: str
    alias: str

if __name__ == "__main__":
    client = DockerClient()
    # client.update_cpu("67959d3ff81a", 5)
    stream = client.get_container_stats("elastic-workbench-video-processing-a-1", stream_p=True)

    for s in stream:
        # print(utils.calculate_cpu_percentage(s))
        print(1)
    # print(utils.calculate_cpu_percentage(client.get_container_stats("multiscaler-video-processing-1")))
