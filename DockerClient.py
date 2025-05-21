import logging

import docker

import utils

logger = logging.getLogger("multiscale")
logging.getLogger("multiscale").setLevel(logging.INFO)

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")


class DockerClient:
    def __init__(self):
        self.client = docker.DockerClient(base_url=DOCKER_SOCKET)

    # @utils.print_execution_time
    def update_cpu(self, container_ref, cpus):
        try:
            container = self.client.containers.get(container_ref)
            container.update(cpu_quota=int(cpus * 100000)) # This again needs to be integer nanoseconds
            logger.info(f"Container set to work with {cpus} cores")
        except Exception as e:
            logger.error("Could not connect to docker container", e)

    # @utils.print_execution_time # 3ms
    def get_container_stats(self, container_ref):
        try:
            container = self.client.containers.get(container_ref)
            return container
        except Exception as e:
            logger.error("Could not connect to docker container", e)

    def get_container_ip(self, container_ref):
        if container_ref is None or container_ref == 'Unknown':
            return "Unknown"

        c_stats = self.get_container_stats(container_ref)
        return c_stats.attrs['NetworkSettings']['Networks']['elastic-workbench_docker_network']['IPAddress']

    def get_container_cores(self, container_ref):
        c_stats = self.get_container_stats(container_ref)
        return int(c_stats.attrs['HostConfig'].get('CpuQuota', 0) / 100000)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = DockerClient()
    # ip = client.get_container_ip("elastic-workbench-qr-detector-1")
    # print(ip)
    print(client.get_container_cores("elastic-workbench-qr-detector-1"))
