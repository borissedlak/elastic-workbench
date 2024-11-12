import logging

import docker

logger = logging.getLogger("multiscale")


class DockerClient:
    def __init__(self, url):
        self.client = docker.DockerClient(base_url=url)  # 'unix:///home/boris/.docker/desktop/docker.sock')

    def update_cpu(self, container_id, cpus):
        try:
            container = self.client.containers.get(container_id)
            container.update(cpu_quota=cpus * 100000)
        except Exception as e:
            logger.error("Could not connect to docker container", e)
            # print(e)

if __name__ == "__main__":
    client = DockerClient('unix:///home/boris/.docker/desktop/docker.sock')
    client.update_cpu("c73449861df8", 5)