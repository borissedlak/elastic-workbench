import logging

import docker

import utils


logger = logging.getLogger("multiscale")
logging.getLogger("multiscale").setLevel(logging.INFO)

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")


class DockerClient:
    def __init__(self, url):
        self.client = docker.DockerClient(base_url=url)

    # TODO: Takes too long with 90ms
    # @utils.print_execution_time
    def update_cpu(self, container_ref, cpus):
        try:
            container = self.client.containers.get(container_ref)
            container.update(cpu_quota=cpus * 100000)
            logger.info(f"Container set to work with {cpus} cores")
        except Exception as e:
            logger.error("Could not connect to docker container", e)

    # @utils.print_execution_time # Takes 6 ms
    def get_container_id(self, container_name):
        container = self.client.containers.list(filters={'name': container_name})
        if container:
            return str(container[0].id)[:12]
        else:
            return "Unknown"

    @utils.print_execution_time
    def get_container_stats(self, container_ref, stream_p=False):
        try:
            container = self.client.containers.get(container_ref)
            stats = container.stats(stream=stream_p, decode=stream_p)
            return stats
        except Exception as e:
            logger.error("Could not connect to docker container", e)

    # def get_max_cpus(self):
    #     container = self.client.containers.get("multiscaler-video-processing-1")
    #     cpu_set = container.attrs['HostConfig']['CpusetCpus']
    #
    #     # Calculate the number of cores
    #     max_cores = len(cpu_set.split(',')) if cpu_set else "No CPU limits set"
    #     print(f"Maximum cores: {max_cores}")


if __name__ == "__main__":
    client = DockerClient(DOCKER_SOCKET)
    # client.update_cpu("67959d3ff81a", 5)
    stream = client.get_container_stats("multiscaler-video-processing-1", stream_p=True)

    for s in stream:
        # print(utils.calculate_cpu_percentage(s))
        print(1)
    # print(utils.calculate_cpu_percentage(client.get_container_stats("multiscaler-video-processing-1")))
