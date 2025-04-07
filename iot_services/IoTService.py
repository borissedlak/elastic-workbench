import utils

from DockerClient import DockerClient

# DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")
CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")

class IoTService:
    def __init__(self):
        self.service_id = CONTAINER_REF
        self._terminated = True
        self._running = False
        self.service_conf = {}

        self.docker_client = DockerClient()

    def process_one_iteration(self, params, frame) -> None:
        pass

    def start_process(self):
        pass

    def terminate(self):
        pass

    def change_config(self, service_d):
        pass

    def vertical_scaling(self, threads_num):
        pass
