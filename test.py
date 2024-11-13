import utils
from DockerClient import DockerClient
from HttpClient import HttpClient

DOCKER_PREFIX = utils.get_ENV_PARAM('DOCKER_PREFIX', "unix://")
DOCKER_SOCKET_PATH = utils.get_ENV_PARAM('DOCKER_SOCKET_PATH', "/home/boris/.docker/desktop/docker.sock")

docker_client = DockerClient(DOCKER_PREFIX + DOCKER_SOCKET_PATH)
http_client = HttpClient()

THREADS_AND_CORES = 8

docker_client.update_cpu("a780851d157d",THREADS_AND_CORES)
http_client.change_threads("localhost", THREADS_AND_CORES)