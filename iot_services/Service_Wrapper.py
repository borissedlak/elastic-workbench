import ast
import logging

from flask import Flask, request

import utils
from DockerClient import DockerClient
from IoTService import IoTService
from iot_services.QrDetector.QrDetector import QrDetector

app = Flask(__name__)

# logger = logging.getLogger("multiscale")
# logging.getLogger('multiscale').setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")
CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")


def init_service(s_type):
    if s_type == "QR":
        return QrDetector()
    else:
        raise RuntimeError("Must pass type!")


class ServiceWrapper:
    def __init__(self, s_type, start_processing=False):
        self.service: IoTService = init_service(s_type)
        self.docker_client = DockerClient(DOCKER_SOCKET)
        if start_processing:
            self.start_processing()
        app.run(host='0.0.0.0', port=8080)

    # @utils.print_execution_time
    @app.route("/start_processing", methods=['POST'])
    def start_processing(self):
        self.service.start_process()
        return ""

    @app.route("/stop_all", methods=['POST'])
    def terminate_processing(self):
        self.service.terminate()
        return ""

    @app.route("/change_config", methods=['PUT'])
    def change_config(self):
        service_d = ast.literal_eval(request.args.get('service_description'))
        self.service.change_config(service_d)
        return ""

    @app.route("/change_threads", methods=['PUT'])
    def change_threads(self):
        threads_num = int(request.args.get('thread_number'))

        # 1) Change the number of threads of the application
        self.service.change_threads(threads_num)
        # 2) Change the number of cores available for docker
        self.docker_client.update_cpu(CONTAINER_REF, threads_num)

        return ""


if __name__ == '__main__':
    s_wrapper = ServiceWrapper(s_type="QR", start_processing=True)
    # s_wrapper.start_processing()
