import ast
import logging

from flask import Flask, request

import utils
from DockerClient import DockerClient
from IoTService import IoTService
from iot_services.CvAnalyzer.CvAnalyzer import CvAnalyzer
from iot_services.QrDetector.QrDetector import QrDetector

app = Flask(__name__)

# logger = logging.getLogger("multiscale")
# logging.getLogger('multiscale').setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")
DEFAULT_CORES = utils.get_env_param("DEFAULT_CORES", 2)
DEFAULT_CLIENTS = utils.get_env_param("DEFAULT_CLIENTS", None)
SERVICE_TYPE = utils.get_env_param("SERVICE_TYPE", None)


def init_service(s_type):
    if s_type == "QR":
        return QrDetector()
    if s_type == "CV":
        return CvAnalyzer()
    else:
        raise RuntimeError("Must pass type!")


class ServiceWrapper:
    def __init__(self, s_type, start_processing=False):
        self.service: IoTService = init_service(s_type)
        self.docker_client = DockerClient()

        if start_processing:
            self.start_processing()
            # TODO: Must add again!!
            # self.scale_cores(DEFAULT_CORES)
            if DEFAULT_CLIENTS:
                clients = DEFAULT_CLIENTS.split(",")
                for client in clients:
                    self.service.change_request_arrival(client.split(":")[0], int(client.split(":")[1]))

        self.app = Flask(__name__)
        self.app.add_url_rule('/start_processing', 'start_processing', self.start_processing, methods=['POST'])
        self.app.add_url_rule('/stop_all', 'stop_all', self.terminate_processing, methods=['POST'])
        self.app.add_url_rule('/change_config', 'change_config', self.change_config, methods=['PUT'])
        self.app.add_url_rule('/quality_scaling', 'quality_scaling', self.quality_scaling, methods=['PUT'])
        self.app.add_url_rule('/model_scaling', 'model_scaling', self.model_scaling, methods=['PUT'])
        self.app.add_url_rule('/resource_scaling', 'resource_scaling', self.resource_scaling, methods=['PUT'])
        self.app.add_url_rule('/change_rps', 'change_rps', self.alter_client_connection, methods=['PUT'])
        self.app.run(host='0.0.0.0', port=8080)

    # @utils.print_execution_time
    # @app.route("/start_processing", methods=['POST'])
    def start_processing(self):
        self.service.start_process()
        return ""

    # @app.route("/stop_all", methods=['POST'])
    def terminate_processing(self):
        self.service.terminate()
        return ""

    def alter_client_connection(self):
        client_id = str(request.args.get('client_id'))
        rps = int(request.args.get('rps'))

        self.service.change_request_arrival(client_id, rps)
        return ""

    ######################################

    # @app.route("/change_config", methods=['PUT'])
    def change_config(self):
        service_d = ast.literal_eval(request.args.get('service_description'))
        self.service.change_config(service_d)
        return ""

    # @app.route("/change_config", methods=['PUT'])
    def quality_scaling(self):
        quality = int(request.args.get('quality'))
        s_conf = self.service.service_conf
        s_conf['quality'] = quality
        self.service.change_config(s_conf)
        return ""

    # @app.route("/change_config", methods=['PUT'])
    def model_scaling(self):
        model_size = int(request.args.get('model_size'))
        s_conf = self.service.service_conf
        s_conf['model_size'] = model_size
        self.service.change_config(s_conf)
        return ""

    # @app.route("/vertical_scaling", methods=['PUT'])
    def resource_scaling(self):
        cpu_cores = int(request.args.get('cores'))
        self.scale_cores(cpu_cores)
        return ""

    def scale_cores(self, cores):
        # 1) Change the number of threads of the application
        self.service.vertical_scaling(cores)
        # 2) Change the number of cores available for docker
        self.docker_client.update_cpu(CONTAINER_REF, cores)


if __name__ == '__main__':
    s_wrapper = ServiceWrapper(s_type=SERVICE_TYPE, start_processing=True)
