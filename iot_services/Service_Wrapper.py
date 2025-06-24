import logging

import numpy as np
from flask import Flask, request

import utils
from DockerClient import DockerClient
from agent.es_registry import ESType
from iot_services.IoTService import IoTService

app = Flask(__name__)

logger = logging.getLogger("multiscale")
# logging.getLogger('multiscale').setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

CONTAINER_REF = utils.get_env_param("CONTAINER_REF", "Unknown")
DEFAULT_CORES = float(utils.get_env_param("DEFAULT_CORES", 2.0))
DEFAULT_CLIENTS = utils.get_env_param("DEFAULT_CLIENTS", None)
SERVICE_TYPE = utils.get_env_param("SERVICE_TYPE", None)
MAX_CORES = utils.get_env_param("MAX_CORES", None)


def init_service(s_type):
    if s_type == "QR":
        from iot_services.QrDetector.QrDetector import QrDetector
        return QrDetector()
    if s_type == "CV":
        from iot_services.CvAnalyzer_Yolo.CvAnalyzer import CvAnalyzer
        return CvAnalyzer()
    if s_type == "PC":
        from iot_services.PcVisualizer.PcVisualizer import PcVisualizer
        return PcVisualizer()
    else:
        raise RuntimeError("Must pass type!")


class ServiceWrapper:
    def __init__(self, s_type, start_processing=False):
        self.service: IoTService = init_service(s_type)
        self.docker_client = DockerClient()
        self.bounds = self.service.es_registry.get_boundaries_minimalistic(self.service.service_type, MAX_CORES)

        if start_processing:
            self.start_processing()
            self.scale_cores(DEFAULT_CORES)

            if DEFAULT_CLIENTS:
                clients = DEFAULT_CLIENTS.split(",")
                for client in clients:
                    self.service.change_request_arrival(client.split(":")[0], int(client.split(":")[1]))

        self.app = Flask(__name__)
        self.app.add_url_rule('/start_processing', 'start_processing', self.start_processing, methods=['POST'])
        self.app.add_url_rule('/stop_all', 'stop_all', self.terminate_processing, methods=['POST'])
        # self.app.add_url_rule('/change_config', 'change_config', self.change_config, methods=['PUT'])
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
    # def change_config(self):
    #     service_d = ast.literal_eval(request.args.get('service_description'))
    #     self.service.change_config(service_d)
    #     self.service.set_flag_and_cooldown(EsType.QUALITY_SCALE)
    #     return ""

    def quality_scaling(self):
        data_quality = round(float(request.args.get('data_quality')))
        data_quality_corrected = np.clip(data_quality, self.bounds['data_quality']['min'], self.bounds['data_quality']['max'])

        if data_quality != data_quality_corrected:
            logger.warning(f"Manually corrected data_quality from {data_quality} to {data_quality_corrected}")

        s_conf = self.service.service_conf.copy()
        s_conf['data_quality'] = data_quality_corrected

        self.service.change_config(s_conf)
        self.service.set_flag_and_cooldown(ESType.QUALITY_SCALE)
        return ""

    def model_scaling(self):
        model_size = round(float(request.args.get('model_size')))
        model_size_corrected = np.clip(model_size, self.bounds['model_size']['min'], self.bounds['model_size']['max'])

        if model_size != model_size_corrected:
            logger.warning(f"Manually corrected model size from {model_size} to {model_size_corrected}")

        s_conf = self.service.service_conf.copy()
        s_conf['model_size'] = model_size_corrected

        self.service.change_config(s_conf)
        self.service.set_flag_and_cooldown(ESType.MODEL_SCALE)
        return ""

    def resource_scaling(self):
        cpu_cores = round(float(request.args.get('cores')), 2)
        cores_corrected = np.clip(cpu_cores, self.bounds['cores']['min'], self.bounds['cores']['max'])

        if cpu_cores != cores_corrected:
            logger.warning(f"Manually corrected resources from {cpu_cores} to {cores_corrected}")

        self.scale_cores(cores_corrected)
        return ""

    def scale_cores(self, fractional_cores: float):
        # 1) Change the number of threads of the application; cannot start fractional # of threads
        self.service.vertical_scaling(fractional_cores)
        # 2) Change the number of cores available for docker; can scale continuously
        self.docker_client.update_cpu(CONTAINER_REF, fractional_cores)


if __name__ == '__main__':
    s_wrapper = ServiceWrapper(s_type=SERVICE_TYPE, start_processing=True)
