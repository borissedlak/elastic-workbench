import json
import logging
import random
from enum import Enum
from typing import NamedTuple, List, Dict

from HttpClient import HttpClient

logger = logging.getLogger("multiscale")


class ServiceType(Enum):
    QR = "elastic-workbench-qr-detector"
    CV = "elastic-workbench-cv-analyzer"
    UNKNOWN = "unknown"


class EsType(Enum):
    STARTUP = 'startup'
    QUALITY_SCALE = 'quality_scaling'
    RESOURCE_SCALE = 'resource_scaling'
    MODEL_SCALE = 'model_scaling'
    RESOURCE_SWAP = 'resource_swapping'
    OFFLOADING = 'offloading'
    UNKNOWN = 'unknown'


class ServiceID(NamedTuple):
    host: str
    service_type: ServiceType
    container_id: str


class ES_Registry:
    # TODO: This is super buggy because the order is important for the variables in the Policy Solver!!
    # _ES_activate_default = {'elastic-workbench-qr-detector': ['quality_scaling', 'resource_scaling'],
    #                         'elastic-workbench-cv-analyzer': ['resource_scaling', 'model_scaling']}

    def __init__(self, es_registry_path):
        self.http_client = HttpClient()

        with open(es_registry_path, 'r') as f:
            self.es_api = json.load(f)

    def get_ES_information(self, service_type: ServiceType, es_type: EsType):
        if service_type.value in self.es_api and es_type.value in self.es_api[service_type.value]:
            return self.es_api[service_type.value][es_type.value]

        logger.warning(f"Trying to find unknown strategy {es_type.value} for service {service_type.value}")
        return None

    def is_ES_supported(self, service_type: ServiceType, es_type: EsType) -> bool:
        return self.get_ES_information(service_type, es_type) is not None

    def get_supported_ES_for_service(self, service_type: ServiceType) -> List[EsType]:
        strategies = self.es_api.get(service_type.value, {})
        return [EsType(es) for es in strategies]

    def get_random_ES_for_service(self, service_type: ServiceType) -> EsType:
        return random.choice(self.get_supported_ES_for_service(service_type))

    def get_parameter_bounds_for_active_ES(self, service_type: ServiceType, available_cores=None) -> Dict[EsType, Dict]:
        parameter_bounds = {}
        for es_type in self.get_supported_ES_for_service(service_type):
            info = self.get_ES_information(service_type, es_type)
            if not info or "parameters" not in info:
                continue

            params = info["parameters"]

            if es_type == EsType.RESOURCE_SCALE and available_cores is not None:
                params["cores"]["max"] = available_cores

            parameter_bounds[es_type] = params

        return parameter_bounds

    def get_ES_cooldown(self, service_type: ServiceType, es_type: EsType) -> int:
        if es_type == EsType.STARTUP:
            return 3500

        service_info = self.get_ES_information(service_type, es_type)
        if service_info is None:
            logger.warning(f"Trying to get cooldown for unknown strategy {es_type.value}, {service_type.value}")
            return 0

        return service_info['cooldown']
