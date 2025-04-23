import json
import logging
import random
from enum import Enum
from typing import NamedTuple, List

from HttpClient import HttpClient

logger = logging.getLogger("multiscale")


class ServiceType(Enum):
    QR = "elastic-workbench-qr-detector"
    CV = "elastic-workbench-cv-analyzer"


class EsType(Enum):
    STARTUP = 'startup'
    QUALITY_SCALE = 'quality_scaling'
    RESOURCE_SCALE = 'resource_scaling'
    MODEL_SCALE = 'model_scaling'
    RESOURCE_SWAP = 'resource_swapping'
    OFFLOADING = 'offloading'


class ServiceID(NamedTuple):
    host: str
    service_type: ServiceType
    container_id: str


class ES_Registry:
    # TODO: This is super buggy because the order is important for the variables in the Policy Solver!!
    _ES_activate_default = {'elastic-workbench-qr-detector': ['quality_scaling', 'resource_scaling'],
                            'elastic-workbench-cv-analyzer': ['resource_scaling', 'model_scaling']}

    def __init__(self, es_registry_path):
        self.http_client = HttpClient()
        self.ES_activated = self._ES_activate_default

        with open(es_registry_path, 'r') as f:
            self.es_api = json.load(f)

        logger.info(self.es_api)

    def is_ES_supported(self, service_type: ServiceType, es_type: EsType) -> bool:
        services = self.es_api['services']
        for service in services:
            if service['name'] == service_type.value:
                for es in service['elasticity_strategies']:
                    if (
                            # es.get("service_type") == service_type and
                            es.get("ES_name") == es_type.value
                    ):
                        if es_type.value in self.ES_activated.get(service_type.value, []):
                            return True
                        else:
                            logger.info(
                                f"Strategy <{service_type.value},{es_type.value}> is registered, but not activated")
                            return False

        logger.info("No corresponding strategy registered")
        return False

    def get_active_ES_for_s(self, service_type: ServiceType) -> List[EsType]:
        if service_type.value in self.ES_activated.keys():
            return [EsType(x) for x in self.ES_activated[service_type.value]]
        else:
            logger.warning(f"Querying active ES for unknown service type {service_type.value}")
            return []

    def get_parameter_bounds_for_active_ES(self, service_type: ServiceType, max_cores=None):
        active_ES = self.get_active_ES_for_s(service_type)
        param_list = []
        for es in active_ES:
            endpoints = self.get_ES_information(service_type, es)["endpoints"]
            params = [params for params in [e["parameters"] for e in endpoints]][0][0]

            if max_cores is not None and es == EsType.RESOURCE_SCALE:
                params["max"] = max_cores

            param_list.append(params | {"es_type": es})
        return param_list

    def get_ES_information(self, service_type: ServiceType, es_type: EsType):
        for service in self.es_api["services"]:
            if service["name"] == service_type.value:
                for es in service["elasticity_strategies"]:
                    if es["ES_name"] == es_type.value:
                        return es
        logger.warning(f"Trying to find unknown strategy {es_type.value} for service {service_type.value}")
        return None

    def get_random_ES_for_service(self, service_type: ServiceType) -> EsType:
        if service_type.value in self.ES_activated and bool(self.ES_activated[service_type.value]):
            return EsType(random.choice(self.ES_activated[service_type.value]))

        raise RuntimeError(f"Requesting strategy for unknown service {service_type.value}")

    def get_ES_cooldown(self, service_type: ServiceType, es_type: EsType) -> int:
        if es_type == EsType.STARTUP:
            return 3000

        service_info = self.get_ES_information(service_type, es_type)

        return service_info['cooldown']
