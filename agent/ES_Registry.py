import json
import logging
import os
import random
from enum import Enum
from typing import NamedTuple, List

from HttpClient import HttpClient

logger = logging.getLogger("multiscale")


class ServiceType(Enum):
    QR = "elastic-workbench-video-processing"


class EsType(Enum):
    STARTUP = 'startup'
    RESOURCE_SCALE = 'resource_scaling'
    QUALITY_SCALE = 'quality_scaling'
    RESOURCE_SWAP = 'resource_swapping'
    OFFLOADING = 'offloading'


class ServiceID(NamedTuple):
    host: str
    service_type: ServiceType
    container_id: str


class ES_Registry:
    # TODO: This is super buggy because the order is important for the variables in the Policy Solver!!
    _ES_activate_default = {'elastic-workbench-video-processing': ['quality_scaling', 'resource_scaling']}

    def __init__(self):
        self.http_client = HttpClient()
        self.ES_activated = self._ES_activate_default
        ROOT = os.path.dirname(__file__)
        with open(ROOT + '/conf/es_registry.json', 'r') as f:
            self.es_api = json.load(f)

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

    def get_parameter_bounds_for_active_ES(self, service_type: ServiceType):
        active_ES = self.get_active_ES_for_s(service_type)
        param_list = []
        for es in active_ES:
            endpoints = self.get_ES_information(service_type, es)["endpoints"]
            params = [params for params in [e["parameters"] for e in endpoints]]
            param_list.append(params[0][0])
        return param_list

    def get_ES_information(self, service_type: ServiceType, es_type: EsType):
        for service in self.es_api["services"]:
            if service["name"] == service_type.value:
                for es in service["elasticity_strategies"]:
                    if es["ES_name"] == es_type.value:
                        return es
        return None

    def get_random_ES_for_service(self, service_type: ServiceType) -> EsType:
        if service_type.value in self.ES_activated and bool(self.ES_activated[service_type.value]):
            return EsType(random.choice(self.ES_activated[service_type.value]))

        raise RuntimeError(f"Requesting strategy for unknown service {service_type.value}")

    def get_ES_cooldown(self, service_type: ServiceType, es_type: EsType) -> int:
        service_info = self.get_ES_information(service_type, es_type)

        return service_info['cooldown'] if es_type != EsType.STARTUP else 2000


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    es_reg = ES_Registry()
    # print(es_reg.is_ES_supported(ServiceType.QR, 'resource_scaling'))
    # print(es_reg.get_ES_information(ServiceType.QR, 'resource_scaling'))
    # print(es_reg.get_random_ES_for_service(ServiceType.QR))
    print(es_reg.get_parameter_bounds_for_active_ES(ServiceType.QR))
