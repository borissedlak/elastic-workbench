import json
import logging
import os
import random
from enum import Enum
from typing import NamedTuple

from HttpClient import HttpClient

logger = logging.getLogger("multiscale")


class ServiceType(Enum):
    QR = "elastic-workbench-video-processing"


class EsType(Enum):
    STARTUP = 'resource_scaling'  # Do the same as when scaling resources
    RESOURCE_S = 'resource_scaling'
    QUALITY_S = 'quality_scaling'
    OFFLOADING = 'offloading'


class ServiceID(NamedTuple):
    host: str
    service_type: ServiceType
    container_id: str


class ES_Registry:
    _ES_activate_default = {'elastic-workbench-video-processing': ['resource_scaling', 'quality_scaling']}

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

    def get_active_ES_for_s(self, service_type: ServiceType):
        if service_type.name in self.ES_activated.keys():
            return self.ES_activated[service_type.value]
        else:
            logger.warning(f"Querying active ES for unknown service type {service_type.name}")
            return []

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
        return self.get_ES_information(service_type, es_type)['cooldown']


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    es_reg = ES_Registry()
    # print(es_reg.is_ES_supported(ServiceType.QR, 'resource_scaling'))
    # print(es_reg.get_ES_information(ServiceType.QR, 'resource_scaling'))
    # print(es_reg.get_random_ES_for_service(ServiceType.QR))
    print(es_reg.get_ES_cooldown(ServiceType.QR, EsType.RESOURCE_S))
