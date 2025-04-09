import json
import logging
import random
from enum import Enum
from typing import NamedTuple

from HttpClient import HttpClient

logger = logging.getLogger("multiscale")


class ServiceType(Enum):
    QR = "elastic-workbench-video-processing"


class ServiceID(NamedTuple):
    host: str
    service_type: ServiceType
    container_id: str


class ES_Registry:
    _ES_activate_default = {'elastic-workbench-video-processing': ['resource_scaling', 'quality_scaling']}

    def __init__(self):
        self.http_client = HttpClient()
        self.ES_activated = self._ES_activate_default
        with open('es_registry.json', 'r') as f:
            self.es_api = json.load(f)

    def is_ES_supported(self, service_type: ServiceType, es_name):
        services = self.es_api['services']
        # service_ES = services.get(service_type, [])
        for service in services:
            if service['name'] == service_type.value:
                for es in service['elasticity_strategies']:
                    if (
                            # es.get("service_type") == service_type and
                            es.get("ES_name") == es_name
                    ):
                        if es_name in self.ES_activated.get(service_type.value, []):
                            return True
                        else:
                            logger.info(f"Strategy <{service_type.value},{es_name}> is registered, but not activated")
                            return False

        logger.info("No corresponding strategy registered")
        return False

    def get_ES_information(self, service_type: ServiceType, es_name):
        for service in self.es_api["services"]:
            if service["name"] == service_type.value:
                for es in service["elasticity_strategies"]:
                    if es["ES_name"] == es_name:
                        return es["endpoints"]
        return []

    def get_random_ES_for_service(self, service_type: ServiceType):
        if service_type.value in self.ES_activated and bool(self.ES_activated[service_type.value]):
            return random.choice(self.ES_activated[service_type.value])

        raise RuntimeError(f"Requesting strategy for unknown service {service_type.value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    es_reg = ES_Registry()
    print(es_reg.is_ES_supported(ServiceType.QR, 'resource_scaling'))
    print(es_reg.get_ES_information(ServiceType.QR, 'resource_scaling'))
    print(es_reg.get_random_ES_for_service(ServiceType.QR))
