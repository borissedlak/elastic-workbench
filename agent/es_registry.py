import json
import logging
import random
from enum import Enum
from os import PathLike
from typing import NamedTuple, List, Dict, Tuple

from agent import agent_utils

logger = logging.getLogger("multiscale")


class ServiceType(Enum):
    QR = "elastic-workbench-qr-detector"
    CV = "elastic-workbench-cv-analyzer"
    PC = "elastic-workbench-pc-visualizer"
    UNKNOWN = "unknown"


class ESType(Enum):
    STARTUP = 'startup'
    QUALITY_SCALE = 'quality_scaling'
    PARALLELISM_SCALE = 'parallelism_scaling'
    RESOURCE_SCALE = 'resource_scaling'
    MODEL_SCALE = 'model_scaling'
    RESOURCE_SWAP = 'resource_swapping'
    OFFLOADING = 'offloading'
    IDLE = 'idle'
    UNKNOWN = 'unknown'


class ServiceID(NamedTuple):
    host: str
    service_type: ServiceType
    container_id: str


class ESRegistry:

    def __init__(self, es_registry_path: PathLike):

        with open(es_registry_path, 'r') as f:
            self.es_api = json.load(f)

    def get_es_information(self, service_type: ServiceType, es_type: ESType):
        if service_type.value in self.es_api and es_type.value in self.es_api[service_type.value]:
            return self.es_api[service_type.value][es_type.value]

        logger.warning(f"Trying to find unknown strategy {es_type.value} for service {service_type.value}")
        return None

    def is_es_supported(self, service_type: ServiceType, es_type: ESType) -> bool:
        return self.get_es_information(service_type, es_type) is not None

    def get_active_ES_for_service(self, service_type: ServiceType) -> List[ESType]:
        # TODO: Here I can filter if it should be active or not
        strategies = self.es_api.get(service_type.value, {})

        active = {
            k: v for k, v in strategies.items()
            if v.get('active') is True
        }

        return [ESType(es) for es in active]

    # def _get_random_ES_for_service(self, service_type: ServiceType) -> ESType:
    #     return random.choice(self.get_supported_ES_for_service(service_type))

    # def get_random_ES_and_params(self, service_type: ServiceType) -> Tuple[ESType, Dict]:
    #     rand_ES = self._get_random_ES_for_service(service_type)
    #     parameter_bounds = self.get_parameter_bounds_for_active_ES(service_type).get(rand_ES, {})
    #     random_params = agent_utils.get_random_parameter_assignments(parameter_bounds)
    #
    #     return rand_ES, random_params

    def get_parameter_bounds_for_active_ES(self, service_type: ServiceType, available_cores=None) -> Dict[ESType, Dict]:
        parameter_bounds = {}
        for es_type in self.get_active_ES_for_service(service_type):
            info = self.get_es_information(service_type, es_type)
            if not info or "parameters" not in info:
                continue

            params = info["parameters"]

            if es_type == ESType.RESOURCE_SCALE and available_cores is not None:
                params["cores"]["max"] = max(available_cores, 1.0)

            parameter_bounds[es_type] = params

        return parameter_bounds

    def get_boundaries_minimalistic(self, service_type, max_cores) -> Dict[str, Dict]:
        boundaries = {}

        param_bounds = self.get_parameter_bounds_for_active_ES(service_type, max_cores)
        for es_type in param_bounds.values():
            param_bound = list(es_type.items())[0]
            boundaries[param_bound[0]] = param_bound[1]

        return boundaries

    def get_es_cooldown(self, service_type: ServiceType, es_type: ESType) -> int:
        if es_type == ESType.STARTUP:
            return 3500

        service_info = self.get_es_information(service_type, es_type)
        if service_info is None:
            logger.warning(f"Trying to get cooldown for unknown strategy {es_type.value}, {service_type.value}")
            return 0

        return service_info['cooldown']
