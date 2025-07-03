import logging
import os
from enum import Enum
from typing import Tuple

import utils
from HttpClient import HttpClient
from RedisClient import RedisClient
from agent.components.es_registry import ServiceID

ROOT = os.path.dirname(__file__)
REDIS_INSTANCE = utils.get_env_param("REDIS_INSTANCE", "localhost")

logger = logging.getLogger("multiscale")


class RequestPattern(Enum):
    BURSTY = "bursty" # max val 397
    DIURNAL = "diurnal" # max val 387


class PatternRPS:
    def __init__(self):
        self.redis_client = RedisClient(host=REDIS_INSTANCE)
        self.http_client = HttpClient()

    def get_current_rps(self, req_pattern: RequestPattern, seconds_passed: int) -> Tuple[int, float]:
        with open(ROOT + "/" + req_pattern.value + ".csv", "r") as f:
            rps_lines = f.readlines()

        line = rps_lines[seconds_passed - 1].strip()
        rps_abs, rps_norm = line.split(",")[0], line.split(",")[1]
        return int(rps_abs), float(rps_norm)

    def reconfigure_rps(self, req_pattern: RequestPattern, service_id: ServiceID, max_rps: int, seconds_passed: int = None):
        _, rps_norm = self.get_current_rps(req_pattern, seconds_passed)
        rps = round(rps_norm * max_rps)
        self.http_client.update_service_rps(service_id, rps)
        logger.info(f"Reconfigured {service_id.service_type.value} after {seconds_passed}s to RPS: {rps} ")


if __name__ == '__main__':
    pattern = PatternRPS()
    pattern.get_current_rps(RequestPattern.BURSTY, 1000)
