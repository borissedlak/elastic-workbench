import logging
import os
from enum import Enum

import utils
from HttpClient import HttpClient
from RedisClient import RedisClient
from agent.es_registry import ServiceType, ServiceID

ROOT = os.path.dirname(__file__)
REDIS_INSTANCE = utils.get_env_param("REDIS_INSTANCE", "localhost")

logger = logging.getLogger("multiscale")

class RequestPattern(Enum):
    BURSTY = "bursty.txt"
    DIURNAL = "diurnal.txt"


class PatternRPS:
    def __init__(self):
        self.redis_client = RedisClient(host=REDIS_INSTANCE)
        self.http_client = HttpClient()

    def get_current_rps(self, req_pattern: RequestPattern, seconds_passed: int) -> int:
        with open(ROOT + "/" + req_pattern.value, "r") as f:
            rps_lines = f.readlines()

        rps = rps_lines[seconds_passed - 1].strip()
        return int(rps)

    def reconfigure_rps(self, req_pattern: RequestPattern, service_id: ServiceID, seconds_passed: int):
        rps = self.get_current_rps(req_pattern, seconds_passed)
        self.http_client.update_service_rps(service_id, rps)
        # self.redis_client.store_assignment(service_id, {"C_1": rps})
        logger.info(f"Reconfigured after {seconds_passed}s to RPS: {rps} ")


# Load RPS values from the .txt file

if __name__ == '__main__':
    pattern = PatternRPS()
    pattern.get_current_rps(RequestPattern.BURSTY, 1000)
