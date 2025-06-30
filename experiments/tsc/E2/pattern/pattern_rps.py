import logging
import os
from enum import Enum

import utils
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

    def get_current_rps(self, req_pattern: RequestPattern, seconds_passed: int) -> int:
        with open(ROOT + "/" + req_pattern.value, "r") as f:
            rps_lines = f.readlines()

        rps = rps_lines[seconds_passed - 1].strip()
        return int(rps)

    # TODO: Means changing the rps in Redis; I can switch to completion rate for the SLO!
    def reconfigure_rps(self, req_pattern: RequestPattern, service_id: ServiceID, seconds_passed: int):
        rps = self.get_current_rps(req_pattern, seconds_passed)
        self.redis_client.store_assignment(service_id, {"C_1": rps})
        logger.info(f"Reconfigured after {seconds_passed}s to RPS: {rps} ")


# Load RPS values from the .txt file

if __name__ == '__main__':
    pattern = PatternRPS()
    pattern.get_current_rps(RequestPattern.BURSTY, 1000)

# # Generate time axis
# duration_seconds = len(rps_values)
# time_minutes = [t / 60 for t in range(duration_seconds)]
#
# # Plot
# plt.figure(figsize=(12, 4))
# plt.plot(time_minutes, rps_values, color="blue", label="RPS")
# plt.title("Request Pattern Over 1 Hour")
# plt.xlabel("Time (minutes)")
# plt.ylabel("Requests per Second (RPS)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
