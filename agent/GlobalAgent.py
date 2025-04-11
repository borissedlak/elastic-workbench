import logging

from agent.ES_Registry import ServiceType, ServiceID
from agent.ScalingAgent import ScalingAgent

# logger = logging.getLogger("multiscale")
# logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

EVALUATION_CYCLE = 3  # Might be less frequent for GSA


class GlobalAgent(ScalingAgent):
    def __init__(self, prom_server, services_monitored):
        super().__init__(prom_server, services_monitored, EVALUATION_CYCLE)


if __name__ == '__main__':
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-video-processing-1")
    GlobalAgent(services_monitored=[qr_local], prom_server=ps).start()
