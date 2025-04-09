import logging

from agent.ES_Registry import ServiceType, ServiceID
from agent.ScalingAgent import ScalingAgent

logger = logging.getLogger("multiscale")
logger.setLevel(logging.INFO)


class GlobalAgent(ScalingAgent):
    def __init__(self, prom_server, services_monitored):
        super().__init__(prom_server, services_monitored)


# TODO: This should not crash down only because the service is not available
if __name__ == '__main__':
    ps = "http://172.20.0.2:9090"

    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-video-processing-1")
    ScalingAgent(services_monitored=[qr_local], prom_server=ps).start()
