import time

from prometheus_api_client import PrometheusConnect

import utils
from agent.es_registry import ServiceID, ServiceType


class PrometheusClient:
    def __init__(self, url):
        self.client = PrometheusConnect(url=url, disable_ssl=True)

    # @utils.print_execution_time  # only around 3ms
    def get_metrics(self, metric_names: list[str], service_id: ServiceID = None, period=None):
        # avg_str = "avg_over_time"
        start = f"avg_over_time(" if period is not None else ""
        end = f"[{period}])" if period is not None else ""

        instance_filter = f',container_id="{service_id.container_id}"' if service_id else ""

        metric_str = "|".join(metric_names)
        metric_data = self.client.custom_query(query=f'{start}{{__name__=~"{metric_str}"{instance_filter}}}{end}')
        transformed = utils.convert_prom_multi(metric_data, decimal=True, avg=(period is not None))
        return transformed


if __name__ == "__main__":
    client = PrometheusClient("http://128.131.172.182:9090")
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")

    for i in range(1, 100000):
        print("Metric assignments:", client.get_metrics(["data_quality", "cores", "model_size"], service_id=cv_local))
        time.sleep(0.25)
