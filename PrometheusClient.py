from prometheus_api_client import PrometheusConnect

import utils
from agent.ES_Registry import ServiceID, ServiceType


class PrometheusClient:
    def __init__(self, url):
        self.client = PrometheusConnect(url=url, disable_ssl=True)

    # TODO: Not AVG still not working
    # @utils.print_execution_time  # only around 3ms
    def get_metrics(self, metric_name, service_id: ServiceID = None, period=None, avg=True):
        avg_str = "avg_over_time" if avg else ""
        start = f"{avg_str}(" if period is not None else ""
        end = f"[{period}])" if period is not None else ""

        instance_filter = f',instance="{service_id.host}:8000",container_id="{service_id.container_id}"' if service_id else ""

        metric_data = self.client.custom_query(query=f'{start}{{__name__=~"{metric_name}"{instance_filter}}}{end}')
        transformed = utils.convert_prom_multi(metric_data, decimal=True, avg=avg)
        return transformed


if __name__ == "__main__":
    client = PrometheusClient("http://localhost:9090")
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-video-processing-1")
    print("Metric assignments:", client.get_metrics("|".join(["fps"]), period="10s", service_id=qr_local))
    # print("Parameter assignments:", client.get_metrics("|".join(MB['parameter']), instance="172.18.0.4"))
