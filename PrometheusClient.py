import time

from prometheus_api_client import PrometheusConnect

import utils
from agent.components.es_registry import ServiceID, ServiceType


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

    def get_container_cpu_utilization(self, service_id: ServiceID, rate_interval="10s") -> float:
        """Returns normalized CPU usage (fraction of quota used) for the container."""
        name = service_id.container_id

        query = f"""
        sum by (container) (
          rate(container_cpu_usage_seconds_total{{name="{name}"}}[{rate_interval}])
        )
        /
        on(container)
        group_left
        max by (container) (
          container_spec_cpu_quota{{name="{name}"}}
        )
        /
        max by (container) (
          container_spec_cpu_period{{name="{name}"}}
        )
        """.strip()

        result = self.client.custom_query(query=query)
        # Extract float value from result
        if result and isinstance(result, list) and 'value' in result[0]:
            return round(float(result[0]['value'][1]) * 1e10, 3)
        else:
            return float('nan')  # or raise an exception if you prefer


if __name__ == "__main__":
    client = PrometheusClient("http://localhost:9090")
    qr_local = ServiceID("localhost", ServiceType.QR, "elastic-workbench-qr-detector-1", port="8080")
    cv_local = ServiceID("localhost", ServiceType.CV, "elastic-workbench-cv-analyzer-1", port="8081")
    pc_local = ServiceID("localhost", ServiceType.PC, "elastic-workbench-pc-visualizer-1", port="8082")

    for i in range(1, 100000):
        # print("Metric assignments:", client.get_metrics(["data_quality", "cores", "model_size"], service_id=cv_local))
        print(client.get_container_cpu_utilization(cv_local))
        time.sleep(0.25)
