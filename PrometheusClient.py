from prometheus_api_client import PrometheusConnect

import utils
from slo_config import MB


class PrometheusClient:
    def __init__(self, url):
        self.client = PrometheusConnect(url=url, disable_ssl=True)

    # @utils.print_execution_time  # only around 3ms
    def get_metrics(self, metric_name, period=None, instance=None):
        start = f"avg_over_time(" if period is not None else ""
        end = f"[{period}])" if period is not None else ""

        instance_filter = f',instance="{instance}:8000"' if instance else ""

        metric_data = self.client.custom_query(query=f'{start}{{__name__=~"{metric_name}"{instance_filter}}}{end}')
        transformed = utils.convert_prom_multi(metric_data, item_name="metric_id", decimal=True)
        return transformed


if __name__ == "__main__":
    client = PrometheusClient("http://172.18.0.2:9090")
    print("Metric assignments:",
          client.get_metrics("|".join(list(set(MB['variables']) - set(MB['parameter']))), period="10s", instance="172.18.0.4:8000"))
    print("Parameter assignments:", client.get_metrics("|".join(MB['parameter']), instance="172.18.0.4:8000"))
