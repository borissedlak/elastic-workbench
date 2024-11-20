from prometheus_api_client import PrometheusConnect

import utils

INTERVAL = "3s"
MB = {'variables': ['fps', 'pixel', 'energy', 'cores'],
      'parameter': ['pixel', 'cores'],
      'slos': [('pixel', utils.sigmoid, 0.015, 450, 0.85),
               ('fps', utils.sigmoid, 0.35, 25, 1.5)]}


class PrometheusClient:
    def __init__(self, url="http://localhost:9090"):
        self.client = PrometheusConnect(url=url, disable_ssl=True)

    # @utils.print_execution_time  # only around 3ms
    def get_metric_values(self, metric_name, period=None):
        start = f"avg_over_time(" if period is not None else ""
        end = f"[{period}])" if period is not None else ""

        metric_data = self.client.custom_query(query=f'{start}{{__name__=~"{metric_name}"}}{end}')
        transformed = utils.convert_prom_multi(metric_data, item_name="metric_id", decimal=True)
        return transformed


if __name__ == "__main__":
    client = PrometheusClient()
    print("Metric assignments:", client.get_metric_values("|".join(list(set(MB['variables']) - set(MB['parameter']))), period="10s"))
    print("Parameter assignments:", client.get_metric_values("|".join(MB['parameter'])))
