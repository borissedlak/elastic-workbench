from prometheus_api_client import PrometheusConnect

import utils


class PrometheusClient:
    def __init__(self, url="http://localhost:9090"):
        self.client = PrometheusConnect(url=url, disable_ssl=True)

    @utils.print_execution_time
    def fetch_metric(self, metric_name = "fps", threshold = 20):
        metric_data = self.client.custom_query(query=f"avg_over_time({metric_name}[1m])")
        current_value = float(metric_data[0]['value'][1])
        print(current_value / threshold)


if __name__ == "__main__":
    client = PrometheusClient()

    client.fetch_metric()
