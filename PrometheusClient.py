import math

from prometheus_api_client import PrometheusConnect

import utils

MB = {'variables': ['fps', 'pixel', 'energy', 'cores'],
      'parameter': ['pixel', 'cores'],
      'slos': [('pixel', 180, math.inf), ('fps', 20, math.inf), ('energy', 0, 1)]}


class PrometheusClient:
    def __init__(self, url="http://localhost:9090"):
        self.client = PrometheusConnect(url=url, disable_ssl=True)
        self.MB = MB
        # TODO: If I take an action, the agent should suspend any actions until 2 * (interval) has passed
        self.observation_interval = "10s"

    # @utils.print_execution_time  # only around 3ms
    def get_param_assignments(self, metric_name="|".join(MB['parameter'])):
        raw_result = self.client.custom_query(f'{{__name__=~"{metric_name}"}}')
        transformed = utils.convert_prom_multi(raw_result, decimal=False)
        return transformed

    # @utils.print_execution_time  # only around 3ms
    def get_slo_evaluations(self, metric_name="|".join([item[0] for item in MB['slos']]), threshold=20):
        metric_data = self.client.custom_query(
            query=f'avg_over_time({{__name__=~"{metric_name}"}}[{self.observation_interval}])')
        transformed = utils.convert_prom_multi(metric_data, item_name="metric_id", decimal=True)
        # TODO: Return fuzzy SLO fulfillment --> compare against thresh from MB object

        fuzzy_slof = []
        for slo_var_obs in transformed:
            var, t_min, t_max = utils.filter_tuple(MB['slos'], slo_var_obs[0], 0)

            # TODO: There is surely a better way for this, like a function expressing this over all possible metric values
            if slo_var_obs[1] < t_max:
                fuzzy_slof.append((var, slo_var_obs[1] / t_min))
            elif slo_var_obs[1] > t_max:
                fuzzy_slof.append((var, t_max / slo_var_obs[1]))
            else:
                raise RuntimeError("How?")

        return fuzzy_slof


if __name__ == "__main__":
    client = PrometheusClient()
    # print(client.evaluate_slos("|".join(client.MB['variables'])))
    print("Parameter assignments:", client.get_param_assignments())
    print("SLO evaluation:", client.get_slo_evaluations())
