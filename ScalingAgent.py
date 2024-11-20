import os
import time
from threading import Thread

import numpy as np

import test_gpt
import utils
from DockerClient import DockerClient
from HttpClient import HttpClient
from PrometheusClient import PrometheusClient, MB, INTERVAL

DOCKER_SOCKET = utils.get_env_param('DOCKER_SOCKET', "unix:///var/run/docker.sock")


# TODO: So what the agent must do on a high level is:
#  1) Collect sensory state from Prometheus --> Easy ✓
#  2) Evaluate if SLOs are fulfilled --> Easy ✓
#  3) Retrain its interpretation model --> Medium
#  4) Act so that SLO-F is optimized --> Hard
#  _
#  However, I assume that the agent has the variable DAG and the SLO thresholds
#  And I dont have to resolve variable names dynamically, but keep them hardcoded


class AIFAgent(Thread):
    def __init__(self):
        super().__init__()

        self.prom_client = PrometheusClient()
        self.docker_client = DockerClient(DOCKER_SOCKET)
        self.http_client = HttpClient()

    def run(self):
        while True:
            initial_state = self.get_current_state()
            print("Current State:", initial_state)
            initial_state_f = [initial_state['pixel'], initial_state['fps']]

            action_vectors = test_gpt.get_action(initial_state_f)
            agent.act_on_env(action_vectors[0])

            time.sleep(8)
            updated_state = self.get_current_state()
            updated_state_f = [updated_state['pixel'], updated_state['fps']]

            value_factors = calculate_value_slo(updated_state)
            print("Value of new state:", value_factors)
            value = np.sum(value_factors)

            test_gpt.evaluate_result(initial_state_f, action_vectors, value, updated_state_f)

    # TODO: Also, I should probably look into better ways to query the metric values, like EMA
    def get_current_state(self):
        metric_vars = list(set(MB['variables']) - set(MB['parameter']))
        prom_metric_states = self.prom_client.get_metric_values("|".join(metric_vars), period=INTERVAL)
        prom_parameter_states = self.prom_client.get_metric_values("|".join(MB['parameter']))

        cpu_cores = os.cpu_count()
        return prom_metric_states | prom_parameter_states | {"max_cores": cpu_cores}

    # TODO: If I take an action, the agent should suspend any actions until 2 * (interval) has passed
    def act_on_env(self, a_pixel):
        self.http_client.change_config("localhost", {'pixel': int(a_pixel)})


def calculate_value_slo(state, slos=MB['slos']):
    fuzzy_slof = []

    for var_name, value in state.items():
        if var_name not in [v[0] for v in slos]:
            continue

        var, func, k, c = utils.filter_tuple(slos, var_name, 0)
        fuzzy_slof.append(func(value, k, c))

    return fuzzy_slof


if __name__ == '__main__':
    agent = AIFAgent()
    agent.run()
