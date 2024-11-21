import os
from random import randint
from threading import Thread

import numpy as np
import pandas as pd
import torch

import utils
from PrometheusClient import MB, INTERVAL
from test_DQN import DQNAgent

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

        # self.prom_client = PrometheusClient()
        # self.docker_client = DockerClient(DOCKER_SOCKET)
        # self.http_client = HttpClient()
        self.round_counter = 0
        self.dqn = DQNAgent()
        self.simulated_env = ScalingEnv()

    def run(self):
        while True:
            # initial_state = self.get_current_state()
            # print("Initial State:", initial_state)
            # initial_state_f = [initial_state['pixel'], initial_state['fps']]
            initial_state_f = self.simulated_env.get_current_state()

            random = self.round_counter % 10 == 0 or self.round_counter < 1000  # e - greedy with 0.1
            # action_vectors = test_gpt.get_action(initial_state_f, random)
            action_vectors = self.dqn.choose_action(torch.FloatTensor(np.array([initial_state_f[1] > 20])), random)

            # agent.act_on_env(action_vectors[0])
            delta_pixel = initial_state_f[0] + action_vectors
            punishment_off = 0
            if delta_pixel < 100 or delta_pixel > 2000:
                delta_pixel = np.clip(initial_state_f[0] + action_vectors, 100, 2000)
                punishment_off = - 5
            self.simulated_env.act_on_env(delta_pixel)

            # time.sleep(5.0)
            updated_state = {'pixel': self.simulated_env.get_current_state()[0],
                             'fps': self.simulated_env.get_current_state()[1]}
            # updated_state = self.get_current_state()
            # print("Following State:", updated_state)
            # updated_state_f = [updated_state['pixel'], updated_state['fps']]
            updated_state_f = self.simulated_env.get_current_state()
            # print(f"State transition {initial_state_f} --> {updated_state_f}")

            value_factors = calculate_value_slo(updated_state)
            value = np.sum(value_factors) + punishment_off
            # print(f"Reward for {value_factors} = {value}\n")
            print(f"{self.round_counter}| Reward: {value} for {updated_state_f}")

            if self.dqn.memory.size() > 1000:
                # Probably due to buffer that is trained
                self.dqn.train_agent()

            # test_gpt.evaluate_result(initial_state_f, action_vectors, value, updated_state_f)
            self.round_counter += 1

            # if self.dqn.epsilon > self.dqn.epsilon_min:
            #     self.dqn.epsilon *= self.dqn.epsilon_decay

            if self.round_counter % 1000 == 0:
                # self.dqn.epsilon = 1.0
                self.simulated_env.reset_env()

    # TODO: Also, I should probably look into better ways to query the metric values, like EMA
    def get_current_state(self):
        metric_vars = list(set(MB['variables']) - set(MB['parameter']))
        prom_metric_states = self.prom_client.get_metric_values("|".join(metric_vars), period=INTERVAL)
        prom_parameter_states = self.prom_client.get_metric_values("|".join(MB['parameter']))

        cpu_cores = os.cpu_count()
        return prom_metric_states | prom_parameter_states | {"max_cores": cpu_cores}

    def act_on_env(self, a_pixel):
        self.http_client.change_config("localhost", {'pixel': int(a_pixel)})


class ScalingEnv:
    def __init__(self):
        self.regression_model = utils.get_regression_model(pd.read_csv("regression_data.csv"))
        self.pixel = None
        self.reset_env()

    def get_current_state(self):
        try:
            return [self.pixel, int(self.regression_model.predict([[self.pixel, 2.0]])[0])]
        except ValueError:
            print("Error")
            self.pixel = randint(100, 2000)
            return self.get_current_state()

    def act_on_env(self, pixel):
        self.pixel = int(pixel)

    def reset_env(self):
        self.pixel = randint(1, 20) * 100


def calculate_value_slo(state, slos=MB['slos']):
    fuzzy_slof = []

    for var_name, value in state.items():
        if var_name not in [v[0] for v in slos]:
            continue

        # var, func, k, c, boost = utils.filter_tuple(slos, var_name, 0)
        # fuzzy_slof.append(boost * func(value, k, c))

        if var_name == "pixel":
            fuzzy_slof.append(value >=800)
        elif var_name == "fps":
            fuzzy_slof.append(value >=20)
        else:
            raise RuntimeError("WHY??")

    return fuzzy_slof


if __name__ == '__main__':
    agent = AIFAgent()
    agent.run()
