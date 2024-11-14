import time
from threading import Thread

from PrometheusClient import PrometheusClient, MB


# TODO: So what the agent must do on a high level is:
#  1) Collect sensory state from Prometheus --> Easy
#  2) Evaluate if SLOs are fulfilled --> Easy
#  3) Retrain its interpretation model --> Difficult
#  4) Act so that SLO-F is optimized --> Difficult
#  _
#  However, I assume that the agent has the variable DAG and the SLO thresholds
#  And I dont have to resolve variable names dynamically, but keep them hardcoded


class AIFAgent(Thread):
    def __init__(self):
        super().__init__()
        
        self.prom_client = PrometheusClient()
        self.MB = MB

    def run(self):
        while True:
            slo_f = self.prom_client.get_slo_evaluations()
            parameter = self.prom_client.get_param_assignments()
            print(slo_f)
            print(parameter)
            time.sleep(1)


if __name__ == '__main__':
    agent = AIFAgent()
    agent.run()
