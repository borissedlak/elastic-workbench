import time
from threading import Thread

from PrometheusClient import PrometheusClient


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

    def run(self):
        while True:
            slof_fps = self.prom_client.fetch_metric()
            print(slof_fps)
            time.sleep(1)


if __name__ == '__main__':
    agent = AIFAgent()
    agent.run()
