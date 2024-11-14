import time
from threading import Thread

from PrometheusClient import PrometheusClient


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
