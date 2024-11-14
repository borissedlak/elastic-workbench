import time
from threading import Thread


class AIFAgent(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        while True:
            print("pull")
            time.sleep(0.5)


if __name__ == '__main__':
    agent = AIFAgent()
    agent.run()
