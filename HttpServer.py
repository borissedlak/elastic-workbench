import logging

from flask import Flask

import utils
from HttpClient import HttpClient
from QrDetector import QrDetector

app = Flask(__name__)

DEVICE_NAME = utils.get_ENV_PARAM('DEVICE_NAME', "Unknown")

logging.getLogger('multiscale').setLevel(logging.INFO)

http_client = HttpClient()
qd = QrDetector(show_results=False)


# @utils.print_execution_time
@app.route("/start_video", methods=['POST'])
def start_video_processing():
    qd.start_process()
    return "Started service successfully"

@app.route("/stop_all", methods=['POST'])
def terminate_processing():
    qd.terminate()
    return "Service stopped successfully"


if __name__ == '__main__':
    start_video_processing()
    app.run(host='0.0.0.0', port=8080)
