import ast
import logging

from flask import Flask, request

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
    return ""


@app.route("/stop_all", methods=['POST'])
def terminate_processing():
    qd.terminate()
    return ""


@app.route("/change_config", methods=['PUT'])
def change_config():
    service_d = ast.literal_eval(request.args.get('service_description'))
    qd.change_config(service_d)

    return ""

@app.route("/change_threads", methods=['PUT'])
def change_threads():
    threads_num = int(request.args.get('thread_number'))
    qd.change_threads(threads_num)

    return ""


if __name__ == '__main__':
    start_video_processing()
    app.run(host='0.0.0.0', port=8080)
