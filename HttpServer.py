import logging

from flask import Flask

import utils
from HttpClient import HttpClient

# from pgmpy.readwrite import XMLBIFReader

# from orchestration.HttpClient import HttpClient
# from orchestration.ServiceWrapper import start_service
# from orchestration.models import model_trainer

app = Flask(__name__)

MODEL_DIRECTORY = "./"

logger = logging.getLogger("vehicle")
logging.getLogger('pgmpy').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('vehicle').setLevel(logging.INFO)

DEVICE_NAME = utils.get_ENV_PARAM('DEVICE_NAME', "Unknown")

http_client = HttpClient()


# MEMBER ROUTES ######################################

# @utils.print_execution_time
@app.route("/start_service", methods=['POST'])
def start():
    pass

    return "M| Started service successfully"


def run_server():
    print(f"Start HttpServer at ??")
    app.run(host='0.0.0.0', port=8080)


run_server()
