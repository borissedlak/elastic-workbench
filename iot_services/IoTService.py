import utils


class IoTService:
    def __init__(self):
        self.device_metric_reporter = None

    def process_one_iteration(self, params, frame) -> None:
        pass

    def start_process(self):
        pass

    def report_to_mongo(self, metrics) -> None:
        self.device_metric_reporter.report_metrics(utils.COLLECTION_NAME, metrics)

    def terminate(self):
        pass

    def change_config(self, service_d):
        pass
