import json
import logging

from HttpClient import HttpClient
from agent import agent_utils

logger = logging.getLogger("multiscale")


class ES_Registry:
    _ES_activate_default = {'elastic-workbench-video-processing': ['resource_scaling', 'quality_scaling']}

    def __init__(self):
        self.http_client = HttpClient()
        self.ES_activated = self._ES_activate_default
        with open('es_registry.json', 'r') as f:
            self.es_api = json.load(f)

    def is_ES_supported(self, service_type, es_name):
        strategies = self.es_api.get("strategies", [])
        for strategy in strategies:
            if (
                    strategy.get("service_type") == service_type and
                    strategy.get("ES_name") == es_name
            ):
                if es_name in self.ES_activated.get(service_type, []):
                    return True
                else:
                    logger.info(f"Strategy <{service_type},{es_name}> is registered, but not activated")
                    return False

        logger.info("No corresponding strategy registered")
        return False

    def get_ES_information(self, service_type, es_name):
        for strategy in self.es_api.get("strategies", []):
            if strategy["service_type"] == service_type and strategy["ES_name"] == es_name:
                return strategy.get("endpoints", [])
        return []

    def ES_random_execution(self, host, service_type, es_name):
        if not self.is_ES_supported(service_type, es_name):
            raise RuntimeError(f"Requesting unknown strategy <{service_type},{es_name}>")

        ES_endpoint = self.get_ES_information(service_type, es_name)
        for endpoint in ES_endpoint:
            random_params = agent_utils.get_random_parameter_assignments(endpoint['parameters'])
            # print(random_params)
            self.http_client.call_ES_endpoint(host, endpoint['target'], random_params)

            logger.info(f"Calling random ES <{service_type},{es_name}> with {random_params}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    es_reg = ES_Registry()
    print(es_reg.get_ES_information('elastic-workbench-video-processing', 'vertical_scaling'))
